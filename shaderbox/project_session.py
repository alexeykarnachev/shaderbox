"""The headless project core (feature 025).

`ProjectSession` owns the project-lifecycle + copilot state that has no UI/GL dependency — the
state and methods the copilot backend reaches that App used to own directly. It imports no
imgui/glfw (the headless invariant): every UI side effect of a project mutation flows back to the
owner through injected `on_*` callbacks the core invokes (the `ShaderLibFileManager` idiom), so the
core never touches notifications, editor sessions, or any imgui state itself.

App constructs one `ProjectSession` and forwards project state/ops to it via explicit
`@property` accessors; a headless harness (feature 026) constructs it directly on a standalone
EGL context, passing no `on_*` callbacks (they default to no-ops). A moderngl context must be
current on the constructing thread before any node load (Node/Canvas do
`self._gl = gl or moderngl.get_context()`).
"""

import contextlib
import shutil
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import moderngl
from loguru import logger

from shaderbox.copilot.backend import CopilotBackend
from shaderbox.copilot.capabilities import CopilotCapabilities
from shaderbox.copilot.config import COPILOT_CONFIG
from shaderbox.copilot.llm.openrouter import OpenRouterLLMClient
from shaderbox.copilot.persistence import archive_conversation
from shaderbox.copilot.revert import RevertExecutor
from shaderbox.copilot.session import CopilotSession
from shaderbox.core import ENGINE_DRIVEN_UNIFORMS, Node
from shaderbox.exporters.integrations import IntegrationsStore
from shaderbox.exporters.registry import ExporterRegistry
from shaderbox.paths import ProjectPaths, shader_lib_root
from shaderbox.scripting import (
    EXPORT_MOUSE,
    BrainStatus,
    EngineContext,
    MouseState,
    ScriptEngine,
    ScriptProbe,
    brain_stub_for,
    is_scriptable,
    normalize_script_tabs,
)
from shaderbox.shader_lib import ShaderLibIndex
from shaderbox.shader_lib import set_active as set_active_lib_index
from shaderbox.shader_lib.favorites import ShaderLibFavoritesStore
from shaderbox.shader_lib.file_ops import ShaderLibFileManager
from shaderbox.shader_lib.tags import ShaderLibTagsStore
from shaderbox.templates_descriptions import TemplateDescriptionsStore
from shaderbox.ui_models import (
    UIAppState,
    UINode,
    load_node_from_dir,
    load_nodes_from_dir,
)
from shaderbox.util import select_next_value

# Prepended to the engine stub when the COPILOT reads a script-less node (feature 043). The actor
# copies verbatim, so a no-op commented stub teaches the binding but not MOTION — this gives one
# concrete ctx.t-driven pattern to adapt (a reference, not a body to save back unchanged).
_AGENT_STUB_EXAMPLE = (
    "# EXAMPLE -- a uniform animated over ctx.t (adapt the names + math, don't save this verbatim):\n"
    "#     pulse = 0.2 + 0.1 * math.sin(ctx.t * 2.0)        # a float oscillates\n"
    "#     cx = 0.5 + 0.3 * math.sin(ctx.t)                 # a Vec2 drifts\n"
    "#     cy = 0.5 + 0.3 * math.sin(ctx.t * 2.0)\n"
    "#     return {'u_radius': pulse, 'u_center': Vec2(cx, cy)}\n\n"
)


def _noop_current_node_changed(old_id: str, new_id: str) -> None:
    pass


def _noop_node_source_synced(node_id: str, source: str) -> None:
    pass


def _noop_node_deleted(node_id: str, source_path: Path) -> None:
    pass


class ProjectSession:
    def __init__(
        self,
        *,
        node_templates_dir: Path,
        starter_template_id: str,
        template_order: list[str],
        get_exporter_registry: Callable[[], ExporterRegistry],
        get_shader_lib_files: Callable[[], ShaderLibFileManager],
        # UI reactions to project mutations ride these callbacks (the core never touches imgui
        # state): the owner clears the sticky-focus bit / rehydrates the editor session / drops
        # the editor session + delete-arm. All default to no-ops so a headless caller omits them.
        on_current_node_changed: Callable[
            [str, str], None
        ] = _noop_current_node_changed,
        on_node_source_synced: Callable[[str, str], None] = _noop_node_source_synced,
        on_node_deleted: Callable[[str, Path], None] = _noop_node_deleted,
    ) -> None:
        self._node_templates_dir = node_templates_dir
        self._starter_template_id = starter_template_id
        # Authored template display order (filesystem ctime isn't preserved through git/zip);
        # templates not listed sort last.
        self._template_order = template_order
        # Injected refs: exporter_registry + shader_lib_files stay App-side (imgui-coupled).
        self._get_exporter_registry = get_exporter_registry
        self._get_shader_lib_files = get_shader_lib_files
        # UI-reaction callbacks the core invokes after a mutation (the owner does the imgui work).
        self._on_current_node_changed = on_current_node_changed
        self._on_node_source_synced = on_node_source_synced
        self._on_node_deleted = on_node_deleted

        # ---- per-project state, (re)populated by _load ----
        self.paths: ProjectPaths
        self.project_dir: Path
        self.integrations_store = IntegrationsStore()
        self.ui_nodes: dict[str, UINode] = {}
        self.ui_node_templates: dict[str, UINode] = {}
        self.app_state = UIAppState()
        # The active library index; rebuilt per project by rebuild_shader_lib_index.
        self.shader_lib_index: ShaderLibIndex = ShaderLibIndex.empty()
        # Cross-project stores (global, survive a project switch).
        self.shader_lib_favorites: ShaderLibFavoritesStore = (
            ShaderLibFavoritesStore.load()
        )
        self.shader_lib_tags: ShaderLibTagsStore = ShaderLibTagsStore.load()
        self.template_descriptions: TemplateDescriptionsStore = (
            TemplateDescriptionsStore.load()
        )
        # Working set: every node/lib address the agent touched this turn, reset per turn.
        self._copilot_working_set: list[str] = []

        # The CPU-script engine (feature 041): per-node uniform-compute behaviors, ticked once
        # per frame before render. Populated per project by _resolve_scripts in load().
        self.script_engine = ScriptEngine(ENGINE_DRIVEN_UNIFORMS)

        # Built LAST: _build_copilot_capabilities reads the project-state fields above. The
        # client reads the key/model LIVE through getters — _load reassigns integrations_store,
        # so capturing it here would go stale. project_dir isn't set until _load -> the trace
        # slug falls back to "project" until the first reset_conversation rotates it.
        self.copilot = CopilotSession(
            caps=self._build_copilot_capabilities(),
            client=OpenRouterLLMClient(
                get_api_key=lambda: self.integrations_store.copilot.openrouter_key,
                get_model=lambda: self.integrations_store.copilot.model,
            ),
            get_project_slug=lambda: getattr(self, "project_dir", Path("project")).name,
            get_checkpoints_root=lambda: self.paths.copilot_checkpoints_dir,
        )

    def _build_copilot_capabilities(self) -> CopilotCapabilities:
        # Construct the CopilotBackend — it satisfies the CopilotCapabilities Protocol
        # structurally, so it IS the capabilities object. Project-dependent deps are getters
        # (a project switch retargets them); deps that reference self.copilot are lazy (it
        # doesn't exist yet). exporter_registry + shader_lib_files stay App-side, reached
        # via injected getters.
        self.copilot_backend = CopilotBackend(
            get_bridge=lambda: self.copilot.bridge,
            node_templates_dir=self.node_templates_dir,
            starter_template_id=self._starter_template_id,
            get_renders_dir=lambda: self.paths.renders_dir,
            get_ui_nodes=lambda: self.ui_nodes,
            get_ui_node_templates=lambda: self.ui_node_templates,
            get_exporter_registry=self._get_exporter_registry,
            get_shader_lib_index=lambda: self.shader_lib_index,
            get_shader_lib_files=self._get_shader_lib_files,
            get_current_node_id=lambda: self.current_node_id,
            get_is_cancelled=lambda: self.copilot.is_cancelled(),
            get_script_driven_uniforms=self.get_script_driven_uniforms,
            get_script_path=self.script_path_for,
            get_script_source_view=self.script_source_view,
            read_script_source=self.read_script_source,
            write_script_source=self.write_script_source,
            set_current_node_id=self.set_current_node_id,
            save_ui_node=self.save_ui_node,
            sync_editor_from_disk=self.sync_editor_from_disk,
            delete_node_unguarded=self._delete_node_unguarded,
            template_description=self.template_description,
            working_set_reader=lambda: self._copilot_working_set,
            working_set_add=self._copilot_ws_add,
            get_active_checkpoint=lambda: self.copilot.checkpoints.active,
        )
        self.revert_executor = RevertExecutor(
            get_nodes_dir=lambda: self.paths.nodes_dir,
            get_trash_dir=lambda: self.paths.trash_dir,
            get_ui_nodes=lambda: self.ui_nodes,
            get_checkpoints=lambda: self.copilot.checkpoints,
            get_shader_lib_files=self._get_shader_lib_files,
            set_current_node_id=self.set_current_node_id,
            sync_editor_from_disk=self.sync_editor_from_disk,
            delete_node_unguarded=self._delete_node_unguarded,
            invalidate_lib_consumers=self.copilot_backend.invalidate_lib_consumers,
        )
        return self.copilot_backend

    def set_current_node_id(self, id: str = "") -> None:
        old_id = self.app_state.current_node_id
        self.app_state.current_node_id = id
        if id != old_id:
            self._on_current_node_changed(old_id, id)

    def save_ui_node(
        self,
        ui_node: UINode,
        root_dir: Path | None = None,
        dir_name: str | None = None,
    ) -> Path:
        # No toast here: the copilot calls this mid-turn (create_node) where a "Node saved"
        # toast is spurious (the chat already reports it). The user-initiated toast lives in
        # App.save_ui_node, the forwarder the UI paths call.
        root_dir = root_dir or self.paths.nodes_dir
        dir = ui_node.save(root_dir, dir_name)
        logger.info(f"Node '{ui_node.ui_state.ui_name}' saved: {dir}")
        return dir

    def sync_editor_from_disk(self, node_id: str, source: str) -> None:
        # The whole reaction is UI (push new disk text into the live editor session), so the
        # core just fires the callback; the owner's handler does the editor work.
        self._on_node_source_synced(node_id, source)

    def _delete_node_unguarded(self, node_id: str) -> str:
        # Teardown shared by the public + copilot delete: release GL, drop the editor session,
        # reselect current, move the dir to trash. Returns the trash dir-NAME (id, or id_<ts>
        # on collision) so a caller can offer a Recover. Caller guarantees node_id in ui_nodes.
        new_node_id = select_next_value(
            values=list(self.ui_nodes.keys()),
            current_value=node_id,
            default_value="",
        )
        if new_node_id == node_id:
            new_node_id = ""

        # Capture the source path BEFORE the pop (it's gone after; the owner's editor sessions
        # are path-keyed). The on_node_deleted handler drops the editor session + delete-arm.
        path = self.ui_nodes[node_id].node.source.path
        self.ui_nodes.pop(node_id).node.release()
        self.script_engine.drop_node(
            node_id
        )  # free its behaviors + stale errors (feature 041)
        if node_id in self._copilot_working_set:
            self._copilot_working_set.remove(node_id)
        self._on_node_deleted(node_id, path)
        if node_id == self.current_node_id or not self.current_node_id:
            self.set_current_node_id(new_node_id)
        trash_name = node_id
        dest = self.paths.trash_dir / trash_name
        if dest.exists():  # a prior node with this id was already trashed
            trash_name = f"{node_id}_{int(time.time() * 1000)}"
            dest = self.paths.trash_dir / trash_name
        shutil.move(self.paths.nodes_dir / node_id, dest)

        logger.info(f"Node deleted: {node_id}")
        return trash_name

    @property
    def node_templates_dir(self) -> Path:
        return self._node_templates_dir

    @property
    def current_node_id(self) -> str:
        return self.app_state.current_node_id

    def template_description(self, template_uuid: str) -> str:
        # Effective description: the user override if present, else the shipped node.json
        # default. ui_state is NOT mutated, so a 'reset' = delete the sidecar key.
        override = self.template_descriptions.get(template_uuid)
        if override is not None:
            return override
        ui_node = self.ui_node_templates.get(template_uuid)
        return ui_node.ui_state.description if ui_node is not None else ""

    def _copilot_ws_add(self, address: str) -> None:
        # Add a node full-id or "lib:" address to the working set, order-preserved, no dupes.
        if address not in self._copilot_working_set:
            self._copilot_working_set.append(address)

    def rebuild_shader_lib_index(self) -> None:
        # Walk shader_lib_root, extract every top-level function, publish via the module-level
        # accessor that Node.compile() reads.
        self.shader_lib_index = ShaderLibIndex.build(shader_lib_root())
        set_active_lib_index(self.shader_lib_index)
        logger.debug(f"Lib index: {len(self.shader_lib_index.functions)} functions")

    def _order_templates(self, templates: dict[str, UINode]) -> dict[str, UINode]:
        rank = {tid: i for i, tid in enumerate(self._template_order)}
        ordered_ids = sorted(templates, key=lambda tid: rank.get(tid, len(rank)))
        return {tid: templates[tid] for tid in ordered_ids}

    def load(self, project_dir: Path) -> None:
        # Load the project's GL-free state: paths, lib index, nodes + templates, app_state,
        # integrations. A moderngl context must already be current (node warm-up compiles).
        self.ui_nodes.clear()

        self.paths = ProjectPaths.for_root(project_dir)
        self.project_dir = self.paths.root
        logger.info(f"Project loaded: {self.project_dir}")

        # Build the lib index before loading nodes — every node's first compile (warm-up in
        # load_nodes_from_dir) reads the active index.
        self.rebuild_shader_lib_index()

        self.ui_nodes = load_nodes_from_dir(self.paths.nodes_dir)
        self.ui_node_templates = self._order_templates(
            load_nodes_from_dir(self._node_templates_dir)
        )

        if self.paths.app_state_file.exists():
            self.app_state = UIAppState.load_and_migrate(self.paths.app_state_file)

        self.integrations_store = IntegrationsStore.load()
        self.integrations_store.copilot.apply_limits()

        self._resolve_scripts()

    def _resolve_scripts(self) -> None:
        # Per project (feature 041): reset the engine, resolve each node's scripts/u_*.py against its
        # active uniforms, and wire each Node's script hooks. The live path re-polls mtimes + re-wires
        # any newly-inserted node via reload_scripts() in ui.py.
        self.script_engine = ScriptEngine(ENGINE_DRIVEN_UNIFORMS)
        for node_id, ui_node in self.ui_nodes.items():
            self.script_engine.reload(
                node_id,
                self.paths.scripts_dir_for(node_id),
                ui_node.node,
            )
            self._wire_node_hooks(node_id, ui_node.node)

    def _wire_node_hooks(self, node_id: str, node: Node) -> None:
        # Inject the export-isolation factory (Node.render_media enters it around every export, so an
        # exported integrator starts from a clean per-export instance). Wired ONCE on first sight —
        # called from reload_scripts each frame, so a node inserted AFTER load (copilot create /
        # template / revert-replace) gets it too. The live preview path does NOT ride on_pre_render
        # (ui.py ticks via session.tick); on_pre_render is the swap target the isolation factory uses.
        if node.export_isolation is not contextlib.nullcontext:
            return  # already wired (the factory never resets it, so this sentinel is unambiguous)
        node.export_isolation = self._make_export_isolation(node_id)

    def _make_export_isolation(
        self, node_id: str
    ) -> Callable[[], contextlib.AbstractContextManager[None]]:
        # The factory Node.render_media enters around EVERY export (feature 041). It swaps the node's
        # on_pre_render to tick a FRESH behavior set (recompiled from cached source, independent of the
        # live instances) so an exported integrator starts from a clean __init__ regardless of how long
        # the live preview ran, and restores the live hook in finally. Because render_media itself
        # enters it, no export caller can forget to isolate.
        @contextlib.contextmanager
        def _isolation() -> Iterator[None]:
            ui_node = self.ui_nodes.get(node_id)
            if ui_node is None:
                yield
                return
            node = ui_node.node
            live_hook = node.on_pre_render
            brain = self.script_engine.fresh_behavior_for(node_id)
            if brain is None:
                yield
                return

            def _export_pre_render(t: float, dt: float, frame: int) -> None:
                # EXPORT_MOUSE (the EngineContext default) freezes the cursor at center so an
                # exported render is deterministic. No stopped set — an export always plays the script.
                self.script_engine.tick_export(
                    node_id, node, EngineContext(t=t, dt=dt, frame=frame), brain
                )

            node.on_pre_render = _export_pre_render
            try:
                yield
            finally:
                node.on_pre_render = live_hook

        return _isolation

    def reload_scripts(self) -> None:
        # The live hot-reload poll: re-stat each node's scripts dir, recompiling only changed files
        # (a recompile makes a fresh instance — state resets on edit), and re-wire hooks so a node
        # inserted after load (copilot create / template / revert) is covered. Invoked from
        # ui.py::update_and_draw before the live tick.
        for node_id, ui_node in self.ui_nodes.items():
            self.script_engine.reload(
                node_id,
                self.paths.scripts_dir_for(node_id),
                ui_node.node,
            )
            self._wire_node_hooks(node_id, ui_node.node)

    def tick(
        self,
        node_ids: list[str],
        t: float,
        dt: float,
        frame: int,
        *,
        mouse: MouseState = EXPORT_MOUSE,
    ) -> None:
        # The live per-frame tick: tick exactly the nodes this frame will render (the ui.py render
        # gate), so a scripted uniform animates identically live and in export. `mouse` is the live
        # cursor App passes in (headless callers omit it → EXPORT_MOUSE, deterministic).
        for node_id in node_ids:
            ui_node = self.ui_nodes.get(node_id)
            if ui_node is None:
                continue
            self.script_engine.tick(
                node_id,
                ui_node.node,
                EngineContext(t=t, dt=dt, frame=frame, mouse=mouse),
                self._stopped_for(node_id),
            )

    def _stopped_for(self, node_id: str) -> frozenset[str]:
        # The uniform names frozen for manual edit this frame (048): the node's explicit
        # `stopped_uniforms` UNION every driven name when the node is `all_stopped`. Built fresh each
        # tick (never cached across the tick/draw boundary) and passed to the engine as a param — the
        # engine never learns UINodeState (the headless boundary holds, as `engine_driven` does).
        ui_node = self.ui_nodes.get(node_id)
        if ui_node is None:
            return frozenset()
        state = ui_node.ui_state
        stopped = set(state.stopped_uniforms)
        if state.all_stopped:
            stopped |= self.script_engine.script_driven_uniforms(node_id)
        return frozenset(stopped)

    def get_brain_status(self, node_id: str) -> BrainStatus | None:
        # The node-brain's UI status for 042's strip (sentinel error + driven count + homeless
        # soft-key errors), or None when the node has no script.py.
        return self.script_engine.brain_status(node_id)

    def has_script(self, node_id: str) -> bool:
        # Whether the node's `script.py` exists on disk (the open-script glyph state + the play/stop
        # affordance gate). Disk presence so a create lands instantly, before the next reload.
        return (self.paths.scripts_dir_for(node_id) / "script.py").is_file()

    def script_has_error(self, node_id: str) -> bool:
        # Whether the node's brain has a recorded compile/run error (the open-script glyph error tint).
        return (node_id, "script.py") in self.script_engine.errors

    def _scriptable_uniforms_for(self, node_id: str) -> list[moderngl.Uniform]:
        # The uniforms a brain can drive: scriptable + not engine-owned. The engine silently drops a
        # brain key naming an engine uniform, so listing one as a stub example invites a silent no-op
        # (the legibility gap 048 targets).
        return [
            u
            for u in self.ui_nodes[node_id].node.get_active_uniforms()
            if is_scriptable(u) and u.name not in ENGINE_DRIVEN_UNIFORMS
        ]

    def create_script(self, node_id: str) -> Path:
        # Write the node-brain `script.py` + return its path; the next reload_scripts binds it (048 —
        # the file's existence IS the binding, no activate step). The skeleton is the engine's stub
        # (explicit imports + an empty-dict body + the node's uniforms as commented examples).
        scripts_dir = self.paths.scripts_dir_for(node_id)
        scripts_dir.mkdir(parents=True, exist_ok=True)
        path = self.script_path_for(node_id)
        path.write_text(
            brain_stub_for(self._scriptable_uniforms_for(node_id)), encoding="utf-8"
        )
        return path

    def script_path_for(self, node_id: str) -> Path:
        # The scripts/ path for the node-brain `script.py` (048 — one script per node).
        return self.paths.scripts_dir_for(node_id) / "script.py"

    def read_script_source(self, node_id: str) -> tuple[str, bool]:
        # The copilot read_script source (feature 043): the live scripts/script.py text, or — when the
        # node has no brain — the AGENT stub (the engine stub + one un-commented math.sin(ctx.t)
        # example, so the actor has a concrete animating pattern to copy). The stub is NOT persisted;
        # returns (text, is_stub).
        path = self.script_path_for(node_id)
        if path.is_file():
            return path.read_text(encoding="utf-8"), False
        stub = brain_stub_for(self._scriptable_uniforms_for(node_id))
        return _AGENT_STUB_EXAMPLE + stub, True

    def write_script_source(self, node_id: str, new_text: str) -> ScriptProbe:
        # The copilot write_script (feature 043): overwrite (or create) scripts/script.py, reload so
        # the compile verdict is live, then dry-run for the tick-gated facts. Returns the probe; the
        # backend renders it into the tool result + the motion facts.
        path = self.script_path_for(node_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(normalize_script_tabs(new_text), encoding="utf-8")
        ui_node = self.ui_nodes[node_id]
        self.script_engine.reload(
            node_id, self.paths.scripts_dir_for(node_id), ui_node.node
        )
        return self.script_engine.dry_run(
            node_id,
            ui_node.node,
            COPILOT_CONFIG.motion_sample_times,
            COPILOT_CONFIG.motion_fps,
        )

    def script_source_view(self, node_id: str) -> tuple[str, BrainStatus | None]:
        # The working-set script sub-view (feature 043): the live script source ("" = no brain) + its
        # brain status (sentinel error for the working-set error line). GL-free reads.
        path = self.script_path_for(node_id)
        if not path.is_file():
            return "", None
        return path.read_text(encoding="utf-8"), self.get_brain_status(node_id)

    def uniform_is_driven(self, node_id: str, name: str) -> bool:
        # Whether the brain TARGETS this uniform (playing OR stopped) — the gate for showing the row's
        # play/stop button at all (a never-scripted MANUAL uniform shows nothing). Reads the engine's
        # last-tick driven set (decision 4/10).
        return name in self.script_engine.script_driven_uniforms(node_id)

    def is_uniform_stopped(self, node_id: str, name: str) -> bool:
        # Whether the user has STOPPED this uniform (explicitly, or via the node-level all_stopped).
        ui_node = self.ui_nodes.get(node_id)
        if ui_node is None:
            return False
        state = ui_node.ui_state
        return state.all_stopped or name in state.stopped_uniforms

    def set_uniform_stopped(self, node_id: str, name: str, stopped: bool) -> None:
        # Add/remove a uniform from the node's stopped set (the row's play/stop toggle + the auto-stop
        # on manual edit). Node-scoped + name-keyed, so it survives a retype + works before any row
        # draws (no lazy-row trap). Persists in node.json on the next save.
        ui_node = self.ui_nodes.get(node_id)
        if ui_node is None:
            return
        names = ui_node.ui_state.stopped_uniforms
        if stopped and name not in names:
            names.append(name)
        elif not stopped and name in names:
            names.remove(name)

    def set_node_all_stopped(self, node_id: str, stopped: bool) -> None:
        # The whole-node play/stop: freeze/resume every driven uniform's write at once. The brain keeps
        # ticking either way (stop freezes WRITES, not ticking — so a later play resumes from advanced
        # state). Node-play clears ONLY this blanket, never explicit per-uniform stops.
        ui_node = self.ui_nodes.get(node_id)
        if ui_node is not None:
            ui_node.ui_state.all_stopped = stopped

    def get_script_driven_uniforms(self, node_id: str) -> set[str]:
        # The uniform names the brain drove on its last tick — the copilot set_uniform reject queries
        # this so it won't no-op a script-driven uniform.
        return self.script_engine.script_driven_uniforms(node_id)

    def clear_conversation(self) -> None:
        # Archive the live conversation (recoverable), delete checkpoints, reset to a fresh empty
        # chat + persist the empty store. No-op mid-turn (the reset_conversation invariant needs an
        # idle worker). The copilot resumes with ZERO memory of prior turns — only the nodes on disk
        # remain. App.copilot_clear_chat forwards here; the dogfood harness calls it for a
        # context-wipe (a fresh agent on an existing project).
        if self.copilot.state.in_flight:
            return
        archive_conversation(
            self.paths.copilot_conversation_path, time.strftime("%Y-%m-%d_%H-%M-%S")
        )
        self.copilot.clear_checkpoints()
        self.copilot.reset_conversation()
        self.copilot.save_conversation(self.paths.copilot_conversation_path)

    def seed_starter_node(self, seed_current: Callable[[str], None]) -> None:
        # First-run only: seed a starter into an empty project. A node load + save + select;
        # `seed_current` is the owner's set-current hook (the setter lives in App until C3).
        template_dir = self._node_templates_dir / self._starter_template_id
        if not template_dir.is_dir():
            logger.warning(f"Starter template missing ({template_dir}); skipping seed")
            return
        try:
            new_node = load_node_from_dir(template_dir)
            new_node.reset_id()
            new_node.save(self.paths.nodes_dir, new_node.id)
            self.ui_nodes[new_node.id] = new_node
            seed_current(new_node.id)
            logger.debug(f"Seeded starter node {new_node.id} (first run)")
        except Exception as e:
            logger.error(f"Failed to seed starter node: {e}")
