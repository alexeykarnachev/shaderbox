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

import shutil
import time
from collections.abc import Callable
from pathlib import Path

from loguru import logger

from shaderbox.copilot.backend import CopilotBackend
from shaderbox.copilot.capabilities import CopilotCapabilities
from shaderbox.copilot.llm.openrouter import OpenRouterLLMClient
from shaderbox.copilot.persistence import archive_conversation
from shaderbox.copilot.revert import RevertExecutor
from shaderbox.copilot.session import CopilotSession
from shaderbox.exporters.integrations import IntegrationsStore
from shaderbox.exporters.registry import ExporterRegistry
from shaderbox.paths import ProjectPaths, shader_lib_root
from shaderbox.scripting import EngineContext, ScriptEngine
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

        # The CPU-script engine (feature 040): per-node uniform-compute behaviors, ticked once
        # per frame before render. Populated per project by _resolve_scripts in load().
        self.script_engine = ScriptEngine()

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
        # Per project (feature 040): reset the engine, resolve each node's scripts/u_*.py against
        # its active uniforms, and inject the per-frame export tick hook onto each Node. The live
        # path re-polls mtimes via reload_scripts() in ui.py; export ticks the resolved behaviors.
        self.script_engine = ScriptEngine()
        for node_id, ui_node in self.ui_nodes.items():
            self.script_engine.reload(
                node_id, self.paths.scripts_dir_for(node_id), ui_node.node
            )
            ui_node.node.on_pre_render = self._make_pre_render(node_id)

    def _make_pre_render(self, node_id: str) -> Callable[[float, float, int], None]:
        def _pre_render(t: float, dt: float, frame: int) -> None:
            self._tick_node(node_id, t, dt, frame)

        return _pre_render

    def _tick_node(self, node_id: str, t: float, dt: float, frame: int) -> None:
        ui_node = self.ui_nodes.get(node_id)
        if ui_node is None:
            return
        # uniforms snapshots the ticked node's own values (an integrator reads its prev value);
        # mouse/state are 041/042 reserved (state is the shared phase-A scratchpad).
        ctx = EngineContext(
            t=t,
            dt=dt,
            frame=frame,
            state=self.script_engine.state,
            uniforms=dict(ui_node.node.uniform_values),
        )
        self.script_engine.tick(node_id, ui_node.node, ctx)

    def reload_scripts(self) -> None:
        # The live hot-reload poll (decision 9): re-stat each node's scripts dir, recompiling only
        # changed files. Invoked from ui.py::update_and_draw before the live tick.
        for node_id, ui_node in self.ui_nodes.items():
            self.script_engine.reload(
                node_id, self.paths.scripts_dir_for(node_id), ui_node.node
            )

    def tick(self, node_ids: list[str], t: float, dt: float, frame: int) -> None:
        # The live per-frame tick (decision 4): tick exactly the nodes this frame will render
        # (the ui.py render gate), so a scripted uniform animates identically live and in export.
        for node_id in node_ids:
            self._tick_node(node_id, t, dt, frame)

    def get_script_driven_uniforms(self, node_id: str) -> set[str]:
        # The set of uniform names with a u_<name>.py file on `node_id` — the copilot set_uniform
        # reject (decision 5) queries this so it won't no-op a script-driven uniform.
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
