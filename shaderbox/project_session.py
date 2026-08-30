"""The headless project core (feature 025).

`ProjectSession` owns the project-lifecycle + copilot state that has no UI/GL dependency — the
state and methods the copilot backend reaches that App used to own directly. It imports no
imgui/glfw (the headless invariant): every UI side effect of a project mutation flows back to the
owner through injected `on_*` callbacks the core invokes (the `ShaderLibFileManager` idiom), so the
core never touches notifications, editor sessions, or any imgui state itself.

App constructs one `ProjectSession` and forwards project state/ops to it via explicit
`@property` accessors; a headless harness (feature 026) constructs it directly on a standalone
EGL context, passing no `on_*` callbacks (they default to no-ops). A moderngl context must be
current on the constructing thread before any document load (Document/Canvas do
`self._gl = gl or moderngl.get_context()`).
"""

import contextlib
import re
import shutil
import time
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path

import moderngl
from loguru import logger

from shaderbox.copilot.backend import CopilotBackend
from shaderbox.copilot.capabilities import CopilotCapabilities
from shaderbox.copilot.config import COPILOT_ENGINE
from shaderbox.copilot.llm.openrouter import OpenRouterLLMClient
from shaderbox.copilot.persistence import archive_conversation
from shaderbox.copilot.revert import RevertExecutor
from shaderbox.copilot.session import CopilotSession
from shaderbox.core import ENGINE_DRIVEN_UNIFORMS, Pass
from shaderbox.document import Document
from shaderbox.exporters.registry import ExporterRegistry
from shaderbox.integrations import IntegrationsStore
from shaderbox.pass_graph import PassEntry, PassGraph, TargetConfig
from shaderbox.paths import (
    DOCUMENT_JSON_BASENAME,
    DOCUMENT_SCRIPT_BASENAME,
    PASS_SHADER_SUFFIX,
    PASSES_DIR_NAME,
    ProjectPaths,
    shader_lib_root,
)
from shaderbox.scripting import (
    EXPORT_MOUSE,
    EngineContext,
    MouseState,
    ScriptEngine,
    ScriptProbe,
    ScriptStatus,
    is_scriptable,
    normalize_script_tabs,
    script_stub_for,
)
from shaderbox.shader_lib import ShaderLibIndex
from shaderbox.shader_lib import set_active as set_active_lib_index
from shaderbox.shader_lib.favorites import ShaderLibFavoritesStore
from shaderbox.shader_lib.file_ops import ShaderLibFileManager
from shaderbox.shader_lib.tags import ShaderLibTagsStore
from shaderbox.shader_source import ShaderSource
from shaderbox.ui_models import (
    UIAppState,
    UIDocument,
    load_document_from_dir,
    load_documents_from_dir,
)
from shaderbox.util import select_next_value

# Prepended to the engine stub when the COPILOT reads a script-less document (feature 043). The actor
# copies verbatim, so a no-op commented stub teaches the binding but not MOTION — this gives one
# concrete ctx.t-driven pattern to adapt (a reference, not a body to save back unchanged).
_AGENT_STUB_EXAMPLE = (
    "# EXAMPLE -- a uniform animated over ctx.t (adapt the names + math, don't save this verbatim):\n"
    "#     pulse = 0.2 + 0.1 * math.sin(ctx.t * 2.0)        # a float oscillates\n"
    "#     cx = 0.5 + 0.3 * math.sin(ctx.t)                 # a Vec2 drifts\n"
    "#     cy = 0.5 + 0.3 * math.sin(ctx.t * 2.0)\n"
    "#     return {'u_radius': pulse, 'u_center': Vec2(cx, cy)}\n\n"
)


def _noop_current_document_changed(old_id: str, new_id: str) -> None:
    pass


def _noop_document_source_synced(path: Path, source: str) -> None:
    pass


def _noop_pass_renamed(old_path: Path, new_path: Path) -> None:
    pass


def _noop_document_deleted(document_id: str, source_path: Path) -> None:
    pass


# A new pass draws nothing until it is wired or authored: opaque black, so the first render is a
# blank slate rather than an error strip.
PASS_STUB = """#version 460 core

in vec2 vs_uv;
out vec4 fs_color;

void main() {
    fs_color = vec4(0.0, 0.0, 0.0, 1.0);
}
"""

# A pass name is a FILENAME and a graph key, so it stays to the characters both accept.
_PASS_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _pass_name_error(name: str, existing: dict[str, Pass]) -> str:
    if not _PASS_NAME_RE.match(name):
        return (
            "a pass name starts with a letter and holds letters, digits and underscores"
        )
    if name in existing:
        return f"'{name}' already exists"
    return ""


def _graph_without(graph: PassGraph, removed: str, kept: dict[str, Pass]) -> PassGraph:
    entries = {
        name: entry.model_copy(
            update={
                "inputs": {u: src for u, src in entry.inputs.items() if src != removed}
            }
        )
        for name, entry in graph.passes.items()
        if name != removed
    }
    output = graph.output if graph.output != removed else next(iter(kept), "")
    layout = {n: pos for n, pos in graph.layout.items() if n != removed}
    return graph.with_passes(entries, output=output, layout=layout)


def _graph_renamed(graph: PassGraph, old: str, new: str) -> PassGraph:
    def moved(entry: PassEntry) -> PassEntry:
        inputs = {u: (new if src == old else src) for u, src in entry.inputs.items()}
        return entry.model_copy(update={"inputs": inputs})

    entries = {
        (new if name == old else name): moved(entry)
        for name, entry in graph.passes.items()
    }
    layout = {(new if n == old else n): pos for n, pos in graph.layout.items()}
    return graph.with_passes(
        entries,
        output=new if graph.output == old else graph.output,
        layout=layout,
    )


class ProjectSession:
    def __init__(
        self,
        *,
        document_examples_dir: Path,
        starter_example_id: str,
        example_order: list[str],
        get_exporter_registry: Callable[[], ExporterRegistry],
        get_shader_lib_files: Callable[[], ShaderLibFileManager],
        # UI reactions to project mutations ride these callbacks (the core never touches imgui
        # state): the owner clears the sticky-focus bit / rehydrates the editor session / drops
        # the editor session + delete-arm. All default to no-ops so a headless caller omits them.
        on_current_document_changed: Callable[
            [str, str], None
        ] = _noop_current_document_changed,
        on_document_source_synced: Callable[
            [Path, str], None
        ] = _noop_document_source_synced,
        on_document_deleted: Callable[[str, Path], None] = _noop_document_deleted,
        on_pass_renamed: Callable[[Path, Path], None] = _noop_pass_renamed,
    ) -> None:
        self._document_examples_dir = document_examples_dir
        self._starter_example_id = starter_example_id
        # Authored example display order (filesystem ctime isn't preserved through git/zip);
        # examples not listed sort last.
        self._example_order = example_order
        # Injected refs: exporter_registry + shader_lib_files stay App-side (imgui-coupled).
        self._get_exporter_registry = get_exporter_registry
        self._get_shader_lib_files = get_shader_lib_files
        # UI-reaction callbacks the core invokes after a mutation (the owner does the imgui work).
        self._on_current_document_changed = on_current_document_changed
        self._on_document_source_synced = on_document_source_synced
        self._on_pass_renamed = on_pass_renamed
        self._on_document_deleted = on_document_deleted

        # ---- per-project state, (re)populated by _load ----
        self.paths: ProjectPaths
        self.project_dir: Path
        self.integrations_store = IntegrationsStore()
        self.ui_documents: dict[str, UIDocument] = {}
        # document.json mtime per loaded document — the diff baseline for sync_documents_from_disk.
        self._document_json_mtimes: dict[str, float] = {}
        self.ui_document_examples: dict[str, UIDocument] = {}
        self.app_state = UIAppState()
        # The active library index; rebuilt per project by rebuild_shader_lib_index.
        self.shader_lib_index: ShaderLibIndex = ShaderLibIndex.empty()
        # Cross-project stores (global, survive a project switch).
        self.shader_lib_favorites: ShaderLibFavoritesStore = (
            ShaderLibFavoritesStore.load()
        )
        self.shader_lib_tags: ShaderLibTagsStore = ShaderLibTagsStore.load()
        # Working set: every document/lib address the agent touched this turn, reset per turn (by the
        # copilot session at enqueue). LRU-ordered — oldest first, so the size cap evicts the
        # least-recently-touched member into `_copilot_working_set_evicted` (reported to the agent).
        self._copilot_working_set: list[str] = []
        self._copilot_working_set_evicted: list[str] = []

        # The CPU-script engine (feature 041): per-document uniform-compute behaviors, ticked once
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
            get_gate=lambda: self.copilot.gate,
            document_examples_dir=self.document_examples_dir,
            starter_example_id=self._starter_example_id,
            get_renders_dir=lambda: self.paths.renders_dir,
            get_ui_documents=lambda: self.ui_documents,
            get_ui_document_examples=lambda: self.ui_document_examples,
            get_exporter_registry=self._get_exporter_registry,
            get_shader_lib_index=lambda: self.shader_lib_index,
            get_shader_lib_files=self._get_shader_lib_files,
            get_current_document_id=lambda: self.current_document_id,
            get_is_cancelled=lambda: self.copilot.is_cancelled(),
            get_script_driven_uniforms=self.get_script_driven_uniforms,
            get_script_path=self.script_path_for,
            get_script_source_view=self.script_source_view,
            read_script_source=self.read_script_source,
            write_script_source=self.write_script_source,
            set_current_document_id=self.set_current_document_id,
            save_ui_document=self.save_ui_document,
            sync_editor_from_disk=self.sync_editor_from_disk,
            delete_document_unguarded=self._delete_document_unguarded,
            example_description=self.example_description,
            working_set_reader=lambda: self._copilot_working_set,
            working_set_add=self._copilot_ws_add,
            working_set_evicted=lambda: self._copilot_working_set_evicted,
            working_set_reset=self._copilot_ws_reset,
            get_active_checkpoint=lambda: self.copilot.checkpoints.active,
        )
        self.revert_executor = RevertExecutor(
            get_documents_dir=lambda: self.paths.documents_dir,
            get_trash_dir=lambda: self.paths.trash_dir,
            get_ui_documents=lambda: self.ui_documents,
            get_checkpoints=lambda: self.copilot.checkpoints,
            get_shader_lib_files=self._get_shader_lib_files,
            set_current_document_id=self.set_current_document_id,
            sync_editor_from_disk=self.sync_editor_from_disk,
            delete_document_unguarded=self._delete_document_unguarded,
            invalidate_lib_consumers=self.copilot_backend.invalidate_lib_consumers,
        )
        return self.copilot_backend

    def set_current_document_id(self, id: str = "") -> None:
        old_id = self.app_state.current_document_id
        self.app_state.current_document_id = id
        if id != old_id:
            self._on_current_document_changed(old_id, id)

    def save_ui_document(
        self,
        ui_document: UIDocument,
        root_dir: Path | None = None,
        dir_name: str | None = None,
    ) -> Path:
        # No toast here: the copilot calls this mid-turn (create_document) where a "Document saved"
        # toast is spurious (the chat already reports it). The user-initiated toast lives in
        # App.save_ui_document, the forwarder the UI paths call.
        root_dir = root_dir or self.paths.documents_dir
        dir = ui_document.save(root_dir, dir_name)
        # Our own write bumped document.json's mtime; rebaseline so sync_documents_from_disk doesn't read it
        # straight back as an external "change" and clobber the live document next frame.
        if root_dir == self.paths.documents_dir:
            with contextlib.suppress(OSError):
                self._document_json_mtimes[dir.name] = (
                    (dir / DOCUMENT_JSON_BASENAME).lstat().st_mtime
                )
        logger.info(f"Document '{ui_document.ui_state.ui_name}' saved: {dir}")
        return dir

    def sync_editor_from_disk(self, path: Path, source: str) -> None:
        # The whole reaction is UI (push new disk text into the live editor session), so the
        # core just fires the callback; the owner's handler does the editor work. Keyed by PATH,
        # which is what an editor session is keyed by — a document has one file per pass.
        self._on_document_source_synced(path, source)

    def _delete_document_unguarded(self, document_id: str) -> str:
        # Teardown shared by the public + copilot delete: release GL, drop the editor session,
        # reselect current, move the dir to trash. Returns the trash dir-NAME (id, or id_<ts>
        # on collision) so a caller can offer a Recover. Caller guarantees document_id in ui_documents.
        new_document_id = select_next_value(
            values=list(self.ui_documents.keys()),
            current_value=document_id,
            default_value="",
        )
        if new_document_id == document_id:
            new_document_id = ""

        # Capture the source path BEFORE the pop (it's gone after; the owner's editor sessions
        # are path-keyed). The on_document_deleted handler drops the editor session + delete-arm.
        path = self.ui_documents[document_id].document.render_pass.source.path
        self.ui_documents.pop(document_id).document.release()
        self._document_json_mtimes.pop(document_id, None)
        self.script_engine.drop_document(
            document_id
        )  # free its behaviors + stale errors (feature 041)
        if document_id in self._copilot_working_set:
            self._copilot_working_set.remove(document_id)
        self._on_document_deleted(document_id, path)
        if document_id == self.current_document_id or not self.current_document_id:
            self.set_current_document_id(new_document_id)
        trash_name = document_id
        dest = self.paths.trash_dir / trash_name
        if dest.exists():  # a prior document with this id was already trashed
            trash_name = f"{document_id}_{int(time.time() * 1000)}"
            dest = self.paths.trash_dir / trash_name
        shutil.move(self.paths.documents_dir / document_id, dest)

        logger.info(f"Document deleted: {document_id}")
        return trash_name

    @property
    def document_examples_dir(self) -> Path:
        return self._document_examples_dir

    @property
    def current_document_id(self) -> str:
        return self.app_state.current_document_id

    def example_description(self, example_uuid: str) -> str:
        ui_document = self.ui_document_examples.get(example_uuid)
        return ui_document.ui_state.description if ui_document is not None else ""

    def _copilot_ws_add(self, address: str) -> None:
        # Add a document full-id or "lib:" address to the working set, no dupes, MOVE-TO-END on a
        # re-touch (so the document being hammered is never the one the cap evicts). Past the cap the
        # oldest member is dropped and recorded; a re-added address leaves the record, or the
        # rendered "dropped" line would claim something the block still shows.
        if address in self._copilot_working_set:
            self._copilot_working_set.remove(address)
        self._copilot_working_set.append(address)
        if address in self._copilot_working_set_evicted:
            self._copilot_working_set_evicted.remove(address)
        cap = COPILOT_ENGINE.copilot_working_set_max_documents
        while cap > 0 and len(self._copilot_working_set) > cap:  # 0 = uncapped
            dropped = self._copilot_working_set.pop(0)
            if dropped not in self._copilot_working_set_evicted:
                self._copilot_working_set_evicted.append(dropped)

    def _copilot_ws_reset(self) -> None:
        self._copilot_working_set = []
        self._copilot_working_set_evicted = []

    def rebuild_shader_lib_index(self) -> None:
        # Walk shader_lib_root, extract every top-level function, publish via the module-level
        # accessor that Pass.compile() reads.
        self.shader_lib_index = ShaderLibIndex.build(shader_lib_root())
        set_active_lib_index(self.shader_lib_index)
        logger.debug(f"Lib index: {len(self.shader_lib_index.functions)} functions")

    def _order_examples(self, examples: dict[str, UIDocument]) -> dict[str, UIDocument]:
        rank = {eid: i for i, eid in enumerate(self._example_order)}
        ordered_ids = sorted(examples, key=lambda eid: rank.get(eid, len(rank)))
        return {eid: examples[eid] for eid in ordered_ids}

    def load(self, project_dir: Path) -> None:
        # Load the project's GL-free state: paths, lib index, documents + examples, app_state,
        # integrations. A moderngl context must already be current (document warm-up compiles).
        self.ui_documents.clear()

        self.paths = ProjectPaths.for_root(project_dir)
        self.project_dir = self.paths.root
        logger.info(f"Project loaded: {self.project_dir}")

        # Build the lib index before loading documents — every document's first compile (warm-up in
        # load_documents_from_dir) reads the active index.
        self.rebuild_shader_lib_index()

        self.ui_documents = load_documents_from_dir(self.paths.documents_dir)
        self._seed_document_json_mtimes()
        self.ui_document_examples = self._order_examples(
            load_documents_from_dir(self._document_examples_dir)
        )

        if self.paths.app_state_file.exists():
            self.app_state = UIAppState.load(self.paths.app_state_file)

        self.integrations_store = IntegrationsStore.load()
        self.integrations_store.copilot.apply_limits()

        self._resolve_scripts()

    def _resolve_scripts(self) -> None:
        # Per project (feature 041): reset the engine, resolve each document's scripts/u_*.py against its
        # active uniforms, and wire each Document's script hooks. The live path re-polls mtimes + re-wires
        # any newly-inserted document via reload_scripts() in ui.py.
        self.script_engine = ScriptEngine(ENGINE_DRIVEN_UNIFORMS)
        for document_id, ui_document in self.ui_documents.items():
            self.script_engine.reload(
                document_id,
                self.paths.scripts_dir_for(document_id),
                ui_document.document.render_pass,
            )
            self._wire_document_hooks(document_id, ui_document.document)

    def _seed_document_json_mtimes(self) -> None:
        # Baseline the sync cache from the just-loaded documents, so sync_documents_from_disk's first frame
        # sees no spurious "changed". A document whose dir/json vanished between load and seed is skipped.
        self._document_json_mtimes = {}
        for document_id in self.ui_documents:
            meta = self.paths.document_json_for(document_id)
            try:
                self._document_json_mtimes[document_id] = meta.lstat().st_mtime
            except OSError:
                continue

    def sync_documents_from_disk(self) -> None:
        # Per-frame document-dir watcher: disk is the source of truth, so reconcile ui_documents to it.
        # Globs documents/*/ + diffs each dir's document.json mtime against the cache, then ADDS new dirs,
        # REMOVES vanished ones, and RE-READS a dir whose document.json changed (a new uniform value /
        # ui_state / canvas size edited externally). Shader TEXT of a loaded document is NOT handled here
        # — reload_document_if_changed (ui.py) already hot-reloads it by source mtime; script.py likewise
        # rides reload_scripts. So this owns exactly the three things those miss: dir add/remove +
        # document.json. Cheap when nothing changed: one glob + a stat per dir.
        current: dict[str, float] = {}
        for document_dir in self.paths.documents_dir.iterdir():
            meta = document_dir / DOCUMENT_JSON_BASENAME
            passes = document_dir / PASSES_DIR_NAME
            # A dir is loadable only once document.json AND at least one pass file exist — skip a
            # half-written document (a document.json already on disk while its passes are still
            # being written); it syncs in once complete.
            if (
                not document_dir.is_dir()
                or not meta.is_file()
                or not any(passes.glob(f"*{PASS_SHADER_SUFFIX}"))
            ):
                continue
            try:
                current[document_dir.name] = meta.lstat().st_mtime
            except OSError:
                continue

        removed = [nid for nid in self.ui_documents if nid not in current]
        added = [nid for nid in current if nid not in self.ui_documents]
        changed = [
            nid
            for nid, mtime in current.items()
            if nid in self.ui_documents and self._document_json_mtimes.get(nid) != mtime
        ]
        if not (removed or added or changed):
            return

        for document_id in removed:
            path = self.ui_documents[document_id].document.render_pass.source.path
            self.ui_documents.pop(document_id).document.release()
            self.script_engine.drop_document(document_id)
            self._document_json_mtimes.pop(document_id, None)
            self._on_document_deleted(document_id, path)

        for document_id in added + changed:
            self._load_one_document_from_disk(document_id)
            self._document_json_mtimes[document_id] = current[document_id]

        # A removed dir may have dropped the current document; reselect (mirrors _delete_document_unguarded).
        if self.current_document_id not in self.ui_documents:
            self.set_current_document_id(next(iter(self.ui_documents), ""))

        logger.debug(
            f"Document sync: +{len(added)} -{len(removed)} ~{len(changed)} (now {len(self.ui_documents)})"
        )

    def _load_one_document_from_disk(self, document_id: str) -> None:
        # (Re)read one document dir from disk and install it: release a prior live copy, load fresh,
        # re-resolve its scripts + wire hooks, then push the disk shader text into any open editor.
        old = self.ui_documents.get(document_id)
        if old is not None:
            old.document.release()
        ui_document = load_document_from_dir(self.paths.documents_dir / document_id)
        self.ui_documents[document_id] = ui_document
        self.script_engine.reload(
            document_id,
            self.paths.scripts_dir_for(document_id),
            ui_document.document.render_pass,
        )
        self._wire_document_hooks(document_id, ui_document.document)
        for render_pass in ui_document.document.passes.values():
            self._on_document_source_synced(
                render_pass.source.path, render_pass.source.text
            )

    def _wire_document_hooks(self, document_id: str, document: Document) -> None:
        # Inject the export-isolation factory (Document.render_media enters it around every export, so an
        # exported integrator starts from a clean per-export instance). Wired ONCE on first sight —
        # called from reload_scripts each frame, so a document inserted AFTER load (copilot create /
        # example / revert-replace) gets it too. The live preview path does NOT ride on_pre_render
        # (ui.py ticks via session.tick); on_pre_render is the swap target the isolation factory uses.
        if document.export_isolation is not contextlib.nullcontext:
            return  # already wired (the factory never resets it, so this sentinel is unambiguous)
        document.export_isolation = self._make_export_isolation(document_id)

    def _make_export_isolation(
        self, document_id: str
    ) -> Callable[[], contextlib.AbstractContextManager[None]]:
        # The factory Document.render_media enters around EVERY export (feature 041). It swaps the document's
        # on_pre_render to tick a FRESH behavior set (recompiled from cached source, independent of the
        # live instances) so an exported integrator starts from a clean __init__ regardless of how long
        # the live preview ran, and restores the live hook in finally. Because render_media itself
        # enters it, no export caller can forget to isolate.
        @contextlib.contextmanager
        def _isolation() -> Iterator[None]:
            ui_document = self.ui_documents.get(document_id)
            if ui_document is None:
                yield
                return
            document = ui_document.document
            live_hook = document.on_pre_render
            behavior = self.script_engine.fresh_behavior_for(document_id)
            if behavior is None:
                yield
                return

            def _export_pre_render(t: float, dt: float, frame: int) -> None:
                # EXPORT_MOUSE (the EngineContext default) freezes the cursor at center so an
                # exported render is deterministic. No stopped set — an export always plays the script.
                self.script_engine.tick_export(
                    document_id,
                    document.render_pass,
                    EngineContext(t=t, dt=dt, frame=frame),
                    behavior,
                )

            document.on_pre_render = _export_pre_render
            try:
                yield
            finally:
                document.on_pre_render = live_hook

        return _isolation

    def reload_scripts(self) -> None:
        # The live hot-reload poll: re-stat each document's scripts dir, recompiling only changed files
        # (a recompile makes a fresh instance — state resets on edit), and re-wire hooks so a document
        # inserted after load (copilot create / example / revert) is covered. Invoked from
        # ui.py::update_and_draw before the live tick.
        for document_id, ui_document in self.ui_documents.items():
            self.script_engine.reload(
                document_id,
                self.paths.scripts_dir_for(document_id),
                ui_document.document.render_pass,
            )
            self._wire_document_hooks(document_id, ui_document.document)

    def tick(
        self,
        document_ids: list[str],
        t: float,
        dt: float,
        frame: int,
        *,
        mouse: MouseState = EXPORT_MOUSE,
    ) -> None:
        # The live per-frame tick: tick exactly the documents this frame will render (the ui.py render
        # gate), so a scripted uniform animates identically live and in export. `mouse` is the live
        # cursor App passes in (headless callers omit it → EXPORT_MOUSE, deterministic).
        for document_id in document_ids:
            ui_document = self.ui_documents.get(document_id)
            if ui_document is None:
                continue
            self.script_engine.tick(
                document_id,
                ui_document.document.render_pass,
                EngineContext(t=t, dt=dt, frame=frame, mouse=mouse),
                self._stopped_for(document_id),
            )

    def _stopped_for(self, document_id: str) -> frozenset[str]:
        # The uniform names frozen for manual edit this frame (048): the document's explicit
        # `stopped_uniforms` UNION every driven name when the document is `all_stopped`. Built fresh each
        # tick (never cached across the tick/draw boundary) and passed to the engine as a param — the
        # engine never learns UIDocumentState (the headless boundary holds, as `engine_driven` does).
        ui_document = self.ui_documents.get(document_id)
        if ui_document is None:
            return frozenset()
        state = ui_document.ui_state
        stopped = set(state.stopped_uniforms)
        if state.all_stopped:
            stopped |= self.script_engine.script_driven_uniforms(document_id)
        return frozenset(stopped)

    def get_script_status(self, document_id: str) -> ScriptStatus | None:
        # The document script's UI status for 042's strip (sentinel error + driven count + homeless
        # soft-key errors), or None when the document has no script.py.
        return self.script_engine.script_status(document_id)

    def has_script(self, document_id: str) -> bool:
        # Whether the document's `script.py` exists on disk (the open-script glyph state + the play/stop
        # affordance gate). Disk presence so a create lands instantly, before the next reload.
        return self.paths.document_script_for(document_id).is_file()

    def script_has_error(self, document_id: str) -> bool:
        # Whether the document's script has a recorded compile/run error (the open-script glyph error tint).
        return (document_id, DOCUMENT_SCRIPT_BASENAME) in self.script_engine.errors

    def _scriptable_uniforms_for(self, document_id: str) -> list[moderngl.Uniform]:
        # The uniforms a script can drive: scriptable + not engine-owned. The engine silently drops a
        # script key naming an engine uniform, so listing one as a stub example invites a silent no-op
        # (the legibility gap 048 targets).
        return [
            u
            for u in self.ui_documents[
                document_id
            ].document.render_pass.get_active_uniforms()
            if is_scriptable(u) and u.name not in ENGINE_DRIVEN_UNIFORMS
        ]

    def create_script(self, document_id: str) -> Path:
        # Write the document script `script.py` + return its path; the next reload_scripts binds it (048 —
        # the file's existence IS the binding, no activate step). The skeleton is the engine's stub
        # (explicit imports + an empty-dict body + the document's uniforms as commented examples).
        scripts_dir = self.paths.scripts_dir_for(document_id)
        scripts_dir.mkdir(parents=True, exist_ok=True)
        path = self.script_path_for(document_id)
        path.write_text(
            script_stub_for(self._scriptable_uniforms_for(document_id)),
            encoding="utf-8",
        )
        return path

    def script_path_for(self, document_id: str) -> Path:
        # The scripts/ path for the document script `script.py` (048 — one script per document).
        return self.paths.document_script_for(document_id)

    def read_script_source(self, document_id: str) -> tuple[str, bool]:
        # The copilot read_script source (feature 043): the live scripts/script.py text, or — when the
        # document has no script — the AGENT stub (the engine stub + one un-commented math.sin(ctx.t)
        # example, so the actor has a concrete animating pattern to copy). The stub is NOT persisted;
        # returns (text, is_stub).
        path = self.script_path_for(document_id)
        if path.is_file():
            return path.read_text(encoding="utf-8"), False
        stub = script_stub_for(self._scriptable_uniforms_for(document_id))
        return _AGENT_STUB_EXAMPLE + stub, True

    def write_script_source(self, document_id: str, new_text: str) -> ScriptProbe:
        # The copilot write_script (feature 043): overwrite (or create) scripts/script.py, reload so
        # the compile verdict is live, then dry-run for the tick-gated facts. Returns the probe; the
        # backend renders it into the tool result + the motion facts.
        path = self.script_path_for(document_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(normalize_script_tabs(new_text), encoding="utf-8")
        ui_document = self.ui_documents[document_id]
        self.script_engine.reload(
            document_id,
            self.paths.scripts_dir_for(document_id),
            ui_document.document.render_pass,
        )
        return self.script_engine.dry_run(
            document_id,
            ui_document.document.render_pass,
            COPILOT_ENGINE.motion_sample_times,
            COPILOT_ENGINE.motion_fps,
        )

    def script_source_view(self, document_id: str) -> tuple[str, ScriptStatus | None]:
        # The working-set script sub-view (feature 043): the live script source ("" = no script) + its
        # script status (sentinel error for the working-set error line). GL-free reads.
        path = self.script_path_for(document_id)
        if not path.is_file():
            return "", None
        return path.read_text(encoding="utf-8"), self.get_script_status(document_id)

    def uniform_is_driven(self, document_id: str, name: str) -> bool:
        # Whether the script TARGETS this uniform (playing OR stopped) — the gate for showing the row's
        # play/stop button at all (a never-scripted MANUAL uniform shows nothing). Reads the engine's
        # last-tick driven set (decision 4/10).
        return name in self.script_engine.script_driven_uniforms(document_id)

    def is_uniform_stopped(self, document_id: str, name: str) -> bool:
        # Whether the user has STOPPED this uniform (explicitly, or via the document-level all_stopped).
        ui_document = self.ui_documents.get(document_id)
        if ui_document is None:
            return False
        state = ui_document.ui_state
        return state.all_stopped or name in state.stopped_uniforms

    def set_uniform_stopped(self, document_id: str, name: str, stopped: bool) -> None:
        # Add/remove a uniform from the document's stopped set (the row's play/stop toggle + the auto-stop
        # on manual edit). Document-scoped + name-keyed, so it survives a retype + works before any row
        # draws (no lazy-row trap). Persists in document.json on the next save.
        ui_document = self.ui_documents.get(document_id)
        if ui_document is None:
            return
        names = ui_document.ui_state.stopped_uniforms
        if stopped and name not in names:
            names.append(name)
        elif not stopped and name in names:
            names.remove(name)

    def set_document_all_stopped(self, document_id: str, stopped: bool) -> None:
        # The whole-document play/stop: freeze/resume every driven uniform's write at once. The script keeps
        # ticking either way (stop freezes WRITES, not ticking — so a later play resumes from advanced
        # state). Document-play also CLEARS every explicit per-uniform stop, so a full stop->play cycle
        # returns the whole document to playing (a uniform stopped mid-play doesn't survive the round trip).
        ui_document = self.ui_documents.get(document_id)
        if ui_document is not None:
            ui_document.ui_state.all_stopped = stopped
            if not stopped:
                ui_document.ui_state.stopped_uniforms.clear()

    def get_script_driven_uniforms(self, document_id: str) -> set[str]:
        # The uniform names the script drove on its last tick — the copilot set_uniform reject queries
        # this so it won't no-op a script-driven uniform.
        return self.script_engine.script_driven_uniforms(document_id)

    def clear_conversation(self) -> None:
        # Archive the live conversation (recoverable), delete checkpoints, reset to a fresh empty
        # chat + persist the empty store. No-op mid-turn (the reset_conversation invariant needs an
        # idle worker). The copilot resumes with ZERO memory of prior turns — only the documents on disk
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

    # ---- the pass graph's six verbs (D15) -------------------------------------------------
    # Every one mutates the live document AND saves, so `passes/` and `graph.json` never
    # disagree with what is on screen. Each returns an error string, or "" on success.

    def add_pass(self, document_id: str, name: str) -> str:
        ui_document = self.ui_documents.get(document_id)
        if ui_document is None:
            return f"no such document '{document_id}'"
        error = _pass_name_error(name, ui_document.document.passes)
        if error:
            return error
        document = ui_document.document
        source_path = self.paths.pass_shader_for(document_id, name)
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(PASS_STUB, encoding="utf-8")
        render_pass = Pass(
            gl=document.gl,
            source=ShaderSource.load(source_path),
            canvas_size=document.canvas_size,
            target=TargetConfig(),
        )
        render_pass.compile()
        document.passes[name] = render_pass
        document.graph = document.graph.with_passes(
            {**document.graph.passes, name: PassEntry()}
        )
        self.save_ui_document(ui_document)
        return ""

    def delete_pass(self, document_id: str, name: str) -> str:
        ui_document = self.ui_documents.get(document_id)
        if ui_document is None:
            return f"no such document '{document_id}'"
        document = ui_document.document
        if name not in document.passes:
            return f"no such pass '{name}'"
        if len(document.passes) == 1:
            return "a document needs at least one pass"
        document.passes.pop(name).release()
        # Every edge that named it goes too: an input left pointing at a deleted pass would read
        # black (D3), which is silent — the panel's own delete must not leave that behind.
        document.graph = _graph_without(document.graph, name, document.passes)
        self.save_ui_document(ui_document)
        return ""

    def rename_pass(self, document_id: str, old: str, new: str) -> str:
        ui_document = self.ui_documents.get(document_id)
        if ui_document is None:
            return f"no such document '{document_id}'"
        document = ui_document.document
        if old not in document.passes:
            return f"no such pass '{old}'"
        if new == old:
            return ""
        error = _pass_name_error(new, document.passes)
        if error:
            return error
        # Transactional (D15): the file, every edge that references it, the output choice, and the
        # open editor tab move together. Any one of them left behind fails SILENTLY — an edge
        # naming a pass that no longer exists just reads black.
        render_pass = document.passes.pop(old)
        old_path = render_pass.source.path
        new_path = self.paths.pass_shader_for(document_id, new)
        old_path.replace(new_path)
        render_pass.source = replace(render_pass.source, path=new_path)
        document.passes[new] = render_pass
        document.graph = _graph_renamed(document.graph, old, new)
        self._on_pass_renamed(old_path, new_path)
        self.save_ui_document(ui_document)
        return ""

    def set_output_pass(self, document_id: str, name: str) -> str:
        ui_document = self.ui_documents.get(document_id)
        if ui_document is None:
            return f"no such document '{document_id}'"
        document = ui_document.document
        if name not in document.passes:
            return f"no such pass '{name}'"
        document.graph = document.graph.with_output(name)
        self.save_ui_document(ui_document)
        return ""

    def wire_pass_input(
        self, document_id: str, consumer: str, uniform: str, producer: str
    ) -> str:
        """Fill `consumer`'s `uniform` from `producer`, or unwire it when `producer` is empty.

        A closed set by construction: the caller picks `producer` from the document's own pass
        names, which is what makes SHADERed's positional-slot footgun impossible here.
        """
        ui_document = self.ui_documents.get(document_id)
        if ui_document is None:
            return f"no such document '{document_id}'"
        document = ui_document.document
        if consumer not in document.passes:
            return f"no such pass '{consumer}'"
        if producer and producer not in document.passes:
            return f"no such pass '{producer}'"
        document.graph = document.graph.with_input(consumer, uniform, producer)
        self.save_ui_document(ui_document)
        return ""

    def set_pass_target(self, document_id: str, name: str, target: TargetConfig) -> str:
        ui_document = self.ui_documents.get(document_id)
        if ui_document is None:
            return f"no such document '{document_id}'"
        document = ui_document.document
        if name not in document.passes:
            return f"no such pass '{name}'"
        document.graph = document.graph.with_target(name, target)
        document.passes[name].set_target(target)
        self.save_ui_document(ui_document)
        return ""

    def seed_starter_document(self, seed_current: Callable[[str], None]) -> None:
        # First-run only: seed a starter into an empty project. A document load + save + select;
        # `seed_current` is the owner's set-current hook (the setter lives in App until C3).
        starter_dir = self._document_examples_dir / self._starter_example_id
        if not starter_dir.is_dir():
            logger.warning(f"Starter example missing ({starter_dir}); skipping seed")
            return
        try:
            new_document = load_document_from_dir(starter_dir)
            new_document.reset_id()
            new_document.save(self.paths.documents_dir, new_document.id)
            self.ui_documents[new_document.id] = new_document
            seed_current(new_document.id)
            logger.debug(f"Seeded starter document {new_document.id} (first run)")
        except Exception as e:
            logger.error(f"Failed to seed starter document: {e}")
