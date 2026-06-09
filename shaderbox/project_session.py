"""The headless project core (feature 025).

`ProjectSession` owns the project-lifecycle + (later, C3) copilot state that has no UI/GL
dependency — the state and methods the copilot backend reaches that App used to own directly.
It imports no imgui/glfw (the headless invariant): every UI side effect of a project mutation
flows back to the owner through injected callbacks the core invokes (the `ShaderLibFileManager`
idiom), and the `notifier` is injected, never constructed here (`Notifications` imports imgui).

App constructs one `ProjectSession` and forwards project state/ops to it via explicit
`@property` accessors; a headless harness (feature 026) constructs it directly on a standalone
EGL context. A moderngl context must be current on the constructing thread before any node load
(Node/Canvas do `self._gl = gl or moderngl.get_context()`).

Slices: C1 pure-core STATE + pure methods; C2 the GL-free project load (`load` / `seed_starter_node`).
The copilot cluster and the UI-tail method splits land in C3-C4.
"""

from collections.abc import Callable
from pathlib import Path

from loguru import logger

from shaderbox.exporters.integrations import IntegrationsStore
from shaderbox.notifications import Notifications
from shaderbox.paths import ProjectPaths, shader_lib_root
from shaderbox.shader_lib import ShaderLibIndex
from shaderbox.shader_lib import set_active as set_active_lib_index
from shaderbox.shader_lib.favorites import ShaderLibFavoritesStore
from shaderbox.shader_lib.tags import ShaderLibTagsStore
from shaderbox.templates_descriptions import TemplateDescriptionsStore
from shaderbox.ui_models import (
    UIAppState,
    UINode,
    load_node_from_dir,
    load_nodes_from_dir,
)


class ProjectSession:
    def __init__(
        self,
        *,
        node_templates_dir: Path,
        starter_template_id: str,
        template_order: list[str],
        notifier: Notifications,
    ) -> None:
        # notifier is injected (Notifications imports imgui — never built by the core).
        self._notifier = notifier
        self._node_templates_dir = node_templates_dir
        self._starter_template_id = starter_template_id
        # Authored template display order (filesystem ctime isn't preserved through git/zip);
        # templates not listed sort last.
        self._template_order = template_order

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
