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

This is the C1 slice: pure-core STATE + the pure methods. The project-load orchestration
(`_init` body), the copilot cluster, and the UI-tail method splits land in C2-C4.
"""

from pathlib import Path

from loguru import logger

from shaderbox.notifications import Notifications
from shaderbox.paths import shader_lib_root
from shaderbox.shader_lib import ShaderLibIndex
from shaderbox.shader_lib import set_active as set_active_lib_index
from shaderbox.shader_lib.favorites import ShaderLibFavoritesStore
from shaderbox.shader_lib.tags import ShaderLibTagsStore
from shaderbox.templates_descriptions import TemplateDescriptionsStore
from shaderbox.ui_models import UIAppState, UINode


class ProjectSession:
    def __init__(
        self,
        *,
        node_templates_dir: Path,
        starter_template_id: str,
        notifier: Notifications,
    ) -> None:
        # notifier is injected (Notifications imports imgui — never built by the core).
        self._notifier = notifier
        self._node_templates_dir = node_templates_dir
        self._starter_template_id = starter_template_id

        # ---- pure-core project state (loaded from disk in _init today, C2 moves the load) ----
        self.ui_nodes: dict[str, UINode] = {}
        self.ui_node_templates: dict[str, UINode] = {}
        self.app_state = UIAppState()
        # The active library index; rebuilt per project by rebuild_shader_lib_index.
        self.shader_lib_index: ShaderLibIndex = ShaderLibIndex.empty()
        # Cross-project stores (global, survive a project switch).
        self.shader_lib_favorites: ShaderLibFavoritesStore = ShaderLibFavoritesStore.load()
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
