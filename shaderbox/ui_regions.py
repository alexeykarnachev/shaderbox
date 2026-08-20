"""The two keyboard/navigation enums shared by the command registry, the persisted UI state and
the draw layer. A leaf on purpose: `ActiveRegion` and `NodeTab` are plain names with no imgui in
them, and `ui_models.py` persists `active_node_tab` — so keeping them beside the imgui-evaluating
command table would drag imgui into the headless model layer (`commands.py` builds `K = imgui.Key`
at module scope, so importing it really does load the library)."""

from enum import StrEnum, auto


class ActiveRegion(StrEnum):
    # The three keyboard-nav focus regions; CYCLE_REGION moves between them and
    # nav operates within the focused one.
    EDITOR = auto()
    GRID = auto()
    PANEL = auto()


class NodeTab(StrEnum):
    # The settings-panel inner tabs; FOCUS_TAB_* jump to one directly.
    NODE = auto()
    RENDER = auto()
    SHARE = auto()
