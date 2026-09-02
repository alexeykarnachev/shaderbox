"""The settings-panel tab enum shared by the command registry, the persisted UI state and the
draw layer. A leaf on purpose: `DocumentTab` is a plain name with no imgui in it, and
`ui_models.py` persists `active_document_tab` — so keeping it beside the imgui-evaluating
command table would drag imgui into the headless model layer (`commands.py` builds `K = imgui.Key`
at module scope, so importing it really does load the library)."""

from enum import StrEnum, auto


class DocumentTab(StrEnum):
    # The settings-panel inner tabs; FOCUS_TAB_* jump to one directly.
    DOCUMENT = auto()
    RENDER = auto()
    SHARE = auto()
