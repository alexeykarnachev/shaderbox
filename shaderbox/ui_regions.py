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


class ChannelView(StrEnum):
    # What the viewer shows of the output texture. COLOR is the plain frame over the quiet
    # checker; COLOR_ALPHA composites over a loud checker so transparency reads; ALPHA shows
    # the alpha channel alone as grayscale.
    COLOR = auto()
    COLOR_ALPHA = auto()
    ALPHA = auto()


_CHANNEL_VIEW_CYCLE: list[ChannelView] = [
    ChannelView.COLOR,
    ChannelView.COLOR_ALPHA,
    ChannelView.ALPHA,
]


def next_channel_view(view: ChannelView) -> ChannelView:
    return _CHANNEL_VIEW_CYCLE[
        (_CHANNEL_VIEW_CYCLE.index(view) + 1) % len(_CHANNEL_VIEW_CYCLE)
    ]


CHANNEL_VIEW_LABELS: dict[ChannelView, str] = {
    ChannelView.COLOR: "Color",
    ChannelView.COLOR_ALPHA: "Color+Alpha",
    ChannelView.ALPHA: "Alpha",
}
