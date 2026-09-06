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
    # What the viewer shows of the output texture, as the three questions that can be asked
    # of it: COLOR composites the frame over the checker, so transparency reads; ALPHA shows
    # the alpha channel alone as grayscale; RGB discards alpha entirely, so a frame whose
    # background is transparent still shows the color the shader wrote there -- the checker
    # is never seen under it.
    COLOR = auto()
    ALPHA = auto()
    RGB = auto()


_CHANNEL_VIEW_CYCLE: list[ChannelView] = [
    ChannelView.COLOR,
    ChannelView.ALPHA,
    ChannelView.RGB,
]


def next_channel_view(view: ChannelView) -> ChannelView:
    return _CHANNEL_VIEW_CYCLE[
        (_CHANNEL_VIEW_CYCLE.index(view) + 1) % len(_CHANNEL_VIEW_CYCLE)
    ]


CHANNEL_VIEW_LABELS: dict[ChannelView, str] = {
    ChannelView.COLOR: "Color",
    ChannelView.ALPHA: "Alpha",
    ChannelView.RGB: "RGB",
}
