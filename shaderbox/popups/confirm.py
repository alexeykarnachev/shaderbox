"""The confirm modal: one client for every destructive verb (093 W4).

`App.request_confirm` builds the request at the verb, so the confirm names its target and its
consequence identically from a menu, a bar item, a button, a chord and the palette.
"""

from imgui_bundle import imgui

from shaderbox.app import App, ModalId
from shaderbox.popups import Modal
from shaderbox.theme import COLOR, SPACE
from shaderbox.ui_primitives import (
    danger_button,
    modal_footer,
    standard_button,
    wrapped_caption,
)

_LABEL = "Confirm##confirm"
_POPUP_W = 380.0


def _size(app: App) -> tuple[float, float]:
    _ = app
    return (_POPUP_W, 0.0)


def _draw_body(app: App) -> bool:
    keep_open = True
    request = app.confirm
    if request is None:
        return False

    wrapped_caption(request.title, COLOR.FG_TITLE)
    imgui.dummy((0.0, float(SPACE.XS)))
    wrapped_caption(request.line)

    # The modal auto-sizes to its own two lines, so it has no scrollable content region --
    # the footer alone (`modal_content`'s docstring).
    with modal_footer():
        # ONE decision per frame: the button and the key cannot both fire `on_confirm`. Enter
        # is declined on the appearing frame, which a keyboard menu activation shares with the
        # press that opened the modal (the bar draws before the popup block).
        confirmed = danger_button(request.verb) or (
            imgui.is_key_pressed(imgui.Key.enter, repeat=False)
            and not imgui.is_window_appearing()
        )
        if confirmed:
            request.on_confirm()
            keep_open = False
        imgui.same_line()
        if standard_button("Cancel"):
            keep_open = False
    return keep_open


def _on_close(app: App) -> None:
    app.clear_confirm()


MODAL = Modal(
    id=ModalId.CONFIRM,
    label=_LABEL,
    size=_size,
    body=_draw_body,
    on_close=_on_close,
)
