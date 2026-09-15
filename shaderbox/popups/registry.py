"""The modal registry: one `Modal` value per `ModalId`, and the two verbs every surface uses.

A modal is its body plus one `Modal(...)` constant in its own module; the draw call, the Esc
and Close funnel, the per-modal cleanup and the gates all derive from `MODALS` here. This
module is the leaf of the popups layer -- it imports `App`, the shared `Modal` type and every
popup module, no popup module imports it, and `app.py` imports nothing from `shaderbox.popups`.
"""

from imgui_bundle import imgui

from shaderbox.app import App, ModalId
from shaderbox.popups import (
    Modal,
    confirm,
    emoji_picker,
    examples,
    help,
    import_passes,
    lib_picker,
    pass_settings,
    projects,
    settings,
)
from shaderbox.ui_primitives import modal_window

MODALS: tuple[Modal, ...] = (
    examples.MODAL,
    help.MODAL,
    settings.MODAL,
    pass_settings.MODAL,
    import_passes.MODAL,
    emoji_picker.MODAL,
    lib_picker.MODAL,
    projects.MODAL,
    confirm.MODAL,
)

BY_ID: dict[ModalId, Modal] = {modal.id: modal for modal in MODALS}


def close_modal(app: App, forced: bool = False) -> bool:
    """The ONE close funnel -- Esc, and a body returning False. Reports whether it closed.

    An unforced close defers to `owns_esc`: an inline input owns that Esc and runs its own
    cancel later in the same frame. A Close the user clicked is always forced, since a
    dismissal the user asked for cannot be refused by a rename input inside the modal.

    A modal opened STACKED restores the one it covered, whose own `on_close` runs only when
    that one closes in its turn; an unstacked modal restores `None`.
    """
    modal = BY_ID.get(app.modal) if app.modal is not None else None
    if modal is None:
        return False
    if not forced and modal.owns_esc is not None and modal.owns_esc(app):
        return False
    if modal.on_close is not None:
        modal.on_close(app)
    app.modal = app.modal_below
    app.modal_below = None
    return True


def draw_modal(app: App) -> None:
    """Draw whichever modal is open. The one popup call in `ui.py`."""
    modal = BY_ID.get(app.modal) if app.modal is not None else None
    if modal is None:
        return
    with modal_window(
        modal.label, modal.size(app), modal.sizing, flags=modal.flags
    ) as visible:
        if not visible:
            return
        # `close_current_popup` is legal only inside the popup's scope, and only after a close
        # that happened -- imgui's stack would otherwise desync from `app.modal`. A Close the
        # user clicked is FORCED: an inline input inside the body cannot refuse it.
        if not modal.body(app) and close_modal(app, forced=True):
            imgui.close_current_popup()
