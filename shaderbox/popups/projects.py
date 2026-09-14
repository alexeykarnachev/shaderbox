"""The Projects modal (Alt+O) — every project verb in one surface (feature 084).

A row per project: name, document count, path. The path is the column that earns the list — it is
the only thing distinguishing a project under the projects root from one opened from anywhere else,
and a project living somewhere surprising is how real work ends up in a directory the OS clears.

One rule places every verb: a row click SELECTS, and the verb row acts on the selection. Switching
is a double-click. Nothing here switches the project inline — the modal draws after the editor
panel and the document image have pushed their textures into the frame's draw list, so the switch
is requested and consumed by the frame tick before any drawing.
"""

from collections.abc import Callable
from pathlib import Path

from imgui_bundle import imgui

from shaderbox.app import App, ModalId
from shaderbox.popups import Modal
from shaderbox.project_session import ProjectInfo
from shaderbox.theme import COLOR, SIZE, SPACE
from shaderbox.ui_primitives import (
    InlineInput,
    danger_button,
    modal_content,
    modal_footer,
    name_input_row,
    primary_button,
    standard_button,
)

_LABEL = "Projects##projects"
_POPUP_W = 620.0
_POPUP_H = 420.0
# Where the path column begins, clear of the longest project name.
_PATH_X = 240.0


def _draw_body(app: App) -> bool:
    with modal_content():
        for info in app.projects_rows:
            _draw_row(app, info)
        # The error sits with the list rather than in the footer: the footer is one row tall
        # by contract, and a message appearing under the verbs would push them out of it.
        if app.projects_error:
            imgui.text_colored(COLOR.STATE_ERROR, app.projects_error)

    # Three alternative footers, one at a time: the verb row a plain open shows, the armed
    # delete row, and the name-entry row.
    if app.projects_new_input.is_open:
        return _draw_name_input(app, app.projects_new_input, "New", _commit_new)
    if app.projects_delete_armed is not None:
        return _draw_delete_confirm(app)
    return _draw_verb_row(app)


def _select_row(app: App, path: Path) -> None:
    # Selecting elsewhere disarms a pending delete, so Yes can never reach a project the user is
    # no longer looking at. Split out of the draw so the rule is reachable without a frame.
    app.projects_selected = path
    app.projects_delete_armed = None
    app.projects_error = ""


def _draw_row(app: App, info: ProjectInfo) -> None:
    selected = app.projects_selected == info.path
    # The id keys on the PATH, never the display name: two projects in different roots can share
    # a name, and the path is what the verbs act on.
    if imgui.selectable(f"##project_{info.path}", selected)[0]:
        _select_row(app, info.path)
    if (
        selected
        and not app.projects_input_focused
        and imgui.is_mouse_double_clicked(0)
        and imgui.is_item_hovered()
    ):
        app.request_project_switch(info.path)
    imgui.same_line(SPACE.MD)
    # The open project reads as the bright one and every other as secondary — the same
    # weight difference the pass strip uses for the output pass. Selection is the row
    # highlight, which is a different question and already drawn by `selectable`.
    imgui.text_colored(
        COLOR.FG_PRIMARY if info.is_open else COLOR.FG_SECONDARY, info.name
    )
    imgui.same_line(_PATH_X)
    imgui.text_colored(COLOR.FG_DIM, str(info.path.parent))


def _draw_verb_row(app: App) -> bool:
    keep_open = True
    selected = app.projects_selected
    # Enter on the selection switches, so a row reached by keyboard can be activated (the rows are
    # `selectable`, which IS a nav stop). Suppressed while a name input holds focus, or typing a
    # name and pressing Enter would both create the project and switch to the unrelated selection.
    if (
        selected is not None
        and not app.projects_input_focused
        and imgui.is_key_pressed(imgui.Key.enter, repeat=False)
    ):
        app.request_project_switch(selected)
        return False
    # Open is the modal's primary action -- switching is why it was opened -- and it acts on the
    # selection like every other verb here. Nothing rides a row: a control that appears inside one
    # on selection shifts the row it lives in, which is the overlay trap.
    is_open_project = selected is not None and selected == app.project_dir.resolve()
    with modal_footer():
        imgui.begin_disabled(selected is None or is_open_project)
        if primary_button("Open") and selected is not None:
            app.request_project_switch(selected)
            keep_open = False
        imgui.end_disabled()
        imgui.same_line()
        if standard_button("New"):
            app.reset_projects_state()
            app.projects_new_input.open(app.default_projects_root_dir)
        imgui.same_line()
        if standard_button("Open other..."):
            app.pick_project_dir()
            keep_open = False
        imgui.same_line()
        imgui.begin_disabled(selected is None or is_open_project)
        if danger_button("Delete"):
            app.projects_delete_armed = selected
        imgui.end_disabled()
        imgui.same_line()
        if standard_button("Close", width=float(SIZE.BTN_SM_W)):
            keep_open = False
    return keep_open


def _draw_delete_confirm(app: App) -> bool:
    armed = app.projects_delete_armed
    if armed is None:
        return True
    with modal_footer():
        imgui.text_colored(COLOR.STATE_ERROR, "Delete to trash?")
        imgui.same_line(imgui.get_content_region_avail().x - float(SIZE.BTN_SM_W) * 2.0)
        # Yes is the PRIMARY tier, not a filled red: the armed row already carries the danger.
        if primary_button("Yes"):
            app.projects_error = app.delete_project(armed)
            app.projects_delete_armed = None
            app.projects_rows = [r for r in app.projects_rows if r.path != armed]
            if app.projects_selected == armed:
                app.projects_selected = app.project_dir.resolve()
        imgui.same_line()
        if standard_button("No"):
            app.projects_delete_armed = None
    return True


def _draw_name_input(
    app: App, state: InlineInput, verb: str, commit: Callable[[App, str], str]
) -> bool:
    with modal_footer():
        imgui.text_colored(COLOR.FG_DIM, verb)
        imgui.same_line()
        result = name_input_row("project_name", state, width=float(SIZE.NAME_INPUT_W))
        # The outer Enter (a switch) must not also fire while this input holds focus.
        app.projects_input_focused = result.focused
        if result.cancelled:
            # The close funnel leaves the modal open for exactly this; without the cancel
            # here, Esc would be a dead key.
            state.close()
            app.projects_error = ""
            return True
        imgui.same_line()
        accepted = primary_button(verb) or result.committed
    if accepted:
        app.projects_error = commit(app, state.buf)
        if not app.projects_error:
            # Close rather than stay: the switch is queued, so the list this modal is showing is
            # already stale.
            state.close()
            return False
    return True


def _commit_new(app: App, name: str) -> str:
    return app.new_project(name)


MODAL = Modal(
    id=ModalId.PROJECTS,
    label=_LABEL,
    size=lambda app: (_POPUP_W, _POPUP_H),
    body=_draw_body,
    on_close=lambda app: app.reset_projects_state(),
    owns_esc=lambda app: app.projects_input_owns_esc(),
)
