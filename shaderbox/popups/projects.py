"""The Projects modal (Ctrl+O) — every project verb in one surface (feature 084).

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

from shaderbox.app import App, PopupState
from shaderbox.editor_types import InlineInput
from shaderbox.project_session import ProjectInfo
from shaderbox.theme import COLOR, SIZE, SPACE
from shaderbox.ui_primitives import (
    danger_button,
    modal_window,
    primary_button,
    standard_button,
)

_LABEL = "Projects##projects"
_POPUP_W = 620.0
_POPUP_H = 420.0
# The name column, wide enough for a long project name before the count column starts.
_NAME_W = 220.0
_COUNT_W = 72.0
# The `open` marker column, between the count and the path.
_OPEN_W = 56.0


def draw_projects(app: App) -> None:
    if app.popup_state != PopupState.PROJECTS:
        return
    with modal_window(_LABEL, (_POPUP_W, _POPUP_H)) as visible:
        if not visible:
            return
        if not _draw_body(app):
            app.popup_state = PopupState.CLOSED
            app.reset_projects_state()
            imgui.close_current_popup()


def _draw_body(app: App) -> bool:
    rows_h = -imgui.get_frame_height_with_spacing() * 2.0
    if imgui.begin_child("##projects_rows", size=(0.0, rows_h)):
        for info in app.projects_rows:
            _draw_row(app, info)
    imgui.end_child()
    imgui.dummy((0.0, float(SPACE.MD)))

    if app.projects_new_input.is_open:
        return _draw_name_input(app, app.projects_new_input, "New", _commit_new)
    if app.projects_duplicate_input.is_open:
        return _draw_name_input(
            app, app.projects_duplicate_input, "Duplicate", _commit_duplicate
        )
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
    imgui.text_colored(
        COLOR.ACCENT_PRIMARY if selected else COLOR.FG_PRIMARY, info.name
    )
    imgui.same_line(_NAME_W)
    plural = "" if info.document_count == 1 else "s"
    imgui.text_colored(COLOR.FG_DIM, f"{info.document_count} doc{plural}")
    imgui.same_line(_NAME_W + _COUNT_W)
    if info.is_open:
        imgui.text_colored(COLOR.ACCENT_PRIMARY, "open")
    imgui.same_line(_NAME_W + _COUNT_W + _OPEN_W)
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
    if primary_button("New"):
        app.reset_projects_state()
        app.projects_new_input.open(app.default_projects_root_dir)
    imgui.same_line()
    imgui.begin_disabled(selected is None)
    if standard_button("Duplicate") and selected is not None:
        app.reset_projects_state()
        app.projects_duplicate_input.open(selected, f"{selected.name} copy")
    imgui.end_disabled()
    imgui.same_line()
    if standard_button("Open other..."):
        app.pick_project_dir()
        keep_open = False
    imgui.same_line()
    is_open_project = selected is not None and selected == app.project_dir.resolve()
    imgui.begin_disabled(selected is None or is_open_project)
    if danger_button("Delete"):
        app.projects_delete_armed = selected
    imgui.end_disabled()
    imgui.same_line(imgui.get_content_region_avail().x - float(SIZE.BTN_SM_W))
    if standard_button("Close", width=float(SIZE.BTN_SM_W)):
        keep_open = False
    if is_open_project:
        imgui.text_colored(COLOR.FG_DIM, "switch away first")
    elif app.projects_error:
        imgui.text_colored(COLOR.STATE_ERROR, app.projects_error)
    return keep_open


def _draw_delete_confirm(app: App) -> bool:
    armed = app.projects_delete_armed
    if armed is None:
        return True
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
    imgui.text_colored(COLOR.FG_DIM, verb)
    imgui.same_line()
    if state.needs_focus:
        # ONE-SHOT: re-grabbing every frame resets the caret blink and fights other inputs.
        imgui.set_keyboard_focus_here(0)
        state.needs_focus = False
    imgui.set_next_item_width(SIZE.NAME_INPUT_W)
    changed, state.buf = imgui.input_text(
        "##project_name", state.buf, imgui.InputTextFlags_.enter_returns_true
    )
    # Read the deactivate IMMEDIATELY after the input, before any same_line: the item-scoped
    # queries answer for the last submitted item.
    deactivated = imgui.is_item_deactivated_after_edit()
    # The outer Enter (a switch) must not also fire while this input holds focus.
    app.projects_input_focused = imgui.is_item_focused()
    if imgui.is_key_pressed(imgui.Key.escape, repeat=False):
        # `hotkeys._handle_escape` leaves the modal open for exactly this; without the cancel
        # here, Esc would be a dead key.
        state.close()
        app.projects_error = ""
        return True
    imgui.same_line()
    cancelled = standard_button("x")
    if cancelled:
        # Cancel wins over the deactivate its own click produced.
        state.close()
        app.projects_error = ""
        return True
    if changed or deactivated:
        error = commit(app, state.buf)
        app.projects_error = error
        if not error:
            # Close rather than stay: the switch is queued, so the list this modal is showing is
            # already stale -- and a stale `open` marker is how Delete reaches the wrong project.
            state.close()
            return False
    imgui.text_colored(
        COLOR.STATE_ERROR if app.projects_error else COLOR.FG_DIM,
        app.projects_error or "Enter creates",
    )
    return True


def _commit_new(app: App, name: str) -> str:
    return app.new_project(name)


def _commit_duplicate(app: App, name: str) -> str:
    source = app.projects_duplicate_input.target
    if source is None:
        return "no project selected"
    return app.duplicate_project(source, name)
