"""The confirm modal, and the close funnel every modal reaches (093 W4).

Two questions, two rigs. The confirm's own behavior is asked of REAL frames -- the modal that
names its target, the one decision per frame, the Enter that does not fire on the appearing
frame, and the Esc that runs nothing -- because a body that parses and confirms nothing is
exactly the failure a confirm must not have. The funnel's per-row cleanup and its `owns_esc`
decline are asked of the registry directly, since a second frame-driving App in one process
hits a torn-down imgui font atlas.
"""

from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from imgui_bundle import imgui

from shaderbox import hotkeys, ui_primitives
from shaderbox.app import ModalId
from shaderbox.commands import CommandId, command_label
from shaderbox.constants import STARTER_EXAMPLE_ID
from shaderbox.copilot.state import Message
from shaderbox.paths import shader_lib_root
from shaderbox.popups import confirm, lib_picker
from shaderbox.popups.lib_picker import tree
from shaderbox.popups.registry import BY_ID, MODALS, close_modal, draw_modal
from shaderbox.ui_models import ConfirmRequest
from shaderbox.widgets import copilot_chat

# This module drives real frames, so it gets its own worker: the imgui font atlas is per
# PROCESS and a second App that renders a full frame in one interpreter dies on a texture the
# first released (`conventions.md ## Known quirks`).
pytestmark = pytest.mark.xdist_group("gl_frames_confirm")


def _pump(app: Any, frames: int = 1) -> None:
    """Draw the open modal for real, one frame per call."""
    for _ in range(frames):
        imgui.new_frame()
        draw_modal(app)
        imgui.end_frame()


def _press(key: imgui.Key) -> None:
    imgui.get_io().add_key_event(key, True)


def _release(key: imgui.Key) -> None:
    imgui.get_io().add_key_event(key, False)


# ---------------------------------------------------------------------------
# R4 — one decision per frame, and the appearing frame swallows Enter
# ---------------------------------------------------------------------------


def test_enter_confirms_but_never_on_the_appearing_frame(app: Any) -> None:
    """A keyboard menu activation shares its frame with the press that opened the modal (the
    bar draws before the popup block), so the appearing frame declines.

    Falsifier: drop `not imgui.is_window_appearing()` -- the Enter that opened the menu item
    fires the verb instantly, and the first assertion below goes red.
    """
    ran: list[int] = []
    app.request_confirm(_request(lambda: ran.append(1)))

    _press(imgui.Key.enter)
    _pump(app)
    assert ran == [], "Enter fired on the modal's appearing frame"
    assert app.modal is ModalId.CONFIRM

    _release(imgui.Key.enter)
    _pump(app)
    _press(imgui.Key.enter)
    _pump(app)
    assert ran == [1], "a later Enter did not confirm"
    assert app.modal is None and app.confirm is None
    _release(imgui.Key.enter)
    _pump(app)


def test_the_verb_runs_once_when_the_button_and_enter_land_together(
    app: Any, monkeypatch: Any
) -> None:
    """The body draws the button and reads the key in the same pass, so a click while Enter
    is also pressed must still be ONE decision.

    Falsifier: give the Enter branch its own `if` -- the spy counts two.
    """
    ran: list[int] = []
    app.request_confirm(_request(lambda: ran.append(1)))
    # Settle past the appearing frame, so the Enter below is the body's to read.
    _pump(app)
    _release(imgui.Key.enter)
    _pump(app)

    real_danger = ui_primitives.danger_button

    def clicking(label: str, *args: Any, **kwargs: Any) -> bool:
        real_danger(label, *args, **kwargs)
        return True

    monkeypatch.setattr(confirm, "danger_button", clicking)
    _press(imgui.Key.enter)
    _pump(app)
    assert ran == [1], f"the verb ran {len(ran)} times on one frame"
    _release(imgui.Key.enter)


def test_escape_cancels_and_runs_nothing(app: Any) -> None:
    """Esc closes through the funnel, which nulls the request. Falsifier: run `on_confirm`
    from the modal's `on_close`."""
    ran: list[int] = []
    app.request_confirm(_request(lambda: ran.append(1)))
    assert close_modal(app) is True
    assert ran == [], "the cancel ran the verb"
    assert app.modal is None and app.confirm is None


def test_cancel_closes_and_runs_nothing(app: Any, monkeypatch: Any) -> None:
    """The Cancel button's path, driven through a real frame. Falsifier: return `keep_open`
    from the Cancel branch."""
    ran: list[int] = []
    app.request_confirm(_request(lambda: ran.append(1)))
    _pump(app)

    real_standard = ui_primitives.standard_button

    def clicking(label: str, *args: Any, **kwargs: Any) -> bool:
        real_standard(label, *args, **kwargs)
        return label.split("##", 1)[0] == "Cancel"

    monkeypatch.setattr(confirm, "standard_button", clicking)
    _pump(app)
    assert ran == []
    assert app.modal is None and app.confirm is None


def _request(on_confirm: Any) -> ConfirmRequest:
    return ConfirmRequest(
        title="Delete pass b?",
        verb="Delete",
        on_confirm=on_confirm,
    )


# ---------------------------------------------------------------------------
# R5 / R7 — every destructive surface asks the same question
# ---------------------------------------------------------------------------


def test_the_pass_delete_verb_asks_then_deletes(app: Any) -> None:
    """The pass survives the request and goes on the confirm. Falsifier: call
    `session.delete_pass` from `delete_pass_confirmed` -- the first assertion goes red."""
    document_id = app.current_document_id
    name = next(iter(app.ui_documents[document_id].document.passes))
    assert app.session.add_pass(document_id, "other") == ""

    with mock.patch.object(
        app.session, "delete_pass", wraps=app.session.delete_pass
    ) as verb:
        app.delete_pass_confirmed(document_id, name)
        assert verb.call_count == 0, "the verb deleted before the confirm"
        assert app.modal is ModalId.CONFIRM
        assert app.confirm is not None
        assert app.confirm.title == f"Delete pass {name}?"
        assert app.confirm.verb == "Delete"
        app.confirm.on_confirm()
    assert verb.call_count == 1, [c.args for c in verb.call_args_list]
    assert verb.call_args.args[:2] == (document_id, name)
    assert name not in app.ui_documents[document_id].document.passes


def test_the_pass_delete_confirm_through_the_real_modal(app: Any) -> None:
    """The whole path, frame-driven: the tile's verb, the modal, the Enter, the pass gone.

    Falsifier: drop the `on_confirm` call from the confirm body -- the pass survives Enter.
    """
    document_id = app.current_document_id
    name = next(iter(app.ui_documents[document_id].document.passes))
    assert app.session.add_pass(document_id, "other") == ""

    app.delete_pass_confirmed(document_id, name)
    _pump(app)
    assert name in app.ui_documents[document_id].document.passes
    _release(imgui.Key.enter)
    _pump(app)
    _press(imgui.Key.enter)
    _pump(app)
    _release(imgui.Key.enter)
    assert name not in app.ui_documents[document_id].document.passes
    assert app.modal is None


def test_the_document_delete_verb_names_the_document(app: Any) -> None:
    document_id = app.current_document_id
    ui_name = app.ui_documents[document_id].ui_state.ui_name
    with mock.patch.object(app, "delete_document", wraps=app.delete_document) as verb:
        app.delete_current_document_confirmed()
        assert verb.call_count == 0
        assert app.confirm is not None
        assert app.confirm.title == f"Move {ui_name} to the trash?"
        assert app.confirm.verb == "Delete"
        app.confirm.on_confirm()
    assert verb.call_count == 1
    assert verb.call_args.args[:1] == (document_id,)


def test_the_document_reset_verb_names_the_document(app: Any) -> None:
    ui_name = app.ui_documents[app.current_document_id].ui_state.ui_name
    with mock.patch.object(
        app.session, "reset_document", wraps=app.session.reset_document
    ) as verb:
        app.reset_document_confirmed()
        assert verb.call_count == 0
        assert app.confirm is not None
        assert app.confirm.title == f"Reset {ui_name}?"
        assert app.confirm.verb == "Reset"
        app.confirm.on_confirm()
    assert verb.call_count == 1


def test_the_tile_menus_reset_targets_its_own_document(app: Any) -> None:
    """A tile's Reset restarts THAT document, not whichever one is current.

    A right-click does not select the tile it opens on (`document_grid.draw`), so a menu
    routed through the current-document verb would restart the wrong document with the right
    name on the confirm. Falsifier: point the menu at `reset_document_confirmed` -- the
    non-current id below stops reaching `session.reset_document`.
    """
    other_id = app.create_document_from_example(STARTER_EXAMPLE_ID)
    app.select_document(other_id)
    target = next(i for i in app.ui_documents if i != other_id)
    assert app.current_document_id == other_id

    with mock.patch.object(
        app.session, "reset_document", wraps=app.session.reset_document
    ) as verb:
        app.reset_document_for_confirmed(target)
        assert verb.call_count == 0, "the menu restarted the document with no confirm"
        assert app.modal is ModalId.CONFIRM
        assert app.confirm is not None and app.confirm.verb == "Reset"
        app.confirm.on_confirm()
    assert verb.call_count == 1
    assert verb.call_args.args[:1] == (target,), (
        "the confirm restarted the current document, not the tile's"
    )


def test_the_chat_clear_verb_asks_first(app: Any) -> None:
    with mock.patch.object(
        app, "copilot_clear_chat", wraps=app.copilot_clear_chat
    ) as verb:
        app.copilot_clear_chat_confirmed()
        assert verb.call_count == 0
        assert app.modal is ModalId.CONFIRM
        assert app.confirm is not None and app.confirm.verb == "Clear"
        app.confirm.on_confirm()
    assert verb.call_count == 1


def test_the_chats_clear_button_asks_first(app: Any, monkeypatch: Any) -> None:
    """The chat's own Clear control routes through the confirming verb.

    Falsifier: call `app.copilot_clear_chat()` from the button -- the transcript and its
    checkpoints go on one click.
    """
    real_danger = copilot_chat.danger_button

    def clicking(label: str, *args: Any, **kwargs: Any) -> bool:
        real_danger(label, *args, **kwargs)
        return label.split("##", 1)[0] == command_label(CommandId.CLEAR_COPILOT_CHAT)

    monkeypatch.setattr(copilot_chat, "danger_button", clicking)
    with mock.patch.object(
        app, "copilot_clear_chat", wraps=app.copilot_clear_chat
    ) as verb:
        imgui.new_frame()
        imgui.begin("chat_rig")
        copilot_chat._draw_top_bar(app)
        imgui.end()
        imgui.end_frame()
    assert verb.call_count == 0, (
        "the Clear button dropped the transcript with no confirm"
    )
    assert app.modal is ModalId.CONFIRM
    assert app.confirm is not None and app.confirm.verb == "Clear"


def test_the_revert_glyph_asks_before_reverting(app: Any) -> None:
    """`open_copilot_revert` builds the request with the message's own head. Falsifier:
    revert inside the opener -- the verb spy counts one before the confirm."""
    msg = Message(role="user", text="make the blur wider", turn_id="t1")
    with mock.patch.object(app, "revert_turn", wraps=app.revert_turn) as verb:
        app.open_copilot_revert(msg)
        assert verb.call_count == 0
        assert app.modal is ModalId.CONFIRM
        assert app.confirm is not None
        assert app.confirm.verb == "Revert"
        assert "make the blur wider" in app.confirm.title
        app.confirm.on_confirm()
    assert verb.call_count == 1


def test_the_lib_tree_delete_asks_through_its_menu(app: Any, monkeypatch: Any) -> None:
    """The tree's Delete item routes through `request_confirm` with the trash line.

    Falsifier: call `shader_lib_files.delete_file` from the item -- the file is gone with
    no confirm.
    """
    victim = shader_lib_root() / "confirm_probe.glsl"
    victim.write_text("float SB_probe() { return 1.0; }\n", encoding="utf-8")

    with mock.patch.object(
        app.shader_lib_files, "delete_file", wraps=app.shader_lib_files.delete_file
    ) as verb:
        tree._confirm_file_delete(app, victim)
        assert verb.call_count == 0, "the item deleted with no confirm"
        assert victim.exists()
        assert app.modal is ModalId.CONFIRM
        assert app.confirm is not None
        assert app.confirm.title == f"Delete {victim.name}?"
        app.confirm.on_confirm()
    assert verb.call_count == 1
    assert not victim.exists()


def _fire_tree_item(monkeypatch: Any, str_id: str, label: str) -> None:
    """Open ONE of the lib tree's context menus and fire ONE of its items, by id and label.

    The item bodies are only submitted while their popup is open, and a frame-driven
    right-click on a tree row cannot be aimed at a row whose rect the test does not know --
    so the popup gate and the click are supplied for that one menu, and the item's own body
    is what runs. `end_popup` is neutered only for the frames this menu is faked open, since
    nothing was pushed for it.
    """
    faked: list[bool] = []
    real_begin = imgui.begin_popup_context_item
    real_end = imgui.end_popup

    def begin(item_id: Any = None, *args: Any, **kwargs: Any) -> bool:
        if item_id == str_id:
            faked.append(True)
            return True
        return real_begin(item_id, *args, **kwargs)

    def end() -> None:
        if faked:
            faked.pop()
            return
        real_end()

    monkeypatch.setattr(imgui, "begin_popup_context_item", begin)
    monkeypatch.setattr(imgui, "end_popup", end)
    monkeypatch.setattr(
        imgui,
        "menu_item_simple",
        lambda item_label, *args, **kwargs: item_label == label,
    )


def test_a_confirm_from_inside_the_picker_leaves_the_picker_below(
    app: Any, monkeypatch: Any
) -> None:
    """The lib tree's Delete runs INSIDE the open picker, so its confirm is STACKED: the file
    is trashed and the user is back in the picker, with the rename it had armed still armed
    and `close_lib_picker` never run.

    Falsifier: open the confirm unstacked (`request_confirm` without `stacked=True`) --
    `app.modal` is None after the confirm and the picker is gone.
    """
    victim = shader_lib_root() / "stacked_probe.glsl"
    victim.write_text("float SB_stacked() { return 1.0; }\n", encoding="utf-8")
    app.open_shader_lib_picker()
    app.shader_lib_files.begin_file_rename(victim)
    assert app.shader_lib_files.inline_input_owns_esc()

    with mock.patch.object(
        app, "close_lib_picker", wraps=app.close_lib_picker
    ) as cleanup:
        tree._confirm_file_delete(app, victim)
        assert app.modal is ModalId.CONFIRM
        assert app.modal_below is ModalId.SHADER_LIB_PICKER
        assert victim.exists(), "the item deleted with no confirm"

        _release(imgui.Key.enter)
        _pump(app)
        _press(imgui.Key.enter)
        _pump(app)
        _release(imgui.Key.enter)

        assert not victim.exists(), "confirming did not trash the file"
        assert app.modal is ModalId.SHADER_LIB_PICKER, (
            "confirming closed the picker the confirm was asked from"
        )
        assert app.modal_below is None
        assert cleanup.call_count == 0, (
            "the picker's on_close ran while the picker is still open"
        )
    assert app.shader_lib_files.inline_input_owns_esc(), (
        "the armed rename did not survive the confirm"
    )


def test_escape_on_a_stacked_confirm_returns_to_the_picker(app: Any) -> None:
    """Cancelling lands the user back in the picker with nothing trashed.

    Falsifier: have `close_modal` write `None` rather than `app.modal_below`.
    """
    victim = shader_lib_root() / "stacked_cancel_probe.glsl"
    victim.write_text("float SB_cancel() { return 1.0; }\n", encoding="utf-8")
    app.open_shader_lib_picker()
    tree._confirm_file_delete(app, victim)
    assert app.modal is ModalId.CONFIRM

    assert close_modal(app) is True
    assert app.modal is ModalId.SHADER_LIB_PICKER
    assert app.modal_below is None
    assert app.confirm is None, "the confirm's own cleanup did not run"
    assert victim.exists(), "Escape trashed the file"


def test_a_stacked_confirm_leaves_the_chat_focus_capture_alone(app: Any) -> None:
    """The capture belongs to the modal that was opened from the app, not to the confirm
    stacked over it: a stacked open reads `copilot_focused` mid-draw, where it is False by
    construction, so re-capturing would send a chat user back to the editor.

    Falsifier: drop the `stacked` branch from `_open_modal` -- the flag reads False after the
    confirm.
    """
    victim = shader_lib_root() / "focus_probe.glsl"
    victim.write_text("float SB_focus() { return 1.0; }\n", encoding="utf-8")
    app.copilot_focused = True
    app.open_shader_lib_picker()
    assert app._chat_focused_before_popup is True
    app.copilot_focused = False

    tree._confirm_file_delete(app, victim)
    assert app._chat_focused_before_popup is True, (
        "the stacked confirm clobbered the pre-popup chat focus"
    )
    close_modal(app)
    assert app._chat_focused_before_popup is True


def test_the_lib_tree_delete_item_asks_from_inside_the_open_picker(
    app: Any, monkeypatch: Any
) -> None:
    """The real menu item, submitted from inside the picker's own body: the file survives the
    click and the confirm stacks over the picker.

    Falsifier: call `shader_lib_files.delete_file` from the item -- the file is gone with no
    confirm.
    """
    victim = shader_lib_root() / "menu_probe.glsl"
    victim.write_text("float SB_menu() { return 1.0; }\n", encoding="utf-8")
    app.session.rebuild_shader_lib_index()
    app.open_shader_lib_picker()
    _fire_tree_item(monkeypatch, f"##filectx_{victim}", "Delete")

    with mock.patch.object(
        app.shader_lib_files, "delete_file", wraps=app.shader_lib_files.delete_file
    ) as verb:
        _pump(app)
    assert verb.call_count == 0, "the item deleted with no confirm"
    assert victim.exists()
    assert app.modal is ModalId.CONFIRM
    assert app.modal_below is ModalId.SHADER_LIB_PICKER
    assert app.confirm is not None
    assert app.confirm.title == f"Delete {victim.name}?"


def test_the_lib_tree_dir_delete_asks_through_its_menu(app: Any) -> None:
    """The directory item's own confirm, which had no test at all.

    Falsifier: call `shader_lib_files.delete_dir` from `_confirm_dir_delete`.
    """
    victim = shader_lib_root() / "confirm_probe_dir"
    victim.mkdir(exist_ok=True)

    with mock.patch.object(
        app.shader_lib_files, "delete_dir", wraps=app.shader_lib_files.delete_dir
    ) as verb:
        tree._confirm_dir_delete(app, victim)
        assert verb.call_count == 0, "the item deleted with no confirm"
        assert victim.exists()
        assert app.modal is ModalId.CONFIRM
        assert app.confirm is not None
        assert app.confirm.title == f"Delete {victim.name}?"
        assert app.confirm.verb == "Delete"
        app.confirm.on_confirm()
    assert verb.call_count == 1
    assert not victim.exists()


# ---------------------------------------------------------------------------
# R2 — the close funnel, per registry row
# ---------------------------------------------------------------------------

_OPENERS: dict[ModalId, Any] = {
    ModalId.EXAMPLES: lambda app: app.open_examples(),
    ModalId.HELP: lambda app: app.open_help(),
    ModalId.SETTINGS: lambda app: app.open_settings(),
    ModalId.PASS_SETTINGS: lambda app: app.open_add_pass(),
    ModalId.IMPORT_PASSES: lambda app: app.open_import_passes(),
    ModalId.EMOJI_PICKER: lambda app: app.open_emoji_picker(lambda glyph: None),
    ModalId.SHADER_LIB_PICKER: lambda app: app.open_shader_lib_picker(),
    ModalId.PROJECTS: lambda app: app.open_projects(),
    ModalId.CONFIRM: lambda app: app.request_confirm(_request(lambda: None)),
}

_WITH_CLEANUP = [modal for modal in MODALS if modal.on_close is not None]
_WITH_ESC_OWNER = [modal for modal in MODALS if modal.owns_esc is not None]


# The `App` method each row's `on_close` calls -- spying it is what makes "the row lost its
# cleanup" red, since a dropped `on_close` never reaches the method.
_CLEANUP_METHOD: dict[ModalId, str] = {
    ModalId.SETTINGS: "apply_editor_settings",
    ModalId.PASS_SETTINGS: "close_pass_settings",
    ModalId.IMPORT_PASSES: "close_import_passes",
    ModalId.EMOJI_PICKER: "close_emoji_picker",
    ModalId.SHADER_LIB_PICKER: "close_lib_picker",
    ModalId.PROJECTS: "reset_projects_state",
    ModalId.CONFIRM: "clear_confirm",
}


def test_every_row_that_owns_state_carries_a_cleanup() -> None:
    """The parametrized test below derives its cases from the rows that HAVE an `on_close`,
    so a row that loses one would silently drop its own case. This pins the set instead.

    Falsifier: drop `on_close` from `emoji_picker.MODAL` -- the picker's cleanup then never
    runs and no case is left to notice.
    """
    assert {modal.id for modal in _WITH_CLEANUP} == set(_CLEANUP_METHOD), (
        f"rows with on_close: {sorted(m.id.value for m in _WITH_CLEANUP)}; "
        f"named here: {sorted(k.value for k in _CLEANUP_METHOD)}"
    )


@pytest.mark.parametrize(
    "modal_id",
    sorted(_CLEANUP_METHOD),
    ids=[modal_id.value for modal_id in sorted(_CLEANUP_METHOD)],
)
def test_the_funnel_runs_each_rows_cleanup(app: Any, modal_id: ModalId) -> None:
    """Every row with an `on_close` runs its cleanup exactly once through the funnel.

    Falsifier: drop `on_close` from one row -- the spy on its `App` method reads 0, and the
    state the cleanup owns (the emoji target, the import draft, the picker's rename) survives
    a close.
    """
    assert BY_ID[modal_id].on_close is not None
    method = _CLEANUP_METHOD[modal_id]
    _OPENERS[modal_id](app)
    assert app.modal is modal_id
    with mock.patch.object(app, method, wraps=getattr(app, method)) as cleanup:
        assert close_modal(app) is True
        count = cleanup.call_count
    assert count == 1, f"{modal_id.value}'s {method} ran {count} times"
    assert app.modal is None


@pytest.mark.parametrize(
    "modal_id",
    [modal.id for modal in _WITH_ESC_OWNER],
    ids=[modal.id.value for modal in _WITH_ESC_OWNER],
)
def test_an_armed_inline_input_declines_escape(app: Any, modal_id: ModalId) -> None:
    """Esc belongs to the armed input, whose own cancel runs later in the same frame.

    Falsifier: drop `owns_esc` from one row -- Esc then closes the whole modal out from
    under a half-typed name.
    """
    modal = BY_ID[modal_id]
    assert modal.owns_esc is not None
    _OPENERS[modal_id](app)
    _arm_inline_input(app, modal_id)
    assert modal.owns_esc(app), "the probe did not arm an inline input"

    assert close_modal(app) is False, "an unforced close ignored owns_esc"
    assert app.modal is modal_id


def test_the_lib_pickers_close_button_works_with_a_rename_armed(
    app: Any, monkeypatch: Any
) -> None:
    """A Close the user CLICKED is forced: the armed rename cannot refuse a dismissal.

    Driven through the REAL `draw_modal`, with the picker's own Close button reporting the
    click -- the whole point is that the funnel's `forced` flag reaches this path. Falsifier:
    drop `forced` from `draw_modal`'s call to `close_modal` -- the Close button goes dead
    and the picker the user dismissed stays on screen.
    """
    app.open_shader_lib_picker()
    app.shader_lib_files.begin_file_rename(shader_lib_root() / "probe.glsl")
    assert app.shader_lib_files.inline_input_owns_esc()

    real_standard = lib_picker.standard_button

    def clicking(label: str, *args: Any, **kwargs: Any) -> bool:
        real_standard(label, *args, **kwargs)
        return label.split("##", 1)[0] == "Close"

    monkeypatch.setattr(lib_picker, "standard_button", clicking)
    _pump(app)
    assert app.modal is None, "the Close the user clicked left the picker on screen"
    assert not app.shader_lib_files.inline_input_owns_esc(), (
        "the row's on_close did not cancel the armed rename"
    )


def _arm_inline_input(app: Any, modal_id: ModalId) -> None:
    if modal_id is ModalId.PROJECTS:
        app.projects_new_input.open(app.default_projects_root_dir, "half typed")
        return
    if modal_id is ModalId.SHADER_LIB_PICKER:
        app.shader_lib_files.begin_file_rename(shader_lib_root() / "probe.glsl")
        return
    raise AssertionError(f"no inline-input probe for {modal_id.value}")


def test_escape_reaches_the_funnel_through_the_hotkey_dispatch(app: Any) -> None:
    """The WIRE the funnel hangs from: `hotkeys._handle_escape` calls `close_modal`.

    Driven through a real frame with Esc pressed, since the funnel being correct says
    nothing about whether anything calls it. Falsifier: replace the `close_modal(app)` call
    in `_handle_escape` with `pass` -- Esc then dismisses nothing, on every modal.
    """
    app.open_emoji_picker(lambda glyph: None)
    app.emoji_picker_query = "smi"
    assert app.modal is ModalId.EMOJI_PICKER

    imgui.get_io().add_key_event(imgui.Key.escape, True)
    imgui.new_frame()
    hotkeys._handle_escape(app)
    imgui.end_frame()
    imgui.get_io().add_key_event(imgui.Key.escape, False)

    assert app.modal is None, "Esc did not reach the close funnel"
    assert app.emoji_pick_target is None, (
        "the row's cleanup did not run on the Esc path"
    )
    assert app.emoji_picker_query == ""


def test_the_confirm_modal_needs_no_popup_import_in_app() -> None:
    """R4's payload type lives in `ui_models.py`, which is what keeps the registry a leaf.

    Falsifier: move `ConfirmRequest` into `popups/confirm.py` -- `app.py` must then import
    the popups layer, which `test_modal_chrome` rejects.
    """
    source = (
        Path(__file__).resolve().parent.parent / "shaderbox/popups/confirm.py"
    ).read_text(encoding="utf-8")
    assert "class ConfirmRequest" not in source
    assert ConfirmRequest.__module__ == "shaderbox.ui_models"
