"""The pass-settings modal: a pass's name, how many times it runs, and the target it draws
into (065, 068), and the one place a pass is CREATED (078 D5).

These are set-up-once choices, and as an always-open block under the strip they spent the
panel's vertical budget on rows nobody touches after the first minute. The modal opens from a
tile's gear, the tile's context menu, and from `add pass` — in create mode it edits a draft
(`App.pass_draft`) that becomes a pass only on `Create`; `Cancel` or Escape makes nothing.
What a pass READS is not here: a sampler's source is chosen on its own row of the uniforms
panel (072).
"""

from imgui_bundle import imgui

from shaderbox.app import App, ModalId
from shaderbox.pass_graph import MAX_ITERATIONS, PassEntry, PassGraph
from shaderbox.popups import Modal
from shaderbox.theme import SIZE, SPACE
from shaderbox.ui_primitives import (
    ModalSizing,
    help_marker,
    label_row,
    modal_footer,
    primary_button,
    standard_button,
)

_LABEL = "Pass settings##popup"
_ROW_LABEL_W = 110.0
_CTRL_W = 168.0

# What a pass's target format MEANS, in place of moderngl's `f1`/`f2`/`f4` dtype strings. The
# tuple is (code, menu label, what it is for) — the label is what a person picks from, the last
# half is the tooltip, because "16-bit" alone does not tell you when to want it.
_FORMATS: list[tuple[str, str, str]] = [
    ("f1", "8-bit", "clamps to 0-1, the smallest"),
    ("f2", "16-bit float", "holds values above 1, the default"),
    ("f4", "32-bit float", "full precision, twice the memory"),
]
_FORMAT_LABELS = [label for _, label, _ in _FORMATS]
_FORMAT_CODES = [code for code, _, _ in _FORMATS]


def _size(app: App) -> tuple[float, float]:
    _ = app
    return (float(SIZE.PASS_SETTINGS_W), 0.0)


def _draw_modal_body(app: App) -> bool:
    """One modal, two modes: the draft is what `add pass` opens, the pass what the gear does."""
    return _draw_draft(app) if app.pass_draft is not None else _draw_body(app)


def _draw_draft(app: App) -> bool:
    """Create mode: every control edits the draft; nothing reaches the document until
    `Create`. Enter in the name field is `Create` too."""
    draft = app.pass_draft
    assert draft is not None
    document_id = app.current_document_id
    ui_document = app.ui_documents.get(document_id)
    if ui_document is None:
        return False

    imgui.separator_text("Pass")
    label_row(app.font_12, "name", _CTRL_W, _ROW_LABEL_W)
    if draft.needs_focus:
        imgui.set_keyboard_focus_here(0)
        draft.needs_focus = False
    entered, draft.name_buf = imgui.input_text(
        "##pass_draft_name",
        draft.name_buf,
        flags=imgui.InputTextFlags_.enter_returns_true,
    )
    imgui.same_line()
    help_marker("names its shader file and its wires")
    group, _ = _draw_group(
        app, "draft", draft.entry.group, _existing_groups(ui_document.document.graph)
    )
    if group != draft.entry.group:
        draft.entry = draft.entry.model_copy(update={"group": group})

    imgui.dummy((0.0, float(SPACE.MD)))
    draft.entry = _draw_target(
        app, "draft", draft.entry, ui_document.document.canvas_size, is_output=False
    )
    draft.entry = _draw_repeat(app, "draft", draft.entry)

    with modal_footer():
        created = (primary_button("Create") or entered) and app.create_pass_from_draft()
        imgui.same_line()
        keep_open = not (created or standard_button("Cancel"))
    return keep_open


def _draw_body(app: App) -> bool:
    document_id = app.current_document_id
    ui_document = app.ui_documents.get(document_id)
    name = app.pass_settings_name
    if ui_document is None or name not in ui_document.document.passes:
        return False

    imgui.separator_text("Pass")
    renamed = _draw_name(app, document_id, name)
    if not renamed:
        document = ui_document.document
        entry = document.graph.passes.get(name, PassEntry())
        app.pass_settings_group_buf, committed = _draw_group(
            app, name, app.pass_settings_group_buf, _existing_groups(document.graph)
        )
        if committed:
            app.commit_pass_group()
        imgui.dummy((0.0, float(SPACE.MD)))
        new_entry = _draw_target(
            app,
            name,
            entry,
            document.canvas_size,
            is_output=name == document.graph.output,
        )
        new_entry = _draw_repeat(app, name, new_entry)
        _apply_entry(app, document_id, name, entry, new_entry)

    with modal_footer():
        keep_open = not standard_button("Close")
    return keep_open


def _apply_entry(
    app: App, document_id: str, name: str, before: PassEntry, after: PassEntry
) -> None:
    # Edit mode writes straight through the session's verbs, one per changed field, so the
    # document (and its `graph.json`) follows every control the frame it moves.
    if after.target != before.target:
        error = app.session.set_pass_target(document_id, name, after.target)
        if error:
            app.notifications.push(error)
    if after.iterations != before.iterations:
        error = app.session.set_pass_iterations(document_id, name, after.iterations)
        if error:
            app.notifications.push(error)


def _draw_name(app: App, document_id: str, name: str) -> bool:
    """Draw the name row; return True when a rename landed this frame (`name` is now dead)."""
    # A successful rename moves the file and every edge, and `_on_pass_renamed` re-points this
    # modal's own target (App.pass_settings_name) so the next frame draws the live pass.
    label_row(app.font_12, "name", _CTRL_W, _ROW_LABEL_W)
    committed, app.pass_settings_name_buf = imgui.input_text(
        "##pass_settings_name",
        app.pass_settings_name_buf,
        flags=imgui.InputTextFlags_.enter_returns_true,
    )
    # Read on the line after the input: the item-scoped queries see the LAST submitted item.
    deactivated = imgui.is_item_deactivated_after_edit()
    renamed = (
        _commit_pass_name(app, document_id, name) if committed or deactivated else False
    )
    imgui.same_line()
    help_marker("names its shader file and its wires")
    return renamed


def _existing_groups(graph: PassGraph) -> list[str]:
    return sorted({entry.group for entry in graph.passes.values() if entry.group})


def _draw_group(
    app: App, id_: str, group: str, existing: list[str]
) -> tuple[str, bool]:
    """The group row (091 D9): a field beside a picker of the document's groups. Returns the
    buffer as the row left it and whether it was committed this frame (Enter, leaving the
    field, or a pick)."""
    label_row(app.font_12, "group", _CTRL_W, _ROW_LABEL_W)
    entered, group = imgui.input_text(
        f"##group_{id_}", group, flags=imgui.InputTextFlags_.enter_returns_true
    )
    deactivated = imgui.is_item_deactivated_after_edit()
    picked = False
    imgui.same_line()
    imgui.begin_disabled(not existing)
    if imgui.begin_combo(f"##group_pick_{id_}", "", imgui.ComboFlags_.no_preview):
        for name in existing:
            if imgui.selectable(name, name == group)[0]:
                group, picked = name, True
        if imgui.selectable("none", group == "")[0]:
            group, picked = "", True
        imgui.end_combo()
    imgui.end_disabled()
    imgui.same_line()
    help_marker("the tile's group label")
    return group, entered or deactivated or picked


def _commit_pass_name(app: App, document_id: str, name: str) -> bool:
    """Rename `name` to the buffer's text; return True only when the rename landed."""
    new_name = app.pass_settings_name_buf.strip()
    if not new_name or new_name == name:
        app.pass_settings_name_buf = name
        return False
    error = app.session.rename_pass(document_id, name, new_name)
    if error:
        app.notifications.push(error)
        app.pass_settings_name_buf = name
        return False
    return True


def _draw_target(
    app: App,
    name: str,
    entry: PassEntry,
    canvas_size: tuple[int, int],
    is_output: bool,
) -> PassEntry:
    """What this pass DRAWS INTO, drawn over `entry`; returns the entry as the controls left
    it (the caller decides whether that is a document write or a draft)."""
    # Every control names the thing it changes about the picture, not the field it writes: a
    # target's dtype is "format / 16-bit float", not "f2".
    target = entry.target
    new_target = target

    imgui.separator_text("Draws into")

    label_row(app.font_12, "format", _CTRL_W, _ROW_LABEL_W)
    changed, picked = imgui.combo(
        f"##dtype_{name}", _FORMAT_CODES.index(target.dtype), _FORMAT_LABELS
    )
    if changed:
        new_target = target.model_copy(update={"dtype": _FORMAT_CODES[picked]})
    imgui.same_line()
    help_marker(_FORMATS[_FORMAT_CODES.index(target.dtype)][2])

    # The OUTPUT keeps full size and its `scale` is ignored, which is why the slider below is
    # disabled for it -- so the label has to say the size the canvas HAS, not the one the stored
    # scale would imply.
    w, h = canvas_size if is_output else target.target_size(canvas_size)
    label_row(app.font_12, "size", _CTRL_W, _ROW_LABEL_W)
    # The slider runs over 5-100 so `%.0f%%` formats the number a person reads; the model
    # keeps the 0-1 scale. The derived dims ride the format string, not the label column.
    imgui.begin_disabled(is_output)
    scale_changed, percent = imgui.slider_float(
        f"##scale_{name}",
        target.scale * 100.0,
        5.0,
        100.0,
        f"%.0f%% · {w}x{h}",
    )
    imgui.end_disabled()
    if scale_changed:
        new_target = new_target.model_copy(update={"scale": percent / 100.0})
    imgui.same_line()
    help_marker("share of the canvas, output always full")

    label_row(app.font_12, "sampling", _CTRL_W, _ROW_LABEL_W)
    smooth_changed, smooth = imgui.checkbox(f"smooth##{name}", target.filter_linear)
    if smooth_changed:
        new_target = new_target.model_copy(update={"filter_linear": smooth})

    label_row(app.font_12, "edges", _CTRL_W, _ROW_LABEL_W)
    tile_changed, tile = imgui.checkbox(f"repeat##{name}", target.wrap)
    if tile_changed:
        new_target = new_target.model_copy(update={"wrap": tile})

    return (
        entry
        if new_target == target
        else entry.model_copy(update={"target": new_target})
    )


def _draw_repeat(app: App, name: str, entry: PassEntry) -> PassEntry:
    # How many times this pass draws per frame (068). Named for what it does to the picture --
    # "runs" of the same shader, each seeing the one before it -- not for the model field.
    imgui.separator_text("Runs")

    label_row(app.font_12, "runs", _CTRL_W, _ROW_LABEL_W)
    changed, runs = imgui.slider_int(
        f"##iterations_{name}", entry.iterations, 1, MAX_ITERATIONS
    )
    imgui.same_line()
    help_marker("redraws per frame, each reading the last")

    # No engine-side "your count is short" warning: the engine cannot tell a base-2 jump flood
    # from the base-4 cascade stack beside it, so a check assuming one warns falsely on the
    # other -- it fired on this repo's own shipped cascade pass at its shipped size. The help
    # text explains the number; the shader's author owns it.
    if changed and runs != entry.iterations:
        return entry.model_copy(update={"iterations": runs})
    return entry


MODAL = Modal(
    id=ModalId.PASS_SETTINGS,
    label=_LABEL,
    size=_size,
    body=_draw_modal_body,
    sizing=ModalSizing.AUTO,
    on_close=lambda app: app.close_pass_settings(),
)
