"""The pass-settings modal: what fills one pass's inputs, and the target it draws into (065).

Wiring and target controls live here, off the strip — they are set-up-once choices, and as an
always-open block under the strip they spent the panel's vertical budget on rows nobody touches
after the first minute. The modal opens from a tile's gear, the tile's context menu, and
automatically when a pass is created (the one moment the choices are actually made).

Wiring is a closed-set combo over the document's own pass names, never a free-text field, so an
input can never name a pass that does not exist. That is what makes SHADERed's positional-slot
footgun impossible here, and it is why an unfilled input reading black (D3) stays a state you are
building toward rather than a typo you cannot see.

The combo carries two synthetic items ahead of the names (069 D9): `auto: <x>` is what the
uniform's own NAME resolves to and stores no key, `(none)` stores an explicit black the name rule
must not undo. Three stored states, three distinct readings -- a combo that showed one label for
a working wire and a black one is what this replaced.
"""

from imgui_bundle import imgui
from OpenGL.GL import GL_SAMPLER_2D

from shaderbox.app import App, PopupState
from shaderbox.core import Pass
from shaderbox.document import _is_user_bound
from shaderbox.pass_graph import MAX_ITERATIONS, PassEntry, effective_inputs
from shaderbox.theme import COLOR, SIZE, SPACE
from shaderbox.ui_primitives import (
    ghost_button,
    help_marker,
    label_row,
    modal_window,
)

_LABEL = "Pass settings##popup"
_ROW_LABEL_W = 110.0
_CTRL_W = 168.0
_UNWIRED = "(none)"

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


def draw_pass_settings(app: App) -> None:
    if app.popup_state != PopupState.PASS_SETTINGS:
        return
    # `always_auto_resize` IGNORES set_next_window_size, so the width token only holds
    # through a constraint: min and max both PASS_SETTINGS_W pins the axis the user reads
    # across, while the height follows the content up to the display. The scrollbar is left
    # enabled deliberately -- it can only appear once content exceeds the display, which is
    # exactly when the user needs to be told the panel continues.
    display_h = imgui.get_io().display_size.y
    width = float(SIZE.PASS_SETTINGS_W)
    imgui.set_next_window_size_constraints(
        (width, 0.0), (width, max(1.0, display_h - float(SIZE.PASS_SETTINGS_MARGIN)))
    )
    with modal_window(
        _LABEL,
        (width, 0.0),
        flags=imgui.WindowFlags_.always_auto_resize,
    ) as visible:
        if not visible:
            return
        if not _draw_body(app):
            app.close_pass_settings()
            imgui.close_current_popup()


def _draw_body(app: App) -> bool:
    document_id = app.current_document_id
    ui_document = app.ui_documents.get(document_id)
    name = app.pass_settings_name
    if ui_document is None or name not in ui_document.document.passes:
        return False

    document = ui_document.document
    imgui.separator_text("Pass")
    renamed = _draw_name(app, document_id, name)
    if not renamed:
        imgui.dummy((0.0, float(SPACE.MD)))
        _draw_inputs(app, document_id, name, document.passes[name])
        imgui.dummy((0.0, float(SPACE.MD)))
        _draw_target(app, document_id, name)

    imgui.dummy((0.0, float(SPACE.MD)))
    return not ghost_button("Close")


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


def _sampler_names(render_pass: Pass) -> list[str]:
    return [
        u.name
        for u in render_pass.get_active_uniforms()
        if getattr(u, "gl_type", None) == GL_SAMPLER_2D
    ]


def _draw_inputs(app: App, document_id: str, name: str, render_pass: Pass) -> None:
    # One closed-set combo per sampler uniform the pass actually declares, with two synthetic
    # items in front of the pass names: the name rule's own answer, and an explicit none. Neither
    # label is a reachable pass name (`_PASS_NAME_RE` admits no colon, space or parenthesis), so
    # indexing this mixed list is safe.
    imgui.separator_text("Reads")
    samplers = _sampler_names(render_pass)
    if not samplers:
        imgui.text_colored(COLOR.FG_DIM, "no sampler2D uniforms")
        return
    document = app.ui_documents[document_id].document
    entry = document.graph.passes.get(name, PassEntry())
    names = set(document.passes)
    bound = [
        uniform
        for uniform in samplers
        if _is_user_bound(render_pass.uniform_values.get(uniform))
    ]
    for uniform in samplers:
        undecided = entry.model_copy(
            update={"inputs": {u: s for u, s in entry.inputs.items() if u != uniform}}
        )
        auto = effective_inputs(undecided, [uniform], names, name, bound).get(
            uniform, ""
        )
        choices = [f"auto: {auto or 'none'}", _UNWIRED, *sorted(document.passes)]
        stored = entry.inputs.get(uniform)
        if stored is None:
            index = 0
        elif stored == "":
            index = 1
        else:
            # A stale explicit name -- its pass is gone -- reads as `(none)`, which is what it
            # renders as; picking anything then rewrites the key to something valid.
            index = choices.index(stored) if stored in choices else 1
        label_row(app.font_12, uniform, _CTRL_W, _ROW_LABEL_W)
        changed, picked = imgui.combo(f"##wire_{name}_{uniform}", index, choices)
        if not changed:
            continue
        if picked == 0:
            error = app.session.unwire_pass_input(document_id, name, uniform)
        else:
            producer = "" if picked == 1 else choices[picked]
            error = app.session.wire_pass_input(document_id, name, uniform, producer)
        if error:
            app.notifications.push(error)


def _draw_target(app: App, document_id: str, name: str) -> None:
    # What this pass DRAWS INTO. Every control names the thing it changes about the picture, not
    # the field it writes: a target's dtype is "format / 16-bit float", not "f2".
    document = app.ui_documents[document_id].document
    entry = document.graph.passes.get(name, PassEntry())
    target = entry.target
    new_target = target
    is_output = name == document.graph.output

    imgui.separator_text("Draws into")

    label_row(app.font_12, "format", _CTRL_W, _ROW_LABEL_W)
    changed, picked = imgui.combo(
        f"##dtype_{name}", _FORMAT_CODES.index(target.dtype), _FORMAT_LABELS
    )
    if changed:
        new_target = target.model_copy(update={"dtype": _FORMAT_CODES[picked]})
    imgui.same_line()
    help_marker(_FORMATS[_FORMAT_CODES.index(target.dtype)][2])

    canvas_w, canvas_h = document.canvas_size
    w = max(1, round(canvas_w * target.scale))
    h = max(1, round(canvas_h * target.scale))
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

    if new_target != target:
        error = app.session.set_pass_target(document_id, name, new_target)
        if error:
            app.notifications.push(error)

    _draw_repeat(app, document_id, name, entry, document.canvas_size)


def _draw_repeat(
    app: App,
    document_id: str,
    name: str,
    entry: PassEntry,
    canvas_size: tuple[int, int],
) -> None:
    # How many times this pass draws per frame (068). Named for what it does to the picture --
    # "runs" of the same shader, each seeing the one before it -- not for the model field.
    imgui.separator_text("Runs")

    label_row(app.font_12, "runs", _CTRL_W, _ROW_LABEL_W)
    changed, runs = imgui.slider_int(
        f"##iterations_{name}", entry.iterations, 1, MAX_ITERATIONS
    )
    imgui.same_line()
    help_marker("redraws per frame, each reading the last")
    if changed and runs != entry.iterations:
        error = app.session.set_pass_iterations(document_id, name, runs)
        if error:
            app.notifications.push(error)

    # No engine-side "your count is short" warning: the engine cannot tell a base-2 jump flood
    # from the base-4 cascade stack beside it, so a check assuming one warns falsely on the
    # other -- it fired on this repo's own shipped cascade pass at its shipped size. The help
    # text explains the number; the shader's author owns it.
