"""The pass-settings modal: what fills one pass's inputs, and the target it draws into (065).

Wiring and target controls live here, off the strip — they are set-up-once choices, and as an
always-open block under the strip they spent the panel's vertical budget on rows nobody touches
after the first minute. The modal opens from a tile's gear, the tile's context menu, and
automatically when a pass is created (the one moment the choices are actually made).

Wiring is a closed-set combo over the document's own pass names, never a free-text field, so an
input can never name a pass that does not exist. That is what makes SHADERed's positional-slot
footgun impossible here, and it is why an unfilled input reading black (D3) stays a state you are
building toward rather than a typo you cannot see.
"""

from imgui_bundle import imgui
from OpenGL.GL import GL_SAMPLER_2D

from shaderbox.app import App, PopupState
from shaderbox.core import Pass
from shaderbox.pass_graph import (
    MAX_ITERATIONS,
    PassEntry,
    PassGraph,
    iteration_shortfalls,
)
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
    (
        "f1",
        "8-bit",
        "Smallest. Values clamp to 0-1 — the right choice for a final image.",
    ),
    (
        "f2",
        "16-bit float",
        "Holds values above 1 (bright highlights, accumulated light). The default, and what "
        "bloom and feedback need.",
    ),
    (
        "f4",
        "32-bit float",
        "Full precision. Rarely needed; costs twice the memory of 16-bit.",
    ),
]
_FORMAT_LABELS = [label for _, label, _ in _FORMATS]
_FORMAT_CODES = [code for code, _, _ in _FORMATS]


def draw_pass_settings(app: App) -> None:
    if app.popup_state != PopupState.PASS_SETTINGS:
        return
    with modal_window(
        _LABEL, (float(SIZE.PASS_SETTINGS_W), float(SIZE.PASS_SETTINGS_H))
    ) as visible:
        if not visible:
            return
        if not _draw_body(app):
            app.popup_state = PopupState.CLOSED
            app.pass_settings_name = ""
            imgui.close_current_popup()


def _draw_body(app: App) -> bool:
    document_id = app.current_document_id
    ui_document = app.ui_documents.get(document_id)
    name = app.pass_settings_name
    if ui_document is None or name not in ui_document.document.passes:
        return False

    document = ui_document.document
    imgui.separator_text("Pass")
    _draw_name(app, document_id, name)
    imgui.dummy((0.0, float(SPACE.MD)))

    _draw_inputs(app, document_id, name, document.passes[name])
    imgui.dummy((0.0, float(SPACE.MD)))
    _draw_target(app, document_id, name)

    imgui.dummy((0.0, float(SPACE.MD)))
    return not ghost_button("Close")


def _draw_name(app: App, document_id: str, name: str) -> None:
    # Enter commits; a successful rename moves the file and every edge, and `_on_pass_renamed`
    # re-points this modal's own target (App.pass_settings_name) so the body keeps drawing.
    label_row(app.font_12, "name", _CTRL_W, _ROW_LABEL_W)
    committed, app.pass_settings_name_buf = imgui.input_text(
        "##pass_settings_name",
        app.pass_settings_name_buf,
        flags=imgui.InputTextFlags_.enter_returns_true,
    )
    if committed:
        new_name = app.pass_settings_name_buf.strip()
        if new_name and new_name != name:
            error = app.session.rename_pass(document_id, name, new_name)
            if error:
                app.notifications.push(error)
    imgui.same_line()
    help_marker(
        "The pass's name: its shader file under passes/ and what other passes' Reads "
        "call it. Enter applies; a rename re-points every wire and open tab."
    )


def _sampler_names(render_pass: Pass) -> list[str]:
    return [
        u.name
        for u in render_pass.get_active_uniforms()
        if getattr(u, "gl_type", None) == GL_SAMPLER_2D
    ]


def _draw_inputs(app: App, document_id: str, name: str, render_pass: Pass) -> None:
    # One closed-set combo per sampler uniform the pass actually declares; one help for the whole
    # section — per-row help repeated the same sentence under every combo. separator_text is an
    # item, so the section title itself hosts the hover; "(?)" in the label marks it.
    imgui.separator_text("Reads (?)")
    if imgui.is_item_hovered(imgui.HoveredFlags_.delay_short):
        imgui.set_tooltip(
            "Every sampler2D uniform this pass declares gets a row here; pick which pass "
            f"fills it ({_UNWIRED} leaves it black). To read something new, declare another "
            "sampler2D in this pass's shader."
        )
    samplers = _sampler_names(render_pass)
    if not samplers:
        imgui.text_colored(
            COLOR.FG_DIM, "nothing — declare a sampler2D uniform to read another pass"
        )
        return
    document = app.ui_documents[document_id].document
    entry = document.graph.passes.get(name, PassEntry())
    choices = [_UNWIRED, *sorted(document.passes)]
    for uniform in samplers:
        current = entry.inputs.get(uniform, "")
        index = choices.index(current) if current in choices else 0
        label_row(app.font_12, uniform, _CTRL_W, _ROW_LABEL_W)
        changed, picked = imgui.combo(f"##wire_{name}_{uniform}", index, choices)
        if changed:
            producer = "" if picked == 0 else choices[picked]
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
    # The derived resolution lives in the LABEL — inside the slider only the percent fits.
    label_row(app.font_12, f"size ({w}, {h})", _CTRL_W, _ROW_LABEL_W)
    # The slider runs over 5-100 so `%.0f%%` formats the number a person reads; the model
    # keeps the 0-1 scale.
    scale_changed, percent = imgui.slider_float(
        f"##scale_{name}",
        target.scale * 100.0,
        5.0,
        100.0,
        "%.0f%%",
    )
    if scale_changed:
        new_target = new_target.model_copy(update={"scale": percent / 100.0})
    imgui.same_line()
    help_marker(
        "How big this pass's own image is, relative to the canvas. Half size is a quarter of "
        "the pixels — the usual choice for a blur, which looks the same and costs less. The "
        "output pass always draws at full size."
        if is_output
        else "How big this pass's own image is, relative to the canvas. Half size is a quarter "
        "of the pixels — the usual choice for a blur, which looks the same and costs less."
    )

    label_row(app.font_12, "sampling", _CTRL_W, _ROW_LABEL_W)
    smooth_changed, smooth = imgui.checkbox(f"smooth##{name}", target.filter_linear)
    if smooth_changed:
        new_target = new_target.model_copy(update={"filter_linear": smooth})
    imgui.same_line()
    help_marker(
        "How another pass reads this one BETWEEN pixels: smooth blends neighbours (right for "
        "a blur or an upscale), off gives hard pixel edges."
    )

    label_row(app.font_12, "edges", _CTRL_W, _ROW_LABEL_W)
    tile_changed, tile = imgui.checkbox(f"repeat##{name}", target.wrap)
    if tile_changed:
        new_target = new_target.model_copy(update={"wrap": tile})
    imgui.same_line()
    help_marker(
        "What a read PAST this pass's edge returns: repeat wraps to the far side (tiling), "
        "off clamps to the edge pixel — which is what a feedback trail wants."
    )

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
    imgui.separator_text("Runs per frame")

    label_row(app.font_12, "runs", _CTRL_W, _ROW_LABEL_W)
    changed, runs = imgui.slider_int(
        f"##iterations_{name}", entry.iterations, 1, MAX_ITERATIONS
    )
    imgui.same_line()
    help_marker(
        "How many times this pass draws each frame, each run reading what the one before it "
        "wrote. One is an ordinary pass. More builds a chain inside a single shader -- a jump "
        "flood, a cascade stack -- with u_pass_iteration telling the shader which run it is "
        "and u_pass_iterations how many there are."
    )
    if changed and runs != entry.iterations:
        error = app.session.set_pass_iterations(document_id, name, runs)
        if error:
            app.notifications.push(error)

    # The D3 warning, shown where the number is SET and against the live canvas: a count that
    # spanned the old canvas goes quietly short after a resize, and the render stays plausible.
    for shortfall in iteration_shortfalls(PassGraph(passes={name: entry}), canvas_size):
        imgui.push_text_wrap_pos(0.0)
        imgui.text_colored(COLOR.STATE_WARN, shortfall.message)
        imgui.pop_text_wrap_pos()
