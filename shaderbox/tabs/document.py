from dataclasses import dataclass
from enum import Enum, auto

from imgui_bundle import imgui

from shaderbox.app import App
from shaderbox.commands import CommandId, command_label
from shaderbox.media import MediaWithTexture
from shaderbox.pass_graph import clamp_canvas_size
from shaderbox.render_preset import resolve_dims
from shaderbox.render_shape import (
    ASPECT_PRESETS,
    MENU_SHAPES,
    SHAPE_TABLE,
    RenderShape,
    ResolutionMode,
    aspect_label,
    aspect_of,
    reduce_aspect,
    shape_to_preset,
)
from shaderbox.theme import COLOR, SIZE, SPACE
from shaderbox.ui_models import (
    UIDocument,
)
from shaderbox.ui_primitives import (
    ComboRow,
    grouped_combo,
    play_stop_toggle,
    small_caption,
    standard_button,
)
from shaderbox.util import get_resolution_str
from shaderbox.widgets import pass_list

_SQUARE_PRESETS: tuple[int, ...] = (256, 512, 1024, 2048)


# One entry of the canvas popup: what it sets, and how it reads. An AUTO entry carries the
# aspect it selects and no size; a FIXED entry carries the pixel pair.
class CanvasChoiceKind(Enum):
    AUTO = auto()
    FIXED = auto()


@dataclass(frozen=True)
class CanvasChoice:
    kind: CanvasChoiceKind
    label: str
    aspect: tuple[int, int]
    size: tuple[int, int] | None = None


def _aspect_terms(label: str) -> tuple[int, int]:
    width, _, height = label.partition(":")
    return (int(width), int(height))


def _row_label(label: str, aspect: tuple[int, int]) -> str:
    """A size's label with the ratio its GROUP already carries taken off the end.

    `Wide 720p (16:9)` and `512x512 (1:1)` both name their shape, which the caption above the
    row repeats -- the suffixes are single-homed in `render_shape` and `get_resolution_str`
    for the surfaces that show one size alone, so the trim lives here rather than there.
    """
    suffix = f" ({aspect[0]}:{aspect[1]})"
    return label[: -len(suffix)] if label.endswith(suffix) else label


def canvas_choice_groups(
    ui_document: UIDocument,
) -> list[tuple[str, list[CanvasChoice]]]:
    """The canvas popup, grouped by aspect: one group per ratio, its first row `Auto`.

    The two modes pick from ONE list because they answer the same question -- what shape is
    this document, and does its size follow the viewer or a stored pair. `Auto` leads each
    group because it is the mode a new document starts in, and the sizes under it are that
    same ratio at three scales. A ratio no fixed size covers keeps its group, so every aspect
    the model accepts stays reachable.

    Reads `uniform_values`, never `get_active_uniforms()`: the latter compiles a
    never-attempted pass (066 D1), which on the Document tab's every frame would compile the
    whole graph.
    """
    by_aspect: dict[tuple[int, int], list[CanvasChoice]] = {
        preset: [] for preset in ASPECT_PRESETS
    }

    current = ui_document.document.resolution
    sizes: list[tuple[str, tuple[int, int]]] = [
        (get_resolution_str(None, n, n), (n, n)) for n in _SQUARE_PRESETS
    ]
    for shape in MENU_SHAPES:
        if shape is RenderShape.NATIVE:
            continue
        sizes.append(
            (
                SHAPE_TABLE[shape].menu_label,
                resolve_dims(
                    shape_to_preset(
                        shape,
                        is_video=False,
                        fps=None,
                        container=None,
                        duration_max=None,
                    ),
                    current,
                ),
            )
        )
    for render_pass in ui_document.document.passes.values():
        for uniform_name, value in sorted(render_pass.uniform_values.items()):
            if not isinstance(value, MediaWithTexture):
                continue
            sizes.append(
                (
                    get_resolution_str(uniform_name, *value.texture.size),
                    value.texture.size,
                )
            )

    seen: set[tuple[int, int]] = set()
    for label, size in sizes:
        if size in seen:
            continue
        seen.add(size)
        # `aspect_of` reduces exactly, spelling an encoder-aligned 1920x1088 as `30:17`. The
        # group a size belongs to is the one a reader would name it, so the snap decides.
        snapped = reduce_aspect(_aspect_terms(aspect_label(size)))
        by_aspect.setdefault(snapped, []).append(
            CanvasChoice(
                CanvasChoiceKind.FIXED, _row_label(label, snapped), snapped, size
            )
        )

    return [
        (
            f"{aspect[0]}:{aspect[1]}",
            [CanvasChoice(CanvasChoiceKind.AUTO, "Auto", aspect), *rows],
        )
        for aspect, rows in by_aspect.items()
    ]


def canvas_choice_label(ui_document: UIDocument) -> str:
    """What the closed chip reads: the mode, and the value that mode resolves to.

    Under Auto the stored value is a RATIO and the pixels follow the viewer, so the chip names
    the ratio and the readout line carries the live size. Under Fixed the pair IS the value.
    """
    document = ui_document.document
    if document.resolution_mode is ResolutionMode.AUTO:
        aspect = ui_document.ui_state.aspect
        return f"Auto {aspect[0]}:{aspect[1]}"
    width, height = document.resolution
    return f"{width}x{height}"


def _apply_canvas_choice(
    app: App, ui_document: UIDocument, choice: CanvasChoice
) -> None:
    """Commit one popup row: the mode it names, then the value that mode reads."""
    if choice.kind is CanvasChoiceKind.AUTO:
        _switch_resolution_mode(app, ui_document, ResolutionMode.AUTO)
        _apply_aspect(app, ui_document, choice.aspect)
        return
    assert choice.size is not None
    _switch_resolution_mode(app, ui_document, ResolutionMode.FIXED)
    _apply_canvas_size(app, ui_document, choice.size)


def _draw_canvas_control(app: App, ui_document: UIDocument) -> None:
    """The whole canvas control: one combo over every shape the document can take.

    One control, not a mode toggle beside a value field: `Auto | Fixed` named a mode whose two
    positions edited different things (a ratio, a pixel pair), so the row carried a control
    that changed meaning under it. Here a row names both at once.
    """
    groups = canvas_choice_groups(ui_document)
    flat = [choice for _, rows in groups for choice in rows]
    current_label = canvas_choice_label(ui_document)
    combo_groups: list[tuple[str, list[ComboRow]]] = [
        (caption, [(choice.label, COLOR.FG_SECONDARY) for choice in rows])
        for caption, rows in groups
    ]
    width = imgui.calc_text_size("Auto 16:9").x + 4.0 * float(SPACE.MD)
    picked = grouped_combo(
        "##canvas_choice",
        (current_label, COLOR.FG_PRIMARY),
        combo_groups,
        width,
    )
    if picked is not None:
        _apply_canvas_choice(app, ui_document, flat[picked])


def _switch_resolution_mode(
    app: App, ui_document: UIDocument, mode: ResolutionMode
) -> None:
    """Change the mode, seeding the field the new mode reads from the live canvas.

    A switch must never jump the picture (revision 1 D3): Auto -> Fixed takes the size the
    document is rendering at right now, so the canvas keeps its pixels; Fixed -> Auto takes
    that size's reduced ratio, so the shape survives and only the sizing rule changes.
    """
    document = ui_document.document
    if mode is ResolutionMode.FIXED:
        seeded = document.clamped_size(document.canvas_size)
        ui_document.ui_state.resolution = seeded
        document.resolution = seeded
        # The live size is already `seeded`; the deferred write is what makes the Fixed branch
        # of the next tick agree rather than resize on its first frame.
        app.pending_resolution[ui_document.id] = seeded
    else:
        seeded_aspect = aspect_of(document.canvas_size)
        ui_document.ui_state.aspect = seeded_aspect
        document.aspect = seeded_aspect
    ui_document.ui_state.resolution_mode = mode
    document.resolution_mode = mode


def _apply_aspect(app: App, ui_document: UIDocument, ratio: tuple[int, int]) -> None:
    reduced = reduce_aspect(ratio)
    if reduced == ui_document.ui_state.aspect:
        return
    ui_document.ui_state.aspect = reduced
    ui_document.document.aspect = reduced
    app.notifications.push(f"Aspect: {reduced[0]}:{reduced[1]}")


def _draw_canvas_caption(app: App, ui_document: UIDocument) -> None:
    """The `Canvas` caption and, beside it, the number the control below does not carry.

    Under Auto the control names a RATIO, so this is the live pixel size; under Fixed the
    control names a SIZE, so this is its aspect. Each mode shows the other half of the same
    fact. It sits on the CAPTION row (his call) rather than under the control: a line below the
    combo read as a second field, and the caption row is where a label belongs.
    """
    document = ui_document.document
    if document.resolution_mode is ResolutionMode.FIXED:
        detail = aspect_label(document.resolution)
    else:
        width, height = document.canvas_size
        detail = f"{width}x{height}"
    small_caption(app.font_12, f"Canvas  {detail}")


def _apply_canvas_size(
    app: App, ui_document: UIDocument, size: tuple[int, int]
) -> None:
    """Commit the picker's W x H as the document's stored `resolution` (090 D5).

    The write is DEFERRED, never applied in place: this runs inside the draw phase, after
    `_draw_document_image` pushed the output texture into this frame's draw list, and a resize
    releases that texture and every feedback history with it. Step 4 of the next
    `_tick_frame_state` consumes `pending_resolution` through the same path Auto takes, where
    every release precedes any draw — the shape `pending_project_switch` uses for the same
    084 D5 hazard.
    """
    w, h = clamp_canvas_size(size)
    if (w, h) == ui_document.ui_state.resolution:
        return
    ui_document.ui_state.resolution = (w, h)
    ui_document.document.resolution = (w, h)
    app.pending_resolution[ui_document.id] = (w, h)
    label = (
        "Canvas"
        if ui_document.document.resolution_mode is ResolutionMode.FIXED
        else "Export"
    )
    app.notifications.push(f"{label}: {w}x{h}")


def draw(app: App) -> None:
    if not (ui_document := app.ui_documents.get(app.current_document_id)):
        return

    imgui.spacing()

    combo_offset = SIZE.NAME_INPUT_W + SPACE.XL

    imgui.begin_disabled(app.copilot_turn_active)

    small_caption(app.font_12, "Document name")
    imgui.same_line(combo_offset)
    _draw_canvas_caption(app, ui_document)

    imgui.set_next_item_width(SIZE.NAME_INPUT_W)
    ui_document.ui_state.ui_name = imgui.input_text_with_hint(
        "##document_name", "document name", ui_document.ui_state.ui_name
    )[1]

    imgui.same_line(combo_offset)

    # Per-document widget ids: imgui keeps an ITEM active across a document switch, so a
    # shared id would let the outgoing document's open popup land on the incoming one.
    imgui.push_id(ui_document.id)

    _draw_canvas_control(app, ui_document)
    _draw_script_toggle(app)

    imgui.pop_id()

    imgui.end_disabled()

    imgui.dummy((0, SPACE.MD))
    _draw_passes(app, app.current_document_id)


def _draw_script_toggle(app: App) -> None:
    """The document script's play/stop, under a `Script` label naming what it controls.

    The bare word needs that label HERE and not on a uniform row: there the toggle sits at the
    end of the row of the uniform it stops, which is its subject, while on this row it had no
    neighbor at all and read as a loose `stop` (his finding). The `open` summoners it used to
    sit beside are gone (093 W8) -- opening a script is a verb, and verbs live on the
    document's context menu, the Document menu and their chords. What cannot live there is a
    STATE the user watches while a shader runs, so the toggle stays, and is absent when the
    document has no script. The caller's copilot-turn bracket covers it (a write races the
    reload).
    """
    document_id = app.current_document_id
    if not app.session.has_script(document_id):
        return
    imgui.same_line(spacing=float(SPACE.LG))
    imgui.align_text_to_frame_padding()
    imgui.text_colored(COLOR.FG_DIM, "Script")
    imgui.same_line(spacing=float(SPACE.MD))
    playing = not app.current_document_ui_state_or_default.all_stopped
    if play_stop_toggle(
        "document",
        playing,
        tooltip="Stop the whole script" if playing else "Resume the whole script",
    ):
        app.set_document_all_stopped(document_id, playing)


def _draw_passes(app: App, document_id: str) -> None:
    # A plain caption over the strip: the graph's summoner lives on the entry-point row beside
    # the script's (093 W2-1), so this row names the tiles and nothing else. The add / import
    # row sits under them, inside its own copilot-turn bracket.
    imgui.begin_disabled(app.copilot_turn_active)
    small_caption(app.font_12, "Passes")
    imgui.end_disabled()
    pass_list.draw(app, document_id)
    imgui.begin_disabled(app.copilot_turn_active)
    imgui.dummy((0, float(SPACE.SM)))
    if standard_button(command_label(CommandId.ADD_PASS)):
        app.open_add_pass()
    imgui.same_line()
    if standard_button(command_label(CommandId.IMPORT_PASSES)):
        app.open_import_passes()
    imgui.end_disabled()
