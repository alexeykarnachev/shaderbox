"""Which uniform rows a pass has, and how a compact one draws on a graph node (094 D4/D10a).

Two things live here because both the node's rows and the focused node's full rows need them
and neither owns the other:

- `pass_rows` is the loop the deleted Uniforms tab was the only home for. It is not a pure
  read: it CREATES a `UIUniform` for a hash not yet seen and calls `snap_input_type()` on every
  row, both writes to persisted document state. Delete the tab without rehoming it and a
  freshly-compiled pass has no rows at all.
- `draw_compact_row` is the node's version of `widgets/uniform.py`'s row. A full row is 320px of
  control against a 240px card, so the node draws the value alone and the rest is reached by
  focusing the node.
"""

from imgui_bundle import imgui, imgui_ctx

from shaderbox.core import Pass
from shaderbox.glyph_tables import TABLE_UNIFORMS
from shaderbox.theme import COLOR, SIZE
from shaderbox.ui_models import (
    UIUniform,
    UniformSortKey,
    sort_uniform_hashes,
)
from shaderbox.ui_primitives import ellipsize
from shaderbox.util import format_auto_value, get_uniform_hash


def pass_rows(
    render_pass: Pass,
    pass_name: str,
    ui_uniforms: dict[int, UIUniform],
    sort_key: UniformSortKey = "code",
    sort_desc: bool = False,
) -> tuple[list[int], list[int]]:
    """`(value_hashes, auto_hashes)` for one pass, creating any row it does not have yet.

    `sort_uniform_hashes` is what produces DECLARATION order: `get_active_uniforms()` yields
    GL's own order, which is driver-defined and reshuffles between recompiles.
    """
    value_hashes: list[int] = []
    auto_hashes: list[int] = []
    for uniform in render_pass.get_active_uniforms():
        if (
            uniform.name in TABLE_UNIFORMS
        ):  # engine glyph tables — pure machinery, no row
            continue
        hash_key = get_uniform_hash(uniform, pass_name)
        if hash_key not in ui_uniforms:
            ui_uniforms[hash_key] = UIUniform.from_uniform(uniform)
        ui_uniforms[hash_key].snap_input_type()
        if ui_uniforms[hash_key].input_type == "auto":
            auto_hashes.append(hash_key)
        elif ui_uniforms[hash_key].input_type in node_row_types():
            value_hashes.append(hash_key)
    return (
        sort_uniform_hashes(value_hashes, ui_uniforms, sort_key, sort_desc),
        auto_hashes,
    )


def compact_row_count(value_hashes: list[int]) -> int:
    """How many rows a node actually draws: the scroll window, never the whole list (094 D6)."""
    return min(len(value_hashes), SIZE.GRAPH_ROWS_VISIBLE)


def node_row_types() -> frozenset[str]:
    """The input types that get a row ON THE NODE.

    `texture` is absent because a sampler is already a PORT with the source on the wire's other
    end; drawing its value too would say it twice, and it is what filled the canvas with
    `AutoSource()` in the first cut. `buffer`, `array` and `text` need a control the card has no
    room for, so they are reached by focusing the node.
    """
    return frozenset({"drag", "color"})


def draw_compact_row(
    ui_uniform: UIUniform,
    render_pass: Pass,
    origin: tuple[float, float],
    width: float,
    row_id: str,
) -> bool:
    """One uniform's compact row, drawn at `origin` in SCREEN space.

    Positioned absolutely, never flowed: the canvas is a pannable draw list with no layout
    cursor of its own, so `same_line` measures from the child's content origin and every row
    lands in the same left column whatever node it belongs to. That is what the first cut did.
    """
    name = ui_uniform.name
    value = render_pass.uniform_values.get(name)
    # A FRACTION of the row, not a fixed cap: the cap was sized against the 136px card and
    # survived D4e's widening to 240, which truncated every name to `u_de...` on a card with
    # room for all of it. The floor keeps a name readable when the row is narrow.
    label_w = max(float(SIZE.GRAPH_ROW_NAME_W), width * 0.55)
    control_w = max(24.0, width - label_w - 4.0)

    imgui.set_cursor_screen_pos(origin)
    imgui.text_colored(COLOR.FG_DIM, ellipsize(name, label_w))

    imgui.set_cursor_screen_pos((origin[0] + label_w + 4.0, origin[1]))
    imgui.set_next_item_width(control_w)
    imgui.set_next_item_allow_overlap()
    # The control's own frame is transparent: the row already sits on the card's panel, and a
    # second filled rect inside it reads as a separate box rather than as a value in a row.
    # The hover and active fills stay, so the control still says it is one.
    with imgui_ctx.push_style_color(
        imgui.Col_.frame_bg, imgui.ImVec4(0.0, 0.0, 0.0, 0.0)
    ):
        return _draw_control(ui_uniform, render_pass, value, row_id)


def _draw_control(
    ui_uniform: UIUniform,
    render_pass: Pass,
    value: object,
    row_id: str,
) -> bool:
    """The row's value control, inside the caller's transparent-frame scope."""
    name = ui_uniform.name

    if ui_uniform.input_type == "drag":
        if isinstance(value, float | int) and not isinstance(value, bool):
            changed, new = imgui.drag_float(f"##{row_id}", float(value), 0.01)
            if changed:
                render_pass.uniform_values[name] = new
            return changed
        if isinstance(value, (list, tuple)) and 2 <= len(value) <= 4:
            fn = getattr(imgui, f"drag_float{len(value)}")
            changed, new = fn(f"##{row_id}", list(value), 0.01)
            if changed:
                render_pass.uniform_values[name] = new
            return changed
    elif ui_uniform.input_type == "color":
        if isinstance(value, (list, tuple)) and 3 <= len(value) <= 4:
            fn = getattr(imgui, f"color_edit{len(value)}")
            changed, new = fn(
                f"##{row_id}", list(value), imgui.ColorEditFlags_.no_inputs.value
            )
            if changed:
                render_pass.uniform_values[name] = new
            return changed

    # A type with no control never reaches here (`node_row_types` filters the list), so this is
    # the "declared a shape the row cannot draw" tail rather than a fallthrough.
    imgui.text_colored(COLOR.FG_DORMANT, format_auto_value(value)[:18])
    return False
