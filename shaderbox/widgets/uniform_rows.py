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

from imgui_bundle import imgui

from shaderbox.core import Pass
from shaderbox.glyph_tables import TABLE_UNIFORMS
from shaderbox.theme import COLOR, SIZE
from shaderbox.ui_models import (
    UIUniform,
    UniformSortKey,
    sort_uniform_hashes,
)
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
        else:
            value_hashes.append(hash_key)
    return (
        sort_uniform_hashes(value_hashes, ui_uniforms, sort_key, sort_desc),
        auto_hashes,
    )


def compact_row_count(value_hashes: list[int]) -> int:
    """How many rows a node actually draws: the scroll window, never the whole list (094 D6)."""
    return min(len(value_hashes), SIZE.GRAPH_ROWS_VISIBLE)


def draw_compact_row(
    ui_uniform: UIUniform,
    render_pass: Pass,
    width: float,
    row_id: str,
) -> bool:
    """One uniform's compact row. Returns True when the value changed.

    Four of the seven input types carry no control here: a `texture` is already a PORT on the
    node and drawing its picture would say it twice, while `buffer`, `array` and `text` need a
    control the card has no room for (a randomize button, a comma list, a 72px box). Those draw
    DIM with their name and state, so the node tells the truth about what the pass declares, and
    the control is one right-click away on the focused node.
    """
    name = ui_uniform.name
    value = render_pass.uniform_values.get(name)
    label_w = min(width * 0.34, float(SIZE.GRAPH_ROW_NAME_W))
    imgui.text_colored(COLOR.FG_MUTED, "")
    imgui.same_line(0.0, 0.0)
    imgui.set_next_item_width(label_w)
    imgui.text_colored(COLOR.FG_DIM, name[: max(1, int(label_w / 6))])
    imgui.same_line(label_w)

    control_w = max(24.0, width - label_w)
    imgui.set_next_item_width(control_w)
    imgui.set_next_item_allow_overlap()

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

    # Every other type: the state, no control.
    imgui.text_colored(COLOR.FG_DORMANT, format_auto_value(value)[:24])
    return False
