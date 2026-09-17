"""Which uniform rows a pass has, and how a compact one draws on a graph node (094 D4/D10a).

Two things live here because both the node's rows and the focused node's full rows need them
and neither owns the other:

- `pass_rows` is the loop the deleted Uniforms tab was the only home for. It is not a pure
  read: it CREATES a `UIUniform` for a hash not yet seen and calls `snap_input_type()` on every
  row, both writes to persisted document state. Delete the tab without rehoming it and a
  freshly-compiled pass has no rows at all.
The node draws the REAL row (`widgets/uniform.py::draw_ui_uniform`), not a variant of it: the
input-type chip, the name in `STATE_INFO` blue while a script drives it, the jump-to-declaration
bridge, the play/stop toggle and every input type's own control. A second spelling of a uniform
row is how the two drift, and the first cut of 094 proved it -- it lost the chip, the colors and
the play/stop, and drew `AutoSource()` where the real row draws a source combo. The CARD is
sized to hold the row instead.
"""

from shaderbox.core import Pass
from shaderbox.glyph_tables import TABLE_UNIFORMS
from shaderbox.theme import SIZE
from shaderbox.ui_models import (
    UIUniform,
    UniformSortKey,
    sort_uniform_hashes,
)
from shaderbox.util import get_uniform_hash


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
            # Every input type, because the node draws the REAL row: a texture gets its source
            # combo and preview, a text uniform its box. No type filter, so there is no table
            # here to drift from the one in `widgets/uniform.py`.
            value_hashes.append(hash_key)
    return (
        sort_uniform_hashes(value_hashes, ui_uniforms, sort_key, sort_desc),
        auto_hashes,
    )


def compact_row_count(value_hashes: list[int]) -> int:
    """How many rows a node actually draws: the scroll window, never the whole list (094 D6)."""
    return min(len(value_hashes), SIZE.GRAPH_ROWS_VISIBLE)
