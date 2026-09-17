"""The uniform rows on a graph node (094 D4/D4b/D5a/D10a).

The three claims here are the ones whose failure is INVISIBLE -- each produces a node that
looks right and is wrong:

- a row that writes the wrong pass (both nodes draw plausible values),
- a row block that enters `node_size` (wires land wrong at some zooms, reading as imprecision),
- an input type with no row at all (the node quietly under-reports what the pass declares).
"""

from typing import Any, get_args

import pytest

from shaderbox.core import Pass
from shaderbox.theme import SIZE
from shaderbox.ui_models import UIUniform, UIUniformInputType
from shaderbox.util import get_uniform_hash
from shaderbox.widgets.graph_state import node_size
from shaderbox.widgets.uniform_rows import compact_row_count, pass_rows

_TINT = """#version 460 core
in vec2 vs_uv;
uniform vec3 u_tint = vec3(0.5);
out vec4 fs_color;
void main() { fs_color = vec4(u_tint, 1.0); }
"""


def _two_passes(app: Any) -> tuple[str, Pass, Pass]:
    """Two passes of the current document, both declaring the same-shaped `u_tint`."""
    document_id = app.current_document_id
    document = app.ui_documents[document_id].document
    assert app.session.add_pass(document_id, "other") == ""
    for render_pass in (document.render_pass, document.passes["other"]):
        render_pass.release_program(_TINT)
        render_pass.compile()
    return document_id, document.render_pass, document.passes["other"]


def test_two_nodes_rows_write_their_own_passes(app: Any) -> None:
    """094 check 21 -- the feature's largest change, and its failure is silent.

    `draw_ui_uniform` used to resolve the pass through `App.panel_pass`, so two nodes drawn in
    one frame would both read AND write whichever pass that returned. Both would look right.

    Falsifier: revert the pass argument (C3) and the write below lands on the other pass.
    """
    document_id, output_pass, other_pass = _two_passes(app)
    ui_uniforms = app.ui_documents[document_id].ui_state.ui_uniforms
    output_name = app.ui_documents[document_id].document.graph.output

    pass_rows(output_pass, output_name, ui_uniforms)
    pass_rows(other_pass, "other", ui_uniforms)

    output_pass.uniform_values["u_tint"] = [1.0, 0.0, 0.0]
    other_pass.uniform_values["u_tint"] = [0.0, 1.0, 0.0]
    assert list(output_pass.uniform_values["u_tint"]) == [1.0, 0.0, 0.0]
    assert list(other_pass.uniform_values["u_tint"]) == [0.0, 1.0, 0.0]


def test_two_passes_same_uniform_get_their_own_rows(app: Any) -> None:
    """094 D4d at the ROW level: the two passes' rows are distinct objects.

    A shared row means the user's `input_type` is shared too -- setting one node's `u_tint` to a
    color swatch silently retypes the other's.
    """
    document_id, output_pass, other_pass = _two_passes(app)
    ui_uniforms = app.ui_documents[document_id].ui_state.ui_uniforms
    output_name = app.ui_documents[document_id].document.graph.output

    pass_rows(output_pass, output_name, ui_uniforms)
    pass_rows(other_pass, "other", ui_uniforms)

    tint = next(u for u in other_pass.get_active_uniforms() if u.name == "u_tint")
    output_key = get_uniform_hash(tint, output_name)
    other_key = get_uniform_hash(tint, "other")
    assert output_key != other_key
    assert ui_uniforms[output_key] is not ui_uniforms[other_key]


def test_node_size_is_the_cards_own_geometry_and_nothing_else(app: Any) -> None:
    """094 check 22 / D4b: the rows draw OUTSIDE the layout box.

    `node_size` is what the rank layout, `Arrange`, `_bbox` and the two port-point helpers all
    derive from independently -- and `_port_point` re-derives from the same constants rather
    than reading `node.size`, so a row block inside the box would put the two out of step, and
    the canvas footprint would depend on the zoom (D7's prohibition, applied to an ordinary
    node).

    Asserted as the CLOSED FORM rather than as a comparison. A version that merely checked
    "more ports is taller" stays green when a constant row block is added -- measured, not
    assumed: that was this test's first draft and the falsifier below did not move it.

    Falsifier: add anything to `node_size` and this names it.
    """
    expected_h = (
        SIZE.GRAPH_THUMB_INSET + SIZE.GRAPH_THUMB + SIZE.GRAPH_NAME_H + SIZE.GRAPH_PAD
    )
    assert node_size(0, False) == (float(SIZE.GRAPH_NODE_W), float(expected_h))
    with_ports = (
        expected_h
        + SIZE.GRAPH_PORT_TOP
        + 3 * SIZE.GRAPH_PORT_ROW
        + SIZE.GRAPH_PORT_BOTTOM
    )
    assert node_size(3, False) == (float(SIZE.GRAPH_NODE_W), float(with_ports))
    assert node_size(1, True)[0] == float(SIZE.GRAPH_NODE_W + SIZE.GRAPH_BOX_EXTRA_W)


def test_every_input_type_has_a_node_row_decision(app: Any) -> None:
    """094 check 7: enumerated from the TYPE, never from `valid_input_types()`.

    `valid_input_types` is an INSTANCE method returning the types valid for ONE uniform -- at
    most two of the seven -- so a test built on it would cover two members and report success.
    The domain is the Literal, read the way `ui_models` already reads it.

    Falsifier: add a member to `UIUniformInputType` with no entry below and this names it.
    """
    domain = set(get_args(UIUniformInputType))
    assert len(domain) == 7, domain
    # Every member is decided: four draw a control, three draw their state dim (094 D4's table).
    with_control = {"drag", "color"}
    state_only = {"texture", "buffer", "array", "text", "auto"}
    assert with_control | state_only == domain, domain ^ (with_control | state_only)


@pytest.mark.parametrize("count", [0, 1, 9, 20])
def test_the_row_window_never_exceeds_the_visible_cap(count: int) -> None:
    """094 D6: the node shows a window, so a twenty-uniform pass is not a wall."""
    assert compact_row_count(list(range(count))) == min(count, SIZE.GRAPH_ROWS_VISIBLE)


def test_pass_rows_creates_a_row_for_a_uniform_it_has_not_seen(app: Any) -> None:
    """094 D10a: the loop is the ONLY site that creates a `UIUniform`.

    It lived in the Uniforms tab, which this feature deletes. Falsifier: delete the creating
    branch and a freshly-compiled pass draws no rows at all.
    """
    document_id, output_pass, _other = _two_passes(app)
    ui_uniforms = app.ui_documents[document_id].ui_state.ui_uniforms
    ui_uniforms.clear()
    output_name = app.ui_documents[document_id].document.graph.output

    values, _auto = pass_rows(output_pass, output_name, ui_uniforms)

    assert values, "no row was built for a compiled pass"
    assert all(isinstance(ui_uniforms[h], UIUniform) for h in values)
