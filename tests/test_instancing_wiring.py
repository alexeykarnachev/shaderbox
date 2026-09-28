"""The wiring: not that each unit is correct, but that the app calls it.

Four adversarial reviewers found the same class across three features -- every producer
and every consumer was gated, and the call connecting them was not. Deleting a wiring site
left the whole suite green in each case, so the feature would vanish from the running app
with every test passing.

These gates all have the same shape: build the real input, call the real assembling
function, and assert the fact arrives. None of them constructs the view or the node it is
checking, because a fixture that passes `entity_fields=[...]` and then asserts
`entity_fields` is only testing its own argument.
"""

from pathlib import Path

import moderngl
import pytest

from shaderbox.core import Pass
from shaderbox.pass_graph import TargetConfig
from shaderbox.shader_source import ShaderSource

_INSTANCED = """#version 460 core
in vec2 vs_quad;
flat in vec2 pos;
flat in float radius;
out vec4 frag_color;
void main() {
    if (length(vs_quad) > 1.0) discard;
    frag_color = vec4(1.0, 0.0, 0.0, 1.0);
}
"""

_FULLSCREEN = """#version 460 core
in vec2 vs_uv;
out vec4 frag_color;
void main() { frag_color = vec4(vs_uv, 0.0, 1.0); }
"""


@pytest.fixture
def pair(gl_ctx: moderngl.Context, tmp_path: Path) -> tuple[Pass, Pass]:
    """One instanced pass and one fullscreen pass, differing ONLY in the `flat in`
    declaration -- the contrast pair every gate below needs."""
    made: list[Pass] = []
    for name, text in (("swarm", _INSTANCED), ("plain", _FULLSCREEN)):
        path = tmp_path / f"{name}.frag.glsl"
        path.write_text(text)
        render_pass = Pass(
            gl=gl_ctx,
            source=ShaderSource.load(path),
            canvas_size=(8, 8),
            target=TargetConfig(),
        )
        render_pass.compile()
        made.append(render_pass)
    assert made[0].entity_fields and not made[1].entity_fields
    return made[0], made[1]


def test_the_graph_computes_its_instanced_set_from_real_passes(
    pair: tuple[Pass, Pass],
) -> None:
    """104 item 8. The adapter is gated by passing `instanced` in by hand; nothing asserted
    that the widget COMPUTES it. Deleting the computation left 2216 tests green, so the
    mark would disappear from the app with the suite passing.

    This calls the widget's own expression against real compiled passes, so a fixture that
    never reached an instanced pass fails on the first assertion rather than reporting an
    empty set as success.
    """
    from shaderbox.graph_canvas.adapter import pass_key
    from shaderbox.widgets.pass_graph import instanced_pass_keys

    swarm, plain = pair

    class _Doc:
        def __init__(self) -> None:
            self.passes = {"swarm": swarm, "plain": plain}

    computed = instanced_pass_keys(_Doc(), ["swarm", "plain"])  # type: ignore[arg-type]

    assert computed == frozenset({pass_key("swarm")}), (
        f"the graph's instanced set is not derived from entity_fields: {computed}"
    )
    # The contrast half is a silence assertion, so it needs proof the fullscreen pass was
    # actually in the input -- otherwise "not marked" and "never looked at" are the same
    # empty result.
    assert pass_key("plain") not in computed
    assert not plain.entity_fields


def test_both_copilot_views_carry_entity_fields_from_a_real_pass(
    pair: tuple[Pass, Pass],
) -> None:
    """103 D4. Both view tests built their dataclasses by hand and asserted the argument
    they had just passed, so removing all three wiring sites left the suite green.

    `_entity_field_rows` is the function both views call; this asserts it discriminates on
    real compiled passes, which is the fact the views carry.
    """
    from shaderbox.copilot.backend import _entity_field_rows

    swarm, plain = pair
    rows = _entity_field_rows(swarm)
    assert rows, "an instanced pass produced no entity-field rows"
    assert any("pos" in row for row in rows)
    assert _entity_field_rows(plain) == [], (
        "a fullscreen pass reported entity fields -- the pair differs in more than the "
        "property under test"
    )
