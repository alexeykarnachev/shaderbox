"""The shipped multi-step example must compile, order correctly, and render.

The examples browser is where a user meets a feature. This one is the only place the
step syntax is discoverable by clicking rather than by reading, so a rot in it is worse
than a rot in a test -- it teaches the feature wrong.
"""

import json
from pathlib import Path

import moderngl
import numpy as np
import pytest

from shaderbox.constants import EXAMPLE_ORDER, NODE_EXAMPLES_DIR
from shaderbox.media import texture_to_rgba8
from shaderbox.paths import shader_lib_root
from shaderbox.shader_lib import ShaderLibIndex, set_active
from shaderbox.ui_models import load_node_from_dir

_EXAMPLE_ID = "b41e7c9d-2a68-4f53-8e17-6c0d95a3f2b8"
_EXAMPLE_DIR = (
    Path(__file__).resolve().parent.parent
    / "shaderbox/resources/node_examples"
    / _EXAMPLE_ID
)


@pytest.fixture(scope="module")
def gl() -> moderngl.Context:
    ctx = moderngl.create_standalone_context(require=460)
    set_active(ShaderLibIndex.build(shader_lib_root()))
    return ctx


def test_the_example_is_listed_in_the_browser() -> None:
    # An example not in EXAMPLE_ORDER sorts last silently; this one is the feature's
    # only discoverable surface, so its listing is part of the contract.
    assert _EXAMPLE_ID in EXAMPLE_ORDER


def test_the_example_compiles_and_declares_its_chain(gl: moderngl.Context) -> None:
    ui_node = load_node_from_dir(_EXAMPLE_DIR)
    assert ui_node.node.compile_unit.errors == [], [
        e.message for e in ui_node.node.compile_unit.errors
    ]
    names = [s.name for s in ui_node.node.steps]
    assert names == ["sparks", "bright", "blur", "trail"]
    # Order is derived, not declared: bright reads sparks, blur reads bright.
    assert ui_node.node.step_plan.order == ["sparks", "bright", "blur", "trail"]
    assert ui_node.node.step_plan.self_reads == {"trail"}


def test_the_example_exercises_every_part_of_the_feature(gl: moderngl.Context) -> None:
    ui_node = load_node_from_dir(_EXAMPLE_DIR)
    by_name = {s.name: s for s in ui_node.node.steps}
    # Differing resolutions (R2), float targets (R7), and a filter choice (R8) -- all of
    # it from node state, since the shader carries no configuration.
    assert by_name["bright"].config.scale == 0.5
    assert by_name["blur"].config.scale == 0.25
    assert by_name["blur"].config.filter_linear is True
    assert all(s.config.dtype == "f2" for s in ui_node.node.steps)
    # And the shipped configs actually reached the engine's targets.
    ui_node.node.render(u_time=0.0)
    assert ui_node.node._step_targets["blur"][0].texture.size == (240, 180)


def test_the_example_renders_and_its_trail_accumulates(gl: moderngl.Context) -> None:
    ui_node = load_node_from_dir(_EXAMPLE_DIR)
    node = ui_node.node

    node.render(u_time=0.5)
    early = texture_to_rgba8(node.step_texture("trail"))
    early_lit = int((early[:, :, :3].max(axis=2) > 8).sum())

    for i in range(40):
        node.render(u_time=0.5 + i / 30.0)
    later = texture_to_rgba8(node.step_texture("trail"))
    later_lit = int((later[:, :, :3].max(axis=2) > 8).sum())

    # Feedback: the trail covers more ground after 40 frames than after one.
    assert later_lit > early_lit * 2

    final = texture_to_rgba8(node.canvas.texture)
    assert final[:, :, :3].max() > 0, "the composed frame is black"


def test_the_example_sparks_actually_move(gl: moderngl.Context) -> None:
    # The emitters are recomputed per frame rather than accumulated, so a static
    # `spark_pos` would leave the sparks target identical at every time.
    ui_node = load_node_from_dir(_EXAMPLE_DIR)
    node = ui_node.node

    node.render(u_time=0.5)
    first = texture_to_rgba8(node.step_texture("sparks")).copy()
    node.render(u_time=3.0)
    second = texture_to_rgba8(node.step_texture("sparks"))
    assert not np.array_equal(first, second)


def test_every_shipped_example_has_a_description() -> None:
    """The examples browser renders "(no description)" for one that lacks it.

    Pinned for every example, not just this one: the browser is where a user meets the
    app, and the slot exists to be filled.
    """
    missing = []
    for example_dir in sorted(NODE_EXAMPLES_DIR.iterdir()):
        if not example_dir.is_dir():
            continue
        meta = json.loads((example_dir / "node.json").read_text())
        state = meta.get("ui_state", {})
        if not state.get("description", "").strip():
            missing.append(state.get("ui_name", example_dir.name))
    assert missing == [], f"examples with no description: {missing}"
