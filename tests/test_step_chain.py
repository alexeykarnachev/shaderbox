"""The step chain on a real GL context: does it render, in order, once each?

These are the checks that decide whether the feature works. Everything else in the 064
suite is GL-free reasoning about a plan; this renders pixels and reads them back.
"""

from pathlib import Path

import moderngl
import numpy as np
import pytest

from shaderbox.core import Canvas, Node
from shaderbox.shader_lib import ShaderLibIndex, set_active
from shaderbox.shader_source import ShaderSource
from shaderbox.paths import shader_lib_root


@pytest.fixture(scope="module")
def gl() -> moderngl.Context:
    ctx = moderngl.create_standalone_context(require=460)
    set_active(ShaderLibIndex.build(shader_lib_root()))
    return ctx


def _node(gl: moderngl.Context, tmp_path: Path, text: str, name: str = "n") -> Node:
    path = tmp_path / f"{name}.frag.glsl"
    path.write_text(text, encoding="utf-8")
    node = Node(gl=gl, source=ShaderSource.load(path), canvas_size=(16, 16))
    node.compile()
    return node


def _pixels(canvas: Canvas) -> np.ndarray:
    raw = canvas.texture.read()
    return np.frombuffer(raw, dtype=np.uint8).reshape(
        canvas.texture.height, canvas.texture.width, 4
    )


def test_a_node_with_no_steps_is_unchanged(gl: moderngl.Context, tmp_path: Path) -> None:
    node = _node(
        gl,
        tmp_path,
        "#version 330\nout vec4 f_color;\nvoid main() { f_color = vec4(1.0, 0.0, 0.0, 1.0); }\n",
    )
    assert node.steps == []
    assert node.step_plan.order == []
    node.render(u_time=0.0)
    px = _pixels(node.canvas)
    assert px[0, 0, 0] == 255 and px[0, 0, 1] == 0


def test_a_two_step_chain_reads_the_earlier_step(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    # step_a writes 0.25; main doubles it. Reading the DEFAULT image instead would give
    # something else entirely, so this fails loudly if the wiring is wrong.
    node = _node(
        gl,
        tmp_path,
        "#version 330\n"
        "out vec4 f_color;\n"
        "in vec2 vs_uv;\n"
        "uniform sampler2D u_a;  // step, f4\n"
        "void step_a(out vec4 o) { o = vec4(0.25, 0.0, 0.0, 1.0); }\n"
        "void main() { f_color = vec4(texture(u_a, vs_uv).r * 2.0, 0.0, 0.0, 1.0); }\n",
    )
    assert node.compile_unit.errors == [], node.compile_unit.errors
    assert [s.name for s in node.steps] == ["a"]
    node.render(u_time=0.0)
    px = _pixels(node.canvas)
    assert px[8, 8, 0] == pytest.approx(127, abs=2)


def test_a_three_step_chain_composes_in_order(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    node = _node(
        gl,
        tmp_path,
        "#version 330\n"
        "out vec4 f_color;\n"
        "in vec2 vs_uv;\n"
        "uniform sampler2D u_a;  // step, f4\n"
        "uniform sampler2D u_b;  // step, f4\n"
        "void step_a(out vec4 o) { o = vec4(0.1, 0.0, 0.0, 1.0); }\n"
        "void step_b(out vec4 o) { o = texture(u_a, vs_uv) + vec4(0.2, 0.0, 0.0, 1.0); }\n"
        "void main() { f_color = texture(u_b, vs_uv); }\n",
    )
    assert node.compile_unit.errors == [], node.compile_unit.errors
    assert node.step_plan.order == ["a", "b"]
    node.render(u_time=0.0)
    px = _pixels(node.canvas)
    # 0.1 + 0.2 = 0.3 -> ~77
    assert px[8, 8, 0] == pytest.approx(77, abs=3)


def test_a_step_target_honours_its_declared_format(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    node = _node(
        gl,
        tmp_path,
        "#version 330\n"
        "out vec4 f_color;\n"
        "in vec2 vs_uv;\n"
        "uniform sampler2D u_big;   // step, f4, scale: 1.0\n"
        "uniform sampler2D u_half;  // step, f2, scale: 0.5, nearest\n"
        "void step_big(out vec4 o) { o = vec4(1.0); }\n"
        "void step_half(out vec4 o) { o = vec4(1.0); }\n"
        "void main() { f_color = texture(u_big, vs_uv) + texture(u_half, vs_uv); }\n",
    )
    assert node.compile_unit.errors == [], node.compile_unit.errors
    node.render(u_time=0.0)
    big = node._step_targets["big"][0]
    half = node._step_targets["half"][0]
    assert big.texture.size == (16, 16)
    assert big.texture.dtype == "f4"
    assert half.texture.size == (8, 8)
    assert half.texture.dtype == "f2"
    # clamp by default, inverting moderngl's repeat.
    assert half.texture.repeat_x is False


def test_a_float_step_target_exceeds_one(gl: moderngl.Context, tmp_path: Path) -> None:
    # R7's whole point: 8-bit saturates on the first accumulate pass. Read the target
    # directly, because the final canvas is f1 and would clamp regardless.
    node = _node(
        gl,
        tmp_path,
        "#version 330\n"
        "out vec4 f_color;\n"
        "in vec2 vs_uv;\n"
        "uniform sampler2D u_hot;  // step, f4\n"
        "void step_hot(out vec4 o) { o = vec4(7.0, 0.0, 0.0, 1.0); }\n"
        "void main() { f_color = texture(u_hot, vs_uv); }\n",
    )
    assert node.compile_unit.errors == [], node.compile_unit.errors
    node.render(u_time=0.0)
    hot = node._step_targets["hot"][0]
    values = np.frombuffer(hot.texture.read(), dtype=np.float32)
    assert values[0] == pytest.approx(7.0)


def test_a_self_reading_step_gets_two_buffers_and_accumulates(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    node = _node(
        gl,
        tmp_path,
        "#version 330\n"
        "out vec4 f_color;\n"
        "in vec2 vs_uv;\n"
        "uniform sampler2D u_acc;  // step, f4\n"
        "void step_acc(out vec4 o) { o = texture(u_acc, vs_uv) + vec4(1.0, 0.0, 0.0, 1.0); }\n"
        "void main() { f_color = texture(u_acc, vs_uv); }\n",
    )
    assert node.compile_unit.errors == [], node.compile_unit.errors
    assert node.step_plan.self_reads == {"acc"}

    for _ in range(5):
        node.render(u_time=0.0)
    assert node._step_targets["acc"][1] is not None  # ping-pong pair
    texture = node.step_texture("acc")
    assert texture is not None
    values = np.frombuffer(texture.read(), dtype=np.float32)
    assert values[0] == pytest.approx(5.0)


def test_advance_state_false_does_not_move_the_feedback_clock(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    # D13: the live loop renders the current node twice per frame and the copilot probe
    # renders twice back to back. Advancing per CALL would run feedback at 2x on the
    # focused node and 1x everywhere else.
    node = _node(
        gl,
        tmp_path,
        "#version 330\n"
        "out vec4 f_color;\n"
        "in vec2 vs_uv;\n"
        "uniform sampler2D u_acc;  // step, f4\n"
        "void step_acc(out vec4 o) { o = texture(u_acc, vs_uv) + vec4(1.0, 0.0, 0.0, 1.0); }\n"
        "void main() { f_color = texture(u_acc, vs_uv); }\n",
    )
    preview = Canvas(gl=gl, size=(8, 8))
    for _ in range(3):
        node.render(u_time=0.0, canvas=preview, advance_state=False)
        node.render(u_time=0.0)
    texture = node.step_texture("acc")
    assert texture is not None
    values = np.frombuffer(texture.read(), dtype=np.float32)
    assert values[0] == pytest.approx(3.0)
    preview.release()


def test_step_targets_size_off_the_node_canvas_not_the_passed_one(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    # D12: ui.py renders the current node into a ~200px preview canvas. Sizing targets
    # off it would reallocate twice a frame and discard the ping-pong history.
    node = _node(
        gl,
        tmp_path,
        "#version 330\n"
        "out vec4 f_color;\n"
        "in vec2 vs_uv;\n"
        "uniform sampler2D u_a;  // step, scale: 0.5\n"
        "void step_a(out vec4 o) { o = vec4(1.0); }\n"
        "void main() { f_color = texture(u_a, vs_uv); }\n",
    )
    small = Canvas(gl=gl, size=(4, 4))
    node.render(u_time=0.0, canvas=small)
    assert node._step_targets["a"][0].texture.size == (8, 8)  # half of 16, not of 4
    small.release()


def test_a_step_sampler_never_enters_uniform_values(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    # D11: a bare moderngl.Texture in uniform_values is what UINode.save writes to
    # textures/*.bin -- megabytes of transient float target, reloaded as a stale frame.
    node = _node(
        gl,
        tmp_path,
        "#version 330\n"
        "out vec4 f_color;\n"
        "in vec2 vs_uv;\n"
        "uniform sampler2D u_a;  // step\n"
        "uniform float u_gain;\n"
        "void step_a(out vec4 o) { o = vec4(1.0); }\n"
        "void main() { f_color = texture(u_a, vs_uv) * u_gain; }\n",
    )
    node.render(u_time=0.0)
    assert node.is_step_sampler("u_a") is True
    assert node.is_step_sampler("u_gain") is False
    assert "u_a" not in node.uniform_values
    assert "u_gain" in node.uniform_values


def test_a_malformed_rider_refuses_the_whole_compile(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    # D14: compiling the final variant alone would leave u_a an ordinary sampler bound
    # to the default image -- a picture that looks fine and is wrong.
    node = _node(
        gl,
        tmp_path,
        "#version 330\n"
        "out vec4 f_color;\n"
        "in vec2 vs_uv;\n"
        "uniform sampler2D u_a;  // stp\n"
        "void step_a(out vec4 o) { o = vec4(1.0); }\n"
        "void main() { f_color = texture(u_a, vs_uv); }\n",
    )
    assert node.compile_unit.errors
    assert node.steps == []
    assert any("did you mean" in e.message for e in node.compile_unit.errors)


def test_release_frees_every_step_target(gl: moderngl.Context, tmp_path: Path) -> None:
    node = _node(
        gl,
        tmp_path,
        "#version 330\n"
        "out vec4 f_color;\n"
        "in vec2 vs_uv;\n"
        "uniform sampler2D u_a;  // step\n"
        "uniform sampler2D u_b;  // step\n"
        "void step_a(out vec4 o) { o = vec4(1.0); }\n"
        "void step_b(out vec4 o) { o = texture(u_a, vs_uv); }\n"
        "void main() { f_color = texture(u_b, vs_uv); }\n",
    )
    node.render(u_time=0.0)
    assert len(node._step_targets) == 2
    node.release()
    assert node._step_targets == {}
    assert node._step_programs == {}
    assert node._step_vaos == {}


def test_a_cycle_between_steps_is_reported(gl: moderngl.Context, tmp_path: Path) -> None:
    node = _node(
        gl,
        tmp_path,
        "#version 330\n"
        "out vec4 f_color;\n"
        "in vec2 vs_uv;\n"
        "uniform sampler2D u_a;  // step\n"
        "uniform sampler2D u_b;  // step\n"
        "void step_a(out vec4 o) { o = texture(u_b, vs_uv); }\n"
        "void step_b(out vec4 o) { o = texture(u_a, vs_uv); }\n"
        "void main() { f_color = texture(u_b, vs_uv); }\n",
    )
    assert any("cycle" in e.message for e in node.compile_unit.errors)


def test_evaluation_order_beats_declaration_order_on_a_real_render(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    """The producer is declared LAST and must still render first.

    This is freska's bug made visible: it iterated an unordered_map under its own
    `// TODO: this is incorrect!`, so a chain lagged one frame per hop. Every other test
    here happens to declare steps in dependency order, which makes declaration order
    look correct -- so only this one fails when the topological sort is bypassed.
    """
    node = _node(
        gl,
        tmp_path,
        "#version 330\n"
        "out vec4 f_color;\n"
        "in vec2 vs_uv;\n"
        # consumer first, producer second
        "uniform sampler2D u_late;   // step, f4\n"
        "uniform sampler2D u_early;  // step, f4\n"
        "void step_late(out vec4 o) { o = texture(u_early, vs_uv) + vec4(0.5, 0.0, 0.0, 1.0); }\n"
        "void step_early(out vec4 o) { o = vec4(0.25, 0.0, 0.0, 1.0); }\n"
        "void main() { f_color = texture(u_late, vs_uv); }\n",
    )
    assert node.compile_unit.errors == [], node.compile_unit.errors
    assert node.step_plan.order == ["early", "late"]

    # ONE render. In declaration order `late` would read an unwritten `early` (black)
    # and land at 0.5; correctly ordered it reads 0.25 and lands at 0.75.
    node.render(u_time=0.0)
    late = node.step_texture("late")
    assert late is not None
    assert np.frombuffer(late.read(), dtype=np.float32)[0] == pytest.approx(0.75)
