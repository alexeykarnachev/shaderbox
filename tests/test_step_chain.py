"""The step chain on a real GL context: does it render, in order, once each?

These are the checks that decide whether the feature works. Everything else in the 064
suite is GL-free reasoning about a plan; this renders pixels and reads them back.
"""

from pathlib import Path

import moderngl
import numpy as np
import pytest
from PIL import Image as PILImage

from shaderbox.core import Canvas, Node
from shaderbox.media import FileDetails, MediaDetails, ResolutionDetails
from shaderbox.paths import shader_lib_root
from shaderbox.shader_lib import ShaderLibIndex, set_active
from shaderbox.shader_source import ShaderSource
from shaderbox.step_spec import StepConfig


@pytest.fixture(scope="module")
def gl() -> moderngl.Context:
    ctx = moderngl.create_standalone_context(require=460)
    set_active(ShaderLibIndex.build(shader_lib_root()))
    return ctx


def _node(
    gl: moderngl.Context,
    tmp_path: Path,
    text: str,
    name: str = "n",
    configs: dict[str, StepConfig] | None = None,
) -> Node:
    """A node built the way the app builds one: source from disk, configs from state.

    Steps default to `f4` here rather than the engine's `f2`, because these tests read
    exact float values back out of a target and half-float would round them. A test that
    cares about the DEFAULT format says so explicitly.
    """
    path = tmp_path / f"{name}.frag.glsl"
    path.write_text(text, encoding="utf-8")
    node = Node(gl=gl, source=ShaderSource.load(path), canvas_size=(16, 16))
    node.compile()  # a first pass so the step names are known
    node.step_configs = {step.name: StepConfig(dtype="f4") for step in node.steps}
    if configs:
        node.step_configs.update(configs)
    node.compile()
    return node


def _pixels(canvas: Canvas) -> np.ndarray:
    raw = canvas.texture.read()
    return np.frombuffer(raw, dtype=np.uint8).reshape(
        canvas.texture.height, canvas.texture.width, 4
    )


def test_a_node_with_no_steps_is_unchanged(
    gl: moderngl.Context, tmp_path: Path
) -> None:
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
        "uniform sampler2D u_step_a;\n"
        "void step_a(out vec4 o) { o = vec4(0.25, 0.0, 0.0, 1.0); }\n"
        "void main() { f_color = vec4(texture(u_step_a, vs_uv).r * 2.0, 0.0, 0.0, 1.0); }\n",
        configs={"a": StepConfig(dtype="f4")},
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
        "uniform sampler2D u_step_a;\n"
        "uniform sampler2D u_step_b;\n"
        "void step_a(out vec4 o) { o = vec4(0.1, 0.0, 0.0, 1.0); }\n"
        "void step_b(out vec4 o) { o = texture(u_step_a, vs_uv) + vec4(0.2, 0.0, 0.0, 1.0); }\n"
        "void main() { f_color = texture(u_step_b, vs_uv); }\n",
    )
    assert node.compile_unit.errors == [], node.compile_unit.errors
    assert node.step_plan.order == ["a", "b"]
    node.render(u_time=0.0)
    px = _pixels(node.canvas)
    # 0.1 + 0.2 = 0.3 -> ~77
    assert px[8, 8, 0] == pytest.approx(77, abs=3)


def test_a_step_target_honours_its_configured_format(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    # Format, scale and filter are node state, not shader text: the shader says WHAT the
    # steps are, the config says how each target is set up.
    node = _node(
        gl,
        tmp_path,
        "#version 330\n"
        "out vec4 f_color;\n"
        "in vec2 vs_uv;\n"
        "uniform sampler2D u_step_big;\n"
        "uniform sampler2D u_step_half;\n"
        "void step_big(out vec4 o) { o = vec4(1.0); }\n"
        "void step_half(out vec4 o) { o = vec4(1.0); }\n"
        "void main() { f_color = texture(u_step_big, vs_uv) + texture(u_step_half, vs_uv); }\n",
        configs={
            "big": StepConfig(dtype="f4"),
            "half": StepConfig(scale=0.5, dtype="f2", filter_linear=False),
        },
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
        "uniform sampler2D u_step_hot;\n"
        "void step_hot(out vec4 o) { o = vec4(7.0, 0.0, 0.0, 1.0); }\n"
        "void main() { f_color = texture(u_step_hot, vs_uv); }\n",
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
        "uniform sampler2D u_step_acc;\n"
        "void step_acc(out vec4 o) { o = texture(u_step_acc, vs_uv) + vec4(1.0, 0.0, 0.0, 1.0); }\n"
        "void main() { f_color = texture(u_step_acc, vs_uv); }\n",
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
        "uniform sampler2D u_step_acc;\n"
        "void step_acc(out vec4 o) { o = texture(u_step_acc, vs_uv) + vec4(1.0, 0.0, 0.0, 1.0); }\n"
        "void main() { f_color = texture(u_step_acc, vs_uv); }\n",
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
        "uniform sampler2D u_step_a;\n"
        "void step_a(out vec4 o) { o = vec4(1.0); }\n"
        "void main() { f_color = texture(u_step_a, vs_uv); }\n",
        configs={"a": StepConfig(scale=0.5)},
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
        "uniform sampler2D u_step_a;\n"
        "uniform float u_gain;\n"
        "void step_a(out vec4 o) { o = vec4(1.0); }\n"
        "void main() { f_color = texture(u_step_a, vs_uv) * u_gain; }\n",
    )
    node.render(u_time=0.0)
    assert node.is_step_sampler("u_step_a") is True
    assert node.is_step_sampler("u_gain") is False
    assert "u_step_a" not in node.uniform_values
    assert "u_gain" in node.uniform_values


def test_a_step_with_no_body_refuses_the_whole_compile(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    # Compiling the final variant alone would leave u_step_a an ordinary sampler bound
    # to the default image -- a picture that looks fine and is wrong.
    node = _node(
        gl,
        tmp_path,
        "#version 330\n"
        "out vec4 f_color;\n"
        "in vec2 vs_uv;\n"
        "uniform sampler2D u_step_a;\n"
        "void main() { f_color = texture(u_step_a, vs_uv); }\n",
    )
    assert node.compile_unit.errors
    assert node.steps == []
    assert any("has no body" in e.message for e in node.compile_unit.errors)


def test_an_ordinary_sampler_with_any_comment_still_compiles(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    # The reason steps are named rather than commented: a shader that never heard of the
    # feature cannot be broken by what it writes in its own comments.
    node = _node(
        gl,
        tmp_path,
        "#version 330\n"
        "out vec4 f_color;\n"
        "in vec2 vs_uv;\n"
        "uniform sampler2D u_image;  // step, scale: 0.5, f2 -- inert prose\n"
        "void main() { f_color = texture(u_image, vs_uv); }\n",
    )
    assert node.compile_unit.errors == [], node.compile_unit.errors
    assert node.steps == []
    assert node.program is not None


def test_release_frees_every_step_target(gl: moderngl.Context, tmp_path: Path) -> None:
    node = _node(
        gl,
        tmp_path,
        "#version 330\n"
        "out vec4 f_color;\n"
        "in vec2 vs_uv;\n"
        "uniform sampler2D u_step_a;\n"
        "uniform sampler2D u_step_b;\n"
        "void step_a(out vec4 o) { o = vec4(1.0); }\n"
        "void step_b(out vec4 o) { o = texture(u_step_a, vs_uv); }\n"
        "void main() { f_color = texture(u_step_b, vs_uv); }\n",
    )
    node.render(u_time=0.0)
    assert len(node._step_targets) == 2
    node.release()
    assert node._step_targets == {}
    assert node._step_programs == {}
    assert node._step_vaos == {}


def test_a_cycle_between_steps_is_reported(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    node = _node(
        gl,
        tmp_path,
        "#version 330\n"
        "out vec4 f_color;\n"
        "in vec2 vs_uv;\n"
        "uniform sampler2D u_step_a;\n"
        "uniform sampler2D u_step_b;\n"
        "void step_a(out vec4 o) { o = texture(u_step_b, vs_uv); }\n"
        "void step_b(out vec4 o) { o = texture(u_step_a, vs_uv); }\n"
        "void main() { f_color = texture(u_step_b, vs_uv); }\n",
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
        "uniform sampler2D u_step_late;\n"
        "uniform sampler2D u_step_early;\n"
        "void step_late(out vec4 o) { o = texture(u_step_early, vs_uv) + vec4(0.5, 0.0, 0.0, 1.0); }\n"
        "void step_early(out vec4 o) { o = vec4(0.25, 0.0, 0.0, 1.0); }\n"
        "void main() { f_color = texture(u_step_late, vs_uv); }\n",
    )
    assert node.compile_unit.errors == [], node.compile_unit.errors
    assert node.step_plan.order == ["early", "late"]

    # ONE render. In declaration order `late` would read an unwritten `early` (black)
    # and land at 0.5; correctly ordered it reads 0.25 and lands at 0.75.
    node.render(u_time=0.0)
    late = node.step_texture("late")
    assert late is not None
    assert np.frombuffer(late.read(), dtype=np.float32)[0] == pytest.approx(0.75)


def test_step_targets_follow_a_canvas_resize(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    # A scale-derived target that does not track the canvas samples at the wrong ratio,
    # with no error to show for it.
    node = _node(
        gl,
        tmp_path,
        "#version 330\n"
        "out vec4 f_color;\n"
        "in vec2 vs_uv;\n"
        "uniform sampler2D u_step_half;\n"
        "void step_half(out vec4 o) { o = vec4(3.0, 0.0, 0.0, 1.0); }\n"
        "void main() { f_color = texture(u_step_half, vs_uv); }\n",
        configs={"half": StepConfig(scale=0.5, dtype="f4")},
    )
    node.render(u_time=0.0)
    assert node._step_targets["half"][0].texture.size == (8, 8)

    node.canvas.set_size((64, 64))
    node.render(u_time=0.0)
    assert node._step_targets["half"][0].texture.size == (32, 32)
    texture = node.step_texture("half")
    assert texture is not None
    assert np.frombuffer(texture.read(), dtype=np.float32)[0] == pytest.approx(3.0)


def test_an_observer_step_sees_a_self_readers_current_frame(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    # The ping-pong swap must land before downstream steps read: a self-reader gets its
    # PREVIOUS frame, while everyone else gets its CURRENT one.
    node = _node(
        gl,
        tmp_path,
        "#version 330\n"
        "out vec4 f_color;\n"
        "in vec2 vs_uv;\n"
        "uniform sampler2D u_step_acc;\n"
        "uniform sampler2D u_step_watch;\n"
        "void step_acc(out vec4 o) { o = texture(u_step_acc, vs_uv) + vec4(1.0, 0.0, 0.0, 1.0); }\n"
        "void step_watch(out vec4 o) { o = texture(u_step_acc, vs_uv); }\n"
        "void main() { f_color = texture(u_step_watch, vs_uv); }\n",
    )
    assert node.compile_unit.errors == [], node.compile_unit.errors
    assert node.step_plan.self_reads == {"acc"}
    for expected in (1.0, 2.0, 3.0):
        node.render(u_time=0.0)
        acc = node.step_texture("acc")
        watch = node.step_texture("watch")
        assert acc is not None and watch is not None
        assert np.frombuffer(acc.read(), dtype=np.float32)[0] == pytest.approx(expected)
        assert np.frombuffer(watch.read(), dtype=np.float32)[0] == pytest.approx(
            expected
        )


def test_a_step_reading_three_others_binds_distinct_texture_units(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    # R3. A unit collision would make the sum wrong while everything still renders.
    node = _node(
        gl,
        tmp_path,
        "#version 330\n"
        "out vec4 f_color;\n"
        "in vec2 vs_uv;\n"
        "uniform sampler2D u_step_a;\n"
        "uniform sampler2D u_step_b;\n"
        "uniform sampler2D u_step_c;\n"
        "uniform sampler2D u_step_sum;\n"
        "void step_a(out vec4 o) { o = vec4(1.0, 0.0, 0.0, 1.0); }\n"
        "void step_b(out vec4 o) { o = vec4(2.0, 0.0, 0.0, 1.0); }\n"
        "void step_c(out vec4 o) { o = vec4(4.0, 0.0, 0.0, 1.0); }\n"
        "void step_sum(out vec4 o) {\n"
        "    o = texture(u_step_a, vs_uv) + texture(u_step_b, vs_uv) + texture(u_step_c, vs_uv);\n"
        "}\n"
        "void main() { f_color = texture(u_step_sum, vs_uv); }\n",
    )
    assert node.compile_unit.errors == [], node.compile_unit.errors
    node.render(u_time=0.0)
    total = node.step_texture("sum")
    assert total is not None
    # 1 + 2 + 4: any collision gives a different, still-plausible number.
    assert np.frombuffer(total.read(), dtype=np.float32)[0] == pytest.approx(7.0)


def test_a_multi_step_node_exports_its_chain(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    """Export must run the chain, not the bare final shader.

    Every export funnels through `render_media`, which brackets itself in
    `export_isolation`. A chain that did not run leaves the final step sampling an
    unwritten target, so the file exports black -- plausible, and wrong.
    """

    node = _node(
        gl,
        tmp_path,
        "#version 330\n"
        "out vec4 f_color;\n"
        "in vec2 vs_uv;\n"
        "uniform sampler2D u_step_a;\n"
        "void step_a(out vec4 o) { o = vec4(vs_uv.x, vs_uv.y, 0.5, 1.0); }\n"
        "void main() { f_color = texture(u_step_a, vs_uv); }\n",
    )
    assert node.compile_unit.errors == [], node.compile_unit.errors

    out = tmp_path / "out.png"
    node.render_media(
        MediaDetails(
            file_details=FileDetails(path=str(out)),
            resolution_details=ResolutionDetails(width=16, height=16),
            duration=0.0,
            fps=1.0,
            is_video=False,
        )
    )
    assert out.is_file()
    pixels = np.array(PILImage.open(out))
    assert pixels[:, :, :3].max() > 0, "the chain did not run: exported black"
    # The step writes a uv gradient, so opposite corners must differ.
    assert tuple(pixels[0, 0][:3]) != tuple(pixels[-1, -1][:3])


def test_a_persist_step_survives_a_recompile(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    """`persist` means "survives a recompile", and every real flow is a recompile.

    Ctrl+S, a lib hot-reload and a copilot edit are all invalidate-then-compile, so a
    target preserved by the first call and freed by the second makes the flag a no-op --
    which is what it was, while the Help panel advertised it.
    """
    node = _node(
        gl,
        tmp_path,
        "#version 330\n"
        "out vec4 f_color;\n"
        "in vec2 vs_uv;\n"
        "uniform sampler2D u_step_acc;\n"
        "void step_acc(out vec4 o) { o = texture(u_step_acc, vs_uv) + vec4(1.0, 0.0, 0.0, 1.0); }\n"
        "void main() { f_color = texture(u_step_acc, vs_uv); }\n",
        configs={"acc": StepConfig(dtype="f4", persist=True)},
    )
    for _ in range(3):
        node.render(u_time=0.0)
    accumulated = node.step_texture("acc")
    assert accumulated is not None
    assert np.frombuffer(accumulated.read(), dtype=np.float32)[0] == pytest.approx(3.0)

    node.invalidate()
    node.compile()
    node.render(u_time=0.0)

    after = node.step_texture("acc")
    assert after is not None
    # 4.0 = the accumulation continued. 1.0 would mean it restarted cold.
    assert np.frombuffer(after.read(), dtype=np.float32)[0] == pytest.approx(4.0)


def test_a_non_persist_step_restarts_cold_on_a_recompile(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    node = _node(
        gl,
        tmp_path,
        "#version 330\n"
        "out vec4 f_color;\n"
        "in vec2 vs_uv;\n"
        "uniform sampler2D u_step_acc;\n"
        "void step_acc(out vec4 o) { o = texture(u_step_acc, vs_uv) + vec4(1.0, 0.0, 0.0, 1.0); }\n"
        "void main() { f_color = texture(u_step_acc, vs_uv); }\n",
    )
    for _ in range(3):
        node.render(u_time=0.0)
    node.invalidate()
    node.compile()
    node.render(u_time=0.0)
    after = node.step_texture("acc")
    assert after is not None
    assert np.frombuffer(after.read(), dtype=np.float32)[0] == pytest.approx(1.0)


def test_exporting_a_feedback_node_twice_gives_the_same_frames(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    """D10: an export must not depend on how long the app has been open.

    `export_isolation` already re-instantiates a stateful SCRIPT per export for exactly
    this reason. A feedback step is the same class of state, so without a reset the same
    node exported before and after a few minutes of live preview produces different
    video -- measured at 13 vs 255 before this landed.
    """
    node = _node(
        gl,
        tmp_path,
        "#version 330\n"
        "out vec4 f_color;\n"
        "in vec2 vs_uv;\n"
        "uniform sampler2D u_step_acc;\n"
        "void step_acc(out vec4 o) { o = texture(u_step_acc, vs_uv) + vec4(0.05, 0.0, 0.0, 1.0); }\n"
        "void main() { f_color = vec4(texture(u_step_acc, vs_uv).r, 0.0, 0.0, 1.0); }\n",
    )

    def _export(name: str) -> bytes:
        out = tmp_path / name
        node.render_media(
            MediaDetails(
                file_details=FileDetails(path=str(out)),
                resolution_details=ResolutionDetails(width=16, height=16),
                duration=0.0,
                fps=1.0,
                is_video=False,
            )
        )
        return out.read_bytes()

    first = _export("a.png")
    for _ in range(20):  # the app being left open
        node.render(u_time=0.0)
    second = _export("b.png")
    assert first == second
