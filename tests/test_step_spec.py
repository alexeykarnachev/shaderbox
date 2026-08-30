"""The `// step` rider parser: every token, and every way of getting it wrong.

The load-bearing property is D2 -- a rider that merely LOOKS like a step must be an error,
never a silently-ignored comment. Falling through would leave an ordinary sampler bound to
the shipped default image, so the user gets a picture and it is the wrong one.
"""

from pathlib import Path

import pytest

from shaderbox.step_spec import DEFAULT_DTYPE, StepSpec, parse_steps

_PATH = Path("node.frag.glsl")


def _src(decl: str, body: str = "void step_blur(out vec4 o) { o = vec4(1.0); }") -> str:
    return f"#version 330\nout vec4 f_color;\n{decl}\n{body}\nvoid main() {{}}\n"


def test_the_float_default_is_the_measured_safe_one() -> None:
    # 063 measured f1 saturating at 255 on the FIRST accumulate pass where f2 reached
    # exactly 7.0. Both the dataclass default and the parser's must agree on f2, or an
    # accumulating step silently gets an 8-bit target.
    assert DEFAULT_DTYPE == "f2"
    assert StepSpec(name="x", sampler="u_x").dtype == "f2"


def test_bare_marker_uses_every_default() -> None:
    result = parse_steps(_src("uniform sampler2D u_blur;  // step"), _PATH)
    assert result.errors == []
    (step,) = result.steps
    assert step.name == "blur"
    assert step.sampler == "u_blur"
    assert step.fn_name == "step_blur"
    assert step.scale == 1.0
    assert step.size is None
    # f2, not f1: 8-bit saturates on the first accumulate pass (063), so the safe
    # value is the default and f1 is the opt-in.
    assert step.dtype == "f2"
    assert step.filter_linear is True
    # clamp, inverting moderngl's repeat_x/y=True, which is wrong for a feedback border.
    assert step.wrap is False
    assert step.persist is False


def test_every_option_parses() -> None:
    result = parse_steps(
        _src("uniform sampler2D u_blur;  // step, scale: 0.25, f4, nearest, repeat, persist"),
        _PATH,
    )
    assert result.errors == []
    (step,) = result.steps
    assert step.scale == 0.25
    assert step.dtype == "f4"
    assert step.filter_linear is False
    assert step.wrap is True
    assert step.persist is True


def test_absolute_size_wins_over_scale() -> None:
    result = parse_steps(
        _src("uniform sampler2D u_blur;  // step, size: 320x240, scale: 0.5"), _PATH
    )
    assert result.errors == []
    (step,) = result.steps
    assert step.size == (320, 240)
    assert step.target_size((1280, 960)) == (320, 240)


def test_scale_resolves_against_the_canvas_and_never_hits_zero() -> None:
    result = parse_steps(_src("uniform sampler2D u_blur;  // step, scale: 0.25"), _PATH)
    (step,) = result.steps
    assert step.target_size((1280, 960)) == (320, 240)
    # A deep cascade level must not round down to a zero-sized target.
    assert step.target_size((2, 2)) == (1, 1)


def test_an_ordinary_comment_is_left_alone() -> None:
    result = parse_steps(
        _src("uniform sampler2D u_blur;  // the scene texture", body="void main() {}"),
        _PATH,
    )
    assert result.errors == []
    assert result.steps == []


def test_a_sampler_with_no_comment_is_left_alone() -> None:
    result = parse_steps(
        _src("uniform sampler2D u_blur;", body="void main() {}"), _PATH
    )
    assert result.errors == []
    assert result.steps == []


@pytest.mark.parametrize("typo", ["stp", "stemp", "setp", "steps", "ste"])
def test_a_near_miss_marker_is_a_loud_error(typo: str) -> None:
    # D2: the whole point. Without this the sampler silently stays an ordinary
    # texture input and renders the default image.
    result = parse_steps(_src(f"uniform sampler2D u_blur;  // {typo}, scale: 0.5"), _PATH)
    assert result.steps == []
    assert len(result.errors) == 1
    assert "did you mean" in result.errors[0].message
    assert result.errors[0].line == 2


def test_an_unknown_option_is_an_error() -> None:
    result = parse_steps(
        _src("uniform sampler2D u_blur;  // step, floaty"), _PATH
    )
    assert result.steps == []
    assert any("unknown option" in e.message for e in result.errors)


@pytest.mark.parametrize(
    "rider",
    [
        "step, scale: wide",
        "step, scale: 0",
        "step, scale: -1",
        "step, size: 320",
        "step, size: 320xtall",
        "step, size: 0x0",
    ],
)
def test_a_malformed_value_is_an_error(rider: str) -> None:
    result = parse_steps(_src(f"uniform sampler2D u_blur;  // {rider}"), _PATH)
    assert result.steps == []
    assert len(result.errors) == 1


def test_a_declared_step_with_no_body_is_an_error() -> None:
    result = parse_steps(
        _src("uniform sampler2D u_blur;  // step", body="void main() {}"), _PATH
    )
    assert result.steps == []
    assert any("has no body" in e.message for e in result.errors)


def test_a_body_with_no_declaration_is_an_error() -> None:
    # A `step_x` body nobody declares would never run -- a typo, not intent.
    result = parse_steps(
        _src("uniform sampler2D u_other;", body="void step_blur(out vec4 o) { o = vec4(1.0); }"),
        _PATH,
    )
    assert result.steps == []
    assert any("is never run" in e.message for e in result.errors)


def test_a_duplicate_step_is_an_error() -> None:
    src = (
        "#version 330\n"
        "uniform sampler2D u_blur;  // step\n"
        "uniform sampler2D u_blur;  // step\n"
        "void step_blur(out vec4 o) { o = vec4(1.0); }\n"
        "void main() {}\n"
    )
    result = parse_steps(src, _PATH)
    assert len(result.steps) == 1
    assert any("already declared" in e.message for e in result.errors)


def test_several_steps_keep_declaration_order() -> None:
    src = (
        "#version 330\n"
        "uniform sampler2D u_a;  // step, scale: 0.5\n"
        "uniform sampler2D u_b;  // step, f1\n"
        "void step_a(out vec4 o) { o = vec4(1.0); }\n"
        "void step_b(out vec4 o) { o = vec4(2.0); }\n"
        "void main() {}\n"
    )
    result = parse_steps(src, _PATH)
    assert result.errors == []
    assert [s.name for s in result.steps] == ["a", "b"]
    assert result.steps[1].dtype == "f1"


def test_a_sampler_without_the_u_prefix_still_names_its_step() -> None:
    src = (
        "#version 330\n"
        "uniform sampler2D blur;  // step\n"
        "void step_blur(out vec4 o) { o = vec4(1.0); }\n"
        "void main() {}\n"
    )
    result = parse_steps(src, _PATH)
    assert result.errors == []
    assert result.steps[0].name == "blur"


def test_errors_carry_the_declaration_line_for_click_to_jump() -> None:
    src = (
        "#version 330\n"
        "\n"
        "\n"
        "uniform sampler2D u_blur;  // step, nonsense\n"
        "void main() {}\n"
    )
    result = parse_steps(src, _PATH)
    assert result.errors[0].line == 3
    assert result.errors[0].path == _PATH
