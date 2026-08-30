"""Finding steps by NAME, and configuring their targets separately.

A sampler named `u_step_<name>` is a render step. Nothing is parsed out of comments: a
comment is not part of the language, so it cannot be checked, it collides with prose a
user writes for themselves, and a typo in one is indistinguishable from a sentence.

The consequence worth testing is the absence of a whole error class -- a shader that
never heard of steps cannot be broken by one, whatever it writes in its comments.
"""

from pathlib import Path

import pytest

from shaderbox.step_spec import (
    DEFAULT_DTYPE,
    StepConfig,
    find_steps,
    step_name_for,
)

_PATH = Path("node.frag.glsl")


def _src(decl: str, body: str = "void step_blur(out vec4 o) { o = vec4(1.0); }") -> str:
    return f"#version 330\nout vec4 f_color;\n{decl}\n{body}\nvoid main() {{}}\n"


def test_a_prefixed_sampler_is_a_step() -> None:
    result = find_steps(_src("uniform sampler2D u_step_blur;"), _PATH)
    assert result.errors == []
    (step,) = result.steps
    assert step.name == "blur"
    assert step.sampler == "u_step_blur"
    assert step.fn_name == "step_blur"


def test_an_ordinary_sampler_is_not_a_step() -> None:
    result = find_steps(
        _src("uniform sampler2D u_image;", body="void main() {}"), _PATH
    )
    assert result.errors == []
    assert result.steps == []


@pytest.mark.parametrize(
    "comment",
    [
        "// step",
        "// steps of the animation",
        "// stop",
        "// the step function below",
        "// step, scale: 0.5, f2",
    ],
)
def test_a_comment_never_makes_or_breaks_a_step(comment: str) -> None:
    # The whole point of the naming convention: comments carry no meaning, so an
    # ordinary English one cannot declare a step and cannot break a shader either.
    result = find_steps(
        _src(f"uniform sampler2D u_image;  {comment}", body="void main() {}"), _PATH
    )
    assert result.errors == []
    assert result.steps == []


def test_a_comment_does_not_configure_a_step() -> None:
    # A rider-looking comment on a REAL step is inert: the config comes from node state.
    result = find_steps(
        _src("uniform sampler2D u_step_blur;  // step, scale: 0.25, f4"), _PATH
    )
    assert result.errors == []
    (step,) = result.steps
    assert step.config.scale == 1.0
    assert step.config.dtype == DEFAULT_DTYPE


def test_config_comes_from_node_state() -> None:
    configs = {"blur": StepConfig(scale=0.25, dtype="f4", filter_linear=False)}
    result = find_steps(_src("uniform sampler2D u_step_blur;"), _PATH, configs=configs)
    (step,) = result.steps
    assert step.config.scale == 0.25
    assert step.config.dtype == "f4"
    assert step.config.filter_linear is False
    assert step.target_size((800, 600)) == (200, 150)


def test_a_step_with_no_config_gets_working_defaults() -> None:
    # A freshly-declared step must render correctly before anyone opens the panel.
    result = find_steps(_src("uniform sampler2D u_step_blur;"), _PATH)
    (step,) = result.steps
    # f2 because 063 measured f1 saturating at 255 on the first accumulate pass.
    assert step.config.dtype == "f2"
    assert step.config.filter_linear is True
    assert step.config.wrap is False  # clamp, inverting moderngl's repeat default
    assert step.config.persist is False
    assert step.target_size((1280, 960)) == (1280, 960)


def test_every_config_field_is_reachable_from_the_panel() -> None:
    """No field the user cannot set.

    `StepConfig` previously carried an absolute `size` that nothing could write, which is
    a field nobody maintains and every reader has to reason about. The panel's combos are
    the contract: what the panel can set is what the config holds.
    """
    from dataclasses import fields

    settable = {"scale", "dtype", "filter_linear", "wrap", "persist"}
    assert {f.name for f in fields(StepConfig)} == settable


def test_a_scaled_target_never_rounds_to_zero() -> None:
    configs = {"blur": StepConfig(scale=0.05)}
    result = find_steps(_src("uniform sampler2D u_step_blur;"), _PATH, configs=configs)
    assert result.steps[0].target_size((2, 2)) == (1, 1)


def test_config_for_an_unknown_step_is_simply_unused() -> None:
    # A sampler was renamed or deleted; its stale config must not resurrect it.
    configs = {"ghost": StepConfig(scale=0.5)}
    result = find_steps(_src("uniform sampler2D u_step_blur;"), _PATH, configs=configs)
    assert [s.name for s in result.steps] == ["blur"]
    assert result.errors == []


def test_a_declared_step_with_no_body_is_an_error() -> None:
    result = find_steps(
        _src("uniform sampler2D u_step_blur;", body="void main() {}"), _PATH
    )
    assert result.steps == []
    assert any("has no body" in e.message for e in result.errors)


def test_a_body_with_no_sampler_is_an_error() -> None:
    # A `step_x` body nobody declares never runs -- a step the author wrote that does
    # nothing, with nothing on screen to say so.
    result = find_steps(
        _src(
            "uniform sampler2D u_image;",
            body="void step_blur(out vec4 o) { o = vec4(1.0); }",
        ),
        _PATH,
    )
    assert result.steps == []
    assert any("is never run" in e.message for e in result.errors)


@pytest.mark.parametrize(
    "helper",
    [
        "void step_warp() { }",
        "float step_curve(float x) { return x; }",
        "void step_march(vec2 p) { }",
    ],
)
def test_a_step_prefixed_helper_with_another_signature_is_left_alone(
    helper: str,
) -> None:
    # `step_` is an ordinary prefix in shader code. Only the exact step signature counts.
    result = find_steps(f"#version 330\n{helper}\nvoid main() {{}}\n", _PATH)
    assert result.errors == []
    assert result.steps == []


def test_a_duplicate_step_is_an_error() -> None:
    src = (
        "#version 330\n"
        "uniform sampler2D u_step_blur;\n"
        "uniform sampler2D u_step_blur;\n"
        "void step_blur(out vec4 o) { o = vec4(1.0); }\n"
        "void main() {}\n"
    )
    result = find_steps(src, _PATH)
    assert len(result.steps) == 1
    assert any("already declared" in e.message for e in result.errors)


def test_several_steps_keep_declaration_order() -> None:
    src = (
        "#version 330\n"
        "uniform sampler2D u_step_a;\n"
        "uniform sampler2D u_step_b;\n"
        "void step_a(out vec4 o) { o = vec4(1.0); }\n"
        "void step_b(out vec4 o) { o = vec4(2.0); }\n"
        "void main() {}\n"
    )
    result = find_steps(src, _PATH)
    assert result.errors == []
    assert [s.name for s in result.steps] == ["a", "b"]


def test_errors_carry_the_declaration_line_for_click_to_jump() -> None:
    src = "#version 330\n\n\nuniform sampler2D u_step_blur;\nvoid main() {}\n"
    result = find_steps(src, _PATH)
    assert result.errors[0].line == 3
    assert result.errors[0].path == _PATH


@pytest.mark.parametrize(
    ("sampler", "expected"),
    [
        ("u_step_blur", "blur"),
        ("u_step_c0", "c0"),
        ("u_image", None),
        ("u_step_", None),
        ("step_blur", None),
    ],
)
def test_step_name_for(sampler: str, expected: str | None) -> None:
    assert step_name_for(sampler) == expected
