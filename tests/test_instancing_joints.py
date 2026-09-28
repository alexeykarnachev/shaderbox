"""The seams where one flow's output is another's input, through the real system.

Each test drives a path no single flow could build or gate alone: the script engine
decides a state the draw cannot see, the draw decides states the engine cannot see, and a
surface reads both.

Every test here drives real objects. An earlier version asserted against
`inspect.getsource(...)` for three of them, and a reviewer broke all three by keeping the
searched text and deleting the mechanism -- the defect these gates exist to catch,
reproduced inside the gates themselves.
"""

from pathlib import Path

import moderngl
import numpy as np
import pytest

from shaderbox.app import App
from shaderbox.core import Pass
from shaderbox.instanced_outcome import InstancedOutcome
from shaderbox.notifications import Notifications
from shaderbox.pass_graph import BLEND_MODES, TargetConfig
from shaderbox.scripting.engine import DocumentScripts, ScriptEngine
from shaderbox.shader_source import ShaderSource

# A pass with NO `flat in`: the shape the engine's `no_fields` verdict is about.
_PLAIN_FRAGMENT = """#version 460 core
in vec2 vs_uv;
out vec4 frag_color;
void main() { frag_color = vec4(vs_uv, 0.0, 1.0); }
"""

# The same pass once the author declares the fields the verdict asked for.
_ENTITY_FRAGMENT = """#version 460 core
in vec2 vs_quad;
flat in vec2 pos;
flat in float radius;
out vec4 frag_color;
void main() {
    if (length(vs_quad) > 1.0) discard;
    frag_color = vec4(1.0, 0.0, 0.0, 1.0);
}
"""


def _pass_at(gl_ctx: moderngl.Context, path: Path, text: str) -> Pass:
    path.write_text(text)
    render_pass = Pass(
        gl=gl_ctx,
        source=ShaderSource.load(path),
        canvas_size=(8, 8),
        target=TargetConfig(),
    )
    render_pass.compile()
    return render_pass


def test_no_fields_survives_the_frame_that_follows_it(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    """The engine decides `no_fields` during the tick; the draw runs after and must not
    erase it.

    Both verdicts are true of the same frame -- the draw really did draw a full-screen
    quad -- but only the engine's is about the population, and letting the draw win is
    exactly the silence I1 names. Driven through a REAL draw: an earlier version read
    `render`'s source for the word `no_fields`, and a reviewer broke it by keeping the
    word and deleting the mechanism.
    """
    render_pass = _pass_at(gl_ctx, tmp_path / "plain.frag.glsl", _PLAIN_FRAGMENT)
    assert render_pass.program is not None
    assert not render_pass.entity_fields, (
        "fixture must be a pass with NO flat in fields"
    )

    render_pass.last_outcome = InstancedOutcome("plain", "no_fields")
    render_pass.render(u_time=0.0)

    assert render_pass.last_outcome.state == "no_fields", (
        "the draw erased the engine's verdict -- a population offered to a pass with no "
        "`flat in` is dropped and nothing says so, which is I1's silence restored"
    )


def test_every_blend_mode_round_trips_through_a_target_config() -> None:
    # F3 gates this against graph.json; here it is the type itself, so a mode that cannot
    # survive validation is caught even if no example document uses it yet.
    for mode in BLEND_MODES:
        config = TargetConfig(blend=mode)
        assert TargetConfig.model_validate(config.model_dump()).blend == mode


def test_a_blend_change_alone_never_reallocates() -> None:
    """The two-path feedback wipe (102 D5), asserted where both paths read it.

    `Document.set_pass_target` and `Pass.set_target` both decide reallocation, and both
    asked `==` before this wave. A blend-only change must be invisible to both.
    """
    base = TargetConfig()
    for mode in BLEND_MODES:
        changed = base.model_copy(update={"blend": mode})
        assert base.allocates_same_as(changed)


@pytest.mark.parametrize("dtype", ["f1", "f2", "f4"])
def test_a_population_column_must_be_f4_i4_or_u4(dtype: str) -> None:
    """The dtype contract is CHECKED, never cast (100). The copilot's prompt block states
    this rule, so it is gated here rather than only in the text that teaches it."""
    from shaderbox.instanced import EntityField, validate_population

    fields = (EntityField(name="pos", glsl_type="vec2", line=0),)
    numpy_dtype = {"f1": np.uint8, "f2": np.float16, "f4": np.float32}[dtype]
    column = np.zeros((4, 2), dtype=numpy_dtype)
    _, problem = validate_population(fields, {"pos": column})
    if dtype == "f4":
        assert problem is None
    else:
        assert problem is not None, (
            f"{dtype} column accepted -- the check was cast away"
        )


def test_the_no_fields_verdict_clears_when_the_pass_recompiles(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    """A sticky verdict is as wrong as a missing one.

    `no_fields` outranks the draw, which is what makes it visible -- and that precedence
    would pin it forever once set.

    The fixture recompiles WITHOUT invalidating, which is the path the reset is the only
    guard for: a lib file changed, or a forced rebuild. Going through `release_program`
    instead passes with the reset deleted, because `invalidate` drops the program and
    `render` then returns early -- a fixture that would report the same thing whether or
    not the line exists. MEASURED both ways before choosing this one.
    """
    render_pass = _pass_at(gl_ctx, tmp_path / "gains.frag.glsl", _ENTITY_FRAGMENT)
    assert render_pass.entity_fields, "fixture must compile with `flat in` fields"

    render_pass.last_outcome = InstancedOutcome("gains", "no_fields")
    render_pass.compile()

    assert render_pass.last_outcome.state != "no_fields", (
        "the verdict survived the compile that fixed it -- the strip keeps reporting a "
        "defect the author has already corrected"
    )


class _FakeApp:
    """Stands in for App so its handler runs without a GL context or a window.

    It invokes the REAL `App._on_script_key_warning` unbound rather than reimplementing
    it, and provides only `notifications` -- so a handler that grows a second dependency
    fails loudly here instead of quietly passing, which a Mock would not do.
    """

    def __init__(self, notifications: Notifications) -> None:
        self.notifications = notifications

    def on_key_warning(
        self, document_id: str, pass_name: str, name: str, reason: str, message: str
    ) -> None:
        App._on_script_key_warning(
            self,  # type: ignore[arg-type]
            document_id,
            pass_name,
            name,
            reason,
            message,
        )


def test_a_non_landing_key_reaches_the_app_through_the_whole_seam(
    tmp_path: Path,
) -> None:
    """102 D-F: a non-landing key warns in the logs AND in the notifications, routed
    through an injected callback because the engine imports no imgui (D3a).

    Traverses the whole seam, engine to notification stack. An earlier version searched
    `App.__init__` and the handler for two substrings, and severing the MIDDLE link in
    `project_session.py` -- which is in neither searched function -- left it green.
    """
    from shaderbox.engine_uniforms import ENGINE_DRIVEN_UNIFORMS
    from shaderbox.project_session import ProjectSession

    notifications = Notifications()
    app = _FakeApp(notifications)
    engine = ScriptEngine(ENGINE_DRIVEN_UNIFORMS, on_key_warning=app.on_key_warning)

    engine._warn(
        document_id="d",
        pass_name="blur",
        name="u_absent",
        reason="no_such_uniform",
        message="no pass declares 'u_absent'",
        errors={},
        skipped=set(),
        scripts=DocumentScripts(tmp_path),
    )
    assert notifications._stack, "the warning did not reach the notification stack"
    assert "u_absent" in notifications._stack[0].text

    # The MIDDLE link, and the reason this test exists: `ProjectSession` must hand the
    # callback to the engine it builds. Asserting the parameter merely EXISTS passes when
    # the constructor accepts it and drops it on the floor -- which is the shape a
    # reviewer used to defeat the previous version of this gate.
    session = ProjectSession.__new__(ProjectSession)
    session._on_key_warning = app.on_key_warning
    built = ProjectSession._build_script_engine(session)
    assert built._on_key_warning == app.on_key_warning, (
        "ProjectSession built its engine without forwarding the warn callback -- the "
        "engine fires into the no-op and nothing reaches the screen"
    )


def test_a_population_that_returns_to_its_start_is_not_reported_static() -> None:
    """A population is diffed FIRST against EVERY later sample, not first against last.

    The fixture's endpoints are IDENTICAL and its middle differs -- an orbit that comes
    back round, which is ordinary for a flock. Comparing the two ends reports it STATIC,
    which is C1's own false verdict in a narrower window, and a fixture whose endpoints
    differ cannot tell the two implementations apart.
    """
    from shaderbox.copilot.backend import _population_motion
    from shaderbox.scripting.engine import ScriptProbe

    def stats(
        extent: float,
    ) -> dict[
        str, tuple[int, dict[str, tuple[str, tuple[int, ...], float, float]], str]
    ]:
        return {"swarm": (100, {"pos": ("float32", (100, 2), -extent, extent)}, "")}

    probe = ScriptProbe(
        compile_error=None,
        driven=set(),
        per_key_errors=[],
        orphan_keys=[],
        samples=[],
        population_samples=[
            (0.0, stats(1.0)),
            (0.5, stats(0.2)),
            (1.0, stats(1.0)),
        ],
    )

    lines = _population_motion(probe, eps=1e-6)
    assert lines, "no verdict produced at all -- the fixture never reached the code"
    assert "ANIMATING" in lines[0], (
        f"a population that moved and returned was reported static: {lines[0]}"
    )


def test_the_instanced_stub_hint_is_valid_python(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    """The stub is text the model COPIES, so it has to parse.

    An earlier version carried a doubled comment marker on every column line
    (`#         #`) and indented the whole block to a `return` it merely sits after,
    describing a nesting that is not there. Both are invisible to `ast.parse`, since a
    comment parses whatever it says, so they are asserted against the text.

    The `}}` in the snippet is CORRECT and deliberately not asserted against: the line
    opens `{"@instances": {`, so two closing braces is what balances it. A review read it
    as an f-string escaping bug, and checking the braces actually balance is the
    assertion that distinguishes the two readings.
    """
    import ast

    from shaderbox.copilot.backend import _teach_instances_in_stub

    render_pass = _pass_at(gl_ctx, tmp_path / "swarm.frag.glsl", _ENTITY_FRAGMENT)
    assert render_pass.entity_fields, "fixture must declare `flat in` fields"

    class _Doc:
        def __init__(self) -> None:
            self.passes = {"swarm": render_pass}

    stub = "class Behavior:\n    def update(self, context):\n        return {}\n"
    taught = _teach_instances_in_stub(stub, _Doc())  # type: ignore[arg-type]

    assert taught != stub, "the hint was not appended -- the fixture never reached it"
    assert "@instances" in taught
    ast.parse(taught)  # raises SyntaxError on the `}}` defect

    hint = taught[len(stub) :]
    assert "#         #" not in hint, "doubled comment marker on a column line"
    # The snippet the model copies must be balanced, which is the real question behind
    # the `}}`: count them rather than banning a spelling.
    snippet = hint.replace("#", " ")
    assert snippet.count("{") == snippet.count("}"), (
        f"the copied snippet's braces do not balance: {hint}"
    )
    for line in hint.splitlines():
        if line.strip():
            assert line.startswith("#"), (
                f"hint line is indented as if it were inside the class: {line!r}"
            )
