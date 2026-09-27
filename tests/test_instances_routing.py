"""`@instances`: how a script's entity columns reach the pass that draws them.

The reserved `@` namespace exists so a mistyped engine key cannot be mistaken for a
uniform the shader has yet to declare -- that case is deliberately SILENT (079 D5), which
is right for an author mid-edit and wrong for engine vocabulary, where the cost is a blank
frame with an empty error strip.
"""

from pathlib import Path

import moderngl
import numpy as np
import pytest

from shaderbox.core import Pass
from shaderbox.document import Document
from shaderbox.instanced import validate_population
from shaderbox.intel.glsl import entity_fields
from shaderbox.intel.script import _is_number
from shaderbox.scripting.context import ScriptContext
from shaderbox.scripting.engine import ScriptEngine
from shaderbox.shader_source import ShaderSource
from shaderbox.uniform_coerce import is_number

_FRAGMENT = """#version 460 core
in vec2 vs_uv;
in vec2 vs_quad;
flat in vec2  pos;
flat in float radius;
out vec4 frag_color;
void main(){
    if (length(vs_quad) > 1.0) discard;
    frag_color = vec4(1.0);
}
"""

_SCRIPT = """
import numpy as np
from shaderbox.scripting import ScriptBehavior, ScriptContext


class Behavior(ScriptBehavior):
    def __init__(self) -> None:
        self.pos = np.array([[-0.5, 0.0], [0.5, 0.0]], dtype="f4")
        self.radius = np.array([0.2, 0.2], dtype="f4")

    def update(self, context: ScriptContext) -> dict:
        return {"swarm": {"@instances": {"pos": self.pos, "radius": self.radius}}}
"""


def _document(tmp_path: Path, gl_ctx: moderngl.Context, script: str) -> tuple:
    (tmp_path / "passes").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "passes" / "swarm.frag.glsl").write_text(_FRAGMENT)
    (tmp_path / "scripts" / "script.py").write_text(script)
    (tmp_path / "graph.json").write_text(
        '{"version": 2, "output": "swarm", "passes": {"swarm": {}}}'
    )
    (tmp_path / "document.json").write_text('{"uniforms": {}}')
    document, _ = Document.load_from_dir(tmp_path, gl=gl_ctx, canvas_size=(320, 180))
    document.render()  # the first render is what compiles the pass
    engine = ScriptEngine()
    engine.reload("doc", tmp_path / "scripts")
    return document, engine


def test_a_script_s_columns_reach_the_pass_and_become_entities(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    document, engine = _document(tmp_path, gl_ctx, _SCRIPT)
    engine.tick("doc", document, ScriptContext(t=0.0, dt=1 / 60, frame=0))
    assert set(document.passes["swarm"].pending_instances) == {"pos", "radius"}

    document.render()
    pixels = np.frombuffer(
        document.render_pass.canvas.fbo.read(components=4, dtype="f2"), dtype="f2"
    ).reshape(180, 320, 4)
    lit = pixels[..., 3] > 0.5
    columns = np.where(lit.any(axis=0))[0]
    # Two entities at two positions must draw in two places. One entity could not tell a
    # per-instance binding from a per-vertex one.
    assert 1 + len(np.where(np.diff(columns) > 1)[0]) == 2


def test_an_unrecognised_engine_key_is_an_error_not_a_silent_orphan(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    document, engine = _document(
        tmp_path, gl_ctx, _SCRIPT.replace("@instances", "@instance")
    )
    engine.tick("doc", document, ScriptContext(t=0.0, dt=1 / 60, frame=0))
    status = engine.script_status("doc")
    assert status is not None
    messages = " ".join(error.message for _, _, error in status.soft_errors)
    assert "@instance" in messages and "@instances" in messages
    assert document.passes["swarm"].pending_instances == {}


def test_the_reserved_key_outside_a_pass_block_says_so(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # Without the wrapper the key looks like a pass name, and the honest-looking report
    # is "no pass named '@instances'" -- true, and useless to the author who forgot the
    # pass.
    bare = _SCRIPT.replace(
        'return {"swarm": {"@instances": {"pos": self.pos, "radius": self.radius}}}',
        'return {"@instances": {"pos": self.pos, "radius": self.radius}}',
    )
    document, engine = _document(tmp_path, gl_ctx, bare)
    engine.tick("doc", document, ScriptContext(t=0.0, dt=1 / 60, frame=0))
    status = engine.script_status("doc")
    assert status is not None
    messages = " ".join(e.message for _, _, e in status.soft_errors)
    assert "INSIDE a pass block" in messages


@pytest.mark.parametrize(
    ("columns", "expected"),
    [
        ({"pos": np.zeros((4, 2), "f4")}, "no column for radius"),
        (
            {"pos": np.zeros((4, 2), "f4"), "radius": np.zeros(4, "f8")},
            "expected f4",
        ),
        (
            {"pos": np.zeros((4, 2), "f4"), "radius": np.zeros(3, "f4")},
            "disagree",
        ),
        (
            {"pos": np.zeros((4, 3), "f4"), "radius": np.zeros(4, "f4")},
            "wants (N, 2)",
        ),
        (
            {
                "pos": np.zeros((4, 2), "f4"),
                "radius": np.zeros(4, "f4"),
                "mass": np.zeros(4, "f4"),
            },
            "nothing declares mass",
        ),
        (
            {"pos": np.zeros((2, 4), "f4").T, "radius": np.zeros(4, "f4")},
            "not contiguous",
        ),
    ],
)
def test_a_bad_population_is_refused_with_the_author_s_own_field_named(
    columns: dict, expected: str
) -> None:
    # Checked rather than converted: numpy's own assignment casts silently, so an f8
    # column would be accepted and 1e40 would arrive as inf. Nothing is truncated either
    # -- a data array quietly losing its tail is corruption wearing a plausible picture.
    fields = entity_fields(_FRAGMENT)
    count, problem = validate_population(fields, columns)
    assert problem is not None and expected in problem
    assert count == 0


def test_a_numpy_scalar_is_a_number_at_every_width_but_a_numpy_bool_is_not() -> None:
    # This feature makes numpy routine in scripts, and the old rule split on whether a
    # type happened to subclass `float`: f8 passed, f4 did not, and `np.mean` over an f4
    # array returns f4 -- so the rejected case was the common one.
    assert is_number(np.float32(1.0)) and is_number(np.float64(1.0))
    assert is_number(np.int32(1)) and is_number(np.uint8(3))
    assert not is_number(np.bool_(True))
    assert not is_number(True)
    # Assert the MECHANISM, not the answer: `np.bool_` is refused today by falling
    # through both checks, so the explicit guard against it is a barrier for a future
    # widening and deleting it changes nothing measurable. What WOULD change the answer
    # is numpy making `np.bool_` a number, and that is what this pins.
    assert not isinstance(np.bool_(True), np.number), (
        "np.bool_ became an np.number -- the explicit guard is now load-bearing"
    )


def test_the_two_number_rules_agree() -> None:
    """`uniform_coerce.is_number` and `intel.script._is_number` must answer the same.

    They are deliberately separate -- the intel package stays GL-free and importing the
    coercion would pull in moderngl -- so nothing but this stops them drifting. The cost
    of drift is a completion offering a declaration the next tick refuses, which reads as
    the editor being wrong about the author's own script.

    Enumerated over the whole domain rather than spot-checked: the pair already drifted
    once on `np.float32`, where one said number and the other did not.
    """
    domain: list[object] = [
        1,
        0,
        -3,
        1.5,
        0.0,
        True,
        False,
        np.bool_(True),
        np.bool_(False),
        "s",
        None,
        [1.0],
        (),
        {},
    ]
    for dtype in ("f2", "f4", "f8", "i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8"):
        domain.append(np.dtype(dtype).type(1))
    for value in domain:
        assert is_number(value) is _is_number(value), (
            f"{type(value).__name__} splits the two rules"
        )


def test_a_population_never_enters_the_probe_s_samples(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    """The copilot's dry-run samples carry scalars, never a population.

    `ScriptProbe.samples` exists so an agent can see a value MOVE between frames, and the
    comparison behind it (`_values_differ`) falls through to `a != b` for anything it does
    not recognise. Two ndarrays answer that elementwise, and the caller then takes the
    truth value of an array -- a ValueError on a live simulation. Copying megabytes per
    sample to reach that crash would be the other half of the cost.

    A population stays out because it is never a DRIVEN uniform: it drives no uniform row
    at all. This pins that, since the cheapest future change -- reporting it as driven so
    the strip can show it -- reintroduces both problems at once.
    """
    (tmp_path / "passes").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "passes" / "swarm.frag.glsl").write_text(
        _FRAGMENT.replace(
            "out vec4 frag_color;", "uniform float u_glow = 0.5;\nout vec4 frag_color;"
        ).replace("vec4(1.0)", "vec4(u_glow)")
    )
    (tmp_path / "scripts" / "script.py").write_text(
        _SCRIPT.replace(
            'return {"swarm": {"@instances": {"pos": self.pos, "radius": self.radius}}}',
            'return {"swarm": {"@instances": {"pos": self.pos, '
            '"radius": self.radius}, "u_glow": 0.5 + 0.1 * context.frame}}',
        )
    )
    (tmp_path / "graph.json").write_text(
        '{"version": 2, "output": "swarm", "passes": {"swarm": {}}}'
    )
    (tmp_path / "document.json").write_text('{"uniforms": {}}')
    document, _ = Document.load_from_dir(tmp_path, gl=gl_ctx, canvas_size=(64, 64))
    document.render()
    engine = ScriptEngine()
    engine.reload("doc", tmp_path / "scripts")

    probe = engine.dry_run("doc", document, (0.0, 0.5, 1.0), 60.0)

    # The scalar IS sampled, which is what proves the probe ran and reached this script
    # rather than returning empty for some unrelated reason.
    assert ("swarm", "u_glow") in probe.driven
    assert any(("swarm", "u_glow") in values for _, values in probe.samples)
    for _, values in probe.samples:
        for key, value in values.items():
            assert not isinstance(value, np.ndarray), (
                f"{key} carries a whole population"
            )


def test_a_field_declared_but_not_yet_read_does_not_crash_the_frame(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    """Declare a field, read it in the next edit -- the ordinary authoring step.

    The driver DEAD-STRIPS an attribute the fragment shader does not read, so the linked
    program has no `a_mass` while the declaration plainly does, and binding it raised a
    KeyError out of `Document.render` -- the frame loop.
    """
    source = tmp_path / "swarm.frag.glsl"
    source.write_text(
        _FRAGMENT.replace(
            "flat in float radius;", "flat in float radius;\nflat in float mass;"
        )
    )
    render_pass = Pass(
        gl=gl_ctx, source=ShaderSource.load(source), canvas_size=(64, 64)
    )
    render_pass.compile()
    assert [f.name for f in render_pass.entity_fields] == ["pos", "radius", "mass"]
    # The premise, checked: the driver really did drop it. Without this the test could
    # pass because the attribute survived, proving nothing about the guard.
    assert render_pass.program is not None and "a_mass" not in render_pass.program

    render_pass.render(
        u_time=0.0,
        instances={
            "pos": np.array([[-0.4, 0.0], [0.4, 0.0]], dtype="f4"),
            "radius": np.full(2, 0.2, dtype="f4"),
            "mass": np.ones(2, dtype="f4"),
        },
    )
    pixels = np.frombuffer(
        render_pass.canvas.fbo.read(components=4, dtype="f2"), dtype="f2"
    )
    assert float(pixels.max()) > 0.0, "the population still has to draw"


def test_an_invalid_population_draws_nothing_rather_than_the_whole_canvas(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    """A wrong dtype must not paint the entity shader over everything.

    "No population" and "a population that does not match" returned the same value, and
    the draw read it as "go fullscreen" -- so a single `dtype="f8"`, which is what numpy
    gives you unless you ask otherwise, filled the canvas.
    """
    source = tmp_path / "swarm.frag.glsl"
    # No `discard`: a fullscreen draw then covers every pixel, so the failure is visible
    # rather than masked by the shape test.
    source.write_text(
        """#version 460 core
in vec2 vs_quad;
flat in vec2  pos;
flat in float radius;
out vec4 frag_color;
void main(){ frag_color = vec4(1.0); }
"""
    )
    render_pass = Pass(
        gl=gl_ctx, source=ShaderSource.load(source), canvas_size=(64, 64)
    )
    render_pass.compile()
    render_pass.render(
        u_time=0.0,
        instances={
            "pos": np.zeros((4, 2), dtype="f4"),
            "radius": np.zeros(4, dtype="f8"),  # the mistake
        },
    )
    pixels = np.frombuffer(
        render_pass.canvas.fbo.read(components=4, dtype="f2"), dtype="f2"
    )
    assert float(pixels.max()) == 0.0, "an invalid population drew something"

    # The SAME pass with a valid population draws, so the silence above is about the
    # invalid case rather than about a pass that never draws at all.
    render_pass.render(
        u_time=0.0,
        instances={
            "pos": np.zeros((4, 2), dtype="f4"),
            "radius": np.full(4, 0.2, dtype="f4"),
        },
    )
    good = np.frombuffer(
        render_pass.canvas.fbo.read(components=4, dtype="f2"), dtype="f2"
    )
    assert float(good.max()) > 0.0


def test_a_dry_run_leaves_the_live_population_alone(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    """`dry_run` promises the live document is byte-identical afterward.

    Every uniform write routes through `values_sink` to keep that promise. A population
    is live state too, and writing it unconditionally replaced the real document's
    entities with the throwaway probe instance's.
    """
    growing = """
import numpy as np
from shaderbox.scripting import ScriptBehavior, ScriptContext


class Behavior(ScriptBehavior):
    def __init__(self) -> None:
        self.n = 0

    def update(self, context: ScriptContext) -> dict:
        self.n += 1
        return {"swarm": {"@instances": {
            "pos": np.full((self.n, 2), float(self.n), dtype="f4"),
            "radius": np.full(self.n, 0.1, dtype="f4"),
        }}}
"""
    document, engine = _document(tmp_path, gl_ctx, growing)
    engine.tick("doc", document, ScriptContext(t=0.0, dt=1 / 60, frame=0))
    before = document.passes["swarm"].pending_instances["pos"].copy()

    engine.dry_run("doc", document, (0.0, 0.5, 1.0), 60.0)

    after = document.passes["swarm"].pending_instances["pos"]
    # The probe's own instance counts up, so a leak shows as BOTH a different shape and
    # a different value -- the fixture is built where the two cannot coincide.
    assert after.shape == before.shape and float(after[0, 0]) == float(before[0, 0])


def test_a_bad_population_says_why_on_the_strip(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    """The message reaches the author, not a log file nobody opens.

    Validation happens in the engine rather than at draw time for exactly this reason:
    `Pass` can only log, and an author running the app from a launcher never sees it.
    """
    wrong_dtype = _SCRIPT.replace(
        'dtype="f4")\n        self.radius', 'dtype="f8")\n        self.radius'
    )
    document, engine = _document(tmp_path, gl_ctx, wrong_dtype)
    engine.tick("doc", document, ScriptContext(t=0.0, dt=1 / 60, frame=0))

    status = engine.script_status("doc")
    assert status is not None
    messages = " ".join(error.message for _, _, error in status.soft_errors)
    assert "pos" in messages and "f4" in messages
    assert document.passes["swarm"].pending_instances == {}


def test_a_pass_that_stops_returning_a_population_does_not_keep_the_last_one(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # The write is unconditional for exactly this reason. Making it conditional on the
    # pass appearing in the returned dict -- the natural-looking optimisation -- leaves
    # a pass rendering last frame's entities forever once the script stops sending them.
    alternating = """
import numpy as np
from shaderbox.scripting import ScriptBehavior, ScriptContext


class Behavior(ScriptBehavior):
    def __init__(self) -> None:
        self.frame = 0

    def update(self, context: ScriptContext) -> dict:
        self.frame += 1
        if self.frame % 2 == 1:
            return {"swarm": {"@instances": {
                "pos": np.zeros((2, 2), dtype="f4"),
                "radius": np.full(2, 0.2, dtype="f4"),
            }}}
        return {}
"""
    document, engine = _document(tmp_path, gl_ctx, alternating)
    engine.tick("doc", document, ScriptContext(t=0.0, dt=1 / 60, frame=0))
    assert document.passes["swarm"].pending_instances, "the odd frame must populate"
    engine.tick("doc", document, ScriptContext(t=1 / 60, dt=1 / 60, frame=1))
    assert document.passes["swarm"].pending_instances == {}, "the even frame must clear"
