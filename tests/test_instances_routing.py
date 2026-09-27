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

from shaderbox.document import Document
from shaderbox.instanced import validate_population
from shaderbox.intel.glsl import entity_fields
from shaderbox.scripting.context import ScriptContext
from shaderbox.scripting.engine import ScriptEngine

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
