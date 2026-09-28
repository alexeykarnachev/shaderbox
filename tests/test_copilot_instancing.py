"""Feature 103: the copilot learns instancing.

C1's two false verdicts, broken and watched to fail before believed: the shipped Entity Flock
(20000 entities, orbiting, `@instances` only -- no scalar uniform) must not read as "drives 0
uniforms", and adding one CONSTANT scalar beside the population must not read as "values
UNCHANGED across t (STATIC)" with no honest word about the population that IS moving.
"""

from pathlib import Path

import moderngl

from shaderbox.copilot.backend import (
    _motion_verdict,
    _population_facts_text,
    _population_motion,
)
from shaderbox.copilot.capabilities import PassView, ShaderView, WorkingSetView
from shaderbox.copilot.config import COPILOT_ENGINE
from shaderbox.document import Document
from shaderbox.engine_uniforms import ENGINE_DRIVEN_UNIFORMS
from shaderbox.instanced import MODE_UNIFORM
from shaderbox.scripting.engine import ScriptEngine

_COUNT = 20000

# A fragment shader shaped like the shipped Entity Flock's `swarm.frag.glsl`: a `flat in`
# population, discard outside the entity's own quad.
_FRAGMENT = """#version 460 core
in vec2 vs_quad;
flat in vec2 pos;
flat in float radius;
out vec4 frag_color;
void main() {
    if (length(vs_quad) > 1.0) discard;
    frag_color = vec4(1.0);
}
"""

# A fullscreen sibling with no entity fields -- the real contrast pair for the "both views"
# gate, differing from _FRAGMENT only in the property under test (instanced or not).
_FULLSCREEN_FRAGMENT = """#version 460 core
in vec2 vs_uv;
out vec4 frag_color;
void main() {
    frag_color = vec4(vs_uv, 0.0, 1.0);
}
"""

# The flock AS SHIPPED: 20000 entities orbiting via context.t, ONLY `@instances` -- no scalar
# uniform driven at all. This is exactly the C1 shape: probe.driven is empty.
_FLOCK_SCRIPT = """
import numpy as np
from shaderbox.scripting import ScriptBehavior, ScriptContext

COUNT = 20000


class Behavior(ScriptBehavior):
    def __init__(self) -> None:
        # A narrow ARC, not the full circle: the real flock's entities do not already span
        # the whole range at t=0, so a shift over t is visible as a range change rather than
        # hidden by a population that already covers the full extent at every sample.
        self.angle = np.linspace(0.0, 0.3, COUNT).astype("f4")
        self.radius = np.full(COUNT, 0.01, dtype="f4")

    def update(self, context: ScriptContext) -> dict:
        t = np.float32(context.t)
        a = self.angle + t
        pos = np.stack(
            [0.5 * np.cos(a), 0.5 * np.sin(a)], axis=1
        ).astype("f4")
        return {"swarm": {"@instances": {"pos": pos, "radius": self.radius}}}
"""

# The same flock PLUS one constant scalar uniform -- the second false-verdict case: a scalar
# that never varies is correctly STATIC, but the population beside it IS moving, and nothing
# in the reply may say the whole script is static.
_FLOCK_WITH_CONSTANT_SCRIPT = _FLOCK_SCRIPT.replace(
    'return {"swarm": {"@instances": {"pos": pos, "radius": self.radius}}}',
    'return {"swarm": {"@instances": {"pos": pos, "radius": self.radius}, '
    '"u_glow": 0.5}}',
)

_FRAGMENT_WITH_GLOW = _FRAGMENT.replace(
    "out vec4 frag_color;", "uniform float u_glow = 0.5;\nout vec4 frag_color;"
).replace("vec4(1.0)", "vec4(u_glow)")


def _flock_document(
    tmp_path: Path, gl_ctx: moderngl.Context, script: str, fragment: str = _FRAGMENT
) -> tuple[Document, ScriptEngine]:
    (tmp_path / "passes").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "passes" / "swarm.frag.glsl").write_text(fragment)
    (tmp_path / "scripts" / "script.py").write_text(script)
    (tmp_path / "graph.json").write_text(
        '{"version": 2, "output": "swarm", "passes": {"swarm": {}}}'
    )
    (tmp_path / "document.json").write_text('{"uniforms": {}}')
    document, _ = Document.load_from_dir(tmp_path, gl=gl_ctx, canvas_size=(64, 64))
    document.render()  # the first render is what compiles the pass
    engine = ScriptEngine()
    engine.reload("doc", tmp_path / "scripts")
    return document, engine


# ---- (a) sb_instanced is not script-drivable (D4a) ----


def test_sb_instanced_is_engine_driven() -> None:
    # Break by removing it from ENGINE_DRIVEN_UNIFORMS and watching a script "drive" it
    # cleanly instead of being silently overwritten every frame.
    assert MODE_UNIFORM in ENGINE_DRIVEN_UNIFORMS


# ---- gate: the shipped flock must not report "drives 0 uniforms" (C1, case 1) ----


def test_the_shipped_flock_does_not_report_drives_zero_uniforms(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    document, engine = _flock_document(tmp_path, gl_ctx, _FLOCK_SCRIPT)
    probe = engine.dry_run(
        "doc", document, COPILOT_ENGINE.motion_sample_times, COPILOT_ENGINE.motion_fps
    )
    # The premise this gate exists to catch: a pure-@instances script drives no SCALAR
    # uniform, so probe.driven really is empty -- that is not itself the bug.
    assert not probe.driven

    verdict = _motion_verdict(probe, "", COPILOT_ENGINE.motion_value_eps)
    assert "drives 0 uniforms" not in verdict, verdict

    facts = _population_facts_text(probe, COPILOT_ENGINE.motion_value_eps)
    # The count must APPEAR (not merely "some string is absent" -- a silence assertion
    # cannot tell you it missed).
    assert str(_COUNT) in facts, facts
    assert "swarm" in facts


# ---- gate: the flock + one constant scalar must not report a blanket STATIC (C1, case 2) ----


def test_the_flock_plus_a_constant_scalar_does_not_report_a_blanket_static(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    document, engine = _flock_document(
        tmp_path, gl_ctx, _FLOCK_WITH_CONSTANT_SCRIPT, _FRAGMENT_WITH_GLOW
    )
    probe = engine.dry_run(
        "doc", document, COPILOT_ENGINE.motion_sample_times, COPILOT_ENGINE.motion_fps
    )
    # u_glow really is constant, so probe.driven is non-empty and the SCALAR verdict is
    # legitimately STATIC -- that line is correct. What must not happen is the reply
    # containing ONLY that line with no honest word about the moving population.
    assert probe.driven

    scalar_verdict = _motion_verdict(probe, "", COPILOT_ENGINE.motion_value_eps)
    assert (
        "values UNCHANGED across t (STATIC)" in scalar_verdict
    )  # the scalar IS static

    population_facts = _population_facts_text(probe, COPILOT_ENGINE.motion_value_eps)
    assert population_facts, "the population's own facts must not be silent"
    assert str(_COUNT) in population_facts
    motion_lines = _population_motion(probe, COPILOT_ENGINE.motion_value_eps)
    assert any("ANIMATING" in line for line in motion_lines), motion_lines
    # The combined agent-facing text (both fields, as tools/script.py concatenates them)
    # must not read as a single blanket STATIC verdict.
    combined = f"{scalar_verdict}\n{population_facts}"
    assert "ANIMATING" in combined


# ---- gate: "not validated" is never a false zero (D2's conditional-validation case) ----


def test_a_population_reaching_an_uncompiled_pass_reports_not_validated(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # A pass whose fragment source fails to compile has no entity_fields -- engine.py's
    # "if not fields:" branch. The statistics sink must say "not validated", never a false
    # zero-entity report (103 D2).
    broken_fragment = "#version 460 core\nBROKEN SYNTAX HERE\n"
    document, engine = _flock_document(tmp_path, gl_ctx, _FLOCK_SCRIPT, broken_fragment)
    probe = engine.dry_run(
        "doc", document, COPILOT_ENGINE.motion_sample_times, COPILOT_ENGINE.motion_fps
    )
    facts = _population_facts_text(probe, COPILOT_ENGINE.motion_value_eps)
    assert facts, "a population reaching an uncompiled pass must still be reported"
    assert "not validated" in facts
    assert str(_COUNT) not in facts  # never a false count alongside "not validated"


# ---- gate: both views report entity fields for an instanced pass, none for a fullscreen one ----


def test_shader_view_carries_entity_fields_only_for_the_instanced_pass() -> None:
    instanced = ShaderView(
        document_id="d1",
        name="swarm",
        listing="",
        uniforms=[],
        errors=[],
        entity_fields=["pos vec2", "radius float"],
    )
    fullscreen = ShaderView(
        document_id="d2", name="plain", listing="", uniforms=[], errors=[]
    )
    assert instanced.entity_fields == ["pos vec2", "radius float"]
    assert fullscreen.entity_fields == []


def test_working_set_view_and_pass_view_carry_entity_fields_only_for_the_instanced_one() -> (
    None
):
    instanced_single = WorkingSetView(
        address="d1",
        name="swarm",
        listing="",
        is_current=True,
        is_lib=False,
        uniforms=[],
        errors=[],
        entity_fields=["pos vec2", "radius float"],
    )
    fullscreen_single = WorkingSetView(
        address="d2",
        name="plain",
        listing="",
        is_current=False,
        is_lib=False,
        uniforms=[],
        errors=[],
    )
    assert instanced_single.entity_fields == ["pos vec2", "radius float"]
    assert fullscreen_single.entity_fields == []

    instanced_pass = PassView(
        name="swarm",
        address="d1#swarm",
        listing="",
        uniforms=[],
        errors=[],
        is_output=True,
        entity_fields=["pos vec2"],
    )
    fullscreen_pass = PassView(
        name="plain",
        address="d1#plain",
        listing="",
        uniforms=[],
        errors=[],
        is_output=False,
    )
    assert instanced_pass.entity_fields == ["pos vec2"]
    assert fullscreen_pass.entity_fields == []


def test_backend_entity_field_rows_reads_a_real_compiled_pass(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # The real contrast pair: same helper, two passes differing ONLY in whether they declare
    # `flat in` fields.
    from shaderbox.copilot.backend import _entity_field_rows
    from shaderbox.core import Pass
    from shaderbox.shader_source import ShaderSource

    instanced_path = tmp_path / "swarm.frag.glsl"
    instanced_path.write_text(_FRAGMENT)
    instanced_pass = Pass(
        gl=gl_ctx, source=ShaderSource.load(instanced_path), canvas_size=(64, 64)
    )
    instanced_pass.compile()
    assert _entity_field_rows(instanced_pass) == ["pos vec2", "radius float"]

    fullscreen_path = tmp_path / "plain.frag.glsl"
    fullscreen_path.write_text(_FULLSCREEN_FRAGMENT)
    fullscreen_pass = Pass(
        gl=gl_ctx, source=ShaderSource.load(fullscreen_path), canvas_size=(64, 64)
    )
    fullscreen_pass.compile()
    assert _entity_field_rows(fullscreen_pass) == []
