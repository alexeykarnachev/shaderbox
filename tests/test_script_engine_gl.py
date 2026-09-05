"""GL-integration test for the CPU-script engine (feature 041, redesigned by 048 to ONE document script):
a script drives a scalar + a vec2 (one stateful class, dict return), then renders on a real GL context,
asserting the computed values reach document.uniform_values, that the rendered pixel CHANGES between two t
values (the value reaches the GPU), that a shape-mismatch freezes + records a ScriptError, that int
uniforms reach the GPU (not popped on a failed write), and that a FRESH export instance renders cold
(export-isolation, including render_media auto-entering it).

Needs a real GL context. On the display-less dev box use the EGL backend + the MESA version
overrides (set at process top, read at context creation); skips cleanly if no context is available.
"""

import contextlib
import os
from collections.abc import Iterator
from pathlib import Path

import moderngl
import pytest

from shaderbox.core import Pass
from shaderbox.document import Document
from shaderbox.media import MediaDetails, ResolutionDetails, texture_to_rgba8
from shaderbox.pass_graph import PassEntry, PassGraph
from shaderbox.scripting import EngineContext, ScriptEngine

_SRC = """#version 460 core
in vec2 vs_uv;
uniform float u_wave;
uniform vec2 u_offset;
out vec4 fs_color;
void main() {
    fs_color = vec4(u_wave, u_offset.x, u_offset.y, 1.0);
}
"""


@pytest.fixture(scope="module")
def gl_ctx() -> Iterator[moderngl.Context]:
    # The MESA overrides give the display-less box's V3D driver #version 460 — read at context
    # creation, so set them here before create_standalone_context (no effect on a desktop driver).
    os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "4.6")
    os.environ.setdefault("MESA_GLSL_VERSION_OVERRIDE", "460")
    # Default-backend like every other GL module's fixture — an EXPLICIT backend="egl" context
    # released here poisons the process's EGL display and the NEXT module's first program
    # compile segfaults (module-order-only; one context recipe per process is the rule).
    try:
        ctx = moderngl.create_standalone_context()
    except Exception as e:
        pytest.skip(f"no standalone GL context available: {e}")
    yield ctx
    ctx.release()


def _document(gl: moderngl.Context, src: str = _SRC) -> Document:
    document = Document(gl=gl)
    document.render_pass.release_program(src)
    document.render_pass.compile()
    document.render(u_time=0.0)  # warm-up so get_active_uniforms is populated
    return document


def _write_script(scripts_dir: Path, body: str) -> None:
    # The document script file (048): documents/<id>/scripts/script.py — the only script on a document.
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / "script.py").write_text(body, encoding="utf-8")


def _pixel(document: Document) -> tuple[int, int, int, int]:
    data = document.render_pass.canvas.texture.read()
    return tuple(data[:4])


# A script driving a t-pure scalar + a constant vec2 from ONE instance.
_WAVE_SCRIPT = (
    "import math\n"
    "class Behavior(ScriptBehavior):\n"
    "    def update(self, ctx: Ctx) -> dict:\n"
    "        return {\n"
    "            'u_wave': 0.5 + 0.5 * math.sin(ctx.t),\n"
    "            'u_offset': [0.25, 0.75],\n"
    "        }\n"
)
# A stateful ramp driving BOTH u_wave (the integrator) and u_offset from ONE instance.
_RAMP_SCRIPT = (
    "class Behavior(ScriptBehavior):\n"
    "    def __init__(self) -> None:\n"
    "        self.v = 0.0\n"
    "    def update(self, ctx: Ctx) -> dict:\n"
    "        self.v += ctx.dt\n"
    "        return {'u_wave': self.v % 1.0, 'u_offset': [0.25, 0.75]}\n"
)


def test_script_value_reaches_gpu(gl_ctx: moderngl.Context, tmp_path: Path) -> None:
    # One script drives a float + a vec2: both reach document.uniform_values AND the scripted u_wave
    # changes the rendered pixel between two t values. Falsifier: px_a[0] == px_b[0] (u_wave never
    # reached the GPU) or the uniforms aren't written.
    scripts_dir = tmp_path / "scripts"
    _write_script(scripts_dir, _WAVE_SCRIPT)
    document = _document(gl_ctx)
    eng = ScriptEngine()
    eng.reload("n", scripts_dir, document)

    eng.tick("n", document, EngineContext(t=0.0, dt=0.0, frame=0))
    assert abs(document.render_pass.uniform_values["u_wave"] - 0.5) < 1e-6
    assert document.render_pass.uniform_values["u_offset"] == (0.25, 0.75)
    document.render(u_time=0.0)
    px_a = _pixel(document)

    eng.tick(
        "n", document, EngineContext(t=1.5708, dt=0.0, frame=1)
    )  # sin(pi/2)=1 -> u_wave≈1.0
    document.render(u_time=1.5708)
    px_b = _pixel(document)
    assert px_a[0] != px_b[0], "scripted u_wave did not reach the GPU"

    with contextlib.suppress(Exception):
        document.release()


_UTIME_SRC = """#version 460 core
in vec2 vs_uv;
uniform float u_time;
out vec4 fs_color;
void main() {
    fs_color = vec4(0.5 + 0.5 * sin(u_time), 0.0, 0.0, 1.0);
}
"""


def test_render_clock_honors_passed_u_time(gl_ctx: moderngl.Context) -> None:
    # The 043 polish render-clock invariant (the consumer side of _render_facts_for(document, t=mid[0])):
    # a u_time-reading shader rendered at an EXPLICIT t produces the frame for THAT t, not wall-clock 0.
    # The corroborating script-probe render passes t=mid[0]; if Document.render ignored it (or the caller
    # rendered at wall-clock 0 while injecting t=0.5 values), the agent would get a frame that never
    # existed. Falsifier: px at t=pi/2 == px at t=0 -> u_time didn't reach the GPU at the passed t.
    document = _document(gl_ctx, _UTIME_SRC)
    document.render(u_time=0.0)
    px_t0 = _pixel(document)
    document.render(u_time=1.5708)  # sin(pi/2)=1 -> red channel ~255
    px_half = _pixel(document)
    assert px_t0[0] != px_half[0], "Document.render ignored the passed u_time"
    assert px_half[0] > px_t0[0]  # brighter at the later phase
    with contextlib.suppress(Exception):
        document.release()


def test_script_shape_mismatch_freezes_and_records(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # A vec3 into a float uniform: the per-key coercion mismatch freezes the uniform at its seeded
    # value + records a (document, name) runtime error — never corrupts the GPU write. Falsifier: u_wave
    # changes off the seed, or no error recorded.
    scripts_dir = tmp_path / "scripts"
    _write_script(
        scripts_dir,
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        return {'u_wave': [0.1, 0.2, 0.3]}\n",  # vec3 into a float uniform
    )
    document = _document(gl_ctx)
    document.render_pass.seed_uniform_values()
    seeded = document.render_pass.uniform_values.get("u_wave")
    eng = ScriptEngine()
    eng.reload("n", scripts_dir, document)
    eng.tick("n", document, EngineContext(t=0.0, dt=0.0, frame=0))
    assert (
        document.render_pass.uniform_values.get("u_wave") == seeded
    )  # frozen, not corrupted
    pass_name = next(iter(document.passes))
    assert eng.errors[("n", pass_name, "u_wave")].kind == "runtime"

    with contextlib.suppress(Exception):
        document.release()


def test_render_media_auto_enters_export_isolation(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # The structural guarantee (feature 041): render_media ITSELF enters document.export_isolation, so a
    # caller cannot forget to isolate. Inject an isolation factory + a LIVE hook, warm the live
    # integrator, then call the real export entry (render_media) and assert the factory was entered
    # exactly once AND the export pre-render fired a FRESH instance (not the warmed live one).
    # Falsifier: entered != 1, or the exported frame-0 value equals the warmed live value.
    scripts_dir = tmp_path / "scripts"
    _write_script(scripts_dir, _RAMP_SCRIPT)
    document = _document(gl_ctx)
    eng = ScriptEngine()
    eng.reload("n", scripts_dir, document)

    entered = {"count": 0}

    @contextlib.contextmanager
    def _isolation() -> Iterator[None]:
        entered["count"] += 1
        live_hook = document.on_pre_render
        fresh = eng.fresh_behavior_for("n")
        assert fresh is not None
        document.on_pre_render = lambda t, dt, f: eng.tick_export(
            "n", document, EngineContext(t=t, dt=dt, frame=f), fresh
        )
        try:
            yield
        finally:
            document.on_pre_render = live_hook

    document.on_pre_render = lambda t, dt, f: eng.tick(
        "n", document, EngineContext(t=t, dt=dt, frame=f)
    )
    document.export_isolation = _isolation

    # Warm the live instance well past the ramp wrap.
    for i in range(120):
        eng.tick("n", document, EngineContext(t=i / 60, dt=1 / 60, frame=i))
    live_wave = document.render_pass.uniform_values["u_wave"]

    out = tmp_path / "out.png"
    cw, ch = document.render_pass.canvas.texture.size
    details = MediaDetails(
        is_video=False, resolution_details=ResolutionDetails(width=cw, height=ch)
    )
    details.file_details.path = str(out)
    document.render_media(details)
    assert entered["count"] == 1, "render_media did not enter export_isolation"
    assert document.render_pass.uniform_values["u_wave"] != live_wave
    assert out.exists()

    with contextlib.suppress(Exception):
        document.release()


_INT_SRC = """#version 460 core
in vec2 vs_uv;
uniform int u_i;
uniform uint u_count;
uniform ivec2 u_iv;
out vec4 fs_color;
void main() {
    fs_color = vec4(float(u_i) + float(u_count) + float(u_iv.x) + float(u_iv.y));
}
"""


def test_script_int_uniforms_reach_gpu_not_popped(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # The review-swarm bug: a script returning a float for an int/uint/ivec uniform passed coercion
    # but moderngl raised on write, render swallowed it, and uniform_values.pop'd the value EVERY
    # frame — silently. A naive "didn't raise" check passes while broken; assert the value is RETAINED
    # in uniform_values after render() (NOT popped). Falsifier: a value missing post-render.
    scripts_dir = tmp_path / "scripts"
    _write_script(
        scripts_dir,
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        return {'u_i': 2.7, 'u_count': 4.2, 'u_iv': [1.6, 2.4]}\n",
    )
    document = Document(gl=gl_ctx)
    document.render_pass.release_program(_INT_SRC)
    document.render_pass.compile()
    document.render(u_time=0.0)
    eng = ScriptEngine()
    eng.reload("n", scripts_dir, document)
    eng.tick("n", document, EngineContext(t=0.0, dt=0.0, frame=0))
    document.render(u_time=0.0)
    # If a write raised, render's except pops the value — these reads would be missing.
    assert document.render_pass.uniform_values["u_i"] == 3
    assert document.render_pass.uniform_values["u_count"] == 4
    assert document.render_pass.uniform_values["u_iv"] == (
        2,
        2,
    )  # round(1.6)=2, round(2.4)=2
    assert not any(k[0] == "n" for k in eng.errors)

    with contextlib.suppress(Exception):
        document.release()


def test_script_drives_two_uniforms_to_gpu_and_export_clean(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # One document script drives a float + a vec2 from a single stateful instance: both reach the GPU,
    # the scripted u_wave changes the rendered pixel, and a FRESH export instance renders cold (the
    # headline 048 goal — one script per document, export-isolated). Falsifier: px_warm == px_cold (the
    # value never reached the GPU) or the export inherited live state.
    scripts_dir = tmp_path / "scripts"
    _write_script(scripts_dir, _RAMP_SCRIPT)
    document = _document(gl_ctx)
    eng = ScriptEngine()
    eng.reload("n", scripts_dir, document)

    # Cold-start reference: a fresh instance, one tick at frame 0.
    cold = eng.fresh_behavior_for("n")
    assert cold is not None
    eng.tick_export("n", document, EngineContext(t=0.0, dt=1 / 60, frame=0), cold)
    assert document.render_pass.uniform_values["u_offset"] == (
        0.25,
        0.75,
    )  # both driven from one script
    document.render(u_time=0.0)
    px_cold = _pixel(document)

    # Warm the LIVE instance past the ramp wrap.
    for i in range(120):
        eng.tick("n", document, EngineContext(t=i / 60, dt=1 / 60, frame=i))
    live_wave = document.render_pass.uniform_values["u_wave"]
    document.render(u_time=2.0)
    px_warm = _pixel(document)
    assert px_warm[0] != px_cold[0], "scripted u_wave did not reach the GPU"

    # A fresh export instance reproduces the cold pixel, NOT the warmed value.
    fresh = eng.fresh_behavior_for("n")
    assert fresh is not None
    eng.tick_export("n", document, EngineContext(t=0.0, dt=1 / 60, frame=0), fresh)
    document.render(u_time=0.0)
    assert _pixel(document) == px_cold
    assert document.render_pass.uniform_values["u_wave"] != live_wave

    with contextlib.suppress(Exception):
        document.release()


_SEED_SRC = """#version 460 core
in vec2 vs_uv;
out vec4 fs_color;
uniform float u_wave;
void main() { fs_color = vec4(u_wave, 0.0, 0.0, 1.0); }
"""
_OUT_SRC = """#version 460 core
in vec2 vs_uv;
out vec4 fs_color;
uniform float u_wave;
uniform sampler2D u_seed;
void main() { fs_color = vec4(u_wave, texture(u_seed, vs_uv).r, 0.0, 1.0); }
"""


def test_a_broadcast_reaches_both_passes_on_the_gpu(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # The whole point of 069 W-G: a bare key drives EVERY pass declaring the uniform, and the write
    # reaches the GPU on the NON-output pass too. Falsifier: route to the output pass only — the
    # `seed` pixel does not change.
    scripts_dir = tmp_path / "scripts"
    _write_script(
        scripts_dir,
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        return {'u_wave': ctx.t}\n",
    )
    document = Document(gl=gl_ctx, canvas_size=(8, 8))
    for render_pass in document.passes.values():
        render_pass.release()
    document.passes = {}
    # `out` READS `seed`, so `seed` is on the output chain and actually renders — a pass nothing
    # consumes is never drawn, which would make the assertion below unfalsifiable.
    graph = PassGraph(
        output="out",
        passes={"seed": PassEntry(), "out": PassEntry()},
    )
    for name, src in (("seed", _SEED_SRC), ("out", _OUT_SRC)):
        # The pass's target comes from its OWN graph entry, exactly as the loader builds it. A
        # bare `Pass(...)` takes Canvas's 8-bit default while the entry says f2, and a float target
        # read as uint8 truncates to a plausible wrong value — the trap `_red_of` names in
        # test_document_graph.py.
        render_pass = Pass(
            gl=gl_ctx, canvas_size=(8, 8), target=graph.passes[name].target
        )
        render_pass.release_program(src)
        render_pass.compile()
        assert render_pass.compile_unit.errors == []
        document.passes[name] = render_pass
    document.graph = graph

    eng = ScriptEngine()
    eng.reload("n", scripts_dir, document)

    # An ABSOLUTE read per t, never a diff between two: an unrendered `seed` canvas reads 0, which
    # a "these two differ" assertion would satisfy for the wrong reason at t=0. Both sample times
    # are non-zero so every expected pixel is distinguishable from black.
    for t, expected in ((0.25, 64), (1.0, 255)):
        eng.tick("n", document, EngineContext(t=t, dt=1.0, frame=int(t * 4)))
        document.begin_frame(int(t * 4))
        document.render(u_time=t)
        # Read back only once the GPU has actually finished: llvmpipe under suite-wide memory
        # pressure otherwise hands back the PREVIOUS frame's mapping, which reads as this test
        # failing intermittently with exactly the prior sample's value.
        gl_ctx.finish()
        seed_px = int(texture_to_rgba8(document.passes["seed"].canvas.texture)[0][0][0])
        assert abs(seed_px - expected) <= 2, (
            f"the broadcast did not reach the NON-output pass at t={t} "
            f"(read {seed_px}, expected ~{expected})"
        )
        assert document.passes["seed"].uniform_values["u_wave"] == t
        assert document.passes["out"].uniform_values["u_wave"] == t

    with contextlib.suppress(Exception):
        document.release()
