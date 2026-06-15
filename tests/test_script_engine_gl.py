"""GL-integration test for the CPU-script engine (feature 041, redesigned by 048 to ONE node script):
a script drives a scalar + a vec2 (one stateful class, dict return), then renders on a real GL context,
asserting the computed values reach node.uniform_values, that the rendered pixel CHANGES between two t
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

from shaderbox.core import Node
from shaderbox.media import MediaDetails, ResolutionDetails
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
    ctx = None
    for kwargs in ({"backend": "egl"}, {}):
        try:
            ctx = moderngl.create_standalone_context(**kwargs)  # type: ignore[arg-type]
            break
        except Exception:
            continue
    if ctx is None:
        pytest.skip("no standalone GL context available")
    yield ctx
    ctx.release()


def _node(gl: moderngl.Context, src: str = _SRC) -> Node:
    node = Node(gl=gl)
    node.release_program(src)
    node.compile()
    node.render(u_time=0.0)  # warm-up so get_active_uniforms is populated
    return node


def _write_script(scripts_dir: Path, body: str) -> None:
    # The node script file (048): nodes/<id>/scripts/script.py — the only script on a node.
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / "script.py").write_text(body, encoding="utf-8")


def _pixel(node: Node) -> tuple[int, int, int, int]:
    data = node.canvas.texture.read()
    return tuple(data[:4])  # type: ignore[return-value]


# A script driving a t-pure scalar + a constant vec2 from ONE instance.
_WAVE_SCRIPT = (
    "import math\n"
    "class Behavior(ScriptBehavior):\n"
    "    def update(self, ctx: Ctx) -> dict:\n"
    "        return {\n"
    "            'u_wave': 0.5 + 0.5 * math.sin(ctx.t),\n"
    "            'u_offset': Vec2(0.25, 0.75),\n"
    "        }\n"
)
# A stateful ramp driving BOTH u_wave (the integrator) and u_offset from ONE instance.
_RAMP_SCRIPT = (
    "class Behavior(ScriptBehavior):\n"
    "    def __init__(self) -> None:\n"
    "        self.v = 0.0\n"
    "    def update(self, ctx: Ctx) -> dict:\n"
    "        self.v += ctx.dt\n"
    "        return {'u_wave': self.v % 1.0, 'u_offset': Vec2(0.25, 0.75)}\n"
)


def test_script_value_reaches_gpu(gl_ctx: moderngl.Context, tmp_path: Path) -> None:
    # One script drives a float + a vec2: both reach node.uniform_values AND the scripted u_wave
    # changes the rendered pixel between two t values. Falsifier: px_a[0] == px_b[0] (u_wave never
    # reached the GPU) or the uniforms aren't written.
    scripts_dir = tmp_path / "scripts"
    _write_script(scripts_dir, _WAVE_SCRIPT)
    node = _node(gl_ctx)
    eng = ScriptEngine()
    eng.reload("n", scripts_dir, node)

    eng.tick("n", node, EngineContext(t=0.0, dt=0.0, frame=0))
    assert abs(node.uniform_values["u_wave"] - 0.5) < 1e-6
    assert node.uniform_values["u_offset"] == (0.25, 0.75)
    node.render(u_time=0.0)
    px_a = _pixel(node)

    eng.tick(
        "n", node, EngineContext(t=1.5708, dt=0.0, frame=1)
    )  # sin(pi/2)=1 -> u_wave≈1.0
    node.render(u_time=1.5708)
    px_b = _pixel(node)
    assert px_a[0] != px_b[0], "scripted u_wave did not reach the GPU"

    with contextlib.suppress(Exception):
        node.release()


_UTIME_SRC = """#version 460 core
in vec2 vs_uv;
uniform float u_time;
out vec4 fs_color;
void main() {
    fs_color = vec4(0.5 + 0.5 * sin(u_time), 0.0, 0.0, 1.0);
}
"""


def test_render_clock_honors_passed_u_time(gl_ctx: moderngl.Context) -> None:
    # The 043 polish render-clock invariant (the consumer side of _render_facts_for(node, t=mid[0])):
    # a u_time-reading shader rendered at an EXPLICIT t produces the frame for THAT t, not wall-clock 0.
    # The corroborating script-probe render passes t=mid[0]; if Node.render ignored it (or the caller
    # rendered at wall-clock 0 while injecting t=0.5 values), the agent would get a frame that never
    # existed. Falsifier: px at t=pi/2 == px at t=0 -> u_time didn't reach the GPU at the passed t.
    node = _node(gl_ctx, _UTIME_SRC)
    node.render(u_time=0.0)
    px_t0 = _pixel(node)
    node.render(u_time=1.5708)  # sin(pi/2)=1 -> red channel ~255
    px_half = _pixel(node)
    assert px_t0[0] != px_half[0], "Node.render ignored the passed u_time"
    assert px_half[0] > px_t0[0]  # brighter at the later phase
    with contextlib.suppress(Exception):
        node.release()


def test_script_shape_mismatch_freezes_and_records(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # A vec3 into a float uniform: the per-key coercion mismatch freezes the uniform at its seeded
    # value + records a (node, name) runtime error — never corrupts the GPU write. Falsifier: u_wave
    # changes off the seed, or no error recorded.
    scripts_dir = tmp_path / "scripts"
    _write_script(
        scripts_dir,
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx: Ctx) -> dict:\n"
        "        return {'u_wave': Vec3(0.1, 0.2, 0.3)}\n",  # vec3 into a float uniform
    )
    node = _node(gl_ctx)
    node.seed_uniform_values()
    seeded = node.uniform_values.get("u_wave")
    eng = ScriptEngine()
    eng.reload("n", scripts_dir, node)
    eng.tick("n", node, EngineContext(t=0.0, dt=0.0, frame=0))
    assert node.uniform_values.get("u_wave") == seeded  # frozen, not corrupted
    assert eng.errors[("n", "u_wave")].kind == "runtime"

    with contextlib.suppress(Exception):
        node.release()


def test_fresh_export_instance_renders_clean(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # Warm a stateful ramp on the LIVE instance, then render via a FRESH export instance —
    # the export's frame-0 pixel must match a cold-start render, NOT the warmed live value.
    # Falsifier: px_export != px_cold (the export inherited live state).
    scripts_dir = tmp_path / "scripts"
    _write_script(scripts_dir, _RAMP_SCRIPT)
    node = _node(gl_ctx)
    eng = ScriptEngine()
    eng.reload("n", scripts_dir, node)

    # Cold-start reference: a fresh instance, one tick at frame 0.
    cold = eng.fresh_behavior_for("n")
    assert cold is not None
    eng.tick_export("n", node, EngineContext(t=0.0, dt=1 / 60, frame=0), cold)
    node.render(u_time=0.0)
    px_cold = _pixel(node)

    # Warm the LIVE instance well past the ramp's wrap.
    for i in range(120):
        eng.tick("n", node, EngineContext(t=i / 60, dt=1 / 60, frame=i))
    live_wave = node.uniform_values["u_wave"]

    # A fresh export instance must reproduce the cold value, not the warmed one.
    fresh = eng.fresh_behavior_for("n")
    assert fresh is not None
    eng.tick_export("n", node, EngineContext(t=0.0, dt=1 / 60, frame=0), fresh)
    node.render(u_time=0.0)
    px_export = _pixel(node)
    assert px_export == px_cold
    assert node.uniform_values["u_wave"] != live_wave

    with contextlib.suppress(Exception):
        node.release()


def test_render_media_auto_enters_export_isolation(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # The structural guarantee (feature 041): render_media ITSELF enters node.export_isolation, so a
    # caller cannot forget to isolate. Inject an isolation factory + a LIVE hook, warm the live
    # integrator, then call the real export entry (render_media) and assert the factory was entered
    # exactly once AND the export pre-render fired a FRESH instance (not the warmed live one).
    # Falsifier: entered != 1, or the exported frame-0 value equals the warmed live value.
    scripts_dir = tmp_path / "scripts"
    _write_script(scripts_dir, _RAMP_SCRIPT)
    node = _node(gl_ctx)
    eng = ScriptEngine()
    eng.reload("n", scripts_dir, node)

    entered = {"count": 0}

    @contextlib.contextmanager
    def _isolation() -> Iterator[None]:
        entered["count"] += 1
        live_hook = node.on_pre_render
        fresh = eng.fresh_behavior_for("n")
        assert fresh is not None
        node.on_pre_render = lambda t, dt, f: eng.tick_export(
            "n", node, EngineContext(t=t, dt=dt, frame=f), fresh
        )
        try:
            yield
        finally:
            node.on_pre_render = live_hook

    node.on_pre_render = lambda t, dt, f: eng.tick(
        "n", node, EngineContext(t=t, dt=dt, frame=f)
    )
    node.export_isolation = _isolation

    # Warm the live instance well past the ramp wrap.
    for i in range(120):
        eng.tick("n", node, EngineContext(t=i / 60, dt=1 / 60, frame=i))
    live_wave = node.uniform_values["u_wave"]

    out = tmp_path / "out.png"
    cw, ch = node.canvas.texture.size
    details = MediaDetails(
        is_video=False, resolution_details=ResolutionDetails(width=cw, height=ch)
    )
    details.file_details.path = str(out)
    node.render_media(details)
    assert entered["count"] == 1, "render_media did not enter export_isolation"
    assert node.uniform_values["u_wave"] != live_wave
    assert out.exists()

    with contextlib.suppress(Exception):
        node.release()


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
        "        return {'u_i': 2.7, 'u_count': 4.2, 'u_iv': Vec2(1.6, 2.4)}\n",
    )
    node = Node(gl=gl_ctx)
    node.release_program(_INT_SRC)
    node.compile()
    node.render(u_time=0.0)
    eng = ScriptEngine()
    eng.reload("n", scripts_dir, node)
    eng.tick("n", node, EngineContext(t=0.0, dt=0.0, frame=0))
    node.render(u_time=0.0)
    # If a write raised, render's except pops the value — these reads would be missing.
    assert node.uniform_values["u_i"] == 3
    assert node.uniform_values["u_count"] == 4
    assert node.uniform_values["u_iv"] == (2, 2)  # round(1.6)=2, round(2.4)=2
    assert not any(k[0] == "n" for k in eng.errors)

    with contextlib.suppress(Exception):
        node.release()


def test_script_drives_two_uniforms_to_gpu_and_export_clean(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # One node script drives a float + a vec2 from a single stateful instance: both reach the GPU,
    # the scripted u_wave changes the rendered pixel, and a FRESH export instance renders cold (the
    # headline 048 goal — one script per node, export-isolated). Falsifier: px_warm == px_cold (the
    # value never reached the GPU) or the export inherited live state.
    scripts_dir = tmp_path / "scripts"
    _write_script(scripts_dir, _RAMP_SCRIPT)
    node = _node(gl_ctx)
    eng = ScriptEngine()
    eng.reload("n", scripts_dir, node)

    # Cold-start reference: a fresh instance, one tick at frame 0.
    cold = eng.fresh_behavior_for("n")
    assert cold is not None
    eng.tick_export("n", node, EngineContext(t=0.0, dt=1 / 60, frame=0), cold)
    assert node.uniform_values["u_offset"] == (
        0.25,
        0.75,
    )  # both driven from one script
    node.render(u_time=0.0)
    px_cold = _pixel(node)

    # Warm the LIVE instance past the ramp wrap.
    for i in range(120):
        eng.tick("n", node, EngineContext(t=i / 60, dt=1 / 60, frame=i))
    live_wave = node.uniform_values["u_wave"]
    node.render(u_time=2.0)
    px_warm = _pixel(node)
    assert px_warm[0] != px_cold[0], "scripted u_wave did not reach the GPU"

    # A fresh export instance reproduces the cold pixel, NOT the warmed value.
    fresh = eng.fresh_behavior_for("n")
    assert fresh is not None
    eng.tick_export("n", node, EngineContext(t=0.0, dt=1 / 60, frame=0), fresh)
    node.render(u_time=0.0)
    assert _pixel(node) == px_cold
    assert node.uniform_values["u_wave"] != live_wave

    with contextlib.suppress(Exception):
        node.release()
