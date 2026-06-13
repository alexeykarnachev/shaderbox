"""CPU-script engine (feature 040) — pure, no GL. A SimpleNamespace stands in for
moderngl.Uniform (the coercion/shape logic is GL-free; the GL write reaches the GPU is
verified in test_script_engine_gl.py). Covers: out.set shapes, the no-set / last-wins /
freeze-on-error contract, compile + runtime + shape errors as data, the (path, mtime) cache,
and the scoped-determinism rule (ctx.t-pure same, integrator diverges by design).
"""

import time
import types
from pathlib import Path

from shaderbox.scripting import EngineContext, ScriptEngine, UniformOut

_GL_FLOAT = 0x1406


def _u(
    name: str, dim: int = 1, n: int = 1, gl_type: int = _GL_FLOAT
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        name=name, dimension=dim, array_length=n, gl_type=gl_type, value=0.0
    )


class _FakeNode:
    # The EngineNode slice: a uniform_values dict + get_active_uniforms().
    def __init__(self, uniforms: list[types.SimpleNamespace]) -> None:
        self.uniform_values: dict[str, object] = {}
        self._uniforms = uniforms

    def get_active_uniforms(self) -> list[types.SimpleNamespace]:
        return self._uniforms


def _write(tmp: Path, name: str, body: str) -> Path:
    scripts_dir = tmp / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    path = scripts_dir / f"{name}.py"
    path.write_text(body, encoding="utf-8")
    return path


def _ctx(t: float, dt: float = 1 / 60, frame: int = 0) -> EngineContext:
    return EngineContext(t=t, dt=dt, frame=frame)


# ---- UniformOut shapes (direct) ----


def test_out_set_scalar() -> None:
    out = UniformOut(_u("u_x"))
    out.set(0.5)
    assert out.was_set and out.value == 0.5 and out.error is None


def test_out_set_vec_components() -> None:
    out = UniformOut(_u("u_off", dim=2))
    out.set(0.3, 0.7)
    assert out.value == (0.3, 0.7) and out.error is None


def test_out_set_vec_sequence() -> None:
    out = UniformOut(_u("u_off", dim=2))
    out.set((0.3, 0.7))
    assert out.value == (0.3, 0.7)


def test_out_set_shape_mismatch_records_error() -> None:
    out = UniformOut(_u("u_off", dim=3))
    out.set(0.5)  # scalar into a vec3
    assert not out.was_set and out.error is not None and out.error.kind == "runtime"


# ---- engine tick: bodies ----


def test_scalar_body(tmp_path: Path) -> None:
    _write(tmp_path, "u_wave", "out.set(0.5 + 0.3 * sin(ctx.t))")
    node = _FakeNode([_u("u_wave")])
    eng = ScriptEngine()
    eng.reload("n0", tmp_path / "scripts", node)
    eng.tick("n0", node, _ctx(0.0))
    assert abs(node.uniform_values["u_wave"] - 0.5) < 1e-9


def test_temps_then_set(tmp_path: Path) -> None:
    _write(tmp_path, "u_off", "r = 0.4\nout.set(r * cos(ctx.t), r * sin(ctx.t))")
    node = _FakeNode([_u("u_off", dim=2)])
    eng = ScriptEngine()
    eng.reload("n0", tmp_path / "scripts", node)
    eng.tick("n0", node, _ctx(0.0))
    assert node.uniform_values["u_off"] == (0.4, 0.0)


def test_branching_body(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "u_color",
        "if ctx.t > 1.0:\n    out.set(1.0, 0.0, 0.0)\nelse:\n    out.set(0.0, 0.0, 1.0)",
    )
    node = _FakeNode([_u("u_color", dim=3)])
    eng = ScriptEngine()
    eng.reload("n0", tmp_path / "scripts", node)
    eng.tick("n0", node, _ctx(0.0))
    assert node.uniform_values["u_color"] == (0.0, 0.0, 1.0)
    eng.tick("n0", node, _ctx(2.0))
    assert node.uniform_values["u_color"] == (1.0, 0.0, 0.0)


def test_no_set_holds_last_good(tmp_path: Path) -> None:
    # A body that conditionally sets: first tick sets, second tick is a no-op -> holds.
    _write(tmp_path, "u_x", "if ctx.frame == 0:\n    out.set(0.9)")
    node = _FakeNode([_u("u_x")])
    eng = ScriptEngine()
    eng.reload("n0", tmp_path / "scripts", node)
    eng.tick("n0", node, _ctx(0.0, frame=0))
    assert node.uniform_values["u_x"] == 0.9
    eng.tick("n0", node, _ctx(0.1, frame=1))  # no out.set this tick
    assert node.uniform_values["u_x"] == 0.9  # frozen, not cleared
    assert ("n0", "u_x") not in eng.errors  # a no-op is NOT an error


def test_multiple_set_last_wins(tmp_path: Path) -> None:
    _write(tmp_path, "u_x", "out.set(0.1)\nout.set(0.2)")
    node = _FakeNode([_u("u_x")])
    eng = ScriptEngine()
    eng.reload("n0", tmp_path / "scripts", node)
    eng.tick("n0", node, _ctx(0.0))
    assert node.uniform_values["u_x"] == 0.2


# ---- errors as data ----


def test_syntax_error_is_compile_error_at_user_line(tmp_path: Path) -> None:
    _write(tmp_path, "u_x", "out.set(0.5 +")  # incomplete expression
    node = _FakeNode([_u("u_x")])
    eng = ScriptEngine()
    eng.reload("n0", tmp_path / "scripts", node)
    err = eng.errors[("n0", "u_x")]
    assert err.kind == "compile" and err.line == 1


def test_runtime_error_freezes_last_good(tmp_path: Path) -> None:
    # First tick computes a value; then break it to a runtime error -> freezes at last-good.
    path = _write(tmp_path, "u_x", "out.set(0.7)")
    node = _FakeNode([_u("u_x")])
    eng = ScriptEngine()
    eng.reload("n0", tmp_path / "scripts", node)
    eng.tick("n0", node, _ctx(0.0))
    assert node.uniform_values["u_x"] == 0.7

    time.sleep(0.01)
    path.write_text(
        "out.set(1.0 / 0.0)", encoding="utf-8"
    )  # ZeroDivisionError at runtime
    eng.reload("n0", tmp_path / "scripts", node)
    eng.tick("n0", node, _ctx(0.1))
    assert node.uniform_values["u_x"] == 0.7  # frozen at last-good
    assert eng.errors[("n0", "u_x")].kind == "runtime"


def test_shape_mismatch_freezes_and_records(tmp_path: Path) -> None:
    _write(tmp_path, "u_x", "out.set(1.0, 2.0, 3.0)")  # vec3 into a scalar uniform
    node = _FakeNode([_u("u_x")])
    node.uniform_values["u_x"] = 0.0  # node-intrinsic default seeded
    eng = ScriptEngine()
    eng.reload("n0", tmp_path / "scripts", node)
    eng.tick("n0", node, _ctx(0.0))
    assert node.uniform_values["u_x"] == 0.0  # frozen
    assert eng.errors[("n0", "u_x")].kind == "runtime"


def test_orphan_script_warns_and_is_ignored(tmp_path: Path) -> None:
    _write(tmp_path, "u_ghost", "out.set(0.5)")  # no such uniform on the node
    node = _FakeNode([_u("u_x")])
    eng = ScriptEngine()
    eng.reload("n0", tmp_path / "scripts", node)
    assert "u_ghost" not in eng.script_driven_uniforms("n0")


# ---- scoped determinism (decision 1) ----


def test_t_pure_script_is_deterministic(tmp_path: Path) -> None:
    # Same ctx.t -> same value, regardless of dt path. This is the determinism guarantee.
    _write(tmp_path, "u_x", "out.set(sin(ctx.t))")
    node = _FakeNode([_u("u_x")])
    eng = ScriptEngine()
    eng.reload("n0", tmp_path / "scripts", node)
    eng.tick("n0", node, _ctx(1.234, dt=1 / 30))
    a = node.uniform_values["u_x"]
    eng.tick("n0", node, _ctx(1.234, dt=1 / 120))  # different dt, same t
    assert node.uniform_values["u_x"] == a


def test_integrator_diverges_by_design(tmp_path: Path) -> None:
    # A dt-reading integrator is path-dependent: a NONLINEAR step (the new value feeds the next
    # rate) makes the SAME elapsed time reached via different dt yield different values. This is
    # the live(variable dt) vs export(1/fps) divergence — documented as expected (decision 1),
    # NOT a determinism violation. The determinism guarantee is scoped to ctx.t-pure scripts.
    _write(
        tmp_path,
        "u_x",
        "prev = ctx.uniforms.get('u_x', 1.0)\nout.set(prev + prev * ctx.dt)",
    )
    node_a = _FakeNode([_u("u_x")])
    node_b = _FakeNode([_u("u_x")])
    eng = ScriptEngine()
    eng.reload("a", tmp_path / "scripts", node_a)
    eng.reload("b", tmp_path / "scripts", node_b)

    # node_a: two steps of dt=0.5 (variable-dt live path) to elapse 1.0s of sim.
    for _i in range(2):
        ctx = EngineContext(
            t=0.0, dt=0.5, frame=0, uniforms=dict(node_a.uniform_values)
        )
        eng.tick("a", node_a, ctx)
    # node_b: one step of dt=1.0 (a coarser fixed-dt export path) over the same 1.0s.
    ctx = EngineContext(t=0.0, dt=1.0, frame=0, uniforms=dict(node_b.uniform_values))
    eng.tick("b", node_b, ctx)

    # 1*(1.5)^2 = 2.25 vs 1*(1+1) = 2.0 — divergent by design.
    assert node_a.uniform_values["u_x"] != node_b.uniform_values["u_x"]


# ---- (path, mtime) cache (decision 9) ----


def test_cache_no_recompile_when_mtime_unchanged(tmp_path: Path) -> None:
    _write(tmp_path, "u_x", "out.set(0.5)")
    node = _FakeNode([_u("u_x")])
    eng = ScriptEngine()
    eng.reload("n0", tmp_path / "scripts", node)
    behavior_first = eng._nodes["n0"].behaviors["u_x"]
    eng.reload("n0", tmp_path / "scripts", node)  # nothing changed
    assert (
        eng._nodes["n0"].behaviors["u_x"] is behavior_first
    )  # same object, not recompiled


def test_cache_recompiles_on_mtime_change(tmp_path: Path) -> None:
    path = _write(tmp_path, "u_x", "out.set(0.5)")
    node = _FakeNode([_u("u_x")])
    eng = ScriptEngine()
    eng.reload("n0", tmp_path / "scripts", node)
    behavior_first = eng._nodes["n0"].behaviors["u_x"]
    time.sleep(0.01)
    path.write_text("out.set(0.9)", encoding="utf-8")
    eng.reload("n0", tmp_path / "scripts", node)
    assert eng._nodes["n0"].behaviors["u_x"] is not behavior_first  # recompiled


def test_removed_script_drops_binding(tmp_path: Path) -> None:
    path = _write(tmp_path, "u_x", "out.set(0.5)")
    node = _FakeNode([_u("u_x")])
    eng = ScriptEngine()
    eng.reload("n0", tmp_path / "scripts", node)
    assert "u_x" in eng.script_driven_uniforms("n0")
    path.unlink()
    eng.reload("n0", tmp_path / "scripts", node)
    assert "u_x" not in eng.script_driven_uniforms("n0")


# ---- perf sanity (not a gate) ----


def test_tick_is_cheap(tmp_path: Path) -> None:
    _write(tmp_path, "u_x", "out.set(0.5 + 0.3 * sin(ctx.t))")
    node = _FakeNode([_u("u_x")])
    eng = ScriptEngine()
    eng.reload("n0", tmp_path / "scripts", node)
    start = time.perf_counter()
    for i in range(1000):
        eng.tick("n0", node, _ctx(i / 60))
    per_tick_us = (time.perf_counter() - start) / 1000 * 1e6
    assert per_tick_us < 500  # generous ceiling; the proto measured ~3-4us
