"""An engine-driven uniform in the working set shows a marker, never a number: its cache entry is
whatever the last render wrote (0.0 before any), and a model read `u_aspect float = 0.0` and
built its layout around it (deepseek-v4-flash on the station). Enumerated from the engine's own
list so a new builtin is covered without a new test."""

from typing import Any

from shaderbox.core import ENGINE_UNIFORM_TYPES

_SHADER = (
    "#version 460 core\nin vec2 vs_uv;\nout vec4 fs_color;\n"
    + "".join(f"uniform {t} {n};\n" for n, t in ENGINE_UNIFORM_TYPES.items())
    + "uniform float u_glow = 0.4;\nvoid main() { fs_color = vec4(u_glow"
    + "".join(
        f" + float({n}{'.x' if t.startswith('vec') else ''})"
        for n, t in ENGINE_UNIFORM_TYPES.items()
    )
    + ", 0.0, 0.0, 1.0); }\n"
)


def test_engine_uniform_rows_carry_a_marker_not_a_value(app: Any) -> None:
    backend = app.copilot_backend
    assert backend.apply_full_rewrite(_SHADER, "").errors == []
    views, _evicted = backend.read_working_set()
    rows = {r.split(" ")[0]: r for v in views for r in v.uniforms}
    for name in ENGINE_UNIFORM_TYPES:
        assert rows[name].endswith("= <set by the engine each frame>"), rows[name]
    assert rows["u_glow"].startswith("u_glow float = 0.4"), rows["u_glow"]
