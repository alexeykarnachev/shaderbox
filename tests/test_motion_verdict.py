"""The motion-verdict honesty helpers (feature 043), unit-tested without GL. These are the value-diff
side of the make-or-break "is it animating" signal: drives-0, STATIC, ANIMATING, and the per-uniform
"which one is constant" detail — the four ways the agent reads its own animation headlessly. The one
corroborating pixel render (the visible/FLAT case) is GL and exercised live in the dogfood drive."""

from shaderbox.copilot.backend import (
    _motion_verdict,
    _uniform_changes,
    _values_differ,
)
from shaderbox.scripting import ScriptProbe

_EPS = 1e-4


def _probe(
    driven: set[str], samples: list[tuple[float, dict[str, object]]]
) -> ScriptProbe:
    return ScriptProbe(
        compile_error=None,
        driven=driven,
        per_key_errors=[],
        orphan_keys=[],
        samples=samples,
    )


def test_values_differ_scalar_and_nested() -> None:
    assert _values_differ(0.0, 0.5, _EPS)
    assert not _values_differ(0.5, 0.5 + 1e-9, _EPS)  # within epsilon
    assert _values_differ((0.1, 0.2), (0.1, 0.9), _EPS)  # one component moved
    assert not _values_differ((0.1, 0.2), (0.1, 0.2), _EPS)
    assert _values_differ([1.0, 2.0], [1.0], _EPS)  # length mismatch = differ
    assert _values_differ([(0.0, 0.0)], [(0.0, 1.0)], _EPS)  # nested


def test_uniform_changes_detects_motion() -> None:
    samples = [(0.0, {"u_x": 0.0}), (0.5, {"u_x": 0.3}), (1.0, {"u_x": 0.6})]
    assert _uniform_changes("u_x", samples, _EPS)
    held = [(0.0, {"u_c": 0.7}), (0.5, {"u_c": 0.7}), (1.0, {"u_c": 0.7})]
    assert not _uniform_changes("u_c", held, _EPS)


def test_verdict_drives_nothing_is_loud() -> None:
    verdict = _motion_verdict(_probe(set(), []), "", _EPS)
    assert "drives 0 uniforms" in verdict and "Nothing" in verdict


def test_verdict_animating_names_changing_and_constant() -> None:
    samples = [
        (0.0, {"u_x": 0.0, "u_c": 0.7}),
        (0.5, {"u_x": 0.3, "u_c": 0.7}),
        (1.0, {"u_x": 0.6, "u_c": 0.7}),
    ]
    verdict = _motion_verdict(_probe({"u_x", "u_c"}, samples), "", _EPS)
    assert "u_x CHANGE across t (ANIMATING)" in verdict
    assert "u_c constant" in verdict


def test_verdict_static_when_no_value_varies() -> None:
    samples = [
        (0.0, {"u_x": 0.4}),
        (0.5, {"u_x": 0.4}),
        (1.0, {"u_x": 0.4}),
    ]
    verdict = _motion_verdict(_probe({"u_x"}, samples), "", _EPS)
    assert "STATIC" in verdict
    assert "ctx.t" in verdict  # steers the agent to vary by ctx.t


def test_verdict_carries_the_render_line() -> None:
    # The corroborating render line (the visible/FLAT honesty fact) rides the verdict so the agent
    # sees "values animate" AND "but the frame is FLAT" together.
    samples = [(0.0, {"u_x": 0.0}), (1.0, {"u_x": 0.5})]
    render = "render@t=0.5s FLAT -- one uniform color rgba(0,0,0,0)"
    verdict = _motion_verdict(_probe({"u_x"}, samples), render, _EPS)
    assert "FLAT" in verdict
    assert "ANIMATING" in verdict  # both facts present for the agent to reconcile
