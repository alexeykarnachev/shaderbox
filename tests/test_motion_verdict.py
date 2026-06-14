"""The motion-verdict honesty helpers (feature 043), unit-tested without GL. These are the value-diff
side of the make-or-break "is it animating" signal: drives-0, STATIC, ANIMATING, and the per-uniform
"which one is constant" detail — the four ways the agent reads its own animation headlessly. The one
corroborating pixel render (the visible/FLAT case) is GL and exercised live in the dogfood drive."""

from shaderbox.copilot.backend import (
    _motion_verdict,
    _reindent,
    _script_match_spans,
    _splice_script,
    _uniform_changes,
    _values_differ,
)
from shaderbox.scripting import ScriptProbe, normalize_script_tabs

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


# ---- the edit_script indent-aware matcher + splice (feature 043) ----


def test_exact_match_fast_path() -> None:
    src = "a = 1\nb = 1\nc = 2\n"
    spans = _script_match_spans(src, "= 1")
    assert [(s, e) for s, e, _ in spans] == [(2, 5), (8, 11)]  # all occurrences
    assert _script_match_spans(src, "nope") == []
    assert _script_match_spans(src, "") == []  # empty matches nothing


def test_indent_forgiving_match_8_vs_6_spaces() -> None:
    # The real breakout miss: source at 8/12 spaces, the agent's old_str re-typed at 6/8 (NOT a verbatim
    # substring, so the exact path fails). The structural fallback must match AND re-indent the
    # replacement onto the real source column.
    src = (
        "        for c in cols:\n"
        "            if hit:\n"
        "                self.b[c] = 0.0\n"
        "        paddle = 1\n"
    )
    old = (
        "      if hit:\n          self.b[c] = 0.0\n"  # 6-vs-8 step, NOT in src verbatim
    )
    assert old.rstrip("\n") not in src  # the exact path can't find it
    spans = _script_match_spans(src, old)
    assert len(spans) == 1
    new = "      if hit:\n          self.b[c] = 1.0\n"
    out = _splice_script(src, spans, new)
    # the replacement landed at the source's real 12/16-space columns, not the agent's 6/8
    assert "            if hit:\n                self.b[c] = 1.0\n" in out


def test_structural_match_rejects_different_nesting() -> None:
    # Same content, DIFFERENT relative nesting must NOT match (Python indentation is semantic). Use a
    # form that is NOT a verbatim substring so only the structural path can (wrongly) fire.
    src = "for i in x:\n  a = 1\n  b = 2\n"  # a, b at the SAME 2-space level
    old = "a = 1\n      b = 2\n"  # b nested deeper than a -> different structure; not a substring
    assert old.rstrip("\n") not in src
    assert _script_match_spans(src, old) == []


def test_duplicate_structural_form_is_multi_match() -> None:
    # Two blocks with the same relative form, neither a verbatim substring of the same-indent kind,
    # surface as >1 (loud reject upstream), never a silent wrong-region edit.
    src = "if a:\n    x = 1\nif b:\n        x = 1\n"  # x at 4 and at 8 spaces
    old = "  x = 1\n"  # 2-space — not verbatim at either site -> structural path, both forms match
    assert len(_script_match_spans(src, old)) == 2


def test_normalize_tabs() -> None:
    assert normalize_script_tabs("\tx = 1\r\n\t\ty\r") == "    x = 1\n        y\n"


def test_reindent_shifts_block() -> None:
    assert _reindent("a\n    b\n", 4) == "    a\n        b\n"
    assert _reindent("        a\n            b\n", -4) == "    a\n        b\n"
    assert _reindent("a\n\nb\n", 2) == "  a\n\n  b\n"  # blank lines untouched
