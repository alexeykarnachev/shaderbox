"""The jedi-backed Python side of the intel module (078 D10), on the real stub."""

from shaderbox.intel.python import python_completions, python_lookup
from shaderbox.intel.symbols import SymbolKind
from shaderbox.scripting.engine import script_stub_for


def _stub_with(line: str) -> tuple[str, int, int]:
    """(text, line index, caret column at the END of `line`). The column is derived rather
    than written out: hardcoded offsets silently point at the wrong token when a name in the
    fixture changes length."""
    stub = script_stub_for({"main": []}).rstrip()
    body = "        " + line
    text = stub + "\n" + body + "\n"
    return text, len(text.split("\n")) - 2, len(body)


def test_ctx_members_complete_with_the_engine_gloss() -> None:
    text, line, col = _stub_with("x = context.")
    found = python_completions(text, line, col)
    names = [s.name for s in found]
    assert names[:4] == ["dt", "frame", "mouse", "t"]
    assert all(s.kind == SymbolKind.PY_MEMBER for s in found[:4])
    assert "__class__" not in names
    assert next(s for s in found if s.name == "t").doc.startswith("Seconds since")


def test_math_members_and_the_api_complete_by_kind() -> None:
    text, line, col = _stub_with("y = math.si")
    assert [s.name for s in python_completions(text, line, col)][:2] == ["sin", "sinh"]
    text, line, col = _stub_with("Scr")
    api = python_completions(text, line, col)
    assert {s.name for s in api} >= {"ScriptContext"}
    assert all(s.kind == SymbolKind.PY_API for s in api if s.name == "ScriptContext")
    text, line, col = _stub_with("se")
    assert any(
        s.name == "self" and s.kind == SymbolKind.PY_LOCAL
        for s in python_completions(text, line, col)
    )


def test_lookup_on_ctx_field_and_on_a_builtin_and_past_the_line_end() -> None:
    text, line, col = _stub_with("x = context.t")
    found = python_lookup(text, line, col)
    assert found is not None and found.name == "t"
    assert found.doc.startswith("Seconds since")
    text, line, col = _stub_with("y = math.sin")
    found = python_lookup(text, line, col)
    assert found is not None and found.name == "sin" and "sin" in found.signature
    past = python_lookup(text, line, 999)
    assert past is not None and past.name == "sin", "clamped to the line end"
    assert python_lookup("", 5, 0) is None


def test_the_api_gloss_wins_for_an_injected_name_imported_or_not() -> None:
    # The stub imports `ScriptContext`; jedi has no doc of its own for the name at a bare
    # reference, so the engine's gloss is the answer in completion and under `K` alike.
    text, line, col = _stub_with("Scr")
    found = next(
        s for s in python_completions(text, line, col) if s.name == "ScriptContext"
    )
    assert found.kind == SymbolKind.PY_API
    assert found.doc.startswith("The engine state for one frame")
    text, line, _col = _stub_with("c: ScriptContext = context")
    looked = python_lookup(text, line, 12)
    assert looked is not None and looked.name == "ScriptContext"
    assert looked.kind == SymbolKind.PY_API
    assert looked.doc.startswith("The engine state for one frame")


def test_a_member_spelled_like_an_api_name_is_a_member_under_k() -> None:
    text = "class P:\n    Text = 1\n\nz = P.Text\n"
    looked = python_lookup(text, 3, 7)
    assert looked is not None and looked.name == "Text"
    assert looked.kind == SymbolKind.PY_MEMBER, "reached through a dot, not the API"
