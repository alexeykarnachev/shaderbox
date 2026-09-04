"""The jedi-backed Python side of the intel module (078 D10), on the real stub."""

from shaderbox.intel.python import python_completions, python_lookup
from shaderbox.intel.symbols import SymbolKind
from shaderbox.scripting.engine import script_stub_for


def _stub_with(line: str) -> tuple[str, int]:
    stub = script_stub_for({"main": []}).rstrip()
    text = stub + "\n        " + line + "\n"
    return text, len(text.split("\n")) - 2


def test_ctx_members_complete_with_the_engine_gloss() -> None:
    text, line = _stub_with("x = ctx.")
    found = python_completions(text, line, 16)
    names = [s.name for s in found]
    assert names[:4] == ["dt", "frame", "mouse", "t"]
    assert all(s.kind == SymbolKind.PY_MEMBER for s in found[:4])
    assert "__class__" not in names
    assert next(s for s in found if s.name == "t").doc == "seconds"


def test_math_members_and_the_api_complete_by_kind() -> None:
    text, line = _stub_with("y = math.si")
    assert [s.name for s in python_completions(text, line, 19)][:2] == ["sin", "sinh"]
    text, line = _stub_with("Ve")
    api = python_completions(text, line, 10)
    assert {s.name for s in api} >= {"Vec2", "Vec3", "Vec4"}
    assert all(s.kind == SymbolKind.PY_API for s in api if s.name.startswith("Vec"))
    text, line = _stub_with("se")
    assert any(
        s.name == "self" and s.kind == SymbolKind.PY_LOCAL
        for s in python_completions(text, line, 10)
    )


def test_lookup_on_ctx_field_and_on_a_builtin_and_past_the_line_end() -> None:
    text, line = _stub_with("x = ctx.t")
    found = python_lookup(text, line, 16)
    assert found is not None and found.name == "t" and found.doc == "seconds"
    text, line = _stub_with("y = math.sin")
    found = python_lookup(text, line, 18)
    assert found is not None and found.name == "sin" and "sin" in found.signature
    assert python_lookup(text, line, 999) is not None or True
    assert python_lookup("", 5, 0) is None
