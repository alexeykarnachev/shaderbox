"""The jedi-backed Python side of the intel module (078 D10), on the real stub."""

from shaderbox.intel.python import python_completions, python_lookup, python_spans
from shaderbox.intel.symbols import SymbolKind
from shaderbox.scripting.engine import script_stub_for
from shaderbox.syntax_colors import kind_capture, kind_color


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
    assert found.doc.startswith("The engine state for one tick")
    text, line, _col = _stub_with("c: ScriptContext = context")
    looked = python_lookup(text, line, 12)
    assert looked is not None and looked.name == "ScriptContext"
    assert looked.kind == SymbolKind.PY_API
    assert looked.doc.startswith("The engine state for one tick")


def test_a_member_spelled_like_an_api_name_is_a_member_under_k() -> None:
    text = "class P:\n    Text = 1\n\nz = P.Text\n"
    looked = python_lookup(text, 3, 7)
    assert looked is not None and looked.name == "Text"
    assert looked.kind == SymbolKind.PY_MEMBER, "reached through a dot, not the API"


def test_a_caret_inside_a_string_literal_gets_nothing() -> None:
    # jedi completes a literal as a file path (`"u` offered `uv.lock"`), and the injected API
    # names would otherwise leak in by prefix. A closed literal is not "inside".
    for line in ('s = "u', "s = 'Scr", 'x = f("Scr', 's = """Scr'):
        text, line_index, col = _stub_with(line)
        assert python_completions(text, line_index, col) == [], line
    text, line_index, col = _stub_with('s = "u" + Scr')
    assert {s.name for s in python_completions(text, line_index, col)} >= {
        "ScriptContext"
    }


def test_a_class_object_does_not_offer_the_metaclass_protocol() -> None:
    text, line_index, col = _stub_with("y = Behavior.m")
    names = {s.name for s in python_completions(text, line_index, col)}
    assert "mro" not in names


def test_a_definition_takes_the_kind_treesitter_would_give_it() -> None:
    """`class`, `__init__` and an ordinary `def` are three kinds, because nvim draws three
    colours: `@type`, `@constructor` and `@function`.

    `__init__` is the one worth pinning. Treesitter's captures are layered and the LAST
    wins, so it ends at `@constructor` (orange) having passed through `@function.method`
    (green) -- reading the list top-down gives the wrong answer, which is how it was
    coloured green before.

    Falsifier: drop the `_CONSTRUCTOR_NAMES` branch and `__init__` comes back green.
    """
    source = "\n".join(
        [
            "class Behavior(ScriptBehavior):",
            "    def __init__(self, seed: int):",
            "        self.seed = seed",
            "",
            "    def step(self) -> None:",
            "        pass",
            "",
            "    def __new__(cls):",
            "        pass",
        ]
    )
    by_name = {span.name: span.kind for span in python_spans(source)}
    assert by_name["Behavior"] == SymbolKind.PY_CLASS
    assert by_name["__init__"] == SymbolKind.PY_CONSTRUCTOR
    assert by_name["__new__"] == SymbolKind.PY_CONSTRUCTOR
    assert by_name["step"] == SymbolKind.PY_DEFINITION
    # And the three really are three colours, not three names for one.
    assert (
        len(
            {
                kind_color(by_name["Behavior"]),
                kind_color(by_name["__init__"]),
                kind_color(by_name["step"]),
            }
        )
        == 3
    )


def test_a_parameter_and_a_base_class_take_the_captures_nvim_uses() -> None:
    """The two the maintainer reported as mismatched: a signature's parameter names and
    the names inside the inheritance brackets.

    nvim gives a parameter `@variable.parameter` (blue) and a base class `@type` (yellow);
    both had been drawn as something else. Pinned by CAPTURE rather than by hex, so a
    theme swap does not break this test -- `test_theme_file.py` owns the hex.
    """
    source = "\n".join(
        [
            "class Behavior(ScriptBehavior):",
            "    def __init__(self, seed: int, scale: float = 1.0):",
            "        pass",
        ]
    )
    spans = python_spans(source)
    by_name = {span.name: span.kind for span in spans}
    assert kind_capture(by_name["seed"]) == "@variable.parameter"
    assert kind_capture(by_name["scale"]) == "@variable.parameter"
    assert kind_capture(by_name["ScriptBehavior"]) == "@type"
    # `self` stays the language's own name even in the parameter list.
    assert kind_capture(by_name["self"]) == "@variable.builtin"
    # A parameter and a base class are not the same colour, which is the report itself.
    assert kind_color(by_name["seed"]) != kind_color(by_name["ScriptBehavior"])
