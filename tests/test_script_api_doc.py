"""Feature 059 D3: the generated SCRIPT API block. Drift pins — each test goes red when the Python
script surface changes without the block following it.

`vars(cls)` is the introspection, deliberately not `inspect.getmembers`: getmembers leaks `tuple`'s
`count`/`index`, returns nothing for `Array`/`Text` (their surface is the constructor), and skips
dunders — so it would not see `__mul__` at all."""

import ast
from inspect import signature
from pathlib import Path

from shaderbox.copilot.prompt import _context_block
from shaderbox.copilot.prompt_context import build_context
from shaderbox.scripting import api_doc
from shaderbox.scripting.api_doc import (
    _CTX_GLOSS,
    _CTX_HELP,
    _VALUE_SHAPE_GLOSS,
    _VEC_OPERATOR_GLOSS,
    API_NAMES,
    api_symbol_doc,
    ctx_field_gloss,
    script_api_summary,
)
from shaderbox.scripting.context import EXPORT_MOUSE, EngineContext, MouseState
from shaderbox.scripting.engine import _stub_kind
from shaderbox.scripting.outputs import Array, Text, Vec2, Vec3, Vec4, _Vec
from tests._caps import minimal_caps

_VEC_CLASSES: tuple[type, ...] = (_Vec, Vec2, Vec3, Vec4)

# Dunders `vars()` reports that describe the class rather than its operator surface — everything else
# a Vec class defines must be an allowlisted operator, or the coverage check has gone stale.
_STRUCTURAL_DUNDERS: frozenset[str] = frozenset(
    {
        "__new__",
        "__init__",
        "__slots__",
        "__module__",
        "__qualname__",
        "__doc__",
        "__dict__",
        "__weakref__",
        "__annotations__",
        "__type_params__",
        "__orig_bases__",
        "__firstlineno__",
        "__static_attributes__",
    }
)


def _flat(text: str) -> str:
    # The block is wrapped for the prompt, so a multi-word gloss can straddle a line break; assert
    # against the unwrapped text or the pin fails on layout rather than on content.
    return " ".join(text.split())


def _public_vec_members(classes: tuple[type, ...]) -> set[str]:
    return {n for cls in classes for n in vars(cls) if not n.startswith("_")}


def _operator_dunders(classes: tuple[type, ...]) -> set[str]:
    return {
        n
        for cls in classes
        for n in vars(cls)
        if n.startswith("__") and n not in _STRUCTURAL_DUNDERS
    }


def _uncovered_members(summary: str, classes: tuple[type, ...]) -> set[str]:
    return {n for n in _public_vec_members(classes) if f".{n}" not in summary}


def test_summary_covers_every_public_vec_member() -> None:
    # A new `Vec3.reflect()` that nobody documents goes red here.
    assert _uncovered_members(script_api_summary(), _VEC_CLASSES) == set()


def test_the_vec_member_check_actually_fails_on_an_undocumented_member() -> None:
    # The falsifier for the test above: a check that cannot go red pins nothing.
    class _FakeVec3(Vec3):
        def reflect(self) -> "Vec3":
            return self

    assert _uncovered_members(script_api_summary(), (_FakeVec3,)) == {"reflect"}


def test_operator_dunders_match_the_allowlist_and_all_reach_the_summary() -> None:
    # Both directions: dropping `__truediv__` from outputs.py goes red, and adding `__mod__` without
    # documenting it goes red too.
    assert _operator_dunders(_VEC_CLASSES) == set(_VEC_OPERATOR_GLOSS)
    # The map IS the rendered operator list, so this half fails if the bullet is dropped or reworded.
    summary = _flat(script_api_summary())
    for dunder, gloss in _VEC_OPERATOR_GLOSS.items():
        assert gloss in summary, dunder


def test_ctx_gloss_keys_are_exactly_the_dataclass_fields() -> None:
    # A new `ctx` field is undocumented until someone writes its gloss — in BOTH renderings, the
    # prompt's terse one and the `K` note's prose (079 D3).
    assert set(_CTX_GLOSS) == set(EngineContext.__dataclass_fields__)
    assert set(_CTX_HELP) == set(EngineContext.__dataclass_fields__)


def test_no_help_text_is_empty_or_a_semicolon_joined_list() -> None:
    # 079 D3, from the maintainer's reading of the notes: `ctx.dt` and `ctx.frame` opened EMPTY,
    # and `ctx.mouse` was six facts joined by `;` on one line. Every human-facing string now
    # opens with a summary sentence and puts one fact per line. Falsifier: restore either gap and
    # the emptiness or the `;` check goes red.
    human = {f"ctx.{k}": v for k, v in _CTX_HELP.items()}
    human |= {name: api_symbol_doc(name)[1] for name in sorted(API_NAMES)}
    for where, text in human.items():
        assert text.strip(), f"{where} opens an empty note"
        summary = text.splitlines()[0]
        assert summary.endswith("."), f"{where} has no summary sentence: {summary!r}"
        assert summary[0].isupper(), f"{where}'s summary is not a sentence: {summary!r}"
        for line in text.splitlines():
            assert line.count(";") <= 1, f"{where} joins a list with `;`: {line!r}"


def test_every_ctx_field_answers_under_k() -> None:
    # The `K` path itself, not just the table: an undocumented field would return "".
    for name in EngineContext.__dataclass_fields__:
        assert ctx_field_gloss(name), name


def test_summary_lists_every_ctx_field_and_the_mouse_subfields() -> None:
    summary = script_api_summary()
    for name in EngineContext.__dataclass_fields__:
        assert f"`{name}`" in summary, name
    for name in MouseState.__dataclass_fields__:
        assert f"`{name}`" in summary, name


def test_the_mouse_gloss_carries_the_frozen_at_center_caveat() -> None:
    # 17ab552 inlined this fact next to the ctx intro because a live leak showed the agent trusting
    # mouse motion in a probe. The ctx intro moved to the generated block; the caveat moves with it.
    at = f"{EXPORT_MOUSE.x:g},{EXPORT_MOUSE.y:g}"
    caveat = f"FROZEN at {at} on export and in the headless probe"
    assert caveat in _CTX_GLOSS["mouse"]
    assert caveat in _flat(script_api_summary())


def test_every_stub_kind_type_name_has_a_value_shape_gloss() -> None:
    # The drift pin WITHOUT lifting `_stub_kind`'s 7-outcome moderngl.Uniform dispatch into the
    # GL-free module: the two share the type-name -> prose map, nothing more. Names read off the
    # dispatch's own return statements, so a new outcome shows up here.
    source = Path(_stub_kind.__code__.co_filename).read_text(encoding="utf-8")
    tree = ast.parse(source)
    returned: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_stub_kind":
            for ret in ast.walk(node):
                if isinstance(ret, ast.Return) and isinstance(ret.value, ast.Tuple):
                    first = ret.value.elts[0]
                    assert isinstance(first, ast.Constant), ast.dump(first)
                    returned.add(str(first.value))
    assert returned, "found no type names in _stub_kind"
    assert returned <= set(_VALUE_SHAPE_GLOSS), returned - set(_VALUE_SHAPE_GLOSS)


def test_array_and_text_are_described_by_their_real_constructor_params() -> None:
    # Their surface IS the constructor, so `vars()` says nothing useful — pin the signature instead.
    summary = script_api_summary()
    for cls, ctor in ((Array, Array.__init__), (Text, Text.__init__)):
        assert cls.__name__ in summary
        for param in signature(ctor).parameters:
            if param != "self":
                assert f"{cls.__name__}({param})" in summary, (cls.__name__, param)


def test_the_block_reaches_the_rare_prompt_tier() -> None:
    # The WIRE, not the definition: a generated block nobody renders is worth nothing.
    block = _context_block(build_context(minimal_caps()))
    assert "SCRIPT API" in block
    assert script_api_summary() in block
    # It sits AFTER the example library and BEFORE conventions, so the GLSL cluster stays contiguous.
    assert block.index("EXAMPLE LIBRARY") < block.index("SCRIPT API")
    assert block.index("SCRIPT API") < block.index("CONVENTIONS")


def test_api_doc_reaches_only_for_the_gl_free_half_of_the_package() -> None:
    # api_doc must never import `engine`/`behavior` (both pull moderngl) — it is built off the main
    # thread. A runtime sys.modules assertion cannot express this: importing ANY submodule executes
    # `shaderbox/scripting/__init__.py`, which re-exports the GL half, so the invariant is over this
    # module's OWN imports.
    tree = ast.parse(Path(api_doc.__file__).read_text(encoding="utf-8"))
    reached: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            reached.add(node.module)
        elif isinstance(node, ast.Import):
            reached.update(alias.name for alias in node.names)
    assert {m for m in reached if m.startswith("shaderbox")} == {
        "shaderbox.scripting.context",
        "shaderbox.scripting.outputs",
    }
    assert not [m for m in reached if "moderngl" in m or "OpenGL" in m]


def test_the_contract_bullet_states_both_addressing_forms() -> None:
    # 069 D3's grammar is what the copilot reads before writing a script, so the block must state
    # BOTH forms and the precedence. Falsifier: revert the api_doc contract-bullet edit.
    summary = script_api_summary()
    assert "EVERY pass declaring it" in summary
    assert "{pass: {uniform: value}}" in summary
    assert "WINS over a bare key" in summary


def test_the_mouse_gloss_states_the_button_and_the_previous_position() -> None:
    # A bare field list tells the agent a NAME and not a meaning. Falsifier: revert the mouse-gloss
    # edit — `down`/`prev_x` still appear as field names (the field-list pin covers that), but the
    # sentences describing them are gone.
    summary = script_api_summary()
    assert "LMB" in summary
    assert "PREVIOUS cursor position" in summary
