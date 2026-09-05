"""Feature 059 D3: the generated SCRIPT API block. Drift pins — each test goes red when the Python
script surface changes without the block following it.

`vars(cls)` is the introspection, deliberately not `inspect.getmembers`: getmembers leaks `tuple`'s
`count`/`index`, returns nothing for `Array`/`Text` (their surface is the constructor), and skips
dunders — so it would not see `__mul__` at all."""

import ast
from inspect import cleandoc
from pathlib import Path

from shaderbox.copilot.prompt import _context_block
from shaderbox.copilot.prompt_context import build_context
from shaderbox.scripting import api_doc
from shaderbox.scripting.api_doc import (
    _CTX_GLOSS,
    _CTX_HELP,
    _IMPORT_NAMES,
    _VALUE_SHAPE_GLOSS,
    API_NAMES,
    api_symbol_doc,
    ctx_field_gloss,
    script_api_summary,
)
from shaderbox.scripting.behavior import _INJECTED_NAMES
from shaderbox.scripting.context import EXPORT_MOUSE, EngineContext, MouseState
from shaderbox.scripting.engine import _stub_kind, script_stub_for
from tests._caps import minimal_caps


def _flat(text: str) -> str:
    # The block is wrapped for the prompt, so a multi-word gloss can straddle a line break; assert
    # against the unwrapped text or the pin fails on layout rather than on content.
    return " ".join(text.split())


def test_ctx_gloss_keys_are_exactly_the_dataclass_fields() -> None:
    # A new `ctx` field is undocumented until someone writes its gloss — in BOTH renderings, the
    # prompt's terse one and the `K` note's prose (079 D3).
    assert set(_CTX_GLOSS) == set(EngineContext.__dataclass_fields__)
    assert set(_CTX_HELP) == set(EngineContext.__dataclass_fields__)


def test_every_help_text_follows_pep_257() -> None:
    """PEP 257's multi-line shape, which is what 079 D3 settles on.

    From the source (peps.python.org/pep-0257): a summary line that fits on one line and
    ends in a period, then a BLANK line, then the elaboration. The maintainer's findings were
    both violations of it — `ctx.dt` and `ctx.frame` opened empty, and `ctx.mouse` ran six
    facts together on one line with semicolons. Falsifier: restore either and this goes red.
    """
    human = {f"ctx.{k}": v for k, v in _CTX_HELP.items()}
    human |= {name: api_symbol_doc(name)[1] for name in sorted(API_NAMES)}
    for where, text in human.items():
        assert text.strip(), f"{where} opens an empty note"
        lines = text.splitlines()
        summary = lines[0]
        assert summary.endswith("."), f"{where} has no summary sentence: {summary!r}"
        assert summary[0].isupper(), f"{where}'s summary is not a sentence: {summary!r}"
        assert len(summary) <= 79, f"{where}'s summary runs to {len(summary)} chars"
        if len(lines) > 1:
            assert not lines[1].strip(), (
                f"{where} runs into its description with no blank line after the summary: "
                f"{lines[1]!r}"
            )
        for line in lines:
            assert line.count(";") <= 1, f"{where} joins a list with `;`: {line!r}"


def test_every_google_section_is_spelled_and_indented_as_google_writes_it() -> None:
    """The Google style guide's section shape (google.github.io/styleguide/pyguide.html 3.8).

    A section is a known heading on its own line, and each entry under it is `name: text`
    indented beneath. A misspelled heading (`Arguments:`, `Return:`) renders as prose in every
    tool that reads these, which is the failure worth catching.
    """
    known = ("Args:", "Returns:", "Yields:", "Raises:", "Attributes:", "Example:")
    sources: dict[str, str] = {f"ctx.{name}": text for name, text in _CTX_HELP.items()}
    sources |= {name: api_symbol_doc(name)[1] for name in sorted(API_NAMES)}
    sources |= {
        cls.__name__: cleandoc(cls.__doc__ or "") for cls in (EngineContext, MouseState)
    }
    sources["script stub"] = script_stub_for({"main": []})
    misspelled = ("Arguments:", "Return:", "Parameters:", "Raise:", "Attribute:")
    for where, text in sources.items():
        for line in text.splitlines():
            stripped = line.strip()
            assert stripped not in misspelled, (
                f"{where} spells a section heading {stripped!r}; Google's headings are "
                f"{known}"
            )
            if stripped in known:
                assert line.rstrip() == line, f"{where}'s {stripped} has trailing space"


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


def test_the_advertised_import_is_one_the_gate_accepts() -> None:
    # Three lists must agree or the app tells the user (and the copilot) to write an import that
    # raises: the names the prompt advertises, the names the engine injects, and the names the
    # script import gate lets through. Falsifier: advertise a name the gate refuses — a script
    # written from the prompt block then fails to compile.
    advertised = set(_IMPORT_NAMES.split(", "))
    assert advertised <= _INJECTED_NAMES, (
        f"the prompt advertises {sorted(advertised - _INJECTED_NAMES)}, which a script cannot "
        "import"
    )
    assert API_NAMES == _INJECTED_NAMES, (
        "the editor's API names and the engine's injected names have drifted apart: "
        f"{sorted(API_NAMES ^ _INJECTED_NAMES)}"
    )
