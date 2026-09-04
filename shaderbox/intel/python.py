"""The document script read by jedi (078 D10): completion candidates and `K` answers for
Python, in-process, synchronous. The caller decides the thread; this module only turns a
(text, cursor) into symbols. The context fields carry the engine's own gloss, since jedi has
no docstring for a dataclass field."""

import re

import jedi
from jedi.api.classes import BaseName, Completion

from shaderbox.intel.symbols import Symbol, SymbolKind
from shaderbox.scripting.api_doc import API_NAMES, api_symbol_doc, ctx_field_gloss

_CONTEXT_CLASS = "shaderbox.scripting.context.EngineContext"

# In-process inference: no child interpreter to reap, and the one shape measured safe when
# the calls are serialized on one thread (`worker.py`). Built on first use, by that thread.
_ENVIRONMENT: list[jedi.InterpreterEnvironment] = []


def _script(text: str) -> jedi.Script:
    if not _ENVIRONMENT:
        _ENVIRONMENT.append(jedi.InterpreterEnvironment())
    return jedi.Script(text, environment=_ENVIRONMENT[0])


def _first_paragraph(doc: str) -> str:
    return doc.strip().split("\n\n", 1)[0].strip()


def _kind(completion: Completion, after_dot: bool) -> SymbolKind:
    if completion.type == "keyword":
        return SymbolKind.PY_KEYWORD
    if after_dot:
        return SymbolKind.PY_MEMBER
    if completion.name in API_NAMES:
        return SymbolKind.PY_API
    if completion.module_name == "builtins":
        return SymbolKind.PY_BUILTIN
    return SymbolKind.PY_LOCAL


_MEMBER_AT_END = re.compile(r"\.\s*\w*$")
_WORD_AT_END = re.compile(r"\w*$")


def _after_dot(before_caret: str) -> bool:
    # The word at the caret is reached through a dot: `ctx.`, `ctx.t`, `math.si`.
    return _MEMBER_AT_END.search(before_caret) is not None


def _gloss_for(full_name: str | None) -> str:
    if full_name and full_name.startswith(_CONTEXT_CLASS + "."):
        return ctx_field_gloss(full_name.rsplit(".", 1)[1])
    return ""


def _signature_and_doc(name: BaseName) -> tuple[str, str]:
    # jedi's rich docstring opens with the call form for a callable; the raw one is the
    # body. The context gloss wins over an empty body.
    rich = name.docstring()
    raw = name.docstring(raw=True)
    signature = rich.split("\n", 1)[0].strip() if rich and rich != raw else ""
    doc = _gloss_for(name.full_name) or _first_paragraph(raw)
    return signature or f"{name.type} {name.name}", doc


def python_completions(text: str, line: int, column: int) -> list[Symbol]:
    """Candidates at a 0-based (line, column), jedi's order. Names starting with `_` are
    offered only when the typed prefix starts with `_`."""
    lines = text.split("\n")
    if not 0 <= line < len(lines):
        return []
    column = min(column, len(lines[line]))
    before = lines[line][:column]
    after_dot = _after_dot(before)
    found: list[Symbol] = []
    for completion in _script(text).complete(line + 1, column):
        typed = completion.name[: len(completion.name) - len(completion.complete or "")]
        if completion.name.startswith("_") and not typed.startswith("_"):
            continue
        kind = _kind(completion, after_dot=after_dot)
        signature, doc = (
            api_symbol_doc(completion.name)
            if kind == SymbolKind.PY_API
            else _signature_and_doc(completion)
        )
        found.append(Symbol(completion.name, kind, signature=signature, doc=doc))
    if not after_dot:
        # The engine injects the API into every script's globals, which jedi cannot see
        # unless the stub's import line names them; offer them from the engine's own list.
        typed = _WORD_AT_END.search(before)
        prefix = typed.group(0) if typed else ""
        offered = {symbol.name for symbol in found}
        for api_name in sorted(API_NAMES - offered):
            if api_name.startswith(prefix):
                signature, doc = api_symbol_doc(api_name)
                found.append(
                    Symbol(api_name, SymbolKind.PY_API, signature=signature, doc=doc)
                )
    return found


def python_lookup(text: str, line: int, column: int) -> Symbol | None:
    """What `K` shows for the name at a 0-based (line, column), or None."""
    lines = text.split("\n")
    if not 0 <= line < len(lines):
        return None
    column = min(column, len(lines[line]))
    names = _script(text).help(line + 1, column)
    if not names:
        return None
    name = names[0]
    if name.name in API_NAMES and not _after_dot(lines[line][:column]):
        # `Ctx` is a bare alias of the context class, so jedi has no docstring for it and
        # its raw answer is "statement Ctx": the engine's own gloss is the answer.
        signature, doc = api_symbol_doc(name.name)
        return Symbol(name.name, SymbolKind.PY_API, signature=signature, doc=doc)
    signature, doc = _signature_and_doc(name)
    if name.type == "keyword":
        kind = SymbolKind.PY_KEYWORD
    elif _after_dot(lines[line][:column]):
        kind = SymbolKind.PY_MEMBER
    else:
        kind = SymbolKind.PY_LOCAL
    return Symbol(name.name, kind, signature=signature, doc=doc)
