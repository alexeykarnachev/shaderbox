"""The document script read by jedi (078 D10): completion candidates and `K` answers for
Python, in-process, synchronous. The caller decides the thread; this module only turns a
(text, cursor) into symbols. The context fields carry the engine's own gloss, since jedi has
no docstring for a dataclass field."""

import io
import re
import tokenize
from collections.abc import Iterator
from dataclasses import dataclass

import jedi
import parso
from jedi.api.classes import BaseName, Completion
from parso.python.tree import Class, Decorator, Function, Name, Operator
from parso.tree import BaseNode, NodeOrLeaf

from shaderbox.intel.symbols import Symbol, SymbolKind
from shaderbox.scripting.api_doc import API_NAMES, api_symbol_doc, context_field_gloss

_CONTEXT_CLASS = "shaderbox.scripting.context.ScriptContext"

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
    # The word at the caret is reached through a dot: `context.`, `context.t`, `math.si`.
    return _MEMBER_AT_END.search(before_caret) is not None


# The tokenizer's two verdicts for a caret inside an unclosed literal; an open bracket reports
# "EOF in multi-line statement" instead, which is not a string.
_IN_STRING_VERDICTS = ("unterminated string literal", "EOF in multi-line string")


def _inside_string(text: str, line: int, column: int) -> bool:
    """Whether the caret sits inside a string literal: the source up to the caret tokenizes
    to an unterminated string. jedi completes a literal as a FILE PATH, which a document
    script never wants."""
    lines = text.split("\n")
    head = "\n".join([*lines[:line], lines[line][:column]])
    try:
        list(tokenize.generate_tokens(io.StringIO(head).readline))
    except tokenize.TokenError as error:
        return str(error.args[0]).startswith(_IN_STRING_VERDICTS)
    return False


_METACLASS_PREFIX = "builtins.type."


def _gloss_for(full_name: str | None) -> str:
    if full_name and full_name.startswith(_CONTEXT_CLASS + "."):
        return context_field_gloss(full_name.rsplit(".", 1)[1])
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
    offered only when the typed prefix starts with `_`; a caret inside a string literal gets
    nothing; a class object's completions are its own attributes, never `type`'s (`mro`)."""
    lines = text.split("\n")
    if not 0 <= line < len(lines):
        return []
    column = min(column, len(lines[line]))
    if _inside_string(text, line, column):
        return []
    before = lines[line][:column]
    after_dot = _after_dot(before)
    found: list[Symbol] = []
    for completion in _script(text).complete(line + 1, column):
        typed = completion.name[: len(completion.name) - len(completion.complete or "")]
        if completion.name.startswith("_") and not typed.startswith("_"):
            continue
        if (completion.full_name or "").startswith(_METACLASS_PREFIX):
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
        # `ScriptContext` is a bare alias of the context class, so jedi has no docstring for it and
        # its raw answer is "statement ScriptContext": the engine's own gloss is the answer.
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


# `self` and `cls` mean the same thing wherever they appear, so they are name-keyed facts and
# ride the word table (105 D2): no parser, no positions, nothing to go stale on an edit.
_SELF_NAMES: frozenset[str] = frozenset({"self", "cls"})

# The two names treesitter's python grammar draws as `@constructor` rather than
# `@function.method`, listed by its own `#any-of?` predicate in `highlights.scm`.
_CONSTRUCTOR_NAMES: frozenset[str] = frozenset({"__init__", "__new__"})

_DUNDER = re.compile(r"^__\w+__$")
_WORD = re.compile(r"[A-Za-z_]\w*")


def python_word_classes(text: str) -> dict[str, SymbolKind]:
    """The edit-invariant half of a script's semantic colouring, keyed by spelling.

    `GlslIndex.classes()`'s Python counterpart: the host feeds these through
    `ed_set_word_class`, where the library re-applies them against new positions on every
    retokenization. Only names the buffer actually contains are returned, so the table stays
    the size of the file rather than the size of the vocabulary."""
    found: dict[str, SymbolKind] = {}
    for match in _WORD.finditer(text):
        word = match.group(0)
        if word in _SELF_NAMES:
            found[word] = SymbolKind.PY_SELF
        elif _DUNDER.match(word):
            found[word] = SymbolKind.PY_DUNDER
    return found


@dataclass(frozen=True)
class SpanSymbol:
    """A name at a place: what the word table cannot express. Lines and columns are 0-based,
    the unit every other host-to-library call uses; `column_end` is exclusive."""

    name: str
    kind: SymbolKind
    line: int
    column: int
    column_end: int


def _definition_spans(module: NodeOrLeaf) -> list[SpanSymbol]:
    """The name after `def` or `class`, which is the case the word table cannot express:
    `update` at its definition and `update` two lines down are the same spelling.

    Read off the parso tree rather than jedi's `get_names`, which reports an IMPORTED name as
    a definition at its import site -- `from dataclasses import dataclass` would colour
    `dataclass` as though the script defined it. A `Function`/`Class` node is structural, so
    the import case cannot arise."""
    found: list[SpanSymbol] = []
    for node in _walk(module):
        if not isinstance(node, (Function, Class)):
            continue
        # Three kinds because treesitter draws three colours: a `class` name is `@type`,
        # `__init__` is `@constructor` (the last capture wins, over `@function.method`),
        # and every other `def` is `@function`.
        if isinstance(node, Class):
            kind = SymbolKind.PY_CLASS
        elif node.name.value in _CONSTRUCTOR_NAMES:
            kind = SymbolKind.PY_CONSTRUCTOR
        else:
            kind = SymbolKind.PY_DEFINITION
        found.append(_span_of(node.name, kind))
    return found


def _walk(node: NodeOrLeaf) -> Iterator[NodeOrLeaf]:
    stack: list[NodeOrLeaf] = [node]
    while stack:
        current = stack.pop()
        yield current
        if isinstance(current, BaseNode):
            stack.extend(current.children)


def _span_of(leaf: Name, kind: SymbolKind) -> SpanSymbol:
    line, column = leaf.start_pos
    return SpanSymbol(leaf.value, kind, line - 1, column, column + len(leaf.value))


def _name_spans(node: NodeOrLeaf, kind: SymbolKind) -> list[SpanSymbol]:
    # Every NAME leaf under a subtree, which is what a dotted decorator or a subscripted
    # annotation (`@a.b`, `dict[str, Any]`) is made of.
    return [_span_of(leaf, kind) for leaf in _walk(node) if isinstance(leaf, Name)]


def _decorator_name_spans(node: NodeOrLeaf) -> list[SpanSymbol]:
    # The decorator's NAME only: `@a.b.c` is three name leaves and `@a(x)` is one, because
    # everything from the call's `trailer` onward is an ordinary expression.
    if not isinstance(node, BaseNode):
        return _name_spans(node, SymbolKind.PY_DECORATOR)
    found: list[SpanSymbol] = []
    for child in node.children:
        if child.type == "trailer" and not _starts_with_dot(child):
            break
        found.extend(_decorator_name_spans(child))
    return found


def _starts_with_dot(node: BaseNode) -> bool:
    first = node.children[0]
    return isinstance(first, Operator) and first.value == "."


def _annotation_spans(node: BaseNode) -> list[SpanSymbol]:
    # `: <annotation>` optionally followed by `= <default>`; the default is a value
    # expression and is not part of the type.
    found: list[SpanSymbol] = []
    for child in node.children[1:]:
        if isinstance(child, Operator) and child.value == "=":
            break
        found.extend(_name_spans(child, SymbolKind.PY_ANNOTATION))
    return found


def _decorator_and_annotation_spans(module: NodeOrLeaf) -> list[SpanSymbol]:
    """The two cases jedi cannot answer (105 D7).

    MEASURED: `get_names` reports a decorator's `@property` and a bare reference to
    `property` identically -- both `type="statement"`, `is_definition()` false, same
    `description` -- and the same for `float` as an annotation versus `float` anywhere else.
    The parso tree distinguishes them structurally: a `Decorator` node, an `annassign` under
    an assignment, a `tfpdef` parameter, a `Function`'s own `annotation`."""
    found: list[SpanSymbol] = []
    for node in _walk(module):
        if isinstance(node, Decorator):
            # `@`, then the dotted name, then optionally a call. The call's `trailer` hangs
            # BELOW the name inside an `atom_expr` rather than beside it, so stopping at a
            # top-level trailer never sees it and a decorator's ARGUMENTS get coloured as
            # though they were the decorator -- which is what `@register(key=helper)` showed.
            for child in node.children[1:]:
                if child.type in ("operator", "newline"):
                    break
                found.extend(_decorator_name_spans(child))
        elif isinstance(node, Function):
            if node.annotation is not None:
                found.extend(_name_spans(node.annotation, SymbolKind.PY_ANNOTATION))
        elif isinstance(node, BaseNode) and node.type in ("annassign", "tfpdef"):
            found.extend(_annotation_spans(node))
    return found


def python_spans(text: str) -> tuple[SpanSymbol, ...]:
    """Every positional distinction a script's colouring needs, from one parso parse.

    MEASURED on the 123-line flock script: 6.63 ms, against 7.46 ms for jedi's `get_names`
    alone -- so all three positional cases cost LESS than the one producer an earlier
    estimate was sized against. `parso.parse` recovers from broken text rather than raising,
    which matters because a buffer mid-edit is normally unparseable: an unclosed paren, a
    dangling `def`, a half-typed annotation and an unterminated string all measured between
    6.26 and 6.74 ms and kept the whole file's structure. A 4x buffer costs 24.9 ms."""
    module = parso.parse(text)
    found = _definition_spans(module)
    found.extend(_decorator_and_annotation_spans(module))
    found.extend(_base_class_spans(module))
    found.extend(_parameter_spans(module))
    return tuple(found)


def _base_class_spans(module: NodeOrLeaf) -> list[SpanSymbol]:
    """The names a `class` inherits from -- a TYPE at a use site, like an annotation.

    Positional by nature: `ScriptBehavior` in the inheritance list and the same name
    anywhere else are the same spelling, so the word table cannot tell them apart.
    """
    found: list[SpanSymbol] = []
    for node in _walk(module):
        if not isinstance(node, Class):
            continue
        for child in node.children:
            if isinstance(child, Operator) and child.value == "(":
                depth = node.children.index(child)
                for base in node.children[depth + 1 :]:
                    if isinstance(base, Operator) and base.value == ")":
                        break
                    found.extend(_name_spans(base, SymbolKind.PY_ANNOTATION))
                break
    return found


def _parameter_name_spans(node: NodeOrLeaf) -> list[SpanSymbol]:
    # `self` and `cls` are the language's own names wherever they appear, INCLUDING in the
    # parameter list -- a span here would otherwise override the word table that already
    # classifies them and colour the first parameter as an ordinary one.
    if isinstance(node, Name) and node.value in _SELF_NAMES:
        return _name_spans(node, SymbolKind.PY_SELF)
    return _name_spans(node, SymbolKind.PY_PARAMETER)


def _parameter_spans(module: NodeOrLeaf) -> list[SpanSymbol]:
    """A function's parameter NAMES -- not its annotations, which `_annotation_spans`
    already claims as types.
    """
    found: list[SpanSymbol] = []
    for node in _walk(module):
        if not isinstance(node, BaseNode) or node.type != "param":
            continue
        for child in node.children:
            if isinstance(child, Operator):
                continue
            if isinstance(child, BaseNode) and child.type == "tfpdef":
                # `name : annotation` -- the name only; the annotation is a type and is
                # claimed by `_annotation_spans`.
                found.extend(_parameter_name_spans(child.children[0]))
                break
            found.extend(_parameter_name_spans(child))
            break
    return found
