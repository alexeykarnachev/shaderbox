"""Host-side completion policy (073 W-B, re-based on the intel index in 078 W-A): WHAT the code
panel offers and WHEN.

The editor library owns the popup and the word prefix; the host owns the vocabulary and the
decision to offer (`docs/embedding.md` in the editor repo: pushing IS opening). This module is
the decision: a table of providers, each a context predicate on the line before the caret plus
a symbol source on the context, evaluated in order, every eligible provider's matches offered
without repeats. The vocabulary itself is the index (`intel/`): the buffer's declarations, the
engine's uniforms, the script's returns, the other passes, the library, the language -- and,
on a script tab, what the jedi worker answered for this cursor. GL-free; creates no context.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from shaderbox.intel.index import GlslIndex
from shaderbox.intel.symbols import Symbol

MAX_CANDIDATES = 50


@dataclass(frozen=True)
class CompletionContext:
    tab_kind: str
    # The caret's line up to the caret; what a provider's context predicate reads.
    line_before_caret: str
    # The library's word prefix at the caret (identifier characters only).
    prefix: str
    # Ctrl+N / Ctrl+P, as opposed to a keystroke.
    explicit: bool
    # A shader or lib tab's index; None on a script tab.
    index: GlslIndex | None = None
    # A script tab's candidates, as the jedi worker answered for THIS cursor; empty while
    # the answer is still on its way (nothing is offered until it lands).
    python_candidates: tuple[Symbol, ...] = ()


@dataclass(frozen=True)
class CompletionProvider:
    name: str
    tab_kinds: frozenset[str]
    # Anchored at the caret; None fires on any line.
    context: re.Pattern[str] | None
    # The shortest prefix that opens the popup by itself, and on Ctrl+N.
    min_prefix_auto: int
    min_prefix_explicit: int
    candidates: Callable[[CompletionContext], Sequence[Symbol]]


_SITE = re.compile(r"\buniform\s+(?:(\w+)\s+)?(\w*)$")


def _declaration_parts(symbol: Symbol) -> tuple[str, str]:
    # A declaration symbol's insert text is `uniform <type> <name>;`.
    words = symbol.inserted.rstrip(";").split()
    return (words[1], words[2]) if len(words) == 3 else ("", symbol.name)


def _declarations(context: CompletionContext) -> Sequence[Symbol]:
    """Whole declarations shaped to the site: after `uniform ` the type and name; after a
    typed type only the names of that type (a name-only script uniform fits any type)."""
    if context.index is None:
        return ()
    site = _SITE.search(context.line_before_caret)
    typed_type = site.group(1) if site is not None else None
    shaped: list[Symbol] = []
    for symbol in context.index.declarations:
        glsl_type, name = _declaration_parts(symbol)
        if typed_type is None:
            shaped.append(replace(symbol, insert_text=f"{glsl_type} {name};"))
        elif glsl_type in ("", typed_type):
            shaped.append(replace(symbol, insert_text=f"{name};"))
    return shaped


def _glsl_words(context: CompletionContext) -> Sequence[Symbol]:
    # After `uniform <type> ` the line wants a NEW name: the buffer's declared names and the
    # language's words are not candidates there, only the declarations provider's.
    site = _SITE.search(context.line_before_caret)
    if context.index is None or (site is not None and site.group(1)):
        return ()
    return context.index.words


def _python_words(context: CompletionContext) -> Sequence[Symbol]:
    return context.python_candidates


# `uniform`, `uniform u_`, `uniform vec4 u_`, `uniform sampler2D u_`: a declaration site up to
# and including a partial name; not `vec3 color = u_`, and not past the name.
DECLARATION_SITE = re.compile(r"\buniform\s+(?:\w+\s+)?\w*$")
# Where a Python completion makes sense: after a dot (member access, prefix may be empty) or
# inside an identifier.
PYTHON_SITE = re.compile(r"(?:\.\w*|\w+)$")

PROVIDERS: tuple[CompletionProvider, ...] = (
    # A declaration site wants whole declarations the buffer lacks: the engine's uniforms,
    # the script's returns, the samplers another pass could feed.
    CompletionProvider(
        name="declarations",
        tab_kinds=frozenset({"shader"}),
        context=DECLARATION_SITE,
        min_prefix_auto=0,
        min_prefix_explicit=0,
        candidates=_declarations,
    ),
    CompletionProvider(
        name="glsl",
        tab_kinds=frozenset({"shader", "lib"}),
        context=None,
        min_prefix_auto=2,
        min_prefix_explicit=1,
        candidates=_glsl_words,
    ),
    # A member site (`ctx.`, `math.si`) opens by itself with an empty prefix; a bare
    # identifier needs one letter. Nothing is asked of jedi anywhere else on a line.
    CompletionProvider(
        name="python",
        tab_kinds=frozenset({"script"}),
        context=PYTHON_SITE,
        min_prefix_auto=0,
        min_prefix_explicit=0,
        candidates=_python_words,
    ),
)


def matches(candidate: str, prefix: str) -> bool:
    """A candidate is offered when it extends the prefix; a multi-word candidate (a
    declaration) also when any of its words does, so `u_ti` finds `uniform float u_time;`."""
    if candidate == prefix:
        return False
    if candidate.startswith(prefix):
        return True
    return " " in candidate and any(
        word.startswith(prefix) for word in candidate.replace(";", "").split()
    )


_LINE_COMMENT: dict[str, str] = {"shader": "//", "lib": "//", "script": "#"}


def in_line_comment(tab_kind: str, line_before_caret: str) -> bool:
    marker = _LINE_COMMENT.get(tab_kind)
    return marker is not None and marker in line_before_caret


def eligible_providers(context: CompletionContext) -> list[CompletionProvider]:
    """The providers whose tab kind, context and prefix floor all hold, in table order."""
    if in_line_comment(context.tab_kind, context.line_before_caret):
        return []
    found: list[CompletionProvider] = []
    for provider in PROVIDERS:
        if context.tab_kind not in provider.tab_kinds:
            continue
        if provider.context is not None and not provider.context.search(
            context.line_before_caret
        ):
            continue
        floor = (
            provider.min_prefix_explicit
            if context.explicit
            else provider.min_prefix_auto
        )
        if len(context.prefix) < floor:
            continue
        found.append(provider)
    return found


def offer(context: CompletionContext) -> list[Symbol]:
    """The candidates to push: every eligible provider's matches, in table order, without
    repeats of the inserted text; empty means cancel."""
    seen: set[str] = set()
    found: list[Symbol] = []
    for provider in eligible_providers(context):
        for symbol in provider.candidates(context):
            text = symbol.inserted
            if text in seen or not matches(text, context.prefix):
                continue
            seen.add(text)
            found.append(symbol)
    return found[:MAX_CANDIDATES]


_WORD = re.compile(r"[A-Za-z0-9_]+")


def word_at(line: str, column: int) -> str:
    """The identifier under `column`, or the first one after it on the line (vim's rule for
    the word under the cursor); "" when the line has none there."""
    for found in _WORD.finditer(line):
        if found.end() > column:
            return found.group(0)
    return ""
