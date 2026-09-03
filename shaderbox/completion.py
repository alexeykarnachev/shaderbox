"""Host-side completion policy (073 W-B): WHAT the code panel offers and WHEN.

The editor library owns the popup and the word prefix; the host owns the vocabulary and the
decision to offer (`docs/embedding.md` in the editor repo: pushing IS opening). This module is
the decision, GL-free: a table of providers, each a context predicate on the line before the
caret plus a candidate list, evaluated in order, the first that fires is offered. `K`'s lookup
reads the same tables the other way, word -> what it is.
"""

import keyword
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from shaderbox.help_content import ENGINE_UNIFORM_DOCS
from shaderbox.shader_lib.index import ShaderLibFunction

MAX_CANDIDATES = 50

# GLSL completion seeds beyond the live lib index + uniforms: the keywords and
# builtins the lexer knows are a fine floor for a fragment shader.
_GLSL_WORDS: tuple[str, ...] = (
    "attribute",
    "bool",
    "break",
    "const",
    "continue",
    "discard",
    "else",
    "float",
    "for",
    "highp",
    "if",
    "in",
    "int",
    "ivec2",
    "ivec3",
    "ivec4",
    "lowp",
    "mat2",
    "mat3",
    "mat4",
    "mediump",
    "out",
    "return",
    "sampler2D",
    "uniform",
    "uint",
    "varying",
    "vec2",
    "vec3",
    "vec4",
    "void",
    "while",
    "abs",
    "ceil",
    "clamp",
    "cos",
    "cross",
    "distance",
    "dot",
    "exp",
    "floor",
    "fract",
    "length",
    "max",
    "min",
    "mix",
    "mod",
    "normalize",
    "pow",
    "reflect",
    "sin",
    "smoothstep",
    "sqrt",
    "step",
    "tan",
    "texture",
)


@dataclass(frozen=True)
class CompletionContext:
    tab_kind: str
    # The caret's line up to the caret; what a provider's context predicate reads.
    line_before_caret: str
    # The library's word prefix at the caret (identifier characters only).
    prefix: str
    lib_functions: tuple[str, ...]
    pass_uniforms: tuple[str, ...]
    # Ctrl+N / Ctrl+P, as opposed to a keystroke.
    explicit: bool


@dataclass(frozen=True)
class CompletionProvider:
    name: str
    tab_kinds: frozenset[str]
    # Anchored at the caret; None fires on any line.
    context: re.Pattern[str] | None
    # The shortest prefix that opens the popup by itself, and on Ctrl+N.
    min_prefix_auto: int
    min_prefix_explicit: int
    candidates: Callable[[CompletionContext], list[str]]


def builtin_uniform_declarations() -> list[str]:
    return [f"{glsl_type} {name};" for name, (glsl_type, _doc) in ENGINE_UNIFORM_DOCS.items()]


def _glsl_words(context: CompletionContext) -> list[str]:
    return [*context.lib_functions, *context.pass_uniforms, *_GLSL_WORDS]


def _python_words(_context: CompletionContext) -> list[str]:
    return list(keyword.kwlist)


PROVIDERS: tuple[CompletionProvider, ...] = (
    # `uniform ` wants a builtin: the whole declaration lands, type and name, so the user
    # never retypes what the engine already knows.
    CompletionProvider(
        name="builtin uniforms",
        tab_kinds=frozenset({"shader"}),
        context=re.compile(r"\buniform\s+\w*$"),
        min_prefix_auto=0,
        min_prefix_explicit=0,
        candidates=lambda _context: builtin_uniform_declarations(),
    ),
    CompletionProvider(
        name="glsl",
        tab_kinds=frozenset({"shader", "lib"}),
        context=None,
        min_prefix_auto=2,
        min_prefix_explicit=1,
        candidates=_glsl_words,
    ),
    CompletionProvider(
        name="python",
        tab_kinds=frozenset({"script"}),
        context=None,
        min_prefix_auto=2,
        min_prefix_explicit=1,
        candidates=_python_words,
    ),
)


def matches(candidate: str, prefix: str) -> bool:
    """A candidate is offered when it extends the prefix; a multi-word candidate (a
    declaration) also when any of its words does, so `u_ti` finds `float u_time;`."""
    if candidate == prefix:
        return False
    if candidate.startswith(prefix):
        return True
    return " " in candidate and any(
        word.startswith(prefix) for word in candidate.replace(";", "").split()
    )


def choose_provider(context: CompletionContext) -> CompletionProvider | None:
    for provider in PROVIDERS:
        if context.tab_kind not in provider.tab_kinds:
            continue
        if provider.context is not None and not provider.context.search(
            context.line_before_caret
        ):
            continue
        floor = (
            provider.min_prefix_explicit if context.explicit else provider.min_prefix_auto
        )
        if len(context.prefix) < floor:
            continue
        return provider
    return None


def offer(context: CompletionContext) -> list[str]:
    """The candidates to push, in order; empty means cancel."""
    provider = choose_provider(context)
    if provider is None:
        return []
    found = [c for c in provider.candidates(context) if matches(c, context.prefix)]
    return found[:MAX_CANDIDATES]


def symbol_doc(
    word: str, lib_functions: Mapping[str, ShaderLibFunction]
) -> tuple[str, str] | None:
    """What `K` shows for the word under the caret: (signature or typed declaration, doc),
    or None when nothing in the repo documents it."""
    function = lib_functions.get(word)
    if function is not None:
        return function.signature, function.doc
    builtin = ENGINE_UNIFORM_DOCS.get(word)
    if builtin is not None:
        glsl_type, doc = builtin
        return f"uniform {glsl_type} {word};", doc
    return None
