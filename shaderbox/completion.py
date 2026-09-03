"""Host-side completion policy (073 W-B): WHAT the code panel offers and WHEN.

The editor library owns the popup and the word prefix; the host owns the vocabulary and the
decision to offer (`docs/embedding.md` in the editor repo: pushing IS opening). This module is
the decision, creating no GL context: a table of providers, each a context predicate on the line before the
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


# GLSL builtins the completion vocabulary offers: signature + one line of what it does, so
# `K` and the completion popup can say what a name IS. Not the full ES 3.0 library -- the
# entries here are exactly the callable names in _GLSL_WORDS, pinned by a test.
#
# Nearly all of these are OVERLOADED. `genType` is the spec's own notation for "float, vec2,
# vec3 or vec4, the same one throughout"; where a parameter does NOT follow the others
# (mix's t, clamp's bounds, pow's exponent) the overload set is spelled out, because that is
# the difference a caller actually trips over.
GLSL_BUILTIN_DOCS: dict[str, tuple[str, str]] = {
    "abs": ("genType abs(genType x)", "distance from zero, sign discarded"),
    "ceil": ("genType ceil(genType x)", "up to the nearest integer"),
    "clamp": (
        "genType clamp(genType x, genType lo, genType hi) | (genType x, float lo, float hi)",
        "x held inside [lo, hi]",
    ),
    "cos": ("genType cos(genType angle)", "cosine, angle in radians"),
    "cross": ("vec3 cross(vec3 a, vec3 b)", "vector perpendicular to both"),
    "distance": ("float distance(genType a, genType b)", "length of a - b"),
    "dot": (
        "float dot(genType a, genType b)",
        "sum of the products; 1 when parallel, 0 when perpendicular for unit vectors",
    ),
    "exp": ("genType exp(genType x)", "e raised to x"),
    "floor": ("genType floor(genType x)", "down to the nearest integer"),
    "fract": ("genType fract(genType x)", "the part after the point: x - floor(x)"),
    "length": ("float length(genType x)", "the vector's magnitude"),
    "max": (
        "genType max(genType a, genType b) | (genType a, float b)",
        "the larger of the two",
    ),
    "min": (
        "genType min(genType a, genType b) | (genType a, float b)",
        "the smaller of the two",
    ),
    "mix": (
        "genType mix(genType a, genType b, genType t) | (genType a, genType b, float t)",
        "linear blend: a at t=0, b at t=1; a bvec t selects per component",
    ),
    "mod": (
        "genType mod(genType x, genType y) | (genType x, float y)",
        "remainder, sign follows y",
    ),
    "normalize": ("genType normalize(genType x)", "same direction, length 1"),
    "pow": ("genType pow(genType x, genType y)", "x raised to y; x must be >= 0"),
    "reflect": (
        "genType reflect(genType i, genType n)",
        "i bounced off the surface with unit normal n",
    ),
    "sin": ("genType sin(genType angle)", "sine, angle in radians"),
    "smoothstep": (
        "genType smoothstep(genType e0, genType e1, genType x) | (float e0, float e1, genType x)",
        "0 below e0, 1 above e1, an S-curve between -- step without the hard edge",
    ),
    "sqrt": ("genType sqrt(genType x)", "square root"),
    "step": (
        "genType step(genType edge, genType x) | (float edge, genType x)",
        "0 below edge, 1 at or above",
    ),
    "tan": ("genType tan(genType angle)", "tangent, angle in radians"),
    "texture": (
        "vec4 texture(sampler2D s, vec2 uv [, float bias])",
        "sample s at uv; uv 0..1 spans the image",
    ),
}


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
    return [
        f"{glsl_type} {name};"
        for name, (glsl_type, _doc) in ENGINE_UNIFORM_DOCS.items()
    ]


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


def offer(context: CompletionContext) -> list[str]:
    """The candidates to push: every eligible provider's matches, in table order, without
    repeats; empty means cancel. Concatenation rather than first-wins, so `uniform sam`
    still finds `sampler2D` from the glsl provider after the builtin one found nothing."""
    seen: set[str] = set()
    found: list[str] = []
    for provider in eligible_providers(context):
        for candidate in provider.candidates(context):
            if candidate in seen or not matches(candidate, context.prefix):
                continue
            seen.add(candidate)
            found.append(candidate)
    return found[:MAX_CANDIDATES]


_WORD = re.compile(r"[A-Za-z0-9_]+")


def word_at(line: str, column: int) -> str:
    """The identifier under `column`, or the first one after it on the line (vim's rule for
    the word under the cursor); "" when the line has none there."""
    for found in _WORD.finditer(line):
        if found.end() > column:
            return found.group(0)
    return ""


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
    return GLSL_BUILTIN_DOCS.get(word)


def candidate_doc(
    candidate: str, lib_functions: Mapping[str, ShaderLibFunction]
) -> tuple[str, str] | None:
    """What the popup shows beside the highlighted row. A candidate is either a bare name or
    a whole declaration (`float u_time;`), so the name is taken off the end."""
    word = candidate.rstrip(";").split()[-1] if " " in candidate else candidate
    return symbol_doc(word.rstrip(";"), lib_functions)
