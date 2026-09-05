"""The vocabulary every reader of the index shares: what a name IS, and what is said about it."""

from dataclasses import dataclass
from enum import StrEnum, auto


class SymbolKind(StrEnum):
    """Every kind of name the editor can know. Its color and its syntax slot are functions of
    this enum (`theme.kind_color`, `theme.kind_slot`), walked by the enum-domain test, so a
    kind added here without both fails before a frame draws it."""

    GLSL_KEYWORD = auto()
    GLSL_TYPE = auto()
    GLSL_BUILTIN = auto()
    LIB_FUNCTION = auto()
    # An engine-driven uniform (`u_time`, `u_resolution`, ...), declared or not.
    ENGINE_UNIFORM = auto()
    # A uniform the buffer declares that is neither engine-driven nor wired to a pass.
    PASS_UNIFORM = auto()
    # A sampler2D the buffer declares whose live source is another pass.
    PASS_SAMPLER = auto()
    # A sampler2D the buffer could declare to read another pass (`u_<pass>`, `u_prev`).
    WIRABLE_SAMPLER = auto()
    # A uniform the document script returns, declared in this pass or not.
    SCRIPT_UNIFORM = auto()
    # The fragment output the buffer declares (`out vec4 fragColor;`) or `gl_FragColor`: the
    # one name a shader WRITES rather than reads.
    OUTPUT_VARIABLE = auto()
    # A function, constant or define the buffer itself declares, or a plain word in it.
    BUFFER_SYMBOL = auto()
    PY_KEYWORD = auto()
    PY_BUILTIN = auto()
    # The script API the engine injects: `ScriptContext`, `ScriptBehavior`, `Vec3`, ...
    PY_API = auto()
    # A member reached through a dot: `context.t`, `math.sin`, `self.phase`.
    PY_MEMBER = auto()
    # A name the script defines: a class, a function, a variable in scope.
    PY_LOCAL = auto()


@dataclass(frozen=True)
class Symbol:
    name: str
    kind: SymbolKind
    # What `K` shows first: a declaration, a signature, a call form.
    signature: str = ""
    doc: str = ""
    # What completion inserts; the name unless a whole declaration is the useful thing.
    insert_text: str = ""

    @property
    def inserted(self) -> str:
        return self.insert_text or self.name


# The order candidates sort in (079 D2), the document's own vocabulary before the language's:
# the buffer's own names, the engine's uniforms, the script's, the samplers, the library, the
# language. Python's kinds parallel GLSL's — what the script itself defines first, the API next,
# the language last. Names inside a tier sort alphabetically, so a tie here is deliberate. It
# lives here rather than beside `kind_color` / `kind_slot` because a rank is not a color and
# `completion.py` is GL-free — importing `theme` from it pulls imgui into the intel layer.
_KIND_RANK: dict[SymbolKind, int] = {
    SymbolKind.PASS_UNIFORM: 0,
    SymbolKind.BUFFER_SYMBOL: 0,
    SymbolKind.OUTPUT_VARIABLE: 0,
    SymbolKind.ENGINE_UNIFORM: 1,
    SymbolKind.SCRIPT_UNIFORM: 2,
    SymbolKind.PASS_SAMPLER: 3,
    SymbolKind.WIRABLE_SAMPLER: 3,
    SymbolKind.LIB_FUNCTION: 4,
    SymbolKind.GLSL_BUILTIN: 5,
    SymbolKind.GLSL_KEYWORD: 6,
    SymbolKind.GLSL_TYPE: 6,
    SymbolKind.PY_LOCAL: 0,
    SymbolKind.PY_MEMBER: 0,
    SymbolKind.PY_API: 2,
    SymbolKind.PY_BUILTIN: 5,
    SymbolKind.PY_KEYWORD: 6,
}


def kind_rank(kind: SymbolKind) -> int:
    """Where a kind sorts among candidates (079 D2). Lower comes first; ties sort by name."""
    return _KIND_RANK[kind]
