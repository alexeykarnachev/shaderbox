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
    # A function, constant or define the buffer itself declares, or a plain word in it.
    BUFFER_SYMBOL = auto()
    PY_KEYWORD = auto()
    PY_BUILTIN = auto()
    # The script API the engine injects: `Ctx`, `ScriptBehavior`, `Vec3`, ...
    PY_API = auto()
    # A member reached through a dot: `ctx.t`, `math.sin`, `self.phase`.
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
