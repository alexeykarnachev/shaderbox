"""The GLSL index: every symbol a shader or lib buffer can mean, built from explicit inputs
and read three ways -- completion candidates, `K`, the classes that color the text.

The buffer is the source of truth for what is declared: a uniform the body never reads is
still declared here (finding 6 of 078). A sampler's class follows its VALUE joined to the
graph's pass names through `wired_pass`, never the compiled program's sampler set.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from shaderbox.glsl_docs import BUILTINS, KEYWORDS, TYPES
from shaderbox.intel.glsl import (
    buffer_declarations,
    buffer_words,
    uniform_declarations,
)
from shaderbox.intel.script import ScriptReturn
from shaderbox.intel.symbols import Symbol, SymbolKind
from shaderbox.pass_graph import AutoSource, wired_pass

FEEDBACK_SAMPLER = "u_prev"


@dataclass(frozen=True)
class GlslContext:
    """Everything the index is a function of. GL-free values only: text, names, tables."""

    text: str
    # Engine uniform name -> GLSL type, and -> doc.
    engine_types: Mapping[str, str]
    engine_docs: Mapping[str, str]
    # Library function name -> (signature, doc).
    lib_functions: Mapping[str, tuple[str, str]]
    script_returns: tuple[ScriptReturn, ...] = ()
    # The pass this buffer is; None for a lib file (no samplers, no script scope).
    pass_name: str | None = None
    # Every pass of the document, for the wirable samplers and the wiring rule.
    passes: tuple[str, ...] = ()
    # Declared sampler name -> its value (`AutoSource` / `NoSource` / `PassSource` / a
    # texture); a sampler absent here is a fresh declaration, which reads as `AutoSource`.
    sampler_values: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class GlslIndex:
    # Name -> symbol; the document's own names shadow the language's.
    symbols: Mapping[str, Symbol]
    # What a `uniform ...` line offers: whole declarations the buffer lacks (their
    # `insert_text`); the same names in `words` and `symbols` insert as bare names.
    declarations: tuple[Symbol, ...]
    # Identifier completion, in offer order: the document's names first, the language's last.
    words: tuple[Symbol, ...]

    def lookup(self, word: str) -> Symbol | None:
        return self.symbols.get(word)

    def classes(self) -> dict[str, SymbolKind]:
        """The names the text colors by kind (the host classes the lexer cannot know)."""
        return {
            name: symbol.kind
            for name, symbol in self.symbols.items()
            if symbol.kind
            in (
                SymbolKind.ENGINE_UNIFORM,
                SymbolKind.SCRIPT_UNIFORM,
                SymbolKind.PASS_SAMPLER,
                SymbolKind.LIB_FUNCTION,
            )
        }


def _language_symbols() -> list[Symbol]:
    found: list[Symbol] = []
    for name, (signatures, purpose) in sorted(BUILTINS.items()):
        found.append(
            Symbol(
                name,
                SymbolKind.GLSL_BUILTIN,
                signature="\n".join(signatures),
                doc=purpose,
            )
        )
    found.extend(
        Symbol(t, SymbolKind.GLSL_TYPE, signature=t, doc="GLSL type") for t in TYPES
    )
    found.extend(
        Symbol(k, SymbolKind.GLSL_KEYWORD, signature=k, doc="GLSL keyword")
        for k in KEYWORDS
    )
    return found


def build_glsl_index(context: GlslContext) -> GlslIndex:
    declared = uniform_declarations(context.text)
    declared_names = {u.name for u in declared}
    returned = {
        r.name: r
        for r in context.script_returns
        if r.pass_name is None or r.pass_name == context.pass_name
    }
    document: list[Symbol] = []
    declarations: list[Symbol] = []

    for uniform in declared:
        if uniform.name in context.engine_types:
            kind = SymbolKind.ENGINE_UNIFORM
            doc = context.engine_docs.get(uniform.name, "")
        elif (
            uniform.glsl_type == "sampler2D"
            and context.pass_name is not None
            and (
                source := wired_pass(
                    context.sampler_values.get(uniform.name, AutoSource()),
                    uniform.name,
                    context.pass_name,
                    context.passes,
                )
            )
            is not None
        ):
            kind = SymbolKind.PASS_SAMPLER
            doc = (
                "this pass's previous frame"
                if source == context.pass_name
                else f"reads pass {source}"
            )
        elif uniform.name in returned:
            kind = SymbolKind.SCRIPT_UNIFORM
            doc = "driven by the script"
        else:
            kind = SymbolKind.PASS_UNIFORM
            doc = ""
        document.append(
            Symbol(uniform.name, kind, signature=uniform.declaration, doc=doc)
        )

    for name, glsl_type in context.engine_types.items():
        if name not in declared_names:
            declaration = f"uniform {glsl_type} {name};"
            symbol = Symbol(
                name,
                SymbolKind.ENGINE_UNIFORM,
                signature=declaration,
                doc=context.engine_docs.get(name, ""),
                insert_text=declaration,
            )
            document.append(replace(symbol, insert_text=""))
            declarations.append(symbol)

    for name, ret in returned.items():
        if name in declared_names:
            continue
        doc = "returned by the script"
        if ret.glsl_type is None:
            document.append(Symbol(name, SymbolKind.SCRIPT_UNIFORM, name, doc))
            continue
        declaration = f"uniform {ret.glsl_type} {name};"
        symbol = Symbol(
            name,
            SymbolKind.SCRIPT_UNIFORM,
            signature=declaration,
            doc=doc,
            insert_text=declaration,
        )
        document.append(replace(symbol, insert_text=""))
        declarations.append(symbol)

    if context.pass_name is not None:
        wirable = [
            (f"u_{other}", f"reads pass {other}")
            for other in context.passes
            if other != context.pass_name
        ]
        wirable.append((FEEDBACK_SAMPLER, "this pass's previous frame"))
        for name, doc in wirable:
            if name in declared_names:
                continue
            declaration = f"uniform sampler2D {name};"
            symbol = Symbol(
                name,
                SymbolKind.WIRABLE_SAMPLER,
                signature=declaration,
                doc=doc,
                insert_text=declaration,
            )
            document.append(replace(symbol, insert_text=""))
            declarations.append(symbol)

    for decl in buffer_declarations(context.text):
        document.append(
            Symbol(decl.name, SymbolKind.BUFFER_SYMBOL, signature=decl.signature)
        )

    library = [
        Symbol(name, SymbolKind.LIB_FUNCTION, signature=signature, doc=doc)
        for name, (signature, doc) in context.lib_functions.items()
    ]
    language = _language_symbols()

    symbols: dict[str, Symbol] = {}
    for symbol in [*document, *library, *language]:
        symbols.setdefault(symbol.name, symbol)
    plain = [
        Symbol(word, SymbolKind.BUFFER_SYMBOL)
        for word in sorted(buffer_words(context.text))
        if word not in symbols and len(word) > 1
    ]
    for symbol in plain:
        symbols[symbol.name] = symbol

    words = tuple(
        symbols[name]
        for name in dict.fromkeys(
            [*(s.name for s in document), *(s.name for s in plain)]
            + [s.name for s in library]
            + [s.name for s in language]
        )
    )
    return GlslIndex(symbols=symbols, declarations=tuple(declarations), words=words)
