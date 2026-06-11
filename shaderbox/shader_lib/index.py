"""Library of reusable GLSL helpers, auto-resolved by identifier.

`set_active(index)` / `active()` hold module-level state for the loaded
`ShaderLibIndex`; `App` rebuilds on mtime change, `Node.compile()` reads `active()`.
Single-process GUI app — no concurrency concern.

A shader calls `SB_perlin_noise_3(...)` directly (no `#include`); on compile the host
scans for `SB_\\w+` idents, intersects with this index, and prepends only the matching
functions plus transitive deps.

`ShaderLibIndex` snapshots every top-level GLSL function AND top-level
`const`/`uniform` declaration in `<shader_lib_root>/**.glsl` (body + signature +
callees/referenced idents + optional `///` doc). Usage pruner lives in
`resolver.py`, regex/brace machinery in `parser.py`. Pure functions, no GL, no
imgui.
"""

from dataclasses import dataclass
from pathlib import Path

from shaderbox.shader_lib import parser
from shaderbox.shader_source import ShaderSource


@dataclass(frozen=True)
class ShaderLibFunction:
    # One spliceable lib entry: a top-level function, or a top-level
    # `const`/`uniform` declaration (its `body` is then the whole declaration,
    # `calls` the initializer's identifiers).
    name: str
    signature: str  # declaration head, e.g. "float SB_hash(vec2 p)" / "uniform vec4 SBT_STROKES[520]"
    body: str  # full entry text: signature + braced body (or `const ... = ...;`)
    file: Path
    line_in_file: int  # 0-based line of the signature
    calls: frozenset[str]  # other identifiers referenced inside the body
    doc: str  # `///` comment block immediately above (may be empty)


@dataclass(frozen=True)
class ShaderLibIndex:
    functions: dict[str, ShaderLibFunction]
    sources: dict[Path, ShaderSource]

    @classmethod
    def empty(cls) -> "ShaderLibIndex":
        return cls(functions={}, sources={})

    @classmethod
    def build(cls, shader_lib_root: Path) -> "ShaderLibIndex":
        # Unreadable files are skipped — a half-finished file must not break compile
        # of every other shader.
        functions: dict[str, ShaderLibFunction] = {}
        sources: dict[Path, ShaderSource] = {}
        if not shader_lib_root.exists():
            return cls(functions=functions, sources=sources)
        for path in sorted(shader_lib_root.glob("**/*.glsl")):
            if not is_shader_lib_path(path, shader_lib_root):
                continue
            try:
                source = ShaderSource.load(path)
            except OSError:
                continue
            sources[path] = source
            for fn in _extract_functions(source):
                # Duplicate name: first wins (sorted walk → deterministic).
                if fn.name not in functions:
                    functions[fn.name] = fn
        return cls(functions=functions, sources=sources)


def is_shader_lib_path(path: Path, shader_lib_root: Path) -> bool:
    # Exclude paths under dot-dirs (e.g. `.trash/`). The build and the mtime
    # watcher's glob MUST share this filter, else the cache compare loops rebuilds.
    try:
        rel = path.relative_to(shader_lib_root)
    except ValueError:
        return False
    return not any(part.startswith(".") for part in rel.parts)


_ACTIVE_INDEX: ShaderLibIndex = ShaderLibIndex.empty()


def set_active(index: ShaderLibIndex) -> None:
    global _ACTIVE_INDEX
    _ACTIVE_INDEX = index


def active() -> ShaderLibIndex:
    return _ACTIVE_INDEX


def _extract_functions(source: ShaderSource) -> list[ShaderLibFunction]:
    # Brace-match function bodies manually — regex can't handle nested braces.
    # Also extracts top-level `const`/`uniform` declarations (shareable lib
    # constants + engine-bound tables like the glyph strokes). Only depth-0
    # lines are declaration candidates: a matched function's body is jumped
    # past wholesale; an unmatched opening brace (Allman-style signature, a
    # half-edited body) is depth-counted instead, so its interior is never
    # indexed — the function itself just stays unindexed.
    stripped = parser.strip_comments(source.text)
    raw_lines = source.text.splitlines()
    stripped_lines = stripped.splitlines()
    functions: list[ShaderLibFunction] = []
    i = 0
    depth = 0
    while i < len(stripped_lines):
        line = stripped_lines[i]
        if depth > 0:
            depth = parser.advance_brace_depth(depth, line)
            i += 1
            continue
        match = parser.FN_SIG_RE.match(line)
        if match is not None:
            type_, name, args = match.group(1), match.group(2), match.group(3)
            signature = f"{type_.strip()} {name}({args.strip()})"
            end = parser.find_body_end(stripped_lines, i)
        else:
            dmatch = parser.DECL_SIG_RE.match(line)
            if dmatch is None:
                depth = parser.advance_brace_depth(depth, line)
                i += 1
                continue
            kind, type_, name, arr = dmatch.groups()
            signature = f"{kind} {type_.strip()} {name}{arr or ''}"
            end = parser.find_decl_end(stripped_lines, i)
        if end is None:
            depth = parser.advance_brace_depth(depth, line)
            i += 1
            continue
        # Body keeps original (un-stripped) lines.
        body = "\n".join(raw_lines[i : end + 1])
        # Callees over-collect (keywords, types); the lib-index intersection at
        # compile time is the real filter.
        body_idents = set(
            parser.IDENT_RE.findall("\n".join(stripped_lines[i : end + 1]))
        )
        body_idents.discard(name)
        # Doc-comment block: contiguous `///` lines immediately above.
        doc = parser.extract_doc(raw_lines, i)
        functions.append(
            ShaderLibFunction(
                name=name,
                signature=signature,
                body=body,
                file=source.path,
                line_in_file=i,
                calls=frozenset(body_idents),
                doc=doc,
            )
        )
        i = end + 1
    return functions
