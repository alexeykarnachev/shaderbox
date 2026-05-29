"""Library of reusable GLSL helpers, auto-resolved by identifier.

Active-index accessor:

- `set_active(index)` / `active()` — module-level mutable state for the currently
  loaded ShaderLibIndex. `App` owns the rebuild lifecycle (mtime fan-out triggers
  `set_active` with a fresh index). `Node.compile()` reads `active()` at compile
  time. Single-process, GUI app, no concurrency concern.

A user's shader can call `SB_perlin_noise_3(...)` directly — no `#include`. On
compile, the host scans the user's text for `SB_\\w+` identifiers, intersects them
with this index, and prepends ONLY the matching functions (plus transitive deps)
to the source handed to the driver. Lib functions feel like built-ins.

`ShaderLibIndex` is a snapshot of every top-level GLSL function in
`<shader_lib_root>/**.glsl`, with its body + signature + callees + optional `///`
docstring. Rebuilt when any lib file's mtime changes. The per-compile usage pruner
lives in `resolver.py`; the regex/brace machinery in `parser.py`.

Pure functions, no GL, no imgui. `Node.compile()` is the single integration site.
"""

from dataclasses import dataclass
from pathlib import Path

from shaderbox.shader_lib import parser
from shaderbox.shader_source import ShaderSource


@dataclass(frozen=True)
class ShaderLibFunction:
    name: str
    signature: str  # full declaration line, e.g. "float SB_hash(vec2 p)"
    body: str  # full function text: signature + braced body
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
        # Walk every .glsl under shader_lib_root and extract top-level function defs.
        # Files that fail to read are silently skipped — the user authoring a
        # half-finished file shouldn't break compile of every other shader.
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
                # Two lib files defining the same name: first one wins (sorted
                # walk → deterministic). A real collision is a lib-author bug;
                # we don't try to be clever.
                if fn.name not in functions:
                    functions[fn.name] = fn
        return cls(functions=functions, sources=sources)


def is_shader_lib_path(path: Path, shader_lib_root: Path) -> bool:
    # Exclude any path under a dot-prefixed dir (e.g. `.trash/`). `Path.glob`
    # walks into dot-dirs by default; both the ShaderLibIndex build and the mtime
    # watcher's independent glob walk MUST apply this same filter or the
    # current/cached dict comparison loops rebuilds forever on trashed files.
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
    # Find every top-level function definition. Brace-match the body manually
    # because regex can't handle nested braces.
    stripped = parser.strip_comments(source.text)
    raw_lines = source.text.splitlines()
    stripped_lines = stripped.splitlines()
    functions: list[ShaderLibFunction] = []
    i = 0
    while i < len(stripped_lines):
        line = stripped_lines[i]
        match = parser.FN_SIG_RE.match(line)
        if match is None:
            i += 1
            continue
        type_, name, args = match.group(1), match.group(2), match.group(3)
        signature = f"{type_.strip()} {name}({args.strip()})"
        # Find the closing `}` by brace counting, starting from this line.
        body_end = parser.find_body_end(stripped_lines, i)
        if body_end is None:
            i += 1
            continue
        # Body = original (un-stripped) lines from i to body_end inclusive.
        body = "\n".join(raw_lines[i : body_end + 1])
        # Callees = identifiers inside the body, MINUS the function's own name +
        # GLSL keywords + common type names. We don't need a perfect filter; the
        # transitive close at compile time intersects with the lib index, which
        # is the real filter.
        body_idents = set(
            parser.IDENT_RE.findall("\n".join(stripped_lines[i : body_end + 1]))
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
        i = body_end + 1
    return functions
