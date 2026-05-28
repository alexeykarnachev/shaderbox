"""Library of reusable GLSL helpers, auto-resolved by identifier.

Active-index accessor:

- `set_active(index)` / `active()` — module-level mutable state for the currently
  loaded LibIndex. `App` owns the rebuild lifecycle (mtime fan-out triggers
  `set_active` with a fresh index). `Node.compile()` reads `active()` at compile
  time. Single-process, GUI app, no concurrency concern.

A user's shader can call `SB_perlin_noise_3(...)` directly — no `#include`. On
compile, the host scans the user's text for `SB_\\w+` identifiers, intersects them
with this index, and prepends ONLY the matching functions (plus transitive deps)
to the source handed to the driver. Lib functions feel like built-ins.

Two artifacts:
- `LibIndex` — a snapshot of every top-level GLSL function in `<lib_root>/**.glsl`,
  with its body + signature + callees + optional `///` docstring. Rebuilt when any
  lib file's mtime changes.
- `resolve_usage(root, index)` — the per-compile usage pruner. Returns the
  flattened text + sources list + `SourceMap` + any resolver errors.

Pure functions, no GL, no imgui. `Node.compile()` is the single integration site.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from shaderbox.shader_source import ShaderSource
from shaderbox.util import SourceMap

# Top-level function signature: `<type> <name>(<args>)` followed by `{`. The type
# permits a trailing `[]` (rare in lib code but cheap to allow). Captures: 1=type,
# 2=name, 3=args. Anchored to start-of-line so we don't pick up function calls
# inside other functions.
_FN_SIG_RE = re.compile(
    r"^\s*(\w+(?:\s*\[\s*\d*\s*\])?)\s+(\w+)\s*\(([^)]*)\)\s*\{",
    re.MULTILINE,
)

# Identifier references inside a function body (used to build the call graph).
_IDENT_RE = re.compile(r"\b([A-Za-z_]\w*)\b")

# A `SB_` identifier — what the user types in their shader to call a lib function.
_SB_IDENT_RE = re.compile(r"\bSB_\w+\b")

# A function definition IN THE USER'S text (so we can suppress the lib version when
# the user shadows it). The regex deliberately matches more loosely than _FN_SIG_RE
# — it just needs to find `<type> SB_foo(`.
_USER_FN_DEF_RE = re.compile(
    r"\b\w+(?:\s*\[\s*\d*\s*\])?\s+(SB_\w+)\s*\(", re.MULTILINE
)


@dataclass(frozen=True)
class LibFunction:
    name: str
    signature: str  # full declaration line, e.g. "float SB_hash(vec2 p)"
    body: str  # full function text: signature + braced body
    file: Path
    line_in_file: int  # 0-based line of the signature
    calls: frozenset[str]  # other identifiers referenced inside the body
    doc: str  # `///` comment block immediately above (may be empty)


@dataclass(frozen=True)
class LibIndex:
    functions: dict[str, LibFunction]
    sources: dict[Path, ShaderSource]

    @classmethod
    def empty(cls) -> "LibIndex":
        return cls(functions={}, sources={})

    @classmethod
    def build(cls, lib_root: Path) -> "LibIndex":
        # Walk every .glsl under lib_root and extract top-level function defs.
        # Files that fail to read are silently skipped — the user authoring a
        # half-finished file shouldn't break compile of every other shader.
        functions: dict[str, LibFunction] = {}
        sources: dict[Path, ShaderSource] = {}
        if not lib_root.exists():
            return cls(functions=functions, sources=sources)
        for path in sorted(lib_root.glob("**/*.glsl")):
            if not is_lib_path(path, lib_root):
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


def is_lib_path(path: Path, lib_root: Path) -> bool:
    # Exclude any path under a dot-prefixed dir (e.g. `.trash/`). `Path.glob`
    # walks into dot-dirs by default; both the LibIndex build and the mtime
    # watcher's independent glob walk MUST apply this same filter or the
    # current/cached dict comparison loops rebuilds forever on trashed files.
    try:
        rel = path.relative_to(lib_root)
    except ValueError:
        return False
    return not any(part.startswith(".") for part in rel.parts)


_ACTIVE_INDEX: LibIndex = LibIndex.empty()


def set_active(index: LibIndex) -> None:
    global _ACTIVE_INDEX
    _ACTIVE_INDEX = index


def active() -> LibIndex:
    return _ACTIVE_INDEX


@dataclass(frozen=True)
class ResolveError:
    path: Path
    line: int
    message: str


@dataclass
class _ResolveState:
    flattened_lines: list[str] = field(default_factory=list)
    sources: list[ShaderSource] = field(default_factory=list)
    file_id_to_path: dict[int, Path] = field(default_factory=dict)
    path_to_file_id: dict[Path, int] = field(default_factory=dict)
    errors: list[ResolveError] = field(default_factory=list)


def resolve_usage(
    root: ShaderSource, index: LibIndex
) -> tuple[str, list[ShaderSource], SourceMap, list[ResolveError]]:
    """Prepend lib functions used by `root` to its source.

    Returns:
      - `flattened`: text handed to the driver. `root.text` with a preamble of the
        topo-sorted, transitively-closed set of lib functions inserted after any
        `#version`/`#extension` lines.
      - `sources`: every distinct source contributing to `flattened`, in first-seen
        order. `sources[0]` is always `root`.
      - `source_map`: `file_id -> path`, populated for the root + every lib file
        whose content is in the preamble. `parse_shader_errors` uses this to remap
        driver errors back to source files.
      - `errors`: cycle in the lib call graph, missing transitive dep. Rare; they
        surface to the user as synthetic `ShaderError`s in `CompileUnit.errors`.
    """
    state = _ResolveState()
    # Root is always file_id 0.
    _intern(state, root)

    used = _collect_used_lib_names(root, index)
    if not used:
        # Fast path: no lib functions referenced; the flattened text IS the root.
        flattened = root.text
        source_map = SourceMap(file_id_to_path=dict(state.file_id_to_path))
        return flattened, list(state.sources), source_map, list(state.errors)

    ordered = _topo_sort(used, index, state.errors)

    # Build the preamble: each lib function gets its own `#line` marker so errors
    # remap to its (file, line_in_file).
    preamble_lines: list[str] = []
    for name in ordered:
        fn = index.functions[name]
        lib_source = index.sources[fn.file]
        file_id = _intern(state, lib_source)
        # `#line N M` sets the NEXT physical line to logical line N in file M.
        preamble_lines.append(f"#line {fn.line_in_file + 1} {file_id}")
        preamble_lines.append(fn.body)

    # Splice preamble into root.text AFTER `#version`/`#extension` lines.
    header_end, header_lines, body_lines = _split_root_header(root.text)
    out_lines: list[str] = []
    out_lines.extend(header_lines)
    out_lines.extend(preamble_lines)
    # Restore root's line numbering for the body. `header_end` is the 0-based
    # index of the first body line; physical-line N (1-based) should map back to
    # source line (header_end + 1).
    out_lines.append(f"#line {header_end + 1} 0")
    out_lines.extend(body_lines)
    flattened = "\n".join(out_lines)

    source_map = SourceMap(file_id_to_path=dict(state.file_id_to_path))
    return flattened, list(state.sources), source_map, list(state.errors)


# ----------------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------------


def _intern(state: _ResolveState, source: ShaderSource) -> int:
    if source.path in state.path_to_file_id:
        return state.path_to_file_id[source.path]
    new_id = len(state.file_id_to_path)
    state.file_id_to_path[new_id] = source.path
    state.path_to_file_id[source.path] = new_id
    state.sources.append(source)
    return new_id


def _strip_comments(text: str) -> str:
    # GLSL has `//` line comments and `/* ... */` block comments. No strings, no
    # nested block comments. Block-strip is non-greedy + multi-line.
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return text


def _split_root_header(text: str) -> tuple[int, list[str], list[str]]:
    # Return (header_end_index, header_lines, body_lines).
    # Header = any prefix of: blank lines, line-comments (kept verbatim), and
    # `#version` / `#extension` / `#pragma` directives. First line that's not one
    # of those starts the body.
    lines = text.splitlines()
    header_end = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("//"):
            continue
        if stripped.startswith(("#version", "#extension", "#pragma")):
            header_end = i + 1
            continue
        # First non-header line.
        header_end = i
        break
    else:
        # All lines were header.
        header_end = len(lines)
    return header_end, lines[:header_end], lines[header_end:]


def _extract_functions(source: ShaderSource) -> list[LibFunction]:
    # Find every top-level function definition. Brace-match the body manually
    # because regex can't handle nested braces.
    stripped = _strip_comments(source.text)
    raw_lines = source.text.splitlines()
    stripped_lines = stripped.splitlines()
    functions: list[LibFunction] = []
    i = 0
    while i < len(stripped_lines):
        line = stripped_lines[i]
        match = _FN_SIG_RE.match(line)
        if match is None:
            i += 1
            continue
        type_, name, args = match.group(1), match.group(2), match.group(3)
        signature = f"{type_.strip()} {name}({args.strip()})"
        # Find the closing `}` by brace counting, starting from this line.
        body_end = _find_body_end(stripped_lines, i)
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
            _IDENT_RE.findall("\n".join(stripped_lines[i : body_end + 1]))
        )
        body_idents.discard(name)
        # Doc-comment block: contiguous `///` lines immediately above.
        doc = _extract_doc(raw_lines, i)
        functions.append(
            LibFunction(
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


def _find_body_end(lines: list[str], start: int) -> int | None:
    # Starting at `start` (line with the function signature + opening `{`), walk
    # forward counting braces; return the 0-based line index containing the
    # matching `}`.
    depth = 0
    in_string = False  # GLSL has no strings, but keep the toggle harmless
    for i in range(start, len(lines)):
        for ch in lines[i]:
            if ch == "{" and not in_string:
                depth += 1
            elif ch == "}" and not in_string:
                depth -= 1
                if depth == 0:
                    return i
    return None


def _extract_doc(lines: list[str], sig_line: int) -> str:
    # Walk backwards from `sig_line - 1` collecting contiguous `///` lines.
    doc_lines: list[str] = []
    j = sig_line - 1
    while j >= 0:
        stripped = lines[j].lstrip()
        if stripped.startswith("///"):
            doc_lines.append(stripped[3:].strip())
            j -= 1
        else:
            break
    doc_lines.reverse()
    return "\n".join(doc_lines).strip()


def _collect_used_lib_names(root: ShaderSource, index: LibIndex) -> set[str]:
    # Tokens the user actually references, minus tokens they themselves define.
    stripped = _strip_comments(root.text)
    referenced = set(_SB_IDENT_RE.findall(stripped))
    user_defined = set(_USER_FN_DEF_RE.findall(stripped))
    return {n for n in referenced if n not in user_defined and n in index.functions}


def _topo_sort(
    roots: set[str], index: LibIndex, errors: list[ResolveError]
) -> list[str]:
    # Iterative DFS post-order over the call graph rooted at `roots`. Functions
    # are emitted in dependency order (callees before callers).
    visited: set[str] = set()
    on_stack: set[str] = set()  # for cycle detection
    order: list[str] = []

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in on_stack:
            fn = index.functions[name]
            errors.append(
                ResolveError(
                    fn.file,
                    fn.line_in_file,
                    f"library cycle involving '{name}'",
                )
            )
            return
        on_stack.add(name)
        fn = index.functions.get(name)
        if fn is None:
            return  # Not in lib — silently skipped (user-defined or typo; driver will complain)
        # Visit each callee that's also in the lib.
        for callee in fn.calls:
            if callee in index.functions:
                visit(callee)
        on_stack.discard(name)
        visited.add(name)
        order.append(name)

    for name in sorted(roots):  # sorted for deterministic output
        visit(name)
    return order
