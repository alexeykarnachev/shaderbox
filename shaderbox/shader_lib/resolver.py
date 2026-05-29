"""Per-compile usage pruner — flatten the lib functions a shader actually uses.

`resolve_usage(root, index)` scans the user's text for `SB_*` identifiers, takes
the transitive closure over the lib call graph, topo-sorts it, and splices the
needed function bodies into a preamble after the shader's `#version`/`#extension`
header — emitting `#line N M` directives so driver errors remap back to source.

Pure functions, no GL, no imgui. `Node.compile()` is the single integration site.
"""

from dataclasses import dataclass, field
from pathlib import Path

from shaderbox.shader_errors import SourceMap
from shaderbox.shader_lib import parser
from shaderbox.shader_lib.index import ShaderLibIndex
from shaderbox.shader_source import ShaderSource


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
    root: ShaderSource, index: ShaderLibIndex
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
    header_end, header_lines, body_lines = parser.split_root_header(root.text)
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


def _intern(state: _ResolveState, source: ShaderSource) -> int:
    if source.path in state.path_to_file_id:
        return state.path_to_file_id[source.path]
    new_id = len(state.file_id_to_path)
    state.file_id_to_path[new_id] = source.path
    state.path_to_file_id[source.path] = new_id
    state.sources.append(source)
    return new_id


def _collect_used_lib_names(root: ShaderSource, index: ShaderLibIndex) -> set[str]:
    # Tokens the user actually references, minus tokens they themselves define.
    stripped = parser.strip_comments(root.text)
    referenced = set(parser.SB_IDENT_RE.findall(stripped))
    user_defined = set(parser.USER_FN_DEF_RE.findall(stripped))
    return {n for n in referenced if n not in user_defined and n in index.functions}


def _topo_sort(
    roots: set[str], index: ShaderLibIndex, errors: list[ResolveError]
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
