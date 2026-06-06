"""The copilot capability backend — the worker-facing implementation of every
`CopilotCapabilities` method (feature 023, extracted from `app.py`).

`CopilotBackend` owns the node/edit/uniform/render/publish/telegram verbs the copilot
worker calls. It does NOT import `App` (the no-`TYPE_CHECKING` rule): every dependency
is an explicit ref / getter / callback injected by `App._build_copilot_capabilities`,
mirroring `shader_lib/file_ops.py::ShaderLibFileManager`. Project-dependent reads are
getters (re-read every call so a project switch retargets them); the working-set /
batch-mutated state stays on `App` and is reached through accessor callbacks. Every
GL-affine verb marshals to the main thread through `self._bridge.run_on_main`.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeGuard

import moderngl
from loguru import logger
from OpenGL.GL import GL_SAMPLER_2D, GL_UNSIGNED_INT

from shaderbox.copilot.bridge import CopilotBridge
from shaderbox.copilot.capabilities import (
    CompileErrorInfo,
    DeleteNodeResult,
    EditResult,
    GrepHit,
    LibCatalogEntry,
    LibFunctionBody,
    NodeTreeEntry,
    PublishResult,
    RenderResult,
    SetUniformResult,
    ShaderView,
    SwitchNodeResult,
    TelegramConnectResult,
    TelegramOpResult,
    TelegramPackInfo,
    TemplateEntry,
    WorkingSetView,
)
from shaderbox.copilot.config import COPILOT_CONFIG
from shaderbox.copilot.errors import CopilotToolError
from shaderbox.copilot.glsl_lex import span_has_comment, token_match
from shaderbox.copilot.text_render import sanitize_display
from shaderbox.core import Node
from shaderbox.exporters.base import (
    AuthState,
    Exporter,
    ExporterStatus,
    ExportProgress,
)
from shaderbox.exporters.registry import ExporterRegistry
from shaderbox.exporters.telegram import NEEDS_START_ERROR, TelegramExporter
from shaderbox.exporters.youtube import YouTubeExporter
from shaderbox.paths import shader_lib_root
from shaderbox.render_preset import FitPolicy, RenderPreset, ResolutionPolicy
from shaderbox.shader_errors import ShaderError
from shaderbox.shader_lib import ShaderLibIndex
from shaderbox.shader_lib.file_ops import ShaderLibFileManager
from shaderbox.tabs import share_state
from shaderbox.ui_models import UINode, load_node_from_dir
from shaderbox.util import str_to_unicode, try_to_release

# Copilot node-id shortening: the agent sees/passes a short prefix, never the 36-char UUID.
# 4 chars is collision-free for any realistic project (16^4 = 65536 over a handful of nodes);
# _copilot_short_ids grows the prefix only on an actual collision.
_COPILOT_SHORT_ID_LEN = 4
_COPILOT_FULL_ID_LEN = 36


def _to_error_infos(errors: list[ShaderError]) -> list[CompileErrorInfo]:
    # ShaderError.line is 0-based (-1 = unparsed fallback row); the agent reads a cat -n
    # listing, so report 1-based. The fallback row has no real line — report it as 0.
    return [
        CompileErrorInfo(
            path=str(e.path), line=e.line + 1 if e.line >= 0 else 0, message=e.message
        )
        for e in errors
    ]


def _number_lines(text: str) -> str:
    # cat -n style. The prefixes orient the agent but are NOT part of the text it edits
    # against (it matches on content) — spec §16.2.
    lines = text.split("\n")
    width = len(str(len(lines)))
    return "\n".join(f"{i:>{width}}  {line}" for i, line in enumerate(lines, start=1))


@dataclass
class _CopilotEditTarget:
    # A resolved copilot edit target (feature 020·16): either a NODE (recompiles, returns
    # compile errors) or a LIB file (written, no standalone compile). `source` is the current
    # text the edit matches/line-edits against ("" for a not-yet-created lib file). `ws_address` is
    # the working-set + per-batch-guard key (the node full-id, or the "lib:" address — feature 020·29).
    kind: str  # "node" | "lib"
    source: str
    ws_address: str
    node_id: str | None = None
    node: "Node | None" = None
    lib_path: Path | None = None
    lib_create: bool = False


def _ws_normalize(text: str) -> tuple[str, list[int]]:
    # Collapse every run of horizontal whitespace to a single space and drop spaces
    # adjacent to a newline, returning the normalized text plus a parallel map from each
    # normalized-char index to its originating index in `text`. The map lets a match in
    # normalized space be sliced back to the EXACT original bytes (feature 020 · 12 A).
    out: list[str] = []
    src_index: list[int] = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c in " \t":
            j = i
            while j < n and text[j] in " \t":
                j += 1
            # Emit a single space only when it sits between two non-newline chars; a run
            # touching a newline collapses away (indentation / trailing space).
            prev = out[-1] if out else "\n"
            nxt = text[j] if j < n else "\n"
            if prev != "\n" and nxt != "\n":
                out.append(" ")
                src_index.append(i)
            i = j
        else:
            out.append(c)
            src_index.append(i)
            i += 1
    src_index.append(n)  # sentinel: end of the last char maps past the source end
    return "".join(out), src_index


def _whitespace_near_match(src: str, old_str: str) -> str:
    # When old_str doesn't match src exactly, find the UNIQUE region of src that matches
    # it ignoring whitespace, and return that region's EXACT original bytes (so the model
    # copies them verbatim). "" when there is no match or it's not unique.
    norm_src, src_index = _ws_normalize(src)
    norm_old, _ = _ws_normalize(old_str)
    if not norm_old:
        return ""
    first = norm_src.find(norm_old)
    if first == -1 or norm_src.find(norm_old, first + 1) != -1:
        return ""  # no match, or ambiguous — no safe single hint
    return src[src_index[first] : src_index[first + len(norm_old)]]


def _splice(src: str, spans: list[tuple[int, int]], new_str: str) -> str:
    # Replace each non-overlapping (start, end) span (source order, from token_match) with
    # new_str verbatim. Cursor walk is offset-stable because the spans never overlap.
    out: list[str] = []
    cursor: int = 0
    for start, end in spans:
        out.append(src[cursor:start])
        out.append(new_str)
        cursor = end
    out.append(src[cursor:])
    return "".join(out)


def _uniform_type_label(u: moderngl.Uniform | moderngl.UniformBlock) -> str:
    if isinstance(u, moderngl.UniformBlock):
        return "block"
    # moderngl stub gap: Uniform.gl_type (conventions.md sanctioned allowlist).
    gl_type = u.gl_type  # type: ignore
    # A sampler is a live Uniform (Node.render binds it) but is NOT a settable scalar/vector — label it
    # honestly so the agent's type oracle and set_uniform's reject agree (020·16 Decision 6).
    if gl_type == GL_SAMPLER_2D:
        return "sampler2D"
    base = "uint" if gl_type == GL_UNSIGNED_INT else "float"
    scalar = base if u.dimension == 1 else f"vec{u.dimension}"
    return f"{scalar}[{u.array_length}]" if u.array_length > 1 else scalar


def _is_number(v: object) -> TypeGuard[int | float]:
    return isinstance(v, int | float) and not isinstance(v, bool)


def _coerce_uniform_value(
    value: object, uniform: moderngl.Uniform
) -> float | int | list[int] | tuple[float, ...] | list[tuple[float, ...]] | None:
    # Coerce a JSON-decoded value to the EXACT shape moderngl's Uniform write wants (020·16 / 020·23).
    # The shapes are probe-pinned: a scalar -> number; vecN -> a tuple of N; a dim==1 array -> a FLAT
    # list of array_length; a dim>1 array (vecN[M]) -> array_length NESTED dimension-tuples. Returns
    # None on any mismatch (the handler turns that into an explicit error — never the silent render pop).
    # Bools are rejected (JSON true is not a shader number).
    dim = uniform.dimension
    n = uniform.array_length
    if n > 1:
        return _coerce_array(value, uniform, dim, n)
    if dim == 1:
        return value if _is_number(value) else None
    if not isinstance(value, list | tuple) or len(value) != dim:
        return None
    if not all(_is_number(v) for v in value):
        return None
    return tuple(float(v) for v in value)


def _coerce_array(
    value: object, uniform: moderngl.Uniform, dim: int, n: int
) -> list[int] | tuple[float, ...] | list[tuple[float, ...]] | None:
    # A uint TEXT array (uint[N]): accept a str (-> codepoints, the UI's str_to_unicode) OR a list of
    # ints; truncate/null-pad to exactly N (a string genuinely null-terminates). A NUMERIC array: exact
    # length, NO padding (padding numeric data is silent corruption; moderngl raises on a short write).
    gl_type = uniform.gl_type  # type: ignore
    if dim == 1 and gl_type == GL_UNSIGNED_INT:
        if isinstance(value, str):
            return str_to_unicode(value, n)
        if isinstance(value, list | tuple) and all(_is_number(v) for v in value):
            ints = [int(v) for v in value][:n]
            return ints + [0] * (n - len(ints))
        return None
    # numeric array. A str is never valid here.
    if not isinstance(value, list | tuple) or not all(_is_number(v) for v in value):
        return None
    if dim == 1:  # float[N] -> flat list of exactly N
        return tuple(float(v) for v in value) if len(value) == n else None
    if len(value) != n * dim:  # vecN[M] -> N rows of `dim`
        return None
    flat = [float(v) for v in value]
    return [tuple(flat[i : i + dim]) for i in range(0, n * dim, dim)]


def _set_uniform_shape_hint(name: str, uniform: moderngl.Uniform, label: str) -> str:
    # The single shape-mismatch feedback channel (020·23): teach the EXACT shape for what `name` is.
    dim = uniform.dimension
    n = uniform.array_length
    gl_type = uniform.gl_type  # type: ignore
    if n > 1 and dim == 1 and gl_type == GL_UNSIGNED_INT:
        return (
            f"value does not match {label} (a text array) — pass the text as a string e.g. "
            f'"Hello\\nWorld", or a list of up to {n} codepoint ints'
        )
    if n > 1 and dim == 1:
        return f"value does not match {label} — provide a list of exactly {n} numbers"
    if n > 1:
        return (
            f"value does not match {label} — provide a list of {n * dim} numbers "
            f"({n} groups of {dim})"
        )
    if dim > 1:
        return f"value does not match {label} — provide a list of {dim} numbers for a vector"
    return f"value does not match {label} — provide a number"


def _format_uniforms(
    uniforms: list[moderngl.Uniform | moderngl.UniformBlock],
) -> list[str]:
    # "name type = value" rows for the agent's orientation. Blocks have no scalar value.
    rows: list[str] = []
    for u in uniforms:
        label = _uniform_type_label(u)
        if isinstance(u, moderngl.UniformBlock):
            rows.append(f"{u.name} {label}")
        else:
            rows.append(f"{u.name} {label} = {u.value}")
    return rows


# Engine-driven uniforms: Node.render() overwrites these every frame from the engine clock /
# canvas regardless of uniform_values, so set_uniform on them is a per-frame no-op (020·16 Decision 6).
_ENGINE_DRIVEN_UNIFORMS: frozenset[str] = frozenset(
    {"u_time", "u_aspect", "u_resolution"}
)


class CopilotBackend:
    def __init__(
        self,
        *,
        get_bridge: Callable[[], CopilotBridge],
        node_templates_dir: Path,
        starter_template_id: str,
        get_renders_dir: Callable[[], Path],
        get_ui_nodes: Callable[[], dict[str, UINode]],
        get_ui_node_templates: Callable[[], dict[str, UINode]],
        get_exporter_registry: Callable[[], ExporterRegistry],
        get_shader_lib_index: Callable[[], ShaderLibIndex],
        get_shader_lib_files: Callable[[], ShaderLibFileManager],
        get_current_node_id: Callable[[], str],
        get_is_cancelled: Callable[[], bool],
        set_current_node_id: Callable[[str], None],
        save_ui_node: Callable[[UINode], object],
        sync_editor_from_disk: Callable[[str, str], None],
        delete_node_unguarded: Callable[[str], str],
        template_description: Callable[[str], str],
        working_set_reader: Callable[[], list[str]],
        working_set_add: Callable[[str], None],
        batch_mutated_reader: Callable[[], set[str]],
        batch_mutated_add: Callable[[str], None],
    ) -> None:
        self._get_bridge = get_bridge
        self._node_templates_dir = node_templates_dir
        self._starter_template_id = starter_template_id
        self._get_renders_dir = get_renders_dir
        self._get_ui_nodes = get_ui_nodes
        self._get_ui_node_templates = get_ui_node_templates
        self._get_exporter_registry = get_exporter_registry
        self._get_shader_lib_index = get_shader_lib_index
        self._get_shader_lib_files = get_shader_lib_files
        self._get_current_node_id = get_current_node_id
        self._get_is_cancelled = get_is_cancelled
        self._set_current_node_id = set_current_node_id
        self._save_ui_node = save_ui_node
        self._sync_editor_from_disk = sync_editor_from_disk
        self._delete_node_unguarded_cb = delete_node_unguarded
        self._template_description = template_description
        self._working_set_reader = working_set_reader
        self._working_set_add = working_set_add
        self._batch_mutated_reader = batch_mutated_reader
        self._batch_mutated_add = batch_mutated_add

    @property
    def _bridge(self) -> CopilotBridge:
        # Resolved lazily: the bridge lives on the CopilotSession, which is constructed AFTER the
        # backend (the session takes the caps the backend's methods are bound into). Every call is
        # at turn-time, long after the session exists, so the getter always resolves.
        return self._get_bridge()

    def _copilot_short_ids(self) -> dict[str, str]:
        # full node-id -> short display id (feature: the copilot sees/passes a short prefix, not a
        # 36-char UUID — fewer bytes to copy = fewer mangled ids, and the user sees no ugly UUIDs).
        # Length is the shortest prefix (>=_COPILOT_SHORT_ID_LEN) that is unique across the CURRENT
        # nodes; on the rare prefix collision ALL ids grow together so display + resolve agree.
        ids = list(self._get_ui_nodes())
        n = _COPILOT_SHORT_ID_LEN
        while n < _COPILOT_FULL_ID_LEN:
            prefixes = [i[:n] for i in ids]
            if len(set(prefixes)) == len(prefixes):
                break
            n += 1
        return {i: i[:n] for i in ids}

    def _copilot_resolve_node_id(self, handle: str) -> str | None:
        # Resolve a copilot-supplied node handle (a short id, or — defensively — a full id) to the
        # full node-id, or None if it matches no node or is ambiguous. Accepts an exact full id, an
        # exact short id, or any unique prefix (the model may copy a few extra/fewer chars). An
        # empty/whitespace handle is unresolvable (every id startswith(""), so it would silently
        # resolve to the sole node and let delete/switch act on a node the model never named, §020·20
        # D4) — required-target tools must reject it, not the caller's current-node fallback.
        if not handle.strip():
            return None
        if handle in self._get_ui_nodes():
            return handle
        matches = [i for i in self._get_ui_nodes() if i.startswith(handle)]
        return matches[0] if len(matches) == 1 else None

    def node_tree(self) -> list[NodeTreeEntry]:
        # GL-FREE (feature 020·16, Decision 9): name + has_errors (cached compile_unit.errors,
        # no GL) + is_current. NO uniforms — get_active_uniforms() is a GL read and this is
        # called off-main when building the prompt context. Stays cache-stable (no per-frame value).
        # node_id carries the SHORT id (the only handle the copilot ever sees / passes back).
        current = self._get_current_node_id()
        short = self._copilot_short_ids()
        return [
            NodeTreeEntry(
                node_id=short[nid],
                name=ui_node.ui_state.ui_name,
                has_errors=bool(ui_node.node.compile_unit.errors),
                is_current=(nid == current),
            )
            for nid, ui_node in self._get_ui_nodes().items()
        ]

    def template_catalog(self) -> list[TemplateEntry]:
        # GL-FREE: the shipped node templates the agent always sees so it can create_node(template=...)
        # from a named template (e.g. Text Rendering) on intent, instead of writing source blind, AND
        # read_shader/grep them via the template: handle (feature 020·22). The handle is the PREFIXED
        # form `template:<4-char>` (self-describing, mirrors lib:) — never the full uuid in chat. The
        # description is the merged (override-or-shipped) value, sanitized for the prompt + the font.
        return [
            TemplateEntry(
                template_id=f"template:{tid[:4]}",
                name=ui_node.ui_state.ui_name,
                description=sanitize_display(self._template_description(tid)),
            )
            for tid, ui_node in self._get_ui_node_templates().items()
        ]

    def _copilot_resolve_template_id(self, handle: str) -> str | None:
        # Resolve a copilot template handle (the 4-char short id, the `template:`-prefixed form, or —
        # defensively — a full uuid) to the full template dir-uuid, or None if it matches no template /
        # is ambiguous. The catalogue emits `template:` + a 4-char prefix; the resolver strips it.
        h = handle.removeprefix("template:").strip()
        if not h:
            return None
        if h in self._get_ui_node_templates():
            return h
        matches = [tid for tid in self._get_ui_node_templates() if tid.startswith(h)]
        return matches[0] if len(matches) == 1 else None

    def _copilot_resolve_source(self, handle: str) -> tuple[str, str | None]:
        # Unify read/grep source addressing (feature 020·22): a `template:` prefix -> a TEMPLATE
        # (read-only), anything else a NODE. Returns (kind, full_id|None). lib: is NOT a read_shader
        # target (read_lib owns lib), so it falls through to the node resolver and returns None.
        if handle.startswith("template:"):
            return "template", self._copilot_resolve_template_id(handle)
        return "node", self._copilot_resolve_node_id(handle)

    def lib_catalog(self) -> list[LibCatalogEntry]:
        # GL-FREE: the parsed lib index (name + signature + doc + the lib: address). No bodies
        # (that is the read_lib pull). The address is the edit target for the function's file.
        root = shader_lib_root()
        entries: list[LibCatalogEntry] = []
        for fn in self._get_shader_lib_index().functions.values():
            try:
                rel = fn.file.relative_to(root)
            except ValueError:
                rel = fn.file
            entries.append(
                LibCatalogEntry(
                    name=fn.name,
                    signature=fn.signature,
                    doc=fn.doc,
                    lib_address=f"lib:{rel.as_posix()}",
                )
            )
        return entries

    # ---- cross-project reads (feature 020·16) ----

    def read_shaders(self, node_ids: list[str]) -> list[ShaderView]:
        # Marshalled (force-compile + uniform read are GL). `node_ids` are copilot handles (short
        # ids, or the resolved current id when the agent omitted them); each resolves to a full id.
        # For each: ensure a fresh compile, read source + uniforms (type + value) + errors, and ADD
        # it to the working set so its live source rides the scratchpad (feature 020·29). Unknown
        # handles are skipped (the tool layer reports them). The ShaderView carries the SHORT id.
        def _on_main() -> list[ShaderView]:
            short = self._copilot_short_ids()
            # [] / empty -> the current node (resolved here so the tool layer never handles a
            # full id; pin 2a — a concrete id is what gets stamped).
            handles = node_ids or [self._get_current_node_id()]
            views: list[ShaderView] = []
            seen: set[str] = (
                set()
            )  # dedup: two prefixes of one source resolve to the same id
            for handle in handles:
                kind, full_id = self._copilot_resolve_source(handle)
                if full_id is None or full_id in seen:
                    continue
                seen.add(full_id)
                if kind == "template":
                    # Read-only: build the SAME view, BUT don't add it to the working set (no edit ever
                    # targets a template) and address it by the `template:` prefixed handle (feature 020·22).
                    ui_node = self._get_ui_node_templates()[full_id]
                    view_id = f"template:{full_id[:4]}"
                else:
                    ui_node = self._get_ui_nodes()[full_id]
                    view_id = short[full_id]
                node = ui_node.node
                if node.program is None:
                    node.compile()
                text = node.source.text
                if kind == "node":
                    self._working_set_add(full_id)
                views.append(
                    ShaderView(
                        node_id=view_id,
                        name=ui_node.ui_state.ui_name,
                        listing=_number_lines(text),
                        uniforms=_format_uniforms(node.get_active_uniforms()),
                        errors=_to_error_infos(node.compile_unit.errors),
                    )
                )
            return views

        return self._bridge.run_on_main(_on_main)

    def read_working_set(self) -> list[WorkingSetView]:
        # Rebuild the working set into live views (feature 020·29). Bridge-marshalled: uniform reads
        # and the program-is-None recompile are GL. Current node is UNIONED in (never a filter) and
        # sorts FIRST; then the touched addresses in add-order. A gone node is skipped (never a hard
        # KeyError). Coherence invariant: a node with no live program is recompiled HERE so its shown
        # source and its errors are computed in the same rebuild (never a stale `errors: none`).
        def _on_main() -> list[WorkingSetView]:
            short = self._copilot_short_ids()
            current = self._get_current_node_id()
            ordered: list[str] = []
            if current and current in self._get_ui_nodes():
                ordered.append(current)
            for address in self._working_set_reader():
                if address not in ordered:
                    ordered.append(address)
            views: list[WorkingSetView] = []
            for address in ordered:
                if address.startswith("lib:"):
                    view = self._copilot_lib_working_view(address)
                else:
                    view = self._copilot_node_working_view(address, short, current)
                if view is not None:
                    views.append(view)
            return views

        return self._bridge.run_on_main(_on_main)

    def _copilot_node_working_view(
        self, full_id: str, short: dict[str, str], current: str
    ) -> WorkingSetView | None:
        ui_node = self._get_ui_nodes().get(full_id)
        if ui_node is None:
            return None
        node = ui_node.node
        if node.program is None:
            node.compile()
        return WorkingSetView(
            address=short.get(full_id, full_id),
            name=ui_node.ui_state.ui_name,
            listing=_number_lines(node.source.text),
            is_current=(full_id == current),
            is_lib=False,
            uniforms=_format_uniforms(node.get_active_uniforms()),
            errors=_to_error_infos(node.compile_unit.errors),
        )

    def _copilot_lib_working_view(self, address: str) -> WorkingSetView | None:
        # A lib file's live whole-file listing (feature 020·29 D6 — read_lib is function-keyed, so a
        # lib has no other source view). No standalone compile; a consuming node shows the new errors.
        path = self._get_shader_lib_files().resolve_copilot_path(address[len("lib:") :])
        if path is None or not path.exists():
            return None
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return None
        return WorkingSetView(
            address=address,
            name=address,
            listing=_number_lines(text),
            is_current=False,
            is_lib=True,
            uniforms=[],
            errors=[],
        )

    def grep(self, query: str) -> list[GrepHit]:
        # GL-FREE: substring search across every node's source, every shipped TEMPLATE's source, and
        # every lib file's source. Hits are origin-labeled so the agent can hand the origin to a read
        # tool (a node id, a `template:` handle, or a lib: address). Case-sensitive substring match (a
        # comment/#define can false-positive — acceptable for rare discovery, 020·16 Out-of-scope).
        if not query:
            return []
        short = self._copilot_short_ids()
        hits: list[GrepHit] = []
        for node_id, ui_node in self._get_ui_nodes().items():
            label = f"node '{ui_node.ui_state.ui_name}'"
            for i, line in enumerate(ui_node.node.source.text.split("\n"), start=1):
                if query in line:
                    hits.append(
                        GrepHit(
                            origin=short[node_id],
                            location=label,
                            line=i,
                            text=line.strip(),
                        )
                    )
        for tid, ui_node in self._get_ui_node_templates().items():
            origin = f"template:{tid[:4]}"
            label = f"template '{ui_node.ui_state.ui_name}'"
            for i, line in enumerate(ui_node.node.source.text.split("\n"), start=1):
                if query in line:
                    hits.append(
                        GrepHit(
                            origin=origin, location=label, line=i, text=line.strip()
                        )
                    )
        root = shader_lib_root()
        for path, source in self._get_shader_lib_index().sources.items():
            try:
                rel = path.relative_to(root)
            except ValueError:
                rel = path
            address = f"lib:{rel.as_posix()}"
            for i, line in enumerate(source.text.split("\n"), start=1):
                if query in line:
                    hits.append(
                        GrepHit(
                            origin=address, location=address, line=i, text=line.strip()
                        )
                    )
        return hits

    def read_lib(self, names: list[str]) -> list[LibFunctionBody]:
        # GL-FREE: the full body of each named lib function (the explicit pull; the catalogue in
        # the prompt carries signatures only). Unknown names are skipped (tool layer reports them).
        root = shader_lib_root()
        bodies: list[LibFunctionBody] = []
        for name in names:
            fn = self._get_shader_lib_index().functions.get(name)
            if fn is None:
                continue
            try:
                rel = fn.file.relative_to(root)
            except ValueError:
                rel = fn.file
            bodies.append(
                LibFunctionBody(
                    name=fn.name,
                    signature=fn.signature,
                    lib_address=f"lib:{rel.as_posix()}",
                    body=fn.body,
                )
            )
        return bodies

    # ---- cross-project mutations (feature 020·16) ----

    def set_uniform(self, name: str, value: object, node: str) -> SetUniformResult:
        # Set a runtime uniform VALUE on a node (020·16 Decision 6). Marshalled: validating the
        # name against get_active_uniforms() is a GL read, and try_to_release may touch GL. The
        # write itself mirrors the UI uniform widget (release old, dict-assign); the next render
        # picks it up. Up-front validation is the ONLY feedback channel — the render-time shape-pop
        # in Node.render is off-thread and the handler never sees it. Rejects samplers, blocks,
        # and the engine-driven uniforms (which render() overwrites every frame).
        def _on_main() -> SetUniformResult:
            node_id = (
                self._copilot_resolve_node_id(node)
                if node
                else self._get_current_node_id()
            )
            if node_id is None or node_id not in self._get_ui_nodes():
                return SetUniformResult(
                    ok=False,
                    error=f"no node with id '{node}' — check the project map",
                )
            if name in _ENGINE_DRIVEN_UNIFORMS:
                return SetUniformResult(
                    ok=False,
                    error=f"'{name}' is engine-driven (set every frame by ShaderBox) — it "
                    "cannot be set; change the shader code if you need different behavior",
                )
            target = self._get_ui_nodes()[node_id].node
            uniform = next(
                (u for u in target.get_active_uniforms() if u.name == name), None
            )
            if uniform is None:
                return SetUniformResult(
                    ok=False,
                    error=f"node has no active uniform '{name}' — read_shader it to see its "
                    "uniforms",
                )
            label = _uniform_type_label(uniform)
            if not isinstance(uniform, moderngl.Uniform) or label.startswith("sampler"):
                return SetUniformResult(
                    ok=False,
                    error=f"'{name}' is a {label} — only scalar/vector uniforms can be set "
                    "to a value; samplers and uniform blocks are not settable",
                )
            coerced = _coerce_uniform_value(value, uniform)
            if coerced is None:
                return SetUniformResult(
                    ok=False, error=_set_uniform_shape_hint(name, uniform, label)
                )
            try_to_release(target.uniform_values.get(name))
            target.uniform_values[name] = coerced
            return SetUniformResult(ok=True, type_label=label)

        return self._bridge.run_on_main(_on_main)

    def create_node(
        self, name: str, source: str, template: str, switch_to: bool
    ) -> tuple[str, list[CompileErrorInfo]]:
        # Create a node (020·16 Decision 8 / 020·22). `template` = a template_catalog handle (bare or
        # `template:`-prefixed); empty = the DEFAULT starter (UV Mango, the conventional blank canvas).
        # `source` non-empty overrides the instantiated body. Binds the RAW create body (NOT the
        # _copilot_busy_blocked-guarded create_node_from_selected_template). Insert order is save->insert->
        # set-current (mirror _seed_starter_node). Freshness-auto-stamps the new node so the agent can edit
        # it without a re-read. Compiles + returns errors — the same compile-feedback the edit tools give.
        def _on_main() -> tuple[str, list[CompileErrorInfo]]:
            template_id = (
                self._copilot_resolve_template_id(template)
                if template.strip()
                else self._starter_template_id
            )
            if template_id is None:
                raise RuntimeError(f"no template matching '{template}'")
            template_dir = self._node_templates_dir / template_id
            if not template_dir.is_dir():
                # Shipped resource; missing only on a broken install. The bridge re-raises on
                # the worker and the registry turns it into a clean tool error, not a crash.
                raise RuntimeError("starter template is missing")
            new_node = load_node_from_dir(template_dir)
            new_node.reset_id()
            if name.strip():
                new_node.ui_state.ui_name = name.strip()
            if source.strip():
                # release_program sets source.text; save_ui_node then writes it to the new
                # node's OWN dir + rebinds source.path. Do NOT write through source.path here —
                # it still points at the shared starter template until the save rebinds it.
                new_node.node.release_program(source)
            # Compile (GL-affine, must stay on main) BEFORE save so the persisted program matches
            # the reported errors. Empty source keeps the starter's clean program -> compiles clean.
            new_node.node.compile()
            self._save_ui_node(new_node)
            self._get_ui_nodes()[new_node.id] = new_node
            if switch_to:
                self._set_current_node_id(new_node.id)
            self._working_set_add(new_node.id)
            errors = _to_error_infos(new_node.node.compile_unit.errors)
            logger.info(
                f"copilot created node {new_node.id} (switch_to={switch_to}, "
                f"errors={len(errors)})"
            )
            # Return the SHORT id (computed after insert, so it's part of the current id set) —
            # the agent re-uses this handle to read/edit the node it just made.
            return self._copilot_short_ids()[new_node.id], errors

        return self._bridge.run_on_main(_on_main)

    def delete_node(self, node: str) -> DeleteNodeResult:
        # Delete a node on the copilot's behalf (feature 020·17). The agent loop has already
        # cleared a user Yes (GatePolicy.ALWAYS) before this runs. Resolves the short id, then
        # marshals the GL teardown to the main thread via the bridge (node release is GL-affine).
        # Returns the node_id + trash dir-name so the chat can offer a Recover.
        def _on_main() -> DeleteNodeResult:
            node_id = self._copilot_resolve_node_id(node)
            if node_id is None or node_id not in self._get_ui_nodes():
                return DeleteNodeResult(
                    ok=False,
                    error=f"no such node '{node}' — check the project map for ids",
                )
            name = self._get_ui_nodes()[node_id].ui_state.ui_name
            trash_name = self._delete_node_unguarded_cb(node_id)
            logger.info(f"copilot deleted node {node_id} (trash={trash_name})")
            return DeleteNodeResult(
                ok=True, deleted_name=name, node_id=node_id, trash_name=trash_name
            )

        return self._bridge.run_on_main(_on_main)

    def switch_node(self, node: str) -> SwitchNodeResult:
        # Make `node` the current one (so publish/render/edit-without-target act on it).
        # set_current_node_id is a bare state write the frame loop reads — marshal it on the main
        # thread. The switched node joins the working set so it rides the scratchpad (feature 020·29).
        def _on_main() -> SwitchNodeResult:
            node_id = self._copilot_resolve_node_id(node)
            if node_id is None or node_id not in self._get_ui_nodes():
                return SwitchNodeResult(
                    ok=False,
                    error=f"no such node '{node}' — check the project map for ids",
                )
            ui_node = self._get_ui_nodes()[node_id]
            self._set_current_node_id(node_id)
            self._working_set_add(node_id)
            logger.info(f"copilot switched current node to {node_id}")
            return SwitchNodeResult(ok=True, name=ui_node.ui_state.ui_name)

        return self._bridge.run_on_main(_on_main)

    def _copilot_render_path(self, node: UINode, ext: str) -> Path:
        # A non-colliding render filename: <name>_<short-id>_<n>.<ext>, n = the next free
        # index in renders_dir (the agent may render the same node several times a turn).
        base = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in node.ui_state.ui_name
        )
        short = self._copilot_short_ids().get(node.id, node.id[:4])
        renders = self._get_renders_dir()
        n = 0
        while True:
            candidate = renders / f"{base}_{short}_{n}.{ext}"
            if not candidate.exists():
                return candidate
            n += 1

    def render_image(self, node: str, width: int, height: int) -> RenderResult:
        # Render the node's current frame to a PNG (feature 020·18). GL => marshalled via the
        # bridge with the longer render_op_timeout_s (a heavy shader freezes the frame loop).
        def _on_main() -> RenderResult:
            ui_node = self._copilot_render_target(node)
            if ui_node is None:
                return RenderResult(ok=False, error=f"no such node '{node}'")
            w, h = self._copilot_render_dims(ui_node, width, height)
            preset = self._copilot_render_preset(False, None, w, h)
            out = self._copilot_render_path(ui_node, "png")
            art = share_state.render_to(ui_node.node, preset, 0.0, out)
            if art is None:
                return RenderResult(ok=False, error="render failed (see logs)")
            return RenderResult(
                ok=True,
                path=str(art.path),
                is_video=False,
                width=art.size[0],
                height=art.size[1],
            )

        return self._bridge.run_on_main(
            _on_main, timeout=COPILOT_CONFIG.render_op_timeout_s, defer=True
        )

    def render_video(
        self, node: str, seconds: float, fps: int, width: int, height: int
    ) -> RenderResult:
        # Render `seconds` of the node's animation (from t=0) to a WebM (feature 020·18).
        def _on_main() -> RenderResult:
            ui_node = self._copilot_render_target(node)
            if ui_node is None:
                return RenderResult(ok=False, error=f"no such node '{node}'")
            w, h = self._copilot_render_dims(ui_node, width, height)
            preset = self._copilot_render_preset(True, fps, w, h)
            out = self._copilot_render_path(ui_node, "webm")
            art = share_state.render_to(ui_node.node, preset, seconds, out)
            if art is None:
                return RenderResult(ok=False, error="render failed (see logs)")
            return RenderResult(
                ok=True,
                path=str(art.path),
                is_video=True,
                width=art.size[0],
                height=art.size[1],
                duration=art.duration,
            )

        return self._bridge.run_on_main(
            _on_main, timeout=COPILOT_CONFIG.render_op_timeout_s, defer=True
        )

    def _copilot_render_target(self, node: str) -> UINode | None:
        node_id = (
            self._copilot_resolve_node_id(node) if node else self._get_current_node_id()
        )
        if node_id is None or node_id not in self._get_ui_nodes():
            return None
        return self._get_ui_nodes()[node_id]

    def _copilot_render_dims(
        self, node: UINode, width: int, height: int
    ) -> tuple[int, int]:
        # 0 => the node's current canvas size (read on the main thread, GL-affine).
        cw, ch = node.node.canvas.texture.size
        return (width or cw, height or ch)

    def _copilot_render_preset(
        self, is_video: bool, fps: int | None, w: int, h: int
    ) -> RenderPreset:
        # FIXED_DIMS + RENDER_AT_TARGET so the resolved (w, h) actually drives the output —
        # the default FREE/SCALE_DISTORT renders into the node's own canvas and ignores them.
        return RenderPreset(
            is_video=is_video,
            fps=fps,
            target_w=w,
            target_h=h,
            container=".webm" if is_video else None,
            resolution_policy=ResolutionPolicy.FIXED_DIMS,
            fit=FitPolicy.RENDER_AT_TARGET,
        )

    def _copilot_publish(
        self,
        exporter: Exporter,
        kind: str,
        preset: RenderPreset,
        settings: dict[str, Any],
    ) -> PublishResult:
        # Render the node with the exporter's own preset, then enqueue the upload + AWAIT its
        # terminal progress (feature 020·18, Decision 5). Every exporter touch (render, enqueue,
        # the await poll's update()+status()) runs on the MAIN thread via the bridge — the worker
        # only sleeps + checks cancel between polls, so the frame loop stays responsive during the
        # network upload and the render-thread affinity contract holds.
        node_id = self._get_current_node_id()
        if node_id is None or node_id not in self._get_ui_nodes():
            return PublishResult(
                ok=False, error="no current node to publish", kind=kind
            )
        ui_node = self._get_ui_nodes()[node_id]

        def _render_and_enqueue() -> ExportProgress | None:
            duration = float(settings.get("seconds", preset.duration_max or 3.0))
            out = self._copilot_render_path(ui_node, share_state.preset_ext(preset))
            art = share_state.render_to(ui_node.node, preset, duration, out)
            if art is None:
                raise CopilotToolError("render failed")
            baseline = exporter.status().last_progress
            exporter.publish(art, settings)
            return baseline

        try:
            # The returned object is held HERE for the whole wait, so the terminal can never
            # be a different object reused at the baseline's address (no id() ambiguity).
            baseline = self._bridge.run_on_main(
                _render_and_enqueue,
                timeout=COPILOT_CONFIG.render_op_timeout_s,
                defer=True,
            )
        except CopilotToolError:
            return PublishResult(ok=False, error="render failed (see logs)", kind=kind)

        deadline = time.monotonic() + COPILOT_CONFIG.publish_await_timeout_s
        while time.monotonic() < deadline:
            if self._get_is_cancelled():
                return PublishResult(ok=False, error="cancelled", kind=kind)
            time.sleep(COPILOT_CONFIG.publish_poll_interval_s)
            try:
                status: ExporterStatus = self._bridge.run_on_main(
                    lambda: (exporter.update(None), exporter.status())[1]
                )
            except CopilotToolError:
                # A busy main thread timed out this poll; the upload still runs — try again.
                continue
            prog = status.last_progress
            if prog is not None and prog.is_terminal and prog is not baseline:
                if prog.is_error:
                    return PublishResult(ok=False, error=prog.message, kind=kind)
                return PublishResult(ok=True, url=prog.url or "", kind=kind)
        return PublishResult(
            ok=False,
            error="the upload is taking too long — check the Share tab for progress",
            kind=kind,
        )

    def publish_telegram(self, node: str, emoji: str) -> PublishResult:
        _ = node  # publish always renders the current node (the share path's contract)
        exporter = self._get_exporter_registry().get("telegram")
        if not isinstance(exporter, TelegramExporter):
            return PublishResult(
                ok=False, error="Telegram exporter unavailable", kind="telegram"
            )
        preset = exporter.render_preset()
        settings: dict[str, Any] = {
            "pack_set_name": exporter.current_default_pack(),
            "emoji": emoji,
            "seconds": preset.duration_max or 3.0,
        }
        return self._copilot_publish(exporter, "telegram", preset, settings)

    def publish_youtube(
        self, node: str, title: str, description: str, is_short: bool
    ) -> PublishResult:
        _ = node
        exporter = self._get_exporter_registry().get("youtube")
        if not isinstance(exporter, YouTubeExporter):
            return PublishResult(
                ok=False, error="YouTube exporter unavailable", kind="youtube"
            )
        # Drive the render shape from the tool arg so the preset (aspect/duration) and the
        # upload's is_short flag agree; restore the user's Share-tab shape after, so a copilot
        # publish doesn't silently flip their visible Long/Short selection.
        prior_short: bool = exporter.current_is_short()
        exporter.set_shape(is_short)
        try:
            preset = exporter.render_preset()
            settings: dict[str, Any] = {
                "title": title,
                "description": description,
                "is_short": is_short,
                "seconds": preset.duration_max or 6.0,
            }
            return self._copilot_publish(exporter, "youtube", preset, settings)
        finally:
            exporter.set_shape(prior_short)

    def has_current_node(self) -> bool:
        return self._get_current_node_id() in self._get_ui_nodes()

    def telegram_connected(self) -> bool:
        exporter = self._get_exporter_registry().get("telegram")
        return exporter is not None and exporter.is_connected()

    def youtube_connected(self) -> bool:
        exporter = self._get_exporter_registry().get("youtube")
        return exporter is not None and exporter.is_connected()

    def telegram_has_default_pack(self) -> bool:
        exporter = self._get_exporter_registry().get("telegram")
        return isinstance(exporter, TelegramExporter) and bool(
            exporter.current_default_pack()
        )

    # ---- Telegram connect + pack CRUD (feature 020·19) ----

    def _copilot_telegram(self) -> "TelegramExporter | None":
        exporter = self._get_exporter_registry().get("telegram")
        return exporter if isinstance(exporter, TelegramExporter) else None

    def telegram_token_set(self) -> bool:
        tg = self._copilot_telegram()
        return tg is not None and tg.bot_token_present()

    def set_telegram_token(self, secret: str) -> TelegramConnectResult:
        # MARSHAL the token-set + auto-link kickoff to the main thread (the exporter is render-
        # thread); then AWAIT auth_state leaving LINKING. The secret is set into the live store
        # here and nowhere else.
        def _on_main() -> None:
            tg = self._copilot_telegram()
            if tg is not None:
                tg.set_token(secret)
                tg.begin_auth()  # sets auth_state=LINKING, enqueues the link job

        tg = self._copilot_telegram()
        if tg is None:
            return TelegramConnectResult(
                ok=False, error="Telegram exporter unavailable"
            )
        self._bridge.run_on_main(_on_main)
        return self._copilot_await_telegram_connect()

    def telegram_connect(self) -> TelegramConnectResult:
        def _on_main() -> None:
            tg = self._copilot_telegram()
            if tg is not None:
                tg.begin_auth()

        tg = self._copilot_telegram()
        if tg is None:
            return TelegramConnectResult(
                ok=False, error="Telegram exporter unavailable"
            )
        self._bridge.run_on_main(_on_main)
        return self._copilot_await_telegram_connect()

    def _copilot_await_telegram_connect(self) -> TelegramConnectResult:
        # Poll auth_state to LEAVE the LINKING floor (begin_auth set it). Each poll pumps the
        # exporter's typed-event queue via the bridge (update() is what transitions auth_state)
        # — a status-only read would never see the change. cancel/teardown aware like the
        # publish-await. A "no message received" ERROR is surfaced as needs_start (guidance).
        tg = self._copilot_telegram()
        if tg is None:
            return TelegramConnectResult(
                ok=False, error="Telegram exporter unavailable"
            )
        deadline = time.monotonic() + COPILOT_CONFIG.telegram_connect_timeout_s
        while time.monotonic() < deadline:
            if self._get_is_cancelled():
                return TelegramConnectResult(ok=False, error="cancelled")
            time.sleep(COPILOT_CONFIG.publish_poll_interval_s)
            try:
                status: ExporterStatus = self._bridge.run_on_main(
                    lambda t=tg: (t.update(None), t.status())[1]
                )
            except CopilotToolError:
                continue
            if status.auth_state is AuthState.AUTHED:
                return TelegramConnectResult(
                    ok=True, bot_username=tg.bot_username_value()
                )
            if status.auth_state is AuthState.ERROR:
                return TelegramConnectResult(
                    ok=False,
                    error=status.auth_message,
                    needs_start=status.auth_message == NEEDS_START_ERROR,
                )
        return TelegramConnectResult(ok=False, error="link timed out — try again")

    def list_telegram_packs(self) -> list[TelegramPackInfo]:
        tg = self._copilot_telegram()
        if tg is None:
            return []
        active = tg.current_default_pack()
        return [
            TelegramPackInfo(
                title=p.title, set_name=p.set_name, is_default=p.set_name == active
            )
            for p in tg.list_packs()
        ]

    def select_telegram_pack(self, set_name: str) -> TelegramOpResult:
        def _on_main() -> str | None:
            tg = self._copilot_telegram()
            if tg is None:
                return None
            if all(p.set_name != set_name for p in tg.list_packs()):
                return ""
            tg.select_pack(set_name)
            return set_name

        result = self._bridge.run_on_main(_on_main)
        if result is None:
            return TelegramOpResult(ok=False, error="Telegram exporter unavailable")
        if result == "":
            return TelegramOpResult(ok=False, error=f"no pack named '{set_name}'")
        return TelegramOpResult(ok=True, set_name=result)

    def create_telegram_pack(self, title: str) -> TelegramOpResult:
        def _on_main() -> str | None:
            tg = self._copilot_telegram()
            if tg is None:
                return None
            tg.create_pack(title)
            return tg.current_default_pack()

        result = self._bridge.run_on_main(_on_main)
        if result is None:
            return TelegramOpResult(ok=False, error="Telegram exporter unavailable")
        return TelegramOpResult(ok=True, set_name=result)

    def delete_telegram_pack(self, set_name: str) -> TelegramOpResult:
        # delete_pack drops the pack locally + enqueues the Telegram delete (which pushes a
        # terminal ExportProgress) — await it like a publish so the agent can report the outcome.
        def _enqueue() -> ExportProgress | None:
            tg = self._copilot_telegram()
            if tg is None:
                raise CopilotToolError("Telegram exporter unavailable")
            if all(p.set_name != set_name for p in tg.list_packs()):
                raise CopilotToolError(f"no pack named '{set_name}'")
            baseline = tg.status().last_progress
            tg.delete_pack(set_name)
            return baseline

        try:
            baseline = self._bridge.run_on_main(_enqueue)
        except CopilotToolError as e:
            return TelegramOpResult(ok=False, error=str(e))
        deadline = time.monotonic() + COPILOT_CONFIG.publish_await_timeout_s
        while time.monotonic() < deadline:
            if self._get_is_cancelled():
                return TelegramOpResult(ok=False, error="cancelled")
            time.sleep(COPILOT_CONFIG.publish_poll_interval_s)
            tg = self._copilot_telegram()
            if tg is None:
                return TelegramOpResult(ok=False, error="Telegram exporter unavailable")
            try:
                status = self._bridge.run_on_main(
                    lambda t=tg: (t.update(None), t.status())[1]
                )
            except CopilotToolError:
                continue
            prog = status.last_progress
            if prog is not None and prog.is_terminal and prog is not baseline:
                if prog.is_error:
                    return TelegramOpResult(ok=False, error=prog.message)
                return TelegramOpResult(ok=True, set_name=set_name)
        return TelegramOpResult(
            ok=False, error="delete is taking too long — check the Share tab"
        )

    # ---- edit / compile-feedback (target-addressable: node or lib: file, 020·16) ----

    def apply_shader_edit(
        self, old_str: str, new_str: str, replace_all: bool, target: str
    ) -> EditResult:
        # Match + replace against the TARGET's AUTHORITATIVE source, then recompile (node) or
        # write (lib), persist, refresh the read-only editor — ONE bridge round-trip (spec
        # §16.3 / 020·16 Decision 3). Matching here (not on the worker against a re-read copy)
        # gives a single source of truth and no staleness window. A 0/ambiguous match mutates
        # nothing. The failed-compile early-return in Node.compile() keeps the old program live
        # (feature-013), so a broken node edit never blanks the user's preview. A substring edit
        # is NOT subject to the D9 guard (it matches by text, not line number) — but it RECORDS its
        # target as batch-mutated so a later same-batch LINE edit on it is caught (feature 020·29).
        def _on_main() -> EditResult:
            tgt = self._copilot_resolve_target(target, allow_create=False)
            if isinstance(tgt, EditResult):
                return tgt  # an unresolvable target rejects, mutates nothing
            src = tgt.source
            spans = token_match(src, old_str)
            if not spans:
                return EditResult(
                    matches=0, errors=[], hint=_whitespace_near_match(src, old_str)
                )
            if len(spans) > 1 and not replace_all:
                return EditResult(matches=len(spans), errors=[])
            if any(span_has_comment(src, s, e) for s, e in spans):
                return EditResult(matches=0, errors=[], comment_loss=True)
            new_text = _splice(src, spans, new_str)
            return self._copilot_persist_target(tgt, new_text, len(spans))

        return self._bridge.run_on_main(_on_main)

    def _copilot_persist_shader(
        self, node_id: str, node: Node, new_text: str
    ) -> list[CompileErrorInfo]:
        # Swap in the new source, recompile, persist to disk, refresh the read-only editor —
        # the shared main-thread tail of every copilot NODE edit (§16.3 / §14 L4). release_program
        # is what actually adopts new_text; compile()'s failed-compile early-return keeps the
        # old program live (feature-013) so a broken edit never blanks the preview. `node_id`
        # is the edit's TARGET (020·16 Decision 2b): the editor sync must key on it, NOT
        # self._get_current_node_id() — else a non-current edit syncs the wrong editor session.
        # sync_editor_from_disk no-ops when the target has no open editor (the usual non-current case).
        node.release_program(new_text)
        node.compile()
        node.source.path.write_text(new_text, encoding="utf-8")
        self._sync_editor_from_disk(node_id, new_text)
        return _to_error_infos(node.compile_unit.errors)

    def _copilot_resolve_target(
        self, target: str, *, allow_create: bool
    ) -> "_CopilotEditTarget | EditResult":
        # Resolve an edit target to its source + identity, or an EditResult REJECT (020·16
        # Decision 1/3). Parse rule (pin, Decision F1): a target is a LIB file IFF it starts
        # with "lib:"; otherwise it is a node-id; an empty target is the current node. An
        # unknown node-id is a hard error (never a silent lib fallback).
        if target.startswith("lib:"):
            return self._copilot_resolve_lib_target(target, allow_create=allow_create)
        if target.startswith("template:"):
            # Templates are READ-ONLY shipped resources (feature 020·22). An EXPLICIT guard (not
            # incidental non-resolution) with an actionable message, so the agent doesn't loop.
            return EditResult(
                matches=0,
                errors=[],
                unresolved=True,
                unresolved_reason="templates are read-only — create_node(template=...) from it "
                "first, then edit the resulting node",
            )
        if not target:
            node_id = self._get_current_node_id()
        else:
            resolved = self._copilot_resolve_node_id(target)
            if resolved is None:
                return EditResult(
                    matches=0,
                    errors=[],
                    unresolved=True,
                    unresolved_reason=f"no node with id '{target}' — use an id from the "
                    "project map",
                )
            node_id = resolved
        if node_id not in self._get_ui_nodes():
            return EditResult(
                matches=0,
                errors=[],
                unresolved=True,
                unresolved_reason="that shader no longer exists — check the project map for ids",
            )
        node = self._get_ui_nodes()[node_id].node
        return _CopilotEditTarget(
            kind="node",
            node_id=node_id,
            node=node,
            source=node.source.text,
            ws_address=node_id,
        )

    def _copilot_resolve_lib_target(
        self, target: str, *, allow_create: bool
    ) -> "_CopilotEditTarget | EditResult":
        # Resolve a "lib:<rel-path>" address to its file + current source. Reuses the
        # ShaderLibFileManager path-traversal guard (020·16 Decision 5). A non-existent path is
        # an error UNLESS allow_create (insert_after auto-creates the file, written LIVE).
        rel = target[len("lib:") :]
        path = self._get_shader_lib_files().resolve_copilot_path(rel)
        if path is None:
            return EditResult(
                matches=0,
                errors=[],
                unresolved=True,
                unresolved_reason=f"invalid library path '{target}' — copy a lib: address "
                "from the library catalogue or read_lib",
            )
        if not path.exists():
            if not allow_create:
                return EditResult(
                    matches=0,
                    errors=[],
                    unresolved=True,
                    unresolved_reason=f"no library file at '{target}' — use insert_after to "
                    "create a new library file, or copy an existing lib: address",
                )
            return _CopilotEditTarget(
                kind="lib",
                lib_path=path,
                source="",
                lib_create=True,
                ws_address=target,
            )
        return _CopilotEditTarget(
            kind="lib",
            lib_path=path,
            source=path.read_text(encoding="utf-8"),
            ws_address=target,
        )

    def _copilot_persist_target(
        self, tgt: "_CopilotEditTarget", new_text: str, matches: int
    ) -> EditResult:
        # Persist an applied edit and return the target-kind-aware result (020·16 Decision 3/4).
        # A NODE recompiles and returns compile errors; a LIB file is written (created if new) and
        # returns the honest "no standalone compile" string. On success the target joins the working
        # set + is marked batch-mutated (feature 020·29 — a later same-batch LINE edit on it is caught).
        if tgt.kind == "node":
            assert tgt.node is not None and tgt.node_id is not None
            errors = self._copilot_persist_shader(tgt.node_id, tgt.node, new_text)
            self._working_set_add(tgt.ws_address)
            self._batch_mutated_add(tgt.ws_address)
            return EditResult(matches=matches, errors=errors)
        assert tgt.lib_path is not None
        if not self._get_shader_lib_files().write_copilot_lib_file(
            tgt.lib_path, new_text
        ):
            return EditResult(
                matches=0,
                errors=[],
                unresolved=True,
                unresolved_reason="failed to write the library file",
            )
        self._copilot_invalidate_lib_consumers(tgt.lib_path)
        self._working_set_add(tgt.ws_address)
        self._batch_mutated_add(tgt.ws_address)
        verb = "created" if tgt.lib_create else "written"
        note = (
            f"library file {verb}; it has no standalone compile — errors will surface when a "
            "node that calls the function recompiles. Edit (or read) a node that uses it to "
            "confirm it is valid."
        )
        return EditResult(matches=matches, errors=[], lib_note=note)

    def _copilot_invalidate_lib_consumers(self, lib_path: Path) -> None:
        # A lib edit changes only the LIB file's text, not any consumer node's source.text — so the
        # next scratchpad rebuild's program-is-None recompile would NOT fire for the consumer, and it
        # would show stale errors. Invalidate every working-set node whose last compile pulled in this
        # lib file so the rebuild recompiles it with the new lib source (write_copilot_lib_file already
        # rebuilt the index — feature 020·29 D5). Match on the RESOLVED path: lib_path is fully resolved
        # (resolve_copilot_path) but the index glob's source paths are not, and they diverge under a
        # symlinked/relative SHADERBOX_DATA_DIR.
        target = lib_path.resolve()
        for address in self._working_set_reader():
            node = self._get_ui_nodes().get(address)
            if node is None:
                continue
            if any(s.path.resolve() == target for s in node.node.compile_unit.sources):
                node.node.invalidate()

    def apply_line_edit(
        self, start_line: int, end_line: int, new_text: str, target: str
    ) -> EditResult:
        # Replace the 1-based inclusive line range [start_line, end_line] with new_text over
        # the split("\n") line list (§14 L2 — byte-exact by construction), recompile/write +
        # persist. end_line == start_line - 1 is the one legal empty selection (a pure insert at
        # start_line); any other start > end, or an out-of-range bound, fails loud and mutates
        # nothing. A lib: target that does not exist yet is CREATED here (insert path only —
        # allow_create, 020·16 Decision 5). D9 (feature 020·29): a line-addressed edit to a target a
        # PRIOR edit already mutated THIS batch is rejected — its line numbers shifted, and the
        # scratchpad only refreshes BETWEEN batches.
        def _on_main() -> EditResult:
            tgt = self._copilot_resolve_target(target, allow_create=True)
            if isinstance(tgt, EditResult):
                return tgt
            if tgt.ws_address in self._batch_mutated_reader():
                return EditResult(
                    matches=0,
                    errors=[],
                    unresolved=True,
                    unresolved_reason="the line numbers shifted from an edit earlier in this "
                    "same step — the working set refreshes next step with current numbers; "
                    "re-issue then (or use edit_shader, which matches by text not line number)",
                )
            lines = tgt.source.split("\n") if tgt.source else []
            n = len(lines)
            is_insert = end_line == start_line - 1
            # A brand-new lib file (source "") accepts the one bootstrap insert at line 1
            # (insert_after 0); every other target uses the normal range bounds.
            new_lib_bootstrap = tgt.lib_create and is_insert and start_line == 1
            out_of_range = (
                start_line < 1
                or end_line > n
                or (start_line > end_line and not is_insert)
            )
            if out_of_range and not new_lib_bootstrap:
                return EditResult(matches=0, errors=[], hint="")
            repl = new_text.split("\n") if new_text != "" else []
            new_lines = lines[: start_line - 1] + repl + lines[end_line:]
            new_full = "\n".join(new_lines)
            return self._copilot_persist_target(tgt, new_full, 1)

        return self._bridge.run_on_main(_on_main)
