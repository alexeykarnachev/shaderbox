"""The copilot capability backend — the worker-facing implementation of every
`CopilotCapabilities` method (feature 023, extracted from `app.py`).

`CopilotBackend` owns the node/edit/uniform/render/publish/telegram verbs the copilot
worker calls. It does NOT import `App` (the no-`TYPE_CHECKING` rule): every dependency
is an explicit ref / getter / callback injected by `ProjectSession._build_copilot_capabilities`,
mirroring `shader_lib/file_ops.py::ShaderLibFileManager`. Project-dependent reads are
getters (re-read every call so a project switch retargets them); the working-set /
batch-mutated state stays on `App` and is reached through accessor callbacks. Every
GL-affine verb marshals to the main thread through `self._bridge.run_on_main`.
"""

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import glfw
import moderngl
from loguru import logger
from OpenGL.GL import GL_SAMPLER_2D, GL_UNSIGNED_INT

from shaderbox.copilot.address import (
    is_lib_address,
    is_template_address,
    lib_address,
    strip_lib_prefix,
    strip_template_prefix,
    template_address,
)
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
from shaderbox.copilot.checkpoint import TurnCheckpoint
from shaderbox.copilot.config import COPILOT_CONFIG
from shaderbox.copilot.edit_hints import compile_hints, render_facts
from shaderbox.copilot.errors import CopilotToolError
from shaderbox.copilot.glsl_lex import glsl_lex, span_drops_comment, token_match
from shaderbox.copilot.sanitize import sanitize_display
from shaderbox.core import ENGINE_DRIVEN_UNIFORMS, Canvas, Node
from shaderbox.exporters.base import (
    AuthState,
    Exporter,
    ExporterStatus,
    ExportProgress,
)
from shaderbox.exporters.registry import ExporterRegistry
from shaderbox.exporters.telegram import NEEDS_START_ERROR, TelegramExporter
from shaderbox.exporters.youtube import YouTubeExporter
from shaderbox.glyph_tables import TABLE_UNIFORMS
from shaderbox.paths import shader_lib_root
from shaderbox.render_preset import FitPolicy, RenderPreset, ResolutionPolicy
from shaderbox.shader_errors import ShaderError
from shaderbox.shader_lib import ShaderLibIndex, parser
from shaderbox.shader_lib.file_ops import ShaderLibFileManager
from shaderbox.tabs import share_state
from shaderbox.ui_models import UINode, load_node_from_dir
from shaderbox.uniform_coerce import (
    coerce_uniform_value,
    uniform_shape_hint,
)
from shaderbox.util import try_to_release

# Node-id prefix shown to the agent; _copilot_short_ids grows it past the floor only on collision.
_COPILOT_SHORT_ID_LEN = 4
_COPILOT_FULL_ID_LEN = 36


def _to_error_infos(errors: list[ShaderError]) -> list[CompileErrorInfo]:
    # ShaderError.line is 0-based (-1 = unparsed fallback); report 1-based, fallback as 0.
    return [
        CompileErrorInfo(
            path=str(e.path), line=e.line + 1 if e.line >= 0 else 0, message=e.message
        )
        for e in errors
    ]


def _cross_file_note(edited_path: Path, errors: list[CompileErrorInfo]) -> str:
    # "" unless EVERY error lives outside the edited file (a spliced lib) — then the
    # edited file may be fine and hints computed from its text would mislead.
    edited = edited_path.resolve()
    foreign: list[str] = []
    for e in errors:
        if Path(e.path).resolve() == edited:
            return ""
        if e.path not in foreign:
            foreign.append(e.path)
    if not foreign:
        return ""
    root = shader_lib_root()
    labels: list[str] = []
    for p in foreign:
        try:
            labels.append(lib_address(Path(p).relative_to(root)))
        except ValueError:
            try:
                labels.append(
                    lib_address(Path(p).resolve().relative_to(root.resolve()))
                )
            except ValueError:
                labels.append(p)
    return (
        f"the error is in {', '.join(labels)}, which this shader pulls in — the file "
        "you edited may be fine"
    )


def _edit_error_hints(
    edited_path: Path, new_text: str, errors: list[CompileErrorInfo]
) -> tuple[str, ...]:
    # The brace-balance hint reads the EDITED file's text; when every error is in
    # another file it is the wrong signal — dropped in favor of the cross-file note.
    hints = tuple(compile_hints(new_text, [e.message for e in errors]))
    note = _cross_file_note(edited_path, errors)
    if note:
        hints = (note, *(h for h in hints if "'{'" not in h))
    return hints


# D9: one whole-file rewrite per file per step — an earlier edit this batch changed the
# content the rewrite was composed from.
_BATCH_GUARD_REASON = (
    "this file was already edited earlier in this same step, so what you copied from "
    "the working set is stale — the working set refreshes next step; re-issue then "
    "(or use edit_shader, which re-matches the current text)"
)


def _number_lines(text: str) -> str:
    # cat -n style. Prefixes orient the agent but are not part of the text it matches against.
    lines = text.split("\n")
    width = len(str(len(lines)))
    return "\n".join(f"{i:>{width}}  {line}" for i, line in enumerate(lines, start=1))


@dataclass
class _CopilotEditTarget:
    # A resolved edit target: a NODE (recompiles) or a LIB file (written, no standalone compile).
    # `source` is the current text to edit against ("" for a not-yet-created lib file).
    # `ws_address` is the working-set + per-batch-guard key (node full-id, or the "lib:" address).
    # `label` names the target in result messages (EditResult.target_label).
    kind: str  # "node" | "lib"
    source: str
    ws_address: str
    label: str = ""
    node_id: str | None = None
    node: "Node | None" = None
    lib_path: Path | None = None
    lib_create: bool = False


def _ws_normalize(text: str) -> tuple[str, list[int]]:
    # Collapse horizontal-whitespace runs to one space (dropped adjacent to a newline) and
    # return the normalized text + a per-char map back to original indices, so a normalized-space
    # match can be sliced back to exact original bytes.
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
            # One space only between two non-newline chars; a run touching a newline collapses away.
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
    # The unique region of src matching old_str ignoring whitespace, as exact original bytes;
    # "" when no match or not unique.
    norm_src, src_index = _ws_normalize(src)
    norm_old, _ = _ws_normalize(old_str)
    if not norm_old:
        return ""
    first = norm_src.find(norm_old)
    if first == -1 or norm_src.find(norm_old, first + 1) != -1:
        return ""  # no match, or ambiguous — no safe single hint
    return src[src_index[first] : src_index[first + len(norm_old)]]


def _comment_only_spans(src: str, old_str: str) -> list[tuple[int, int]] | None:
    # None = old_str has code tokens (the token matcher owns it). A COMMENT/whitespace-only
    # old_str is invisible to the token matcher (comments lex as trivia), so it matches by
    # whitespace-normalized TEXT instead — comments are editable content too.
    if glsl_lex(old_str):
        return None
    norm_src, src_index = _ws_normalize(src)
    norm_old, _ = _ws_normalize(old_str)
    if not norm_old.strip():
        return []
    spans: list[tuple[int, int]] = []
    pos = norm_src.find(norm_old)
    while pos != -1:
        spans.append((src_index[pos], src_index[pos + len(norm_old)]))
        pos = norm_src.find(norm_old, pos + len(norm_old))
    return spans


def _splice(src: str, spans: list[tuple[int, int]], new_str: str) -> str:
    # Replace each non-overlapping (start, end) span with new_str. Offset-stable: spans don't overlap.
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
    # A sampler is a live Uniform but not a settable scalar/vector — labelled so set_uniform rejects it.
    if gl_type == GL_SAMPLER_2D:
        return "sampler2D"
    base = "uint" if gl_type == GL_UNSIGNED_INT else "float"
    scalar = base if u.dimension == 1 else f"vec{u.dimension}"
    return f"{scalar}[{u.array_length}]" if u.array_length > 1 else scalar


def _format_uniforms(node: Node) -> list[str]:
    # "name type = value" rows. Blocks have no scalar value. The shown value comes from the node's
    # uniform_values cache (the same source tabs/node.py reads) — NOT live u.value, which Node.render()
    # overwrites every frame, so a just-set_uniform value would read back stale and the agent loops.
    # Engine glyph tables are skipped outright: their u.value is ~14KB of stroke data the agent
    # can neither set nor learn from.
    rows: list[str] = []
    for u in node.get_active_uniforms():
        if u.name in TABLE_UNIFORMS:
            continue
        label = _uniform_type_label(u)
        if isinstance(u, moderngl.UniformBlock):
            rows.append(f"{u.name} {label}")
        else:
            value = node.uniform_values.get(u.name, u.value)
            rows.append(f"{u.name} {label} = {value}")
    return rows


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
        get_script_driven_uniforms: Callable[[str], set[str]],
        set_current_node_id: Callable[[str], None],
        save_ui_node: Callable[[UINode], object],
        sync_editor_from_disk: Callable[[str, str], None],
        delete_node_unguarded: Callable[[str], str],
        template_description: Callable[[str], str],
        working_set_reader: Callable[[], list[str]],
        working_set_add: Callable[[str], None],
        get_active_checkpoint: Callable[[], TurnCheckpoint | None],
    ) -> None:
        self._get_bridge = get_bridge
        self._probe_canvas: Canvas | None = None  # lazy 033 render-facts target
        # 033 force-restore bookkeeping: per-node consecutive broken-compile edits +
        # the last source text that compiled clean (the restore target).
        self._broken_streak: dict[str, int] = {}
        self._last_clean: dict[str, str] = {}
        # Oscillation brake (review cycle 2): recent source-state hashes per node —
        # an edit that returns the file to an earlier state is flagged as a fact.
        self._state_history: dict[str, list[int]] = {}
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
        self._get_script_driven_uniforms = get_script_driven_uniforms
        self._set_current_node_id = set_current_node_id
        self._save_ui_node = save_ui_node
        self._sync_editor_from_disk = sync_editor_from_disk
        self._delete_node_unguarded_cb = delete_node_unguarded
        self._template_description = template_description
        self._working_set_reader = working_set_reader
        self._working_set_add = working_set_add
        # Per-batch mutated-target guard: a whole-file rewrite of an address already here is rejected
        # (its lines shifted from an earlier same-batch edit). Cleared per batch via batch_begin.
        self._batch_mutated: set[str] = set()
        self._get_active_checkpoint = get_active_checkpoint

    def batch_begin(self) -> None:
        self._batch_mutated.clear()

    @property
    def _bridge(self) -> CopilotBridge:
        # Lazy: the bridge lives on the CopilotSession, built AFTER the backend. Resolved at turn-time.
        return self._get_bridge()

    # ---- rollback checkpoint capture (feature 020·30) ----
    # All run main-thread inside the bridge _on_main blocks, BEFORE the mutation. Best-effort:
    # TurnCheckpoint's own try/except swallows a capture failure so the edit never fails (decision 10).

    def _capture_node(self, node_id: str) -> None:
        # Serialize the LIVE node (not the stale on-disk dir — set_uniform never writes node.json).
        cp = self._get_active_checkpoint()
        node = self._get_ui_nodes().get(node_id)
        if cp is None or node is None:
            return
        cp.snapshot_node(
            node_id, node, lambda n, dest: n.save(dest.parent, dest.name, rebind=False)
        )

    def _capture_lib(
        self, ws_address: str, pre_edit_source: str, lib_create: bool
    ) -> None:
        cp = self._get_active_checkpoint()
        if cp is None:
            return
        if lib_create:
            cp.mark_created_lib(
                ws_address
            )  # reverse = delete the file, no pre-edit bytes
        else:
            cp.snapshot_lib(ws_address, pre_edit_source)

    def _copilot_short_ids(self) -> dict[str, str]:
        # full node-id -> shortest unique prefix (>=_COPILOT_SHORT_ID_LEN); on collision ALL ids grow
        # together so display + resolve stay consistent.
        ids = list(self._get_ui_nodes())
        n = _COPILOT_SHORT_ID_LEN
        while n < _COPILOT_FULL_ID_LEN:
            prefixes = [i[:n] for i in ids]
            if len(set(prefixes)) == len(prefixes):
                break
            n += 1
        return {i: i[:n] for i in ids}

    def _copilot_resolve_node_id(self, handle: str) -> str | None:
        # Handle (full id, short id, or unique prefix) -> full node-id, or None if no/ambiguous match.
        # Empty handle is unresolvable on purpose (else it'd resolve to the sole node — a required
        # target must reject, not fall back to current).
        if not handle.strip():
            return None
        if handle in self._get_ui_nodes():
            return handle
        matches = [i for i in self._get_ui_nodes() if i.startswith(handle)]
        return matches[0] if len(matches) == 1 else None

    def node_tree(self) -> list[NodeTreeEntry]:
        # GL-FREE (runs off-main building prompt context): name + has_errors (cached) + is_current.
        # No uniforms (that's a GL read). node_id is the short id.
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
        # GL-FREE: the shipped templates, addressed by a `template:<4-char>` handle (never the uuid).
        # Description is the merged override-or-shipped value, sanitized.
        return [
            TemplateEntry(
                template_id=template_address(tid),
                name=ui_node.ui_state.ui_name,
                description=sanitize_display(self._template_description(tid)),
            )
            for tid, ui_node in self._get_ui_node_templates().items()
        ]

    def _copilot_resolve_template_id(self, handle: str) -> str | None:
        # Template handle (`template:`-prefixed, short, or full uuid) -> full uuid, or None if no/ambiguous.
        # Forgiving: also matches a template by its DISPLAY NAME (case-insensitive) — the model copies the
        # human half of the `template:<id> | <name>` catalogue, so a bare name must resolve, not hard-fail.
        templates = self._get_ui_node_templates()
        h = strip_template_prefix(handle).strip()
        if not h:
            return None
        if h in templates:
            return h
        prefix = [tid for tid in templates if tid.startswith(h)]
        if len(prefix) == 1:
            return prefix[0]
        named = [
            tid
            for tid, n in templates.items()
            if n.ui_state.ui_name.casefold() == h.casefold()
        ]
        return named[0] if len(named) == 1 else None

    def _copilot_resolve_source(self, handle: str) -> tuple[str, str | None]:
        # read/grep addressing: `template:` -> TEMPLATE, else NODE. Returns (kind, full_id|None).
        # lib: falls through to the node resolver and returns None (read_shaders short-circuits
        # lib addresses before calling this).
        if is_template_address(handle):
            return "template", self._copilot_resolve_template_id(handle)
        return "node", self._copilot_resolve_node_id(handle)

    def lib_catalog(self) -> list[LibCatalogEntry]:
        # GL-FREE: name + signature + doc + lib: address per function. No bodies (that's read_lib).
        # SB_-prefixed only — the public surface the prompt promises; non-prefixed helpers are
        # file-private (callable only transitively) and would just be catalogue noise.
        root = shader_lib_root()
        entries: list[LibCatalogEntry] = []
        for fn in self._get_shader_lib_index().functions.values():
            if not fn.name.startswith("SB_"):
                continue
            try:
                rel = fn.file.relative_to(root)
            except ValueError:
                rel = fn.file
            entries.append(
                LibCatalogEntry(
                    name=fn.name,
                    signature=fn.signature,
                    doc=fn.doc,
                    lib_address=lib_address(rel),
                )
            )
        return entries

    # ---- cross-project reads (feature 020·16) ----

    def read_shaders(self, node_ids: list[str]) -> list[ShaderView]:
        # Marshalled (compile + uniform read are GL). Per handle: compile, read source + uniforms +
        # errors, add to the working set. Unknown handles skipped. ShaderView carries the short id.
        # A `lib:` handle reads the library file whole (grep origins advertise lib: as a read
        # handle — the read side honors the same address space as edit_shader).
        def _on_main() -> list[ShaderView]:
            short = self._copilot_short_ids()
            # [] -> the current node (resolved here so a concrete id is what gets stamped).
            handles = node_ids or [self._get_current_node_id()]
            views: list[ShaderView] = []
            seen: set[str] = (
                set()
            )  # dedup: two prefixes of one source resolve to the same id
            for handle in handles:
                if is_lib_address(handle):
                    lib_view = self._copilot_lib_working_view(handle)
                    if lib_view is None or handle in seen:
                        continue
                    seen.add(handle)
                    self._working_set_add(handle)
                    views.append(
                        ShaderView(
                            node_id=lib_view.address,
                            name=lib_view.name,
                            listing=lib_view.listing,
                            uniforms=[],
                            errors=[],
                        )
                    )
                    continue
                kind, full_id = self._copilot_resolve_source(handle)
                if full_id is None or full_id in seen:
                    continue
                seen.add(full_id)
                if kind == "template":
                    # Read-only: same view, not added to the working set, addressed by `template:` handle.
                    ui_node = self._get_ui_node_templates()[full_id]
                    view_id = template_address(full_id)
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
                        uniforms=_format_uniforms(node),
                        errors=_to_error_infos(node.compile_unit.errors),
                    )
                )
            return views

        return self._bridge.run_on_main(_on_main)

    def read_working_set(self) -> list[WorkingSetView]:
        # Rebuild the working set into live views (marshalled: uniform read + recompile are GL).
        # Current node unioned in first, then touched addresses in add-order; gone nodes skipped.
        # A program-less node is recompiled here so its source and errors are coherent.
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
                if is_lib_address(address):
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
            uniforms=_format_uniforms(node),
            errors=_to_error_infos(node.compile_unit.errors),
        )

    def _copilot_lib_working_view(self, address: str) -> WorkingSetView | None:
        # A lib file's whole-file listing (read_lib is function-keyed, so a lib has no other view).
        path = self._get_shader_lib_files().resolve_copilot_path(
            strip_lib_prefix(address)
        )
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
        # GL-FREE case-sensitive substring search across node / template / lib sources. Each hit is
        # origin-labelled (node id, `template:` handle, or lib: address) for a follow-up read.
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
            origin = template_address(tid)
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
            address = lib_address(rel)
            for i, line in enumerate(source.text.split("\n"), start=1):
                if query in line:
                    hits.append(
                        GrepHit(
                            origin=address, location=address, line=i, text=line.strip()
                        )
                    )
        return hits

    def read_lib(self, names: list[str]) -> list[LibFunctionBody]:
        # GL-FREE: the full body of each named lib function. Unknown names skipped.
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
                    lib_address=lib_address(rel),
                    body=fn.body,
                )
            )
        return bodies

    # ---- cross-project mutations (feature 020·16) ----

    def set_uniform(self, name: str, value: object, node: str) -> SetUniformResult:
        # Set a uniform value on a node (marshalled: validation + try_to_release touch GL). The write
        # mirrors the UI widget (release old, dict-assign); next render picks it up. Up-front validation
        # is the only feedback channel (the render-time shape-pop is off-thread). Rejects samplers,
        # blocks, and engine-driven uniforms.
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
            if name in ENGINE_DRIVEN_UNIFORMS:
                return SetUniformResult(
                    ok=False,
                    error=f"'{name}' is engine-owned (ShaderBox provides its value) — it "
                    "cannot be set; change the shader code if you need different behavior",
                )
            if name in self._get_script_driven_uniforms(node_id):
                return SetUniformResult(
                    ok=False,
                    error=f"'{name}' is script-driven (a behavior script computes its value "
                    "each frame) — a set here would be overwritten next tick; edit the script "
                    f"at nodes/{node_id}/scripts/{name}.py instead",
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
            coerced = coerce_uniform_value(value, uniform)
            if coerced is None:
                return SetUniformResult(
                    ok=False, error=uniform_shape_hint(uniform, label, value)
                )
            self._capture_node(node_id)  # pre-change rollback snapshot (best-effort)
            try_to_release(target.uniform_values.get(name))
            target.uniform_values[name] = coerced
            return SetUniformResult(
                ok=True, type_label=label, render_facts=self._render_facts_for(target)
            )

        return self._bridge.run_on_main(_on_main)

    def create_node(
        self, name: str, source: str, template: str, switch_to: bool
    ) -> tuple[str, list[CompileErrorInfo], str]:
        # Create a node from `template` (empty = the default starter); `source` overrides the body.
        # Order: save -> insert -> set-current. Adds to the working set; compiles + returns errors.
        def _on_main() -> tuple[str, list[CompileErrorInfo], str]:
            template_id = (
                self._copilot_resolve_template_id(template)
                if template.strip()
                else self._starter_template_id
            )
            if template_id is None:
                raise RuntimeError(f"no template matching '{template}'")
            template_dir = self._node_templates_dir / template_id
            if not template_dir.is_dir():
                # Missing only on a broken install; the registry turns the raise into a tool error.
                raise RuntimeError("starter template is missing")
            new_node = load_node_from_dir(template_dir)
            new_node.reset_id()
            if name.strip():
                new_node.ui_state.ui_name = name.strip()
            if source.strip():
                # release_program sets source.text; save_ui_node writes + rebinds source.path. Do NOT
                # write through source.path here — it still points at the shared starter template.
                new_node.node.release_program(
                    source.replace("\r\n", "\n").replace("\r", "\n")
                )
            # Compile (GL, main-thread) BEFORE save so the persisted program matches the reported errors.
            new_node.node.compile()
            # source.path still points at the template dir here; save rebinds it.
            pre_save_path = str(new_node.node.source.path)
            self._save_ui_node(new_node)
            self._get_ui_nodes()[new_node.id] = new_node
            cp = self._get_active_checkpoint()
            if cp is not None:
                cp.mark_created(new_node.id)  # reverse = delete-to-trash, no snapshot
            if switch_to:
                self._set_current_node_id(new_node.id)
            self._working_set_add(new_node.id)
            persisted_path = str(new_node.node.source.path)
            errors = [
                replace(e, path=persisted_path) if e.path == pre_save_path else e
                for e in _to_error_infos(new_node.node.compile_unit.errors)
            ]
            logger.info(
                f"copilot created node {new_node.id} (switch_to={switch_to}, "
                f"errors={len(errors)})"
            )
            if errors:
                extra = "\n".join(
                    compile_hints(
                        new_node.node.source.text, [e.message for e in errors]
                    )
                )
            else:
                extra = self._render_facts_for(new_node.node)
                self._last_clean[new_node.id] = new_node.node.source.text
            # Short id, computed after insert so it's in the current id set.
            return self._copilot_short_ids()[new_node.id], errors, extra

        return self._bridge.run_on_main(_on_main)

    def delete_node(self, node: str) -> DeleteNodeResult:
        # Delete a node (already user-confirmed). Marshals the GL teardown to main; returns node_id +
        # trash dir-name so the chat can offer a Recover.
        def _on_main() -> DeleteNodeResult:
            node_id = self._copilot_resolve_node_id(node)
            if node_id is None or node_id not in self._get_ui_nodes():
                return DeleteNodeResult(
                    ok=False,
                    error=f"no such node '{node}' — check the project map for ids",
                )
            name = self._get_ui_nodes()[node_id].ui_state.ui_name
            trash_name = self._delete_node_unguarded_cb(node_id)
            cp = self._get_active_checkpoint()
            if cp is not None:
                cp.record_deleted(node_id, trash_name)  # reverse = restore from trash
            logger.info(f"copilot deleted node {node_id} (trash={trash_name})")
            return DeleteNodeResult(
                ok=True, deleted_name=name, node_id=node_id, trash_name=trash_name
            )

        return self._bridge.run_on_main(_on_main)

    def switch_node(self, node: str) -> SwitchNodeResult:
        # Make `node` current (publish/render/untargeted-edit act on it). State write -> main thread;
        # the node joins the working set.
        def _on_main() -> SwitchNodeResult:
            node_id = self._copilot_resolve_node_id(node)
            if node_id is None or node_id not in self._get_ui_nodes():
                return SwitchNodeResult(
                    ok=False,
                    error=f"no such node '{node}' — check the project map for ids",
                )
            ui_node = self._get_ui_nodes()[node_id]
            cp = self._get_active_checkpoint()
            if cp is not None:
                cp.record_pre_switch(self._get_current_node_id())
            self._set_current_node_id(node_id)
            self._working_set_add(node_id)
            logger.info(f"copilot switched current node to {node_id}")
            return SwitchNodeResult(ok=True, name=ui_node.ui_state.ui_name)

        return self._bridge.run_on_main(_on_main)

    def _copilot_render_path(self, node: UINode, ext: str) -> Path:
        # Non-colliding filename <name>_<short-id>_<n>.<ext>, n = next free index in renders_dir.
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
        # Render the current frame to a PNG (GL; marshalled with the longer render_op_timeout_s).
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
        # Render `seconds` of animation (from t=0) to a WebM.
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
        # 0 => the node's current canvas size (GL read, main thread).
        cw, ch = node.node.canvas.texture.size
        return (width or cw, height or ch)

    def _copilot_render_preset(
        self, is_video: bool, fps: int | None, w: int, h: int
    ) -> RenderPreset:
        # FIXED_DIMS + RENDER_AT_TARGET so (w, h) drives the output (the default ignores them).
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
        # Render with the exporter's preset, enqueue the upload, then await its terminal progress.
        # Every exporter touch (render/enqueue/poll) runs on main via the bridge; the worker only
        # sleeps + checks cancel between polls.
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
            # Held for the whole wait so the terminal can't be a different object at the same address.
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
                # Poll timed out on a busy main thread; the upload still runs — retry.
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

    def publish_telegram(self, emoji: str) -> PublishResult:
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
        self, title: str, description: str, is_short: bool
    ) -> PublishResult:
        exporter = self._get_exporter_registry().get("youtube")
        if not isinstance(exporter, YouTubeExporter):
            return PublishResult(
                ok=False, error="YouTube exporter unavailable", kind="youtube"
            )

        # Drive the shape from the arg so preset + is_short agree; restore the user's Share-tab
        # shape after. set_shape writes _render_state.shape, which the main-thread Share tab reads —
        # so the shape mutations marshal to main (the publish itself does its own bridge ops).
        def _set_shape_read_preset(shape: bool) -> tuple[bool, RenderPreset]:
            prior = exporter.current_is_short()
            exporter.set_shape(shape)
            return prior, exporter.render_preset()

        prior_short, preset = self._bridge.run_on_main(
            lambda: _set_shape_read_preset(is_short)
        )
        try:
            settings: dict[str, Any] = {
                "title": title,
                "description": description,
                "is_short": is_short,
                "seconds": preset.duration_max or 6.0,
            }
            return self._copilot_publish(exporter, "youtube", preset, settings)
        finally:
            self._bridge.run_on_main(lambda: exporter.set_shape(prior_short))

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
        # Marshal token-set + auto-link to main, then await auth_state leaving LINKING. The secret is
        # set into the live store here and nowhere else.
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
        # Poll auth_state off the LINKING floor. Each poll pumps the exporter's event queue via the
        # bridge (update() drives the transition; a status-only read wouldn't see it). A "no message"
        # ERROR surfaces as needs_start.
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
        # delete_pack drops it locally + enqueues the Telegram delete; await its terminal progress.
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
        # Match + replace against the target's live source, recompile (node) / write (lib), persist,
        # refresh the editor — one bridge round-trip (matching on main = no staleness window). 0/ambiguous
        # match mutates nothing. A substring edit skips the D9 guard (matches by text) but records its
        # target as batch-mutated so a later same-batch whole-file rewrite is caught.
        def _on_main() -> EditResult:
            tgt = self._copilot_resolve_target(target, allow_create=False)
            if isinstance(tgt, EditResult):
                return tgt  # an unresolvable target rejects, mutates nothing
            src = tgt.source
            spans = token_match(src, old_str)
            if not spans:
                comment_spans = _comment_only_spans(src, old_str)
                if comment_spans is not None:
                    spans = comment_spans
            if not spans:
                return EditResult(
                    matches=0,
                    errors=[],
                    hint=_whitespace_near_match(src, old_str),
                    target_label=tgt.label,
                )
            if len(spans) > 1 and not replace_all:
                return EditResult(matches=len(spans), errors=[])
            if any(span_drops_comment(src, s, e, old_str) for s, e in spans):
                return EditResult(matches=0, errors=[], comment_loss=True)
            new_text = _splice(src, spans, new_str)
            return self._copilot_persist_target(tgt, new_text, len(spans))

        return self._bridge.run_on_main(_on_main)

    def _oscillation_note(self, key: str, prev_text: str, new_text: str) -> str:
        # Deterministic A->B->A detector: hash the post-edit source; if it matches
        # any earlier state of this file, the agent is cycling between versions —
        # tell it as a fact. The pre-edit state seeds the history so the very first
        # A->B->A round trip is caught; a no-op edit (state unchanged) is not an
        # oscillation. History is bounded; clears never needed (stable keys).
        h = hash(new_text)
        hist = self._state_history.setdefault(key, [])
        if not hist:
            hist.append(hash(prev_text))
        if hist[-1] == h:
            return ""
        note = ""
        if h in hist:
            back = hist[::-1].index(h) + 1
            note = (
                f"NOTE: this edit returns the file to a state it already had "
                f"{back} edit(s) ago — you are oscillating between versions. Stop "
                "editing; re-read the working set and reason before the next change."
            )
        hist.append(h)
        del hist[:-8]
        return note

    def _force_restore(
        self, node_id: str, node: Node, streak: int, matches: int
    ) -> EditResult:
        # The 033 unstick: N consecutive broken edits -> put the file back at its last
        # clean-compiling state and tell the agent as a fact. Resets the streak so the
        # next broken run gets a fresh budget.
        restore_errors = self._copilot_persist_shader(
            node_id, node, self._last_clean[node_id]
        )
        self._broken_streak[node_id] = 0
        logger.info(
            f"copilot force-restore | node={node_id} after {streak} broken edits"
        )
        note = (
            f"EDIT UNDONE — {streak} consecutive edits left compile errors, so the file "
            "was restored to its last clean-compiling state (the working set below shows "
            "the restored source). Re-read it and rewrite the whole block in ONE edit."
        )
        if restore_errors:
            err_lines = "\n".join(
                f"{e.path}:{e.line}: {e.message}" for e in restore_errors
            )
            note = (
                f"EDIT UNDONE — {streak} consecutive edits left compile errors; the "
                "file was restored to an earlier state, which itself no longer "
                f"compiles (likely a library change):\n{err_lines}"
            )
        facts = self._render_facts_for(node) if not restore_errors else ""
        return EditResult(
            matches=matches,
            errors=restore_errors,
            restored_note=note,
            render_facts=facts,
        )

    def _render_facts_for(self, node: Node) -> str:
        # Best-effort probe render -> one facts line (feature 033). Runs on the main
        # thread (bridge-marshalled callers) with the GL context current. Never raises
        # into the edit path — facts are advisory.
        if not COPILOT_CONFIG.render_facts_enabled:
            return ""
        try:
            size = COPILOT_CONFIG.render_facts_size
            # Match the node's canvas aspect — a square probe would lay out
            # aspect-corrected shaders (u_aspect) differently from the preview.
            cw, ch = node.canvas.texture.size
            h = min(4 * size, max(8, round(size * ch / cw))) if cw else size
            if self._probe_canvas is None:
                self._probe_canvas = Canvas(size=(size, h))
            else:
                self._probe_canvas.set_size((size, h))
            t = glfw.get_time()
            node.render(u_time=t, canvas=self._probe_canvas)
            raw = self._probe_canvas.texture.read()
            facts = render_facts(raw, size, h)
            # Stamp the sample time: an animated shader's facts change with phase,
            # which otherwise reads as an edit effect.
            return facts.replace("render:", f"render@t={t:.1f}s:", 1) if facts else ""
        except Exception as exc:  # — advisory channel, never break an edit
            logger.debug(f"copilot render facts skipped: {exc}")
            return ""

    def _copilot_persist_shader(
        self, node_id: str, node: Node, new_text: str
    ) -> list[CompileErrorInfo]:
        # Adopt new_text, recompile, persist, refresh the editor — the shared tail of every node edit.
        # sync_editor must key on `node_id` (the edit TARGET), not the current node, else a non-current
        # edit syncs the wrong session; it no-ops when the target has no open editor.
        node.release_program(new_text)
        node.compile()
        node.source.path.write_text(new_text, encoding="utf-8")
        self._sync_editor_from_disk(node_id, new_text)
        return _to_error_infos(node.compile_unit.errors)

    def _copilot_resolve_target(
        self, target: str, *, allow_create: bool
    ) -> "_CopilotEditTarget | EditResult":
        # Resolve an edit target to source + identity, or an EditResult REJECT. "lib:"-prefixed -> lib
        # file; empty -> current node; else a node-id (unknown is a hard error, never a lib fallback).
        if is_lib_address(target):
            return self._copilot_resolve_lib_target(target, allow_create=allow_create)
        if is_template_address(target):
            # Templates are read-only; an explicit guard with an actionable message (not silent non-resolution).
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
        ui_node = self._get_ui_nodes()[node_id]
        short = self._copilot_short_ids().get(node_id, node_id[:_COPILOT_SHORT_ID_LEN])
        label = f"node '{ui_node.ui_state.ui_name}' ({short})"
        if not target:
            label += " — target was empty, so this hit the CURRENT node"
        node = ui_node.node
        return _CopilotEditTarget(
            kind="node",
            node_id=node_id,
            node=node,
            source=node.source.text,
            ws_address=node_id,
            label=label,
        )

    def _copilot_resolve_lib_target(
        self, target: str, *, allow_create: bool
    ) -> "_CopilotEditTarget | EditResult":
        # Resolve "lib:<rel-path>" to file + source (reuses the path-traversal guard). A missing path
        # errors unless allow_create (write_shader auto-creates).
        rel = strip_lib_prefix(target)
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
                    unresolved_reason=f"no library file at '{target}' — use write_shader to "
                    "create a new library file, or copy an existing lib: address",
                )
            return _CopilotEditTarget(
                kind="lib",
                lib_path=path,
                source="",
                lib_create=True,
                ws_address=target,
                label=target,
            )
        return _CopilotEditTarget(
            kind="lib",
            lib_path=path,
            source=path.read_text(encoding="utf-8"),
            ws_address=target,
            label=target,
        )

    def _copilot_persist_target(
        self, tgt: "_CopilotEditTarget", new_text: str, matches: int
    ) -> EditResult:
        # Persist an applied edit. A NODE recompiles + returns errors; a LIB file is written + returns
        # the "no standalone compile" note. On success the target joins the working set + is batch-mutated.
        # Model-supplied text is CRLF-normalized here, the seam every edit write flows through.
        new_text = new_text.replace("\r\n", "\n").replace("\r", "\n")
        if tgt.kind == "node":
            assert tgt.node is not None and tgt.node_id is not None
            self._capture_node(tgt.node_id)  # pre-write rollback snapshot (best-effort)
            # "Clean" requires a LIVE program: an invalidated compile_unit has
            # errors=[] without one (e.g. after a lib edit) and must not anchor.
            prev_clean = (
                not tgt.node.compile_unit.errors and tgt.node.program is not None
            )
            errors = self._copilot_persist_shader(tgt.node_id, tgt.node, new_text)
            self._working_set_add(tgt.ws_address)
            self._batch_mutated.add(tgt.ws_address)
            if errors:
                if prev_clean:
                    # A clean file just broke — this starts a NEW streak (anything
                    # earlier was already fixed, possibly outside the copilot).
                    self._last_clean[tgt.node_id] = tgt.source
                    streak = 1
                else:
                    streak = self._broken_streak.get(tgt.node_id, 0) + 1
                self._broken_streak[tgt.node_id] = streak
                limit = COPILOT_CONFIG.auto_revert_after_failed_edits
                hints = _edit_error_hints(tgt.node.source.path, new_text, errors)
                if limit > 0 and streak >= limit:
                    if tgt.node_id in self._last_clean:
                        return self._force_restore(
                            tgt.node_id, tgt.node, streak, matches
                        )
                    hints = (
                        *hints,
                        f"hint: {streak} broken edits in a row and no clean state "
                        "known for this file this session — stop patching, rewrite "
                        "the whole shader in ONE edit",
                    )
                return EditResult(
                    matches=matches,
                    errors=errors,
                    error_hints=hints,
                    target_label=tgt.label,
                )
            self._broken_streak[tgt.node_id] = 0
            self._last_clean[tgt.node_id] = new_text
            facts = self._render_facts_for(tgt.node)
            loop_note = self._oscillation_note(tgt.node_id, tgt.source, new_text)
            if loop_note:
                facts = f"{facts}\n{loop_note}" if facts else loop_note
            return EditResult(
                matches=matches,
                errors=errors,
                render_facts=facts,
                target_label=tgt.label,
            )
        assert tgt.lib_path is not None
        # pre-write rollback snapshot (a brand-new lib reverses to a delete, not empty bytes)
        self._capture_lib(tgt.ws_address, tgt.source, tgt.lib_create)
        if not self._get_shader_lib_files().write_copilot_lib_file(
            tgt.lib_path, new_text
        ):
            return EditResult(
                matches=0,
                errors=[],
                unresolved=True,
                unresolved_reason="failed to write the library file",
            )
        self.invalidate_lib_consumers(tgt.lib_path)
        self._working_set_add(tgt.ws_address)
        self._batch_mutated.add(tgt.ws_address)
        verb = "created" if tgt.lib_create else "written"
        note = (
            f"library file {verb}; it has no standalone compile — errors will surface when a "
            "node that calls the function recompiles. Edit (or read) a node that uses it to "
            "confirm it is valid."
        )
        opens, closes = parser.brace_counts(new_text)
        if opens != closes:
            note += (
                f"\nwarning: the written file has {opens} '{{' vs {closes} '}}' — a brace "
                "went missing; consumer nodes will fail to compile"
            )
        loop_note = self._oscillation_note(tgt.ws_address, tgt.source, new_text)
        if loop_note:
            note = f"{note}\n{loop_note}"
        return EditResult(matches=matches, errors=[], lib_note=note)

    def invalidate_lib_consumers(self, lib_path: Path) -> None:
        # A lib edit leaves consumer nodes' source.text unchanged, so the next rebuild wouldn't recompile
        # them — invalidate every working-set node that pulled in this lib so they recompile with the new
        # source. Match on the resolved path (the index's source paths aren't resolved; they diverge under
        # a symlinked SHADERBOX_DATA_DIR).
        target = lib_path.resolve()
        for address in self._working_set_reader():
            node = self._get_ui_nodes().get(address)
            if node is None:
                continue
            if any(s.path.resolve() == target for s in node.node.compile_unit.sources):
                node.node.invalidate()

    def apply_full_rewrite(self, new_text: str, target: str) -> EditResult:
        # Whole-file rewrite/create. The removed-names fact makes a truncated rewrite
        # loud; skipped when force-restore undid the write.
        def _on_main() -> EditResult:
            tgt = self._copilot_resolve_target(target, allow_create=True)
            if isinstance(tgt, EditResult):
                return tgt
            if tgt.ws_address in self._batch_mutated:
                return EditResult(
                    matches=0,
                    errors=[],
                    unresolved=True,
                    unresolved_reason=_BATCH_GUARD_REASON,
                    target_label=tgt.label,
                )
            result = self._copilot_persist_target(tgt, new_text, 1)
            if result.unresolved or result.restored_note:
                return result
            opens, closes = parser.brace_counts(new_text)
            if opens != closes:
                # Brace-broken text hides later definitions from the depth-0 scan — the
                # note would claim still-present functions removed; the compile error +
                # brace hint (node) / the lib brace warning (persist) cover it loudly.
                return result
            old_fns, old_decls = parser.top_level_names(tgt.source)
            new_fns, new_decls = parser.top_level_names(new_text)
            # The scan misses restyled signatures (Allman/multi-line) — never claim a
            # name removed while it still TEXTUALLY occurs in the new source; a miss is
            # acceptable, a false "removed" fact is not.
            stripped_new = parser.strip_comments_keep_lines(new_text)
            removed_fns = [
                n
                for n in sorted(old_fns - new_fns)
                if not re.search(rf"\b{re.escape(n)}\s*\(", stripped_new)
            ]
            removed_decls = [
                n
                for n in sorted(old_decls - new_decls)
                if not re.search(rf"\b{re.escape(n)}\b", stripped_new)
            ]
            parts: list[str] = []
            if removed_fns:
                parts.append("function(s): " + ", ".join(removed_fns))
            if removed_decls:
                parts.append("declaration(s): " + ", ".join(removed_decls))
            if not parts:
                return result
            note = "note: this rewrite removed " + "; ".join(parts)
            return replace(result, rewrite_note=note)

        return self._bridge.run_on_main(_on_main)
