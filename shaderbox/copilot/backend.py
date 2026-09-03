"""The copilot capability backend — the worker-facing implementation of every
`CopilotCapabilities` method (feature 023, extracted from `app.py`).

`CopilotBackend` owns the document/edit/uniform/render/publish/telegram verbs the copilot
worker calls. It does NOT import `App` (the no-`TYPE_CHECKING` rule): every dependency
is an explicit ref / getter / callback injected by `ProjectSession._build_copilot_capabilities`,
mirroring `shader_lib/file_ops.py::ShaderLibFileManager`. Project-dependent reads are
getters (re-read every call so a project switch retargets them); the working-set /
batch-mutated state stays on `App` and is reached through accessor callbacks. Every
GL-affine verb marshals to the main thread through `self._bridge.run_on_main`.
"""

import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import moderngl
import numpy as np
from loguru import logger

from shaderbox import render_job
from shaderbox.copilot.address import (
    DOCUMENT_SHORT_ID_LEN,
    example_address,
    is_example_address,
    is_lib_address,
    lib_address,
    pass_address,
    split_pass_address,
    strip_example_prefix,
    strip_lib_prefix,
)
from shaderbox.copilot.bridge import CopilotBridge
from shaderbox.copilot.capabilities import (
    CompileErrorInfo,
    DeleteDocumentResult,
    DocumentImportResult,
    DocumentOpResult,
    DocumentTreeEntry,
    EditResult,
    ExampleEntry,
    GrepHit,
    LibCatalogEntry,
    LibFileResult,
    LibFunctionBody,
    MediaBindResult,
    PassView,
    PublishResult,
    RenderResult,
    ScriptView,
    ScriptWriteResult,
    SetUniformResult,
    ShaderView,
    SwitchDocumentResult,
    TelegramConnectResult,
    TelegramOpResult,
    TelegramPackInfo,
    WorkingSetView,
)
from shaderbox.copilot.checkpoint import TurnCheckpoint
from shaderbox.copilot.config import COPILOT_CONFIG, COPILOT_ENGINE
from shaderbox.copilot.edit_hints import (
    FACTS_PREFIX,
    STAMPED_FACTS_PREFIX,
    compile_hints,
    render_facts,
)
from shaderbox.copilot.edit_match import (
    comment_only_spans,
    script_match_spans,
    splice,
    splice_script,
    whitespace_near_match,
)
from shaderbox.copilot.error_render import format_compile_errors
from shaderbox.copilot.errors import CopilotToolError
from shaderbox.copilot.gate import GateChannel, GateKind, GateRequest
from shaderbox.copilot.glsl_lex import span_drops_comment, token_match
from shaderbox.copilot.sanitize import sanitize_display
from shaderbox.core import ENGINE_DRIVEN_UNIFORMS, Canvas, Pass
from shaderbox.document import Document, document_dir_of
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
from shaderbox.media import (
    MediaWithTexture,
    media_class_for,
)
from shaderbox.pass_graph import AutoSource, NoSource, clamp_canvas_size
from shaderbox.paths import DOCUMENT_SCRIPT_BASENAME, pass_name_of, shader_lib_root
from shaderbox.render_preset import RenderPreset
from shaderbox.render_shape import RenderShape, shape_to_preset
from shaderbox.scripting import (
    ScriptError,
    ScriptProbe,
    ScriptStatus,
    normalize_script_tabs,
)
from shaderbox.shader_errors import ShaderError
from shaderbox.shader_lib import ShaderLibIndex, parser
from shaderbox.shader_lib.file_ops import ShaderLibFileManager
from shaderbox.ui_models import UIDocument, load_document_from_dir
from shaderbox.uniform_coerce import (
    coerce_uniform_value,
    gl_type_label,
    uniform_shape_hint,
)
from shaderbox.util import try_to_release

# Mean abs per-channel pixel delta (0-255) between two probe frames above which a shader counts as
# ANIMATING — small enough to catch any real u_time motion, large enough to ignore FP noise.
_MOTION_EPS = 1.5


def _stamp_facts(facts: str, t: float) -> str:
    # Stamp the probe's sample time onto the facts line ("render:" -> "render@t=Xs:").
    if not facts:
        return ""
    return facts.replace(FACTS_PREFIX, f"{STAMPED_FACTS_PREFIX}{t:.1f}s:", 1)


_COPILOT_FULL_ID_LEN = 36


def _to_error_infos(errors: list[ShaderError]) -> list[CompileErrorInfo]:
    # ShaderError.line is 0-based (-1 = unparsed fallback); report 1-based, fallback as 0.
    return [
        CompileErrorInfo(
            path=str(e.path), line=e.line + 1 if e.line >= 0 else 0, message=e.message
        )
        for e in errors
    ]


def _script_error_info(err: ScriptError) -> CompileErrorInfo:
    # ScriptError.line is 1-based already (the user source line; -1 = unmapped). Report 0 for unmapped.
    return CompileErrorInfo(
        path=DOCUMENT_SCRIPT_BASENAME,
        line=err.line if err.line > 0 else 0,
        message=err.message,
    )


def _no_document_error(handle: str) -> CompileErrorInfo:
    where = f"'{handle}'" if handle else "the current document (none selected)"
    return CompileErrorInfo(path="", line=0, message=f"no document found for {where}")


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
# content the rewrite was composed from. `edit_tool` names the substring editor for the
# artifact kind (a script write must not steer at edit_shader).
def _batch_guard_reason(edit_tool: str) -> str:
    return (
        "this file was already edited earlier in this same step, so what you copied from "
        "the working set is stale — the working set refreshes next step; re-issue then "
        f"(or use {edit_tool}, which re-matches the current text)"
    )


def _number_lines(text: str) -> str:
    # cat -n style. Prefixes orient the agent but are not part of the text it matches against.
    lines = text.split("\n")
    width = len(str(len(lines)))
    return "\n".join(f"{i:>{width}}  {line}" for i, line in enumerate(lines, start=1))


@dataclass
class _CopilotEditTarget:
    # A resolved edit target: a DOCUMENT (recompiles) or a LIB file (written, no standalone compile).
    # `source` is the current text to edit against ("" for a not-yet-created lib file).
    # `ws_address` is the working-set + per-batch-guard key (document full-id, or the "lib:" address).
    # `label` names the target in result messages (EditResult.target_label).
    kind: str  # "document" | "lib"
    source: str
    ws_address: str
    label: str = ""
    document_id: str | None = None
    document: "Document | None" = None
    # Which PASS of that document the edit lands on. A bare document address resolves to its
    # OUTPUT pass, so every tool that predates the graph keeps its meaning.
    render_pass: "Pass | None" = None
    lib_path: Path | None = None
    lib_create: bool = False


def _driven_on(driven: set[tuple[str, str]], pass_name: str) -> set[str]:
    # The uniform NAMES the script drives on one pass, out of the document-scoped (pass, name) set
    # (069). `_format_uniforms` formats one pass, so it keeps its name-keyed signature.
    return {name for pass_, name in driven if pass_ == pass_name}


def _format_uniforms(
    render_pass: Pass, driven: set[str], wired: Mapping[str, str]
) -> list[str]:
    # "name type = value" rows. Blocks have no scalar value. The shown value comes from the document's
    # uniform_values cache (the same source tabs/document.py reads) — NOT live u.value, which Pass.render()
    # overwrites every frame, so a just-set_uniform value would read back stale and the agent loops.
    # Engine glyph tables are skipped outright: their u.value is ~14KB of stroke data the agent
    # can neither set nor learn from. A `driven` uniform shows a marker, not a phantom value: the
    # copilot path never ticks the script, so its uniform_values entry is the stale manual default —
    # showing it as a number contradicts the write that said the script drives it (feature 043).
    rows: list[str] = []
    for u in render_pass.get_active_uniforms():
        if u.name in TABLE_UNIFORMS:
            continue
        label = gl_type_label(u)
        if isinstance(u, moderngl.UniformBlock):
            rows.append(f"{u.name} {label}")
        elif u.name in driven:
            rows.append(f"{u.name} {label} = <driven by script.py>")
        elif label == "sampler2D":
            rows.append(
                f"{u.name} {label} <- {_sampler_reads(render_pass, u.name, wired)}"
            )
        else:
            value = render_pass.uniform_values.get(u.name, u.value)
            rows.append(f"{u.name} {label} = {value}")
    return rows


def _sampler_uniform_names(render_pass: Pass) -> list[str]:
    return [
        u.name
        for u in render_pass.get_active_uniforms()
        if gl_type_label(u) == "sampler2D"
    ]


def _wired_for(document: Document, render_pass: Pass) -> Mapping[str, str]:
    # What the binder fills on THIS pass: uniform -> the pass it reads (072). The name rule
    # resolves from a COMPILED program, so every caller compiles the pass first; asked about a
    # never-compiled pass this reports its name-wired samplers as BLACK.
    return document.effective_wiring().get(pass_name_of(render_pass.source.path), {})


def _sampler_reads(render_pass: Pass, name: str, wired: Mapping[str, str]) -> str:
    """One sampler's `<-` cell: the pass it reads, the media the user bound, or why it is
    black. Whether a filled edge was chosen or derived from the name is not the model's
    business; an EMPTY one is, because "the user chose black" and "nothing decided yet" call
    for different actions. NEVER the source path (corollary-1: the abs path is a model-visible
    leak); only dims + kind for media."""
    if name in wired:
        return wired[name]
    value = render_pass.uniform_values.get(name)
    if isinstance(value, MediaWithTexture):
        res = value.details.resolution_details
        kind = "video" if value.details.is_video else "image"
        return f"({res.width}x{res.height}, {kind})"
    if isinstance(value, moderngl.Texture):
        return "(texture)"
    if isinstance(value, NoSource):
        return "(none; reads BLACK)"
    return "(nothing; reads BLACK)"


def _values_differ(a: object, b: object, eps: float) -> bool:
    # Recursive epsilon compare over the shapes coerce_one emits (scalar / tuple / list / nested),
    # structurally like behavior.py::_all_finite. A shape mismatch counts as differ. Used to tell a
    # driven uniform's value moving across sample times (motion) from a constant (static).
    if isinstance(a, int | float) and isinstance(b, int | float):
        return abs(a - b) > eps
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        if len(a) != len(b):
            return True
        return any(_values_differ(x, y, eps) for x, y in zip(a, b, strict=False))
    return a != b


def _uniform_changes(
    key: tuple[str, str],
    samples: list[tuple[float, dict[tuple[str, str], object]]],
    eps: float,
) -> bool:
    # True when this (pass, uniform)'s value differs by > eps between ANY pair of samples (it
    # animates over t).
    vals = [s.get(key) for _t, s in samples if key in s]
    return any(_values_differ(vals[0], v, eps) for v in vals[1:])


def _dotted(key: tuple[str, str]) -> str:
    # `pass.uniform` for the agent to READ (069). Display only — never parsed back, and never a key
    # grammar (the script addresses a pass by a nested dict, not by a dotted string).
    return f"{key[0]}.{key[1]}"


def _motion_verdict(probe: ScriptProbe, render_line: str, eps: float) -> str:
    # The value-diff motion signal (feature 043): which driven uniforms change across t (exact,
    # GL-free) + ONE render line for the "is it visible / FLAT" honesty case the value-diff misses.
    if not probe.driven:
        return (
            "drives 0 uniforms (update returned an empty dict / only orphan keys). Nothing "
            "animates and every uniform stays manual. Return {name: value} to drive one."
        )
    lines = [
        "values@t={:.1f}: {}".format(
            t, " ".join(f"{_dotted(k)}={s[k]}" for k in sorted(s)) or "(none)"
        )
        for t, s in probe.samples
    ]
    changing = sorted(
        k for k in probe.driven if _uniform_changes(k, probe.samples, eps)
    )
    constant = sorted(probe.driven - set(changing))
    if render_line:
        lines.append(render_line)
    if changing:
        verdict = (
            f"{', '.join(_dotted(k) for k in changing)} CHANGE across t (ANIMATING)"
        )
        if constant:
            verdict += f"; {', '.join(_dotted(k) for k in constant)} constant"
    else:
        verdict = (
            "values UNCHANGED across t (STATIC) -- the script drives these uniforms with values "
            "that do not vary over ctx.t. If you meant motion, vary a value by ctx.t."
        )
    lines.append(f"-> {verdict}")
    return "\n".join(lines)


class CopilotBackend:
    def __init__(
        self,
        *,
        get_bridge: Callable[[], CopilotBridge],
        get_gate: Callable[[], GateChannel],
        document_examples_dir: Path,
        starter_example_id: str,
        get_renders_dir: Callable[[], Path],
        get_ui_documents: Callable[[], dict[str, UIDocument]],
        get_ui_document_examples: Callable[[], dict[str, UIDocument]],
        get_exporter_registry: Callable[[], ExporterRegistry],
        get_shader_lib_index: Callable[[], ShaderLibIndex],
        get_shader_lib_files: Callable[[], ShaderLibFileManager],
        get_current_document_id: Callable[[], str],
        get_is_cancelled: Callable[[], bool],
        get_script_driven_uniforms: Callable[[str], set[tuple[str, str]]],
        get_script_path: Callable[[str], Path],
        get_script_source_view: Callable[[str], tuple[str, ScriptStatus | None]],
        read_script_source: Callable[[str], tuple[str, bool]],
        write_script_source: Callable[[str, str], ScriptProbe],
        set_current_document_id: Callable[[str], None],
        save_ui_document: Callable[[UIDocument], object],
        sync_editor_from_disk: Callable[[Path, str], None],
        delete_document_unguarded: Callable[[str], str],
        example_description: Callable[[str], str],
        working_set_reader: Callable[[], list[str]],
        working_set_add: Callable[[str], None],
        working_set_evicted: Callable[[], list[str]],
        working_set_reset: Callable[[], None],
        get_active_checkpoint: Callable[[], TurnCheckpoint | None],
    ) -> None:
        self._get_bridge = get_bridge
        self._get_gate = get_gate
        self._probe_canvas: Canvas | None = None  # lazy 033 render-facts target
        # Per-document last probe frames (raw0, raw1) — a mutation whose frames match the previous
        # ones changed NOTHING on screen (dead code / wrong target / script-overridden value).
        self._last_probe: dict[str, tuple[bytes, bytes]] = {}
        # The script twin of _last_probe: per-document last clean dry-run samples. A script edit whose
        # sampled driven values match the previous clean edit's changed NOTHING behaviorally (a
        # dead store an own later line overwrites, a text-only change) — the value-diff is the only
        # channel that can catch it (frame-pair facts are pace-blind).
        self._last_script_samples: dict[str, object] = {}
        # 033 force-restore bookkeeping: per-document consecutive broken-compile edits +
        # the last source text that compiled clean (the restore target).
        self._broken_streak: dict[str, int] = {}
        self._last_clean: dict[str, str] = {}
        # The script analog of the shader force-restore (033): N consecutive broken-compile/runtime
        # script writes on ONE document -> revert to its last clean-probing source. A script has TWO failure
        # modes (compile + runtime), so it is at least as prone to a broken-edit loop as a shader.
        self._script_broken_streak: dict[str, int] = {}
        self._script_last_clean: dict[str, str] = {}
        # Oscillation brake (review cycle 2): recent source-state hashes per document —
        # an edit that returns the file to an earlier state is flagged as a fact.
        self._state_history: dict[str, list[int]] = {}
        self._document_examples_dir = document_examples_dir
        self._starter_example_id = starter_example_id
        self._get_renders_dir = get_renders_dir
        self._get_ui_documents = get_ui_documents
        self._get_ui_document_examples = get_ui_document_examples
        self._get_exporter_registry = get_exporter_registry
        self._get_shader_lib_index = get_shader_lib_index
        self._get_shader_lib_files = get_shader_lib_files
        self._get_current_document_id = get_current_document_id
        self._get_is_cancelled = get_is_cancelled
        self._get_script_driven_uniforms = get_script_driven_uniforms
        self._get_script_path = get_script_path
        self._get_script_source_view = get_script_source_view
        self._read_script_source = read_script_source
        self._write_script_source = write_script_source
        self._set_current_document_id = set_current_document_id
        self._save_ui_document = save_ui_document
        self._sync_editor_from_disk = sync_editor_from_disk
        self._delete_document_unguarded_cb = delete_document_unguarded
        self._example_description = example_description
        self._working_set_reader = working_set_reader
        self._working_set_add = working_set_add
        self._working_set_evicted = working_set_evicted
        self._working_set_reset = working_set_reset
        # Per-batch mutated-target guard: a whole-file rewrite of an address already here is rejected
        # (its lines shifted from an earlier same-batch edit). Cleared per batch via batch_begin. A
        # document's script keys as ("script", document_id) so it can't collide with a document address.
        self._batch_mutated: set[str | tuple[str, str]] = set()
        self._get_active_checkpoint = get_active_checkpoint

    def batch_begin(self) -> None:
        self._batch_mutated.clear()

    def reset_working_set(self) -> None:
        self._working_set_reset()

    @property
    def _bridge(self) -> CopilotBridge:
        # Lazy: the bridge lives on the CopilotSession, built AFTER the backend. Resolved at turn-time.
        return self._get_bridge()

    # ---- rollback checkpoint capture (feature 020·30) ----
    # All run main-thread inside the bridge _on_main blocks, BEFORE the mutation. Best-effort:
    # TurnCheckpoint's own try/except swallows a capture failure so the edit never fails (decision 10).

    def _capture_document(self, document_id: str) -> None:
        # Serialize the LIVE document (not the stale on-disk dir — set_uniform never writes document.json).
        # Also carry the document's scripts/script.py into the snapshot dir: UIDocument.save omits scripts/,
        # so without this a document-restore swap would DELETE an existing script.
        cp = self._get_active_checkpoint()
        document = self._get_ui_documents().get(document_id)
        if cp is None or document is None:
            return
        cp.snapshot_document(
            document_id,
            document,
            lambda n, dest: n.save(dest.parent, dest.name, rebind=False),
        )
        cp.snapshot_script(document_id, self._get_script_path(document_id))

    def _capture_script(self, document_id: str) -> None:
        # Pre-write capture for a script-mutating tool (043): snapshot a pre-existing script.py into
        # the document snapshot dir (restored by the document's full-dir swap), or mark a created-this-turn
        # script for delete-on-revert. Best-effort, runs main-thread before the write.
        cp = self._get_active_checkpoint()
        if cp is None:
            return
        if self._get_script_path(document_id).is_file():
            self._capture_document(document_id)
        else:
            cp.mark_created_script(document_id)

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
        # full document-id -> shortest unique prefix (>=DOCUMENT_SHORT_ID_LEN); on collision ALL ids grow
        # together so display + resolve stay consistent.
        ids = list(self._get_ui_documents())
        n = DOCUMENT_SHORT_ID_LEN
        while n < _COPILOT_FULL_ID_LEN:
            prefixes = [i[:n] for i in ids]
            if len(set(prefixes)) == len(prefixes):
                break
            n += 1
        return {i: i[:n] for i in ids}

    def _copilot_resolve_document_id(self, handle: str) -> str | None:
        # Handle (full id, short id, or unique prefix) -> full document-id, or None if no/ambiguous match.
        # Empty handle is unresolvable on purpose (else it'd resolve to the sole document — a required
        # target must reject, not fall back to current).
        if not handle.strip():
            return None
        if handle in self._get_ui_documents():
            return handle
        matches = [i for i in self._get_ui_documents() if i.startswith(handle)]
        return matches[0] if len(matches) == 1 else None

    def document_tree(self) -> list[DocumentTreeEntry]:
        # GL-FREE (runs off-main building prompt context): name + has_errors (cached) + is_current.
        # No uniforms (that's a GL read). document_id is the short id.
        current = self._get_current_document_id()
        short = self._copilot_short_ids()
        return [
            DocumentTreeEntry(
                document_id=short[nid],
                name=ui_document.ui_state.ui_name,
                # EVERY pass, not just the output's: a broken pass that nothing draws would
                # otherwise report the document clean.
                has_errors=any(
                    p.compile_unit.errors for p in ui_document.document.passes.values()
                ),
                is_current=(nid == current),
                passes=tuple(sorted(ui_document.document.passes)),
                output_pass=ui_document.document.graph.output,
            )
            for nid, ui_document in self._get_ui_documents().items()
        ]

    def example_catalog(self) -> list[ExampleEntry]:
        # GL-FREE: the shipped examples, addressed by a `example:<4-char>` handle (never the uuid).
        # Description is the merged override-or-shipped value, sanitized.
        return [
            ExampleEntry(
                example_id=example_address(tid),
                name=ui_document.ui_state.ui_name,
                description=sanitize_display(self._example_description(tid)),
            )
            for tid, ui_document in self._get_ui_document_examples().items()
        ]

    def _copilot_resolve_example_id(self, handle: str) -> str | None:
        # Example handle (`example:`-prefixed, short, or full uuid) -> full uuid, or None if no/ambiguous.
        # Forgiving: also matches an example by its DISPLAY NAME (case-insensitive) — the model copies the
        # human half of the `example:<id> | <name>` catalogue, so a bare name must resolve, not hard-fail.
        examples = self._get_ui_document_examples()
        h = strip_example_prefix(handle).strip()
        if not h:
            return None
        if h in examples:
            return h
        prefix = [tid for tid in examples if tid.startswith(h)]
        if len(prefix) == 1:
            return prefix[0]
        named = [
            tid
            for tid, n in examples.items()
            if n.ui_state.ui_name.casefold() == h.casefold()
        ]
        return named[0] if len(named) == 1 else None

    def _copilot_resolve_source(self, handle: str) -> tuple[str, str | None]:
        # read/grep addressing: `example:` -> EXAMPLE, else DOCUMENT. Returns (kind, full_id|None).
        # lib: falls through to the document resolver and returns None (read_shaders short-circuits
        # lib addresses before calling this).
        if is_example_address(handle):
            return "example", self._copilot_resolve_example_id(handle)
        return "document", self._copilot_resolve_document_id(handle)

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

    def read_shaders(self, document_ids: list[str]) -> list[ShaderView]:
        # Marshalled (compile + uniform read are GL). Per handle: compile, read source + uniforms +
        # errors, add to the working set. Unknown handles skipped. ShaderView carries the short id.
        # A `lib:` handle reads the library file whole (grep origins advertise lib: as a read
        # handle — the read side honors the same address space as edit_shader).
        def _on_main() -> list[ShaderView]:
            short = self._copilot_short_ids()
            # [] -> the current document (resolved here so a concrete id is what gets stamped).
            handles = document_ids or [self._get_current_document_id()]
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
                            document_id=lib_view.address,
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
                if kind == "example":
                    # Read-only: same view, not added to the working set, addressed by `example:` handle.
                    ui_document = self._get_ui_document_examples()[full_id]
                    view_id = example_address(full_id)
                else:
                    ui_document = self._get_ui_documents()[full_id]
                    view_id = short[full_id]
                document = ui_document.document
                if document.render_pass.program is None:
                    document.render_pass.compile()
                text = document.render_pass.source.text
                if kind == "document":
                    self._working_set_add(full_id)
                # A document-scoped set of (pass, name) pairs (069): this listing formats the
                # OUTPUT pass, so filter to its own name — a uniform driven only on `paint` is not
                # driven here.
                driven = (
                    _driven_on(
                        self._get_script_driven_uniforms(full_id),
                        pass_name_of(document.render_pass.source.path),
                    )
                    if kind == "document"
                    else set()
                )
                views.append(
                    ShaderView(
                        document_id=view_id,
                        name=ui_document.ui_state.ui_name,
                        listing=_number_lines(text),
                        uniforms=_format_uniforms(
                            document.render_pass,
                            driven,
                            _wired_for(document, document.render_pass),
                        ),
                        errors=_to_error_infos(
                            document.render_pass.compile_unit.errors
                        ),
                    )
                )
            return views

        return self._bridge.run_on_main(_on_main)

    def read_working_set(self) -> tuple[list[WorkingSetView], list[str]]:
        # Rebuild the working set into live views (marshalled: uniform read + recompile are GL).
        # Current document unioned in first (so the rendered set is the size cap + 1 at most), then
        # touched addresses in add-order; gone documents skipped. A program-less document is recompiled here
        # so its source and errors are coherent. The second element is this turn's evictions, as
        # AGENT-FACING handles and minus anything the block still renders (an evicted document that the
        # current-document union brought back shows its full source — calling it dropped would be a
        # falsehood on the model channel).
        def _on_main() -> tuple[list[WorkingSetView], list[str]]:
            short = self._copilot_short_ids()
            current = self._get_current_document_id()
            ordered: list[str] = []
            if current and current in self._get_ui_documents():
                ordered.append(current)
            for address in self._working_set_reader():
                # A pass address collapses to its document: one member is one DOCUMENT (D11), so
                # an 8-pass document can never evict its own passes out of the size cap.
                document_address = split_pass_address(address)[0]
                if document_address not in ordered:
                    ordered.append(document_address)
            views: list[WorkingSetView] = []
            for address in ordered:
                if is_lib_address(address):
                    view = self._copilot_lib_working_view(address)
                else:
                    view = self._copilot_document_working_view(address, short, current)
                if view is not None:
                    views.append(view)
            rendered = {v.address for v in views}
            evicted = [
                handle
                for handle in (
                    address
                    if is_lib_address(address)
                    else short.get(address, address[:DOCUMENT_SHORT_ID_LEN])
                    for address in self._working_set_evicted()
                )
                if handle not in rendered
            ]
            return views, evicted

        return self._bridge.run_on_main(_on_main)

    def _copilot_document_working_view(
        self, full_id: str, short: dict[str, str], current: str
    ) -> WorkingSetView | None:
        ui_document = self._get_ui_documents().get(full_id)
        if ui_document is None:
            return None
        document = ui_document.document
        for render_pass in document.passes.values():
            if render_pass.program is None:
                render_pass.compile()
        script_text, status = self._get_script_source_view(full_id)
        script_errors: list[CompileErrorInfo] = []
        if status is not None and status.sentinel_error is not None:
            script_errors = [_script_error_info(status.sentinel_error)]
        return WorkingSetView(
            address=short.get(full_id, full_id),
            name=ui_document.ui_state.ui_name,
            listing=_number_lines(document.render_pass.source.text),
            is_current=(full_id == current),
            is_lib=False,
            uniforms=_format_uniforms(
                document.render_pass,
                _driven_on(
                    self._get_script_driven_uniforms(full_id),
                    pass_name_of(document.render_pass.source.path),
                ),
                _wired_for(document, document.render_pass),
            ),
            errors=_to_error_infos(document.render_pass.compile_unit.errors),
            script_listing=_number_lines(script_text) if script_text else "",
            script_errors=script_errors,
            canvas=f"{document.canvas_size[0]}x{document.canvas_size[1]}",
            passes=self._pass_views(full_id, short, document),
        )

    def _pass_views(
        self, full_id: str, short: dict[str, str], document: Document
    ) -> list[PassView]:
        # Empty for a single-pass document: its one pass IS the member's own listing/uniforms/
        # errors, so the ordinary case stays byte-identical to the pre-graph prompt.
        if len(document.passes) < 2:
            return []
        handle = short.get(full_id, full_id)
        driven = self._get_script_driven_uniforms(full_id)
        # Sampler names FIRST, and only then the wiring: `_sampler_uniform_names` goes through
        # `get_active_uniforms`, which compiles a never-attempted pass -- and the name rule
        # resolves from COMPILED programs, so resolving first would see none of them and report
        # every name-wired sampler as BLACK on the first read of a freshly opened document.
        for render_pass in document.passes.values():
            _sampler_uniform_names(render_pass)
        # The EFFECTIVE wiring (069 D9, 072): a sampler the name rule fills has no stored row,
        # and telling the model it reads BLACK while the renderer fills it is a false fact.
        wiring = document.effective_wiring()
        views: list[PassView] = []
        for name in sorted(document.passes):
            render_pass = document.passes[name]
            views.append(
                PassView(
                    name=name,
                    address=pass_address(handle, name),
                    listing=_number_lines(render_pass.source.text),
                    # Per PASS (069): the driven set is document-scoped pairs, so each pass gets
                    # only what the script drives ON IT — the same uniform name on a sibling pass
                    # keeps its real value rather than a phantom marker.
                    uniforms=_format_uniforms(
                        render_pass, _driven_on(driven, name), wiring.get(name, {})
                    ),
                    errors=_to_error_infos(render_pass.compile_unit.errors),
                    is_output=(name == document.graph.output),
                )
            )
        return views

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
        # GL-FREE case-sensitive substring search across document / example / lib sources. Each hit is
        # origin-labelled (document id, `example:` handle, or lib: address) for a follow-up read.
        if not query:
            return []
        short = self._copilot_short_ids()
        hits: list[GrepHit] = []
        for document_id, ui_document in self._get_ui_documents().items():
            label = f"document '{ui_document.ui_state.ui_name}'"
            for i, line in enumerate(
                ui_document.document.render_pass.source.text.split("\n"), start=1
            ):
                if query in line:
                    hits.append(
                        GrepHit(
                            origin=short[document_id],
                            location=label,
                            line=i,
                            text=line.strip(),
                        )
                    )
        for tid, ui_document in self._get_ui_document_examples().items():
            origin = example_address(tid)
            label = f"example '{ui_document.ui_state.ui_name}'"
            for i, line in enumerate(
                ui_document.document.render_pass.source.text.split("\n"), start=1
            ):
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

    def set_uniform(self, name: str, value: object, document: str) -> SetUniformResult:
        # Set a uniform value on a document (marshalled: validation + try_to_release touch GL). The write
        # mirrors the UI widget (release old, dict-assign); next render picks it up. Up-front validation
        # is the only feedback channel (the render-time shape-pop is off-thread). Rejects samplers,
        # blocks, and engine-driven uniforms.
        def _on_main() -> SetUniformResult:
            document_id = (
                self._copilot_resolve_document_id(document)
                if document
                else self._get_current_document_id()
            )
            if document_id is None or document_id not in self._get_ui_documents():
                return SetUniformResult(
                    ok=False,
                    error=f"no document with id '{document}' — check the project map",
                )
            if name in ENGINE_DRIVEN_UNIFORMS:
                return SetUniformResult(
                    ok=False,
                    error=f"'{name}' is engine-owned (ShaderBox provides its value) — it "
                    "cannot be set; change the shader code if you need different behavior",
                )
            target = self._get_ui_documents()[document_id].document
            uniform = next(
                (u for u in target.render_pass.get_active_uniforms() if u.name == name),
                None,
            )
            if uniform is None:
                return SetUniformResult(
                    ok=False,
                    error=f"document has no active uniform '{name}' — read_shader it to see its "
                    "uniforms",
                )
            # The tool addresses the OUTPUT pass, so the reject asks the output PAIR (069): a
            # uniform of the same name driven on another pass is a legitimate manual edit here. It
            # sits below the resolution so a name absent from the output pass gets the "no active
            # uniform" answer rather than the script-driven one.
            if (
                pass_name_of(target.render_pass.source.path),
                name,
            ) in self._get_script_driven_uniforms(document_id):
                # One script per document (048): a driven uniform is always computed by the document script.
                return SetUniformResult(
                    ok=False,
                    error=f"'{name}' is script-driven (the document script computes its value each "
                    "frame) — a set here would be overwritten next tick; edit it with "
                    "edit_script/write_script instead",
                )
            label = gl_type_label(uniform)
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
            self._capture_document(
                document_id
            )  # pre-change rollback snapshot (best-effort)
            try_to_release(target.render_pass.uniform_values.get(name))
            target.render_pass.uniform_values[name] = coerced
            return SetUniformResult(
                ok=True,
                type_label=label,
                render_facts=self._render_facts_for(
                    target, motion=True, cache_key=document_id
                ),
            )

        return self._bridge.run_on_main(_on_main)

    def _create_document_on_main(
        self, name: str, source: str, example: str, switch_to: bool
    ) -> tuple[str, list[CompileErrorInfo], str]:
        # MAIN THREAD. Create a document from `example` (empty = the default starter); `source` overrides
        # the body. Order: save -> insert -> set-current. Adds to the working set; compiles + returns
        # errors. Called marshalled by create_document, and directly by import_picked_document (already on main).
        example_id = (
            self._copilot_resolve_example_id(example)
            if example.strip()
            else self._starter_example_id
        )
        if example_id is None:
            raise RuntimeError(f"no example matching '{example}'")
        example_dir = self._document_examples_dir / example_id
        if not example_dir.is_dir():
            # Missing only on a broken install; the registry turns the raise into a tool error.
            raise RuntimeError("starter example is missing")
        new_document = load_document_from_dir(example_dir)
        new_document.reset_id()
        if name.strip():
            new_document.ui_state.ui_name = name.strip()
        if source.strip():
            # release_program sets source.text; save_ui_document writes + rebinds source.path. Do NOT
            # write through source.path here — it still points at the shared starter example.
            new_document.document.render_pass.release_program(
                source.replace("\r\n", "\n").replace("\r", "\n")
            )
        # Compile (GL, main-thread) BEFORE save so the persisted program matches the reported errors.
        new_document.document.render_pass.compile()
        # source.path still points at the example dir here; save rebinds it.
        pre_save_path = str(new_document.document.render_pass.source.path)
        self._save_ui_document(new_document)
        self._get_ui_documents()[new_document.id] = new_document
        cp = self._get_active_checkpoint()
        if cp is not None:
            cp.mark_created(new_document.id)  # reverse = delete-to-trash, no snapshot
        if switch_to:
            self._set_current_document_id(new_document.id)
        self._working_set_add(new_document.id)
        persisted_path = str(new_document.document.render_pass.source.path)
        errors = [
            replace(e, path=persisted_path) if e.path == pre_save_path else e
            for e in _to_error_infos(
                new_document.document.render_pass.compile_unit.errors
            )
        ]
        logger.info(
            f"copilot created document {new_document.id} (switch_to={switch_to}, "
            f"errors={len(errors)})"
        )
        if errors:
            extra = "\n".join(
                compile_hints(
                    new_document.document.render_pass.source.text,
                    [e.message for e in errors],
                )
            )
        else:
            extra = self._render_facts_for(new_document.document, motion=True)
            self._last_clean[new_document.id] = (
                new_document.document.render_pass.source.text
            )
        # Short id, computed after insert so it's in the current id set.
        return self._copilot_short_ids()[new_document.id], errors, extra

    def create_document(
        self, name: str, source: str, example: str, switch_to: bool
    ) -> tuple[str, list[CompileErrorInfo], str]:
        return self._bridge.run_on_main(
            lambda: self._create_document_on_main(name, source, example, switch_to)
        )

    def delete_document(self, document: str) -> DeleteDocumentResult:
        # Delete a document (already user-confirmed). Marshals the GL teardown to main; returns document_id +
        # trash dir-name so the chat can offer a Recover.
        def _on_main() -> DeleteDocumentResult:
            document_id = self._copilot_resolve_document_id(document)
            if document_id is None or document_id not in self._get_ui_documents():
                return DeleteDocumentResult(
                    ok=False,
                    error=f"no such document '{document}' — check the project map for ids",
                )
            name = self._get_ui_documents()[document_id].ui_state.ui_name
            trash_name = self._delete_document_unguarded_cb(document_id)
            cp = self._get_active_checkpoint()
            if cp is not None:
                cp.record_deleted(
                    document_id, trash_name
                )  # reverse = restore from trash
            logger.info(f"copilot deleted document {document_id} (trash={trash_name})")
            return DeleteDocumentResult(
                ok=True,
                deleted_name=name,
                document_id=document_id,
                trash_name=trash_name,
            )

        return self._bridge.run_on_main(_on_main)

    def switch_document(self, document: str) -> SwitchDocumentResult:
        # Make `document` current (publish/render/untargeted-edit act on it). State write -> main thread;
        # the document joins the working set.
        def _on_main() -> SwitchDocumentResult:
            document_id = self._copilot_resolve_document_id(document)
            if document_id is None or document_id not in self._get_ui_documents():
                return SwitchDocumentResult(
                    ok=False,
                    error=f"no such document '{document}' — check the project map for ids",
                )
            ui_document = self._get_ui_documents()[document_id]
            cp = self._get_active_checkpoint()
            if cp is not None:
                cp.record_pre_switch(self._get_current_document_id())
            self._set_current_document_id(document_id)
            self._working_set_add(document_id)
            logger.info(f"copilot switched current document to {document_id}")
            return SwitchDocumentResult(ok=True, name=ui_document.ui_state.ui_name)

        return self._bridge.run_on_main(_on_main)

    def rename_document(self, document: str, new_name: str) -> DocumentOpResult:
        def _on_main() -> DocumentOpResult:
            document_id = self._copilot_resolve_document_id(document)
            if document_id is None or document_id not in self._get_ui_documents():
                return DocumentOpResult(
                    ok=False,
                    error=f"no such document '{document}' — check the project map for ids",
                )
            name = new_name.strip()
            if not name:
                return DocumentOpResult(ok=False, error="new_name is empty")
            ui_document = self._get_ui_documents()[document_id]
            self._capture_document(
                document_id
            )  # pre-change rollback snapshot (best-effort)
            ui_document.ui_state.ui_name = name
            self._save_ui_document(ui_document)
            logger.info(f"copilot renamed document {document_id} -> {name!r}")
            return DocumentOpResult(ok=True, name=name)

        return self._bridge.run_on_main(_on_main)

    def set_canvas_size(
        self, document: str, width: int, height: int
    ) -> DocumentOpResult:
        def _on_main() -> DocumentOpResult:
            document_id = self._copilot_resolve_document_id(document)
            if document_id is None or document_id not in self._get_ui_documents():
                return DocumentOpResult(
                    ok=False,
                    error=f"no such document '{document}' — check the project map for ids",
                )
            w, h = clamp_canvas_size((width, height))
            ui_document = self._get_ui_documents()[document_id]
            self._capture_document(
                document_id
            )  # pre-change rollback snapshot (best-effort)
            ui_document.document.set_canvas_size((w, h))
            self._save_ui_document(ui_document)
            logger.info(f"copilot set canvas of {document_id} -> {w}x{h}")
            return DocumentOpResult(ok=True, width=w, height=h)

        return self._bridge.run_on_main(_on_main)

    def duplicate_document(
        self, document: str, new_name: str, switch_to: bool
    ) -> tuple[str, list[CompileErrorInfo], str]:
        # Fork a document: persist the live source, load its dir as an independent document (deep copy incl.
        # media/ + script), give it a fresh id, compile, save + insert. Mirrors create_document's tail.
        def _on_main() -> tuple[str, list[CompileErrorInfo], str]:
            document_id = self._copilot_resolve_document_id(document)
            if document_id is None or document_id not in self._get_ui_documents():
                raise RuntimeError(f"no such document '{document}'")
            source_document = self._get_ui_documents()[document_id]
            self._save_ui_document(
                source_document
            )  # persist live state so the load is a full copy
            src_dir = document_dir_of(source_document.document)
            new_document = load_document_from_dir(src_dir)
            new_document.reset_id()
            name = new_name.strip() or f"{source_document.ui_state.ui_name} copy"
            new_document.ui_state.ui_name = name
            new_document.document.render_pass.compile()
            pre_save_path = str(new_document.document.render_pass.source.path)
            self._save_ui_document(new_document)
            self._get_ui_documents()[new_document.id] = new_document
            cp = self._get_active_checkpoint()
            if cp is not None:
                cp.mark_created(new_document.id)  # reverse = delete-to-trash
            if switch_to:
                self._set_current_document_id(new_document.id)
            self._working_set_add(new_document.id)
            persisted_path = str(new_document.document.render_pass.source.path)
            errors = [
                replace(e, path=persisted_path) if e.path == pre_save_path else e
                for e in _to_error_infos(
                    new_document.document.render_pass.compile_unit.errors
                )
            ]
            logger.info(
                f"copilot duplicated {document_id} -> {new_document.id} ({name!r})"
            )
            if errors:
                extra = "\n".join(
                    compile_hints(
                        new_document.document.render_pass.source.text,
                        [e.message for e in errors],
                    )
                )
            else:
                extra = self._render_facts_for(new_document.document, motion=True)
                self._last_clean[new_document.id] = (
                    new_document.document.render_pass.source.text
                )
            return self._copilot_short_ids()[new_document.id], errors, extra

        return self._bridge.run_on_main(_on_main)

    def delete_lib_file(self, path: str) -> LibFileResult:
        # Trash a lib file (already user-confirmed via the ALWAYS gate). Capture pre-delete bytes for
        # revert, invalidate consumers WHILE the path still resolves, then move to .trash.
        def _on_main() -> LibFileResult:
            files = self._get_shader_lib_files()
            resolved = files.resolve_copilot_path(strip_lib_prefix(path))
            if resolved is None or not resolved.exists():
                return LibFileResult(
                    ok=False,
                    error=f"no library file at '{path}' — copy a lib: address from the "
                    "catalogue or grep",
                )
            self._capture_lib(
                path, resolved.read_text(encoding="utf-8"), lib_create=False
            )
            self.invalidate_lib_consumers(resolved)
            files.delete_file(resolved)
            logger.info(f"copilot deleted lib file {path}")
            return LibFileResult(ok=True)

        return self._bridge.run_on_main(_on_main)

    def bind_media(self, document: str, uniform: str) -> MediaBindResult:
        # Validate the sampler on main FIRST (so a bad target rejects BEFORE a picker opens), then
        # block on the FILE gate. The UI poll opens the OS picker, does the load+bind on main
        # (bind_picked_media), and answers with a path-free result — the abs path never reaches here.
        def _validate() -> tuple[str, str]:
            document_id = self._copilot_resolve_document_id(document)
            if document_id is None or document_id not in self._get_ui_documents():
                return (
                    "",
                    f"no such document '{document}' — check the project map for ids",
                )
            n = self._get_ui_documents()[document_id].document
            if n.render_pass.program is None:
                n.render_pass.compile()
            samplers = [
                u.name
                for u in n.render_pass.get_active_uniforms()
                if gl_type_label(u) == "sampler2D"
            ]
            if uniform not in samplers:
                listed = ", ".join(samplers) or "(none)"
                return "", (
                    f"'{uniform}' is not a sampler2D on this document; its samplers: {listed}. "
                    "Declare `uniform sampler2D <name>;` in the source first if you need one."
                )
            return document_id, ""

        document_id, err = self._bridge.run_on_main(_validate)
        if err:
            return MediaBindResult(ok=False, error=err)
        resp = self._get_gate().ask_file(
            GateRequest(
                kind=GateKind.FILE,
                prompt=f"Choose an image or video for {uniform}",
                document_id=document_id,
                uniform=uniform,
                file_kinds=("image", "video"),
            )
        )
        if resp.media_result is None:
            return MediaBindResult(cancelled=True)
        return resp.media_result

    def bind_picked_media(
        self, document_id: str, uniform: str, path: Path
    ) -> MediaBindResult:
        # MAIN THREAD (called by the UI FILE-gate poll, already on the GL thread — NOT bridged). Loads
        # the user-picked file + binds it to the sampler. The path lives ONLY here + the poll; the
        # returned result is path-free.
        ui_document = self._get_ui_documents().get(document_id)
        if ui_document is None:
            return MediaBindResult(ok=False, error="the document is gone")
        self._capture_document(
            document_id
        )  # pre-change rollback snapshot (best-effort)
        try:
            media = media_class_for(path.suffix)(path)
        except Exception as exc:
            logger.warning(f"copilot bind_media load failed for {path.name}: {exc}")
            return MediaBindResult(
                ok=False, error=f"could not load '{path.name}' ({type(exc).__name__})"
            )
        try_to_release(ui_document.document.render_pass.uniform_values.get(uniform))
        ui_document.document.render_pass.uniform_values[uniform] = media
        self._save_ui_document(ui_document)
        d = media.details
        logger.info(f"copilot bound media -> {document_id}.{uniform} ({path.name})")
        return MediaBindResult(
            ok=True,
            basename=path.name,
            width=d.resolution_details.width,
            height=d.resolution_details.height,
            is_video=d.is_video,
        )

    def unbind_media(self, document: str, uniform: str) -> MediaBindResult:
        # Return a sampler to undecided (no picker): the name rule fills it or it reads black,
        # and save() writes no row for it.
        def _on_main() -> MediaBindResult:
            document_id = self._copilot_resolve_document_id(document)
            if document_id is None or document_id not in self._get_ui_documents():
                return MediaBindResult(
                    ok=False,
                    error=f"no such document '{document}' — check the project map for ids",
                )
            ui_document = self._get_ui_documents()[document_id]
            n = ui_document.document
            if n.render_pass.program is None:
                n.render_pass.compile()
            is_sampler = any(
                u.name == uniform and gl_type_label(u) == "sampler2D"
                for u in n.render_pass.get_active_uniforms()
            )
            if not is_sampler:
                return MediaBindResult(
                    ok=False, error=f"'{uniform}' is not a sampler2D on this document"
                )
            self._capture_document(document_id)
            try_to_release(n.render_pass.uniform_values.get(uniform))
            n.render_pass.uniform_values[uniform] = AutoSource()
            self._save_ui_document(ui_document)
            logger.info(f"copilot unbound media on {document_id}.{uniform}")
            return MediaBindResult(ok=True)

        return self._bridge.run_on_main(_on_main)

    def import_document(self, switch_to: bool) -> DocumentImportResult:
        # Worker: block on a FILE gate for a .glsl; the UI poll reads it on main (import_picked_document)
        # and answers with a path-free result.
        resp = self._get_gate().ask_file(
            GateRequest(
                kind=GateKind.FILE,
                prompt="Choose a .glsl shader to import",
                file_kinds=("glsl",),
                file_action="import_document",
                switch_to=switch_to,
            )
        )
        if resp.import_result is None:
            return DocumentImportResult(cancelled=True)
        return resp.import_result

    def import_picked_document(
        self, path: Path, switch_to: bool
    ) -> DocumentImportResult:
        # MAIN THREAD (called by the UI FILE-gate poll). Read the picked file + create a document from it.
        # The path lives only here; the result is path-free (basename only).
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning(
                f"copilot import_document read failed for {path.name}: {exc}"
            )
            return DocumentImportResult(
                ok=False, error=f"could not read '{path.name}' ({type(exc).__name__})"
            )
        try:
            document_id, errors, _extra = self._create_document_on_main(
                path.stem, text, "", switch_to
            )
        except Exception as exc:
            logger.warning(f"copilot import_document create failed: {exc}")
            return DocumentImportResult(
                ok=False, error=f"import failed ({type(exc).__name__})"
            )
        logger.info(f"copilot imported document {document_id} from {path.name}")
        return DocumentImportResult(
            ok=True, document_id=document_id, errors=errors, basename=path.name
        )

    def _copilot_render_path(self, document: UIDocument, ext: str) -> Path:
        # Non-colliding filename <name>_<short-id>_<n>.<ext>, n = next free index in renders_dir.
        base = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in document.ui_state.ui_name
        )
        short = self._copilot_short_ids().get(
            document.id, document.id[:DOCUMENT_SHORT_ID_LEN]
        )
        renders = self._get_renders_dir()
        n = 0
        while True:
            candidate = renders / f"{base}_{short}_{n}.{ext}"
            if not candidate.exists():
                return candidate
            n += 1

    def render_image(self, document: str, shape: RenderShape) -> RenderResult:
        # Render the current frame to a PNG (GL; marshalled with the longer render_op_timeout_s).
        def _on_main() -> RenderResult:
            ui_document = self._copilot_render_target(document)
            if ui_document is None:
                return RenderResult(ok=False, error=f"no such document '{document}'")
            preset = shape_to_preset(
                shape, is_video=False, fps=None, container=None, duration_max=None
            )
            out = self._copilot_render_path(ui_document, "png")
            art = render_job.render_to(ui_document.document, preset, 0.0, out)
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
            _on_main, timeout=COPILOT_ENGINE.render_op_timeout_s, defer=True
        )

    def render_video(
        self, document: str, seconds: float, fps: int, shape: RenderShape
    ) -> RenderResult:
        # Render `seconds` of animation (from t=0) to a WebM.
        def _on_main() -> RenderResult:
            ui_document = self._copilot_render_target(document)
            if ui_document is None:
                return RenderResult(ok=False, error=f"no such document '{document}'")
            preset = shape_to_preset(
                shape, is_video=True, fps=fps, container=".webm", duration_max=None
            )
            out = self._copilot_render_path(ui_document, "webm")
            art = render_job.render_to(ui_document.document, preset, seconds, out)
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
            _on_main, timeout=COPILOT_ENGINE.render_op_timeout_s, defer=True
        )

    def probe_render(self, document: str, t: float) -> str:
        # The aimable read-side probe (feature 050): the one-line facts string the edit path
        # produces, at a chosen `t` (default 0.0 = the export clock). UN-gated + non-mutating - the
        # render-blind agent glances here as often as it likes, vs render_image (gated, writes a
        # deliverable file).
        def _on_main() -> str:
            ui_document = self._copilot_render_target(document)
            if ui_document is None:
                return f"error: no such document '{document}'"
            facts = self._render_facts_for(ui_document.document, t=t)
            return facts or "probe rendered, but produced no readable facts (advisory)."

        return self._bridge.run_on_main(
            _on_main, timeout=COPILOT_ENGINE.render_op_timeout_s, defer=True
        )

    def _copilot_render_target(self, document: str) -> UIDocument | None:
        document_id = (
            self._copilot_resolve_document_id(document)
            if document
            else self._get_current_document_id()
        )
        if document_id is None or document_id not in self._get_ui_documents():
            return None
        return self._get_ui_documents()[document_id]

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
        document_id = self._get_current_document_id()
        if document_id is None or document_id not in self._get_ui_documents():
            return PublishResult(
                ok=False, error="no current document to publish", kind=kind
            )
        ui_document = self._get_ui_documents()[document_id]

        def _render_and_enqueue() -> ExportProgress | None:
            duration = float(settings.get("seconds", preset.duration_max or 3.0))
            out = self._copilot_render_path(ui_document, render_job.preset_ext(preset))
            art = render_job.render_to(ui_document.document, preset, duration, out)
            if art is None:
                raise CopilotToolError("render failed")
            baseline = exporter.status().last_progress
            exporter.publish(art, settings)
            return baseline

        try:
            # Held for the whole wait so the terminal can't be a different object at the same address.
            baseline = self._bridge.run_on_main(
                _render_and_enqueue,
                timeout=COPILOT_ENGINE.render_op_timeout_s,
                defer=True,
            )
        except CopilotToolError:
            return PublishResult(ok=False, error="render failed (see logs)", kind=kind)

        deadline = time.monotonic() + COPILOT_ENGINE.publish_await_timeout_s
        while time.monotonic() < deadline:
            if self._get_is_cancelled():
                return PublishResult(ok=False, error="cancelled", kind=kind)
            time.sleep(COPILOT_ENGINE.publish_poll_interval_s)
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
        self, title: str, description: str, shape: RenderShape
    ) -> PublishResult:
        exporter = self._get_exporter_registry().get("youtube")
        if not isinstance(exporter, YouTubeExporter):
            return PublishResult(
                ok=False, error="YouTube exporter unavailable", kind="youtube"
            )

        # Drive the shape from the arg so the render preset + the #Shorts upload flag agree; restore
        # the user's Share-tab shape after (a FULL RenderShape, so a WIDE_1440 choice round-trips
        # losslessly). set_shape writes _render_state.shape, which the main-thread Share tab reads —
        # so the shape mutations marshal to main (the publish itself does its own bridge ops).
        def _set_shape_read_preset() -> tuple[RenderShape, RenderPreset, bool]:
            prior = exporter.current_shape()
            exporter.set_shape(shape)
            return prior, exporter.render_preset(), exporter.current_is_short()

        prior_shape, preset, is_short_now = self._bridge.run_on_main(
            _set_shape_read_preset
        )
        try:
            settings: dict[str, Any] = {
                "title": title,
                "description": description,
                "is_short": is_short_now,
                "seconds": preset.duration_max or 6.0,
            }
            return self._copilot_publish(exporter, "youtube", preset, settings)
        finally:
            self._bridge.run_on_main(lambda: exporter.set_shape(prior_shape))

    def has_current_document(self) -> bool:
        return self._get_current_document_id() in self._get_ui_documents()

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
        deadline = time.monotonic() + COPILOT_ENGINE.telegram_connect_timeout_s
        while time.monotonic() < deadline:
            if self._get_is_cancelled():
                return TelegramConnectResult(ok=False, error="cancelled")
            time.sleep(COPILOT_ENGINE.publish_poll_interval_s)
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
        deadline = time.monotonic() + COPILOT_ENGINE.publish_await_timeout_s
        while time.monotonic() < deadline:
            if self._get_is_cancelled():
                return TelegramOpResult(ok=False, error="cancelled")
            time.sleep(COPILOT_ENGINE.publish_poll_interval_s)
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

    # ---- edit / compile-feedback (target-addressable: document or lib: file, 020·16) ----

    def apply_shader_edit(
        self, old_str: str, new_str: str, replace_all: bool, target: str
    ) -> EditResult:
        # Match + replace against the target's live source, recompile (document) / write (lib), persist,
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
                comment_spans = comment_only_spans(src, old_str)
                if comment_spans is not None:
                    spans = comment_spans
            if not spans:
                return EditResult(
                    matches=0,
                    errors=[],
                    hint=whitespace_near_match(src, old_str),
                    target_label=tgt.label,
                )
            if len(spans) > 1 and not replace_all:
                return EditResult(matches=len(spans), errors=[])
            if any(span_drops_comment(src, s, e, old_str) for s, e in spans):
                return EditResult(matches=0, errors=[], comment_loss=True)
            new_text = splice(src, spans, new_str)
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
        self,
        ws_address: str,
        document: Document,
        render_pass: Pass,
        streak: int,
        matches: int,
    ) -> EditResult:
        # The 033 unstick: N consecutive broken edits -> put the file back at its last
        # clean-compiling state and tell the agent as a fact. Resets the streak so the
        # next broken run gets a fresh budget. Keyed by the working-set ADDRESS, which names one
        # FILE: a document's passes are separate files, so a document-id key would restore one
        # pass from another's clean state.
        restore_errors = self._copilot_persist_shader(
            render_pass, self._last_clean[ws_address]
        )
        self._broken_streak[ws_address] = 0
        logger.info(
            f"copilot force-restore | target={ws_address} after {streak} broken edits"
        )
        note = (
            f"EDIT UNDONE — {streak} consecutive edits left compile errors, so the file "
            "was restored to its last clean-compiling state (the working set below shows "
            "the restored source). Re-read it and rewrite the whole block in ONE edit."
        )
        if restore_errors:
            err_lines = format_compile_errors(restore_errors)
            note = (
                f"EDIT UNDONE — {streak} consecutive edits left compile errors; the "
                "file was restored to an earlier state, which itself no longer "
                f"compiles (likely a library change):\n{err_lines}"
            )
        facts = (
            self._render_facts_for(document, motion=True, cache_key=ws_address)
            if not restore_errors
            else ""
        )
        return EditResult(
            matches=matches,
            errors=restore_errors,
            restored_note=note,
            render_facts=facts,
        )

    def _render_facts_for(
        self,
        document: Document,
        t: float = 0.0,
        motion: bool = False,
        cache_key: str = "",
    ) -> str:
        # Best-effort probe render -> one facts line (feature 033). Runs on the main
        # thread (bridge-marshalled callers) with the GL context current. Never raises
        # into the edit path — facts are advisory. `t` is the render clock, DEFAULT 0.0 (the
        # export clock the user renders, NOT wall-clock — feature 050: a wall-clock probe drifts
        # with app uptime, so its facts can't be correlated with what the user sees). ONE source
        # for both document.render(u_time=) AND the stamp, so a caller passing its sample time (a
        # script probe, the probe_render tool) can't disagree with the rendered frame.
        # `motion=True` (mutation auto-probes): also render a SECOND frame at render_facts_motion_t
        # and append a STATIC/ANIMATES verdict — t=0 alone is blank for a ramping effect and reads
        # as a failed edit; the verdict + the later frame's facts say it develops over time.
        if not COPILOT_ENGINE.render_facts_enabled:
            return ""
        try:
            size = COPILOT_ENGINE.render_facts_size
            # Match the document's canvas aspect — a square probe would lay out
            # aspect-corrected shaders (u_aspect) differently from the preview.
            cw, ch = document.render_pass.canvas.texture.size
            h = min(4 * size, max(8, round(size * ch / cw))) if cw else size
            if self._probe_canvas is None:
                self._probe_canvas = Canvas(size=(size, h))
            else:
                self._probe_canvas.set_size((size, h))
            document.render(u_time=t, canvas=self._probe_canvas)
            raw0 = self._probe_canvas.texture.read()
            # Stamp the sample time: an animated shader's facts change with phase,
            # which otherwise reads as an edit effect.
            line0 = _stamp_facts(render_facts(raw0, size, h), t)
            if not motion or not line0:
                return line0
            t2 = COPILOT_ENGINE.render_facts_motion_t
            document.render(u_time=t2, canvas=self._probe_canvas)
            raw1 = self._probe_canvas.texture.read()
            a0 = np.frombuffer(raw0, dtype=np.uint8).astype(np.int16)
            a1 = np.frombuffer(raw1, dtype=np.uint8).astype(np.int16)
            if float(np.mean(np.abs(a0 - a1))) < _MOTION_EPS:
                result = f"{line0}\nmotion: STATIC (unchanged from t=0 to t={t2:.1f}s)"
            else:
                line1 = _stamp_facts(render_facts(raw1, size, h), t2)
                result = (
                    f"{line0}\n{line1}\nmotion: ANIMATES (the frame changes over time)"
                )
            # No-op detection: identical to the frames before this mutation = it changed nothing.
            if cache_key:
                prev = self._last_probe.get(cache_key)
                self._last_probe[cache_key] = (raw0, raw1)
                if prev is not None and prev[0] == raw0 and prev[1] == raw1:
                    result = (
                        "this mutation changed NOTHING on screen vs the frame before it — "
                        "dead code, the wrong document/target, a value a script overrides, or a "
                        "change only visible between t=0 and t=1.5s\n" + result
                    )
            return result
        except Exception as exc:  # — advisory channel, never break an edit
            logger.debug(f"copilot render facts skipped: {exc}")
            return ""

    def _script_render_line(
        self,
        document: Document,
        samples: list[tuple[float, dict[tuple[str, str], object]]],
    ) -> str:
        # ONE corroborating render (feature 043): render the mid sample's driven values AT the mid
        # sample's TIME to answer "did the values produce visible ink, or is it FLAT / off-screen / a
        # uniform the shader ignores?" — the honesty case a value-diff alone misses. The render clock IS
        # mid[0] (not wall-clock), so a u_time-reading shader renders the frame the values came from.
        # Rebinds EACH pass's uniform_values to a merged copy (the live dict OBJECTs are never
        # mutated; sampler/Video values are shared and may advance a frame, same as the 033 facts
        # probe), restores them in finally. Advisory — never raises.
        if not samples or not COPILOT_ENGINE.render_facts_enabled:
            return ""
        mid = samples[len(samples) // 2]
        saved = {name: p.uniform_values for name, p in document.passes.items()}
        try:
            # The sample is keyed by (pass, name) since 069, so the merge is per pass: a value the
            # script drives on a non-output pass must reach THAT pass for the render to show what
            # the values describe.
            for pass_name, render_pass in document.passes.items():
                render_pass.uniform_values = {
                    **saved[pass_name],
                    **{n: v for (p, n), v in mid[1].items() if p == pass_name},
                }
            return self._render_facts_for(document, t=mid[0])
        except Exception as exc:
            logger.debug(f"copilot script render facts skipped: {exc}")
            return ""
        finally:
            for pass_name, render_pass in document.passes.items():
                render_pass.uniform_values = saved[pass_name]

    def read_script(self, document: str, /) -> ScriptView:
        def _on_main() -> ScriptView:
            document_id = self._resolve_document_or_current(document)
            if document_id is None:
                return ScriptView(
                    "", "", "", [_no_document_error(document)], is_stub=False
                )
            self._working_set_add(
                document_id
            )  # so its SCRIPT sub-section rides the working set
            text, is_stub = self._read_script_source(document_id)
            _text, status = self._get_script_source_view(document_id)
            errors = (
                [_script_error_info(status.sentinel_error)]
                if status is not None and status.sentinel_error is not None
                else []
            )
            return ScriptView(
                document_id=self._copilot_short_ids().get(document_id, document_id),
                name=self._get_ui_documents()[document_id].ui_state.ui_name,
                listing=_number_lines(text),
                errors=errors,
                is_stub=is_stub,
            )

        return self._bridge.run_on_main(_on_main)

    def _apply_script_text(self, document_id: str, new_text: str) -> ScriptWriteResult:
        # The shared write tail: capture, persist+reload+dry-run, render the probe into a
        # ScriptWriteResult. write_script (whole file) and edit_script (after a splice) both end here, so
        # an edit and a write give IDENTICAL feedback. Runs on the main thread (the caller marshals).
        self._working_set_add(
            document_id
        )  # so the script rides the working set next step
        self._batch_mutated.add(("script", document_id))
        self._capture_script(document_id)
        probe = self._write_script_source(document_id, new_text)
        broken = (
            format_compile_errors([_script_error_info(probe.compile_error)])
            if probe.compile_error
            else (
                "ran, then "
                + format_compile_errors([_script_error_info(probe.runtime_error)])
                if probe.runtime_error
                else ""
            )
        )
        if broken:
            return self._script_broken_write(document_id, broken)
        # A clean probe: reset the streak, snapshot the source as the restore target.
        self._script_broken_streak[document_id] = 0
        self._script_last_clean[document_id] = new_text
        render_line = self._script_render_line(
            self._get_ui_documents()[document_id].document, probe.samples
        )
        prev_samples = self._last_script_samples.get(document_id)
        self._last_script_samples[document_id] = probe.samples
        motion_facts = _motion_verdict(
            probe, render_line, COPILOT_ENGINE.motion_value_eps
        )
        if prev_samples is not None and prev_samples == probe.samples:
            motion_facts += (
                "\nthis edit changed NOTHING in the driven values vs the edit before it "
                "(identical sampled values) — a dead store your own later code overwrites, "
                "dead code, or a text-only change. Do NOT re-apply the same edit; find where "
                "the EFFECTIVE value is computed."
            )
        return ScriptWriteResult(
            ok=True,
            driven=sorted(_dotted(k) for k in probe.driven),
            per_key_errors=[
                f"{_dotted((p, name))}: {err.message}"
                for p, name, err in probe.per_key_errors
            ],
            # A bare key no pass declares carries no pass, so it renders as the bare name.
            orphan_keys=[
                f"{_dotted((p, name)) if p else name}: {err.message}"
                for p, name, err in probe.orphan_keys
            ],
            motion_facts=motion_facts,
        )

    def _script_broken_write(self, document_id: str, error: str) -> ScriptWriteResult:
        # A broken script write/edit (compile OR runtime). After N in a row, restore the last clean
        # source + tell the agent (the 033 force-restore, ported to scripts). Mirrors the shader path.
        streak = self._script_broken_streak.get(document_id, 0) + 1
        self._script_broken_streak[document_id] = streak
        limit = COPILOT_CONFIG.auto_revert_after_failed_edits
        if limit > 0 and streak >= limit and document_id in self._script_last_clean:
            clean = self._script_last_clean[document_id]
            restore = self._write_script_source(document_id, clean)
            self._script_broken_streak[document_id] = 0
            logger.info(
                f"copilot script force-restore | document={document_id} after {streak}"
            )
            note = (
                f"SCRIPT RESTORED — {streak} broken script edits in a row, so the script was "
                "reverted to its last clean-running state (now in the working set). Re-read it and "
                "rewrite the whole script in ONE write_script."
            )
            if restore.compile_error is not None or restore.runtime_error is not None:
                note = (
                    f"{streak} broken edits in a row; the restore target also no longer runs "
                    "(likely a shader/uniform change) -- re-read and rewrite the whole script."
                )
            return ScriptWriteResult(ok=True, restored_note=note)
        return ScriptWriteResult(ok=True, compile_error=error)

    def write_script(self, new_text: str, document: str, /) -> ScriptWriteResult:
        def _on_main() -> ScriptWriteResult:
            document_id = self._resolve_document_or_current(document)
            if document_id is None:
                return ScriptWriteResult(
                    ok=False, error=_no_document_error(document).message
                )
            if ("script", document_id) in self._batch_mutated:
                return ScriptWriteResult(
                    ok=False, error=_batch_guard_reason("edit_script")
                )
            return self._apply_script_text(document_id, new_text)

        return self._bridge.run_on_main(
            _on_main, timeout=COPILOT_ENGINE.render_op_timeout_s
        )

    def apply_script_edit(
        self, old_str: str, new_str: str, replace_all: bool, document: str, /
    ) -> ScriptWriteResult:
        # The script analog of apply_shader_edit (mirror of edit_shader). Plain-TEXT match (a script is
        # Python, not GLSL — the glsl_lex token matcher does NOT apply. INDENT-AWARE match (exact
        # substring first, then an indent-level structural fallback that forgives a re-typed leading
        # indent + re-indents new_str onto the real column), same 0/1/N-match contract, then the same
        # write tail so an edit and a write give identical feedback. old_str/new_str are tab-normalized
        # to match the spaces-only on-disk source.
        def _on_main() -> ScriptWriteResult:
            document_id = self._resolve_document_or_current(document)
            if document_id is None:
                return ScriptWriteResult(
                    ok=False, error=_no_document_error(document).message
                )
            old_norm = normalize_script_tabs(old_str)
            new_norm = normalize_script_tabs(new_str)
            src, _is_stub = self._read_script_source(document_id)
            spans = script_match_spans(src, old_norm)
            if not spans:
                return ScriptWriteResult(
                    ok=False,
                    error="old_str not found in the script -- re-read it with read_script "
                    "and copy an exact substring",
                )
            if len(spans) > 1 and not replace_all:
                return ScriptWriteResult(
                    ok=False,
                    error=f"old_str is not unique ({len(spans)} matches) -- add "
                    "surrounding context to make it unique, or set replace_all=true",
                )
            new_text = splice_script(src, spans, new_norm)
            return self._apply_script_text(document_id, new_text)

        return self._bridge.run_on_main(
            _on_main, timeout=COPILOT_ENGINE.render_op_timeout_s
        )

    def _resolve_document_or_current(self, document: str) -> str | None:
        # A document-id handle (or "" = current) -> full id, or None when it resolves to no live document.
        # Scripts address only documents (one script per document), never lib/example handles.
        if not document:
            cur = self._get_current_document_id()
            return cur if cur and cur in self._get_ui_documents() else None
        kind, full_id = self._copilot_resolve_source(document)
        return full_id if kind == "document" and full_id is not None else None

    def _copilot_persist_shader(
        self, render_pass: Pass, new_text: str
    ) -> list[CompileErrorInfo]:
        # Adopt new_text, recompile, persist, refresh the editor — the shared tail of every source
        # edit. Takes the PASS, not the document: an edit addressed as "<id>#<pass>" must land on
        # that pass's file, and a bare id resolves to the output pass one layer up.
        # sync_editor keys on the edited FILE's path, else a non-current edit syncs the wrong
        # session; it no-ops when that file has no open editor.
        render_pass.release_program(new_text)
        render_pass.compile()
        render_pass.source.path.write_text(new_text, encoding="utf-8")
        self._sync_editor_from_disk(render_pass.source.path, new_text)
        return _to_error_infos(render_pass.compile_unit.errors)

    def _copilot_resolve_target(
        self, target: str, *, allow_create: bool
    ) -> "_CopilotEditTarget | EditResult":
        # Resolve an edit target to source + identity, or an EditResult REJECT. "lib:"-prefixed -> lib
        # file; empty -> current document; else a document-id (unknown is a hard error, never a lib fallback).
        if is_lib_address(target):
            return self._copilot_resolve_lib_target(target, allow_create=allow_create)
        if is_example_address(target):
            # Examples are read-only; an explicit guard with an actionable message (not silent non-resolution).
            return EditResult(
                matches=0,
                errors=[],
                unresolved=True,
                unresolved_reason="examples are read-only — create_document(example=...) from it "
                "first, then edit the resulting document",
            )
        document_target, pass_name = split_pass_address(target)
        if not document_target:
            document_id = self._get_current_document_id()
        else:
            resolved = self._copilot_resolve_document_id(document_target)
            if resolved is None:
                return EditResult(
                    matches=0,
                    errors=[],
                    unresolved=True,
                    unresolved_reason=f"no document with id '{document_target}' — use an id "
                    "from the project map",
                )
            document_id = resolved
        if document_id not in self._get_ui_documents():
            return EditResult(
                matches=0,
                errors=[],
                unresolved=True,
                unresolved_reason="that shader no longer exists — check the project map for ids",
            )
        ui_document = self._get_ui_documents()[document_id]
        short = self._copilot_short_ids().get(
            document_id, document_id[:DOCUMENT_SHORT_ID_LEN]
        )
        document = ui_document.document
        if pass_name and pass_name not in document.passes:
            return EditResult(
                matches=0,
                errors=[],
                unresolved=True,
                unresolved_reason=f"document '{short}' has no pass '{pass_name}' — its passes "
                f"are {sorted(document.passes)}",
            )
        render_pass = document.passes[pass_name] if pass_name else document.render_pass
        label = f"document '{ui_document.ui_state.ui_name}' ({short})"
        if pass_name:
            label += f" pass '{pass_name}'"
        if not target:
            label += " — target was empty, so this hit the CURRENT document"
        return _CopilotEditTarget(
            kind="document",
            document_id=document_id,
            document=document,
            render_pass=render_pass,
            source=render_pass.source.text,
            ws_address=pass_address(document_id, pass_name)
            if pass_name
            else document_id,
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
        # Persist an applied edit. A DOCUMENT recompiles + returns errors; a LIB file is written + returns
        # the "no standalone compile" note. On success the target joins the working set + is batch-mutated.
        # Model-supplied text is CRLF-normalized here, the seam every edit write flows through.
        new_text = new_text.replace("\r\n", "\n").replace("\r", "\n")
        if tgt.kind == "document":
            assert tgt.document is not None and tgt.document_id is not None
            assert tgt.render_pass is not None
            self._capture_document(
                tgt.document_id
            )  # pre-write rollback snapshot (best-effort)
            # "Clean" requires a LIVE program: an invalidated compile_unit has
            # errors=[] without one (e.g. after a lib edit) and must not anchor.
            prev_clean = (
                not tgt.render_pass.compile_unit.errors
                and tgt.render_pass.program is not None
            )
            errors = self._copilot_persist_shader(tgt.render_pass, new_text)
            self._working_set_add(tgt.ws_address)
            self._batch_mutated.add(tgt.ws_address)
            if errors:
                if prev_clean:
                    # A clean file just broke — this starts a NEW streak (anything
                    # earlier was already fixed, possibly outside the copilot).
                    self._last_clean[tgt.ws_address] = tgt.source
                    streak = 1
                else:
                    streak = self._broken_streak.get(tgt.ws_address, 0) + 1
                self._broken_streak[tgt.ws_address] = streak
                limit = COPILOT_CONFIG.auto_revert_after_failed_edits
                hints = _edit_error_hints(tgt.render_pass.source.path, new_text, errors)
                if limit > 0 and streak >= limit:
                    if tgt.ws_address in self._last_clean:
                        return self._force_restore(
                            tgt.ws_address,
                            tgt.document,
                            tgt.render_pass,
                            streak,
                            matches,
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
            self._broken_streak[tgt.document_id] = 0
            self._last_clean[tgt.document_id] = new_text
            facts = self._render_facts_for(
                tgt.document, motion=True, cache_key=tgt.document_id
            )
            loop_note = self._oscillation_note(tgt.document_id, tgt.source, new_text)
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
            "document that calls the function recompiles. Edit (or read) a document that uses it to "
            "confirm it is valid."
        )
        opens, closes = parser.brace_counts(new_text)
        if opens != closes:
            note += (
                f"\nwarning: the written file has {opens} '{{' vs {closes} '}}' — a brace "
                "went missing; consumer documents will fail to compile"
            )
        loop_note = self._oscillation_note(tgt.ws_address, tgt.source, new_text)
        if loop_note:
            note = f"{note}\n{loop_note}"
        return EditResult(matches=matches, errors=[], lib_note=note)

    def invalidate_lib_consumers(self, lib_path: Path) -> None:
        # A lib edit leaves consumer documents' source.text unchanged, so the next rebuild wouldn't recompile
        # them — invalidate every working-set document that pulled in this lib so they recompile with the new
        # source. Match on the resolved path (the index's source paths aren't resolved; they diverge under
        # a symlinked SHADERBOX_DATA_DIR).
        target = lib_path.resolve()
        for address in self._working_set_reader():
            document = self._get_ui_documents().get(address)
            if document is None:
                continue
            if any(
                s.path.resolve() == target
                for s in document.document.render_pass.compile_unit.sources
            ):
                document.document.render_pass.invalidate()

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
                    unresolved_reason=_batch_guard_reason("edit_shader"),
                    target_label=tgt.label,
                )
            result = self._copilot_persist_target(tgt, new_text, 1)
            if result.unresolved or result.restored_note:
                return result
            opens, closes = parser.brace_counts(new_text)
            if opens != closes:
                # Brace-broken text hides later definitions from the depth-0 scan — the
                # note would claim still-present functions removed; the compile error +
                # brace hint (document) / the lib brace warning (persist) cover it loudly.
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
