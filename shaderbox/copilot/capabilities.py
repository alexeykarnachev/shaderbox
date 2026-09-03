from dataclasses import dataclass, field
from typing import Protocol

from shaderbox.render_shape import RenderShape

# The only app surface the copilot package imports: a Protocol the backend
# (copilot/backend.py::CopilotBackend) satisfies structurally — the production session
# passes the backend itself. The package imports ONLY this leaf — never App, imgui, or
# moderngl — so the dependency stays one-way (app -> copilot), cycle-free. No method may
# be annotated with a type that transitively imports App/imgui/moderngl: only leaf types
# (primitives + the GL-free value objects defined HERE). Methods take positional-only
# params (`/`) so a Callable-field dataclass fake (tests/_caps.py) also conforms.


@dataclass(frozen=True)
class DocumentTreeEntry:
    # GL-FREE per-document row for the prompt project map. No uniform names — that's a GL
    # read; has_errors reads cached compile_unit.errors so the tree builds off-main.
    document_id: str
    name: str
    has_errors: bool
    is_current: bool
    # The document's passes, and which one is its output. A model that is never SHOWN that a
    # document has passes cannot construct a "<id>#<pass>" address it has never seen, so the map
    # is where the address kind becomes reachable. A single-pass document renders no pass list —
    # the ordinary case reads exactly as it did before.
    passes: tuple[str, ...] = ()
    output_pass: str = ""


@dataclass(frozen=True)
class LibCatalogEntry:
    # One lib function in the prompt catalogue: signature + doc only, no body (that's the
    # explicit read_lib pull).
    name: str
    signature: str
    doc: str
    lib_address: (
        str  # "lib:<relative-path>" — the edit_shader target for this function's file
    )


@dataclass(frozen=True)
class ExampleEntry:
    # One shipped document example in the prompt catalogue: name + description + a
    # `example:<short>` handle passed to create_document / read_shader / grep. GL-free.
    example_id: str  # the `example:<4-char>` handle (NOT the 36-char dir uuid)
    name: str
    description: str


@dataclass(frozen=True)
class CompileErrorInfo:
    path: str
    line: int  # 1-based
    message: str


@dataclass(frozen=True)
class ShaderView:
    # One document's full view for read_shader: identity + line-numbered listing + uniform
    # rows + compile errors. The read STAMPS the document's freshness so a subsequent edit
    # passes the guard.
    document_id: str
    name: str
    listing: str  # cat -n style
    uniforms: list[str]  # "name type = value" rows
    errors: list[CompileErrorInfo]


@dataclass(frozen=True)
class SetUniformResult:
    # ok=False carries `error` (name not found, sampler/block/engine-driven, or wrong
    # value shape). On ok=True, `type_label` is the uniform's type. The GL uniform
    # converges only after the next render.
    ok: bool
    error: str = ""
    type_label: str = ""
    # Probe-render facts for the frame with the new value applied (feature 033).
    render_facts: str = ""


@dataclass(frozen=True)
class DeleteDocumentResult:
    # ok=False carries `error` (no such document). On ok=True, document_id + trash_name feed the
    # recover card. trash_name is the dir-NAME under the project trash, NOT an absolute
    # path — the project dir is relocatable.
    ok: bool
    error: str = ""
    deleted_name: str = ""
    document_id: str = ""
    trash_name: str = ""


@dataclass(frozen=True)
class SwitchDocumentResult:
    # Makes a document current so the per-current tools (publish/render/edit-without-target)
    # act on it. ok=False carries `error`.
    ok: bool
    error: str = ""
    name: str = ""


@dataclass(frozen=True)
class DocumentOpResult:
    # Shared result for the small document-file ops (feature 052): rename (name set) and set_canvas_size
    # (width/height = the APPLIED, clamped size). ok=False carries `error` (no such document / bad size).
    ok: bool
    error: str = ""
    name: str = ""
    width: int = 0
    height: int = 0


@dataclass(frozen=True)
class MediaBindResult:
    # The result of a bind_media / unbind_media op (feature 052), carried back to the worker through
    # the FILE gate. NO absolute path: the picked path is consumed on the main thread inside the bind
    # and never reaches this object (corollary-1 by construction). `basename` is the source filename
    # (no directory) — safe to echo. `cancelled` = the user dismissed the picker.
    ok: bool = False
    error: str = ""
    cancelled: bool = False
    basename: str = ""
    width: int = 0
    height: int = 0
    is_video: bool = False


@dataclass(frozen=True)
class DocumentImportResult:
    # import_document outcome (feature 052 slice 4), carried back through the FILE gate. NO source path:
    # the picked file is read on the main thread inside import_picked_document. `basename` = the source
    # filename (no directory). `document_id` = the new document's short id; `errors` = its post-compile errors.
    ok: bool = False
    cancelled: bool = False
    error: str = ""
    document_id: str = ""
    errors: list[CompileErrorInfo] = field(default_factory=list)
    basename: str = ""


@dataclass(frozen=True)
class LibFileResult:
    # delete_lib_file outcome (feature 052). ok=False carries `error` (no such lib file).
    ok: bool
    error: str = ""


@dataclass(frozen=True)
class GrepHit:
    # One grep match. `origin` is an addressable handle for a read/edit tool: a document id
    # or a "lib:<path>" address.
    origin: str
    location: str  # human label, e.g. "document 'gradient'" or "lib:noise.glsl"
    line: int  # 1-based
    text: str  # the matched line, stripped


@dataclass(frozen=True)
class LibFunctionBody:
    # One lib function's full body for read_lib (only the found case; misses are
    # handled tool-side).
    name: str
    lib_address: str
    body: str


@dataclass(frozen=True)
class WorkingSetView:
    # One working-set member's live view for the per-turn scratchpad: in-memory source
    # listing + compile-coherent errors, rebuilt every iteration. A DOCUMENT carries uniform
    # rows + errors; a LIB file carries only the listing (no standalone compile). GL-FREE
    # value object — read_working_set marshals the GL/recompile work on the main thread.
    address: (
        str  # the agent-facing handle: a short document id, or a "lib:<path>" address
    )
    name: str  # document display name; the lib: address itself for a lib file
    listing: str  # cat -n style, from live source.text
    is_current: bool
    is_lib: bool
    uniforms: list[str]  # "name type = value" rows (document only; [] for a lib)
    errors: list[CompileErrorInfo]
    # The document's scripts/script.py live source (cat -n; "" = no script) + its compile/run error
    # (feature 043). One script drives EVERY pass (069 D3), so this listing is document-wide even on
    # a multi-pass document. Appended (defaulted) so existing constructors stay valid; a lib view
    # leaves both at default. Rendered as a "=== <document> SCRIPT ===" sub-section only when
    # script_listing is set.
    script_listing: str = ""
    script_errors: list[CompileErrorInfo] = field(default_factory=list)
    # One entry per pass when the document has MORE THAN ONE (D11): its own listing, uniforms and
    # errors, rendered as sub-sections the way SCRIPT already is. A single-pass document leaves
    # this empty and its one pass IS `listing`/`uniforms`/`errors` above, so the common case is
    # byte-identical to what the model saw before the graph existed.
    passes: list["PassView"] = field(default_factory=list)
    # The document's canvas resolution ("WxH"; "" for a lib view) — the render size the user gets
    # (feature 052). Rendered in the working-set document header so the model can see it.
    canvas: str = ""


@dataclass(frozen=True)
class PassView:
    """One pass of a multi-pass document, as the working set shows it."""

    name: str
    address: str  # the "<document>#<pass>" handle an edit tool takes
    listing: str
    uniforms: list[str]
    errors: list[CompileErrorInfo]
    is_output: bool


@dataclass(frozen=True)
class EditResult:
    # The outcome of an edit_shader apply. Match + replace + recompile happen against the
    # document's authoritative source on the main thread, so the handler never re-reads it.
    matches: int  # token-run matches of old_str (0 = not found, >1 = ambiguous)
    errors: list[CompileErrorInfo]  # 1-based; only meaningful when the edit applied
    # On a 0-match, the exact source bytes of the unique region that matches old_str
    # ignoring whitespace — the model copies this instead of re-guessing. "" when there
    # is no unique whitespace-only near-match.
    hint: str = ""
    # Unresolvable-target reject: bad target (unknown document id / invalid lib path), a read-only
    # example, the intra-batch rewrite guard, or a failed lib write. An argument/operation
    # error that DOES count toward the edit-retry cap. unresolved_reason is the message. matches==0.
    unresolved: bool = False
    unresolved_reason: str = ""
    # The honest "no standalone compile" note for a lib: edit target. Empty for a document
    # edit (which returns real errors).
    lib_note: str = ""
    # Feature 033 enriched results: structural compile hints (range bookkeeping,
    # initializer counts, brace balance) and the probe-render facts line on a
    # clean compile. Both engine-computed, both ride the tool result text.
    error_hints: tuple[str, ...] = ()
    render_facts: str = ""
    # Set when the engine force-restored the file to its last clean state after N
    # consecutive broken edits (033) — dominates the result message.
    restored_note: str = ""
    # True when the matched span would verbatim-overwrite an interior comment the
    # whitespace-invariant match can't see — refused so author content isn't silently
    # destroyed; the model is steered to re-quote including the comment. matches==0.
    comment_loss: bool = False
    # The RESOLVED target's display label ("document 'Wave' (f90f)" / "lib:a.glsl"), set once
    # the target resolved — so a failure names WHICH file was checked (an empty target
    # silently means the current document, the dogfooded giveup cause).
    target_label: str = ""
    # Whole-file rewrite fact (feature 039): the top-level functions/declarations the
    # rewrite REMOVED, appended to the result so a truncated rewrite is loud. "" otherwise.
    rewrite_note: str = ""


@dataclass(frozen=True)
class ScriptView:
    # read_script result (feature 043): the document's scripts/script.py source line-numbered + its
    # compile/run error. is_stub = the document had no script, so `listing` is the generated stub (not
    # persisted) the agent adapts; document_id resolves the document it belongs to.
    document_id: str
    name: str
    listing: str  # cat -n style
    errors: list[CompileErrorInfo]
    is_stub: bool


@dataclass(frozen=True)
class ScriptWriteResult:
    # write_script result (feature 043). ok=False + error for an unresolvable target. On ok=True the
    # compile/motion facts are the synchronous feedback: compile_error (a Python SyntaxError/etc., the
    # tool fixes it like a shader compile), driven (the uniforms it now drives, each as
    # `pass.uniform` — empty = the loud no-op fact), per_key_errors/orphan_keys (named + why, in the
    # same dotted form; a bare key no pass declares stays a bare name), motion_facts (the value-diff
    # verdict + the one ink/FLAT render line — the headless "is it animating" signal). The dotted
    # form is for the agent to READ; a script addresses a pass by a nested dict, never by that string.
    ok: bool
    error: str = ""
    compile_error: str = ""
    driven: list[str] = field(default_factory=list)
    per_key_errors: list[str] = field(default_factory=list)
    orphan_keys: list[str] = field(default_factory=list)
    motion_facts: str = ""
    # Set when the engine force-restored the script to its last clean-running state (mirrors
    # EditResult.restored_note). Its own field so a successful restore never reads as
    # applied-with-errors.
    restored_note: str = ""


@dataclass(frozen=True)
class RenderResult:
    # ok=False carries `error` (no such document, or the render failed). On ok=True, `path` is
    # the file under the project renders dir; size is the ACTUAL rendered size (snapped to
    # codec alignment); duration is video-only.
    ok: bool
    error: str = ""
    path: str = ""
    is_video: bool = False
    width: int = 0
    height: int = 0
    duration: float = 0.0


@dataclass(frozen=True)
class PublishResult:
    # ok=False carries the terminal error (or "cancelled" / "timed out"). On ok=True,
    # `url` is the pack/Studio link.
    ok: bool
    error: str = ""
    url: str = ""


@dataclass(frozen=True)
class TelegramConnectResult:
    # ok=True => linked (bot_username set). ok=False carries `error` (link failed / timed
    # out / no message). needs_start=True is the "token set, now open the bot + press
    # Start" guidance state.
    ok: bool
    error: str = ""
    bot_username: str = ""
    needs_start: bool = False


@dataclass(frozen=True)
class TelegramOpResult:
    # ok=False carries `error`.
    ok: bool
    error: str = ""
    set_name: str = ""


@dataclass(frozen=True)
class TelegramPackInfo:
    # One saved pack for list_telegram_packs.
    title: str
    set_name: str
    is_default: bool


class CopilotCapabilities(Protocol):
    # ---- GL-FREE context reads — safe on the worker thread (no bridge). document_tree excludes
    # uniforms ON PURPOSE (uniform names need a GL read; see DocumentTreeEntry).
    def document_tree(self) -> list[DocumentTreeEntry]: ...
    def lib_catalog(self) -> list[LibCatalogEntry]: ...
    # Shipped document examples for create_document(example=...). GL-free.
    def example_catalog(self) -> list[ExampleEntry]: ...

    # ---- cross-project reads ----
    # read_shaders marshals (force-compile + uniform read are GL) and STAMPS freshness per
    # document. grep + read_lib are GL-FREE (string reads over the parsed index / in-memory).
    def read_shaders(self, document_ids: list[str], /) -> list[ShaderView]: ...
    def grep(self, query: str, /) -> list[GrepHit]: ...
    def read_lib(self, names: list[str], /) -> list[LibFunctionBody]: ...
    # The per-turn working set: every shader/lib touched this turn, rebuilt from live source
    # each iteration. read_working_set returns a compile-coherent view (bridge-marshalled —
    # uniform read + the program-is-None recompile are GL) PLUS the addresses the size cap
    # evicted this turn (rendered as a loud line — a silently vanished source is worse than
    # none). reset_working_set clears both, once per turn at enqueue. batch_begin clears the
    # per-batch rewrite guard's mutated-target set; run_turn calls it ONCE before each batch.
    def read_working_set(self) -> tuple[list[WorkingSetView], list[str]]: ...
    def reset_working_set(self) -> None: ...
    def batch_begin(self) -> None: ...

    # ---- mutations the worker REQUESTS but the main thread APPLIES ----
    # Backend methods wrapping bridge.run_on_main(...) (the worker blocks for the result);
    # the marshalling is hidden inside the method, so the tool layer just calls them.
    #
    # Match old_str against the TARGET's CURRENT source, replace, recompile (document) / write
    # (lib), persist, refresh the editor — all on the main thread. `target` "" = current
    # document, a document-id, or a "lib:<path>" address.
    def apply_shader_edit(
        self, old_str: str, new_str: str, replace_all: bool, target: str, /
    ) -> EditResult: ...

    # Replace the target's ENTIRE source with new_text, recompile/write + persist +
    # refresh; for a non-existent lib: target this CREATES the file. An applied rewrite
    # carries the removed top-level names fact in rewrite_note.
    def apply_full_rewrite(self, new_text: str, target: str, /) -> EditResult: ...

    # Set a uniform VALUE. document "" = current. Rejects sampler/block/engine-driven with an
    # explicit error.
    def set_uniform(
        self, name: str, value: object, document: str, /
    ) -> SetUniformResult: ...

    # ---- scripting (feature 043): the document script authoring surface ----
    # read_script returns the document's scripts/script.py source (a fresh document returns the generated
    # stub, unpersisted). write_script create-or-overwrites the whole script, recompiles, dry-runs it,
    # and returns the compile + motion facts. Both marshal main-thread (the dry-tick reads the GL
    # program for active uniforms). document "" = current.
    def read_script(self, document: str, /) -> ScriptView: ...
    def write_script(self, new_text: str, document: str, /) -> ScriptWriteResult: ...
    # edit_script: a substring edit (plain-text match), the script mirror of edit_shader; returns the
    # same ScriptWriteResult as write_script so an edit and a write give identical feedback.
    def apply_script_edit(
        self, old_str: str, new_str: str, replace_all: bool, document: str, /
    ) -> ScriptWriteResult: ...

    # Create a document, then COMPILE it and return its errors. `example` = an example id from
    # example_catalog ("" = the default starter); `source` overrides the example body when
    # non-empty. Returns (new document-id, post-compile errors).
    def create_document(
        self, name: str, source: str, example: str, switch_to: bool, /
    ) -> tuple[str, list[CompileErrorInfo], str]: ...

    # Delete a document (move its dir to the project trash, recoverable). Destructive => always
    # gated; the method marshals the GL teardown via the bridge.
    def delete_document(self, document: str, /) -> DeleteDocumentResult: ...

    # Make a document CURRENT so the per-current tools (publish/render/edit-without-target) act
    # on it. Stamps freshness so a follow-up edit lands.
    def switch_document(self, document: str, /) -> SwitchDocumentResult: ...

    # ---- document file ops (feature 052 slice 3) — checkpoint-revertable document.json mutations ----
    # rename_document sets the display name; set_canvas_size sets the document's render resolution (clamped,
    # returns the applied size). document "" = current. duplicate_document forks a document (dir copy incl.
    # media/script) into a fresh id, returning the create_document shape (short id, post-compile errors,
    # facts).
    def rename_document(self, document: str, new_name: str, /) -> DocumentOpResult: ...
    def set_canvas_size(
        self, document: str, width: int, height: int, /
    ) -> DocumentOpResult: ...
    def duplicate_document(
        self, document: str, new_name: str, switch_to: bool, /
    ) -> tuple[str, list[CompileErrorInfo], str]: ...

    # Delete a shader-library file (a "lib:<path>" address) to the shader-lib trash (recoverable);
    # consumers recompile with a missing-SB_* error next step. Destructive => always gated.
    def delete_lib_file(self, path: str, /) -> LibFileResult: ...

    # ---- media / textures (feature 052 slice 2) ----
    # bind_media validates the sampler, then BLOCKS on a FILE gate while the UI opens the OS file
    # picker; the main thread does the load+bind and answers with a path-free MediaBindResult (the
    # abs path never crosses back to the worker). unbind_media resets a sampler to the default image
    # (a normal main-thread op, no picker). document "" = current.
    def bind_media(self, document: str, uniform: str, /) -> MediaBindResult: ...
    def unbind_media(self, document: str, uniform: str, /) -> MediaBindResult: ...

    # Import a .glsl off the user's disk (feature 052 slice 4): opens the file picker, reads the file
    # on the main thread, and creates a document from it. Reuses the FILE gate; the source path never
    # reaches the worker.
    def import_document(self, switch_to: bool, /) -> DocumentImportResult: ...

    # ---- render / publish (all gated) ----
    # Render a document's current frame to a PNG / `seconds` of animation to a WebM under the
    # project renders dir. GL => the method marshals via the bridge with the longer
    # render_op_timeout_s. `shape` is a named RenderShape tier (NATIVE = the document's canvas size).
    def render_image(self, document: str, shape: RenderShape, /) -> RenderResult: ...
    def render_video(
        self, document: str, seconds: float, fps: int, shape: RenderShape, /
    ) -> RenderResult: ...

    # The aimable read-side probe (feature 050): a one-line facts string off a tiny offscreen
    # render at a chosen `t` (default 0.0). UN-gated + non-mutating, unlike render_image. The
    # returned text is ready to read (the facts line, or an honest error/empty note).
    def probe_render(self, document: str, t: float, /) -> str: ...

    # Render with the exporter's own preset, then enqueue the upload + AWAIT its terminal
    # progress (the method does the bridge-marshalled poll).
    def publish_telegram(self, emoji: str, /) -> PublishResult: ...
    def publish_youtube(
        self, title: str, description: str, shape: RenderShape, /
    ) -> PublishResult: ...

    # GL-free precheck reads backing the pre-gate guided handoff: is there a current document, is
    # the integration connected, and (Telegram) is a pack selected.
    def has_current_document(self) -> bool: ...
    def telegram_connected(self) -> bool: ...
    def youtube_connected(self) -> bool: ...
    def telegram_has_default_pack(self) -> bool: ...

    # ---- Telegram connect + pack CRUD ----
    # set_telegram_token sets the token then auto-kicks the link + awaits AUTHED.
    # telegram_connect re-runs the link (after the user starts the bot). The pack ops are
    # gated state writes; delete awaits its terminal progress.
    def set_telegram_token(self, secret: str, /) -> TelegramConnectResult: ...
    def telegram_connect(self) -> TelegramConnectResult: ...
    def list_telegram_packs(self) -> list[TelegramPackInfo]: ...
    def select_telegram_pack(self, set_name: str, /) -> TelegramOpResult: ...
    def create_telegram_pack(self, title: str, /) -> TelegramOpResult: ...
    def delete_telegram_pack(self, set_name: str, /) -> TelegramOpResult: ...
