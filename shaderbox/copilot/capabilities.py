from collections.abc import Callable
from dataclasses import dataclass

# The only app surface the copilot package imports: a frozen dataclass of bound
# callables App builds in __init__. The package imports ONLY this leaf — never App,
# imgui, or moderngl — so the dependency stays one-way (app -> copilot), cycle-free.
# No field may be annotated with a type that transitively imports App/imgui/moderngl:
# only leaf types (primitives + the GL-free value objects defined HERE).


@dataclass(frozen=True)
class NodeTreeEntry:
    # GL-FREE per-node row for the prompt project map. No uniform names — that's a GL
    # read; has_errors reads cached compile_unit.errors so the tree builds off-main.
    node_id: str
    name: str
    has_errors: bool
    is_current: bool


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
class TemplateEntry:
    # One shipped node template in the prompt catalogue: name + description + a
    # `template:<short>` handle passed to create_node / read_shader / grep. GL-free.
    template_id: str  # the `template:<4-char>` handle (NOT the 36-char dir uuid)
    name: str
    description: str


@dataclass(frozen=True)
class CompileErrorInfo:
    path: str
    line: int  # 1-based
    message: str


@dataclass(frozen=True)
class ShaderView:
    # One node's full view for read_shader: identity + line-numbered listing + uniform
    # rows + compile errors. The read STAMPS the node's freshness so a subsequent edit
    # passes the guard.
    node_id: str
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


@dataclass(frozen=True)
class DeleteNodeResult:
    # ok=False carries `error` (no such node). On ok=True, node_id + trash_name feed the
    # recover card. trash_name is the dir-NAME under the project trash, NOT an absolute
    # path — the project dir is relocatable.
    ok: bool
    error: str = ""
    deleted_name: str = ""
    node_id: str = ""
    trash_name: str = ""


@dataclass(frozen=True)
class SwitchNodeResult:
    # Makes a node current so the per-current tools (publish/render/edit-without-target)
    # act on it. ok=False carries `error`.
    ok: bool
    error: str = ""
    name: str = ""


@dataclass(frozen=True)
class GrepHit:
    # One grep match. `origin` is an addressable handle for a read/edit tool: a node id
    # or a "lib:<path>" address.
    origin: str
    location: str  # human label, e.g. "node 'gradient'" or "lib:noise.glsl"
    line: int  # 1-based
    text: str  # the matched line, stripped


@dataclass(frozen=True)
class LibFunctionBody:
    # One lib function's full body for read_lib (only the found case; misses are
    # handled tool-side).
    name: str
    signature: str
    lib_address: str
    body: str


@dataclass(frozen=True)
class WorkingSetView:
    # One working-set member's live view for the per-turn scratchpad: in-memory source
    # listing + compile-coherent errors, rebuilt every iteration. A NODE carries uniform
    # rows + errors; a LIB file carries only the listing (no standalone compile). GL-FREE
    # value object — read_working_set marshals the GL/recompile work on the main thread.
    address: str  # the agent-facing handle: a short node id, or a "lib:<path>" address
    name: str  # node display name; the lib: address itself for a lib file
    listing: str  # cat -n style, from live source.text
    is_current: bool
    is_lib: bool
    uniforms: list[str]  # "name type = value" rows (node only; [] for a lib)
    errors: list[CompileErrorInfo]


@dataclass(frozen=True)
class EditResult:
    # The outcome of an edit_shader apply. Match + replace + recompile happen against the
    # node's authoritative source on the main thread, so the handler never re-reads it.
    matches: int  # token-run matches of old_str (0 = not found, >1 = ambiguous)
    errors: list[CompileErrorInfo]  # 1-based; only meaningful when the edit applied
    # On a 0-match, the exact source bytes of the unique region that matches old_str
    # ignoring whitespace — the model copies this instead of re-guessing. "" when there
    # is no unique whitespace-only near-match.
    hint: str = ""
    # Unresolvable-target reject: bad target (unknown node id / invalid lib path), a read-only
    # template, the intra-batch line-edit guard, or a failed lib write. An argument/operation
    # error that DOES count toward the edit-retry cap. unresolved_reason is the message. matches==0.
    unresolved: bool = False
    unresolved_reason: str = ""
    # The honest "no standalone compile" note for a lib: edit target. Empty for a node
    # edit (which returns real errors).
    lib_note: str = ""
    # True when the matched span would verbatim-overwrite an interior comment the
    # whitespace-invariant match can't see — refused so author content isn't silently
    # destroyed; the model is steered to a line-addressed edit. matches==0.
    comment_loss: bool = False


@dataclass(frozen=True)
class RenderResult:
    # ok=False carries `error` (no such node, or the render failed). On ok=True, `path` is
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
    # `url` is the pack/Studio link; `kind` is the target ("telegram"/"youtube").
    ok: bool
    error: str = ""
    url: str = ""
    kind: str = ""


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


@dataclass(frozen=True)
class CopilotCapabilities:
    # ---- GL-FREE context reads — safe on the worker thread (no bridge). node_tree excludes
    # uniforms ON PURPOSE (uniform names need a GL read; see NodeTreeEntry).
    node_tree: Callable[[], list[NodeTreeEntry]]
    lib_catalog: Callable[[], list[LibCatalogEntry]]
    # Shipped node templates for create_node(template=...). GL-free.
    template_catalog: Callable[[], list[TemplateEntry]]

    # ---- cross-project reads ----
    # read_shaders marshals (force-compile + uniform read are GL) and STAMPS freshness per
    # node. grep + read_lib are GL-FREE (string reads over the parsed index / in-memory).
    read_shaders: Callable[[list[str]], list[ShaderView]]
    grep: Callable[[str], list[GrepHit]]
    read_lib: Callable[[list[str]], list[LibFunctionBody]]
    # The per-turn working set: every shader/lib touched this turn, rebuilt from live source
    # each iteration. read_working_set returns a compile-coherent view (bridge-marshalled —
    # uniform read + the program-is-None recompile are GL). batch_begin clears the per-batch
    # line-edit guard's mutated-target set; run_turn calls it ONCE before each tool-call batch.
    read_working_set: Callable[[], list[WorkingSetView]]
    batch_begin: Callable[[], None]

    # ---- mutations the worker REQUESTS but the main thread APPLIES ----
    # App-side bridge.run_on_main(...) closures (the worker blocks for the result); the
    # marshalling is hidden inside the closure, so the tool layer just calls them.
    #
    # Match old_str against the TARGET's CURRENT source, replace, recompile (node) / write
    # (lib), persist, refresh the editor — all on the main thread. `target` "" = current
    # node, a node-id, or a "lib:<path>" address. (old_str, new_str, replace_all, target).
    apply_shader_edit: Callable[[str, str, bool, str], EditResult]
    # Replace the 1-based inclusive line range [start, end] with new_text, recompile/write +
    # persist + refresh. An empty selection (end == start - 1) is a pure insert at `start`;
    # for a non-existent lib: target this CREATES the file. (start, end, new_text, target).
    apply_line_edit: Callable[[int, int, str, str], EditResult]
    # Set a uniform VALUE: (name, value, node). node "" = current. Rejects
    # sampler/block/engine-driven with an explicit error.
    set_uniform: Callable[[str, object, str], "SetUniformResult"]
    # Create a node, then COMPILE it and return its errors. `template` = a template id from
    # template_catalog ("" = the default starter); `source` overrides the template body when
    # non-empty. (name, source, template, switch_to) -> (new node-id, post-compile errors).
    create_node: Callable[[str, str, str, bool], tuple[str, list[CompileErrorInfo]]]
    # Delete a node (move its dir to the project trash, recoverable). Destructive => always
    # gated; the closure marshals the GL teardown via the bridge.
    # (node short-id) -> DeleteNodeResult.
    delete_node: Callable[[str], "DeleteNodeResult"]
    # Make a node CURRENT so the per-current tools (publish/render/edit-without-target) act
    # on it. Stamps freshness so a follow-up edit lands. (node short-id) -> SwitchNodeResult.
    switch_node: Callable[[str], "SwitchNodeResult"]

    # ---- render / publish (all gated) ----
    # Render a node's current frame to a PNG / `seconds` of animation to a WebM under the
    # project renders dir. GL => the closure marshals via the bridge with the longer
    # render_op_timeout_s. A 0 width/height means "use the node's canvas size".
    render_image: Callable[[str, int, int], "RenderResult"]
    render_video: Callable[[str, float, int, int, int], "RenderResult"]
    # Render with the exporter's own preset, then enqueue the upload + AWAIT its terminal
    # progress (the closure does the bridge-marshalled poll).
    publish_telegram: Callable[[str, str], "PublishResult"]
    publish_youtube: Callable[[str, str, str, bool], "PublishResult"]
    # GL-free precheck reads backing the pre-gate guided handoff: is there a current node, is
    # the integration connected, and (Telegram) is a pack selected.
    has_current_node: Callable[[], bool]
    telegram_connected: Callable[[], bool]
    youtube_connected: Callable[[], bool]
    telegram_has_default_pack: Callable[[], bool]

    # ---- Telegram connect + pack CRUD ----
    # set_telegram_token sets the token then auto-kicks the link + awaits AUTHED.
    # telegram_connect re-runs the link (after the user starts the bot). The pack ops are
    # gated state writes; delete awaits its terminal progress.
    set_telegram_token: Callable[[str], "TelegramConnectResult"]
    telegram_connect: Callable[[], "TelegramConnectResult"]
    telegram_token_set: Callable[[], bool]
    list_telegram_packs: Callable[[], list["TelegramPackInfo"]]
    select_telegram_pack: Callable[[str], "TelegramOpResult"]
    create_telegram_pack: Callable[[str], "TelegramOpResult"]
    delete_telegram_pack: Callable[[str], "TelegramOpResult"]
