from collections.abc import Callable
from dataclasses import dataclass

# Seam A: the only app surface the copilot package imports. A frozen dataclass of
# bound callables that App builds in __init__ (like _build_command_callbacks). The
# package imports ONLY this leaf — never App, imgui, or moderngl — so the dependency
# is app -> copilot, one-way, cycle-free.
#
# RULE (skeleton 10 §2 rule 5): no field may be annotated with a type that
# transitively imports App / imgui / moderngl. Only leaf types: primitives, and the
# GL-free value objects defined HERE. A field typed Callable[[], moderngl.Texture]
# would silently reintroduce the banned import.


@dataclass(frozen=True)
class NodeTreeEntry:
    # The lean, GL-FREE per-node row for the always-in-prompt project map (feature 020·16,
    # Decision 9). NO uniform names — get_active_uniforms() is a GL read and build_context
    # runs on the worker thread; has_errors reads the cached compile_unit.errors (GL-free)
    # so the tree stays buildable off-main AND cache-stable (it carries no per-frame value).
    node_id: str
    name: str
    has_errors: bool
    is_current: bool


@dataclass(frozen=True)
class LibCatalogEntry:
    # One lib function in the always-in-prompt catalogue (feature 020·16): the signature +
    # doc the agent needs to know a helper EXISTS and how to call it. NO body — that is the
    # explicit read_lib pull. The lib: address is how the agent targets the file for an edit.
    name: str
    signature: str
    doc: str
    lib_address: (
        str  # "lib:<relative-path>" — the edit_shader target for this function's file
    )


@dataclass(frozen=True)
class CompileErrorInfo:
    path: str
    line: int  # 1-based (matches the agent's cat -n orientation)
    message: str


@dataclass(frozen=True)
class ShaderView:
    # One node's full view for read_shader (feature 020·16): identity + the line-numbered
    # listing + uniform rows (type + current value) + compile errors. read_shader returns a
    # LIST of these (one per requested node). The read STAMPS the node's freshness so a
    # subsequent edit on it passes the guard.
    node_id: str
    name: str
    listing: str  # cat -n style
    uniforms: list[str]  # "name type = value" rows
    errors: list[CompileErrorInfo]


@dataclass(frozen=True)
class SetUniformResult:
    # The outcome of a set_uniform (feature 020·16 Decision 6). ok=False carries `error` (the
    # name wasn't found, was a sampler/block/engine-driven, or the value's shape was wrong).
    # On ok=True, `type_label` is the uniform's type for the reply; the value the tool echoes
    # is the one the agent sent (the GL uniform converges only after the next render).
    ok: bool
    error: str = ""
    type_label: str = ""


@dataclass(frozen=True)
class DeleteNodeResult:
    # The outcome of a delete_node (feature 020·17). ok=False carries `error` (no such node).
    # On ok=True, node_id + trash_name feed the recover card (trash_name is the dir-NAME under
    # the project trash, NOT an absolute path — the project dir is relocatable).
    ok: bool
    error: str = ""
    deleted_name: str = ""
    node_id: str = ""
    trash_name: str = ""


@dataclass(frozen=True)
class SwitchNodeResult:
    # The outcome of a switch_node (the copilot makes a node the current one so the per-current
    # tools — publish/render/edit-without-target — act on it). ok=False carries `error`.
    ok: bool
    error: str = ""
    name: str = ""


@dataclass(frozen=True)
class GrepHit:
    # One origin-labeled match for grep (feature 020·16). `origin` is the addressable handle
    # the agent can hand to a read/edit tool: a node id, or a "lib:<path>" address.
    origin: str
    location: str  # human label, e.g. "node 'gradient'" or "lib:noise.glsl"
    line: int  # 1-based
    text: str  # the matched line, stripped


@dataclass(frozen=True)
class LibFunctionBody:
    # One lib function's full body for read_lib (feature 020·16). None-result (missing name)
    # is handled tool-side; this is only the found case.
    name: str
    signature: str
    lib_address: str
    body: str


@dataclass(frozen=True)
class EditResult:
    # The outcome of an edit_shader apply. The match + replace + recompile all happen
    # against the node's authoritative source on the main thread (§16.3), so the handler
    # never re-reads the source — `matches` tells it which §16.4 string to return.
    matches: int  # token-run matches of old_str (0 = not found, >1 = ambiguous)
    errors: list[CompileErrorInfo]  # 1-based; only meaningful when the edit applied
    # On a 0-match, the exact source bytes of the unique region that matches old_str
    # ignoring whitespace — the model copies this instead of re-guessing. "" when there
    # is no unique whitespace-only near-match (feature 020 · 12).
    hint: str = ""
    # Apply-feedback (feature 020 · 14): the post-edit "what changed" excerpt (line-numbered
    # context around the changed region) + its 1-based line range in the NEW source. Set on a
    # single-region apply; both empty/None on a non-apply OR a multi-span replace_all.
    changed_excerpt: str = ""
    changed_range: tuple[int, int] | None = None
    # Freshness reject (feature 020 · 15): True when the edit was refused because the source
    # moved since the agent last read it this turn (or it never read / switched nodes).
    # stale_reason is the App-built message naming the specific cause. matches==0 on a reject.
    stale: bool = False
    stale_reason: str = ""
    # Lib edit (feature 020 · 16 Decision 4): the honest "no standalone compile" note returned
    # when the edit target was a lib: file. Empty for a node edit (which returns real errors).
    lib_note: str = ""


@dataclass(frozen=True)
class RenderResult:
    # The outcome of a render_image / render_video (feature 020·18). ok=False carries `error`
    # (no such node, or the render failed). On ok=True, `path` is the file under the project
    # renders dir + the ACTUAL rendered size (snapped to the codec alignment) + duration (video).
    ok: bool
    error: str = ""
    path: str = ""
    is_video: bool = False
    width: int = 0
    height: int = 0
    duration: float = 0.0


@dataclass(frozen=True)
class PublishResult:
    # The outcome of a publish_telegram / publish_youtube (feature 020·18). ok=False carries the
    # terminal error message (or "cancelled" / "timed out"). On ok=True, `url` is the pack/Studio
    # link the agent relays. `kind` is the target ("telegram"/"youtube") for the reply phrasing.
    ok: bool
    error: str = ""
    url: str = ""
    kind: str = ""


@dataclass(frozen=True)
class TelegramConnectResult:
    # The outcome of set_telegram_token / telegram_connect (feature 020·19). ok=True => linked;
    # bot_username for the reply. ok=False carries `error` (link failed / timed out / no message).
    # needs_start=True is the special "token set, now open the bot + press Start" guidance state.
    ok: bool
    error: str = ""
    bot_username: str = ""
    needs_start: bool = False


@dataclass(frozen=True)
class TelegramOpResult:
    # The outcome of a pack-CRUD op (feature 020·19). ok=False carries `error`.
    ok: bool
    error: str = ""
    set_name: str = ""


@dataclass(frozen=True)
class TelegramPackInfo:
    # One saved pack for list_telegram_packs (feature 020·19).
    title: str
    set_name: str
    is_default: bool


@dataclass(frozen=True)
class CopilotCapabilities:
    # ---- GL-FREE context reads (feature 020·16) — safe to call on the worker thread when
    # building the per-turn prompt context (no bridge). node_tree excludes uniforms ON PURPOSE
    # (uniform names need a GL read; see NodeTreeEntry). lib_catalog reads the parsed index.
    node_tree: Callable[[], list[NodeTreeEntry]]
    lib_catalog: Callable[[], list[LibCatalogEntry]]

    # ---- cross-project reads (feature 020·16) ----
    # read_shader marshals (force-compile + uniform read are GL) and STAMPS freshness per node;
    # it takes the resolved node-id LIST ("" / [] -> current is resolved tool-side). grep +
    # read_lib are GL-FREE (string reads over the parsed index / in-memory sources).
    read_shaders: Callable[[list[str]], list[ShaderView]]
    grep: Callable[[str], list[GrepHit]]
    read_lib: Callable[[list[str]], list[LibFunctionBody]]

    # ---- mutations the worker REQUESTS but the main thread APPLIES ----
    # Implemented App-side as bridge.run_on_main(...) closures (the worker blocks for
    # the result). The tool layer calls these like any other callable — the
    # marshalling is hidden inside the App-supplied closure (Seam C stays invisible here).
    #
    # Match old_str against the TARGET's CURRENT source, replace, recompile (node) / write
    # (lib), persist, refresh the editor — all on the main thread (§16.3). `target` is "" =
    # current node, a node-id, or a "lib:<path>" address (020·16). Returns the match count +
    # the post-compile errors (node) or the honest "no standalone compile" result (lib).
    # (old_str, new_str, replace_all, target).
    apply_shader_edit: Callable[[str, str, bool, str], EditResult]
    # Replace the 1-based inclusive line range [start, end] of the target with new_text,
    # recompile/write + persist + refresh (feature 020 · 14 / 16). An empty selection
    # (end == start - 1) is a pure insert at position `start`; for a non-existent lib: target
    # this CREATES the file (Decision 5). (start, end, new_text, target).
    apply_line_edit: Callable[[int, int, str, str], EditResult]
    # Set a uniform VALUE on a node (020·16 Decision 6): (name, value, node). node "" = current.
    # Validates up front; rejects sampler/block/engine-driven with an explicit error.
    set_uniform: Callable[[str, object, str], "SetUniformResult"]
    # Create a node from source ("" = compiling starter), then COMPILE it and return its errors
    # — the same compile-feedback every edit tool gives (the "every mutation returns its compile
    # result" invariant). switch_to controls the tab. (name, source, switch_to) -> (new node-id,
    # post-compile errors) (020·16 Decision 8).
    create_node: Callable[[str, str, bool], tuple[str, list[CompileErrorInfo]]]
    # Delete a node (move its dir to the project trash, recoverable). Destructive => the loop
    # always gates it (GatePolicy.ALWAYS); the closure marshals the GL teardown via the bridge.
    # (node short-id) -> DeleteNodeResult (feature 020·17).
    delete_node: Callable[[str], "DeleteNodeResult"]
    # Make a node the CURRENT one so the per-current tools (publish/render/edit-without-target)
    # act on it. Non-destructive (the user sees their view switch); stamps freshness so a
    # follow-up edit lands. (node short-id) -> SwitchNodeResult.
    switch_node: Callable[[str], "SwitchNodeResult"]

    # ---- render / publish (feature 020·18; all GatePolicy.ALWAYS) ----
    # Render a node's current frame to a PNG (render_image) / `seconds` of animation to a WebM
    # (render_video) under the project renders dir. GL => the closure marshals via the bridge with
    # the longer render_op_timeout_s. (node, width, height) / (node, seconds, fps, width, height);
    # a 0 width/height means "use the node's canvas size".
    render_image: Callable[[str, int, int], "RenderResult"]
    render_video: Callable[[str, float, int, int, int], "RenderResult"]
    # Render the node with the exporter's own preset, then enqueue the upload + AWAIT its terminal
    # progress (the closure does the bridge-marshalled poll). (node, emoji) / (node, title,
    # description, is_short) -> PublishResult (url or error).
    publish_telegram: Callable[[str, str], "PublishResult"]
    publish_youtube: Callable[[str, str, str, bool], "PublishResult"]
    # GL-free precheck reads backing the pre-gate guided handoff (the tool's `precheck`): is there
    # a current node to render, is the integration connected, and (Telegram) is a pack selected.
    has_current_node: Callable[[], bool]
    telegram_connected: Callable[[], bool]
    youtube_connected: Callable[[], bool]
    telegram_has_default_pack: Callable[[], bool]

    # ---- Telegram connect + pack CRUD (feature 020·19) ----
    # set_telegram_token sets the token (the gate's secret, out-of-band) then auto-kicks the link
    # + awaits AUTHED. telegram_connect re-runs the link (after the user starts the bot). The pack
    # ops are render-side state writes (gated CONFIRM); delete awaits its terminal progress.
    set_telegram_token: Callable[[str], "TelegramConnectResult"]
    telegram_connect: Callable[[], "TelegramConnectResult"]
    telegram_token_set: Callable[[], bool]
    list_telegram_packs: Callable[[], list["TelegramPackInfo"]]
    select_telegram_pack: Callable[[str], "TelegramOpResult"]
    create_telegram_pack: Callable[[str], "TelegramOpResult"]
    delete_telegram_pack: Callable[[str], "TelegramOpResult"]
