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
#
# This is the SCAFFOLD seam: the value objects + a minimal field set that proves the
# shape + testability. The full verb catalog is the later capability brainstorm
# (ai_docs/features/020_copilot_agent — §0 #8), added as more fields here.


@dataclass(frozen=True)
class NodeSummary:
    node_id: str
    name: str
    uniform_names: list[str]
    has_errors: bool


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
    # On ok=True, `applied` echoes the CPU-side value written (NOT the GL uniform — that
    # converges only after the next render); `type_label` is the uniform's type for the reply.
    ok: bool
    error: str = ""
    type_label: str = ""


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
class CopilotCapabilities:
    # ---- scaffold reads (NOT consumed by any slice-1 tool) ----
    # NOTE: list_nodes / get_node_summary read get_active_uniforms() (a GL call), so a
    # FUTURE tool that uses them must marshal via the bridge like the slice-1 caps below —
    # they are NOT GL-free despite the value types they return being plain data.
    list_nodes: Callable[[], list[NodeSummary]]
    get_node_summary: Callable[[str], NodeSummary | None]
    get_shader_source: Callable[[str], str | None]
    get_compile_errors: Callable[[str], list[CompileErrorInfo]]
    current_node_id: Callable[[], str]

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
    # Current-node-only today; target-addressing lands with the write tools (020·16 Phase 3).
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
    # Create a node from source ("" = compiling starter). switch_to controls the tab.
    # (name, source, switch_to) -> new node-id (020·16 Decision 8).
    create_node: Callable[[str, str, bool], str]
