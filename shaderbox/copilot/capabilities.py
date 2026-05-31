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
class CompileErrorInfo:
    path: str
    line: int  # 1-based (matches the agent's cat -n orientation)
    message: str


@dataclass(frozen=True)
class CurrentShaderView:
    # The line-numbered listing + uniforms + errors get_current_shader returns. `text`
    # is the RAW source (no line-number prefixes) — the substring edit_shader matches on.
    text: str
    listing: str  # cat -n style; line-number prefixes are display-only (§16.2)
    uniforms: list[str]  # "name type = value" rows; [] when the shader doesn't compile
    errors: list[CompileErrorInfo]


@dataclass(frozen=True)
class EditResult:
    # The outcome of an edit_shader apply. The match + replace + recompile all happen
    # against the node's authoritative source on the main thread (§16.3), so the handler
    # never re-reads the source — `matches` tells it which §16.4 string to return.
    matches: int  # occurrences of old_str found (0 = not found, >1 = ambiguous)
    errors: list[CompileErrorInfo]  # 1-based; only meaningful when the edit applied


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

    # ---- mutations the worker REQUESTS but the main thread APPLIES ----
    # Implemented App-side as bridge.run_on_main(...) closures (the worker blocks for
    # the result). The tool layer calls these like any other callable — the
    # marshalling is hidden inside the App-supplied closure (Seam C stays invisible here).
    #
    # Slice 1 (edit/compile-feedback) — all current-node-only (no node_id arg):
    get_current_shader_view: Callable[[], CurrentShaderView | None]
    # Match old_str against the node's CURRENT source, replace, recompile, persist,
    # refresh the editor — all on the main thread (§16.3). Returns the match count + the
    # post-compile 1-based errors. (old_str, new_str, replace_all).
    apply_shader_edit: Callable[[str, str, bool], EditResult]
    # Force a compile if stale, return the current 1-based errors.
    get_compile_errors_current: Callable[[], list[CompileErrorInfo]]
