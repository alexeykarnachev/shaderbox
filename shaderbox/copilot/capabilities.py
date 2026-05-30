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
    line: int
    message: str


@dataclass(frozen=True)
class CopilotCapabilities:
    # ---- read-only / GL-free (safe to call on the worker thread) ----
    list_nodes: Callable[[], list[NodeSummary]]
    get_node_summary: Callable[[str], NodeSummary | None]
    get_shader_source: Callable[[str], str | None]
    get_compile_errors: Callable[[str], list[CompileErrorInfo]]
    current_node_id: Callable[[], str]

    # ---- mutations the worker REQUESTS but the main thread APPLIES ----
    # Implemented App-side as bridge.run_on_main(...) closures (the worker blocks for
    # the result). The tool layer calls these like any other callable — the
    # marshalling is hidden inside the App-supplied closure (Seam C stays invisible here).
    edit_shader_source: Callable[[str, str], bool]
