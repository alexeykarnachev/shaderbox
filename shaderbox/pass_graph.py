"""The pass graph: which passes a document has, what fills each one's inputs, and in what
order they draw (065).

A document holds several passes forming a DAG. Each pass is its own `.glsl` file with its own
`main()` and its own render target; one pass is the document's output. A pass declares an input
as an ordinary `uniform sampler2D`, and the graph says which pass fills it -- binding is BY NAME,
never by position.

Everything here is engine-level machinery persisted in `graph.json`, never parsed out of comments,
and everything here is GL-free pure data: the model, the topological order, the cycle check and the
feedback marking are unit-testable with no context and importable from anywhere without a cycle.

Two rules the rest of the engine leans on:

- **A pass reading itself is FEEDBACK, not a cycle.** It consumes the previous frame, so it
  contributes no ordering constraint. Excluding the self-edge from the cycle check while keeping it
  in the model is the trap that sinks a naive implementation.
- **A pass appears at most ONCE in the order.** A diamond's shared ancestor draws once, not once per
  consuming path. A duplicated pass renders the CORRECT picture, just N times, so it reads as slow
  rather than wrong and no pixel test can see it: `assert_plan_invariants` is the observable, and
  `evaluation_order` runs it on the path that actually draws.
"""

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DTYPES: tuple[str, ...] = ("f1", "f2", "f4")

# f2, not f1: 063 measured f1 saturating at 255 on the FIRST accumulate pass where f2 reached
# exactly 7.0, so the safe value is the default and f1 is the opt-in. clamp inverts moderngl's
# repeat_x/y=True, which is wrong for a feedback border.
DEFAULT_DTYPE: Literal["f1", "f2", "f4"] = "f2"
DEFAULT_FILTER_LINEAR = True
DEFAULT_WRAP = False
DEFAULT_SCALE = 1.0

GRAPH_JSON_VERSION = 1


class TargetConfig(BaseModel):
    """How one pass's render target is set up. Document state, not shader text.

    Every field has a working default, so a pass renders correctly before anyone opens the panel.

    CONSTRAINED, because these knobs live in `graph.json` where nothing type-checks them: an
    unbounded `scale` allocates a framebuffer that fails to complete and takes the render loop
    down, and a `dtype` outside this set either raises in `ctx.texture` or loads fine and then
    crashes the panel's combo.
    """

    model_config = ConfigDict(frozen=True)

    # Upper bound as well as lower: a target LARGER than the canvas has no use case and is the
    # shape that exhausts VRAM.
    scale: float = Field(default=DEFAULT_SCALE, gt=0.0, le=1.0)
    dtype: Literal["f1", "f2", "f4"] = DEFAULT_DTYPE
    filter_linear: bool = DEFAULT_FILTER_LINEAR
    wrap: bool = DEFAULT_WRAP
    persist: bool = False

    def target_size(self, canvas_size: tuple[int, int]) -> tuple[int, int]:
        return (
            max(1, round(canvas_size[0] * self.scale)),
            max(1, round(canvas_size[1] * self.scale)),
        )


class PassEntry(BaseModel):
    """One pass's graph entry: what fills its inputs, and how its target is configured.

    `inputs` maps THIS pass's sampler uniform name to the pass that fills it, so an entry naming
    its own pass is feedback. An input naming a pass that does not exist is not an error: it reads
    black (`unresolved_inputs`), which is what keeps a half-built graph usable while you build it.
    """

    model_config = ConfigDict(frozen=True)

    inputs: dict[str, str] = {}
    target: TargetConfig = TargetConfig()


class PassLayout(BaseModel):
    """A pass's position in a spatial editor. Cosmetic, and kept in its own key so losing it never
    costs the effect."""

    model_config = ConfigDict(frozen=True)

    x: float = 0.0
    y: float = 0.0


class PassGraph(BaseModel):
    """The whole `graph.json`: the passes, the wiring, the targets and the output choice.

    App-written derived state, exactly as `node.json` is. The user edits it through the panel,
    never by hand.

    `output` names the pass the preview and export show. It may name a pass that is absent (a
    delete that did not fix it up, a hand-edit); `output_pass` resolves that rather than raising,
    so a document with a stale output still loads and still renders something.
    """

    version: int = GRAPH_JSON_VERSION
    output: str = ""
    passes: dict[str, PassEntry] = {}
    layout: dict[str, PassLayout] = {}

    @model_validator(mode="after")
    def _reject_unnamed_pass(self) -> "PassGraph":
        # A pass named "" makes `output_pass` return a falsy string, which every `if
        # graph.output_pass:` downstream reads as "no output". The name is also a filename.
        if "" in self.passes:
            raise ValueError("a pass name may not be empty")
        return self

    @property
    def output_pass(self) -> str | None:
        """The output pass's name, or None when the graph names none that exists.

        Falls back to the only pass when `output` is unset or stale and there is exactly one --
        a single-pass document has no ambiguity to resolve, so it renders without the panel.
        """
        if self.output in self.passes:
            return self.output
        if len(self.passes) == 1:
            return next(iter(self.passes))
        return None


@dataclass(frozen=True)
class PassPlan:
    """The resolved evaluation order, plus what each pass reads.

    `order` lists producers before consumers, each pass exactly once. `reads` excludes the
    self-edge (that is `feedback`); `unresolved_inputs` names the inputs pointing at a pass that
    does not exist, which read black.
    """

    order: list[str]
    reads: dict[str, set[str]] = field(default_factory=dict)
    feedback: set[str] = field(default_factory=set)
    unresolved_inputs: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphError:
    """A graph-level defect attributed to ONE pass, so the strip can point at its file.

    Not a `ShaderError`: this is about the wiring, not about source text, so it carries no line.
    Whoever renders it owns turning a pass name into a path.
    """

    pass_name: str
    message: str


def _cycle_message(trail: list[str], name: str) -> str:
    cycle = " -> ".join([*trail[trail.index(name) :], name])
    return (
        f"passes form a cycle: {cycle}. A pass may read ITSELF (that is its previous "
        f"frame), but a loop between passes has no order."
    )


def plan_passes(graph: PassGraph) -> tuple[PassPlan, list[GraphError]]:
    """Topologically order `graph`'s passes by which of each other's outputs they read.

    Passes are visited in sorted name order, so the plan is deterministic for a given graph
    regardless of dict insertion order -- an unstable order would make two exports of the same
    document differ.

    A cycle is reported once per pass that sits on it and the pass is left out of the order; the
    rest of the graph still plans, so one bad loop does not cost the document its other passes.
    """
    errors: list[GraphError] = []
    names = sorted(graph.passes)
    known = set(names)

    reads: dict[str, set[str]] = {}
    feedback: set[str] = set()
    unresolved: dict[str, dict[str, str]] = {}
    for name in names:
        entry = graph.passes[name]
        deps: set[str] = set()
        missing: dict[str, str] = {}
        for uniform, source in sorted(entry.inputs.items()):
            if source not in known:
                missing[uniform] = source
                continue
            if source == name:
                feedback.add(name)  # previous frame: no ordering constraint
                continue
            deps.add(source)
        reads[name] = deps
        if missing:
            unresolved[name] = missing

    order: list[str] = []
    state: dict[str, int] = {}  # 0 = visiting, 1 = done, 2 = failed
    failures: dict[str, GraphError] = {}

    def visit(name: str, trail: list[str]) -> bool:
        mark = state.get(name)
        if mark == 1:
            # Memoized: already emitted, so a shared ancestor appears ONCE in the order rather
            # than once per consumer. Appending here instead would re-render a diamond's base for
            # every path that reaches it.
            return True
        if mark == 2:
            # Memoized on the failing side too, so the walk stays linear: without it, every root
            # downstream of a cycle re-walks the whole failed region.
            return False
        if mark == 0:
            failures.setdefault(name, GraphError(name, _cycle_message(trail, name)))
            return False
        state[name] = 0
        ok = True
        for dep in sorted(reads[name]):
            if not visit(dep, [*trail, name]):
                ok = False
        if not ok:
            # A pass whose dependency is on a cycle has no order either. Reported on its own
            # account rather than silently drawn against a target nothing filled.
            state[name] = 2
            failures.setdefault(
                name, GraphError(name, "pass is not ordered: an input is on a cycle.")
            )
            return False
        state[name] = 1
        order.append(name)
        return True

    for name in names:
        visit(name, [])
    errors.extend(failures[name] for name in sorted(failures))

    return PassPlan(
        order=order,
        reads=reads,
        feedback=feedback,
        unresolved_inputs=unresolved,
    ), errors


def assert_plan_invariants(plan: PassPlan, graph: PassGraph) -> None:
    """Assert everything the renderer relies on. Run on EVERY plan a test builds.

    The memoization bug this exists for reads as slow rather than wrong, so a test that only
    eyeballs pixels cannot see it; the observable is the ORDER, and it must be checked every time
    rather than in one dedicated test.
    """
    assert len(plan.order) == len(set(plan.order)), (
        f"a pass appears twice in the order: {plan.order}"
    )
    assert set(plan.order) <= set(graph.passes), (
        f"the order names passes the graph does not have: "
        f"{sorted(set(plan.order) - set(graph.passes))}"
    )
    position = {name: i for i, name in enumerate(plan.order)}
    for name in plan.order:
        # `reads` is audited against the GRAPH, not taken as given: it is produced by the very
        # function under test, so a planner that drops an edge would otherwise be self-consistent
        # and its order would pass while a pass drew before its input.
        expected = {
            source
            for source in graph.passes[name].inputs.values()
            if source in graph.passes and source != name
        }
        assert plan.reads[name] == expected, (
            f"'{name}' reads {sorted(plan.reads[name])}, but the graph wires "
            f"{sorted(expected)}"
        )
        for dep in expected:
            assert dep in position, f"'{name}' is ordered before its input '{dep}' is drawn"
            assert position[dep] < position[name], (
                f"'{name}' draws before its input '{dep}'"
            )
    for name in plan.feedback:
        assert name in graph.passes, f"feedback names a pass that does not exist: '{name}'"
        assert name in graph.passes[name].inputs.values(), (
            f"'{name}' is marked feedback but wires no input to itself"
        )
        assert name not in plan.reads[name], (
            f"'{name}' reads itself as an ordering edge; feedback is the previous frame"
        )


def evaluation_order(graph: PassGraph, target: str) -> list[str]:
    """The passes to draw, in order, to produce `target` -- and nothing else.

    A document whose output is one branch of a wide graph does not pay for the branches nothing
    reads. A pass on a cycle, or absent, yields an empty order rather than a partial draw.

    This is the function the renderer calls, so it asserts the plan itself rather than trusting a
    test to have done it: the draw-once invariant has to hold on the path that actually draws.
    """
    plan, errors = plan_passes(graph)
    assert_plan_invariants(plan, graph)
    if errors and any(e.pass_name == target for e in errors):
        return []
    if target not in plan.order:
        return []
    needed: set[str] = set()
    stack = [target]
    while stack:
        name = stack.pop()
        if name in needed:
            continue
        needed.add(name)
        stack.extend(plan.reads[name])
    return [name for name in plan.order if name in needed]
