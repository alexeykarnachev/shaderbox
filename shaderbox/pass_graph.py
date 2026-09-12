"""The pass graph: which passes a document has, how each target is configured, which pass is the
output, and the planner that orders the draw (065, reshaped by 072).

A document holds several passes forming a DAG. Each pass is its own `.glsl` file with its own
`main()` and its own render target; one pass is the document's output. A pass declares an input
as an ordinary `uniform sampler2D`, and that sampler's VALUE says what fills it (072): a
`PassSource` names a pass, a `NoSource` reads black by decision, an `AutoSource` lets the
sampler's NAME decide (069 D9), and a texture the user bound is itself. `wired_pass` is the one
function that turns a value into "the pass this sampler reads", and the planner takes the
resulting WIRING (pass -> uniform -> pass), so nothing here knows about textures or GL.

`graph.json` holds the rest: the output choice and each pass's target and run count. It is
app-written derived state, never parsed out of comments, and everything here is pure data:
the model, the topological order, the cycle check and the feedback marking are unit-testable
with no context and importable from anywhere without a cycle.

It also owns the canvas-dimension bounds (`MIN_CANVAS_PX` / `MAX_CANVAS_PX`) and the
`clamp_canvas_size` both entry points funnel through, beside `TargetConfig.scale`'s bound, which
is the sibling constraint on the same quantity.

Two rules the rest of the engine leans on:

- **A pass reading itself is FEEDBACK, not a cycle.** It consumes the previous frame, so it
  contributes no ordering constraint. Excluding the self-edge from the cycle check while keeping it
  in the wiring is the trap that sinks a naive implementation.
- **A pass appears at most ONCE in the order.** A diamond's shared ancestor draws once, not once per
  consuming path. A duplicated pass renders the CORRECT picture, just N times, so it reads as slow
  rather than wrong and no pixel test can see it: `assert_plan_invariants` is the observable, and
  `evaluation_order` runs it on the path that actually draws.
"""

import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DTYPES: tuple[str, ...] = ("f1", "f2", "f4")

# f2, not f1: 063 measured f1 saturating at 255 on the FIRST accumulate pass where f2 reached
# exactly 7.0, so the safe value is the default and f1 is the opt-in. clamp inverts moderngl's
# repeat_x/y=True, which is wrong for a feedback border.
TargetDtype = Literal["f1", "f2", "f4"]
TARGET_DTYPES: tuple[TargetDtype, ...] = ("f1", "f2", "f4")
DEFAULT_DTYPE: TargetDtype = "f2"
DEFAULT_FILTER_LINEAR = True
DEFAULT_WRAP = False
DEFAULT_SCALE = 1.0

GRAPH_JSON_VERSION = 2

# A pass name is a FILENAME and a graph key, so it stays to the characters both accept. A group
# name is a pass-name prefix and a border label, so it obeys the same rule.
PASS_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_GROUP_PATTERN = r"^([A-Za-z_][A-Za-z0-9_]*)?$"

# 64 doublings covers a 2^64 canvas, so this bounds the frame cost without bounding any real
# effect: JFA needs ceil(log2(max_dim)) and a cascade stack ceil(log4(diagonal)) + 1.
MAX_ITERATIONS = 64

# A node's stored place on the graph canvas (092 D6), in canvas units at zoom 1. Bounded on the
# model because `graph.json` type-checks nothing: a NaN would poison every bounding box the
# canvas computes and 1e30 would put a node past any reachable pan.
MAX_GRAPH_COORD = 100_000.0
GraphCoord = Annotated[
    float, Field(allow_inf_nan=False, ge=-MAX_GRAPH_COORD, le=MAX_GRAPH_COORD)
]

# A canvas dimension the render path can actually allocate. Both entry points -- the Document
# tab's W x H fields and the copilot's set_canvas_size -- clamp through here.
MIN_CANVAS_PX: int = 16
MAX_CANVAS_PX: int = 4096


def clamp_canvas_size(size: tuple[int, int]) -> tuple[int, int]:
    w, h = size
    return (
        max(MIN_CANVAS_PX, min(MAX_CANVAS_PX, w)),
        max(MIN_CANVAS_PX, min(MAX_CANVAS_PX, h)),
    )


class TargetConfig(BaseModel):
    """How one pass's render target is set up. Document state, not shader text.

    Every field has a working default, so a pass renders correctly before anyone opens the panel.

    CONSTRAINED, because these knobs live in `graph.json` where nothing type-checks them: an
    unbounded `scale` allocates a framebuffer that fails to complete and takes the render loop
    down, and a `dtype` outside this set either raises in `context.texture` or loads fine and then
    crashes the panel's combo.
    """

    model_config = ConfigDict(frozen=True)

    # Upper bound as well as lower: a target LARGER than the canvas has no use case and is the
    # shape that exhausts VRAM.
    scale: float = Field(default=DEFAULT_SCALE, gt=0.0, le=1.0)
    dtype: TargetDtype = DEFAULT_DTYPE
    filter_linear: bool = DEFAULT_FILTER_LINEAR
    wrap: bool = DEFAULT_WRAP

    def target_size(self, canvas_size: tuple[int, int]) -> tuple[int, int]:
        return (
            max(1, round(canvas_size[0] * self.scale)),
            max(1, round(canvas_size[1] * self.scale)),
        )


class PassEntry(BaseModel):
    """One pass's graph entry: how its target is configured and how many times it runs.

    What fills its inputs is not here: that is each sampler's VALUE (072, `wired_pass`).
    """

    model_config = ConfigDict(frozen=True)

    target: TargetConfig = TargetConfig()
    # Draw this pass N times in sequence within one frame, feeding `u_pass_iteration` /
    # `u_pass_iterations` so one shader can be the whole chain -- a cascade level, a jump
    # flood's halving offset. A self-reading iterated pass ping-pongs BETWEEN iterations,
    # not between frames. Bounded for the reason every number here is: `graph.json` type-checks nothing,
    # and an unbounded count is a frame-time bomb.
    #
    # The count is the AUTHOR'S and the engine does not second-guess it. A resize can leave a
    # chain short (a base-2 jump flood spans 2^N, so 512 wants 9 runs and 1024 wants 10) -- but
    # the engine cannot tell a base-2 chain from the base-4 cascade stack beside it, and a check
    # assuming one warns falsely on the other. The shader's author knows its base; the engine
    # only knows the number.
    iterations: int = Field(default=1, ge=1, le=MAX_ITERATIONS)
    # The group this pass belongs to, or "" (091). A label and nothing more: a group exists
    # while a pass carries its name, so there is no member list to keep in step with this.
    group: str = Field(default="", pattern=_GROUP_PATTERN)
    # Where the graph canvas draws this pass (092 D6): None means never placed, and the canvas
    # lays such a pass out by rank every frame. Written only by a drag or by Arrange.
    position: tuple[GraphCoord, GraphCoord] | None = None


class PassGraph(BaseModel):
    """The whole `graph.json`: the passes, their targets and run counts, and the output choice.

    App-written derived state, exactly as `document.json` is. The user edits it through the panel,
    never by hand.

    `output` names the pass the preview and export show. It may name a pass that is absent (a
    delete that did not fix it up, a hand-edit); `output_pass` resolves that rather than raising,
    so a document with a stale output still loads and still renders something.
    """

    version: int = GRAPH_JSON_VERSION
    output: str = ""
    passes: dict[str, PassEntry] = {}

    @model_validator(mode="after")
    def _reject_unnamed_pass(self) -> "PassGraph":
        # A pass named "" makes `output_pass` return a falsy string, which every `if
        # graph.output_pass:` downstream reads as "no output". The name is also a filename.
        if "" in self.passes:
            raise ValueError("a pass name may not be empty")
        return self

    def with_passes(
        self, entries: dict[str, "PassEntry"], output: str | None = None
    ) -> "PassGraph":
        """A copy carrying `entries` (and optionally a new output).

        Every edit funnels through here rather than calling `model_copy` at each call site: the
        field name and the passes/ DIRECTORY share a word, so a bare string is indistinguishable
        from the path and the single-home guard cannot tell them apart.

        The funnel is the point; the COPY inside it still goes through `model_copy`, so a field
        added to this model tomorrow survives every edit rather than silently resetting.
        """
        # The keys come from the model's own field names rather than string literals: the field
        # is spelled the same as the passes/ DIRECTORY, and `test_basename_is_never_respelled`
        # fails a second spelling because a rename would then break some readers silently.
        update: dict[str, object] = {_PASSES_FIELD: entries}
        if output is not None:
            update[_OUTPUT_FIELD] = output
        return self.model_copy(update=update)

    def with_target(self, name: str, target: "TargetConfig") -> "PassGraph":
        entry = self.passes.get(name, PassEntry())
        return self.with_passes(
            {**self.passes, name: entry.model_copy(update={"target": target})}
        )

    def with_output(self, name: str) -> "PassGraph":
        return self.with_passes(self.passes, output=name)

    def with_group(self, name: str, group: str) -> "PassGraph":
        entry = self.passes.get(name, PassEntry())
        return self.with_passes(
            {**self.passes, name: entry.model_copy(update={"group": group})}
        )

    def with_positions(
        self, positions: Mapping[str, tuple[float, float] | None]
    ) -> "PassGraph":
        """A copy with every named pass's `position` replaced; names without an entry get
        a default one, so a placement never depends on a target row having been written."""
        entries = dict(self.passes)
        for name, position in positions.items():
            entry = entries.get(name, PassEntry())
            # Validated, not `model_copy`-ed: a copy skips the field's bounds, and a position
            # is the one entry field a gesture writes with a number the user did not type.
            entries[name] = PassEntry.model_validate(
                {**entry.model_dump(), "position": position}
            )
        return self.with_passes(entries)

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


_AUTO_PREFIX = "u_"
_FEEDBACK_UNIFORM = "u_prev"


@dataclass(frozen=True)
class PassSource:
    """A sampler reads the named pass's live canvas; naming its OWN pass reads the previous
    frame (feedback)."""

    name: str


@dataclass(frozen=True)
class NoSource:
    """A sampler reads black, by decision: the name rule must not fill it."""


@dataclass(frozen=True)
class AutoSource:
    """A sampler nobody decided about: its NAME fills it at bind time (069 D9), or it reads
    black."""


SamplerSource = PassSource | NoSource | AutoSource

# pass -> sampler uniform -> the pass it reads. Every pass of the document is a key, so the
# planner orders all of them; a sampler that reads no pass has no entry.
Wiring = Mapping[str, Mapping[str, str]]


def _auto_source(uniform: str, consumer: str) -> str:
    """The pass `uniform`'s NAME points at, or `""` when the name says nothing (069 D9).

    `u_prev` reads the consumer itself -- the feedback exception D9 writes down, which wins over
    a sibling pass that happens to be called `prev`. A name without the `u_` prefix names no
    pass: D9's rule is about `u_<pass>`, and a bare `tex` is outside it.
    """
    if uniform == _FEEDBACK_UNIFORM:
        return consumer
    if not uniform.startswith(_AUTO_PREFIX):
        return ""
    return uniform[len(_AUTO_PREFIX) :]


def wired_pass(
    source: object, uniform: str, consumer: str, passes: Collection[str]
) -> str | None:
    """The pass `consumer`'s `uniform` reads, given the sampler's VALUE, or None when it reads
    no pass: a `NoSource`, a texture the user bound, or a name that matches no pass.

    A `PassSource` naming a pass that does not exist reads black rather than raising: that is
    what keeps a half-built graph usable while you build it (065 D3).

    GL-free: `passes` are NAMES and `source` is only inspected by type, so nothing here compiles,
    binds or touches a context.
    """
    if isinstance(source, PassSource):
        return source.name if source.name in passes else None
    if isinstance(source, AutoSource):
        auto = _auto_source(uniform, consumer)
        return auto if auto and auto in passes else None
    return None


@dataclass(frozen=True)
class PassPlan:
    """The resolved evaluation order, plus what each pass reads.

    `order` lists producers before consumers, each pass exactly once. `reads` excludes the
    self-edge (that is `feedback`).
    """

    order: list[str]
    reads: dict[str, set[str]] = field(default_factory=dict)
    feedback: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class GraphError:
    """A graph-level defect attributed to ONE pass, so the strip can point at its file.

    Not a `ShaderError`: this is about the wiring, not about source text, so it carries no line.
    Whoever renders it owns turning a pass name into a path.
    """

    pass_name: str
    message: str


_CYCLE_PREFIX = "passes form a cycle: "


def _cycle_message(trail: list[str], name: str) -> str:
    cycle = " -> ".join([*trail[trail.index(name) :], name])
    return (
        f"{_CYCLE_PREFIX}{cycle}. A pass may read ITSELF (that is its previous "
        f"frame), but a loop between passes has no order."
    )


def plan_passes(wiring: Wiring) -> tuple[PassPlan, list[GraphError]]:
    """Topologically order the passes of `wiring` by which of each other's outputs they read.

    Passes are visited in sorted name order, so the plan is deterministic for a given wiring
    regardless of dict insertion order -- an unstable order would make two exports of the same
    document differ.

    A cycle is reported once per pass that sits on it and the pass is left out of the order; the
    rest of the graph still plans, so one bad loop does not cost the document its other passes.
    """
    errors: list[GraphError] = []
    names = sorted(wiring)

    reads: dict[str, set[str]] = {}
    feedback: set[str] = set()
    for name in names:
        deps: set[str] = set()
        for source in wiring[name].values():
            if source == name:
                feedback.add(name)  # previous frame: no ordering constraint
                continue
            deps.add(source)
        reads[name] = deps

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

    return PassPlan(order=order, reads=reads, feedback=feedback), errors


def assert_plan_invariants(plan: PassPlan, wiring: Wiring) -> None:
    """Assert everything the renderer relies on. Run on EVERY plan a test builds.

    The memoization bug this exists for reads as slow rather than wrong, so a test that only
    eyeballs pixels cannot see it; the observable is the ORDER, and it must be checked every time
    rather than in one dedicated test.
    """
    assert len(plan.order) == len(set(plan.order)), (
        f"a pass appears twice in the order: {plan.order}"
    )
    assert set(plan.order) <= set(wiring), (
        f"the order names passes the wiring does not have: "
        f"{sorted(set(plan.order) - set(wiring))}"
    )
    position = {name: i for i, name in enumerate(plan.order)}
    for name in plan.order:
        # `reads` is audited against the WIRING, not taken as given: it is produced by the very
        # function under test, so a planner that drops an edge would otherwise be self-consistent
        # and its order would pass while a pass drew before its input.
        expected = {source for source in wiring[name].values() if source != name}
        assert plan.reads[name] == expected, (
            f"'{name}' reads {sorted(plan.reads[name])}, but the wiring says "
            f"{sorted(expected)}"
        )
        for dep in expected:
            assert dep in position, (
                f"'{name}' is ordered before its input '{dep}' is drawn"
            )
            assert position[dep] < position[name], (
                f"'{name}' draws before its input '{dep}'"
            )
    for name in plan.feedback:
        assert name in wiring, f"feedback names a pass that does not exist: '{name}'"
        assert name in wiring[name].values(), (
            f"'{name}' is marked feedback but wires no input to itself"
        )
        assert name not in plan.reads[name], (
            f"'{name}' reads itself as an ordering edge; feedback is the previous frame"
        )


def plan_for_output(wiring: Wiring, target: str) -> tuple[list[str], list[GraphError]]:
    """`evaluation_order` plus the wiring's errors, so a renderer plans the graph ONCE per frame."""
    plan, errors = plan_passes(wiring)
    assert_plan_invariants(plan, wiring)
    return _order_for(plan, errors, target), errors


def evaluation_order(wiring: Wiring, target: str) -> list[str]:
    """The passes to draw, in order, to produce `target` -- and nothing else.

    A document whose output is one branch of a wide graph does not pay for the branches nothing
    reads. A pass on a cycle, or absent, yields an empty order rather than a partial draw.

    This is the function the renderer calls, so it asserts the plan itself rather than trusting a
    test to have done it: the draw-once invariant has to hold on the path that actually draws.
    """
    return plan_for_output(wiring, target)[0]


def _order_for(plan: PassPlan, errors: list[GraphError], target: str) -> list[str]:
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


# The field names, taken from the model rather than written out: `passes` is spelled the same as
# the passes/ DIRECTORY, and a bare literal here trips the single-spelling guard
# (`test_basename_is_never_respelled`). Deriving them also keeps a rename honest.
_PASSES_FIELD = next(f for f in PassGraph.model_fields if f.startswith("pass"))
_OUTPUT_FIELD = next(f for f in PassGraph.model_fields if f.startswith("out"))


def strip_order(names: Iterable[str], wiring: Wiring) -> list[str]:
    """The pass strip's tile order: producers left of consumers, STABLE across output changes.

    `plan_passes` gives the deterministic topological order and never looks at the output, so
    picking a different output cannot shuffle the tiles. Passes it leaves out (cycle members,
    passes with no wiring entry) are appended by name so every pass still gets a tile.
    """
    known = set(names)
    order = [n for n in plan_passes(wiring)[0].order if n in known]
    return order + sorted(known - set(order))


def entry_points(wiring: Wiring) -> list[str]:
    """The passes of `wiring` that read no OTHER pass: a bundle's inputs (091 D3).

    A self-read is feedback, not an input, so a pass reading only its own previous frame is a
    root. Sorted by name, like everything the planner answers.
    """
    return sorted(
        name
        for name, reads in wiring.items()
        if all(source == name for source in reads.values())
    )


def readers_of(wiring: Wiring, fed: str) -> set[tuple[str, str]]:
    """The `(pass, sampler)` pairs of `wiring` that read `fed`: an import's default handovers
    (091 D6)."""
    return {
        (name, uniform)
        for name, reads in wiring.items()
        for uniform, read in reads.items()
        if read == fed
    }


def group_slug(name: str) -> str:
    """A group name prefilled from a document's display name (091 D2): the first word,
    lowercased, made legal as a pass-name prefix."""
    words = name.split()
    word = re.sub(r"[^A-Za-z0-9_]", "_", words[0].lower()) if words else ""
    if not word:
        return "preset"
    if not re.match(r"^[A-Za-z_]", word):
        word = f"g_{word}"
    return word


def group_runs(order: Sequence[str], groups: Mapping[str, str]) -> list[list[str]]:
    """`order` cut into runs of consecutive passes sharing one non-empty group; an ungrouped
    pass is a run of one. By ADJACENCY, never by name: a group an outside pass interrupts is
    two runs, which is what the strip draws (091 D7)."""
    runs: list[list[str]] = []
    for name in order:
        group = groups.get(name, "")
        if runs and group and groups.get(runs[-1][-1], "") == group:
            runs[-1].append(name)
        else:
            runs.append([name])
    return runs


def step_in_order(order: Sequence[str], current: str, step: int) -> str | None:
    """The name `step` tiles away from `current` in `order`, wrapping at both ends; the first
    tile when `current` is not in the order; None when the order is empty."""
    if not order:
        return None
    if current not in order:
        return order[0]
    return order[(order.index(current) + step) % len(order)]


# ---- the graph canvas's pure half (092) ----------------------------------------------------
# Everything the canvas decides that is not a pixel: where an unplaced pass goes, which ports
# a node has and in what state, what a group's box exposes, whether a drop may land, and
# which edges a cycle message names. Pure over names, sampler values and a wiring, so a test
# asserts each without a window.


def group_name_error(group: str, pass_names: Collection[str]) -> str:
    """Why `group` may not be written, or `""` when it may (092 D17): the pattern first, then
    the one namespace passes and groups share -- the canvas keys nodes and boxes by name."""
    if not group:
        return ""
    if not PASS_NAME_RE.match(group):
        return "a group name starts with a letter and holds letters, digits and underscores"
    if group in pass_names:
        return "a pass and a group cannot share a name"
    return ""


def graph_ranks(wiring: Wiring) -> dict[str, int]:
    """Each pass's column: the longest path from a root over the non-self edges. A pass the
    planner leaves unordered (on a cycle, or downstream of one) sits at rank 0."""
    plan = plan_passes(wiring)[0]
    ranks: dict[str, int] = {}
    for name in plan.order:
        deps = [ranks[d] for d in plan.reads[name] if d in ranks]
        ranks[name] = 1 + max(deps) if deps else 0
    for name in wiring:
        ranks.setdefault(name, 0)
    return ranks


def rank_layout(
    wiring: Wiring,
    names: Collection[str],
    groups: Mapping[str, str],
    sizes: Mapping[str, tuple[float, float]],
    placed: Mapping[str, tuple[float, float]],
    gap_x: float,
    gap_y: float,
) -> dict[str, tuple[float, float]]:
    """Where each of `names` goes (092 D6): columns by `graph_ranks`, left to right at `gap_x`;
    within a column, group members adjacent, then by the mean y of the pass's predecessors
    (a placed predecessor contributes its stored y), then strip order; rows at `gap_y`, each
    column centred on the tallest. Every pass of `wiring` is laid out so the picture is the
    same whichever subset is asked for; only `names` are returned, so a caller placing the
    unplaced passes never overwrites a stored one.
    """
    order = strip_order(wiring, wiring)
    strip_index = {name: i for i, name in enumerate(order)}
    ranks = graph_ranks(wiring)
    columns: dict[int, list[str]] = {}
    for name in order:
        columns.setdefault(ranks[name], []).append(name)
    group_anchor: dict[str, int] = {}
    for name in order:
        group = groups.get(name, "")
        if group:
            group_anchor.setdefault(group, strip_index[name])
    laid: dict[str, tuple[float, float]] = {}
    x = 0.0
    for rank in sorted(columns):
        column = columns[rank]

        def sort_key(name: str) -> tuple[int, float, int]:
            group = groups.get(name, "")
            anchor = group_anchor[group] if group else strip_index[name]
            preds = [
                s
                for s in wiring[name].values()
                if s != name and (s in laid or s in placed)
            ]
            ys = [(placed.get(s) or laid[s])[1] for s in preds]
            bary = sum(ys) / len(ys) if ys else float(strip_index[name])
            return (anchor, bary, strip_index[name])

        column.sort(key=sort_key)
        heights = [sizes.get(name, (0.0, 0.0))[1] for name in column]
        extent = sum(heights) + gap_y * (len(column) - 1)
        widest = max(sizes.get(name, (0.0, 0.0))[0] for name in column)
        y = 0.0
        for name, height in zip(column, heights, strict=True):
            laid[name] = (x, y)
            y += height + gap_y
        # Centre the column once its extent is known: the tallest column is not known until
        # every column is measured, so the vertical centring is applied in a second pass.
        columns[rank] = column
        x += widest + gap_x
    tallest = max(
        (
            sum(sizes.get(n, (0.0, 0.0))[1] for n in col) + gap_y * (len(col) - 1)
            for col in columns.values()
        ),
        default=0.0,
    )
    for column in columns.values():
        extent = sum(sizes.get(n, (0.0, 0.0))[1] for n in column) + gap_y * (
            len(column) - 1
        )
        shift = (tallest - extent) / 2.0
        for name in column:
            px, py = laid[name]
            laid[name] = (px, py + shift)
    return {name: laid[name] for name in names if name in laid}


PortKind = Literal["wired", "unfilled", "none", "prev", "media"]


@dataclass(frozen=True)
class Port:
    """One input of a node (092 D1): the sampler, what it reads, and how that reads on the
    dot -- a pass, nothing yet, black by decision, its own previous frame, or a bound
    texture."""

    sampler: str
    kind: PortKind
    source: str | None = None


def node_ports(
    declared: Sequence[str],
    values: Mapping[str, object],
    wiring_row: Mapping[str, str],
    name: str,
) -> list[Port]:
    """A node's input ports, one per sampler the COMPILED program declares, in the program's
    order with every self-read moved last. The state comes from the sampler's VALUE, the way
    the uniforms panel's row decides it: a value that is no source kind is a bound texture.
    A stored row for a sampler the program no longer declares is no port, which is what keeps
    a drop from writing a row nothing reads."""
    ports: list[Port] = []
    feedback: list[Port] = []
    for sampler in declared:
        value = values.get(sampler, AutoSource())
        source = wiring_row.get(sampler)
        if source == name:
            feedback.append(Port(sampler, "prev", source))
        elif source is not None:
            ports.append(Port(sampler, "wired", source))
        elif isinstance(value, NoSource):
            ports.append(Port(sampler, "none"))
        elif isinstance(value, PassSource | AutoSource):
            ports.append(Port(sampler, "unfilled"))
        else:
            ports.append(Port(sampler, "media"))
    return ports + feedback


@dataclass(frozen=True)
class BoxPort:
    member: str
    port: Port
    label: str


@dataclass(frozen=True)
class Boundary:
    """What a group's box exposes at the root (092 D5): every member slot that reads outside
    the group or nothing, every member read from outside plus the bundle output, and the
    member whose picture the box shows."""

    inputs: list[BoxPort]
    outputs: list[str]
    bundle: str


def bundle_output(members: Sequence[str], wiring: Wiring, output: str) -> str:
    """The member a box stands for (092 D5): the document output when it is a member; else the
    first member (strip order) an outside pass reads; else the last member no member reads;
    else the last member. Total: a box always has a picture."""
    if output in members:
        return output
    inside = set(members)
    read_outside = {
        source
        for reader, row in wiring.items()
        if reader not in inside
        for source in row.values()
        if source in inside
    }
    for member in members:
        if member in read_outside:
            return member
    read_inside = {
        source
        for member in members
        for source in wiring.get(member, {}).values()
        if source in inside and source != member
    }
    unread = [m for m in members if m not in read_inside]
    return (unread or list(members))[-1]


def group_boundary(
    members: Sequence[str],
    ports: Mapping[str, Sequence[Port]],
    wiring: Wiring,
    output: str,
) -> Boundary:
    """`members` in strip order; `ports` each member's `node_ports`."""
    inside = set(members)
    slots: list[tuple[str, Port]] = []
    for member in members:
        for port in ports.get(member, ()):
            if port.kind == "wired" and port.source in inside:
                continue
            if port.kind == "prev":
                continue
            slots.append((member, port))
    counts: dict[str, int] = {}
    for _, port in slots:
        counts[port.sampler] = counts.get(port.sampler, 0) + 1
    inputs = [
        BoxPort(
            member,
            port,
            port.sampler if counts[port.sampler] == 1 else f"{member}.{port.sampler}",
        )
        for member, port in slots
    ]
    read_outside = {
        source
        for reader, row in wiring.items()
        if reader not in inside
        for source in row.values()
        if source in inside
    }
    bundle = bundle_output(members, wiring, output)
    outputs = [m for m in members if m in read_outside or m == bundle]
    return Boundary(inputs=inputs, outputs=outputs, bundle=bundle)


def wiring_with(
    wiring: Wiring, consumer: str, sampler: str, producer: str
) -> dict[str, dict[str, str]]:
    """`wiring` with one more read: what the document would bind after a drop."""
    hypothetical = {name: dict(row) for name, row in wiring.items()}
    hypothetical.setdefault(consumer, {})[sampler] = producer
    return hypothetical


def refuse_drop(wiring: Wiring, consumer: str, sampler: str, producer: str) -> str:
    """Why a wire from `producer` into `consumer.sampler` may not land, or `""` (092 D12):
    the hypothetical wiring is planned and ANY error refuses, since the culprit the planner
    names need not be either endpoint. A self-read is feedback and passes."""
    errors = plan_passes(wiring_with(wiring, consumer, sampler, producer))[1]
    if not errors:
        return ""
    culprit = next(
        (e for e in errors if e.message.startswith(_CYCLE_PREFIX)), errors[0]
    )
    return culprit.message


def cycle_edges(errors: Sequence[GraphError]) -> set[tuple[str, str]]:
    """The `(producer, consumer)` pairs on every cycle the planner reported (092 D11). One
    pass per cycle carries the trail (`a -> b -> a` reads: `a` reads `b`, `b` reads `a`); the
    victims' messages name no edge."""
    edges: set[tuple[str, str]] = set()
    for error in errors:
        if not error.message.startswith(_CYCLE_PREFIX):
            continue
        trail = error.message[len(_CYCLE_PREFIX) :].split(".")[0].split(" -> ")
        for consumer, producer in pairwise(trail):
            edges.add((producer, consumer))
    return edges
