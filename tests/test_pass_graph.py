"""The GL-free pass-graph model and its planner (065 stage 1).

Every plan built here goes through `_plan`, which asserts the plan invariants -- above all that a
shared ancestor appears ONCE in the order. A pass drawn N times renders the CORRECT picture and
only reads as slow, so no assertion about pixels can catch it; asserting on every plan can.
"""

import pytest
from pydantic import ValidationError

from shaderbox import pass_graph
from shaderbox.pass_graph import (
    DEFAULT_DTYPE,
    DTYPES,
    MAX_ITERATIONS,
    AutoSource,
    GraphError,
    NoSource,
    PassEntry,
    PassGraph,
    PassPlan,
    PassSource,
    TargetConfig,
    Wiring,
    assert_plan_invariants,
    evaluation_order,
    plan_passes,
    wired_pass,
)


def _plan(wiring: Wiring) -> tuple[PassPlan, list[GraphError]]:
    plan, errors = plan_passes(wiring)
    assert_plan_invariants(plan, wiring)
    return plan, errors


def test_a_chain_orders_producers_before_consumers() -> None:
    graph = {"a": {}, "b": {"u_src": "a"}, "c": {"u_src": "b"}}
    plan, errors = _plan(graph)
    assert errors == []
    assert plan.order == ["a", "b", "c"]
    assert plan.reads == {"a": set(), "b": {"a"}, "c": {"b"}}


def test_a_diamond_draws_its_shared_ancestor_once() -> None:
    # Appending on every visit instead of memoizing gives ["base", "left", "base", "right",
    # "out"] — the same picture, drawn twice.
    graph = {
        "base": {},
        "left": {"u_src": "base"},
        "right": {"u_src": "base"},
        "out": {"u_l": "left", "u_r": "right"},
    }
    plan, errors = _plan(graph)
    assert errors == []
    assert plan.order.count("base") == 1
    assert plan.order.index("base") < plan.order.index("left")
    assert plan.order.index("right") < plan.order.index("out")


def test_the_order_is_deterministic_regardless_of_dict_order() -> None:
    # Two exports of the same document must not differ because a dict was built in another order.
    forward = {"a": {}, "b": {"u_src": "a"}, "c": {"u_src": "a"}}
    backward = {"c": {"u_src": "a"}, "b": {"u_src": "a"}, "a": {}}
    assert _plan(forward)[0].order == _plan(backward)[0].order


def test_a_pass_reading_itself_is_feedback_not_a_cycle() -> None:
    graph = {"trail": {"u_src": "trail"}}
    plan, errors = _plan(graph)
    assert errors == []
    assert plan.feedback == {"trail"}
    assert plan.reads["trail"] == set()  # the previous frame constrains no ordering
    assert plan.order == ["trail"]


def test_feedback_mixed_with_a_real_input_keeps_both_halves() -> None:
    graph = {"scene": {}, "trail": {"u_src": "scene", "u_prev": "trail"}}
    plan, errors = _plan(graph)
    assert errors == []
    assert plan.feedback == {"trail"}
    assert plan.reads["trail"] == {"scene"}
    assert plan.order == ["scene", "trail"]


def test_a_two_pass_cycle_is_an_error_per_pass_and_does_not_hang() -> None:
    graph = {"a": {"u_src": "b"}, "b": {"u_src": "a"}}
    plan, errors = _plan(graph)
    assert {e.pass_name for e in errors} == {"a", "b"}
    assert all("cycle" in e.message for e in errors)
    assert plan.order == []


def test_a_cycle_reports_each_pass_exactly_once() -> None:
    # Counted, not set-compared: re-walking an already-failed pass from every remaining root
    # gives the right NAMES and quadratically many lines, which the strip shows verbatim.
    ring = {f"p{i}": {"u_src": f"p{(i + 1) % 12}"} for i in range(12)}
    _, errors = _plan(ring)
    assert len(errors) == 12
    assert len({e.pass_name for e in errors}) == 12


def test_a_cycle_does_not_make_the_walk_super_linear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Counted, not timed: every root downstream of a cycle would re-walk the whole failed region
    # without the failure memo, so the visit count goes quadratic in the chain length while the
    # error list stays correct and hides it.
    visits: list[str] = []
    original = pass_graph._cycle_message

    def counting(trail: list[str], name: str) -> str:
        visits.append(name)
        return original(trail, name)

    monkeypatch.setattr(pass_graph, "_cycle_message", counting)
    chain = {"a": {"u": "b"}, "b": {"u": "a"}}
    chain.update({f"c{i}": {"u": "a" if i == 0 else f"c{i - 1}"} for i in range(50)})
    _plan(chain)
    assert len(visits) <= 4  # one detection per pass ON the cycle, not one per consumer


def test_a_cycle_costs_only_the_passes_on_it() -> None:
    graph = {"ok": {}, "a": {"u_src": "b"}, "b": {"u_src": "a"}}
    plan, errors = _plan(graph)
    assert "ok" in plan.order
    assert "ok" not in {e.pass_name for e in errors}


def test_a_consumer_of_a_cycle_is_reported_and_left_unordered() -> None:
    # Without its own error the consumer would silently draw against a target nothing filled.
    graph = {"a": {"u_src": "b"}, "b": {"u_src": "a"}, "sink": {"u_src": "a"}}
    plan, errors = _plan(graph)
    assert "sink" not in plan.order
    assert "sink" in {e.pass_name for e in errors}


def test_an_unfilled_input_is_not_an_error_and_the_pass_still_draws() -> None:
    # D3's graceful degradation: a source naming a pass that does not exist reads black, so a
    # half-built graph stays usable while you build it. `wired_pass` is where it becomes "no
    # read", so the wiring the planner sees never names a missing pass.
    assert wired_pass(PassSource("nope"), "u_src", "blur", {"blur"}) is None
    plan, errors = _plan({"blur": {}})
    assert errors == []
    assert plan.order == ["blur"]
    assert plan.reads["blur"] == set()


def test_wired_pass_over_every_state() -> None:
    # (explicit, undecided, none, a texture) x (the named pass exists or not), plus `u_prev`.
    passes = {"df", "jfa", "edge"}
    assert wired_pass(PassSource("jfa"), "u_df", "edge", passes) == "jfa"
    assert wired_pass(PassSource("gone"), "u_df", "edge", passes) is None
    # Undecided: the NAME decides (069 D9), when it names a pass that exists.
    assert wired_pass(AutoSource(), "u_df", "edge", passes) == "df"
    assert wired_pass(AutoSource(), "u_df", "edge", {"jfa"}) is None
    # No `u_` prefix, no auto edge -- D9's rule is about `u_<pass>` names.
    assert wired_pass(AutoSource(), "df", "edge", passes) is None
    # The feedback exception wins over a sibling that happens to be called `prev`: D9 writes
    # `u_prev` down as reading yourself, so that is the branch a user can predict.
    assert (
        wired_pass(AutoSource(), "u_prev", "cascade", {"cascade", "prev"}) == "cascade"
    )
    # A decision for black, and a texture the user bound, read no pass whatever the name says.
    assert wired_pass(NoSource(), "u_df", "edge", passes) is None
    assert wired_pass(object(), "u_df", "edge", passes) is None


def test_evaluation_order_skips_branches_the_output_does_not_read() -> None:
    graph = {"a": {}, "used": {"u_src": "a"}, "unused": {"u_src": "a"}}
    assert evaluation_order(graph, "used") == ["a", "used"]


def test_evaluation_order_asserts_the_plan_it_was_handed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The wire, not the mechanism: evaluation_order is what the renderer calls, so cutting its
    # assert must fail a test. A planner that emits a duplicated ancestor has to be caught HERE,
    # not only where a test happens to call assert_plan_invariants itself.
    graph = {"base": {}, "top": {"u_src": "base"}}
    good, _ = plan_passes(graph)
    doubled = PassPlan(
        order=["base", "base", "top"],
        reads=good.reads,
        feedback=good.feedback,
    )
    monkeypatch.setattr(pass_graph, "plan_passes", lambda _wiring: (doubled, []))
    with pytest.raises(AssertionError, match="appears twice"):
        evaluation_order(graph, "top")


def test_evaluation_order_of_an_absent_or_cyclic_target_is_empty() -> None:
    graph = {"a": {"u_src": "b"}, "b": {"u_src": "a"}}
    assert evaluation_order(graph, "a") == []
    assert evaluation_order({"a": {}}, "ghost") == []


def test_output_pass_falls_back_for_a_single_pass_document() -> None:
    # A one-pass document renders before anyone has opened the panel to name an output.
    assert PassGraph(passes={"only": PassEntry()}).output_pass == "only"
    assert PassGraph(output="stale", passes={"only": PassEntry()}).output_pass == "only"


def test_output_pass_is_none_when_it_names_nothing_among_several() -> None:
    two = {"a": PassEntry(), "b": PassEntry()}
    assert PassGraph(output="gone", passes=two).output_pass is None
    assert PassGraph(output="b", passes=two).output_pass == "b"


def test_target_defaults_are_the_measured_safe_ones() -> None:
    # f2, linear, clamp — 063 measured f1 saturating at 255 where f2 reached 7.0, and moderngl's
    # repeat default is wrong for a feedback border.
    target = TargetConfig()
    assert target.dtype == DEFAULT_DTYPE == "f2"
    assert target.filter_linear
    assert not target.wrap
    assert target.target_size((800, 600)) == (800, 600)


def test_target_scale_shrinks_and_never_reaches_zero() -> None:
    assert TargetConfig(scale=0.5).target_size((800, 600)) == (400, 300)
    assert TargetConfig(scale=0.25).target_size((800, 603)) == (200, 151)
    assert TargetConfig(scale=0.001).target_size((100, 100)) == (1, 1)


def test_target_scale_is_bounded_on_the_model() -> None:
    # The bound lives on the model, not on the panel's slider: graph.json reaches the loader
    # without passing any widget, and an unbounded scale allocates a framebuffer that fails to
    # complete and takes the render loop down.
    for bad in (0.0, -1.0, 1.5, float("inf"), float("nan")):
        with pytest.raises(ValidationError):
            TargetConfig(scale=bad)
    assert TargetConfig(scale=1.0).scale == 1.0


def test_target_dtype_is_closed() -> None:
    # An unknown dtype either raises inside ctx.texture or loads fine and then crashes the combo.
    for bad in ("f8", "rgba", 3, None):
        with pytest.raises(ValidationError):
            TargetConfig(dtype=bad)
    assert {TargetConfig(dtype=d).dtype for d in DTYPES} == set(DTYPES)


def test_a_pass_may_not_be_unnamed() -> None:
    # "" is falsy, so an unnamed output pass would read as "no output" at every call site.
    with pytest.raises(ValidationError):
        PassGraph(passes={"": PassEntry()})


def test_the_spec_schema_round_trips() -> None:
    data = {
        "version": 2,
        "output": "composite",
        "passes": {
            "scene": {"target": {"scale": 1.0, "dtype": "f2"}},
            "bright": {"target": {"scale": 0.5, "dtype": "f2"}},
            "trail": {"target": {"scale": 1.0, "dtype": "f2"}},
            "composite": {"target": {"scale": 1.0, "dtype": "f1"}},
        },
    }
    graph = PassGraph(**data)
    assert PassGraph(**graph.model_dump()) == graph
    assert graph.output_pass == "composite"
    assert graph.passes["composite"].target.dtype == "f1"
    # What fills the inputs is each sampler's value, not the file (072): the wiring the planner
    # takes is built beside the graph.
    wiring = {
        "scene": {},
        "bright": {"u_src": "scene"},
        "trail": {"u_src": "scene", "u_prev": "trail"},
        "composite": {"u_lit": "scene", "u_glow": "bright", "u_trail": "trail"},
    }
    plan, errors = _plan(wiring)
    assert errors == []
    assert plan.feedback == {"trail"}
    assert plan.order.index("scene") == 0
    assert plan.order[-1] == "composite"
    assert evaluation_order(wiring, "bright") == ["scene", "bright"]


def test_an_empty_graph_plans_to_nothing() -> None:
    plan, errors = _plan({})
    assert plan.order == [] and errors == []
    assert PassGraph().output_pass is None


# --- 068: iteration count -------------------------------------------------------------


def test_iterations_are_bounded() -> None:
    # Same reason every graph.json number is bounded: nothing type-checks this file, and an
    # unbounded count is a frame-time bomb.
    assert PassEntry().iterations == 1
    assert PassEntry(iterations=MAX_ITERATIONS).iterations == MAX_ITERATIONS
    with pytest.raises(ValidationError):
        PassEntry(iterations=0)
    with pytest.raises(ValidationError):
        PassEntry(iterations=MAX_ITERATIONS + 1)


def test_graph_edits_preserve_fields_they_do_not_name() -> None:
    # with_target once REBUILT the entry field-by-field, so retargeting reset iterations 9 -> 1
    # and a JFA chain silently degraded. The falsifier is any edit verb that constructs a
    # PassEntry instead of copying one.
    graph = PassGraph(
        output="jfa",
        passes={"jfa": PassEntry(iterations=9, target=TargetConfig(dtype="f4"))},
    )
    retargeted = graph.with_target("jfa", TargetConfig(dtype="f1"))
    assert retargeted.passes["jfa"].iterations == 9
    assert retargeted.passes["jfa"].target.dtype == "f1"
    assert graph.with_output("jfa").passes["jfa"].iterations == 9
