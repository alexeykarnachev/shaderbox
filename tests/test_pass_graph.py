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
    GraphError,
    PassEntry,
    PassGraph,
    PassPlan,
    TargetConfig,
    assert_plan_invariants,
    evaluation_order,
    plan_passes,
)


def _plan(graph: PassGraph) -> tuple[PassPlan, list[GraphError]]:
    plan, errors = plan_passes(graph)
    assert_plan_invariants(plan, graph)
    return plan, errors


def _graph(
    passes: dict[str, dict[str, str]], output: str = "", **kw: object
) -> PassGraph:
    return PassGraph(
        output=output,
        passes={name: PassEntry(inputs=inputs) for name, inputs in passes.items()},
        **kw,
    )


def test_a_chain_orders_producers_before_consumers() -> None:
    graph = _graph({"a": {}, "b": {"u_src": "a"}, "c": {"u_src": "b"}}, output="c")
    plan, errors = _plan(graph)
    assert errors == []
    assert plan.order == ["a", "b", "c"]
    assert plan.reads == {"a": set(), "b": {"a"}, "c": {"b"}}


def test_a_diamond_draws_its_shared_ancestor_once() -> None:
    # Appending on every visit instead of memoizing gives ["base", "left", "base", "right",
    # "out"] — the same picture, drawn twice.
    graph = _graph(
        {
            "base": {},
            "left": {"u_src": "base"},
            "right": {"u_src": "base"},
            "out": {"u_l": "left", "u_r": "right"},
        },
        output="out",
    )
    plan, errors = _plan(graph)
    assert errors == []
    assert plan.order.count("base") == 1
    assert plan.order.index("base") < plan.order.index("left")
    assert plan.order.index("right") < plan.order.index("out")


def test_the_order_is_deterministic_regardless_of_dict_order() -> None:
    # Two exports of the same document must not differ because a dict was built in another order.
    forward = _graph({"a": {}, "b": {"u_src": "a"}, "c": {"u_src": "a"}})
    backward = _graph({"c": {"u_src": "a"}, "b": {"u_src": "a"}, "a": {}})
    assert _plan(forward)[0].order == _plan(backward)[0].order


def test_a_pass_reading_itself_is_feedback_not_a_cycle() -> None:
    graph = _graph({"trail": {"u_src": "trail"}}, output="trail")
    plan, errors = _plan(graph)
    assert errors == []
    assert plan.feedback == {"trail"}
    assert plan.reads["trail"] == set()  # the previous frame constrains no ordering
    assert plan.order == ["trail"]


def test_feedback_mixed_with_a_real_input_keeps_both_halves() -> None:
    graph = _graph({"scene": {}, "trail": {"u_src": "scene", "u_prev": "trail"}})
    plan, errors = _plan(graph)
    assert errors == []
    assert plan.feedback == {"trail"}
    assert plan.reads["trail"] == {"scene"}
    assert plan.order == ["scene", "trail"]


def test_a_two_pass_cycle_is_an_error_per_pass_and_does_not_hang() -> None:
    graph = _graph({"a": {"u_src": "b"}, "b": {"u_src": "a"}})
    plan, errors = _plan(graph)
    assert {e.pass_name for e in errors} == {"a", "b"}
    assert all("cycle" in e.message for e in errors)
    assert plan.order == []


def test_a_cycle_reports_each_pass_exactly_once() -> None:
    # Counted, not set-compared: re-walking an already-failed pass from every remaining root
    # gives the right NAMES and quadratically many lines, which the strip shows verbatim.
    ring = _graph({f"p{i}": {"u_src": f"p{(i + 1) % 12}"} for i in range(12)})
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
    _plan(_graph(chain))
    assert len(visits) <= 4  # one detection per pass ON the cycle, not one per consumer


def test_a_cycle_costs_only_the_passes_on_it() -> None:
    graph = _graph({"ok": {}, "a": {"u_src": "b"}, "b": {"u_src": "a"}})
    plan, errors = _plan(graph)
    assert "ok" in plan.order
    assert "ok" not in {e.pass_name for e in errors}


def test_a_consumer_of_a_cycle_is_reported_and_left_unordered() -> None:
    # Without its own error the consumer would silently draw against a target nothing filled.
    graph = _graph({"a": {"u_src": "b"}, "b": {"u_src": "a"}, "sink": {"u_src": "a"}})
    plan, errors = _plan(graph)
    assert "sink" not in plan.order
    assert "sink" in {e.pass_name for e in errors}


def test_an_unfilled_input_is_not_an_error_and_the_pass_still_draws() -> None:
    # D3's graceful degradation: an input naming a pass that does not exist reads black, so a
    # half-built graph stays usable while you build it.
    graph = _graph({"blur": {"u_src": "nope"}}, output="blur")
    plan, errors = _plan(graph)
    assert errors == []
    assert plan.order == ["blur"]
    assert plan.unresolved_inputs == {"blur": {"u_src": "nope"}}
    assert plan.reads["blur"] == set()


def test_evaluation_order_skips_branches_the_output_does_not_read() -> None:
    graph = _graph(
        {"a": {}, "used": {"u_src": "a"}, "unused": {"u_src": "a"}}, output="used"
    )
    assert evaluation_order(graph, "used") == ["a", "used"]


def test_evaluation_order_asserts_the_plan_it_was_handed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The wire, not the mechanism: evaluation_order is what the renderer calls, so cutting its
    # assert must fail a test. A planner that emits a duplicated ancestor has to be caught HERE,
    # not only where a test happens to call assert_plan_invariants itself.
    graph = _graph({"base": {}, "top": {"u_src": "base"}}, output="top")
    good, _ = plan_passes(graph)
    doubled = PassPlan(
        order=["base", "base", "top"],
        reads=good.reads,
        feedback=good.feedback,
        unresolved_inputs=good.unresolved_inputs,
    )
    monkeypatch.setattr(
        pass_graph, "plan_passes", lambda _graph: (doubled, [])
    )
    with pytest.raises(AssertionError, match="appears twice"):
        evaluation_order(graph, "top")


def test_evaluation_order_of_an_absent_or_cyclic_target_is_empty() -> None:
    graph = _graph({"a": {"u_src": "b"}, "b": {"u_src": "a"}})
    assert evaluation_order(graph, "a") == []
    assert evaluation_order(_graph({"a": {}}), "ghost") == []


def test_output_pass_falls_back_for_a_single_pass_document() -> None:
    # A one-pass document renders before anyone has opened the panel to name an output.
    assert _graph({"only": {}}).output_pass == "only"
    assert _graph({"only": {}}, output="stale").output_pass == "only"


def test_output_pass_is_none_when_it_names_nothing_among_several() -> None:
    graph = _graph({"a": {}, "b": {}}, output="gone")
    assert graph.output_pass is None
    assert _graph({"a": {}, "b": {}}, output="b").output_pass == "b"


def test_target_defaults_are_the_measured_safe_ones() -> None:
    # f2, linear, clamp — 063 measured f1 saturating at 255 where f2 reached 7.0, and moderngl's
    # repeat default is wrong for a feedback border.
    target = TargetConfig()
    assert target.dtype == DEFAULT_DTYPE == "f2"
    assert target.filter_linear
    assert not target.wrap
    assert not target.persist
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
        "version": 1,
        "output": "composite",
        "passes": {
            "scene": {"inputs": {}, "target": {"scale": 1.0, "dtype": "f2"}},
            "bright": {
                "inputs": {"u_src": "scene"},
                "target": {"scale": 0.5, "dtype": "f2"},
            },
            "trail": {
                "inputs": {"u_src": "scene", "u_prev": "trail"},
                "target": {"scale": 1.0, "dtype": "f2", "persist": True},
            },
            "composite": {
                "inputs": {"u_lit": "scene", "u_glow": "bright", "u_trail": "trail"},
                "target": {"scale": 1.0, "dtype": "f1"},
            },
        },
        "layout": {"scene": {"x": 0, "y": 0}, "bright": {"x": 200, "y": -60}},
    }
    graph = PassGraph(**data)
    assert PassGraph(**graph.model_dump()) == graph
    assert graph.output_pass == "composite"
    assert graph.passes["trail"].target.persist
    assert graph.passes["composite"].target.dtype == "f1"
    plan, errors = _plan(graph)
    assert errors == []
    assert plan.feedback == {"trail"}
    assert plan.order.index("scene") == 0
    assert plan.order[-1] == "composite"
    assert evaluation_order(graph, "bright") == ["scene", "bright"]


def test_an_empty_graph_plans_to_nothing() -> None:
    plan, errors = _plan(PassGraph())
    assert plan.order == [] and errors == []
    assert PassGraph().output_pass is None
