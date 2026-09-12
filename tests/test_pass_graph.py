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
    Port,
    TargetConfig,
    Wiring,
    assert_plan_invariants,
    bundle_output,
    cycle_edges,
    evaluation_order,
    graph_ranks,
    group_boundary,
    group_name_error,
    node_ports,
    plan_passes,
    rank_layout,
    refuse_drop,
    wired_pass,
    wiring_with,
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
    # An unknown dtype either raises inside context.texture or loads fine and then crashes the combo.
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


# ---------------------------------------------------------------------------
# 091 -- entry points, the group slug
# ---------------------------------------------------------------------------


def test_entry_points_are_the_passes_reading_no_other_pass() -> None:
    bloom = {
        "scene": {},
        "bright": {"u_scene": "scene"},
        "trail": {"u_scene": "scene", "u_prev": "trail"},
    }
    assert pass_graph.entry_points(bloom) == ["scene"]
    # A self-read is feedback, not an input: its OWN wiring, since in every real shape the
    # self-reader also reads a sibling and the bug is invisible there. Falsifier: counting the
    # self-read as an input answers [].
    assert pass_graph.entry_points({"acc": {"u_prev": "acc"}}) == ["acc"]
    assert pass_graph.entry_points(
        {"fg": {}, "bg": {}, "mix": {"u_fg": "fg", "u_bg": "bg"}}
    ) == [
        "bg",
        "fg",
    ]
    assert pass_graph.entry_points({"lone": {}}) == ["lone"]


def test_group_slug_is_the_first_word_made_legal() -> None:
    # Falsifier: `2D SDF` -> `2d`, which fails PASS_NAME_RE and rejects every import from it
    # over a name the user never typed.
    assert pass_graph.group_slug("Bloom Chain") == "bloom"
    assert pass_graph.group_slug("Radiance Cascades") == "radiance"
    assert pass_graph.group_slug("2D SDF") == "g_2d"
    assert pass_graph.group_slug("") == "preset"
    for name in ("Bloom Chain", "2D SDF", "", "a-b c"):
        assert pass_graph.PASS_NAME_RE.match(pass_graph.group_slug(name))


# ---- the graph canvas's pure half (092) ----------------------------------------------------

# The bloom fixture as a host sees it: scene feeds the bundle, final reads its output.
_BLOOM: dict[str, dict[str, str]] = {
    "scene": {},
    "b_bright": {"u_scene": "scene"},
    "b_trail": {"u_scene": "scene", "u_prev": "b_trail"},
    "b_blur": {"u_bright": "b_bright"},
    "b_comp": {"u_scene": "scene", "u_blur": "b_blur", "u_trail": "b_trail"},
    "final": {"u_b_comp": "b_comp"},
}
_BLOOM_MEMBERS = ["b_bright", "b_trail", "b_blur", "b_comp"]
_BLOOM_GROUPS = dict.fromkeys(_BLOOM_MEMBERS, "bloom")
_SIZE = dict.fromkeys(_BLOOM, (100.0, 120.0))


def _ports_of(wiring: dict[str, dict[str, str]]) -> dict[str, list[Port]]:
    # Every declared sampler is the wiring's sampler, valued AutoSource (the default).
    return {name: node_ports(list(row), {}, row, name) for name, row in wiring.items()}


def test_rank_layout_puts_producers_left_of_consumers() -> None:
    # Falsifier: rank by insertion order -- `final` sits before `b_comp` in a shuffled dict.
    shuffled = dict(reversed(list(_BLOOM.items())))
    laid = rank_layout(shuffled, list(_BLOOM), _BLOOM_GROUPS, _SIZE, {}, 40.0, 10.0)
    for consumer, row in _BLOOM.items():
        for producer in row.values():
            if producer != consumer:
                assert laid[producer][0] < laid[consumer][0], (producer, consumer)


def test_rank_layout_is_deterministic_over_dict_order() -> None:
    a = rank_layout(_BLOOM, list(_BLOOM), _BLOOM_GROUPS, _SIZE, {}, 40.0, 10.0)
    shuffled = dict(reversed(list(_BLOOM.items())))
    b = rank_layout(shuffled, list(_BLOOM), _BLOOM_GROUPS, _SIZE, {}, 40.0, 10.0)
    assert a == b


def test_rank_layout_places_only_the_names_asked_for() -> None:
    # Falsifier: return a position for every name, which would overwrite a drag.
    placed = {"scene": (5.0, 5.0), "final": (900.0, 5.0)}
    laid = rank_layout(
        _BLOOM, ["b_bright", "b_blur"], _BLOOM_GROUPS, _SIZE, placed, 40.0, 10.0
    )
    assert set(laid) == {"b_bright", "b_blur"}


def test_rank_layout_keeps_cycle_members() -> None:
    # Falsifier: lay out `plan.order` alone -- the two members vanish from the picture.
    wiring = {"a": {"u_b": "b"}, "b": {"u_a": "a"}, "c": {}}
    laid = rank_layout(
        wiring,
        ["a", "b", "c"],
        {},
        _SIZE | {"a": (1, 1), "b": (1, 1), "c": (1, 1)},
        {},
        4.0,
        4.0,
    )
    assert set(laid) == {"a", "b", "c"}
    assert graph_ranks(wiring) == {"a": 0, "b": 0, "c": 0}


def test_rank_layout_keeps_group_members_adjacent() -> None:
    # Rank 1 holds b_bright, b_trail (bloom) and an ungrouped `side`; the two members are
    # contiguous whatever strip order says. Falsifier: sort a column by strip order alone.
    wiring = dict(_BLOOM) | {"side": {"u_scene": "scene"}}
    sizes = _SIZE | {"side": (100.0, 120.0)}
    laid = rank_layout(wiring, list(wiring), _BLOOM_GROUPS, sizes, {}, 40.0, 10.0)
    column = sorted(
        (name for name in ("b_bright", "b_trail", "side")), key=lambda n: laid[n][1]
    )
    members = [column.index("b_bright"), column.index("b_trail")]
    assert abs(members[0] - members[1]) == 1


def test_node_ports_come_from_the_program_never_the_stored_rows() -> None:
    # Falsifier: build the list from `values` keys -- the dead `u_gone` row grows a port.
    values: dict[str, object] = {"u_gone": PassSource("scene"), "u_src": AutoSource()}
    ports = node_ports(["u_src"], values, {"u_src": "scene"}, "cons")
    assert [p.sampler for p in ports] == ["u_src"]


def test_node_ports_classify_every_state_and_put_feedback_last() -> None:
    values: dict[str, object] = {
        "u_prev": AutoSource(),
        "u_none": NoSource(),
        "u_tex": object(),  # a bound texture is any value that is no source kind
        "u_lost": PassSource("vanished"),
    }
    row = {"u_prev": "trail", "u_src": "scene"}
    ports = node_ports(
        ["u_prev", "u_src", "u_none", "u_tex", "u_lost"], values, row, "trail"
    )
    assert [(p.sampler, p.kind) for p in ports] == [
        ("u_src", "wired"),
        ("u_none", "none"),
        ("u_tex", "media"),
        ("u_lost", "unfilled"),
        ("u_prev", "prev"),
    ]
    assert ports[0].source == "scene" and ports[-1].source == "trail"


def test_group_boundary_over_the_bloom_shape() -> None:
    boundary = group_boundary(_BLOOM_MEMBERS, _ports_of(_BLOOM), _BLOOM, "final")
    assert [(b.member, b.label) for b in boundary.inputs] == [
        ("b_bright", "b_bright.u_scene"),
        ("b_trail", "b_trail.u_scene"),
        ("b_comp", "b_comp.u_scene"),
    ]
    assert boundary.outputs == ["b_comp"] and boundary.bundle == "b_comp"


def test_group_boundary_over_the_non_convex_shape() -> None:
    wiring = {
        "a": {},
        "g1": {"u_a": "a"},
        "mid": {"u_g1": "g1"},
        "g2": {"u_mid": "mid"},
        "out": {"u_g2": "g2"},
    }
    boundary = group_boundary(["g1", "g2"], _ports_of(wiring), wiring, "out")
    assert [(b.member, b.port.sampler) for b in boundary.inputs] == [
        ("g1", "u_a"),
        ("g2", "u_mid"),
    ]
    assert boundary.outputs == ["g1", "g2"]


def test_group_boundary_over_a_generator_box() -> None:
    wiring = {"gen": {}, "gen2": {"u_gen": "gen"}, "final": {"u_gen2": "gen2"}}
    boundary = group_boundary(["gen", "gen2"], _ports_of(wiring), wiring, "final")
    assert boundary.inputs == [] and boundary.outputs == ["gen2"]


def test_group_boundary_over_a_one_member_group() -> None:
    boundary = group_boundary(["b_comp"], _ports_of(_BLOOM), _BLOOM, "final")
    assert [b.label for b in boundary.inputs] == ["u_scene", "u_blur", "u_trail"]
    assert boundary.outputs == ["b_comp"]


def test_group_boundary_over_a_split_group() -> None:
    wiring = {"x": {}, "m": {"u_x": "x"}, "y": {"u_m": "m"}, "out": {"u_y": "y"}}
    boundary = group_boundary(["x", "y"], _ports_of(wiring), wiring, "out")
    assert boundary.outputs == ["x", "y"] and boundary.bundle == "x"


def test_group_boundary_over_a_member_whose_sampler_reads_nothing() -> None:
    # The severed chain (mutations case 1): b_comp's u_blur resolves to nothing after b_blur
    # goes. Falsifier: drop the "or unfilled" clause -- the port disappears from the box.
    wiring = {k: v for k, v in _BLOOM.items() if k != "b_blur"}
    wiring["b_comp"] = {"u_scene": "scene", "u_trail": "b_trail"}
    ports = _ports_of(wiring)
    ports["b_comp"] = node_ports(
        ["u_scene", "u_blur", "u_trail"], {}, wiring["b_comp"], "b_comp"
    )
    members = ["b_bright", "b_trail", "b_comp"]
    boundary = group_boundary(members, ports, wiring, "final")
    assert ("b_comp", "u_blur") in [(b.member, b.port.sampler) for b in boundary.inputs]


def test_a_terminal_box_still_has_its_bundle_output_port() -> None:
    # Nothing outside reads any member. Falsifier: make the bundle output conditional on an
    # outside reader -- the mutations review's case 2, a box with a picture and no dot.
    wiring = {k: v for k, v in _BLOOM.items() if k != "final"}
    boundary = group_boundary(_BLOOM_MEMBERS, _ports_of(wiring), wiring, "scene")
    assert boundary.outputs == ["b_comp"] and boundary.bundle == "b_comp"


def test_bundle_output_follows_its_four_branches_in_order() -> None:
    members = ["b_bright", "b_trail", "b_blur", "b_comp"]
    assert bundle_output(members, _BLOOM, "b_trail") == "b_trail"  # the document output
    assert bundle_output(members, _BLOOM, "final") == "b_comp"  # read from outside
    unread = {k: v for k, v in _BLOOM.items() if k != "final"}
    assert bundle_output(members, unread, "scene") == "b_comp"  # last unread by members
    looped = {"m1": {"u_m2": "m2"}, "m2": {"u_m1": "m1"}}
    assert bundle_output(["m1", "m2"], looped, "") == "m2"  # last member
    for wiring, output in ((_BLOOM, "final"), (unread, "scene"), (looped, "")):
        names = members if wiring is not looped else ["m1", "m2"]
        assert bundle_output(names, wiring, output) in names


def test_refuse_drop_refuses_every_cycle_and_allows_every_legal_drop() -> None:
    chain = {"a": {}, "b": {"u_a": "a"}, "c": {"u_b": "b"}, "loner": {}}
    refused = refuse_drop(chain, "a", "u_c", "c")
    assert "passes form a cycle" in refused and "a -> c -> b -> a" in refused
    assert refuse_drop(chain, "c", "u_a2", "a") == ""  # the diamond
    assert refuse_drop(chain, "a", "u_l", "loner") == ""  # unrelated
    assert refuse_drop(chain, "b", "u_prev", "b") == ""  # feedback
    # Falsifier: refuse only when the culprit is an endpoint. The planner names `a` as the
    # culprit for a drop whose endpoints are `c` and `a`, and would name a third pass on a
    # longer loop; the whole-list check is what makes every case refuse.
    longer = wiring_with(chain, "loner", "u_c", "c")
    assert refuse_drop(longer, "a", "u_loner", "loner")


def test_cycle_edges_are_the_pairs_of_the_culprit_message() -> None:
    wiring = {
        "paint": {"u_comp": "composite"},
        "seed": {"u_paint": "paint"},
        "composite": {"u_seed": "seed"},
        "df": {"u_paint": "paint"},
    }
    errors = plan_passes(wiring)[1]
    culprits = [e for e in errors if e.message.startswith("passes form a cycle")]
    assert len(culprits) == 1, "one message per cycle, the rest are victims"
    assert cycle_edges(errors) == {
        ("composite", "paint"),
        ("seed", "composite"),
        ("paint", "seed"),
    }
    assert cycle_edges([]) == set()


def test_group_name_error_covers_the_pattern_and_the_namespace() -> None:
    assert group_name_error("", {"a"}) == ""
    assert "group name" in group_name_error("2bad", {"a"})
    assert group_name_error("a", {"a", "b"}) == "a pass and a group cannot share a name"
    assert group_name_error("c", {"a", "b"}) == ""


@pytest.mark.parametrize(
    "position",
    [
        (float("nan"), 0.0),
        (float("inf"), 0.0),
        (0.0, float("-inf")),
        (1e30, 1e30),
        (-1e30, 0.0),
    ],
)
def test_a_position_is_bounded_on_the_model(position: tuple[float, float]) -> None:
    # Falsifier: a bare `tuple[float, float]`, which the persistence review measured accepting
    # NaN and 1e30.
    with pytest.raises(ValidationError):
        PassEntry(position=position)


def test_a_legal_position_is_accepted_and_carried_by_the_funnel() -> None:
    graph = PassGraph(passes={"a": PassEntry(iterations=3), "b": PassEntry()})
    placed = graph.with_positions({"a": (12.0, -34.5), "c": (1.0, 1.0)})
    assert placed.passes["a"].position == (12.0, -34.5)
    assert placed.passes["a"].iterations == 3
    assert placed.passes["c"].position == (1.0, 1.0)
    assert placed.with_group("a", "g").passes["a"].position == (12.0, -34.5)
    # Falsifier: `model_copy(update=...)`, which skips the field's bounds.
    with pytest.raises(ValidationError):
        graph.with_positions({"a": (1e30, 0.0)})
