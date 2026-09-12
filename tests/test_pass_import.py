"""The import plan (091 D4): what copying another document's passes into this one changes,
decided over two wirings and nothing else.

GL-free. Every case is a hand-built `Wiring`; the falsifiers are the bugs an implementer
would write -- above all skipping the materialization, which leaves a bundle whose every
inter-pass edge is gone while only its feedback survives, silently.
"""

from collections.abc import Collection, Mapping

from shaderbox.pass_graph import Wiring
from shaderbox.pass_import import ImportPlan, plan_import

# The bloom-chain fixture's wiring, as `effective_wiring` answers it once compiled.
_BLOOM = {
    "scene": {},
    "bright": {"u_scene": "scene"},
    "blur": {"u_bright": "bright"},
    "trail": {"u_scene": "scene", "u_prev": "trail"},
    "composite": {"u_scene": "scene", "u_blur": "blur", "u_trail": "trail"},
}
_HOST = {"main": {}, "grade": {"u_main": "main"}, "final": {"u_grade": "grade"}}


def _plan(
    source_wiring: Wiring = _BLOOM,
    source_output: str = "composite",
    group: str = "bloom",
    substitutions: Mapping[str, str] | None = None,
    handovers: Collection[tuple[str, str]] = (),
    host_wiring: Wiring = _HOST,
    host_output: str = "final",
    host_groups: Collection[str] = (),
) -> ImportPlan | str:
    return plan_import(
        source_wiring,
        source_output,
        group,
        {"scene": "main"} if substitutions is None else substitutions,
        handovers,
        host_wiring,
        host_output,
        host_groups,
    )


def test_every_wired_sampler_is_explicit_under_the_new_names() -> None:
    # (a). Falsifier: skip the materialization and `bloom_composite.u_blur` has no row, so the
    # name rule looks for a pass called `blur` in the host and reads black.
    plan = _plan()
    assert isinstance(plan, ImportPlan), plan
    assert plan.renames == {
        "bright": "bloom_bright",
        "blur": "bloom_blur",
        "composite": "bloom_composite",
        "trail": "bloom_trail",
    }
    assert plan.sources["bloom_blur"] == {"u_bright": "bloom_bright"}
    assert plan.sources["bloom_composite"] == {
        "u_scene": "main",
        "u_blur": "bloom_blur",
        "u_trail": "bloom_trail",
    }
    # The self-read is written too, as the renamed self: it resolves by name today and would
    # break the day the group is renamed.
    assert plan.sources["bloom_trail"] == {"u_scene": "main", "u_prev": "bloom_trail"}
    assert plan.output == "bloom_composite"


def test_a_fed_entry_point_is_not_copied_and_its_readers_point_at_the_host() -> None:
    # (b).
    plan = _plan()
    assert isinstance(plan, ImportPlan)
    assert "scene" not in plan.renames
    assert plan.sources["bloom_bright"] == {"u_scene": "main"}


def test_a_colliding_name_rejects_and_names_it() -> None:
    # (c). `grade` already exists on the host.
    plan = _plan(group="", host_wiring={"main": {}, "blur": {}})
    assert plan == "'blur' already exists"


def test_replacing_the_only_pass_rejects() -> None:
    # (d) -- a single-pass source whose one pass is both entry point and output.
    plan = _plan(
        source_wiring={"main": {}},
        source_output="main",
        substitutions={"main": "main"},
    )
    assert isinstance(plan, str) and "output" in plan


def test_an_empty_group_copies_under_bare_names() -> None:
    # (e).
    plan = _plan(group="", host_wiring={"main": {}})
    assert isinstance(plan, ImportPlan)
    assert plan.renames["composite"] == "composite"


def test_a_handover_rewires_that_host_sampler_and_leaves_the_rest() -> None:
    # (f). Falsifier: hand every reader of the fed pass over regardless of the set.
    plan = _plan(handovers={("grade", "u_main")})
    assert isinstance(plan, ImportPlan)
    assert plan.handovers == {"grade": {"u_main": "bloom_composite"}}
    unchecked = _plan(handovers=set())
    assert isinstance(unchecked, ImportPlan) and unchecked.handovers == {}
    assert "does not read" in str(_plan(handovers={("final", "u_grade")}))


def test_the_output_moves_only_when_the_fed_pass_was_the_output() -> None:
    # (g). Falsifier: move it unconditionally.
    stays = _plan()
    assert isinstance(stays, ImportPlan) and stays.becomes_output is False
    moves = _plan(host_wiring={"main": {}}, host_output="main")
    assert isinstance(moves, ImportPlan) and moves.becomes_output is True


def test_a_source_pass_with_no_edges_still_plans() -> None:
    # (h). A broken source pass answers an empty wiring; the plan copies it as is.
    wiring = {**_BLOOM, "blur": {}}
    plan = _plan(source_wiring=wiring)
    assert isinstance(plan, ImportPlan)
    assert "blur" in plan.renames and "bloom_blur" not in plan.sources


def test_a_handover_onto_a_feeding_pass_is_a_loop_and_rejects() -> None:
    # Two entry points fed by two host passes where one reads the other: the default handover
    # would make `b` read the bundle that reads `b`. Falsifier: skip the post-import plan and
    # the loop lands silently (the renderer falls back to drawing the output alone).
    compositor = {"fg": {}, "bg": {}, "mix": {"u_fg": "fg", "u_bg": "bg"}}
    host = {"a": {}, "b": {"u_a": "a"}}
    looped = _plan(
        source_wiring=compositor,
        source_output="mix",
        substitutions={"fg": "a", "bg": "b"},
        handovers={("b", "u_a")},
        host_wiring=host,
        host_output="b",
    )
    assert isinstance(looped, str) and "loop" in looped, looped
    unchecked = _plan(
        source_wiring=compositor,
        source_output="mix",
        substitutions={"fg": "a", "bg": "b"},
        handovers=set(),
        host_wiring=host,
        host_output="b",
    )
    assert isinstance(unchecked, ImportPlan)


def test_rejections_name_what_was_wrong() -> None:
    assert "group name" in str(_plan(group="2bad"))
    assert "not an entry point" in str(_plan(substitutions={"blur": "main"}))
    assert "no such pass" in str(_plan(substitutions={"scene": "nope"}))
