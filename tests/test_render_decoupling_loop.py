"""The throttle and the Auto sizing as the LOOP runs them (090 D2, D6, D9, D10).

Every check here is a wiring check. `plan_render_set`, `fit_to_aspect` and `apply_damping`
are pure and tested on their own in `test_render_plan.py`; what cannot be seen there is whether
anything PRODUCES their inputs and whether anything READS their answers. A plan computed and
never consulted, a recorder nobody calls, a chip handed `None` unconditionally -- each passes
every pure test and ships a feature that does nothing.

**The rig advances `app.frame_idx` itself.** The increment is the LAST statement of
`_update_and_draw`, not of `_tick_frame_state`, so a loop calling the latter twelve times runs
twelve iterations at one index: the `(frame_idx + phase) % k` gate takes the same branch every
time and a `k = 12` document renders either 12 times or 0. Moving the increment into
`_tick_frame_state` is NOT the fix -- `session.tick` and `begin_frame` both read it from inside.
"""

from typing import Any

import pytest

from shaderbox import ui
from shaderbox.app import PopupState
from shaderbox.constants import STARTER_EXAMPLE_ID
from shaderbox.profiling import FrameProfile, Span
from shaderbox.render_plan import AUTO_RESIZE_STABLE_FRAMES, CostRecord, RenderPlan
from shaderbox.render_shape import ResolutionMode, fit_to_aspect
from shaderbox.tabs.document import _apply_canvas_size, _switch_resolution_mode
from shaderbox.theme import SIZE
from shaderbox.ui import _tick_frame_state, update_and_draw
from shaderbox.ui_primitives import profile_rows_plan
from tests.conftest import seed_extra_document

# This module drives real frames, so it gets its own worker: the imgui font atlas is per
# PROCESS and a second App that renders a full frame in one interpreter dies on a texture the
# first released (`conventions.md ## Known quirks`).
pytestmark = pytest.mark.xdist_group("gl_frames_render_decoupling")


def _drive(app: Any, n: int) -> None:
    for _ in range(n):
        _tick_frame_state(app)
        app.frame_idx += 1


def _plant_cost(app: Any, document_id: str, gpu_ms: float) -> None:
    app.document_costs[document_id] = CostRecord(gpu_ms=gpu_ms, cpu_ms=0.0)


def _freeze_costs(app: Any, monkeypatch: Any) -> None:
    """Stop step 3 overwriting a planted cost with the profile's own (zero) numbers."""

    monkeypatch.setattr(ui, "_refresh_document_costs", lambda _app: None)


def _count_renders(
    app: Any, monkeypatch: Any, documents: dict[str, Any]
) -> dict[str, int]:
    """How many times each document's `render` is actually called.

    The render gate and step 8's `begin_frame` gate are two separate reads of the plan, and
    `Document._frame` can only see the second. A plan honored by the feedback advance alone
    would still submit every document's GPU work every frame, which is the whole cost the
    throttle exists to spread.
    """
    counts: dict[str, int] = {}
    by_object = {
        id(ui_document.document): document_id
        for document_id, ui_document in documents.items()
    }
    sample = next(iter(documents.values())).document
    real = type(sample).render

    def counted(self: Any, *args: Any, **kwargs: Any) -> None:
        name = by_object.get(id(self))
        if name is not None:
            counts[name] = counts.get(name, 0) + 1
        real(self, *args, **kwargs)

    monkeypatch.setattr(type(sample), "render", counted)
    return counts


def _count_resizes(app: Any, monkeypatch: Any, document: Any) -> list[tuple[int, int]]:
    """Every size `set_canvas_size` is actually called with, in order."""
    calls: list[tuple[int, int]] = []
    real = type(document).set_canvas_size

    def counted(self: Any, size: tuple[int, int]) -> None:
        calls.append(size)
        real(self, size)

    monkeypatch.setattr(type(document), "set_canvas_size", counted)
    return calls


def _count_begin_frames(app: Any, monkeypatch: Any) -> dict[str, int]:
    """How many times each document's `begin_frame` is actually called.

    `Document._frame` cannot answer this: `begin_frame(frame)` ASSIGNS the frame number, so its
    delta measures the index span the drive covered rather than the number of advances -- a
    throttled document reads the same as an unthrottled one.
    """
    counts: dict[str, int] = {}
    documents = {
        id(ui_document.document): document_id
        for document_id, ui_document in list(app.ui_documents.items())
        + list(app.ui_document_examples.items())
    }
    real = type(app.ui_documents[app.current_document_id].document).begin_frame

    def counted(self: Any, frame: int | None = None) -> None:
        name = documents.get(id(self))
        if name is not None:
            counts[name] = counts.get(name, 0) + 1
        real(self, frame)

    monkeypatch.setattr(
        type(app.ui_documents[app.current_document_id].document), "begin_frame", counted
    )
    return counts


# ---------------------------------------------------------------------------
# V7 / V8 -- the throttle is wired, and off it restores today's set
# ---------------------------------------------------------------------------


def test_an_expensive_current_document_advances_once_in_twelve_frames(
    app: Any, monkeypatch: Any
) -> None:
    # The wiring check: the mutation that finds this is applied at the LOOP, not at
    # `plan_render_set` -- compute the plan and never read it at the render site and every
    # document still advances every frame while the intervals look perfect.
    other = seed_extra_document(app, "cheap-0000-4000-8000-000000000001")
    _freeze_costs(app, monkeypatch)
    _plant_cost(app, app.current_document_id, 100.0)
    _plant_cost(app, other, 0.5)
    # Converge the hysteresis before measuring: an interval only moves after four agreeing
    # frames, so the first frames of any drive are at k = 1 by design.
    _drive(app, 8)

    counts = _count_begin_frames(app, monkeypatch)
    _drive(app, 12)
    assert app.render_plan is not None
    assert app.render_plan.intervals[app.current_document_id] == 12
    advanced = counts.get(app.current_document_id, 0)
    assert advanced <= 2, f"a k = 12 document advanced {advanced} times in 12 frames"
    assert advanced >= 1, "a throttled document stopped advancing altogether"
    assert counts.get(other, 0) == 12, "the cheap document was throttled too"

    # ... and the RENDER site reads the same answer. Dropping only the render-site gate leaves
    # `_frame` correct while every frame still submits the document's whole GPU cost, which is
    # the bug the throttle exists to prevent and the one the feedback counter cannot see.
    renders = _count_renders(app, monkeypatch, app.ui_documents)

    for _ in range(12):
        update_and_draw(app)
    assert renders.get(app.current_document_id, 0) <= 2, (
        f"a k = 12 document rendered {renders.get(app.current_document_id, 0)} times in 12 "
        "frames -- the render site is not reading the plan"
    )


def test_the_throttle_off_advances_every_document_every_frame(
    app: Any, monkeypatch: Any
) -> None:
    # Falsifier: ignore `is_throttle_documents` and the 100 ms document advances once, not
    # twelve times. Without the `frame_idx` bump in `_drive` this would pass vacuously -- it
    # would be green even with the throttle wired wrong.
    app.app_state.is_throttle_documents = False
    _freeze_costs(app, monkeypatch)
    _plant_cost(app, app.current_document_id, 100.0)
    _drive(app, 8)

    counts = _count_begin_frames(app, monkeypatch)
    _drive(app, 12)
    assert counts.get(app.current_document_id, 0) == 12
    assert app.render_plan is not None
    assert set(app.render_plan.intervals.values()) == {1}


def test_a_throttled_document_behind_the_pass_settings_modal_is_gated_too(
    app: Any, monkeypatch: Any
) -> None:
    # The THIRD render branch. The pass-settings modal keeps the current document rendering
    # behind it, and that branch read no interval while step 8 gated the same document's
    # `begin_frame` -- so the throttle was inert exactly where a user sits adjusting an
    # expensive graph, and its feedback pass took many renders per integration step. Falsifier:
    # remove the `renders_this_frame` guard from the PASS_SETTINGS branch and this counts 24.

    _freeze_costs(app, monkeypatch)
    _plant_cost(app, app.current_document_id, 100.0)
    for _ in range(8):
        app.popup_state = PopupState.PASS_SETTINGS
        update_and_draw(app)

    renders = _count_renders(app, monkeypatch, app.ui_documents)
    begins = _count_begin_frames(app, monkeypatch)
    try:
        for _ in range(24):
            app.popup_state = PopupState.PASS_SETTINGS
            update_and_draw(app)
    finally:
        app.popup_state = PopupState.CLOSED

    assert app.render_plan is not None
    interval = app.render_plan.intervals[app.current_document_id]
    assert interval == 12
    drawn = renders.get(app.current_document_id, 0)
    assert drawn <= 3, (
        f"a k = {interval} document rendered {drawn} times in 24 frames behind the "
        "pass-settings modal -- that branch is not reading the plan"
    )
    # The two reads agree: the render gate and the feedback advance admit the same frames.
    assert abs(drawn - begins.get(app.current_document_id, 0)) <= 1, (
        f"{drawn} renders against {begins.get(app.current_document_id, 0)} feedback "
        "advances -- the two gates disagree"
    )


# ---------------------------------------------------------------------------
# V3a -- the loop applies the damping, not the raw request
# ---------------------------------------------------------------------------


def test_the_loop_damps_a_drag_ramp_instead_of_resizing_every_frame(
    app: Any, monkeypatch: Any
) -> None:
    # `apply_damping` being correct says nothing about whether the loop calls it. Falsifier:
    # pass the raw request to `set_canvas_size` and this counts ~60 resizes instead of a handful.
    document_id = app.current_document_id
    document = app.ui_documents[document_id].document
    document.resolution_mode = ResolutionMode.AUTO
    document.aspect = (4, 3)

    calls = _count_resizes(app, monkeypatch, document)
    for width in range(764, 941, 3):
        app.viewer_region = (float(width), width * 3 / 4)
        _drive(app, 1)
    # One more than the pure ramp's bound: the document starts at its loaded size, so the first
    # request is a jump rather than a step.
    assert 0 < len(calls) <= 5, f"{len(calls)} resizes across the ramp: {calls}"


@pytest.mark.parametrize(
    ("region", "label"),
    [
        ((5000, 3750), "past MAX_CANVAS_PX"),
        ((12, 9), "under MIN_CANVAS_PX"),
        ((40, 30), "a small tile, in bounds"),
    ],
)
def test_a_stationary_region_settles_whatever_the_canvas_bounds_do(
    app: Any, monkeypatch: Any, region: tuple[int, int], label: str
) -> None:
    # The damping compares its request against `canvas_size`, which `set_canvas_size` CLAMPS.
    # A region resolving outside 16..4096 is therefore a size the canvas can never equal, so
    # every frame read as "past the dead band" and re-entered the resize funnel forever, with
    # the damping state permanently disarmed. Falsifier: drop the `clamped_size` call in
    # `_resolve_resolutions` and the two out-of-bounds cases report 30 calls instead of 1.
    document = app.ui_documents[app.current_document_id].document
    document.resolution_mode = ResolutionMode.AUTO
    document.aspect = (4, 3)

    calls = _count_resizes(app, monkeypatch, document)
    for _ in range(30):
        app.viewer_region = (float(region[0]), float(region[1]))
        _drive(app, 1)
    assert len(calls) == 1, (
        f"{label}: a stationary region produced {len(calls)} resizes over 30 frames"
    )


# ---------------------------------------------------------------------------
# V9 (revision 1) -- the viewer region is the ONE size source
# ---------------------------------------------------------------------------


def test_the_viewer_records_the_region_it_drew_the_document_image_at(app: Any) -> None:
    # The one recorder left, and cutting it is SILENT: `_resolve_resolutions` skips Auto
    # documents while the region is None, so every document simply keeps its loaded size and
    # nothing else fails. Falsifier: delete the `app.viewer_region = ...` assignment in
    # `_draw_document_image` and this stays None.

    app.ui_documents[
        app.current_document_id
    ].document.resolution_mode = ResolutionMode.AUTO
    app.viewer_region = None
    update_and_draw(app)
    assert app.viewer_region is not None, "the viewer drew and recorded no region"
    width, height = app.viewer_region
    # The VIEWER's own region, not a thumbnail's: the whole point of revision 1 is that the
    # size comes from the big surface, so a tile-sized answer means the wrong site recorded.
    assert max(width, height) > float(SIZE.THUMB_LG), (
        f"the recorded region {width}x{height} is a thumbnail's, not the viewer's"
    )


def test_every_auto_document_renders_at_the_viewer_region_not_only_the_current_one(
    app: Any,
) -> None:
    # Revision 1's core: ONE size source. Before it, a non-current document was sized by
    # whichever surface happened to draw it, so a document nothing had drawn yet stayed at its
    # loaded size -- which is how a new document rendered 64x64 and square. Falsifier: size
    # only `app.current_document_id` in `_resolve_resolutions` and the sibling keeps its own.
    other = seed_extra_document(app, "sibling-0000-4000-8000-000000000004")
    for document_id in (app.current_document_id, other):
        document = app.ui_documents[document_id].document
        document.resolution_mode = ResolutionMode.AUTO
        document.aspect = (16, 9)

    app.viewer_region = (640.0, 480.0)
    _drive(app, AUTO_RESIZE_STABLE_FRAMES + 2)

    expected = fit_to_aspect((640.0, 480.0), (16, 9))
    for document_id in (app.current_document_id, other):
        assert app.ui_documents[document_id].document.canvas_size == expected, (
            f"{document_id} rendered at "
            f"{app.ui_documents[document_id].document.canvas_size}, not the viewer's "
            f"{expected}"
        )


@pytest.mark.parametrize(
    ("region", "aspect", "expected"),
    [
        # A viewer WIDER than the aspect: the height runs out first and the width follows it.
        ((1000.0, 400.0), (16, 9), (711, 400)),
        # A viewer TALLER than the aspect: the width runs out first.
        ((400.0, 1000.0), (16, 9), (400, 225)),
        # Square into a wide viewer, and a tall aspect into a wide one.
        ((1000.0, 400.0), (1, 1), (400, 400)),
        ((1000.0, 400.0), (9, 16), (225, 400)),
    ],
)
def test_an_auto_document_fits_its_aspect_inside_the_viewer(
    app: Any,
    region: tuple[float, float],
    aspect: tuple[int, int],
    expected: tuple[int, int],
) -> None:
    # The fit is the whole of Auto sizing, and it must hold on BOTH sides of the aspect --
    # a rule that only handles the wider case letterboxes correctly and crops the other way.
    # Falsifier: compare the region's ratio the wrong way round in `fit_to_aspect` and the
    # wider and taller cases swap answers.
    document = app.ui_documents[app.current_document_id].document
    document.resolution_mode = ResolutionMode.AUTO
    document.aspect = aspect
    app.viewer_region = region
    _drive(app, AUTO_RESIZE_STABLE_FRAMES + 2)
    assert document.canvas_size == expected


def test_an_auto_document_keeps_its_size_until_the_viewer_has_drawn(app: Any) -> None:
    # Before the first frame nothing knows the region, and a document must not resize to a
    # guess. Falsifier: treat a None region as (0, 0) and every document collapses to the
    # minimum canvas on frame one.
    document = app.ui_documents[app.current_document_id].document
    document.resolution_mode = ResolutionMode.AUTO
    before = document.canvas_size
    app.viewer_region = None
    _drive(app, 3)
    assert document.canvas_size == before


# V9a -- a throttled example is actually skipped in the popup's set
# ---------------------------------------------------------------------------


def test_a_throttled_example_renders_less_often_than_a_cheap_one(
    app: Any, monkeypatch: Any
) -> None:
    # The popup's set is an ALTERNATIVE to `tick_documents`, so an interval computed for an
    # example id has to be read by the EXAMPLES branch or it is read by nothing at all.
    # Falsifier: render the examples loop without the `(frame_idx + phase) % k` gate and both
    # examples advance every frame while the plan's intervals look perfect.

    ids = list(app.ui_document_examples)[:2]
    if len(ids) < 2:
        pytest.skip("needs two shipped examples")
    expensive, cheap = ids
    _freeze_costs(app, monkeypatch)
    app.popup_state = PopupState.EXAMPLES
    app.app_state.selected_example_id = ""
    _plant_cost(app, expensive, 100.0)
    _plant_cost(app, cheap, 0.5)
    # Warm both examples past their first render, which the popup admits one per frame.
    for _ in range(8):
        app.popup_state = PopupState.EXAMPLES
        update_and_draw(app)

    # The RENDER calls, not `_frame`: step 8's `begin_frame` gate and the render gate are two
    # separate reads of the plan, and `_frame` can only see the first. Counting renders is what
    # makes "the examples loop ignores the plan" a red test.
    renders = _count_renders(app, monkeypatch, app.ui_document_examples)
    try:
        for _ in range(24):
            app.popup_state = PopupState.EXAMPLES
            update_and_draw(app)
    finally:
        app.popup_state = PopupState.CLOSED

    assert app.render_plan is not None
    assert app.render_plan.intervals[expensive] > app.render_plan.intervals[cheap]
    assert renders.get(cheap, 0) >= 20, (
        f"the cheap example only rendered {renders.get(cheap, 0)} times in 24 frames"
    )
    assert renders.get(expensive, 0) < renders.get(cheap, 0), (
        f"the throttled example rendered {renders.get(expensive, 0)} times against the cheap "
        f"one's {renders.get(cheap, 0)} -- the popup's render loop is not reading the plan"
    )


# ---------------------------------------------------------------------------
# V10 / V10a -- the chip and the panel rows
# ---------------------------------------------------------------------------


def test_the_chip_carries_the_current_documents_own_rate(
    app: Any, monkeypatch: Any
) -> None:
    # Falsifier: pass `document_fps=None` unconditionally at the `fps_overlay` call -- every
    # prose gate still passes and the chip silently never shows the second number.

    _freeze_costs(app, monkeypatch)
    _plant_cost(app, app.current_document_id, 100.0)
    _drive(app, 8)

    captured: list[Any] = []
    real = ui.fps_overlay

    def spy(**kwargs: Any) -> Any:
        captured.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(ui, "fps_overlay", spy)
    ui.update_and_draw(app)
    assert captured, "the overlay was never drawn"
    document_fps = captured[-1]["document_fps"]
    assert document_fps is not None, "the chip was handed no document rate"
    interval = app.render_plan.intervals[app.current_document_id]
    assert interval > 1
    assert document_fps == round(app.app_state.global_target_fps / interval)


def test_the_panel_rows_match_documents_by_id_not_by_title(app: Any) -> None:
    # Two documents sharing a title are two rows with their OWN numbers. Falsifier: match a
    # `document:` span through its title and one row takes the other's interval.

    root = Span("frame", cpu_ms=20.0)
    root.children.append(Span("document:aaa", cpu_ms=40.0, gpu_ms=40.0))
    root.children.append(Span("document:bbb", cpu_ms=4.0, gpu_ms=4.0))
    plan = RenderPlan(
        intervals={"aaa": 5, "bbb": 1},
        phases={"aaa": 0, "bbb": 1},
        document_fps={"aaa": 12.0, "bbb": 60.0},
    )
    rows = profile_rows_plan(
        FrameProfile(root, 0, complete=True),
        fps=60,
        target_fps=60,
        plan=plan,
        titles={"aaa": "Twin", "bbb": "Twin"},
        budget=0.5,
    )
    twins = [row for row in rows if row.name == "Twin"]
    assert len(twins) == 2, [row.name for row in rows]
    # The throttled one reads as a rate and an interval; its unthrottled namesake keeps today's
    # millisecond number, so the two rows cannot have been filled from one lookup.
    assert "x5" in twins[0].number
    assert twins[0].tooltip == "40.00 ms"
    assert twins[1].number.endswith("ms")


# ---------------------------------------------------------------------------
# V11 -- GPU spans record with the panel closed
# ---------------------------------------------------------------------------


def test_gpu_spans_record_while_the_panel_is_closed(app: Any) -> None:
    # The throttle's cost input is the profiler, so recording cannot follow the panel (D9a).
    # Falsifier: restore `app.profiler.enabled = app.fps_details_open` and `last_profile`
    # carries no `document:` child with a GPU number.

    app.fps_details_open = False
    for _ in range(4):
        update_and_draw(app)
    assert app.profiler.enabled
    profile = app.last_profile
    assert profile is not None
    documents = [s for s in profile.children if s.name.startswith("document:")]
    assert documents, "no document span was recorded with the panel closed"
    passes = [s for s in documents[0].children if s.name.startswith("pass:")]
    assert passes and passes[0].gpu_ms is not None


def test_the_costs_refresh_from_the_profile(app: Any) -> None:
    # Step 3 is the ONE write site (D7), and it is what turns a recorded span into the plan's
    # input. Falsifier: drop the refresh and `document_costs` stays empty forever, so every
    # document plans at k = 1 whatever it costs.

    app.document_costs.clear()
    for _ in range(6):
        update_and_draw(app)
    assert app.current_document_id in app.document_costs, (
        "four frames of recording produced no cost record"
    )


# ---------------------------------------------------------------------------
# V13 -- the copilot's explicit size sticks across a frame
# ---------------------------------------------------------------------------


def test_the_copilots_canvas_size_survives_the_next_frame(app: Any) -> None:
    # An explicit pixel request switches the document to Fixed (D5). Falsifier: leave the mode
    # Auto and the next frame's resolution step resizes it back to the viewer's region -- the
    # tool silently does nothing while every one of its own assertions passes.
    document_id = app.current_document_id
    result = app.copilot_backend.set_canvas_size(document_id, 128, 200)
    assert result.ok and (result.width, result.height) == (128, 200)
    document = app.ui_documents[document_id].document
    assert document.resolution_mode is ResolutionMode.FIXED
    assert document.resolution == (128, 200)

    app.viewer_region = (900.0, 700.0)  # a viewer asking for something else
    _drive(app, 1)
    assert document.canvas_size == (128, 200), (
        "the next frame overwrote the size the copilot set"
    )
    assert document.render_pass.canvas.texture.size == (128, 200)


def test_a_pickers_commit_applies_on_the_next_tick_not_inside_the_draw(
    app: Any,
) -> None:
    # The R2 hazard: `_apply_canvas_size` runs INSIDE the draw phase, after the viewer pushed
    # the output texture into this frame's draw list, so a resize there releases textures imgui
    # is still holding -- now the output canvas AND every feedback history. The commit therefore
    # parks and step 4 applies it. Falsifier: resize in place at the picker and this test's
    # first assertion (the canvas has NOT moved yet) goes red.

    document_id = app.current_document_id
    ui_document = app.ui_documents[document_id]
    ui_document.document.resolution_mode = ResolutionMode.FIXED
    before = ui_document.document.canvas_size

    _apply_canvas_size(app, ui_document, (256, 144))
    assert ui_document.document.canvas_size == before, (
        "the picker resized the canvas inside the draw phase"
    )
    assert app.pending_resolution[document_id] == (256, 144)

    _drive(app, 1)
    assert ui_document.document.canvas_size == (256, 144), (
        "the next tick did not apply the parked commit"
    )
    assert app.pending_resolution == {}, "the commit was applied and not consumed"


def test_closing_a_document_forgets_its_ephemeral_render_state(app: Any) -> None:
    # Every per-document dict the feature keeps is keyed by id and would otherwise hold an
    # entry for a document nothing can render, including a cost the plan would still read.
    # Falsifier: drop the `forget_render_state` call from `_on_document_deleted` and the four
    # entries survive the delete.
    other = seed_extra_document(app, "closeme-0000-4000-8000-000000000003")
    # A region, or no Auto document is ever sized and  stays empty.
    app.viewer_region = (800.0, 600.0)
    _drive(app, 6)
    app.pending_resolution[other] = (200, 150)
    _plant_cost(app, other, 4.0)
    assert other in app.throttle_states
    assert other in app.auto_size_states

    app.delete_document(other)

    for name, holder in (
        ("throttle_states", app.throttle_states),
        ("auto_size_states", app.auto_size_states),
        ("document_costs", app.document_costs),
        ("pending_resolution", app.pending_resolution),
    ):
        assert other not in holder, f"{name} still holds the closed document"


def test_switching_to_fixed_seeds_the_pair_from_the_live_canvas(app: Any) -> None:
    # A mode switch must never jump the picture (revision 1 D3). Auto -> Fixed keeps the size
    # the document is rendering at RIGHT NOW, not the pair it happened to be saved with.
    # Falsifier: seed `resolution` from anything else and the canvas jumps on the switch.

    ui_document = app.ui_documents[app.current_document_id]
    document = ui_document.document
    document.resolution_mode = ResolutionMode.AUTO
    document.aspect = (16, 9)
    document.resolution = (111, 222)  # a stale pair from some earlier life
    app.viewer_region = (640.0, 480.0)
    _drive(app, AUTO_RESIZE_STABLE_FRAMES + 2)
    live = document.canvas_size

    _switch_resolution_mode(app, ui_document, ResolutionMode.FIXED)
    assert document.resolution == live
    assert ui_document.ui_state.resolution == live
    _drive(app, 2)
    assert document.canvas_size == live, "the switch to Fixed moved the picture"


def test_switching_to_auto_seeds_the_aspect_from_the_live_canvas(app: Any) -> None:
    # The mirror: Fixed -> Auto keeps the SHAPE, so only the sizing rule changes. Falsifier:
    # leave `aspect` alone and the document snaps to whatever shape it last had under Auto.

    ui_document = app.ui_documents[app.current_document_id]
    document = ui_document.document
    document.resolution_mode = ResolutionMode.FIXED
    document.aspect = (1, 1)  # a stale ratio from some earlier life
    document.resolution = (1280, 720)
    _drive(app, 2)
    assert document.canvas_size == (1280, 720)

    _switch_resolution_mode(app, ui_document, ResolutionMode.AUTO)
    assert document.aspect == (16, 9)
    assert ui_document.ui_state.aspect == (16, 9)
    # ... and the next frames render at that shape in the viewer, not at the stale square.
    app.viewer_region = (800.0, 800.0)
    _drive(app, AUTO_RESIZE_STABLE_FRAMES + 2)
    assert document.canvas_size == fit_to_aspect((800.0, 800.0), (16, 9))


def test_a_new_document_opens_wide_and_sizes_itself_to_the_viewer(app: Any) -> None:
    # The bug this revision exists for: a new document showed 64x64 and rendered SQUARE. A new
    # document is a copy of the starter example, so its arrival shape is the starter's; nothing
    # is assigned here. Falsifier: give the starter a square aspect, or skip non-current
    # documents in `_resolve_resolutions`, and this goes red.
    before = set(app.ui_documents)
    app.create_document_from_example(STARTER_EXAMPLE_ID)
    (document_id,) = set(app.ui_documents) - before
    document = app.ui_documents[document_id].document
    assert document.resolution_mode is ResolutionMode.AUTO
    assert document.aspect == (16, 9)

    app.viewer_region = (1000.0, 400.0)
    _drive(app, AUTO_RESIZE_STABLE_FRAMES + 2)
    assert document.canvas_size == fit_to_aspect((1000.0, 400.0), (16, 9))
    assert document.canvas_size[0] != document.canvas_size[1], (
        "a fresh Auto document is still rendering square"
    )


def test_the_starter_document_is_the_fixture_it_claims_to_be(app: Any) -> None:
    # A guard on the rig itself: every test above drives the shipped starter, and the shape of
    # its persisted state is what the loop reads.
    document = app.ui_documents[STARTER_EXAMPLE_ID].document
    assert document.resolution == (1280, 720)
    assert document.aspect == (16, 9)
    assert document.resolution_mode is ResolutionMode.AUTO
