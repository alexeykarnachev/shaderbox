"""The throttle and the Auto sizing as the LOOP runs them (090 D2, D6, D9, D10).

Every check here is a wiring check. `plan_render_set`, `auto_canvas_size` and `apply_damping`
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
from imgui_bundle import imgui

from shaderbox.constants import STARTER_EXAMPLE_ID
from shaderbox.render_plan import CostRecord
from shaderbox.render_shape import ResolutionMode
from shaderbox.ui import _tick_frame_state
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
    from shaderbox import ui

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
    from shaderbox.ui import update_and_draw

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
    document.resolution = (800, 600)

    calls: list[tuple[int, int]] = []
    real = type(document).set_canvas_size

    def counted(self: Any, size: tuple[int, int]) -> None:
        calls.append(size)
        real(self, size)

    monkeypatch.setattr(type(document), "set_canvas_size", counted)
    for width in range(764, 941, 3):
        app.displayed_sizes[document_id] = (width, round(width * 3 / 4))
        _drive(app, 1)
    # One more than the pure ramp's bound: the document starts at its loaded 1280x960, so the
    # first request is a jump rather than a step.
    assert 0 < len(calls) <= 5, f"{len(calls)} resizes across the ramp: {calls}"


# ---------------------------------------------------------------------------
# V9 -- the recorders are wired
# ---------------------------------------------------------------------------


def test_the_viewer_records_the_size_it_drew_the_current_document_at(
    app: Any,
) -> None:
    # Cutting a recorder is SILENT: `auto_canvas_size(None, ...)` answers `previous`, so the
    # document keeps its loaded size and every other test still passes. Falsifier: delete the
    # `record_displayed_size` call in `_draw_document_image` and `displayed_sizes` stays empty.
    from shaderbox.theme import SIZE
    from shaderbox.ui import update_and_draw

    document_id = app.current_document_id
    app.ui_documents[document_id].document.resolution_mode = ResolutionMode.AUTO
    app.displayed_sizes.clear()
    update_and_draw(app)
    assert document_id in app.displayed_sizes, (
        "the viewer drew the current document and recorded nothing"
    )
    # The SIZE is the discriminator, not the key: the grid tile below the viewer records the
    # current document too, so an id alone is there whether or not the viewer recorded. The
    # viewer is by far the largest region showing it, and the largest wins.
    width, height = app.displayed_sizes[document_id]
    assert max(width, height) > float(SIZE.THUMB_LG), (
        f"the recorded size {width}x{height} is a thumbnail's, not the viewer's -- the "
        "viewer's own recorder is not wired"
    )


def test_the_document_grid_records_each_tile_it_draws(app: Any) -> None:
    # The grid's own recorder, driven in a headless imgui frame the way `test_canvas_fields.py`
    # drives the Document tab. Falsifier: delete the grid's `record_displayed_size` and the
    # second document -- which the viewer never draws -- has no entry.
    from shaderbox.widgets.document_grid import draw_document_preview_grid

    other = seed_extra_document(app, "gridrec-0000-4000-8000-000000000002")
    app.displayed_sizes.clear()
    imgui.new_frame()
    imgui.begin("rig")
    draw_document_preview_grid(app, 400.0, 400.0)
    imgui.end()
    imgui.end_frame()
    assert other in app.displayed_sizes, "the grid drew a tile and recorded nothing"


def test_the_examples_popup_records_its_own_thumbnails(app: Any) -> None:
    # While the popup is open its examples ARE the displayed set (D10), so its thumbnails are
    # the only recorders those documents have. Falsifier: delete the popup's
    # `record_displayed_size` and no example id appears.
    from shaderbox.popups.examples import _draw_grid

    app.displayed_sizes.clear()
    imgui.new_frame()
    imgui.begin("rig")
    _draw_grid(app)
    imgui.end()
    imgui.end_frame()
    assert set(app.displayed_sizes) & set(app.ui_document_examples), (
        "the examples grid drew thumbnails and recorded nothing"
    )


def test_an_auto_document_with_no_recorder_keeps_its_size(app: Any) -> None:
    # The other half of the rule: a document displayed nowhere must not resize to anything.
    document_id = app.current_document_id
    document = app.ui_documents[document_id].document
    document.resolution_mode = ResolutionMode.AUTO
    before = document.canvas_size
    app.displayed_sizes.clear()
    _drive(app, 3)
    assert document.canvas_size == before


# ---------------------------------------------------------------------------
# V9a -- a throttled example is actually skipped in the popup's set
# ---------------------------------------------------------------------------


def test_a_throttled_example_renders_less_often_than_a_cheap_one(
    app: Any, monkeypatch: Any
) -> None:
    # The popup's set is an ALTERNATIVE to `tick_documents`, so an interval computed for an
    # example id has to be read by the EXAMPLES branch or it is read by nothing at all.
    # Falsifier: render the examples loop without the `(frame_idx + phase) % k` gate and both
    # examples advance every frame while the plan's intervals look perfect.
    from shaderbox.app import PopupState
    from shaderbox.ui import update_and_draw

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
    from shaderbox import ui

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
    from shaderbox.profiling import FrameProfile, Span
    from shaderbox.render_plan import RenderPlan
    from shaderbox.ui_primitives import profile_rows_plan

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
    from shaderbox.ui import update_and_draw

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
    from shaderbox.ui import update_and_draw

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
    # Auto and the next frame's resolution step resizes it back to the panel -- the tool
    # silently does nothing while every one of its own assertions passes.
    document_id = app.current_document_id
    result = app.copilot_backend.set_canvas_size(document_id, 128, 200)
    assert result.ok and (result.width, result.height) == (128, 200)
    document = app.ui_documents[document_id].document
    assert document.resolution_mode is ResolutionMode.FIXED
    assert document.resolution == (128, 200)

    app.displayed_sizes[document_id] = (900, 700)  # a panel asking for something else
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
    from shaderbox.tabs.document import _apply_canvas_size

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


def test_the_starter_document_is_the_fixture_it_claims_to_be(app: Any) -> None:
    # A guard on the rig itself: every test above drives the shipped starter, and the shape of
    # its persisted state is what the loop reads.
    document = app.ui_documents[STARTER_EXAMPLE_ID].document
    assert document.resolution == (1280, 960)
    assert document.resolution_mode is ResolutionMode.AUTO
