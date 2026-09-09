"""The frame profiler's tree, its GPU ring, and the wire from the live loop (feature 088).

The CPU half runs anywhere. The GL half needs a real context and skips without one; the
`update_and_draw` test drives a hidden-window `App` the way `scripts/smoke.py` does.

Each test names the falsifier it was born against in its own body, because a timer that
returns plausible numbers is exactly the shape a passing suite cannot distinguish from a
broken one.
"""

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import moderngl
import pytest

from shaderbox.pass_graph import PassEntry, PassGraph
from shaderbox.profiling import (
    NULL_PROFILER,
    RING_DEPTH,
    SMOOTHING,
    FrameProfile,
    Profiler,
    ProfileSmoother,
    Span,
    by_cost,
)
from shaderbox.ui_primitives import profile_rows_plan

_RED = """#version 460 core
in vec2 vs_uv;
out vec4 fs_color;
void main() { fs_color = vec4(1.0, 0.0, 0.0, 1.0); }
"""

_BLUE = """#version 460 core
in vec2 vs_uv;
uniform sampler2D u_first;
out vec4 fs_color;
void main() { fs_color = vec4(texture(u_first, vs_uv).r, 0.0, 1.0, 1.0); }
"""


@pytest.fixture(scope="module")
def gl_ctx() -> Iterator[moderngl.Context]:
    os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "4.6")
    os.environ.setdefault("MESA_GLSL_VERSION_OVERRIDE", "460")
    # Default backend, like every other GL module's fixture: an explicit EGL context released
    # here poisons the process's EGL display for the next module.
    try:
        context = moderngl.create_standalone_context()
    except Exception as e:
        pytest.skip(f"no standalone GL context available: {e}")
    fbo = context.simple_framebuffer((512, 512))
    fbo.use()
    yield context
    context.release()


def _names(spans: list[Span]) -> list[str]:
    return [span.name for span in spans]


def _find(span: Span, name: str) -> Span | None:
    if span.name == name:
        return span
    for child in span.children:
        found = _find(child, name)
        if found is not None:
            return found
    return None


def _burn(gl: moderngl.Context, clears: int) -> None:
    for _ in range(clears):
        gl.clear(0.1, 0.2, 0.3, 1.0)


# ---------------------------------------------------------------------------
# V1 -- the tree's shape
# ---------------------------------------------------------------------------


def test_a_nested_cpu_span_lands_under_its_parent() -> None:
    # Falsifier: pop the wrong span in `_pop` and `b` lands beside `a` under the root.
    profiler = Profiler()
    profiler.begin_frame()
    with profiler.cpu("a"), profiler.cpu("b"):
        pass
    profile = profiler.end_frame()
    assert profile is not None
    assert _names(profile.children) == ["a"]
    a = profile.children[0]
    assert _names(a.children) == ["b"]
    assert a.children[0].cpu_ms <= a.cpu_ms


def test_the_root_wall_covers_every_child() -> None:
    profiler = Profiler()
    profiler.begin_frame()
    with profiler.cpu("a"):
        pass
    with profiler.cpu("b"):
        pass
    profile = profiler.end_frame()
    assert profile is not None
    assert _names(profile.children) == ["a", "b"]
    assert profile.cpu_ms >= sum(child.cpu_ms for child in profile.children)


def test_two_same_named_siblings_under_one_parent_are_two_spans() -> None:
    profiler = Profiler()
    profiler.begin_frame()
    with profiler.cpu("pass:blur"):
        pass
    with profiler.cpu("pass:blur"):
        pass
    profile = profiler.end_frame()
    assert profile is not None
    assert _names(profile.children) == ["pass:blur", "pass:blur"]


# ---------------------------------------------------------------------------
# V2 -- a nested GPU span raises before it can lie
# ---------------------------------------------------------------------------


def _open_two_gpu_spans(profiler: Profiler, created: list[Any]) -> int:
    """Open a GPU span inside another and report how many queries existed at the inner entry.

    The nesting is the point, so the two `with` blocks cannot be merged into one statement:
    the inner must enter while the outer is open.
    """
    with profiler.gpu("outer"):
        before = len(created)
        with profiler.gpu("inner"):
            pass
    return before


def _count_queries(gl: moderngl.Context, monkeypatch: Any, created: list[Any]) -> None:
    """Route `gl.query` through a counter, so "creates no query" is observable."""
    real_query = gl.query

    def counting_query(**kwargs: Any) -> moderngl.Query:
        query = real_query(**kwargs)
        created.append(query)
        return query

    monkeypatch.setattr(gl, "query", counting_query)


def test_a_gpu_span_inside_another_raises(gl_ctx: moderngl.Context) -> None:
    # Falsifier: delete the assert in `Profiler.gpu` and this sees no raise -- and the
    # numbers go wrong in silence, which is the whole reason the assert is there (the outer
    # query reads garbage, the inner 0, `ctx.error` GL_INVALID_OPERATION, no exception).
    profiler = Profiler()
    profiler.begin_frame()
    with pytest.raises(AssertionError):
        _open_two_gpu_spans(profiler, [])
    profiler.enabled = False


def test_the_nested_gpu_span_raises_before_its_query_begins(
    gl_ctx: moderngl.Context, monkeypatch: Any
) -> None:
    # The raise must precede the begin, or GL has already been put in the bad state the
    # assert exists to prevent. Counting queries created is how that is observable.
    profiler = Profiler()
    created: list[Any] = []
    _count_queries(gl_ctx, monkeypatch, created)
    profiler.begin_frame()
    with pytest.raises(AssertionError):
        _open_two_gpu_spans(profiler, created)
    assert len(created) == 1, "the inner span created a query before the assert fired"
    profiler.enabled = False


# ---------------------------------------------------------------------------
# V3 -- the read lands two frames late
# ---------------------------------------------------------------------------


def test_a_gpu_span_reads_two_frames_late(gl_ctx: moderngl.Context) -> None:
    # Falsifier: read one frame late (drop RING_DEPTH to 2 and read at N+1) and the
    # "still None after frame 2" assertion fails -- which is also the read that measured a
    # 22.3 ms stall under load, the reason the ring is three deep.
    profiler = Profiler()
    profiler.begin_frame()
    with profiler.gpu("pass:heavy"):
        _burn(gl_ctx, 200)
    first = profiler.end_frame()
    assert first is not None
    assert not first.complete
    heavy = _find(first.root, "pass:heavy")
    assert heavy is not None
    assert heavy.gpu_ms is None

    profiler.begin_frame()
    profiler.end_frame()
    assert heavy.gpu_ms is None

    profiler.begin_frame()
    assert first.complete
    assert heavy.gpu_ms is not None
    assert heavy.gpu_ms > 0.0
    profiler.end_frame()
    profiler.enabled = False


def test_two_same_named_gpu_siblings_under_one_parent_read_their_own_loads(
    gl_ctx: moderngl.Context,
) -> None:
    """The live shape: the same document renders twice in one frame (its output, then a
    pending pass's chain), so two `document:X` spans sit under the root and each carries its
    own `pass:main`. The sibling ORDINAL in the ring key is what keeps them apart.

    Falsifier: hard-code `ordinal = 0` in `_push` and the first sibling reads None -- the
    second's query overwrites its ring slot, and one query begun twice in a frame reports
    only the second block.
    """
    profiler = Profiler()
    profiler.begin_frame()
    with profiler.gpu("pass:a"):
        _burn(gl_ctx, 10)
    with profiler.gpu("pass:a"):
        _burn(gl_ctx, 200)
    profile = profiler.end_frame()
    assert profile is not None
    for _ in range(RING_DEPTH):
        profiler.begin_frame()
        profiler.end_frame()

    light, heavy = profile.children
    assert light.gpu_ms is not None and heavy.gpu_ms is not None, (
        f"one sibling lost its number: {[light.gpu_ms, heavy.gpu_ms]}"
    )
    assert light.gpu_ms > 0.0
    assert heavy.gpu_ms > light.gpu_ms * 2.0, (
        f"the 200-clear sibling read {heavy.gpu_ms:.4f} ms against the 10-clear sibling's "
        f"{light.gpu_ms:.4f} ms"
    )
    profiler.enabled = False
    profiler.begin_frame()


def test_same_name_under_two_parents_reads_two_distinct_loads(
    gl_ctx: moderngl.Context,
) -> None:
    # V3a. Falsifier: key the ring by NAME rather than by path and the first span reads
    # 0.0 -- one query object begun twice in a frame reports only the second block.
    profiler = Profiler()
    profiler.begin_frame()
    with profiler.cpu("light"), profiler.gpu("pass:a"):
        _burn(gl_ctx, 10)
    with profiler.cpu("heavy"), profiler.gpu("pass:a"):
        _burn(gl_ctx, 200)
    profile = profiler.end_frame()
    assert profile is not None
    for _ in range(RING_DEPTH):
        profiler.begin_frame()
        profiler.end_frame()

    light = _find(profile.root.children[0], "pass:a")
    heavy = _find(profile.root.children[1], "pass:a")
    assert light is not None and light.gpu_ms is not None
    assert heavy is not None and heavy.gpu_ms is not None
    assert light.gpu_ms > 0.0
    assert heavy.gpu_ms > light.gpu_ms * 2.0, (
        f"the 200-clear span read {heavy.gpu_ms:.4f} ms against the 10-clear span's "
        f"{light.gpu_ms:.4f} ms -- the ring lost one of them"
    )
    profiler.enabled = False


# ---------------------------------------------------------------------------
# V4 -- a disabled profiler costs nothing
# ---------------------------------------------------------------------------


def test_a_disabled_profiler_creates_no_query_and_no_tree(
    gl_ctx: moderngl.Context, monkeypatch: Any
) -> None:
    # Falsifier: create the query eagerly in `gpu(...)` (before the `enabled` check) and the
    # count is 1000 instead of 0.
    profiler = Profiler(enabled=False)
    created: list[Any] = []
    _count_queries(gl_ctx, monkeypatch, created)
    profiler.begin_frame()
    for index in range(1000):
        with profiler.gpu(f"pass:{index}"):
            pass
    profile = profiler.end_frame()

    assert created == []
    assert profile is None


def test_the_null_profiler_is_disabled() -> None:
    assert not NULL_PROFILER.enabled


def test_disabling_drops_the_query_ring(gl_ctx: moderngl.Context) -> None:
    # `moderngl.Query` has no `release()` and no `__del__`, so the ring's only eviction rule
    # is the disable. Falsifier: leave `_ring` standing in the setter and it keeps its key.
    profiler = Profiler()
    profiler.begin_frame()
    with profiler.gpu("pass:x"):
        _burn(gl_ctx, 4)
    profiler.end_frame()
    assert profiler._ring
    profiler.enabled = False
    assert profiler._ring, "the drop must wait for the frame boundary"
    profiler.begin_frame()
    assert not profiler.enabled
    assert not profiler._ring


def test_a_toggle_mid_span_does_not_disturb_the_open_span(
    gl_ctx: moderngl.Context,
) -> None:
    """Disabling from inside an open span leaves that span able to close.

    This is the live shape: `ui.py` writes `app.profiler.enabled` from inside the `ui` span,
    because the panel that toggles it is drawn there. Falsifier: act on the flag in the
    setter and the enclosing span's `finally` pops an empty stack (`IndexError`).
    """
    profiler = Profiler()
    profiler.begin_frame()
    with profiler.cpu("ui"):
        with profiler.gpu("ui:draw"):
            _burn(gl_ctx, 4)
        profiler.enabled = False
    profile = profiler.end_frame()
    assert profile is not None
    assert _names(profile.children) == ["ui"]
    assert profiler.enabled, "the toggle must not land until the next frame"
    profiler.begin_frame()
    assert not profiler.enabled


# ---------------------------------------------------------------------------
# V5 -- a document reports its passes
# ---------------------------------------------------------------------------


def _two_pass_document(gl: moderngl.Context, tmp_path: Path) -> Any:
    from shaderbox.core import Pass
    from shaderbox.document import Document
    from shaderbox.shader_source import ShaderSource

    document = Document(gl=gl, canvas_size=(64, 64))
    for render_pass in document.passes.values():
        render_pass.release()
    document.passes = {}
    for name, source in (("first", _RED), ("second", _BLUE)):
        path = tmp_path / f"{name}.frag.glsl"
        path.write_text(source)
        document.passes[name] = Pass(
            gl=gl, source=ShaderSource.load(path), canvas_size=(64, 64)
        )
    document.graph = PassGraph(
        passes={"first": PassEntry(), "second": PassEntry(iterations=3)},
        output="second",
    )
    return document


def test_a_two_pass_document_yields_one_span_per_pass(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # Falsifier: move the `gpu(...)` inside the iteration loop and `second` appears three
    # times with count 1 instead of once with count 3.
    document = _two_pass_document(gl_ctx, tmp_path)
    profiler = Profiler()
    profiler.begin_frame()
    document.render(u_time=0.0, profiler=profiler)
    profile = profiler.end_frame()
    assert profile is not None
    assert _names(profile.children) == ["pass:first", "pass:second"]
    assert [span.count for span in profile.children] == [1, 3]
    profiler.enabled = False


def test_a_document_rendered_twice_in_one_frame_yields_its_passes_twice(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # The live loop's own shape: both render sites name the span `document:<name>`, so the two
    # parents are same-name siblings too. Falsifier: merge same-name siblings and the second
    # render vanishes -- the tree reports one document where two were drawn.
    document = _two_pass_document(gl_ctx, tmp_path)
    profiler = Profiler()
    profiler.begin_frame()
    for _ in range(2):
        with profiler.cpu("document:one"):
            document.render(u_time=0.0, profiler=profiler)
    profile = profiler.end_frame()
    assert profile is not None
    assert _names(profile.children) == ["document:one", "document:one"]
    for parent in profile.children:
        assert _names(parent.children) == ["pass:first", "pass:second"]
    profiler.enabled = False


def test_an_export_render_opens_no_span(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # The seam's whole point (088 D3): a render that takes the default reports nowhere, so
    # an export's passes cannot land in whatever live frame they ran inside. Falsifier: make
    # the profiler a module-level active object and this render joins the open frame.
    document = _two_pass_document(gl_ctx, tmp_path)
    profiler = Profiler()
    profiler.begin_frame()
    document.render(u_time=0.0)
    profile = profiler.end_frame()
    assert profile is not None
    assert profile.children == []
    profiler.enabled = False


# ---------------------------------------------------------------------------
# V6 and V7 -- the wire from the live loop, and the abort path closing its root
#
# One test, one App: only ONE test per process may drive `ui.update_and_draw`
# (`conventions.md ## Known quirks`), and both facts are properties of that loop rather than
# of the profiler alone.
# ---------------------------------------------------------------------------


def _abort_one_frame(app: Any, monkeypatch: Any) -> Any:
    """Drive one frame whose `_tick_frame_state` returns None -- the abort a vanished shader
    file takes -- and return the profile that frame produced.

    Driven at the function rather than by deleting the file: the disk sync runs first and
    unloads the document, so the file-missing branch is not reachable from outside.
    """
    from shaderbox import ui

    monkeypatch.setattr(ui, "_tick_frame_state", lambda _app: None)
    try:
        ui.update_and_draw(app)
    finally:
        monkeypatch.undo()
    return app.last_profile


def _pass_settings_frames(app: Any) -> Any:
    """Frames with the pass-settings modal up -- the one popup that keeps rendering.

    Enough of them that the profile finally published is itself a modal frame: a profile
    completes two frames after it closes, so fewer would still be reporting the main branch.
    """
    from shaderbox.app import PopupState
    from shaderbox.ui import update_and_draw

    try:
        for _ in range(RING_DEPTH + 2):
            # Re-set each frame: the modal's own draw closes it again, since imgui never sees
            # the `open_popup` a real click would have made.
            app.popup_state = PopupState.PASS_SETTINGS
            update_and_draw(app)
    finally:
        app.popup_state = PopupState.CLOSED
    return app.last_profile


def _capture_the_overlays_profile(app: Any, monkeypatch: Any) -> Any:
    """Drive one frame with the overlay spied on, and report the profile it was handed.

    Which tree reaches the panel is the decision this feature makes, and only the call site
    knows it -- the smoother alone cannot say whether anyone draws what it returns.
    """
    from shaderbox import ui

    handed: list[Any] = []
    real = ui.fps_overlay

    def spy(**kwargs: Any) -> Any:
        handed.append(kwargs["profile"])
        return real(**kwargs)

    monkeypatch.setattr(ui, "fps_overlay", spy)
    try:
        ui.update_and_draw(app)
    finally:
        monkeypatch.undo()
    assert handed, "the overlay was never drawn"
    return handed[-1]


def _capture_the_plan_call(app: Any, monkeypatch: Any) -> Any:
    """Drive one frame with `profile_rows_plan` spied on, and report its arguments.

    The spy goes on `ui_primitives`, the module `fps_overlay` resolves the name in at call
    time; `ui.fps_overlay` stays real, so the plan call actually runs inside a live draw.
    """
    from shaderbox import ui_primitives

    calls: list[Any] = []
    real = ui_primitives.profile_rows_plan

    def spy(profile: Any, fps: int, target_fps: int) -> Any:
        calls.append((profile, fps, target_fps))
        return real(profile, fps, target_fps)

    monkeypatch.setattr(ui_primitives, "profile_rows_plan", spy)
    try:
        from shaderbox import ui

        ui.update_and_draw(app)
    finally:
        monkeypatch.undo()
    return calls


def _close_the_panel(app: Any, monkeypatch: Any) -> None:
    """Make the overlay report a closing click, which is what the live toggle is."""
    from shaderbox import ui

    monkeypatch.setattr(ui, "fps_overlay", lambda **_kwargs: False)


def test_the_wire_and_the_abort_path(app: Any, monkeypatch: Any) -> None:
    """`update_and_draw` with the panel open leaves a complete profile on `app.last_profile`,
    and an aborted frame still closes its root so the next tree is flat.

    This is the "defined is not wired" check, and its falsifiers are: cut `profiler=app.profiler`
    at a render site in `ui.py` and that site's `document:` span is still there while its
    `pass:` child is gone; drop `app.profiler.enabled = app.fps_details_open` and
    `last_profile` stays None; replace the `with app.profiler.frame()` with a bare
    `begin_frame()` / `end_frame()` pair and the aborted frame never publishes; act on the
    `enabled` flag in the setter instead of at the frame boundary and closing the panel
    raises `IndexError: pop from empty list`; drop `_publish`'s ordering check and the
    published index walks backwards across the abort.
    """
    from shaderbox.ui import update_and_draw

    app.fps_details_open = True
    for _ in range(4):
        update_and_draw(app)

    profile = app.last_profile
    assert profile is not None, (
        "no complete profile after four frames with the panel open"
    )
    assert profile.complete
    document_spans = [s for s in profile.children if s.name.startswith("document:")]
    assert document_spans, f"no document span in {_names(profile.children)}"
    pass_spans = [s for s in document_spans[0].children if s.name.startswith("pass:")]
    assert pass_spans, f"no pass span under {document_spans[0].name}"
    assert pass_spans[0].gpu_ms is not None, "the pass span carries no GPU number"
    assert pass_spans[0].gpu_ms >= 0.0

    aborted = _abort_one_frame(app, monkeypatch)
    # An aborted frame is a frame: it opened no GPU span, so its profile is complete the
    # moment it closes and lands on `last_profile` straight away, carrying `tick` alone.
    assert aborted is not None and aborted is not profile
    assert _names(aborted.children) == ["tick"], _names(aborted.children)

    reads: list[float | None] = []
    drew: list[Any] = []
    indexes: list[int] = [aborted.index]
    seen: set[int] = set()
    for _ in range(12):
        update_and_draw(app)
        current = app.last_profile
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        indexes.append(current.index)
        assert current.root.name == "frame"
        assert _names(current.children)[0] == "tick"
        if [s for s in current.children if s.name.startswith("document:")]:
            drew.append(current)
        draw = next((s for s in current.children if s.name == "ui:draw"), None)
        if draw is not None:
            reads.append(draw.gpu_ms)

    # A root the abort left open stalls the frame index, so one ring slot's query is begun
    # twice inside its own lifetime -- and a query begun twice reports 0.0 for the first
    # block, with no GL error to read. Every post-abort read being live is what says the
    # ring stayed aligned with the frames that drew.
    assert len(reads) >= 6, f"only {len(reads)} complete profiles after the abort"
    assert drew, "no post-abort frame rendered a document"
    assert all(value is not None and value > 0.0 for value in reads), (
        f"ui:draw read {reads} after the abort -- the ring lost its alignment"
    )
    # A frame that opens no GPU span completes the instant it closes, while the frame two
    # behind it completes at the next `begin_frame` -- so completion is out of order across
    # an abort, and `last_complete` publishes by frame index rather than by arrival.
    assert indexes == sorted(indexes), (
        f"the published index walked backwards: {indexes}"
    )

    behind = _pass_settings_frames(app)
    assert behind is not None
    modal_documents = [s for s in behind.children if s.name.startswith("document:")]
    assert modal_documents, (
        f"the pass-settings modal's render reported nowhere: {_names(behind.children)}"
    )
    assert [s for s in modal_documents[0].children if s.name.startswith("pass:")]

    # The panel draws the AVERAGE, so the loop must be feeding it -- the smoothed tree
    # carries the same shape as the raw one. Falsifier: cut the `feed` call in
    # `update_and_draw` and the smoother stays empty while the panel is open.
    averaged = app.profile_smoother.smoothed()
    assert averaged is not None, "the loop never fed the smoother"
    assert _names(averaged.children) == _names(behind.children)

    # What the PANEL is handed is the average, not the raw frame. The two agree on shape, so
    # the discriminator is identity: the smoother is seeded with a childless root at an index
    # no live frame reaches, and the overlay is drawn BEFORE the loop's own feed, so the
    # averaged tree still carries that seed while the raw one carries the frame. Falsifier:
    # pass `app.last_profile` at the `fps_overlay` call and the overlay reads the live tree.
    app.profile_smoother.feed(_profile(10_000, 500.0))
    handed = _capture_the_overlays_profile(app, monkeypatch)
    assert handed is not None
    assert handed.index == 10_000 and handed.children == [], (
        f"the overlay drew index {handed.index} with {_names(handed.children)} -- "
        "that is the frame itself, not the average"
    )

    # V11 -- the panel draws from the PLAN, and the plan is handed the same smoothed tree
    # and the live target. Without this seam a sort or a color inside the draw loop would be
    # unfalsifiable: the overlay spy above sees only what goes IN. Falsifier: cut the
    # `profile_rows_plan` call from `fps_overlay` and no call is recorded.
    app.profile_smoother.feed(_profile(10_001, 500.0))
    calls = _capture_the_plan_call(app, monkeypatch)
    assert calls, "the overlay drew without planning its rows"
    planned_profile, _planned_fps, planned_target = calls[-1]
    assert planned_profile is not None
    assert planned_profile.index == 10_001 and planned_profile.children == [], (
        f"the plan was handed index {planned_profile.index} -- that is not the average"
    )
    assert planned_target == app.app_state.global_target_fps

    # Closing the panel writes `enabled` from INSIDE the open `ui` span, so a setter that
    # acted at once would clear the stack under a span whose `finally` has yet to run.
    _close_the_panel(app, monkeypatch)
    update_and_draw(app)
    update_and_draw(app)
    assert not app.profiler.enabled
    assert not app.profiler._ring, "closing the panel must leave no query behind"
    # The profiler dropped its whole state at that boundary; the average goes with it, or a
    # re-opened panel blends its first live frames into the last session's numbers.
    # Falsifier: drop the `else: reset()` branch and this reads the pre-close tree.
    assert app.profile_smoother.smoothed() is None


# ---------------------------------------------------------------------------
# V8 -- the panel's exponential average
#
# GL-free: the smoother reads a tree and writes a tree, so these build their profiles by
# hand rather than measuring anything.
# ---------------------------------------------------------------------------


def _profile(index: int, cpu_ms: float, child_ms: float | None = None) -> FrameProfile:
    root = Span("frame", cpu_ms=cpu_ms)
    if child_ms is not None:
        root.children.append(Span("pass:a", cpu_ms=child_ms))
    return FrameProfile(root, index, complete=True)


def _child_cpu(profile: FrameProfile | None, name: str) -> float:
    assert profile is not None
    span = _find(profile.root, name)
    assert span is not None
    return span.cpu_ms


def test_a_jittering_span_converges_and_never_moves_more_than_the_factor() -> None:
    """A 1/9 alternation settles near 5, and no single step crosses a quarter of its gap.

    The step bound is the literal 0.25, not `SMOOTHING`: reading the constant the code uses
    would leave the assertion true for every value that constant could take, which is the one
    shape a passing run cannot tell from a broken one. The constant is pinned separately.

    Falsifier: build the smoother with `smoothing=1.0` -- the raw numbers straight through --
    and the per-step assertion fails on the first step, which moves the whole 8 ms gap.
    """
    smoother = ProfileSmoother()
    previous: float | None = None
    for index in range(40):
        raw = 1.0 if index % 2 == 0 else 9.0
        smoother.feed(_profile(index, 20.0, raw))
        value = _child_cpu(smoother.smoothed(), "pass:a")
        if previous is not None:
            assert abs(value - previous) <= 0.25 * abs(raw - previous) + 1e-9, (
                f"step {index} moved {abs(value - previous):.4f} ms of a "
                f"{abs(raw - previous):.4f} ms gap"
            )
        previous = value

    assert previous is not None
    assert abs(previous - 5.0) < 1.0, (
        f"a 1/9 jitter settled at {previous:.4f}, not near 5"
    )


def test_the_shipped_factor_is_the_one_the_step_bound_was_written_against() -> None:
    assert SMOOTHING == 0.25


def test_feeding_one_profile_twice_applies_it_once() -> None:
    # The live shape: the panel sees `last_complete` repeat while the ring fills. Falsifier:
    # drop the `profile.index == self._fed_index` check and the second feed pulls the average
    # a second step toward 9, so the two readings differ.
    smoother = ProfileSmoother()
    smoother.feed(_profile(0, 20.0, 1.0))
    repeated = _profile(1, 20.0, 9.0)
    smoother.feed(repeated)
    once = _child_cpu(smoother.smoothed(), "pass:a")
    smoother.feed(repeated)
    assert _child_cpu(smoother.smoothed(), "pass:a") == once, (
        f"the repeat moved the average from {once:.4f} ms"
    )


def test_a_span_seen_for_the_first_time_seeds_at_its_raw_value() -> None:
    # Falsifier: seed at 0.0 (start the blend from a missing key as zero) and a pass that
    # appears mid-run climbs from nothing, reading 2.0 ms on its first frame instead of 8.0.
    smoother = ProfileSmoother()
    smoother.feed(_profile(0, 20.0))
    smoother.feed(_profile(1, 20.0, 8.0))
    assert _child_cpu(smoother.smoothed(), "pass:a") == 8.0


def test_a_span_absent_from_the_newest_profile_leaves_the_smoothed_tree() -> None:
    """A pass deleted from the graph leaves the panel, and one added back starts fresh.

    The tree's SHAPE is the newest profile's, so the row's disappearance is structural; what
    a merge would keep is the stale AVERAGE behind the key, which is why the second half is
    the half with teeth. Falsifier: keep the previous frame's keys (`self._cpu.update(...)`
    instead of replacing it) and the returning pass resumes the deleted one's average,
    reading 8.0 ms on its first frame where the raw value is 20.0.
    """
    smoother = ProfileSmoother()
    smoother.feed(_profile(0, 20.0, 4.0))
    smoother.feed(_profile(1, 20.0))
    smoothed = smoother.smoothed()
    assert smoothed is not None
    assert _find(smoothed.root, "pass:a") is None

    smoother.feed(_profile(2, 20.0, 20.0))
    assert _child_cpu(smoother.smoothed(), "pass:a") == 20.0


def test_reset_empties_the_average() -> None:
    # The disable path: the profiler drops its whole state at the frame boundary and the
    # average goes with it. Falsifier: make `reset` a no-op and the panel re-opens showing
    # the last session's numbers, and the first new frame blends into them.
    smoother = ProfileSmoother()
    smoother.feed(_profile(0, 20.0, 4.0))
    smoother.reset()
    assert smoother.smoothed() is None
    smoother.feed(_profile(0, 20.0, 10.0))
    assert _child_cpu(smoother.smoothed(), "pass:a") == 10.0


def test_two_same_named_siblings_smooth_separately() -> None:
    # The ring's own key shape, reused: `name#ordinal` per level. Falsifier: key by name
    # alone and the two siblings share one average, so the light one reads the heavy one's.
    smoother = ProfileSmoother()
    for index in range(20):
        root = Span("frame", cpu_ms=20.0)
        root.children.append(Span("pass:a", cpu_ms=2.0))
        root.children.append(Span("pass:a", cpu_ms=18.0))
        smoother.feed(FrameProfile(root, index, complete=True))
    smoothed = smoother.smoothed()
    assert smoothed is not None
    light, heavy = smoothed.children
    assert abs(light.cpu_ms - 2.0) < 0.1 and abs(heavy.cpu_ms - 18.0) < 0.1, (
        f"the siblings smoothed together: {light.cpu_ms:.4f} and {heavy.cpu_ms:.4f}"
    )


# ---------------------------------------------------------------------------
# V10 -- the panel's cost order (feature 089 D8)
#
# GL-free: `by_cost` reads a list and writes a list, and the plan is a pure function of a
# tree, so these build their spans by hand.
# ---------------------------------------------------------------------------


def test_by_cost_puts_the_costliest_first() -> None:
    # Falsifier: drop `reverse=True` and recording order comes back as 1, 2, 3.
    children = [Span("a", cpu_ms=1.0), Span("b", cpu_ms=3.0), Span("c", cpu_ms=2.0)]
    assert _names(by_cost(children)) == ["b", "c", "a"]


def test_by_cost_ranks_a_gpu_span_by_its_gpu_number() -> None:
    # The headline number is what the row PRINTS, so a GPU span sorts by its GPU time even
    # where its wall is larger. Falsifier: key on `cpu_ms` and the cheap GPU span, whose
    # CPU issue cost is the highest of the three, leads.
    children = [
        Span("gpu_light", cpu_ms=9.0, gpu_ms=1.0),
        Span("cpu_mid", cpu_ms=4.0),
        Span("gpu_heavy", cpu_ms=1.0, gpu_ms=8.0),
    ]
    assert _names(by_cost(children)) == ["gpu_heavy", "cpu_mid", "gpu_light"]


def test_equal_costs_keep_recording_order() -> None:
    # Falsifier: an unstable sort, or a tie-break on the name, and the two 5 ms spans swap.
    children = [Span("first", cpu_ms=5.0), Span("second", cpu_ms=5.0)]
    assert _names(by_cost(children)) == ["first", "second"]


def test_by_cost_leaves_the_instrument_alone() -> None:
    # The profile is what was measured, so only the reader sorts. Falsifier: sort in place
    # (`children.sort(...)`) and the recorded tree is reordered under every other consumer.
    children = [Span("a", cpu_ms=1.0), Span("b", cpu_ms=3.0)]
    sorted_children = by_cost(children)
    assert sorted_children is not children
    assert _names(children) == ["a", "b"]


def test_the_plans_tree_is_cost_ordered_with_other_last() -> None:
    # The order reaches the panel through the plan, which is what the draw loop walks.
    # Falsifier: iterate `span.children` in the plan and the rows come out a, b, c.
    root = Span("frame", cpu_ms=20.0)
    root.children.append(Span("pass:a", cpu_ms=1.0))
    root.children.append(Span("pass:b", cpu_ms=3.0))
    root.children.append(Span("pass:c", cpu_ms=2.0))
    rows = profile_rows_plan(
        FrameProfile(root, 0, complete=True), fps=60, target_fps=60
    )
    names = [row.name for row in rows]
    assert names == [
        "frame",
        "gpu",
        "budget",
        "fps",
        "target",
        "pass:b",
        "pass:c",
        "pass:a",
        "other",
    ]


def test_the_plan_indents_a_child_under_its_parent() -> None:
    # Depth is the plan's, so the draw loop only multiplies it by the spacing token.
    # Falsifier: pass a constant depth in `_plan_tree` and the tree flattens.
    root = Span("frame", cpu_ms=20.0)
    outer = Span("document:one", cpu_ms=10.0)
    outer.children.append(Span("pass:inner", cpu_ms=6.0))
    root.children.append(outer)
    rows = profile_rows_plan(
        FrameProfile(root, 0, complete=True), fps=60, target_fps=60
    )
    depths = {row.name: row.depth for row in rows}
    assert depths["document:one"] == 0
    assert depths["pass:inner"] == 1


def test_the_gap_before_the_tree_is_marked_on_the_first_tree_row() -> None:
    # The draw reads the boundary off the row, so the panel never re-derives which rows
    # lead. Falsifier: stop setting the flag and no row carries it.
    root = Span("frame", cpu_ms=20.0)
    root.children.append(Span("pass:a", cpu_ms=1.0))
    root.children.append(Span("pass:b", cpu_ms=3.0))
    rows = profile_rows_plan(
        FrameProfile(root, 0, complete=True), fps=60, target_fps=60
    )
    marked = [row.name for row in rows if row.starts_tree]
    assert marked == ["pass:b"]
    assert rows[[row.name for row in rows].index("pass:b") - 1].name == "target"


def test_a_plan_without_a_profile_marks_no_row() -> None:
    # Three static rows and no tree, so there is no boundary to mark.
    rows = profile_rows_plan(None, fps=60, target_fps=60)
    assert [row.name for row in rows] == ["budget", "fps", "target"]
    assert not any(row.starts_tree for row in rows)
