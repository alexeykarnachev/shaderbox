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
from shaderbox.profiling import NULL_PROFILER, RING_DEPTH, Profiler, Span

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

    # Closing the panel writes `enabled` from INSIDE the open `ui` span, so a setter that
    # acted at once would clear the stack under a span whose `finally` has yet to run.
    _close_the_panel(app, monkeypatch)
    update_and_draw(app)
    update_and_draw(app)
    assert not app.profiler.enabled
    assert not app.profiler._ring, "closing the panel must leave no query behind"
