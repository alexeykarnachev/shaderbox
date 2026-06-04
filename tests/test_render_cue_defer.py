"""The Render-tab deferred-render state machine (_run_pending_render). A click sets app.render_request;
the runner must HOLD it one frame (so the "Rendering..." cue paints + swaps) before running the encode
on the next frame — otherwise the encode freezes the loop with no cue ever shown. Pure: a tiny stub
stands in for App (the runner only touches render_request / render_request_shown)."""

import types

from shaderbox.ui import _run_pending_render


def _app() -> types.SimpleNamespace:
    return types.SimpleNamespace(render_request=None, render_request_shown=False)


def test_nothing_pending_is_a_noop() -> None:
    app = _app()
    _run_pending_render(app)
    assert app.render_request is None
    assert app.render_request_shown is False


def test_holds_one_frame_then_runs() -> None:
    app = _app()
    ran: list[str] = []
    app.render_request = lambda: ran.append("render")

    # Frame N (click frame): the request must be HELD, not run — the cue needs a frame to paint.
    _run_pending_render(app)
    assert ran == [], "encode ran on the click frame — the cue never gets a frame to swap"
    assert app.render_request is not None, "request dropped on the hold frame (cue would not paint)"
    assert app.render_request_shown is True

    # Frame N+1: now the encode runs and the request clears.
    _run_pending_render(app)
    assert ran == ["render"], "encode did not run on the frame after the hold"
    assert app.render_request is None
    assert app.render_request_shown is False


def test_shown_flag_resets_when_request_cleared_externally() -> None:
    # If the request is cleared between frames (e.g. never set again), the shown flag must reset so
    # the NEXT request gets its own hold frame rather than running immediately.
    app = _app()
    app.render_request = lambda: None
    _run_pending_render(app)  # hold
    app.render_request = None  # cleared without running
    _run_pending_render(app)
    assert app.render_request_shown is False

    ran: list[str] = []
    app.render_request = lambda: ran.append("x")
    _run_pending_render(app)
    assert ran == [], "a fresh request must be held one frame, not run immediately"
