"""The Render-tab one-frame-hold FSM (RenderDefer). A submitted render must NOT fire until the
"Rendering..." cue frame has been marked shown — so the cue is on the glass before the synchronous
encode freezes the loop. Pure: the GL calls live in ui.py; this owns only request + the latch."""

from shaderbox.render_defer import RenderDefer


def test_holds_one_frame_then_fires_then_clears() -> None:
    defer = RenderDefer()
    ran: list[str] = []

    # Idle: nothing pending.
    assert not defer.has_request()
    assert not defer.ready_to_fire()

    # Submit: a request is pending but NOT ready to fire (the cue hasn't painted yet).
    defer.submit(lambda: ran.append("render"))
    assert defer.has_request()
    assert not defer.ready_to_fire(), "fired before the cue frame was shown"

    # Cue frame painted -> now ready.
    defer.mark_shown()
    assert defer.ready_to_fire()

    # Fire + clear: returns the request and resets to idle.
    request = defer.fire_and_clear()
    assert request is not None
    request()
    assert ran == ["render"]
    assert not defer.has_request()
    assert not defer.ready_to_fire()


def test_submit_resets_shown() -> None:
    # A fresh submit must re-arm the one-frame hold even if a prior cycle left shown True.
    defer = RenderDefer()
    defer.shown = True
    defer.submit(lambda: None)
    assert not defer.ready_to_fire(), "a new submit skipped the cue frame"
