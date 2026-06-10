"""GateChannel reopen-after-release (the conventions' worker<->main latch rule).

App._init tears down the freshly constructed session via release(), which latches the
gate's _shutdown (non-reusable cancel_all). Without a reopen() at the next turn start,
every ask() short-circuits to cancelled — the confirm card resolves instantly with no
buttons. This drives the real ask() across a worker/main split (a direct run_turn test
misses it)."""

import threading

from shaderbox.copilot.gate import GateChannel, GateKind, GateRequest, GateResponse


def test_reopen_after_release_blocks_and_answers() -> None:
    gate = GateChannel()
    gate.cancel_all()  # the latching teardown release() does

    answered: list[GateResponse] = []

    def _worker() -> None:
        answered.append(gate.ask(GateRequest(kind=GateKind.CONFIRM, prompt="ok?")))

    # Without reopen: ask returns cancelled immediately (no block).
    t = threading.Thread(target=_worker)
    t.start()
    t.join(timeout=2.0)
    assert not t.is_alive(), "ask() blocked despite a latched shutdown"
    assert answered and answered[0].cancelled, "latched gate should return cancelled"

    # After reopen: ask blocks until answered (the real turn path). Poll a BOUNDED number
    # of times for the pending request — if reopen didn't take, ask short-circuits and
    # never enqueues, so the loop expires and the assert fails cleanly (never hangs).
    gate.reopen()
    answered.clear()
    t = threading.Thread(target=_worker)
    t.start()
    pending = None
    tick = threading.Event()
    for _ in range(200):  # ~2s at 10ms
        pending = gate.take_pending()
        if pending is not None:
            break
        tick.wait(0.01)
    assert pending is not None, "reopen did not re-arm the gate — ask() short-circuited"
    gate.answer(GateResponse(approved=True, option="Yes"))
    t.join(timeout=2.0)
    assert not t.is_alive(), "ask() never unblocked after reopen + answer"
    assert answered and answered[0].approved and not answered[0].cancelled
