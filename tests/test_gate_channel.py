"""GateChannel reopen-after-release (the conventions' worker<->main latch rule).

App._init tears down the freshly constructed session via release(), which latches the
gate's _shutdown (non-reusable cancel_all). Without a reopen() at the next turn start,
every ask() short-circuits to cancelled — the confirm card resolves instantly with no
buttons. This drives the real ask() across a worker/main split (a direct run_turn test
misses it)."""

import threading

import pytest

from shaderbox.copilot import gate as gate_module
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


def test_ask_racing_cancel_all_never_blocks_forever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Lost-wakeup (feature 060). ask() used to check _shutdown, then build its pending, then
    # publish it to _current. A cancel_all() landing in THAT window swept two empty slots,
    # released nothing, and latched _shutdown — and the worker went on to block on an event
    # nobody would ever set. The copilot stayed latched: on a project switch _ensure_worker saw
    # the thread still alive, so every later turn was queued and silently never ran.
    # Publishing and checking the latch now happen under one lock, so cancel_all either sweeps
    # the published slot or latches before the slot exists (and ask() returns cancelled).
    #
    # The real window is a few instructions wide, so a plain thread race won't find it. This
    # drives the exact interleaving: cancel_all() is forced to complete while ask() is between
    # its check and its publish. Falsifier: drop the lock from ask()/cancel_all() and this hangs
    # until the join times out with the worker still alive.
    gate = GateChannel()
    answered: list[GateResponse] = []
    fired = threading.Event()
    real_pending = gate_module._GatePending

    def _cancel_mid_ask(*args: object, **kwargs: object) -> object:
        # Runs inside ask(), after the _shutdown check and before the slot is published.
        if not fired.is_set():
            fired.set()
            canceller = threading.Thread(target=gate.cancel_all)
            canceller.start()
            canceller.join(timeout=2.0)
        return real_pending(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(gate_module, "_GatePending", _cancel_mid_ask)

    def _worker() -> None:
        answered.append(gate.ask(GateRequest(kind=GateKind.CONFIRM, prompt="ok?")))

    t = threading.Thread(
        target=_worker, daemon=True
    )  # daemon: a regression must not hang exit
    t.start()
    t.join(timeout=5.0)
    assert not t.is_alive(), (
        "ask() blocked forever — cancel_all swept before the slot existed"
    )
    assert answered and answered[0].cancelled
