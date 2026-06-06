import queue
import threading
from dataclasses import dataclass, field
from enum import StrEnum, auto

# Worker->UI blocking round-trip: the worker pushes a request + Event and blocks; the
# main thread draws the widget, fills the response slot, sets the event, worker unblocks.


class GateKind(StrEnum):
    CONFIRM = auto()  # yes/no or a procedurally-generated option list
    CREDENTIAL = auto()  # an inline secret field for a missing integration key
    CONFIG = auto()  # an inline integration setup panel (reuses the exporter's draw_config_ui) + Cancel


@dataclass(frozen=True)
class GateRequest:
    kind: GateKind
    prompt: str
    options: list[str] = field(default_factory=lambda: ["Yes", "No"])  # CONFIRM
    secret_field: str = ""  # CREDENTIAL: which integration key


@dataclass(frozen=True)
class GateResponse:
    approved: bool = False
    option: str = ""
    secret: str = ""  # CREDENTIAL: typed key — never logged/traced/persisted
    cancelled: bool = False  # the wait was released without an answer


@dataclass
class _GatePending:
    request: GateRequest
    done: threading.Event = field(default_factory=threading.Event)
    response: GateResponse = field(default_factory=GateResponse)


class GateChannel:
    """Worker blocks on ask(); the UI answers via answer(). cancel_all() releases every
    pending wait with cancelled=True (Stop / reset / shutdown)."""

    def __init__(self) -> None:
        self._pending: queue.Queue[_GatePending] = queue.Queue()
        self._current: _GatePending | None = None
        self._shutdown: threading.Event = threading.Event()

    def reopen(self) -> None:
        # MAIN THREAD. Clears a `_shutdown` latched by a non-reusable cancel_all() so a
        # reused channel serves again.
        self._shutdown.clear()

    def ask(self, request: GateRequest) -> GateResponse:
        # WORKER THREAD. Enqueue + block until the UI answers or cancel fires.
        if self._shutdown.is_set():
            return GateResponse(cancelled=True)
        pending = _GatePending(request=request)
        self._current = pending
        self._pending.put(pending)
        pending.done.wait()
        return pending.response

    def take_pending(self) -> GateRequest | None:
        # MAIN THREAD. The next request awaiting a UI widget, or None.
        try:
            return self._pending.get_nowait().request
        except queue.Empty:
            return None

    def answer(self, response: GateResponse) -> None:
        # MAIN THREAD. Fill the current pending's slot + unblock the worker.
        pending = self._current
        if pending is None:
            return
        pending.response = response
        pending.done.set()
        self._current = None

    def cancel_all(self, *, reusable: bool = False) -> None:
        # MAIN THREAD. Release every wait with cancelled=True. `reusable=True` leaves the
        # channel live; the default latches `_shutdown` so a late ask() can't block.
        if not reusable:
            self._shutdown.set()
        if self._current is not None:
            self._current.response = GateResponse(cancelled=True)
            self._current.done.set()
            self._current = None
        while True:
            try:
                pending = self._pending.get_nowait()
            except queue.Empty:
                break
            pending.response = GateResponse(cancelled=True)
            pending.done.set()
