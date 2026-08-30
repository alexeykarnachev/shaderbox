import queue
import threading
from dataclasses import dataclass, field
from enum import StrEnum, auto

from shaderbox.copilot.capabilities import DocumentImportResult, MediaBindResult

# Worker->main marshalling seam for USER answers (confirm/credential): the worker pushes a
# request + Event and blocks; the main thread draws the widget, fills the response slot, sets
# the event, worker unblocks. Sibling of `bridge.py`, the same round-trip for GL WORK.


class GateKind(StrEnum):
    CONFIRM = auto()  # a Yes/No confirmation
    CREDENTIAL = auto()  # an inline secret field for a missing integration key
    CONFIG = auto()  # an inline integration setup panel (reuses the exporter's draw_config_ui) + Cancel
    FILE = (
        auto()
    )  # a native OS file picker (feature 052); its own channel slot, drawn UI-side


@dataclass(frozen=True)
class GateRequest:
    kind: GateKind
    prompt: str
    secret_field: str = ""  # CREDENTIAL: which integration key
    # FILE gate targeting: the action ("" = bind_media; "import_document"), the bind target
    # (document_id/uniform), the pickable kinds ("image"/"video"/"glsl"), and switch_to for import.
    document_id: str = ""
    uniform: str = ""
    file_kinds: tuple[str, ...] = ()
    file_action: str = ""
    switch_to: bool = False


@dataclass(frozen=True)
class GateResponse:
    approved: bool = False
    secret: str = ""  # CREDENTIAL: typed key — never logged/traced/persisted
    cancelled: bool = False  # the wait was released without an answer
    media_result: MediaBindResult | None = None  # FILE bind: the path-free bind outcome
    import_result: DocumentImportResult | None = (
        None  # FILE import: the path-free import outcome
    )


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
        # Serializes publish against cancel_all's release sweep. `_generation` is bumped by every
        # sweep: a worker samples it before building its request and re-checks under the lock, so a
        # cancel that landed in between cancels this request too. Without that, a slot published
        # just after a sweep waits on an event nobody will ever set — and under
        # cancel_all(reusable=True) (the Stop button) there is no _shutdown latch to catch it, so
        # the copilot stays latched and every later turn is queued and never runs.
        self._lock: threading.Lock = threading.Lock()
        self._generation: int = 0
        # FILE gate: a SEPARATE slot (feature 052). A FILE gate is raised mid-`execute` (no
        # AgentGateOpened yield), so it can't ride the CONFIRM/CREDENTIAL `_pending`/`take_pending`
        # path (which pump_events drives off that yield); the UI polls this slot every frame instead.
        self._file_pending: queue.Queue[_GatePending] = queue.Queue()
        self._file_current: _GatePending | None = None

    def reopen(self) -> None:
        # MAIN THREAD. Clears a `_shutdown` latched by a non-reusable cancel_all() so a
        # reused channel serves again.
        self._shutdown.clear()

    def ask(self, request: GateRequest) -> GateResponse:
        # WORKER THREAD. Enqueue + block until the UI answers or cancel fires.
        with self._lock:
            generation = self._generation
        pending = _GatePending(request=request)
        with self._lock:
            if self._shutdown.is_set() or generation != self._generation:
                return GateResponse(cancelled=True)
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

    def ask_file(self, request: GateRequest) -> GateResponse:
        # WORKER THREAD (feature 052). Enqueue a FILE request onto its own slot + block until the UI
        # poll answers or cancel fires.
        with self._lock:
            generation = self._generation
        pending = _GatePending(request=request)
        with self._lock:
            if self._shutdown.is_set() or generation != self._generation:
                return GateResponse(cancelled=True)
            self._file_current = pending
            self._file_pending.put(pending)
        pending.done.wait()
        return pending.response

    def take_file_pending(self) -> GateRequest | None:
        # MAIN THREAD (feature 052). The next FILE request awaiting the OS picker, or None.
        try:
            return self._file_pending.get_nowait().request
        except queue.Empty:
            return None

    def file_gate_active(self) -> bool:
        # MAIN THREAD (feature 052). True while a worker is blocked on a FILE pick. Goes False the
        # instant cancel_all fires (Stop/reset) — the UI poll checks this so a dialog still open when
        # the turn was cancelled is ABANDONED (its late pick never mutates a document, never mis-wires the
        # next turn's gate).
        return self._file_current is not None

    def answer_file(self, response: GateResponse) -> None:
        # MAIN THREAD (feature 052). Fill the current FILE pending + unblock the worker. A late answer
        # after cancel_all is a no-op (the guard below), so a pick that lands after Stop is dropped.
        pending = self._file_current
        if pending is None:
            return
        pending.response = response
        pending.done.set()
        self._file_current = None

    def cancel_all(self, *, reusable: bool = False) -> None:
        # MAIN THREAD. Release every wait with cancelled=True. `reusable=True` leaves the
        # channel live; the default latches `_shutdown` so a late ask() can't block.
        with self._lock:
            self._generation += 1
            if not reusable:
                self._shutdown.set()
            for slot in ("_current", "_file_current"):
                pending = getattr(self, slot)
                if pending is not None:
                    pending.response = GateResponse(cancelled=True)
                    pending.done.set()
                    setattr(self, slot, None)
            for q in (self._pending, self._file_pending):
                while True:
                    try:
                        pending = q.get_nowait()
                    except queue.Empty:
                        break
                    pending.response = GateResponse(cancelled=True)
                    pending.done.set()
