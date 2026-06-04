import queue
import threading
from dataclasses import dataclass, field
from enum import StrEnum, auto

# The gate: a worker->UI BLOCKING round-trip — the mirror of CopilotBridge (which is
# worker->main GL). The worker pushes a request + a threading.Event and blocks on it;
# the UI/main thread renders the widget, the user acts, the UI fills the response slot
# and sets the event, the worker unblocks. Same proven primitive as bridge.py, opposite
# direction. See ai_docs/features/020_copilot_agent/11_capability_wave_spec.md §7.
#
# Both gate kinds are wired: CONFIRM (feature 020·17, Yes/No — delete_node + publish) and
# CREDENTIAL (feature 020·19, a masked secret input — set_telegram_token). The agent loop calls
# ask() before a gated tool; pump_events materializes a pending_action Message (stamped with the
# gate_kind) + dequeues via take_pending; the chat draws the matching widget; answer() (CONFIRM)
# or answer_gate_credential() (CREDENTIAL, carries the typed secret) unblocks the worker.


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
    secret: str = (
        ""  # CREDENTIAL: the typed key — out-of-band, never logged/traced/persisted
    )
    cancelled: bool = False  # Stop / window-close / quit released the wait


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
        # MAIN THREAD, at the start of a turn. Clears a `_shutdown` latched by a prior
        # non-reusable cancel_all() (release(), incl. App._init's teardown of the freshly
        # constructed session) so a reused channel serves again — the mirror of bridge.reopen.
        self._shutdown.clear()

    def ask(self, request: GateRequest) -> GateResponse:
        # ON THE WORKER THREAD. Enqueue + block until the UI answers or cancel fires.
        if self._shutdown.is_set():
            return GateResponse(cancelled=True)
        pending = _GatePending(request=request)
        self._current = pending
        self._pending.put(pending)
        pending.done.wait()
        return pending.response

    def take_pending(self) -> GateRequest | None:
        # ON THE MAIN THREAD, per frame. The next request awaiting a UI widget, or None.
        try:
            return self._pending.get_nowait().request
        except queue.Empty:
            return None

    def answer(self, response: GateResponse) -> None:
        # ON THE MAIN THREAD. Fill the current pending's slot + unblock the worker.
        pending = self._current
        if pending is None:
            return
        pending.response = response
        pending.done.set()
        self._current = None

    def cancel_all(self, *, reusable: bool = False) -> None:
        # ON THE MAIN THREAD. Release every wait with cancelled=True. `reusable=True`
        # (reset_conversation / Stop) leaves the channel live for the next turn; the
        # default (release at shutdown) latches it shut so a late ask() can't block.
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
