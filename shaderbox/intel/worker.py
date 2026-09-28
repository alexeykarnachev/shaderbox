"""The one thread that talks to jedi (078 D10).

jedi is not safe to call from two threads at once, so every Python request goes through this
worker and the frame thread never constructs a `Script` itself; the warm-up is the first job
queued, not a call on the frame. A request carries the editor revision and cursor it was made
for; the reader drops a result whose stamp is no longer current. The newest request of each
kind replaces an older one still waiting, so a burst of keystrokes costs one jedi call.
"""

import queue
import threading
from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path

from shaderbox.intel.python import (
    SpanSymbol,
    python_completions,
    python_lookup,
    python_spans,
)
from shaderbox.intel.symbols import Symbol


class PythonRequestKind(StrEnum):
    COMPLETE = auto()
    LOOKUP = auto()
    WARM = auto()
    SPANS = auto()


@dataclass(frozen=True)
class PythonRequest:
    kind: PythonRequestKind
    path: Path
    text: str
    line: int
    column: int
    revision: int
    # Ctrl+N / Ctrl+P as opposed to a keystroke; carried through so the offer keeps its
    # asked-for-ness when the answer lands a frame later.
    explicit: bool = False

    def matches(self, path: Path, revision: int, line: int, column: int) -> bool:
        return (
            self.matches_text(path, revision)
            and self.line == line
            and self.column == column
        )

    def matches_text(self, path: Path, revision: int) -> bool:
        """The caret-blind predicate (105 D4). A completion is an answer about a CARET, so a
        moved caret invalidates it; spans are a property of the TEXT, and dropping a span set
        because the caret moved would throw away a correct answer on every arrow key."""
        return self.path == path and self.revision == revision


@dataclass(frozen=True)
class PythonResult:
    request: PythonRequest
    symbols: tuple[Symbol, ...]
    # A SPANS answer, which `symbols` cannot carry: `Symbol` is name-keyed and has no
    # position, and a span set is exactly the thing a name cannot express.
    spans: tuple[SpanSymbol, ...] = ()


class PythonWorker:
    def __init__(self) -> None:
        self._pending: dict[PythonRequestKind, PythonRequest] = {}
        self._wake = threading.Condition()
        self._results: queue.Queue[PythonResult] = queue.Queue()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, name="shaderbox-intel-python", daemon=True
        )
        self._thread.start()

    def submit(self, request: PythonRequest) -> None:
        with self._wake:
            self._pending[request.kind] = request
            self._wake.notify()

    def poll(self) -> list[PythonResult]:
        """Every result that landed since the last poll, oldest first."""
        found: list[PythonResult] = []
        while True:
            try:
                found.append(self._results.get_nowait())
            except queue.Empty:
                return found

    def close(self) -> None:
        with self._wake:
            self._closed = True
            self._wake.notify()

    def _next(self) -> PythonRequest | None:
        with self._wake:
            while not self._pending and not self._closed:
                self._wake.wait()
            if self._closed:
                return None
            # SPANS goes LAST (105 D3a). Insertion order would let a spans job sit in front
            # of the completion job for the same burst, and MEASURED they cost the same --
            # 8.17 ms against 8.19 ms on the flock script -- so a shipped feature's latency
            # would double on every keystroke that also asked for colour. Completion is what
            # the user is waiting on; colour is not.
            kind = next(
                (k for k in self._pending if k != PythonRequestKind.SPANS),
                next(iter(self._pending)),
            )
            return self._pending.pop(kind)

    def _run(self) -> None:
        while (request := self._next()) is not None:
            if request.kind == PythonRequestKind.COMPLETE:
                symbols = tuple(
                    python_completions(request.text, request.line, request.column)
                )
            elif request.kind == PythonRequestKind.LOOKUP:
                found = python_lookup(request.text, request.line, request.column)
                symbols = (found,) if found is not None else ()
            elif request.kind == PythonRequestKind.SPANS:
                self._results.put(
                    PythonResult(request, (), spans=python_spans(request.text))
                )
                continue
            else:
                python_completions(request.text, request.line, request.column)
                continue
            self._results.put(PythonResult(request, symbols))
