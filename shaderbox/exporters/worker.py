"""The exporter worker half: one background thread draining a job queue, pushing
progress/events back to the main thread.

Both network exporters need the identical machinery — a lazily-spawned worker, a bounded
job queue, a bounded progress queue read by the UI each frame, and a teardown that stops
the worker without ever blocking process exit. `TelegramExporter` layers an asyncio loop on
top (python-telegram-bot forces async); `YouTubeExporter` calls its blocking client
directly. That difference lives in the subclass's `_worker_main`, not here.

Teardown contract (`conventions.md`): push STOP to release the worker, `join(timeout)`, and
on a timeout ABANDON the survivor rather than block shutdown forever. Abandoning is only
safe because the thread is a **daemon** — a non-daemon survivor is re-joined by interpreter
`_shutdown` and hangs the process for as long as it stays blocked.
"""

import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass

from loguru import logger

QUEUE_MAXSIZE = 128
DRAIN_TIMEOUT_SEC = 5.0


@dataclass(frozen=True)
class _Stop:
    """The STOP marker. A distinct type (not a magic string) so `next_job` separates it
    from a real job by type rather than by comparing against a value a Job could equal.

    It carries the generation it was addressed to. A `stop()` whose join times out leaves
    its STOP behind in the queue; without the stamp the NEXT worker consumes that leftover
    and exits immediately, stranding the job the user just asked for (a click that does
    nothing, and an `in_flight` spinner nothing will ever clear)."""

    generation: int


class ExporterWorker[Job, Event]:
    """Owns one worker thread plus the two queues that bracket it.

    `label` names the thread and its log lines. `run` is the worker body; it is called on
    the thread and should loop on `next_job()` until that returns None (the STOP sentinel).
    """

    def __init__(self, label: str) -> None:
        self._label = label
        self._jobs: queue.Queue[Job | _Stop] = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._events: queue.Queue[Event] = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        # Bumped by every stop(). A worker only honours a STOP stamped with the generation
        # it was started under, so a leftover from an abandoned predecessor is skipped.
        self._generation = 0

    # ------------------------------------------------------------------ main thread
    def ensure(self, run: Callable[[], None]) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = threading.Thread(
                target=run,
                name=f"{self._label}-worker",
                daemon=True,
            )
            self._thread.start()

    def submit(self, job: Job, run: Callable[[], None]) -> bool:
        """Enqueue a job, spawning the worker if needed. False = dropped (queue full)."""
        self.ensure(run)
        try:
            self._jobs.put_nowait(job)
            return True
        except queue.Full:
            logger.warning(f"{self._label} job queue full; dropping job")
            return False

    def poll_event(self) -> Event | None:
        try:
            return self._events.get_nowait()
        except queue.Empty:
            return None

    def stop(self) -> threading.Thread | None:
        """Signal STOP and join with a bound. Returns the thread if it had to be abandoned."""
        with self._lock:
            thread = self._thread
            if thread is None or not thread.is_alive():
                self._thread = None
                return None
            self._jobs.put(_Stop(generation=self._generation))
            self._generation += 1
            thread.join(timeout=DRAIN_TIMEOUT_SEC)
            self._thread = None
            if thread.is_alive():
                return thread
            return None

    # ---------------------------------------------------------------- worker thread
    def next_job(self) -> Job | None:
        """Block for the next job. None = STOP for THIS worker; the body must return.
        A STOP addressed to an earlier generation is a leftover from an abandoned worker —
        skip it, or it would kill this one and strand whatever the user just queued."""
        while True:
            job = self._jobs.get()
            if not isinstance(job, _Stop):
                return job
            with self._lock:
                current = self._generation
            if job.generation >= current:
                return None
            logger.debug(
                f"{self._label} skipping a stale STOP from an abandoned worker"
            )

    def push_progress(self, event: Event) -> None:
        """Progress is LOSSY-NEWEST: a full queue drops the oldest so the UI still
        advances rather than freezing on a stale fraction."""
        try:
            self._events.put_nowait(event)
        except queue.Full:
            try:
                self._events.get_nowait()
                self._events.put_nowait(event)
            except (queue.Empty, queue.Full):
                pass

    def push_event(self, event: Event) -> None:
        """A state/terminal event — the kind that clears `in_flight` and moves the auth
        state. These must NOT be droppable: losing one leaves the UI spinning on a job that
        already finished, with nothing left to clear it.

        Dropping a progress item to make room is always the better trade, and the queue is
        full precisely BECAUSE push_progress keeps it full — so "the queue is rarely full"
        would be exactly backwards as a reason to drop the important half."""
        while True:
            try:
                self._events.put_nowait(event)
                return
            except queue.Full:
                try:
                    self._events.get_nowait()
                except queue.Empty:
                    # Drained by the UI between the failed put and this get; retry.
                    continue
