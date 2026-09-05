"""The event log: one JSON object per line, appended as it happens, never rewritten.

The log is the source of truth for a run; every page of the site is regenerated from it. So the
writer's only job is to make each append land whole and in order under the harness's
one-process-per-turn shape (each turn is its own process opening and appending), and the reader's
only job is to give back every complete record even when the last one was torn by a kill.

Every record carries `experiment_id`, `attempt`, `ts` (UTC with an explicit offset), `kind` and a
`payload`; `KINDS` is the closed vocabulary. `reconstruct` folds a record list into the typed
`Experiment` / `Attempt` / `TurnRecord` tree the site builder walks.
"""

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

KINDS: tuple[str, ...] = (
    "experiment_start",
    "attempt_start",
    "turn",
    "context",
    "fix",
    "attempt_end",
    "note",
)

MODES: tuple[str, ...] = ("end_to_end", "babysat", "free_run")

LOG_NAME = "events.jsonl"


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class Event:
    experiment_id: str
    attempt: int
    ts: str
    kind: str
    payload: dict[str, Any]


class EventLog:
    """Append-only writer for one experiment's `events.jsonl`.

    Each append is one `write()` of one line onto a file opened for append and closed again, so
    two processes (or threads) appending at once interleave whole lines, never bytes. A torn
    predecessor (a line with no trailing newline, left by a killed process) is repaired by
    prefixing the next line with a newline, so the torn fragment stays its own unparseable line
    and the reader skips exactly it.
    """

    def __init__(self, path: Path, experiment_id: str) -> None:
        self.path = path
        self.experiment_id = experiment_id
        self._lock = threading.Lock()

    def append(self, kind: str, attempt: int, payload: dict[str, Any]) -> Event:
        if kind not in KINDS:
            raise ValueError(f"unknown event kind {kind!r}; one of {KINDS}")
        event = Event(self.experiment_id, attempt, utc_now(), kind, dict(payload))
        line = json.dumps(asdict(event), ensure_ascii=False, default=str)
        data = line.encode("utf-8") + b"\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not _ends_with_newline(self.path):
                data = b"\n" + data
            with self.path.open("ab") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
        return event


def _ends_with_newline(path: Path) -> bool:
    # An absent or empty file needs no repair.
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return True
    if size == 0:
        return True
    with path.open("rb") as fh:
        fh.seek(size - 1)
        return fh.read(1) == b"\n"


def read_events(path: Path) -> tuple[list[Event], list[str]]:
    """Every complete record in file order, plus one warning per line that could not be read
    (a torn tail, a hand edit). Never raises on content: a damaged line costs that line only."""
    events: list[Event] = []
    warnings: list[str] = []
    if not path.exists():
        return events, warnings
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                events.append(
                    Event(
                        experiment_id=str(obj["experiment_id"]),
                        attempt=int(obj["attempt"]),
                        ts=str(obj["ts"]),
                        kind=str(obj["kind"]),
                        payload=dict(obj["payload"]),
                    )
                )
            except (ValueError, KeyError, TypeError) as exc:
                warnings.append(f"{path.name}:{lineno}: unreadable record ({exc})")
    return events, warnings


# ---- the reconstructed run ----


@dataclass
class TurnRecord:
    n: int
    ts: str
    payload: dict[str, Any]
    contexts: list["ContextRecord"] = field(default_factory=list)

    @property
    def user_text(self) -> str:
        return str(self.payload.get("user_text", ""))

    @property
    def assistant_text(self) -> str:
        return str(self.payload.get("assistant_text", ""))

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        return list(self.payload.get("tool_calls", []))

    @property
    def iterations(self) -> list[dict[str, Any]]:
        return list(self.payload.get("iterations", []))

    @property
    def renders(self) -> list[dict[str, Any]]:
        return list(self.payload.get("renders", []))

    @property
    def gates(self) -> list[dict[str, Any]]:
        return list(self.payload.get("gates", []))

    @property
    def usage(self) -> dict[str, Any]:
        return dict(self.payload.get("usage", {}))

    @property
    def cost_usd(self) -> float:
        return float(self.usage.get("cost_usd", 0.0))

    @property
    def terminal(self) -> str:
        return str(self.payload.get("terminal", ""))

    @property
    def cutoff(self) -> str:
        return str(self.payload.get("cutoff", ""))

    @property
    def peak_input_tokens(self) -> int:
        return max((int(i.get("input_tokens", 0)) for i in self.iterations), default=0)


@dataclass
class ContextRecord:
    turn: int
    iteration: int
    ts: str
    payload: dict[str, Any]

    @property
    def blocks(self) -> list[dict[str, Any]]:
        return list(self.payload.get("blocks", []))

    @property
    def est_total_tokens(self) -> int:
        return int(self.payload.get("est_total_tokens", 0))

    @property
    def billed(self) -> dict[str, Any] | None:
        billed = self.payload.get("billed")
        return dict(billed) if isinstance(billed, dict) else None


@dataclass
class NoteRecord:
    ts: str
    text: str
    axis: str
    turn: int | None


@dataclass
class FixRecord:
    sha: str
    subject: str
    body: str


@dataclass
class Attempt:
    n: int
    started: str
    model: str = ""
    sha: str = ""
    ended: str = ""
    outcome: str = ""
    summary: str = ""
    fixes: list[FixRecord] = field(default_factory=list)
    turns: list[TurnRecord] = field(default_factory=list)
    notes: list[NoteRecord] = field(default_factory=list)
    # Context breakdowns that arrived for a turn whose `turn` record never landed (a killed
    # process): kept so a live attempt's page still shows what the copilot was sent.
    orphan_contexts: list[ContextRecord] = field(default_factory=list)

    @property
    def live(self) -> bool:
        return not self.ended

    @property
    def cost_usd(self) -> float:
        return sum(t.cost_usd for t in self.turns)

    @property
    def last_activity(self) -> str:
        stamps = [self.started, self.ended]
        stamps.extend(t.ts for t in self.turns)
        stamps.extend(n.ts for n in self.notes)
        stamps.extend(c.ts for c in self.orphan_contexts)
        return max(s for s in stamps if s)


@dataclass
class Experiment:
    id: str
    intent: str = ""
    mode: str = ""
    criteria: list[str] = field(default_factory=list)
    started: str = ""
    attempts: list[Attempt] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def cost_usd(self) -> float:
        return sum(a.cost_usd for a in self.attempts)

    @property
    def live(self) -> bool:
        return any(a.live for a in self.attempts)

    @property
    def last_activity(self) -> str:
        stamps = [self.started, *(a.last_activity for a in self.attempts)]
        return max((s for s in stamps if s), default="")

    def attempt(self, n: int) -> Attempt | None:
        return next((a for a in self.attempts if a.n == n), None)


def reconstruct(events: list[Event], experiment_id: str = "") -> Experiment:
    """Fold a record list into one experiment. `experiment_id` names the experiment when the
    list is empty (a store directory with a damaged log); otherwise the records name it."""
    exp = Experiment(id=experiment_id)
    attempts: dict[int, Attempt] = {}
    pending_contexts: dict[int, list[ContextRecord]] = {}

    def attempt_for(n: int, ts: str) -> Attempt:
        if n not in attempts:
            attempts[n] = Attempt(n=n, started=ts)
        return attempts[n]

    for ev in events:
        if not exp.id:
            exp.id = ev.experiment_id
        p = ev.payload
        if ev.kind == "experiment_start":
            exp.intent = str(p.get("intent", ""))
            exp.mode = str(p.get("mode", ""))
            exp.criteria = [str(c) for c in p.get("criteria", [])]
            exp.started = exp.started or ev.ts
        elif ev.kind == "attempt_start":
            a = attempt_for(ev.attempt, ev.ts)
            a.started = ev.ts
            a.model = str(p.get("model", ""))
            a.sha = str(p.get("sha", ""))
        elif ev.kind == "fix":
            attempt_for(ev.attempt, ev.ts).fixes.append(
                FixRecord(
                    sha=str(p.get("sha", "")),
                    subject=str(p.get("subject", "")),
                    body=str(p.get("body", "")),
                )
            )
        elif ev.kind == "context":
            rec = ContextRecord(
                turn=int(p.get("turn", 0)),
                iteration=int(p.get("iteration", 0)),
                ts=ev.ts,
                payload=p,
            )
            pending_contexts.setdefault(ev.attempt, []).append(rec)
        elif ev.kind == "turn":
            a = attempt_for(ev.attempt, ev.ts)
            n = int(p.get("n", len(a.turns) + 1))
            turn = TurnRecord(n=n, ts=ev.ts, payload=p)
            waiting = pending_contexts.get(ev.attempt, [])
            turn.contexts = [c for c in waiting if c.turn == n]
            pending_contexts[ev.attempt] = [c for c in waiting if c.turn != n]
            a.turns.append(turn)
        elif ev.kind == "note":
            raw_turn = p.get("turn")
            attempt_for(ev.attempt, ev.ts).notes.append(
                NoteRecord(
                    ts=ev.ts,
                    text=str(p.get("text", "")),
                    axis=str(p.get("axis", "")),
                    turn=int(raw_turn) if raw_turn is not None else None,
                )
            )
        elif ev.kind == "attempt_end":
            a = attempt_for(ev.attempt, ev.ts)
            a.ended = ev.ts
            a.outcome = str(p.get("outcome", ""))
            a.summary = str(p.get("summary", ""))
        else:
            exp.warnings.append(f"unknown record kind {ev.kind!r} at {ev.ts}")

    for n, leftovers in pending_contexts.items():
        if leftovers:
            attempt_for(n, leftovers[0].ts).orphan_contexts.extend(leftovers)
    exp.attempts = [attempts[k] for k in sorted(attempts)]
    return exp


def load_experiment(exp_dir: Path) -> Experiment:
    events, warnings = read_events(exp_dir / LOG_NAME)
    exp = reconstruct(events, experiment_id=exp_dir.name)
    exp.warnings = warnings + exp.warnings
    return exp


def load_store(store: Path) -> list[Experiment]:
    """Every experiment under the store, newest activity first."""
    if not store.is_dir():
        return []
    found = [
        load_experiment(d)
        for d in sorted(store.iterdir())
        if d.is_dir() and (d / LOG_NAME).exists()
    ]
    return sorted(found, key=lambda e: e.last_activity, reverse=True)
