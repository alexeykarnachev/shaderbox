"""The recorder: turns a harness run into log records without the driver logging anything.

A `StationRecorder` is attached to a copilot session as a trace listener. It hears every
engine event as the structured objects, folds one turn's worth (the user text, each request's
usage, each tool call, the gates, the terminal, the per-request `context_breakdown`) into an
accumulator, and the harness's `dump()` hands it the turn's user-visible reply and new renders
to write the `turn` record. Context records are written as each request's billed usage arrives,
joined to the breakdown estimated for that request; a request whose stream never finished is
flushed without billing when the turn is recorded.

The pointer file `dogfood_station.json` in the project dir carries the experiment and attempt
across the harness's one-process-per-turn shape: a resumed harness finds it and keeps recording
into the same attempt. An attempt's `sha` is the repo HEAD at its start, and every commit
between the previous attempt's sha and this one lands as a `fix` record, so an attempt page can
say what changed beneath it.
"""

import json
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from dogfood.report.log import (
    LOG_NAME,
    MODES,
    EventLog,
    load_experiment,
)
from scripts.dogfood.analyze import TERMINAL_KINDS

DOGFOOD_ROOT = Path(__file__).resolve().parent.parent
STORE = DOGFOOD_ROOT / "runs"
POINTER_NAME = "dogfood_station.json"
MEDIA_DIR = "media"

_HEAVY_KINDS: frozenset[str] = frozenset(
    {"llm_request", "turn_start", "conversation_loaded"}
)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _git(repo: Path, *args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return out.stdout.strip()


def commits_between(
    repo: Path, since_sha: str, until: str = "HEAD"
) -> list[dict[str, str]]:
    """The commits `since_sha..until`, oldest first, as {sha, subject, body}."""
    if not since_sha:
        return []
    raw = _git(
        repo, "log", "--reverse", "--format=%H%x1f%s%x1f%b%x1e", f"{since_sha}..{until}"
    )
    fixes: list[dict[str, str]] = []
    for chunk in raw.split("\x1e"):
        chunk = chunk.strip()
        if not chunk:
            continue
        sha, subject, body = [*chunk.split("\x1f", 2), "", ""][:3]
        fixes.append({"sha": sha, "subject": subject, "body": body.strip()})
    return fixes


class _TurnAccumulator:
    def __init__(self, n: int, user_text: str) -> None:
        self.n = n
        self.user_text = user_text
        self.started = time.monotonic()
        self.iterations: list[dict[str, Any]] = []
        self.tool_calls: list[dict[str, Any]] = []
        self.gates: list[dict[str, Any]] = []
        self.events: list[str] = []
        self.terminal = ""
        self.cutoff = ""
        self.reply = ""
        self.pending_breakdowns: dict[int, dict[str, Any]] = {}

    def usage_total(self) -> dict[str, Any]:
        keys = ("input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens")
        total: dict[str, Any] = {
            k: sum(int(i.get(k, 0)) for i in self.iterations) for k in keys
        }
        total["cost_usd"] = sum(float(i.get("cost_usd", 0.0)) for i in self.iterations)
        return total


class StationRecorder:
    def __init__(
        self,
        store: Path,
        experiment_id: str,
        attempt: int,
        project_dir: Path,
        *,
        repo_root: Path | None = None,
    ) -> None:
        self.store = store
        self.experiment_id = experiment_id
        self.attempt = attempt
        self.project_dir = project_dir
        self.repo_root = repo_root if repo_root is not None else DOGFOOD_ROOT.parent
        self.log = EventLog(store / experiment_id / LOG_NAME, experiment_id)
        self._lock = threading.Lock()
        self._turn: _TurnAccumulator | None = None
        self._next_turn = self._count_next_turn()

    # ---- lifecycle ----

    @classmethod
    def start_experiment(
        cls,
        project_dir: Path,
        experiment_id: str,
        *,
        intent: str,
        mode: str,
        model: str,
        criteria: list[str] | tuple[str, ...] = (),
        store: Path = STORE,
        repo_root: Path | None = None,
    ) -> "StationRecorder":
        if mode not in MODES:
            raise ValueError(f"mode {mode!r} is not one of {MODES}")
        if (store / experiment_id / LOG_NAME).exists():
            raise ValueError(
                f"experiment {experiment_id!r} already exists; start_attempt() to add an attempt"
            )
        log = EventLog(store / experiment_id / LOG_NAME, experiment_id)
        log.append(
            "experiment_start",
            0,
            {"intent": intent, "mode": mode, "criteria": list(criteria)},
        )
        return cls.start_attempt(
            project_dir, experiment_id, model=model, store=store, repo_root=repo_root
        )

    @classmethod
    def start_attempt(
        cls,
        project_dir: Path,
        experiment_id: str,
        *,
        model: str,
        store: Path = STORE,
        repo_root: Path | None = None,
    ) -> "StationRecorder":
        """Open attempt N+1 of an existing experiment, recording every commit landed since
        attempt N started as a `fix`."""
        exp_dir = store / experiment_id
        if not (exp_dir / LOG_NAME).exists():
            raise ValueError(f"no experiment {experiment_id!r} under {store}")
        exp = load_experiment(exp_dir)
        repo = repo_root if repo_root is not None else DOGFOOD_ROOT.parent
        previous = exp.attempts[-1] if exp.attempts else None
        n = (previous.n if previous else 0) + 1
        log = EventLog(exp_dir / LOG_NAME, experiment_id)
        if previous is not None and previous.sha:
            for fix in commits_between(repo, previous.sha):
                log.append("fix", n, fix)
        log.append(
            "attempt_start",
            n,
            {
                "model": model,
                "sha": _git(repo, "rev-parse", "HEAD"),
                "dirty": bool(_git(repo, "status", "--porcelain")),
                "project_dir": str(project_dir),
            },
        )
        rec = cls(store, experiment_id, n, project_dir, repo_root=repo)
        rec._write_pointer()
        return rec

    @classmethod
    def resume(
        cls, project_dir: Path, *, repo_root: Path | None = None
    ) -> "StationRecorder | None":
        pointer = project_dir / POINTER_NAME
        if not pointer.exists():
            return None
        data = json.loads(pointer.read_text(encoding="utf-8"))
        return cls(
            Path(data["store"]),
            str(data["experiment_id"]),
            int(data["attempt"]),
            project_dir,
            repo_root=repo_root,
        )

    def _write_pointer(self) -> None:
        self.project_dir.mkdir(parents=True, exist_ok=True)
        (self.project_dir / POINTER_NAME).write_text(
            json.dumps(
                {
                    "store": str(self.store),
                    "experiment_id": self.experiment_id,
                    "attempt": self.attempt,
                }
            ),
            encoding="utf-8",
        )

    def _count_next_turn(self) -> int:
        # A turn whose process died before dump() left context records but no turn record; its
        # number must not be reused or the next turn would inherit that context.
        exp = load_experiment(self.store / self.experiment_id)
        attempt = exp.attempt(self.attempt)
        if attempt is None:
            return 1
        highest = max(
            [t.n for t in attempt.turns] + [c.turn for c in attempt.orphan_contexts],
            default=0,
        )
        return highest + 1

    @property
    def attempt_page(self) -> Path:
        return self.store / self.experiment_id / f"attempt_{self.attempt}.html"

    # ---- the trace listener (worker thread) ----

    def on_trace(self, kind: str, fields: dict[str, Any]) -> None:
        with self._lock:
            if kind == "turn_start":
                self._turn = _TurnAccumulator(
                    self._next_turn, str(fields.get("user_text", ""))
                )
                self._next_turn += 1
                return
            turn = self._turn
            if turn is None:
                return
            if kind == "context_breakdown":
                bd = _jsonable(fields.get("breakdown"))
                if isinstance(bd, dict):
                    turn.pending_breakdowns[int(bd.get("iteration", 0))] = bd
            elif kind == "llm_response":
                usage = _jsonable(fields.get("usage")) or {}
                iteration = int(fields.get("iteration", 0))
                turn.iterations.append(
                    {
                        "iteration": iteration,
                        "finish_reason": str(fields.get("finish_reason", "")),
                        "text_chars": len(str(fields.get("text", "") or "")),
                        "tool_calls": [
                            str(tc.get("name", ""))
                            for tc in fields.get("tool_calls", []) or []
                        ],
                        **usage,
                    }
                )
                bd = turn.pending_breakdowns.pop(iteration, None)
                if bd is not None:
                    self._write_context(turn, bd, usage)
            elif kind == "tool_call":
                turn.tool_calls.append(
                    {
                        "n": int(fields.get("n", len(turn.tool_calls) + 1)),
                        "name": str(fields.get("name", "")),
                        "args": _jsonable(fields.get("args")),
                        "ok": bool(fields.get("ok", False)),
                        "result": str(fields.get("result", "")),
                        "payload": _jsonable(fields.get("payload")),
                    }
                )
            elif kind == "gate_open":
                turn.gates.append(
                    {
                        "name": str(fields.get("name", "")),
                        "prompt": str(fields.get("prompt", "")),
                        "answer": "open",
                    }
                )
            elif kind in ("gate_approved", "gate_declined", "gate_cancelled"):
                for gate in reversed(turn.gates):
                    if gate["name"] == fields.get("name") and gate["answer"] == "open":
                        gate["answer"] = kind.removeprefix("gate_")
                        break
            if kind in TERMINAL_KINDS or kind in ("turn_cancelled", "stream_torn"):
                turn.terminal = kind
                turn.cutoff = str(fields.get("cutoff", "") or "")
                turn.reply = str(fields.get("reply", "") or "")
            if kind not in _HEAVY_KINDS and kind not in (
                "llm_response",
                "context_breakdown",
                "tool_call",
            ):
                turn.events.append(kind)

    def _write_context(
        self, turn: _TurnAccumulator, bd: dict[str, Any], billed: dict[str, Any] | None
    ) -> None:
        payload = {"turn": turn.n, **bd, "billed": billed or None}
        self.log.append("context", self.attempt, payload)

    # ---- main thread ----

    def record_turn(
        self,
        *,
        assistant_text: str,
        renders: list[Path],
        renders_root: Path,
    ) -> dict[str, Any] | None:
        """Write the `turn` record for the turn the listener accumulated; None when no turn ran
        since the last record. Renders are copied into the experiment's media dir and recorded
        by path relative to the experiment dir, the path the attempt page references."""
        with self._lock:
            turn = self._turn
            self._turn = None
        if turn is None:
            return None
        for iteration, bd in sorted(turn.pending_breakdowns.items()):
            _ = iteration
            self._write_context(turn, bd, None)
        payload: dict[str, Any] = {
            "n": turn.n,
            "user_text": turn.user_text,
            "assistant_text": assistant_text or turn.reply,
            "iterations": turn.iterations,
            "tool_calls": turn.tool_calls,
            "usage": turn.usage_total(),
            "renders": [self._copy_render(turn.n, p, renders_root) for p in renders],
            "gates": turn.gates,
            "terminal": turn.terminal,
            "cutoff": turn.cutoff,
            "events": turn.events,
            "duration_s": round(time.monotonic() - turn.started, 1),
        }
        self.log.append("turn", self.attempt, payload)
        return payload

    def flush(self) -> None:
        """Write what a dying process still holds (the kill-persist path): the accumulated turn
        as a record marked interrupted, so the log shows the turn ran even without a dump."""
        with self._lock:
            turn = self._turn
        if turn is None:
            return
        turn.terminal = turn.terminal or "interrupted"
        self.record_turn(assistant_text="", renders=[], renders_root=Path())

    def _copy_render(
        self, turn_n: int, src: Path, renders_root: Path
    ) -> dict[str, Any]:
        media = self.store / self.experiment_id / MEDIA_DIR / str(self.attempt)
        media.mkdir(parents=True, exist_ok=True)
        try:
            label = src.relative_to(renders_root).as_posix()
        except ValueError:
            label = src.name
        dst = media / f"t{turn_n:03d}_{src.name}"
        shutil.copy2(src, dst)
        return {
            "path": f"{MEDIA_DIR}/{self.attempt}/{dst.name}",
            "label": label,
            "kind": "video"
            if src.suffix.lower() in (".webm", ".mp4", ".mov")
            else "image",
        }

    def note(self, text: str, *, axis: str = "", turn: int | None = None) -> None:
        self.log.append(
            "note", self.attempt, {"text": text, "axis": axis, "turn": turn}
        )

    def end_attempt(self, outcome: str, summary: str = "") -> None:
        self.log.append(
            "attempt_end", self.attempt, {"outcome": outcome, "summary": summary}
        )
