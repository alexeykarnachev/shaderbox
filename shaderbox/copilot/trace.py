import json
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime
from io import TextIOWrapper
from pathlib import Path
from typing import Any

from loguru import logger

from shaderbox.paths import app_data_dir

# A DEDICATED full-transcript log for the copilot, separate from the regular app log
# stream (loguru/stderr). Every prompt, wire message, LLM response, tool call + result,
# and turn boundary is written here IN FULL (no truncation) as one JSON object per line
# (.jsonl). This is the 100%-observability surface for debugging the agent; the concise
# `copilot ...` lines in the normal log stay for at-a-glance flow. One TraceLog per
# CopilotSession; appends across turns; flushed per event so a crash loses nothing.
#
# The file is opened LAZILY on the first event() and re-opened if it was closed: the App
# lifecycle calls release() (-> close()) at the top of every _init, which would otherwise
# null the handle the session opened in __init__ and silently drop the whole transcript.


def _jsonable(value: Any) -> Any:
    # Best-effort full serialization — dataclasses (LLMMessage/LLMToolCall/…) -> dicts,
    # everything else falls back to str via the json default. No truncation anywhere.
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


class TraceLog:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()  # the worker thread writes; serialize appends
        self._fh: TextIOWrapper | None = None

    def _ensure_open(self) -> TextIOWrapper | None:
        # Caller holds the lock. Open (append) on first use / after a close().
        if self._fh is not None:
            return self._fh
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self._path.open("a", encoding="utf-8", buffering=1)
            logger.info(f"copilot trace -> {self._path}")
        except OSError as e:
            logger.warning(f"copilot trace disabled (cannot open {self._path}): {e}")
            self._fh = None
        return self._fh

    def event(self, kind: str, **fields: Any) -> None:
        record = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "kind": kind,
            **{k: _jsonable(v) for k, v in fields.items()},
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            fh = self._ensure_open()
            if fh is None:
                return
            try:
                fh.write(line + "\n")
            except OSError as e:
                logger.warning(f"copilot trace write failed: {e}")

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


def new_trace_log(stamp: str) -> TraceLog:
    # `stamp` is passed in (the caller stamps it; keeps TraceLog construction pure).
    return TraceLog(app_data_dir() / "copilot_traces" / f"copilot_{stamp}.jsonl")


class _NullTraceLog(TraceLog):
    # The agent loop's default when no trace sink is supplied (tests). event() no-ops
    # (its path can never be opened).
    def __init__(self) -> None:
        super().__init__(Path())

    def _ensure_open(self) -> TextIOWrapper | None:
        return None


NULL_TRACE = _NullTraceLog()
