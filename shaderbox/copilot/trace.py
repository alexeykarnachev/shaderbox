import json
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime
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
        self._fh = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("a", encoding="utf-8", buffering=1)  # line-buffered
            logger.info(f"copilot trace -> {path}")
        except OSError as e:
            # Tracing must never break a turn; degrade to a no-op writer.
            logger.warning(f"copilot trace disabled (cannot open {path}): {e}")

    def event(self, kind: str, **fields: Any) -> None:
        if self._fh is None:
            return
        record = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "kind": kind,
            **{k: _jsonable(v) for k, v in fields.items()},
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            try:
                self._fh.write(line + "\n")
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
    # The agent loop's default when no trace sink is supplied (tests). event() no-ops.
    def __init__(self) -> None:
        self._fh = None
        self._lock = threading.Lock()
        self._path = Path()


NULL_TRACE = _NullTraceLog()
