import threading
from dataclasses import is_dataclass
from datetime import datetime
from io import TextIOWrapper
from pathlib import Path
from typing import Any

from loguru import logger

from shaderbox.copilot.llm.api import LLMMessage, LLMToolSpec, LLMUsage
from shaderbox.logging_setup import LOGGING_CONFIG
from shaderbox.paths import copilot_trace_dir

# A DEDICATED full-fidelity transcript for the copilot, separate from the regular log
# stream (loguru). Every prompt, wire message, LLM response, tool call + result, and
# turn boundary is written here IN FULL (no truncation). The ONLY consumer is a human /
# an AI agent reading the file back COLD to debug the copilot flow — so the format is a
# plain-text transcript (real newlines, turn banners, indented blocks), NOT jsonl: a
# 50-line shader reads as a shader, not a `\n`-escaped one-liner, and there are no
# repeated keys per event. One TraceLog per CopilotSession; appends across turns; flushed
# per event so a crash loses nothing.
#
# The file is opened LAZILY on the first event() and re-opened if it was closed: the App
# lifecycle calls release() (-> close()) at the top of every _init, which would otherwise
# null the handle the session opened in __init__ and silently drop the whole transcript.

_INDENT = "    "
_RULE = "=" * 78


def _indent(text: str) -> str:
    return "\n".join(_INDENT + line for line in text.splitlines())


def _render_usage(u: LLMUsage) -> str:
    return f"in={u.input_tokens} out={u.output_tokens} cost=${u.cost_usd:.6f}"


def _render_message(m: LLMMessage) -> str:
    head = f"[{m.role}]"
    if m.tool_call_id is not None:
        head += f" (tool_call_id={m.tool_call_id})"
    parts = [head]
    if m.content:
        parts.append(_indent(m.content))
    if m.tool_calls:
        for tc in m.tool_calls:
            parts.append(_indent(f"-> tool_call {tc.name}(id={tc.id})"))
            parts.append(_indent(_indent(tc.arguments)))
    return "\n".join(parts)


def _render_messages(messages: list[LLMMessage]) -> str:
    return "\n".join(_render_message(m) for m in messages)


def _render_tools(tools: list[LLMToolSpec]) -> str:
    return "\n".join(f"- {t.name}: {t.description}" for t in tools)


def _render_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for tc in tool_calls:
        blocks.append(f"{tc.get('name')}(id={tc.get('id')})")
        blocks.append(_indent(str(tc.get("arguments"))))
    return "\n".join(blocks) if blocks else "(none)"


def _render_value(key: str, value: Any) -> str:
    # Dispatch on the known payload shapes so each renders readably; fall back to str().
    if value is None:
        return f"{key}: (none)"
    if isinstance(value, LLMUsage):
        return f"{key}: {_render_usage(value)}"
    if isinstance(value, list) and value and isinstance(value[0], LLMMessage):
        return f"{key}:\n{_render_messages(value)}"
    if isinstance(value, list) and value and isinstance(value[0], LLMToolSpec):
        return f"{key}:\n{_render_tools(value)}"
    if key in ("tool_calls",) and isinstance(value, list):
        return f"{key}:\n{_indent(_render_tool_calls(value))}"
    if isinstance(value, str) and "\n" in value:
        return f"{key}:\n{_indent(value)}"
    if is_dataclass(value) and not isinstance(value, type):
        return f"{key}: {value}"
    if isinstance(value, dict | list):
        return f"{key}:\n{_indent(str(value))}"
    return f"{key}: {value}"


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
            logger.debug(f"copilot trace -> {self._path}")
        except OSError as e:
            logger.warning(f"copilot trace disabled (cannot open {self._path}): {e}")
            self._fh = None
        return self._fh

    def event(self, kind: str, **fields: Any) -> None:
        ts = datetime.now().isoformat(timespec="milliseconds")
        lines: list[str] = []
        # A turn_start opens a visible banner so turn boundaries are scannable cold.
        if kind == "turn_start":
            lines.append(_RULE)
        lines.append(f"### {kind}  ·  {ts}")
        for key, value in fields.items():
            lines.append(_render_value(key, value))
        block = "\n".join(lines) + "\n\n"
        with self._lock:
            fh = self._ensure_open()
            if fh is None:
                return
            try:
                fh.write(block)
            except OSError as e:
                logger.warning(f"copilot trace write failed: {e}")

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


def _prune_old_traces(keep: int) -> None:
    # Keep only the newest `keep` transcript files — they are large debug ephemera.
    # Sort by mtime, NOT name: the filename is copilot_<slug>_<stamp>, so a name sort
    # orders by slug-then-time and would prune a newer project's files before an older
    # project's whenever the newer slug sorts later.
    try:
        files = sorted(
            copilot_trace_dir().glob("copilot_*.transcript"),
            key=lambda p: p.stat().st_mtime,
        )
    except OSError as e:
        logger.warning(f"copilot trace prune skipped: {e}")
        return
    for stale in files[: max(0, len(files) - keep)]:
        try:
            stale.unlink()
        except OSError as e:
            logger.debug(f"copilot trace prune: could not remove {stale}: {e}")


def new_trace_log(project_slug: str, stamp: str) -> TraceLog:
    # `project_slug` + `stamp` are passed in (the caller composes them; keeps TraceLog
    # construction pure). Prune older transcripts to the retention cap on each new
    # session so the dir stays bounded.
    _prune_old_traces(LOGGING_CONFIG.trace_retention)
    name = f"copilot_{project_slug}_{stamp}.transcript"
    return TraceLog(copilot_trace_dir() / name)


class _NullTraceLog(TraceLog):
    # The agent loop's default when no trace sink is supplied (tests). event() no-ops
    # (its path can never be opened).
    def __init__(self) -> None:
        super().__init__(Path())

    def _ensure_open(self) -> TextIOWrapper | None:
        return None


NULL_TRACE = _NullTraceLog()
