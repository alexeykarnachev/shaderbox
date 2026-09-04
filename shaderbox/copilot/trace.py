import threading
from collections.abc import Callable
from dataclasses import is_dataclass
from datetime import datetime
from io import TextIOWrapper
from pathlib import Path
from typing import Any

from loguru import logger

from shaderbox.copilot.context_breakdown import ContextBreakdown
from shaderbox.copilot.llm.api import LLMMessage, LLMToolSpec, LLMUsage
from shaderbox.logging_setup import LOGGING_CONFIG
from shaderbox.paths import copilot_trace_dir

# Full-fidelity copilot transcript: plain-text (not jsonl) so a 50-line shader reads as a
# shader, not a `\n`-escaped one-liner. One TraceLog per session; appends; flushed per event.
# Opened lazily and re-opened after close — the App lifecycle calls release() at every _init.
#
# Beside the file, every event also goes to the session's listeners as the STRUCTURED
# (kind, fields) pair — the seam an observer (the dogfooding station) records from, so it never
# parses the transcript back. Listeners run on the emitting (worker) thread.

TraceListener = Callable[[str, dict[str, Any]], None]

_INDENT = "    "
_RULE = "=" * 78


def _indent(text: str) -> str:
    return "\n".join(_INDENT + line for line in text.splitlines())


def _render_usage(u: LLMUsage) -> str:
    return (
        f"in={u.input_tokens} out={u.output_tokens} rsn={u.reasoning_tokens} "
        f"cache={u.cached_tokens} cost=${u.cost_usd:.6f}"
    )


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


def _render_breakdown(b: ContextBreakdown) -> str:
    # Sizes only: the block texts are the llm_request just above, already in the file.
    rows = [
        f"{blk.name:<16} {blk.volatility:<9} msgs={blk.messages:<3} chars={blk.chars:<7} "
        f"~tok={blk.est_tokens}"
        + (f"  TRIMMED (-{blk.dropped_messages} msgs)" if blk.trimmed else "")
        for blk in b.blocks
    ]
    rows.append(
        f"{'tools':<16} {'':<9} n={len(b.tools):<6} chars={b.tools_chars:<7} ~tok={b.tools_est_tokens}"
    )
    rows.append(f"~total tok={b.est_total_tokens}")
    return "\n".join(rows)


def _render_value(key: str, value: Any) -> str:
    if value is None:
        return f"{key}: (none)"
    if isinstance(value, LLMUsage):
        return f"{key}: {_render_usage(value)}"
    if isinstance(value, ContextBreakdown):
        return f"{key}:\n{_indent(_render_breakdown(value))}"
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
    def __init__(
        self, path: Path, listeners: list[TraceListener] | None = None
    ) -> None:
        self._path = path
        # Shared with the session so a listener registered after construction is seen.
        self._listeners = listeners if listeners is not None else []
        self._lock = threading.Lock()  # worker thread writes; serialize appends
        self._fh: TextIOWrapper | None = None
        # Permanent-closed latch: `_fh is None` alone can't tell "never opened" from "closed", so a
        # stray event after close() would silently re-open the just-closed file (append) and bleed a
        # later turn's lines into it. A reopen is a NEW TraceLog object (session._new_trace), never a
        # revived closed one — so this latch only ever needs to STAY closed, no re-arm.
        self._closed = False

    def _ensure_open(self) -> TextIOWrapper | None:
        # Caller holds the lock. Open (append) on first use; never re-open after close().
        if self._closed:
            return None
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
        # turn_start gets a banner rule so turn boundaries are scannable.
        if kind == "turn_start":
            lines.append(_RULE)
        lines.append(f"### {kind}  ·  {ts}")
        for key, value in fields.items():
            lines.append(_render_value(key, value))
        block = "\n".join(lines) + "\n\n"
        with self._lock:
            fh = self._ensure_open()
            if fh is not None:
                try:
                    fh.write(block)
                except OSError as e:
                    logger.warning(f"copilot trace write failed: {e}")
        for listen in list(self._listeners):
            # An observer must never break the turn it observes.
            try:
                listen(kind, fields)
            except Exception as e:
                logger.warning(f"copilot trace listener failed on {kind}: {e}")

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._fh is not None:
                self._fh.close()
                self._fh = None


def _prune_old_traces(keep: int) -> None:
    # Sort by mtime, NOT name: the name is copilot_<slug>_<stamp>, so a name sort orders
    # slug-then-time and would prune a newer project's files before an older project's.
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


def new_trace_log(
    project_slug: str, stamp: str, listeners: list[TraceListener] | None = None
) -> TraceLog:
    # Prune to the retention cap each new session so the dir stays bounded.
    _prune_old_traces(LOGGING_CONFIG.trace_retention)
    name = f"copilot_{project_slug}_{stamp}.transcript"
    return TraceLog(copilot_trace_dir() / name, listeners)


class _NullTraceLog(TraceLog):
    # No-op sink (default when none supplied): _ensure_open never opens, so event() no-ops.
    def __init__(self) -> None:
        super().__init__(Path())

    def _ensure_open(self) -> TextIOWrapper | None:
        return None


NULL_TRACE = _NullTraceLog()
