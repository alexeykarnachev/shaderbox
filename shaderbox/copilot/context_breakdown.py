"""What one LLM request CONTAINED, block by block.

`build_messages` composes named prompt tiers, but the request the model sees is assembled
per iteration: the built tiers, then the within-turn tool exchange, then the live working set,
plus the `tools=` block beside the messages. `breakdown_request` measures every one of those
at the moment the request is assembled and keeps each part's text, so a run's record can show
both the map (sizes) and the territory (what the block actually said). Emitted by `run_turn` as
the `context_breakdown` trace event; the dogfooding station is its consumer.
"""

import json
from dataclasses import dataclass

from shaderbox.copilot.llm.api import LLMMessage, LLMToolSpec
from shaderbox.copilot.prompt import (
    CHARS_PER_TOKEN,
    DIALOGUE_BLOCK,
    WORKING_SET_BLOCK,
    RenderedBlock,
    Volatility,
    estimate_tokens,
)

# The within-turn tool exchange: context, but not a PromptBlock. The `tools=` block is
# measured on ContextBreakdown's own tools_* fields rather than through a named block.
EXCHANGE_BLOCK = "turn_exchange"


@dataclass(frozen=True)
class BlockBreakdown:
    name: str
    volatility: str
    messages: int
    chars: int
    est_tokens: int
    text: str
    # Dialogue only: whether the trim dropped history, and how many messages it dropped.
    trimmed: bool = False
    dropped_messages: int = 0


@dataclass(frozen=True)
class ContextBreakdown:
    iteration: int
    blocks: tuple[BlockBreakdown, ...]
    tools: tuple[str, ...]
    tools_chars: int
    tools_est_tokens: int
    tools_text: str
    est_total_tokens: int


def _message_text(m: LLMMessage) -> str:
    head = f"[{m.role}]"
    if m.tool_call_id is not None:
        head += f" (tool_call_id={m.tool_call_id})"
    parts = [head]
    if m.content:
        parts.append(m.content)
    for tc in m.tool_calls or ():
        parts.append(f"-> {tc.name}({tc.id}) {tc.arguments}")
    return "\n".join(parts)


def _measure(
    name: str,
    volatility: Volatility,
    messages: list[LLMMessage],
    *,
    trimmed: bool = False,
    dropped: int = 0,
) -> BlockBreakdown:
    chars = 0
    for m in messages:
        if m.content:
            chars += len(m.content)
        for tc in m.tool_calls or ():
            chars += len(tc.name) + len(tc.arguments)
    return BlockBreakdown(
        name=name,
        volatility=volatility.name,
        messages=len(messages),
        chars=chars,
        est_tokens=estimate_tokens(messages),
        text="\n\n".join(_message_text(m) for m in messages),
        trimmed=trimmed,
        dropped_messages=dropped,
    )


def breakdown_request(
    rendered: list[RenderedBlock],
    exchange: list[LLMMessage],
    scratchpad: list[LLMMessage],
    specs: list[LLMToolSpec],
    *,
    iteration: int,
    history_len: int,
) -> ContextBreakdown:
    """Measure one request. `rendered` is the turn's built tiers in prompt order; `exchange` the
    assistant/tool pairs accumulated within the turn so far; `scratchpad` the working set as
    spliced for THIS iteration (the tier of that name renders empty at build time, so its live
    splice is what gets measured, never the placeholder)."""
    blocks: list[BlockBreakdown] = []
    for rb in rendered:
        if rb.block.name == WORKING_SET_BLOCK:
            continue
        trimmed = rb.block.name == DIALOGUE_BLOCK and len(rb.messages) < history_len
        blocks.append(
            _measure(
                rb.block.name,
                rb.block.volatility,
                rb.messages,
                trimmed=trimmed,
                dropped=history_len - len(rb.messages) if trimmed else 0,
            )
        )
    blocks.append(_measure(EXCHANGE_BLOCK, Volatility.PER_TURN, exchange))
    working_set = next(
        (rb.block for rb in rendered if rb.block.name == WORKING_SET_BLOCK), None
    )
    ws_volatility = working_set.volatility if working_set else Volatility.PER_TURN
    blocks.append(_measure(WORKING_SET_BLOCK, ws_volatility, scratchpad))
    tools_payload = [
        {"name": s.name, "description": s.description, "parameters": s.parameters}
        for s in specs
    ]
    tools_compact = json.dumps(tools_payload, separators=(",", ":"))
    tools_est = int(len(tools_compact) / CHARS_PER_TOKEN)
    return ContextBreakdown(
        iteration=iteration,
        blocks=tuple(blocks),
        tools=tuple(s.name for s in specs),
        tools_chars=len(tools_compact),
        tools_est_tokens=tools_est,
        tools_text=json.dumps(tools_payload, indent=1),
        est_total_tokens=sum(b.est_tokens for b in blocks) + tools_est,
    )
