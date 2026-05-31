from collections.abc import Callable, Iterator
from typing import Any

from loguru import logger
from openai import OpenAI
from openai.types.chat import ChatCompletionChunk

from shaderbox.copilot.llm.api import (
    LLMClient,
    LLMDone,
    LLMMessage,
    LLMStreamEvent,
    LLMTextDelta,
    LLMToolCallCompleted,
    LLMToolCallStarted,
    LLMToolSpec,
    LLMUsage,
)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _to_wire_message(m: LLMMessage) -> dict[str, Any]:
    # content="" + tool_calls breaks grok (the wire wants null content). Coerce (§J5).
    wire: dict[str, Any] = {"role": m.role}
    wire["content"] = m.content if (m.content or not m.tool_calls) else None
    if m.tool_call_id is not None:
        wire["tool_call_id"] = m.tool_call_id
    if m.tool_calls:
        wire["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
            for tc in m.tool_calls
        ]
    return wire


def _tool_to_wire(spec: LLMToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    }


def _log_upstream_error(exc: Exception) -> None:
    # NEVER log the response body (§J4) — OpenRouter error bodies echo the prompt, which
    # carries the user's shader source + the key context. Status/class only.
    status = getattr(exc, "status_code", None)
    logger.error(f"copilot LLM upstream error: {type(exc).__name__} status={status}")


class OpenRouterLLMClient(LLMClient):
    """OpenRouter via the `openai` SDK. Key + model are read LIVE through getters (NOT
    captured) so a project switch that reloads IntegrationsStore — or a key entered
    mid-turn via the credential widget — is seen on the next stream call (§U7).

    Egress is automatic (default dual-stack). A transparent IPv4 fallback for a
    dead-IPv6 route is an impl detail handled inside the client — never a user setting.
    """

    def __init__(
        self,
        get_api_key: Callable[[], str],
        get_model: Callable[[], str],
    ) -> None:
        self._get_api_key = get_api_key
        self._get_model = get_model

    def _client(self) -> OpenAI:
        return OpenAI(
            base_url=_OPENROUTER_BASE_URL,
            api_key=self._get_api_key(),
        )

    def stream(
        self,
        messages: list[LLMMessage],
        *,
        tools: list[LLMToolSpec] | None = None,
        max_tokens: int,
    ) -> Iterator[LLMStreamEvent]:
        try:
            yield from self._stream_impl(messages, tools, max_tokens)
        except Exception as exc:
            _log_upstream_error(exc)
            raise

    def _stream_impl(
        self,
        messages: list[LLMMessage],
        tools: list[LLMToolSpec] | None,
        max_tokens: int,
    ) -> Iterator[LLMStreamEvent]:
        kwargs: dict[str, Any] = {
            "model": self._get_model(),
            "messages": [_to_wire_message(m) for m in messages],
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_completion_tokens": max_tokens,
            "extra_body": {"reasoning": {"effort": "minimal"}},
        }
        if tools:
            kwargs["tools"] = [_tool_to_wire(t) for t in tools]

        builders: dict[int, dict[str, str]] = {}
        started: set[int] = set()
        usage = LLMUsage()
        finish_reason = "stop"
        for chunk in self._client().chat.completions.create(**kwargs):
            chunk: ChatCompletionChunk
            if chunk.usage is not None:
                usage = LLMUsage(
                    input_tokens=chunk.usage.prompt_tokens,
                    output_tokens=chunk.usage.completion_tokens,
                    # usage.cost is OpenRouter-specific + can be None per-chunk (§J7).
                    cost_usd=getattr(chunk.usage, "cost", 0.0) or 0.0,
                )
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            delta = choice.delta
            if delta.content:
                yield LLMTextDelta(delta.content)
            for tc in delta.tool_calls or []:
                idx = tc.index
                b = builders.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                if tc.id:
                    b["id"] = tc.id
                if tc.function and tc.function.name:
                    b["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    b["arguments"] += tc.function.arguments
                if idx not in started and b["id"] and b["name"]:
                    started.add(idx)
                    yield LLMToolCallStarted(index=idx, id=b["id"], name=b["name"])

        # Completion emitted at stream end (arguments fully accumulated, §J8).
        for idx in sorted(builders):
            b = builders[idx]
            yield LLMToolCallCompleted(
                index=idx, id=b["id"], name=b["name"], arguments=b["arguments"]
            )
        yield LLMDone(finish_reason=finish_reason, usage=usage)
