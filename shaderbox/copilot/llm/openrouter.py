from collections.abc import Callable, Iterator
from typing import Any

import httpx
from loguru import logger
from openai import APIStatusError, OpenAI
from openai.types.chat import ChatCompletionChunk

from shaderbox.copilot.config import COPILOT_CONFIG
from shaderbox.copilot.errors import CopilotConfigError
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


class LLMUpstreamError(Exception):
    """Sanitized stand-in for the SDK's APIStatusError family, whose str() embeds the full
    response body (which echoes the prompt + the user's shader source). Status + error
    class only, so no caller's log line or traceback can leak the body."""

    def __init__(self, status_code: int, error_class: str) -> None:
        super().__init__(f"{error_class} (HTTP {status_code})")
        self.status_code: int = status_code


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

    @property
    def model(self) -> str:
        return self._get_model()

    def _client(self) -> OpenAI:
        # Bound the TOTAL wait, not one attempt (043 hang). The SDK default is 600s AND it silently
        # RETRIES (max_retries=2 -> x3) — a bare-float timeout also clobbers the 5s connect default to
        # the full value. So: an explicit httpx.Timeout (small connect, llm_request_timeout_s for
        # read/write/pool) + max_retries=0 (a copilot turn wants a fast-fail stream_error, not three
        # silent slow retries the per-delta cancel can't interrupt mid-create()).
        timeout = httpx.Timeout(COPILOT_CONFIG.llm_request_timeout_s, connect=5.0)
        return OpenAI(
            base_url=_OPENROUTER_BASE_URL,
            api_key=self._get_api_key(),
            timeout=timeout,
            max_retries=0,
        )

    def stream(
        self,
        messages: list[LLMMessage],
        *,
        tools: list[LLMToolSpec] | None = None,
        max_tokens: int,
    ) -> Iterator[LLMStreamEvent]:
        if not self._get_api_key():
            raise CopilotConfigError("no OpenRouter key set (Settings -> Integrations)")
        if not self._get_model():
            raise CopilotConfigError(
                "no OpenRouter model set (Settings -> Integrations)"
            )
        try:
            yield from self._stream_impl(messages, tools, max_tokens)
        except APIStatusError as exc:
            _log_upstream_error(exc)
            # `from None`: the implicit context chain would re-expose the body-bearing
            # original in any logged traceback.
            raise LLMUpstreamError(exc.status_code, type(exc).__name__) from None
        except Exception as exc:
            _log_upstream_error(exc)
            raise

    def _stream_impl(
        self,
        messages: list[LLMMessage],
        tools: list[LLMToolSpec] | None,
        max_tokens: int,
    ) -> Iterator[LLMStreamEvent]:
        model = self._get_model()
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [_to_wire_message(m) for m in messages],
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_completion_tokens": max_tokens,
            "extra_body": {"reasoning": {"effort": "minimal"}},
        }
        if tools:
            kwargs["tools"] = [_tool_to_wire(t) for t in tools]
        logger.debug(
            f"copilot LLM request | model={model} msgs={len(messages)} "
            f"tools={len(tools) if tools else 0} max_tokens={max_tokens}"
        )

        builders: dict[int, dict[str, str]] = {}
        started: set[int] = set()
        usage = LLMUsage()
        finish_reason = "stop"
        for chunk in self._client().chat.completions.create(**kwargs):
            chunk: ChatCompletionChunk
            if chunk.usage is not None:
                details = getattr(chunk.usage, "completion_tokens_details", None)
                prompt_details = getattr(chunk.usage, "prompt_tokens_details", None)
                usage = LLMUsage(
                    input_tokens=chunk.usage.prompt_tokens,
                    output_tokens=chunk.usage.completion_tokens,
                    reasoning_tokens=getattr(details, "reasoning_tokens", 0) or 0,
                    cached_tokens=getattr(prompt_details, "cached_tokens", 0) or 0,
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
