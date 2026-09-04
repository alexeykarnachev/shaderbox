"""A provider that refuses the reasoning-effort setting ("Reasoning is mandatory for this
endpoint and cannot be disabled", HTTP 400) gets the request re-sent once without it, and the
model is remembered so later requests skip the setting. Found on the station: three of five
candidate models rejected every turn at once. Any other 400 still surfaces as before."""

from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from openai import BadRequestError

from shaderbox.copilot.llm.api import LLMDone, LLMMessage, LLMStreamEvent
from shaderbox.copilot.llm.openrouter import LLMUpstreamError, OpenRouterLLMClient


def _bad_request(message: str) -> BadRequestError:
    response = httpx.Response(400, request=httpx.Request("POST", "https://x"))
    return BadRequestError(
        message, response=response, body={"error": {"message": message}}
    )


def _client(calls: list[bool], refuse_reasoning: bool) -> OpenRouterLLMClient:
    client = OpenRouterLLMClient(get_api_key=lambda: "k", get_model=lambda: "m/x")

    def fake_impl(
        messages: list[LLMMessage],
        tools: Any,
        max_tokens: int,
        with_reasoning: bool,
    ) -> Iterator[LLMStreamEvent]:
        calls.append(with_reasoning)
        if with_reasoning and refuse_reasoning:
            raise _bad_request(
                "Reasoning is mandatory for this endpoint and cannot be disabled."
            )
        yield LLMDone("stop")

    client._stream_impl = fake_impl  # type: ignore[method-assign]
    return client


def test_a_reasoning_refusal_resends_without_it_and_remembers() -> None:
    calls: list[bool] = []
    client = _client(calls, refuse_reasoning=True)
    events = list(client.stream([LLMMessage(role="user", content="hi")], max_tokens=10))
    assert isinstance(events[-1], LLMDone)
    assert calls == [True, False]
    list(client.stream([LLMMessage(role="user", content="again")], max_tokens=10))
    assert calls == [True, False, False]


def test_any_other_400_still_surfaces() -> None:
    calls: list[bool] = []
    client = OpenRouterLLMClient(get_api_key=lambda: "k", get_model=lambda: "m/x")

    def fake_impl(*args: Any, **kwargs: Any) -> Iterator[LLMStreamEvent]:
        calls.append(True)
        raise _bad_request("Invalid tool schema.")
        yield LLMDone("stop")

    client._stream_impl = fake_impl  # type: ignore[method-assign]
    with pytest.raises(LLMUpstreamError):
        list(client.stream([LLMMessage(role="user", content="hi")], max_tokens=10))
    assert calls == [True]
