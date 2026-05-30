from collections.abc import Callable, Iterator

import httpx
from openai import OpenAI

from shaderbox.copilot.llm.api import (
    LLMClient,
    LLMMessage,
    LLMStreamEvent,
    LLMToolSpec,
)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _make_http_client(ipv4_only: bool) -> httpx.Client | None:
    # Default: dual-stack (None lets the SDK build its own client) — works on users'
    # machines. ipv4_only is the OPT-IN pin for a dead-IPv6 dev box (do NOT default it;
    # see report 09 §6.1). Mirrors the Telegram _ipv4_request fix without overfitting.
    if not ipv4_only:
        return None
    return httpx.Client(transport=httpx.HTTPTransport(local_address="0.0.0.0"))


class OpenRouterLLMClient(LLMClient):
    """OpenRouter via the `openai` SDK. Key + model are read LIVE through getters (NOT
    captured) so a project switch that reloads IntegrationsStore is seen — see the 7b
    lifecycle pin in skeleton 10 §4.

    SCAFFOLD: the streaming + tool-call accumulation + usage extraction body is the next
    wave (report 09 §3.3) — this wave wires the client construction + the Protocol shape.
    """

    def __init__(
        self,
        get_api_key: Callable[[], str],
        get_model: Callable[[], str],
        *,
        ipv4_only: bool = False,
    ) -> None:
        self._get_api_key = get_api_key
        self._get_model = get_model
        self._http_client = _make_http_client(ipv4_only)

    def _client(self) -> OpenAI:
        return OpenAI(
            base_url=_OPENROUTER_BASE_URL,
            api_key=self._get_api_key(),
            http_client=self._http_client,
        )

    def stream(
        self,
        messages: list[LLMMessage],
        *,
        tools: list[LLMToolSpec] | None = None,
        max_tokens: int,
    ) -> Iterator[LLMStreamEvent]:
        raise NotImplementedError(
            "OpenRouter stream loop lands in the LLM-impl wave (report 09 §3)"
        )
