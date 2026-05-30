from collections.abc import Callable, Iterator

from openai import OpenAI

from shaderbox.copilot.llm.api import (
    LLMClient,
    LLMMessage,
    LLMStreamEvent,
    LLMToolSpec,
)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterLLMClient(LLMClient):
    """OpenRouter via the `openai` SDK. Key + model are read LIVE through getters (NOT
    captured) so a project switch that reloads IntegrationsStore is seen — see the 7b
    lifecycle pin in skeleton 10 §4.

    Egress is automatic (default dual-stack). A transparent IPv4 fallback for a
    dead-IPv6 route is an impl detail handled inside the client — never a user setting.

    SCAFFOLD: the streaming + tool-call accumulation + usage extraction body is the next
    wave (report 09 §3.3) — this wave wires the client construction + the Protocol shape.
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
        raise NotImplementedError(
            "OpenRouter stream loop lands in the LLM-impl wave (report 09 §3)"
        )
