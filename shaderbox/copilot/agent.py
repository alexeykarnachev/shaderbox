from collections.abc import Iterator
from dataclasses import dataclass

from shaderbox.copilot.config import CopilotConfig
from shaderbox.copilot.llm.api import LLMClient
from shaderbox.copilot.tools.registry import ToolRegistry

# The agent loop: stream -> tool_calls -> execute -> repeat, with the usage rollup,
# budget guards, and the action-required pause. The LOOP BODY is the LLM-impl/agent
# wave (report 09 §2.3, §3.5) — it needs the OpenRouter stream (openrouter.py) and a
# non-empty tool catalog, both deferred. This wave ships the shape + the emitted-event
# union the worker pushes to the UI.


@dataclass(frozen=True)
class AgentTextDelta:
    text: str


@dataclass(frozen=True)
class AgentStatus:
    text: str


@dataclass(frozen=True)
class AgentTurnDone:
    pass


@dataclass(frozen=True)
class AgentError:
    message: str


AgentEvent = AgentTextDelta | AgentStatus | AgentTurnDone | AgentError


def run_turn(
    client: LLMClient,
    registry: ToolRegistry,
    config: CopilotConfig,
    user_text: str,
) -> Iterator[AgentEvent]:
    _ = (client, registry, config, user_text)
    raise NotImplementedError(
        "agent loop lands in the LLM-impl wave (report 09 §2.3, §3.5)"
    )
