from shaderbox.copilot.context import CopilotContext
from shaderbox.copilot.llm.api import LLMMessage

# Prompt assembly: (system text, context snapshot, history, the new user turn) ->
# list[LLMMessage]. Knows nothing of tools or the client. The system-prompt TEXT and
# the context-snapshot rendering are the later capability brainstorm (§0 #8) — this is
# the scaffold seam (the assembly fn + its inputs).

_SYSTEM_PROMPT = "You are ShaderBox's in-app coding copilot."  # placeholder — filled later


def build_messages(
    context: CopilotContext,
    history: list[LLMMessage],
    user_text: str,
) -> list[LLMMessage]:
    _ = context  # snapshot rendering lands in a later wave
    return [
        LLMMessage(role="system", content=_SYSTEM_PROMPT),
        *history,
        LLMMessage(role="user", content=user_text),
    ]
