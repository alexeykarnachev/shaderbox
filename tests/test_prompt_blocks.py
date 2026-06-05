"""The block-based prompt constructor (feature 020·25): build_prompt sorts named blocks by volatility,
renders each, drops empty ones, flattens. build_messages composes [static < rare < dialogue < pending].
Pure: no GL, no client."""

from shaderbox.copilot.context import CopilotContext
from shaderbox.copilot.llm.api import LLMMessage
from shaderbox.copilot.prompt import (
    PromptBlock,
    Volatility,
    build_messages,
    build_prompt,
)


def _m(role: str, content: str) -> LLMMessage:
    return LLMMessage(role=role, content=content)


def test_build_prompt_orders_by_volatility() -> None:
    blocks = [
        PromptBlock("pending", Volatility.PER_TURN, lambda: [_m("user", "now")]),
        PromptBlock("static", Volatility.STATIC, lambda: [_m("system", "rules")]),
        PromptBlock("dialogue", Volatility.DIALOGUE, lambda: [_m("user", "earlier")]),
    ]
    out = build_prompt(blocks)
    assert [m.content for m in out] == ["rules", "earlier", "now"]


def test_empty_block_is_dropped() -> None:
    blocks = [
        PromptBlock("static", Volatility.STATIC, lambda: [_m("system", "rules")]),
        PromptBlock("scratch", Volatility.PER_TURN, lambda: []),
        PromptBlock("pending", Volatility.PER_TURN, lambda: [_m("user", "now")]),
    ]
    out = build_prompt(blocks)
    assert [m.content for m in out] == ["rules", "now"]


def test_dialogue_block_expands_to_many_messages() -> None:
    # A block renders a LIST, so dialogue (many) and a singleton are one mechanism.
    history = [_m("user", "a"), _m("assistant", "b"), _m("user", "c")]
    blocks = [PromptBlock("dialogue", Volatility.DIALOGUE, lambda: history)]
    assert build_prompt(blocks) == history


def _ctx() -> CopilotContext:
    return CopilotContext(
        node_tree="- s (id: n1) [current]",
        lib_catalog="(empty)",
        template_catalog="(none)",
        conventions="",
    )


def test_build_messages_tiers_static_rare_dialogue_pending() -> None:
    history = [_m("user", "hi"), _m("assistant", "hey")]
    out = build_messages(_ctx(), history, "what now")
    # static + rare are the two leading system messages; then the dialogue; then the new user last.
    assert out[0].role == "system" and out[1].role == "system"
    assert out[-1].role == "user" and out[-1].content == "what now"
    assert [m.content for m in out[2:-1]] == ["hi", "hey"]
