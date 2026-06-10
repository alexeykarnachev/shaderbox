"""Invariant: every tool is fully carded at its single definition site (feature 029).

One test kills three drift classes: a missing label (the old _TOOL_VERBS gap), a missing
gate prompt on an always-gated tool (the old _GATE_PROMPTS gap), and the dogfood analyzer's
coverage list going stale on a tool add/remove.
"""

from scripts.dogfood.analyze import CANONICAL_TOOLS
from shaderbox.copilot.tools.base import GatePolicy, ToolDefinition
from shaderbox.copilot.tools.registry import ToolRegistry, build_registry
from tests._caps import minimal_caps


def test_every_tool_fully_carded() -> None:
    registry: ToolRegistry = build_registry(minimal_caps())
    definitions: list[ToolDefinition] = registry.definitions()
    assert {d.name for d in definitions} == set(CANONICAL_TOOLS)
    for d in definitions:
        assert d.label_live and d.label_live != d.name
        assert d.label_done and d.label_done != d.name
        if d.gate_policy is GatePolicy.ALWAYS:
            assert d.gate_prompt is not None
        # The LLM-facing contract: hallucinated arg keys must be rejected, not swallowed.
        # Checks the emitted schema, so a stray non-ToolArgs args model can't sneak past.
        schema = d.args_model.model_json_schema()
        assert schema.get("additionalProperties") is False, d.name
