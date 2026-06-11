"""Invariant: every tool is fully carded at its single definition site (feature 029).

One test kills three drift classes: a missing label (the old _TOOL_VERBS gap), a missing
gate prompt on an always-gated tool (the old _GATE_PROMPTS gap), and the dogfood analyzer's
coverage list going stale on a tool add/remove. Plus the structural gate/credential
invariants and the publish precheck handoffs (all counts derived from the registry, never
literals).
"""

from scripts.dogfood.analyze import CANONICAL_TOOLS
from shaderbox.copilot.capabilities import NodeTreeEntry
from shaderbox.copilot.gate import GateKind
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


def test_gate_and_credential_structural_invariants() -> None:
    registry = build_registry(minimal_caps())
    definitions = registry.definitions()

    # eager_specs() is exactly the eager-flagged definitions, and every spec emits a real
    # JSON schema (a pydantic Field/constraint typo would raise or yield an empty dict).
    eager = registry.eager_specs()
    assert {s.name for s in eager} == {d.name for d in definitions if d.eager}
    for spec in eager:
        assert isinstance(spec.parameters, dict) and spec.parameters, spec.name

    # Destructive/external tools are ALWAYS-gated.
    always = {d.name for d in definitions if d.gate_policy is GatePolicy.ALWAYS}
    assert {
        "render_image",
        "render_video",
        "publish_telegram",
        "publish_youtube",
        "delete_node",
    } <= always

    # set_telegram_token is THE credential-gated tool (derived, not a count literal).
    credential = [d for d in definitions if d.gate_kind is GateKind.CREDENTIAL]
    assert [d.name for d in credential] == ["set_telegram_token"]
    assert credential[0].secret_field == "telegram_bot_token"


def test_publish_prechecks_hand_off_until_ready() -> None:
    # Ready: prechecks return None (render tools have no precheck at all).
    ready = build_registry(
        minimal_caps(
            telegram_connected=lambda: True,
            telegram_has_default_pack=lambda: True,
            youtube_connected=lambda: True,
        )
    )
    assert ready.precheck("publish_telegram", {}) is None
    assert ready.precheck("publish_youtube", {}) is None
    assert ready.precheck("render_image", {}) is None

    # Not connected -> a guided handoff (no gate fires for this call).
    no_tg = build_registry(minimal_caps())
    msg = no_tg.precheck("publish_telegram", {})
    assert msg is not None and "connect" in msg.lower()

    # Connected but no pack -> a different handoff.
    no_pack = build_registry(minimal_caps(telegram_connected=lambda: True))
    msg = no_pack.precheck("publish_telegram", {})
    assert msg is not None and "pack" in msg.lower()

    no_yt = build_registry(minimal_caps())
    msg = no_yt.precheck("publish_youtube", {})
    assert msg is not None and "connect" in msg.lower()


def test_delete_gate_prompt_shows_node_name() -> None:
    # The confirm card asks with the node's NAME (resolved via the project map with
    # read_shader's prefix rule), short — no trash/recover tail (034 F01).
    registry = build_registry(
        minimal_caps(
            node_tree=lambda: [
                NodeTreeEntry(
                    node_id="a1b2", name="Blank", has_errors=False, is_current=True
                )
            ]
        )
    )
    d = registry.definition_for("delete_node")
    assert d is not None and d.gate_prompt is not None
    assert d.gate_prompt({"node": "a1b2"}) == "Delete node `Blank`?"
    # Full uuid matching the short id resolves too; an unknown id falls back raw.
    assert d.gate_prompt({"node": "a1b2c3d4-ffff"}) == "Delete node `Blank`?"
    assert d.gate_prompt({"node": "zzzz"}) == "Delete node `zzzz`?"
    assert d.gate_prompt({}) == "Delete node `?`?"
