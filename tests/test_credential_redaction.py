"""Credential redaction (security-critical, feature 020·19): the pasted token must never
appear in the handler's LLM-facing msg, the resolved-card text, the payload, or a
persisted conversation round-trip — only the masked prefix. Pure: no GL, no network."""

from pathlib import Path

from shaderbox.copilot.gate import GateKind
from shaderbox.copilot.llm.openrouter import OpenRouterLLMClient
from shaderbox.copilot.persistence import ConversationStore
from shaderbox.copilot.session import CopilotSession
from shaderbox.copilot.state import ChatState, Message
from shaderbox.copilot.tools.base import mask_secret
from shaderbox.copilot.tools.registry import build_registry
from tests._caps import minimal_caps

_FAKE_TOKEN = "1234567890:AAfake_secret_tail_must_never_appear_anywhere"


def test_mask_secret_hides_the_tail() -> None:
    masked = mask_secret(_FAKE_TOKEN)
    assert _FAKE_TOKEN not in masked
    assert "fake_secret_tail" not in masked


def test_handler_msg_and_payload_are_redacted() -> None:
    # The set_telegram_token handler receives the secret as its 2nd arg (never in args)
    # and must return only the masked prefix in its model-facing msg.
    registry = build_registry(minimal_caps())
    ok, msg, payload = registry.execute("set_telegram_token", {}, _FAKE_TOKEN)
    _ = ok
    assert _FAKE_TOKEN not in msg and "fake_secret_tail" not in msg
    assert payload is None or _FAKE_TOKEN not in str(payload)


def test_resolved_card_and_persistence_are_redacted(tmp_path: Path) -> None:
    # The resolved-card echo (answer_gate_credential) + a persistence round-trip carry only
    # the mask; gate_kind survives the round-trip (a credential card stays one).
    client = OpenRouterLLMClient(get_api_key=lambda: "", get_model=lambda: "stub")
    session = CopilotSession(
        caps=minimal_caps(),
        client=client,
        get_project_slug=lambda: "test",
        get_checkpoints_root=lambda: tmp_path / "checkpoints",
    )
    session.state = ChatState(
        messages=[
            Message(role="user", text="connect telegram"),
            Message(
                role="pending_action",
                text="Paste your Telegram bot token below.",
                gate_kind=GateKind.CREDENTIAL,
            ),
        ]
    )
    session.answer_gate_credential(_FAKE_TOKEN)
    card = session.state.messages[1]
    assert card.resolved
    assert _FAKE_TOKEN not in card.text
    assert card.gate_input == "", "the typed-secret buffer was not cleared on answer"

    store = ConversationStore.from_runtime(session.state, [])
    blob = store.model_dump_json()
    assert _FAKE_TOKEN not in blob and "fake_secret_tail" not in blob

    reloaded = ConversationStore.model_validate_json(blob).to_messages()
    assert reloaded[1].gate_kind is GateKind.CREDENTIAL
    session.release()
