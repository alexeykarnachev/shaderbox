"""Copilot conversation persistence (feature 022) — the ConversationStore round-trip,
fail-soft load, and archive. Pure: no GL, no App."""

import json
from pathlib import Path

from shaderbox.copilot.llm.api import LLMMessage
from shaderbox.copilot.persistence import ConversationStore, archive_conversation
from shaderbox.copilot.state import (
    ChatState,
    Message,
    RecoverInfo,
    ResultWidget,
    StepRecord,
    TurnStats,
)


def _populated_state() -> ChatState:
    state = ChatState()
    # The UI render stream (state.messages) still carries tool_status cards — UI != history.
    state.messages = [
        Message(role="user", text="add a uniform"),
        Message(role="tool_status", text="edit_shader: ok"),
        Message(role="assistant", text="Done — added u_speed."),
    ]
    state.last_turn = TurnStats(context_tokens=1200, reply_tokens=340, cost_usd=0.0042)
    state.session_cost_usd = 0.0042
    return state


def _history() -> list[LLMMessage]:
    # NL-only history (feature 020·28): user + one engine-derived assistant turn-summary. No tool msgs.
    return [
        LLMMessage(role="user", content="add a uniform"),
        LLMMessage(
            role="assistant",
            content="Done — added u_speed.\n(this turn: edit_shader: ok)",
        ),
    ]


def test_round_trip_messages_history_stats(tmp_path: Path) -> None:
    state, history = _populated_state(), _history()
    path = tmp_path / "copilot" / "conversation.json"
    ConversationStore.from_runtime(state, history).save(path)

    loaded = ConversationStore.load(path)
    msgs = loaded.to_messages()
    assert [(m.role, m.text) for m in msgs] == [
        ("user", "add a uniform"),
        ("tool_status", "edit_shader: ok"),
        ("assistant", "Done — added u_speed."),
    ]
    hist = loaded.to_history()
    assert [h.role for h in hist] == ["user", "assistant"]
    assert hist[1].content == "Done — added u_speed.\n(this turn: edit_shader: ok)"
    last_turn = loaded.to_last_turn()
    assert last_turn is not None
    assert (last_turn.context_tokens, last_turn.reply_tokens, last_turn.cost_usd) == (
        1200,
        340,
        0.0042,
    )
    assert loaded.to_session_cost() == 0.0042


def test_save_creates_parent_dir(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "copilot" / "conversation.json"
    ConversationStore().save(path)
    assert path.exists()


def test_missing_file_is_empty_chat(tmp_path: Path) -> None:
    loaded = ConversationStore.load(tmp_path / "nope.json")
    assert loaded.to_messages() == []
    assert loaded.to_history() == []
    assert loaded.to_last_turn() is None
    assert loaded.to_session_cost() == 0.0


def test_corrupt_file_fails_soft(tmp_path: Path) -> None:
    path = tmp_path / "conversation.json"
    path.write_text("{ this is not valid json", encoding="utf-8")
    loaded = ConversationStore.load(path)  # must NOT raise
    assert loaded.to_messages() == []


def test_non_object_json_fails_soft(tmp_path: Path) -> None:
    # Valid JSON that is not an object must degrade to an empty chat, not crash at load.
    for payload in ("[1, 2, 3]", "123"):
        path = tmp_path / "conversation.json"
        path.write_text(payload, encoding="utf-8")
        loaded = ConversationStore.load(path)  # must NOT raise
        assert loaded.to_messages() == []
        assert loaded.to_history() == []


def test_incompatible_schema_fails_soft(tmp_path: Path) -> None:
    path = tmp_path / "conversation.json"
    # Valid JSON, wrong shape (extra=forbid + bad types) -> empty, not a crash.
    path.write_text(
        json.dumps({"messages": "not a list", "bogus": 1}), encoding="utf-8"
    )
    loaded = ConversationStore.load(path)
    assert loaded.to_messages() == []


def test_pre_v5_tool_paired_store_fails_soft_to_empty(tmp_path: Path) -> None:
    # feature 020·28: an old store whose history carries tool_call_id / tool_calls hits extra="forbid"
    # on _HistoryModel -> ValidationError -> empty chat (NL-only by construction, no migration).
    path = tmp_path / "conversation.json"
    path.write_text(
        json.dumps(
            {
                "version": 4,
                "messages": [],
                "history": [
                    {"role": "user", "content": "x"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {"id": "c1", "name": "read_shader", "arguments": "{}"}
                        ],
                    },
                    {"role": "tool", "content": "...source...", "tool_call_id": "c1"},
                ],
            }
        ),
        encoding="utf-8",
    )
    loaded = ConversationStore.load(path)  # must NOT raise
    assert loaded.to_history() == [], (
        "a pre-v5 tool-paired store must reset, not load tool messages"
    )


def test_unknown_role_survives_round_trip(tmp_path: Path) -> None:
    # role is loose (str) on disk so a future MessageRole member loads on an older build.
    path = tmp_path / "conversation.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "messages": [{"role": "future_role", "text": "hi", "resolved": False}],
                "history": [],
            }
        ),
        encoding="utf-8",
    )
    loaded = ConversationStore.load(path)
    msgs = loaded.to_messages()
    assert msgs[0].role == "future_role"  # passed through, not dropped


def test_result_widget_round_trips(tmp_path: Path) -> None:
    # A tool_status card's result widget (feature 020·21) survives a save/load round-trip.
    state = ChatState()
    state.messages = [
        Message(
            role="tool_status",
            text="publish_youtube: ok",
            result_widget=ResultWidget(
                kind="open_url",
                label="Open in YouTube",
                target="https://studio.youtube.com/x",
            ),
        ),
        Message(role="tool_status", text="edit_shader: ok"),  # no widget -> None
    ]
    path = tmp_path / "conversation.json"
    ConversationStore.from_runtime(state, []).save(path)
    msgs = ConversationStore.load(path).to_messages()
    assert msgs[0].result_widget == ResultWidget(
        kind="open_url", label="Open in YouTube", target="https://studio.youtube.com/x"
    )
    assert msgs[1].result_widget is None


def test_v3_message_without_widget_loads_as_none(tmp_path: Path) -> None:
    # A pre-020·21 (v3) file has no result_widget key — it must load as None, not raise.
    path = tmp_path / "conversation.json"
    path.write_text(
        json.dumps(
            {
                "version": 3,
                "messages": [{"role": "tool_status", "text": "render_image: ok"}],
                "history": [],
            }
        ),
        encoding="utf-8",
    )
    msgs = ConversationStore.load(path).to_messages()
    assert msgs[0].result_widget is None


def test_unknown_widget_kind_fails_soft(tmp_path: Path) -> None:
    # A future widget kind (or an empty target) on an older build drops to None — no dead button,
    # no whole-conversation loss.
    path = tmp_path / "conversation.json"
    path.write_text(
        json.dumps(
            {
                "version": 4,
                "messages": [
                    {
                        "role": "tool_status",
                        "text": "x",
                        "result_widget": {
                            "kind": "future_kind",
                            "label": "L",
                            "target": "t",
                        },
                    },
                    {
                        "role": "tool_status",
                        "text": "y",
                        "result_widget": {
                            "kind": "open_url",
                            "label": "L",
                            "target": "",
                        },
                    },
                ],
                "history": [],
            }
        ),
        encoding="utf-8",
    )
    msgs = ConversationStore.load(path).to_messages()
    assert msgs[0].result_widget is None  # unknown kind
    assert msgs[1].result_widget is None  # empty target


def test_turn_snippet_round_trips(tmp_path: Path) -> None:
    # F06: a turn_snippet's step list + its own stats survive a save/load round-trip.
    state = ChatState()
    state.messages = [
        Message(role="user", text="add a uniform", turn_id="turn_0001"),
        Message(
            role="turn_snippet",
            steps=[
                StepRecord(name="read_shader", ok=True),
                StepRecord(name="edit_shader", ok=False),
                StepRecord(name="edit_shader", ok=True),
            ],
            snippet_stats=TurnStats(
                context_tokens=900, reply_tokens=210, cost_usd=0.0031
            ),
        ),
    ]
    path = tmp_path / "conversation.json"
    ConversationStore.from_runtime(state, []).save(path)
    msgs = ConversationStore.load(path).to_messages()
    snip = msgs[1]
    assert snip.role == "turn_snippet"
    assert [(s.name, s.ok) for s in snip.steps] == [
        ("read_shader", True),
        ("edit_shader", False),
        ("edit_shader", True),
    ]
    assert snip.snippet_stats is not None
    assert snip.snippet_stats.reply_tokens == 210
    assert snip.snippet_stats.cost_usd == 0.0031


def test_pre_v9_snippet_fields_default(tmp_path: Path) -> None:
    # A pre-F06 (v8) file has no steps / snippet_stats keys — they must default (empty / None),
    # not raise. A persisted tool_status line still loads as a plain message.
    path = tmp_path / "conversation.json"
    path.write_text(
        json.dumps(
            {
                "version": 8,
                "messages": [
                    {"role": "user", "text": "x"},
                    {"role": "tool_status", "text": "edit_shader: ok"},
                ],
                "history": [],
                "session_cost_usd": 0.0,
            }
        ),
        encoding="utf-8",
    )
    msgs = ConversationStore.load(path).to_messages()
    assert msgs[1].role == "tool_status"
    assert msgs[1].steps == []
    assert msgs[1].snippet_stats is None


def test_archive_moves_file_and_leaves_none(tmp_path: Path) -> None:
    path = tmp_path / "copilot" / "conversation.json"
    ConversationStore.from_runtime(_populated_state(), []).save(path)
    archive_conversation(path, "2026-06-01_18-00-00")
    assert not path.exists()  # moved out
    archived = (
        tmp_path / "copilot" / "archive" / "conversation_2026-06-01_18-00-00.json"
    )
    assert archived.exists()
    # The archived content is the original conversation.
    data = json.loads(archived.read_text(encoding="utf-8"))
    assert data["messages"][0]["text"] == "add a uniform"


def test_archive_missing_file_is_noop(tmp_path: Path) -> None:
    archive_conversation(
        tmp_path / "copilot" / "conversation.json", "stamp"
    )  # no raise
    assert not (tmp_path / "copilot" / "archive").exists()


def test_drop_turn_skips_commit_but_stop_does_not(tmp_path: Path) -> None:
    # The teardown guard: _drop_turn (set by reset/release) makes a finishing aborted turn
    # SKIP its history commit (the orphaned-append fix). A user Stop (cancel_turn) does NOT
    # set _drop_turn, so a stopped turn STILL commits to this conversation's context.
    from collections.abc import Iterator

    from shaderbox.copilot.llm.api import (
        LLMDone,
        LLMStreamEvent,
        LLMTextDelta,
        LLMToolSpec,
        LLMUsage,
    )
    from shaderbox.copilot.session import CopilotSession

    class _OneTextClient:
        model = "test-model"

        def stream(
            self,
            messages: list[LLMMessage],
            *,
            tools: list[LLMToolSpec] | None = None,
            max_tokens: int,
        ) -> Iterator[LLMStreamEvent]:
            _ = (messages, tools, max_tokens)
            return iter([LLMTextDelta("partial reply"), LLMDone("stop", LLMUsage())])

    def _mk() -> CopilotSession:
        from tests.test_copilot_loop import _fake_caps

        return CopilotSession(
            _fake_caps(edit_errors=[]),  # type: ignore[arg-type]
            _OneTextClient(),  # type: ignore[arg-type]
            get_project_slug=lambda: "test",
            get_checkpoints_root=lambda: tmp_path / "checkpoints",
        )

    # Teardown abort -> no commit.
    dropped = _mk()
    dropped._drop_turn.set()
    dropped._run_one_turn("hi")
    assert dropped.history == []
    dropped.release()

    # Plain turn (Stop or normal) -> commits user + assistant.
    kept = _mk()
    kept._run_one_turn("hi")  # _drop_turn not set
    assert [h.role for h in kept.history] == ["user", "assistant"]
    assert kept.history[1].content == "partial reply"
    kept.release()


def test_session_save_then_load_restores(tmp_path: Path) -> None:
    # The session seam: save_conversation writes (state, history); load_conversation on a
    # FRESH session restores both + the turn stats. No worker is spawned (no turn enqueued),
    # and the client is never called by save/load, so a bare stub suffices.
    from shaderbox.copilot.session import CopilotSession
    from tests.test_copilot_loop import _fake_caps

    def _mk() -> CopilotSession:
        return CopilotSession(
            _fake_caps(edit_errors=[]),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]  # client unused by save/load
            get_project_slug=lambda: "test",
            get_checkpoints_root=lambda: tmp_path / "checkpoints",
        )

    sess = _mk()
    sess.state.messages = [
        Message(role="user", text="hi"),
        Message(role="assistant", text="yo"),
    ]
    sess.history = [
        LLMMessage(role="user", content="hi"),
        LLMMessage(role="assistant", content="yo"),
    ]
    sess.state.last_turn = TurnStats(context_tokens=10, reply_tokens=5, cost_usd=0.001)
    sess.state.session_cost_usd = 0.001
    path = tmp_path / "conversation.json"
    sess.save_conversation(path)
    sess.release()

    fresh = _mk()
    fresh.load_conversation(ConversationStore.load(path))
    assert [(m.role, m.text) for m in fresh.state.messages] == [
        ("user", "hi"),
        ("assistant", "yo"),
    ]
    assert [h.role for h in fresh.history] == ["user", "assistant"]
    assert fresh.state.last_turn is not None
    assert fresh.state.last_turn.context_tokens == 10
    assert fresh.state.session_cost_usd == 0.001
    fresh.release()


def test_recover_card_survives_round_trip(tmp_path: Path) -> None:
    # A resolved-Yes delete card carrying RecoverInfo must survive save -> load ->
    # to_messages (the Recover button is rebuilt from it after a restart).
    state = ChatState()
    state.messages = [
        Message(role="user", text="delete the gradient node"),
        Message(
            role="pending_action",
            text="Delete node 'a1b2'?\nYou chose: Yes",
            resolved=True,
            recover=RecoverInfo(
                node_id="full-uuid-1234",
                node_name="Gradient",
                trash_name="full-uuid-1234",
            ),
        ),
        Message(role="assistant", text="Done — moved to trash."),
    ]
    path = tmp_path / "conversation.json"
    ConversationStore.from_runtime(state, []).save(path)
    msgs = ConversationStore.load(path).to_messages()

    assert len(msgs) == 3
    card = msgs[1]
    assert card.role == "pending_action" and card.resolved
    assert card.recover is not None
    assert card.recover.node_id == "full-uuid-1234"
    assert card.recover.trash_name == "full-uuid-1234"
    assert card.recover.node_name == "Gradient"
    assert not card.recover.done
    assert msgs[2].recover is None


def test_pre_recover_v1_file_loads_soft(tmp_path: Path) -> None:
    # A v1-shaped file (no `recover` key) loads into the current schema with recover=None.
    path = tmp_path / "v1.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "messages": [
                    {"role": "user", "text": "hi", "resolved": False},
                    {
                        "role": "pending_action",
                        "text": "Delete node 'x'?",
                        "resolved": True,
                    },
                ],
                "history": [],
            }
        )
    )
    msgs = ConversationStore.load(path).to_messages()
    assert len(msgs) == 2
    assert msgs[1].recover is None


def test_gate_outcome_round_trip_and_pre_v10_default(tmp_path: Path) -> None:
    # The structured outcome (034 F03) survives the round trip; a pre-v10 file
    # (outcome baked into text) loads with gate_outcome="" and renders as plain text.
    state = ChatState()
    state.messages = [
        Message(
            role="pending_action",
            text="Delete node `Blank`?",
            resolved=True,
            gate_outcome="Yes",
        ),
    ]
    path = tmp_path / "conversation.json"
    ConversationStore.from_runtime(state, []).save(path)
    msgs = ConversationStore.load(path).to_messages()
    assert msgs[0].gate_outcome == "Yes"

    pre_v10 = tmp_path / "old.json"
    pre_v10.write_text(
        json.dumps(
            {
                "version": 9,
                "messages": [
                    {
                        "role": "pending_action",
                        "text": "Delete node 'x'?\nYou chose: Yes",
                        "resolved": True,
                    }
                ],
                "history": [],
            }
        )
    )
    old_msgs = ConversationStore.load(pre_v10).to_messages()
    assert old_msgs[0].gate_outcome == ""
    assert "You chose: Yes" in old_msgs[0].text
