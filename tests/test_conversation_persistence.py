"""Copilot conversation persistence (feature 022) — the ConversationStore round-trip,
fail-soft load, and archive. Pure: no GL, no App."""

import json
from pathlib import Path

from shaderbox.copilot.llm.api import LLMMessage, LLMToolCall
from shaderbox.copilot.persistence import ConversationStore, archive_conversation
from shaderbox.copilot.state import ChatState, Message, SessionUsage


def _populated_state() -> ChatState:
    state = ChatState()
    state.messages = [
        Message(role="user", text="add a uniform"),
        Message(role="tool_status", text="edit_shader: ok"),
        Message(role="assistant", text="Done — added u_speed."),
    ]
    state.usage = SessionUsage(input_tokens=1200, output_tokens=340, cost_usd=0.0042)
    return state


def _history() -> list[LLMMessage]:
    return [
        LLMMessage(role="user", content="add a uniform"),
        LLMMessage(
            role="assistant",
            content=None,
            tool_calls=[LLMToolCall(id="c1", name="edit_shader", arguments='{"x":1}')],
        ),
        LLMMessage(role="tool", content="ok — compiled clean", tool_call_id="c1"),
        LLMMessage(role="assistant", content="Done — added u_speed."),
    ]


def test_round_trip_messages_history_usage(tmp_path: Path) -> None:
    state, history = _populated_state(), _history()
    path = tmp_path / "copilot" / "conversation.json"
    ConversationStore.from_runtime(state, history).save(path)

    loaded = ConversationStore.load_and_migrate(path)
    msgs = loaded.to_messages()
    assert [(m.role, m.text) for m in msgs] == [
        ("user", "add a uniform"),
        ("tool_status", "edit_shader: ok"),
        ("assistant", "Done — added u_speed."),
    ]
    hist = loaded.to_history()
    assert [h.role for h in hist] == ["user", "assistant", "tool", "assistant"]
    assert hist[1].tool_calls is not None
    assert hist[1].tool_calls[0].id == "c1"
    assert hist[1].tool_calls[0].arguments == '{"x":1}'
    assert hist[2].tool_call_id == "c1"
    usage = loaded.to_usage()
    assert (usage.input_tokens, usage.output_tokens, usage.cost_usd) == (
        1200,
        340,
        0.0042,
    )


def test_save_creates_parent_dir(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "copilot" / "conversation.json"
    ConversationStore().save(path)
    assert path.exists()


def test_missing_file_is_empty_chat(tmp_path: Path) -> None:
    loaded = ConversationStore.load_and_migrate(tmp_path / "nope.json")
    assert loaded.to_messages() == []
    assert loaded.to_history() == []
    assert loaded.to_usage().input_tokens == 0


def test_corrupt_file_fails_soft(tmp_path: Path) -> None:
    path = tmp_path / "conversation.json"
    path.write_text("{ this is not valid json", encoding="utf-8")
    loaded = ConversationStore.load_and_migrate(path)  # must NOT raise
    assert loaded.to_messages() == []


def test_incompatible_schema_fails_soft(tmp_path: Path) -> None:
    path = tmp_path / "conversation.json"
    # Valid JSON, wrong shape (extra=forbid + bad types) -> empty, not a crash.
    path.write_text(
        json.dumps({"messages": "not a list", "bogus": 1}), encoding="utf-8"
    )
    loaded = ConversationStore.load_and_migrate(path)
    assert loaded.to_messages() == []


def test_unknown_role_survives_round_trip(tmp_path: Path) -> None:
    # role is loose (str) on disk so a future MessageRole member loads on an older build.
    path = tmp_path / "conversation.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "messages": [{"role": "future_role", "text": "hi", "resolved": False}],
                "history": [],
                "usage": {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
            }
        ),
        encoding="utf-8",
    )
    loaded = ConversationStore.load_and_migrate(path)
    msgs = loaded.to_messages()
    assert msgs[0].role == "future_role"  # passed through, not dropped


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
    # FRESH session restores both + usage. No worker is spawned (no turn enqueued), and
    # the client is never called by save/load, so a bare stub suffices.
    from shaderbox.copilot.session import CopilotSession
    from tests.test_copilot_loop import _fake_caps

    def _mk() -> CopilotSession:
        return CopilotSession(
            _fake_caps(edit_errors=[]),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]  # client unused by save/load
            get_project_slug=lambda: "test",
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
    sess.state.usage = SessionUsage(input_tokens=10, output_tokens=5, cost_usd=0.001)
    path = tmp_path / "conversation.json"
    sess.save_conversation(path)
    sess.release()

    fresh = _mk()
    fresh.load_conversation(ConversationStore.load_and_migrate(path))
    assert [(m.role, m.text) for m in fresh.state.messages] == [
        ("user", "hi"),
        ("assistant", "yo"),
    ]
    assert [h.role for h in fresh.history] == ["user", "assistant"]
    assert fresh.state.usage.input_tokens == 10
    fresh.release()
