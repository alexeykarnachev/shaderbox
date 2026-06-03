"""Headless verification for the copilot gate-UI wave (feature 020·17).

Covers the two pure-function-testable locked decisions the GUI smoke can't reach
(`scripts/smoke.py` never opens the chat):

  A. Persistence round-trip — a resolved-Yes delete card carrying RecoverInfo survives
     ConversationStore save -> load -> to_messages; a pre-recover (v1) file loads soft.
  B. Decline path — a declined delete_node does NOT trip the edit-retry cap (it ends in
     AgentTurnDone, not an AgentError giveup), AND every assistant tool_call still gets a
     matching tool result message (an orphaned tool_call_id 400s the real provider).

Usage: `uv run python scripts/copilot_gate_check.py` (exit 0 on success, non-zero on failure).
No GL context, no network — pure logic over stub client + stub gate.
"""

import json
import sys
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path

from loguru import logger

from shaderbox.copilot.agent import (
    AgentError,
    AgentEvent,
    AgentTurnDone,
    run_turn,
)
from shaderbox.copilot.capabilities import (
    CompileErrorInfo,
    CopilotCapabilities,
    DeleteNodeResult,
    EditResult,
    GrepHit,
    LibCatalogEntry,
    LibFunctionBody,
    NodeTreeEntry,
    SetUniformResult,
    ShaderView,
)
from shaderbox.copilot.config import COPILOT_CONFIG
from shaderbox.copilot.context import build_context
from shaderbox.copilot.gate import GateChannel, GateRequest, GateResponse
from shaderbox.copilot.llm.api import (
    LLMDone,
    LLMMessage,
    LLMStreamEvent,
    LLMTextDelta,
    LLMToolCallCompleted,
    LLMToolCallStarted,
    LLMToolSpec,
    LLMUsage,
)
from shaderbox.copilot.persistence import ConversationStore
from shaderbox.copilot.state import ChatState, Message, RecoverInfo, SessionUsage
from shaderbox.copilot.tools.registry import build_registry


# ---- A. persistence round-trip ----------------------------------------------------------


def _check_persistence() -> None:
    state = ChatState(
        messages=[
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
        ],
        usage=SessionUsage(input_tokens=10, output_tokens=5, cost_usd=0.001),
    )
    history = [
        LLMMessage(role="user", content="delete the gradient node"),
        LLMMessage(role="assistant", content="Done — moved to trash."),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "conversation.json"
        ConversationStore.from_runtime(state, history).save(path)
        loaded = ConversationStore.load_and_migrate(path)
        msgs = loaded.to_messages()

    assert len(msgs) == 3, f"round-trip lost messages: {len(msgs)}"
    card = msgs[1]
    assert card.role == "pending_action" and card.resolved, "card role/resolved lost"
    assert card.recover is not None, "recover field dropped on round-trip"
    assert card.recover.node_id == "full-uuid-1234", "recover.node_id corrupted"
    assert card.recover.trash_name == "full-uuid-1234", "recover.trash_name corrupted"
    assert card.recover.node_name == "Gradient", "recover.node_name corrupted"
    assert not card.recover.done, "recover.done should default False"
    # A non-recover card round-trips with recover=None.
    assert msgs[2].recover is None, "non-recover card grew a recover field"
    logger.info("A1 ok: recover card survives save -> load -> to_messages")

    # A v1-shaped file (no `recover` key) loads soft into the v2 schema.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "v1.json"
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
                    "usage": {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
                }
            )
        )
        loaded = ConversationStore.load_and_migrate(path)
    v1msgs = loaded.to_messages()
    assert len(v1msgs) == 2, "v1 file lost messages"
    assert v1msgs[1].recover is None, "absent recover should load as None"
    logger.info("A2 ok: a pre-recover (v1) file loads soft to recover=None")


# ---- B. decline path: 3 declines, no giveup, no orphaned tool_call -----------------------


class _AlwaysDeclineGate(GateChannel):
    # ask() returns a decline without blocking — no UI thread in this harness.
    def ask(self, request: GateRequest) -> GateResponse:
        _ = request
        return GateResponse(approved=False, option="No")


class _DeleteThenStopClient:
    # Emits a `delete_node` tool call for the first `n_deletes` stream() calls, then a final
    # plain-text reply. Captures the messages list it is handed each call so the test can
    # assert tool-result integrity after the turn.
    def __init__(self, n_deletes: int) -> None:
        self._n_deletes = n_deletes
        self._calls = 0
        self.last_messages: list[LLMMessage] = []

    def stream(
        self,
        messages: list[LLMMessage],
        *,
        tools: list[LLMToolSpec] | None = None,
        max_tokens: int,
    ) -> Iterator[LLMStreamEvent]:
        _ = (tools, max_tokens)
        self.last_messages = list(messages)
        self._calls += 1
        if self._calls <= self._n_deletes:
            call_id = f"call_{self._calls}"
            yield LLMToolCallStarted(index=0, id=call_id, name="delete_node")
            yield LLMToolCallCompleted(
                index=0,
                id=call_id,
                name="delete_node",
                arguments=json.dumps({"node": f"n{self._calls}"}),
            )
            yield LLMDone(finish_reason="tool_calls", usage=LLMUsage(output_tokens=1))
        else:
            yield LLMTextDelta("You said no, so I've cancelled the deletion.")
            yield LLMDone(finish_reason="stop", usage=LLMUsage(output_tokens=1))


def _stub_caps() -> CopilotCapabilities:
    # Minimal caps: delete_node never actually runs (the gate declines first); the others are
    # unreachable in this harness but must satisfy the frozen dataclass.
    def _no_views(_ids: list[str]) -> list[ShaderView]:
        return []

    def _no_tree() -> list[NodeTreeEntry]:
        return []

    def _no_catalog() -> list[LibCatalogEntry]:
        return []

    def _no_grep(_q: str) -> list[GrepHit]:
        return []

    def _no_lib(_n: list[str]) -> list[LibFunctionBody]:
        return []

    def _edit(_a: str, _b: str, _c: bool, _t: str) -> EditResult:
        return EditResult(matches=0, errors=[])

    def _line(_s: int, _e: int, _t: str, _g: str) -> EditResult:
        return EditResult(matches=0, errors=[])

    def _set(_n: str, _v: object, _node: str) -> SetUniformResult:
        return SetUniformResult(ok=False, error="n/a")

    def _create(_n: str, _s: str, _w: bool) -> tuple[str, list[CompileErrorInfo]]:
        return "", []

    def _delete(_node: str) -> DeleteNodeResult:
        return DeleteNodeResult(ok=True, deleted_name="n", node_id="n", trash_name="n")

    return CopilotCapabilities(
        node_tree=_no_tree,
        lib_catalog=_no_catalog,
        read_shaders=_no_views,
        grep=_no_grep,
        read_lib=_no_lib,
        apply_shader_edit=_edit,
        apply_line_edit=_line,
        set_uniform=_set,
        create_node=_create,
        delete_node=_delete,
    )


def _check_decline_path() -> None:
    n = COPILOT_CONFIG.max_edit_retries + 1  # one MORE than the cap, to prove it's not tripped
    caps = _stub_caps()
    registry = build_registry(caps)
    client = _DeleteThenStopClient(n_deletes=n)
    context = build_context(caps)
    gate = _AlwaysDeclineGate()
    cancel = threading.Event()

    events: list[AgentEvent] = list(
        run_turn(
            client,
            registry,
            COPILOT_CONFIG,
            context,
            history=[],
            user_text="delete everything",
            gate=gate,
            cancel=cancel,
        )
    )

    errors = [e for e in events if isinstance(e, AgentError)]
    dones = [e for e in events if isinstance(e, AgentTurnDone)]
    assert not errors, f"a declined delete tripped a giveup/cutoff: {[e.message for e in errors]}"
    assert dones, "turn did not end in AgentTurnDone (the model's comment)"
    logger.info(f"B1 ok: {n} declined deletes end in AgentTurnDone, no giveup")

    # B2: the final messages must carry a matching `tool` result for every assistant tool_call.
    msgs = client.last_messages
    open_ids: set[str] = set()
    result_ids: set[str] = set()
    for m in msgs:
        if m.role == "assistant" and m.tool_calls:
            open_ids.update(tc.id for tc in m.tool_calls)
        if m.role == "tool" and m.tool_call_id:
            result_ids.add(m.tool_call_id)
    missing = open_ids - result_ids
    assert not missing, f"orphaned tool_call_id(s) with no tool result: {missing}"
    assert open_ids, "expected at least one tool_call in the replayed messages"
    logger.info(
        f"B2 ok: every tool_call has a matching tool result ({len(open_ids)} calls)"
    )


def main() -> int:
    try:
        _check_persistence()
        _check_decline_path()
    except AssertionError as e:
        logger.error(f"copilot_gate_check: FAIL — {e}")
        return 1
    except Exception as e:
        logger.exception(f"copilot_gate_check: ERROR — {e}")
        return 1
    logger.info("copilot_gate_check: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
