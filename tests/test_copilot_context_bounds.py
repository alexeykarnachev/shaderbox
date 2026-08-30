"""Feature 056 slice D: the working set is bounded and turn-scoped. The per-turn reset is a
CAPABILITY called at the one choke point (so a session driven without the App resets too), and the
add seam is a true LRU with a loud eviction record. Bare objects — no app fixture, no GL."""

import types
from pathlib import Path
from typing import Any, cast

from shaderbox.copilot.backend import CopilotBackend
from shaderbox.copilot.config import COPILOT_ENGINE
from shaderbox.copilot.llm.openrouter import OpenRouterLLMClient
from shaderbox.copilot.session import CopilotSession
from shaderbox.project_session import ProjectSession
from tests._caps import minimal_caps


def _ws_stub() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        _copilot_working_set=[], _copilot_working_set_evicted=[]
    )


def _add(stub: types.SimpleNamespace) -> Any:
    return ProjectSession._copilot_ws_add.__get__(stub)


# ---- D1: the reset rides the session, not the App ----


def test_enqueue_turn_resets_the_working_set(tmp_path: Path) -> None:
    # A CopilotSession driven directly (the harness's multi-send mode, any headless driver) used
    # to accrete the working set across turns — unbounded context and cost semantics diverging
    # from the App. No LLM needed: the reset happens at enqueue.
    working_set: list[str] = ["7f3a", "lib:glow.glsl"]
    caps = minimal_caps(reset_working_set=lambda: working_set.clear())
    sess = CopilotSession(
        caps,
        cast(OpenRouterLLMClient, object()),
        get_project_slug=lambda: "test",
        get_checkpoints_root=lambda: tmp_path / "checkpoints",
    )
    try:
        sess.enqueue_turn("first")
        assert working_set == []
        working_set.extend(["7f3a"])  # the turn touched a document
        sess.enqueue_turn("second")
        assert working_set == []
    finally:
        sess.release()


def test_reset_clears_the_eviction_record_too() -> None:
    stub = _ws_stub()
    stub._copilot_working_set = ["a"]
    stub._copilot_working_set_evicted = ["b"]
    ProjectSession._copilot_ws_reset.__get__(stub)()
    assert stub._copilot_working_set == []
    assert stub._copilot_working_set_evicted == []


# ---- D3: the LRU + the loud eviction ----


def test_add_seam_evicts_the_least_recently_touched() -> None:
    stub = _ws_stub()
    add = _add(stub)
    cap = COPILOT_ENGINE.copilot_working_set_max_documents
    names = [f"n{i}" for i in range(cap)]
    for name in names:
        add(name)
    add("n0")  # re-touch the oldest -> it becomes the NEWEST
    add("overflow")
    assert len(stub._copilot_working_set) == cap
    assert "n0" in stub._copilot_working_set  # re-touched, so not the victim
    assert stub._copilot_working_set_evicted == ["n1"]  # the true LRU went
    assert stub._copilot_working_set[-1] == "overflow"


def test_a_re_added_member_leaves_the_eviction_record() -> None:
    # The rendered line names what the block no longer shows — a re-read address is back in the
    # block, so claiming it was dropped would be a false statement on the model channel.
    stub = _ws_stub()
    add = _add(stub)
    cap = COPILOT_ENGINE.copilot_working_set_max_documents
    for i in range(cap + 1):
        add(f"n{i}")
    assert stub._copilot_working_set_evicted == ["n0"]
    add("n0")  # re-read: back in the block, so out of the dropped list
    assert "n0" not in stub._copilot_working_set_evicted
    assert "n0" in stub._copilot_working_set


def test_zero_cap_means_uncapped() -> None:
    # Sibling 0=off semantics: 0 must not self-evict every add (the naive loop pops the entry it
    # just appended, so the working set would render nothing but "dropped" lines).
    stub = _ws_stub()
    add = _add(stub)
    original = COPILOT_ENGINE.copilot_working_set_max_documents
    COPILOT_ENGINE.copilot_working_set_max_documents = 0
    try:
        for i in range(original + 4):
            add(f"n{i}")
    finally:
        COPILOT_ENGINE.copilot_working_set_max_documents = original
    assert len(stub._copilot_working_set) == original + 4
    assert stub._copilot_working_set_evicted == []


def _read_working_set_stub(
    evicted: list[str], current: str, members: list[str]
) -> types.SimpleNamespace:
    # CopilotBackend.read_working_set bound onto a namespace: the view builders are stubbed, the
    # evicted-list bookkeeping under test is real.
    def _document_view(full_id: str, short: dict[str, str], cur: str) -> Any:
        return types.SimpleNamespace(address=short.get(full_id, full_id[:4]))

    return types.SimpleNamespace(
        _bridge=types.SimpleNamespace(
            run_on_main=lambda fn, timeout=None, defer=False: fn()
        ),
        _copilot_short_ids=lambda: {m: m[:4] for m in [current, *members, *evicted]},
        _get_current_document_id=lambda: current,
        _get_ui_documents=lambda: {m: object() for m in [current, *members]},
        _working_set_reader=lambda: members,
        _working_set_evicted=lambda: evicted,
        _copilot_document_working_view=_document_view,
        _copilot_lib_working_view=lambda address: types.SimpleNamespace(
            address=address
        ),
    )


def test_evicted_addresses_are_agent_facing_handles() -> None:
    # The model has never seen a 36-char uuid — "re-read to view" must name the handle it knows.
    stub = _read_working_set_stub(
        evicted=["9c1dxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "lib:glow.glsl"],
        current="7f3axxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        members=[],
    )
    _views, evicted = CopilotBackend.read_working_set.__get__(stub)()
    assert evicted == ["9c1d", "lib:glow.glsl"]


def test_an_evicted_but_still_rendered_address_is_not_reported_dropped() -> None:
    # The current document is unioned into the block unconditionally, so an evicted current document shows
    # its FULL source — also calling it dropped would be a falsehood on the model channel.
    current = "7f3axxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    stub = _read_working_set_stub(evicted=[current], current=current, members=[])
    views, evicted = CopilotBackend.read_working_set.__get__(stub)()
    assert [v.address for v in views] == ["7f3a"]
    assert evicted == []


def test_no_eviction_under_the_cap() -> None:
    stub = _ws_stub()
    add = _add(stub)
    for i in range(COPILOT_ENGINE.copilot_working_set_max_documents):
        add(f"n{i}")
    add("n0")  # a re-touch is not a growth
    assert stub._copilot_working_set_evicted == []
