"""The station recorder (075 W-3): a trace listener that turns one driven turn into `context`
and `turn` records with no logging call in the driver's command.

Falsifiers: a turn driven through run_turn with only the listener attached must leave a turn
record carrying its tool calls, per-request usage and terminal; each context record must carry
the billed usage of ITS request; a resumed recorder must number the next turn past a turn whose
process died before dump (else that turn's context would be inherited); attempt N+1 must record
the commits landed since attempt N's sha."""

import subprocess
import threading
from pathlib import Path

from dogfood.report.log import LOG_NAME, load_experiment, read_events
from dogfood.report.station import POINTER_NAME, StationRecorder
from shaderbox.copilot.agent import run_turn
from shaderbox.copilot.config import COPILOT_CONFIG
from shaderbox.copilot.gate import GateChannel
from shaderbox.copilot.llm.api import (
    LLMDone,
    LLMMessage,
    LLMStreamEvent,
    LLMTextDelta,
    LLMToolCallCompleted,
    LLMToolCallStarted,
    LLMUsage,
)
from shaderbox.copilot.tools.registry import build_registry
from shaderbox.copilot.trace import TraceLog
from tests._caps import minimal_caps
from tests.test_copilot_loop import _fake_context, _FakeClient


def _git_repo(path: Path) -> Path:
    path.mkdir()
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    (path / "f").write_text("1")
    subprocess.run(["git", "add", "f"], cwd=path, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "first"],
        cwd=path,
        check=True,
        env={**env, "HOME": str(path)},
    )
    return path


def _commit(repo: Path, subject: str) -> str:
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
        "HOME": str(repo),
    }
    (repo / "f").write_text(subject)
    subprocess.run(["git", "commit", "-qam", subject], cwd=repo, check=True, env=env)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _drive(
    recorder: StationRecorder, scripts: list[list[LLMStreamEvent]], user_text: str
) -> None:
    list(
        run_turn(
            _FakeClient(scripts),
            build_registry(minimal_caps()),
            COPILOT_CONFIG,
            _fake_context(),
            history=[],
            user_text=user_text,
            gate=GateChannel(),
            cancel=threading.Event(),
            trace=TraceLog(Path(), [recorder.on_trace]),
            scratchpad_render=lambda: [
                LLMMessage(role="user", content="WORKING SET x")
            ],
        )
    )


def _read_then_reply() -> list[list[LLMStreamEvent]]:
    return [
        [
            LLMToolCallStarted(index=0, id="c1", name="read_shader"),
            LLMToolCallCompleted(
                index=0, id="c1", name="read_shader", arguments='{"documents": ["d"]}'
            ),
            LLMDone(
                finish_reason="tool_calls",
                usage=LLMUsage(
                    input_tokens=1000,
                    output_tokens=20,
                    cached_tokens=800,
                    cost_usd=0.001,
                ),
            ),
        ],
        [
            LLMTextDelta("read it"),
            LLMDone(
                "stop",
                usage=LLMUsage(input_tokens=1200, output_tokens=10, cost_usd=0.002),
            ),
        ],
    ]


def test_a_driven_turn_is_recorded_without_a_logging_call(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path / "repo")
    store = tmp_path / "store"
    project = tmp_path / "proj"
    rec = StationRecorder.start_experiment(
        project,
        "exp",
        intent="i",
        mode="babysat",
        model="m",
        store=store,
        repo_root=repo,
    )
    assert (project / POINTER_NAME).exists()
    _drive(rec, _read_then_reply(), "read the shader")
    render = tmp_path / "renders" / "d_t0.000.png"
    render.parent.mkdir()
    render.write_bytes(b"png")
    recorded = rec.record_turn(
        assistant_text="read it", renders=[render], renders_root=render.parent
    )
    assert recorded is not None and recorded["n"] == 1

    exp = load_experiment(store / "exp")
    attempt = exp.attempts[0]
    assert attempt.model == "m" and len(attempt.sha) == 40
    turn = attempt.turns[0]
    assert turn.user_text == "read the shader" and turn.assistant_text == "read it"
    assert [c["name"] for c in turn.tool_calls] == ["read_shader"]
    assert turn.tool_calls[0]["args"] == {"documents": ["d"]}
    assert turn.terminal == "turn_done" and turn.cutoff == ""
    assert turn.usage["input_tokens"] == 2200 and turn.usage["cached_tokens"] == 800
    assert abs(turn.cost_usd - 0.003) < 1e-9 and turn.peak_input_tokens == 1200
    # One context record per request, each joined to ITS billed usage.
    assert [c.iteration for c in turn.contexts] == [0, 1]
    assert turn.contexts[0].billed == {
        "input_tokens": 1000,
        "output_tokens": 20,
        "reasoning_tokens": 0,
        "cached_tokens": 800,
        "cost_usd": 0.001,
    }
    assert (
        turn.contexts[1].billed is not None
        and turn.contexts[1].billed["input_tokens"] == 1200
    )
    names = {b["name"] for b in turn.contexts[0].blocks}
    assert {"static", "working_set", "turn_exchange"} <= names
    ws = next(b for b in turn.contexts[0].blocks if b["name"] == "working_set")
    assert "WORKING SET x" in ws["text"]
    # The render was copied into the store and recorded relative to the experiment dir.
    assert turn.renders[0]["path"] == "media/1/t001_d_t0.000.png"
    assert (store / "exp" / turn.renders[0]["path"]).read_bytes() == b"png"
    # A second record_turn with no turn driven writes nothing.
    assert (
        rec.record_turn(assistant_text="", renders=[], renders_root=render.parent)
        is None
    )


def test_resume_numbers_past_a_turn_that_died_before_dump(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path / "repo")
    store, project = tmp_path / "store", tmp_path / "proj"
    rec = StationRecorder.start_experiment(
        project,
        "exp",
        intent="i",
        mode="free_run",
        model="m",
        store=store,
        repo_root=repo,
    )
    _drive(rec, _read_then_reply(), "turn one")
    rec.record_turn(assistant_text="", renders=[], renders_root=tmp_path)
    # Turn 2 runs, its context lands, the process dies before dump().
    _drive(rec, _read_then_reply(), "turn two")
    resumed = StationRecorder.resume(project, repo_root=repo)
    assert resumed is not None and resumed.attempt == 1
    _drive(resumed, _read_then_reply(), "turn three")
    resumed.record_turn(assistant_text="", renders=[], renders_root=tmp_path)
    attempt = load_experiment(store / "exp").attempts[0]
    assert [t.n for t in attempt.turns] == [1, 3]
    assert {c.turn for c in attempt.orphan_contexts} == {2}


def test_flush_records_an_interrupted_turn(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path / "repo")
    store, project = tmp_path / "store", tmp_path / "proj"
    rec = StationRecorder.start_experiment(
        project,
        "exp",
        intent="i",
        mode="free_run",
        model="m",
        store=store,
        repo_root=repo,
    )
    _drive(rec, _read_then_reply(), "turn one")
    rec.flush()
    turn = load_experiment(store / "exp").attempts[0].turns[0]
    assert turn.user_text == "turn one" and turn.terminal == "turn_done"


def test_next_attempt_records_the_commits_landed_between(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path / "repo")
    store = tmp_path / "store"
    first = StationRecorder.start_experiment(
        tmp_path / "p1",
        "exp",
        intent="i",
        mode="end_to_end",
        model="m1",
        store=store,
        repo_root=repo,
    )
    first.end_attempt("abandoned", "stuck on the merge")
    sha_a = _commit(repo, "fix the merge")
    sha_b = _commit(repo, "and its test")
    second = StationRecorder.start_attempt(
        tmp_path / "p2", "exp", model="m2", store=store, repo_root=repo
    )
    assert second.attempt == 2
    exp = load_experiment(store / "exp")
    a1, a2 = exp.attempts
    assert a1.outcome == "abandoned" and a1.summary == "stuck on the merge"
    assert [(f.sha, f.subject) for f in a2.fixes] == [
        (sha_a, "fix the merge"),
        (sha_b, "and its test"),
    ]
    assert a2.model == "m2" and a2.sha == sha_b
    # The fixes were written before the attempt_start, in log order.
    kinds = [e.kind for e in read_events(store / "exp" / LOG_NAME)[0]]
    assert kinds[-3:] == ["fix", "fix", "attempt_start"]


def test_start_experiment_refuses_a_duplicate_and_a_bad_mode(tmp_path: Path) -> None:
    repo = _git_repo(tmp_path / "repo")
    store = tmp_path / "store"
    StationRecorder.start_experiment(
        tmp_path / "p",
        "exp",
        intent="i",
        mode="babysat",
        model="m",
        store=store,
        repo_root=repo,
    )
    for kwargs in ({"mode": "babysat"}, {"mode": "scripted"}):
        try:
            StationRecorder.start_experiment(
                tmp_path / "q",
                "exp",
                intent="i",
                model="m",
                store=store,
                repo_root=repo,
                **kwargs,
            )
        except ValueError:
            continue
        raise AssertionError(f"accepted {kwargs}")
