"""The dogfooding station's event log (075 W-0): an append-only JSONL file that stays readable
under the harness's one-process-per-turn shape and after a kill mid-write.

Falsifiers: concurrent appenders from separate writer objects (the per-turn-process shape) must
leave every line whole; a torn tail must cost exactly the torn record and the NEXT append must not
glue itself onto it; `reconstruct` must give back the run from the records alone."""

import json
import threading
from pathlib import Path

from dogfood.report.log import (
    KINDS,
    LOG_NAME,
    EventLog,
    load_store,
    read_events,
    reconstruct,
)


def _log(tmp_path: Path, experiment_id: str = "exp") -> EventLog:
    return EventLog(tmp_path / experiment_id / LOG_NAME, experiment_id)


def test_records_carry_the_required_fields_and_a_utc_offset(tmp_path: Path) -> None:
    log = _log(tmp_path)
    log.append("experiment_start", 0, {"intent": "x", "mode": "free_run"})
    raw = json.loads((tmp_path / "exp" / LOG_NAME).read_text().splitlines()[0])
    assert set(raw) == {"experiment_id", "attempt", "ts", "kind", "payload"}
    assert raw["experiment_id"] == "exp" and raw["kind"] == "experiment_start"
    # An explicit offset, never a bare local stamp.
    assert raw["ts"].endswith("+00:00")


def test_unknown_kind_is_rejected_before_it_lands(tmp_path: Path) -> None:
    log = _log(tmp_path)
    try:
        log.append("verdict", 1, {})
    except ValueError as exc:
        assert "verdict" in str(exc)
    else:
        raise AssertionError("an unknown kind was written")
    assert not (tmp_path / "exp" / LOG_NAME).exists()


def test_concurrent_writers_interleave_whole_lines(tmp_path: Path) -> None:
    # One EventLog per thread models one process per turn: each open+append+close is its own
    # writer; nothing shares a handle or a lock.
    path = tmp_path / "exp" / LOG_NAME
    per_writer, writers = 40, 8

    def run(w: int) -> None:
        log = EventLog(path, "exp")
        for i in range(per_writer):
            log.append("note", 1, {"text": f"w{w}-{i} " + "x" * 300, "writer": w})

    threads = [threading.Thread(target=run, args=(w,)) for w in range(writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    events, warnings = read_events(path)
    assert warnings == []
    assert len(events) == per_writer * writers
    # Every writer's records are all present and each is whole.
    seen = {(e.payload["writer"], e.payload["text"].split()[0]) for e in events}
    assert len(seen) == per_writer * writers


def test_torn_tail_costs_one_record_and_never_glues_the_next(tmp_path: Path) -> None:
    log = _log(tmp_path)
    log.append("experiment_start", 0, {"intent": "i", "mode": "babysat"})
    log.append("attempt_start", 1, {"model": "m"})
    path = tmp_path / "exp" / LOG_NAME
    # A process killed mid-write leaves a partial line with no newline.
    with path.open("ab") as fh:
        fh.write(b'{"experiment_id": "exp", "attempt": 1, "ts": "2026-0')
    events, warnings = read_events(path)
    assert [e.kind for e in events] == ["experiment_start", "attempt_start"]
    assert len(warnings) == 1 and ":3:" in warnings[0]
    # The next append repairs the line boundary: the torn fragment stays its own line.
    log.append("note", 1, {"text": "after the kill"})
    events, warnings = read_events(path)
    assert [e.kind for e in events] == ["experiment_start", "attempt_start", "note"]
    assert len(warnings) == 1
    assert len(path.read_text().splitlines()) == 4


def test_reconstruct_rebuilds_the_run_from_records_alone(tmp_path: Path) -> None:
    log = _log(tmp_path)
    log.append(
        "experiment_start",
        0,
        {"intent": "a working thing", "mode": "end_to_end", "criteria": ["c1"]},
    )
    log.append("attempt_start", 1, {"model": "m1", "sha": "abc"})
    log.append(
        "context",
        1,
        {"turn": 1, "iteration": 0, "blocks": [], "est_total_tokens": 10},
    )
    log.append(
        "turn",
        1,
        {
            "n": 1,
            "user_text": "make it red",
            "assistant_text": "done",
            "iterations": [{"input_tokens": 12}, {"input_tokens": 30}],
            "usage": {"cost_usd": 0.5},
            "terminal": "turn_done",
        },
    )
    log.append("note", 1, {"text": "looks fine", "axis": "fidelity", "turn": 1})
    log.append("context", 1, {"turn": 2, "iteration": 0, "blocks": []})
    log.append("attempt_end", 1, {"outcome": "abandoned", "summary": "s"})
    log.append("fix", 2, {"sha": "def", "subject": "fix it", "body": ""})
    log.append("attempt_start", 2, {"model": "m2", "sha": "def"})

    exp = reconstruct(read_events(tmp_path / "exp" / LOG_NAME)[0])
    assert exp.id == "exp" and exp.mode == "end_to_end" and exp.criteria == ["c1"]
    assert [a.n for a in exp.attempts] == [1, 2]
    a1, a2 = exp.attempts
    assert a1.model == "m1" and a1.outcome == "abandoned" and not a1.live
    turn = a1.turns[0]
    assert turn.user_text == "make it red" and turn.peak_input_tokens == 30
    assert turn.cost_usd == 0.5 and a1.cost_usd == 0.5 and exp.cost_usd == 0.5
    # The context record that arrived before the turn record is joined onto it by number.
    assert [c.iteration for c in turn.contexts] == [0]
    # A context for a turn that never landed (a killed process) is kept, not lost.
    assert [c.turn for c in a1.orphan_contexts] == [2]
    assert a1.notes[0].axis == "fidelity" and a1.notes[0].turn == 1
    assert a2.live and a2.fixes[0].sha == "def" and a2.model == "m2"
    assert exp.live


def test_store_lists_experiments_newest_activity_first(tmp_path: Path) -> None:
    old = _log(tmp_path, "older")
    old.append("experiment_start", 0, {"intent": "o", "mode": "free_run"})
    new = _log(tmp_path, "newer")
    new.append("experiment_start", 0, {"intent": "n", "mode": "free_run"})
    old.append("note", 1, {"text": "activity on the old one"})
    (tmp_path / "not_a_run").mkdir()
    assert [e.id for e in load_store(tmp_path)] == ["older", "newer"]


def test_every_kind_reconstructs_without_a_warning(tmp_path: Path) -> None:
    # The reader's kind dispatch covers the writer's whole vocabulary; a kind added to KINDS
    # without a reconstruct branch surfaces here as an unknown-kind warning.
    log = _log(tmp_path)
    for kind in KINDS:
        log.append(kind, 1, {})
    exp = reconstruct(read_events(tmp_path / "exp" / LOG_NAME)[0])
    assert exp.warnings == []
