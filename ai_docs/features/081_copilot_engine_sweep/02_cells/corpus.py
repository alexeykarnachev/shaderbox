"""Shared loader for the 077 dogfood station corpus.

Every sweep agent imports this so numbers come from ONE code path.
Run from the repo root: sys.path.insert(0, "<scratchpad>/sweep")
"""
import json
from pathlib import Path

ROOT = Path("/home/akarnachev/src/shaderbox")
STATION = ROOT / "dogfood" / "runs"
EXPERIMENTS = ("rc_full_build", "rc_end_to_end")


def events(exp):
    """All events of one experiment, in log order."""
    return [json.loads(l) for l in (STATION / exp / "events.jsonl").open()]


def by_kind(exp, kind):
    return [e for e in events(exp) if e["kind"] == kind]


def attempts(exp):
    """{attempt_no: {model, sha, project_dir, outcome, summary, turns:[payload], notes:[payload]}}"""
    out = {}
    for e in events(exp):
        a = e["attempt"]
        p = e["payload"]
        if e["kind"] == "attempt_start":
            out.setdefault(a, {}).update(
                model=p["model"], sha=p["sha"], project_dir=p["project_dir"],
                turns=[], notes=[], contexts=[], outcome=None, summary=None)
        elif e["kind"] == "attempt_end":
            out.setdefault(a, {}).update(outcome=p["outcome"], summary=p["summary"])
        elif e["kind"] == "turn":
            out.setdefault(a, {}).setdefault("turns", []).append(p)
        elif e["kind"] == "note":
            out.setdefault(a, {}).setdefault("notes", []).append(p)
        elif e["kind"] == "context":
            out.setdefault(a, {}).setdefault("contexts", []).append(p)
    return out


def all_turns():
    """[(exp, attempt, model, turn_payload)] across both experiments."""
    rows = []
    for exp in EXPERIMENTS:
        for a, d in attempts(exp).items():
            for t in d.get("turns", []):
                rows.append((exp, a, d["model"], t))
    return rows


def all_contexts():
    """[(exp, attempt, model, context_payload)] across both experiments."""
    rows = []
    for exp in EXPERIMENTS:
        for a, d in attempts(exp).items():
            for c in d.get("contexts", []):
                rows.append((exp, a, d["model"], c))
    return rows


def all_calls():
    """[(exp, attempt, model, turn_n, call)] — every tool call in the corpus."""
    rows = []
    for exp, a, model, t in all_turns():
        for c in t["tool_calls"]:
            rows.append((exp, a, model, t["n"], c))
    return rows
