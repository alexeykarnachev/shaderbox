"""One dogfood turn from the command line: create or resume a project, optionally open the next
attempt of an experiment, send one message, drive it, render, dump, and print the summary the
driver reads before composing the next message.

    uv run python scripts/dogfood/drive.py --project new --start rc_full_build \\
        --intent "..." --mode babysat "first ask"
    uv run python scripts/dogfood/drive.py --project <dir> "next ask"
    uv run python scripts/dogfood/drive.py --project <dir> --note "..." --axis fidelity --turn 3 "ask"
    uv run python scripts/dogfood/drive.py --project <dir> --end built "summary"

`SHADERBOX_DATA_DIR` is read at import like every harness command (the `/dogfood` skill, §1).
The message is one turn; the driver reads the summary and composes the next -- this is a
command, not a script of replies.
"""

import argparse
import sys
from pathlib import Path

from scripts.dogfood import DogfoodHarness
from shaderbox.copilot.config import COPILOT_CONFIG

_DUMP = Path("scripts/dogfood/runs/turn.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--project", required=True, help="a project dir to resume, or `new`"
    )
    ap.add_argument("--start", metavar="EXPERIMENT", help="open a new experiment here")
    ap.add_argument("--intent", default="")
    ap.add_argument("--mode", default="babysat")
    ap.add_argument(
        "--attempt", metavar="EXPERIMENT", help="open the next attempt here"
    )
    ap.add_argument("--note", default="")
    ap.add_argument("--axis", default="")
    ap.add_argument("--turn", type=int, default=None)
    ap.add_argument("--budget", type=int, default=600, help="turn time budget, seconds")
    ap.add_argument("--strip", default="", help="comma-separated sample times")
    ap.add_argument("--mp4", type=float, default=0.0, help="seconds of mp4 to render")
    ap.add_argument("--size", type=int, default=640, help="longer side of the still")
    ap.add_argument("--end", nargs=2, metavar=("OUTCOME", "SUMMARY"))
    ap.add_argument("message", nargs="?", default="")
    a = ap.parse_args()

    h = DogfoodHarness.create(
        project_dir=None if a.project == "new" else Path(a.project),
        seed_examples=False,
    )
    COPILOT_CONFIG.turn_time_budget_s = a.budget
    if a.start:
        h.start_experiment(a.start, intent=a.intent, mode=a.mode)
    if a.attempt:
        h.start_attempt(a.attempt)
    if a.note:
        h.note(a.note, axis=a.axis, turn=a.turn)
    payload: dict[str, object] | None = None
    if a.message:
        h.send(a.message)
        h.drive_until_idle(auto_approve_gates=True)
        if a.strip:
            h.render_strip([float(x) for x in a.strip.split(",")], size=240)
        if a.mp4:
            h.render_video_mp4(seconds=a.mp4, fps=20, size=320)
        h.render(size=a.size)
        payload = h.dump(_DUMP)
    if a.end:
        h.end_attempt(a.end[0], a.end[1])
    h.release()
    if payload is not None:
        _summary(payload)


def _summary(payload: dict[str, object]) -> None:
    from dogfood.report.log import load_experiment

    station = payload.get("station")
    if not isinstance(station, dict):
        print("(no station attached; the dump is in", _DUMP, ")")
        return
    exp = load_experiment(Path("dogfood/runs") / str(station["experiment_id"]))
    attempt = exp.attempt(int(station["attempt"]))
    if attempt is None or not attempt.turns:
        return
    t = attempt.turns[-1]
    out = sum(int(i.get("output_tokens", 0)) for i in t.iterations)
    rsn = sum(int(i.get("reasoning_tokens", 0)) for i in t.iterations)
    print("\n=== SUMMARY ===")
    print(
        f"project={payload['project_dir']} attempt={attempt.n} turn={t.n} "
        f"terminal={t.terminal} cutoff={t.cutoff} cost=${t.cost_usd:.3f} "
        f"dur={t.payload.get('duration_s')}s reqs={len(t.iterations)} "
        f"reasoning {rsn}/{out} out tokens; renders={[r['label'] for r in t.renders]}"
    )
    for c in t.tool_calls:
        print(f"  {c['name']:14} ok={c['ok']} -> {(c['result'] or '')[:110]!r}")
    print("REPLY:", t.assistant_text[:900])
    sys.stdout.flush()


if __name__ == "__main__":
    main()
