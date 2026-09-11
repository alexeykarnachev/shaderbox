# Run: cd /home/akarnachev/src/shaderbox && DISPLAY=:1 uv run python ai_docs/features/090_render_decoupling/probes/prio_procs.py <heavy_prio> <light_prio> [tiles]
"""P4: the same priority pairing across two OS processes rather than two threads.

Priorities are a per-context attribute, but a driver is free to schedule per process differently,
and the earlier experiment found process isolation bought nothing at equal priority. This asks
whether it buys anything once the priorities differ.
"""

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from prio_guard import wait_for_stable

HERE = Path(__file__).parent
PY = sys.executable


def run(heavy_prio: str, light_prio: str, tiles: int, trial: int) -> None:
    heavy = subprocess.Popen(
        [PY, str(HERE / "prio_heavy_proc.py"), heavy_prio, "14", str(tiles)],
        stdout=subprocess.PIPE, text=True,
    )
    assert heavy.stdout is not None
    print("  " + heavy.stdout.readline().strip(), flush=True)
    time.sleep(1.5)  # let the heavy client fill the queue before measuring
    out = subprocess.run(
        [PY, str(HERE / "prio_light_proc.py"), light_prio, "5",
         f"heavy={heavy_prio} light={light_prio} tiles={tiles} RUNNING t{trial}"],
        capture_output=True, text=True,
    )
    print("  " + out.stdout.strip(), flush=True)
    if out.returncode != 0:
        print("  light proc stderr: " + out.stderr.strip()[-400:], flush=True)
    heavy.wait(timeout=40)
    rest = heavy.stdout.read().strip()
    if rest:
        print("  " + rest, flush=True)


def main() -> None:
    heavy_prio = sys.argv[1] if len(sys.argv) > 1 else "LOW"
    light_prio = sys.argv[2] if len(sys.argv) > 2 else "HIGH"
    tiles = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    wait_for_stable()
    for trial in (1, 2):
        run(heavy_prio, light_prio, tiles, trial)


main()
