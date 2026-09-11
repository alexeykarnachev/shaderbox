# Run: cd /home/akarnachev/src/shaderbox && DISPLAY=:1 uv run python ai_docs/features/090_render_decoupling/probes/probe_a_run.py
"""Config A driver: light process alone, then light process while a heavy process saturates the GPU.

Two OS processes, two glfw windows, two independent GL contexts -- the strongest possible
isolation the driver can offer. If the light client still stalls here, no in-process design helps.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from gpu_clear import wait_for_idle

HERE = Path(__file__).parent
PY = [sys.executable]
DUR = 5.0


def run_light(label: str, env_extra: dict[str, str] | None = None) -> str:
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    out = subprocess.run(
        [*PY, str(HERE / "probe_a_light_proc.py"), str(DUR), label],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return out.stdout.strip()


def main() -> None:
    wait_for_idle()
    env_extra: dict[str, str] = {}
    tag = ""
    if len(sys.argv) > 1 and "=" in sys.argv[1]:
        k, v = sys.argv[1].split("=", 1)
        env_extra[k] = v
        tag = f"_{k}={v}"
        print(f"### env override: {k}={v}")

    for trial in (1, 2):
        print(f"===== trial {trial} =====", flush=True)
        print(run_light(f"idle{tag}_t{trial}", env_extra), flush=True)

        heavy = subprocess.Popen(
            [*PY, str(HERE / "probe_a_heavy_proc.py"), str(DUR + 4.0)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, **env_extra},
        )
        assert heavy.stdout is not None
        line = heavy.stdout.readline()
        print(line.strip(), flush=True)
        time.sleep(1.0)  # let the heavy loop reach steady state
        print(run_light(f"running{tag}_t{trial}", env_extra), flush=True)
        rest = heavy.stdout.read()
        heavy.wait()
        print(rest.strip(), flush=True)


main()
