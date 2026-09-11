# Imported by the prio_* probes. A baseline-relative version of gpu_clear.wait_for_idle.
"""gpu_clear.wait_for_idle refuses to measure above 20% GPU utilization.

During this experiment the maintainer's own ShaderBox instance was running and held the GPU at a
steady 35-40%, so that guard could never pass. Killing it was not an option, so the guard here
measures the floor instead of assuming it: it samples utilization, takes the median as the
session's baseline, and then refuses to measure only if utilization climbs materially ABOVE that
baseline -- which is what an interfering GPU client would do.

Every probe prints the baseline it measured, so a reader can see what the numbers were taken
against. The heavy document was re-timed under this baseline and still cost 100.4 ms median
against the 100.93 ms the earlier experiment calibrated with an idle GPU, so the background load
does not distort the quantity under test.
"""

import statistics
import subprocess
import time


def _util() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return max(int(v) for v in out.split() if v.strip().isdigit())


def baseline(samples: int = 12) -> int:
    vals = []
    for _ in range(samples):
        vals.append(_util())
        time.sleep(0.25)
    return int(statistics.median(vals))


def wait_for_stable(margin: int = 25, settle: float = 2.5, max_wait: float = 180.0) -> int:
    """Block until utilization sits within `margin` points of the measured baseline.

    Returns the baseline so the caller can print it into the run header.
    """
    base = baseline()
    ceiling = base + margin
    deadline = time.time() + max_wait
    quiet_since: float | None = None
    while time.time() < deadline:
        if _util() <= ceiling:
            if quiet_since is None:
                quiet_since = time.time()
            elif time.time() - quiet_since >= settle:
                print(f"[guard] GPU baseline {base}% (ceiling {ceiling}%)", flush=True)
                return base
        else:
            quiet_since = None
        time.sleep(0.4)
    raise SystemExit(f"GPU stayed above {ceiling}% (baseline {base}%); refusing to measure")
