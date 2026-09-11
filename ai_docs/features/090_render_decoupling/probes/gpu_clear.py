# Imported by the probes; refuses to measure while another GPU client is busy.
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


def wait_for_idle(max_wait: float = 300.0, threshold: int = 20, settle: float = 3.0) -> None:
    """Block until the GPU has been quiet for `settle` seconds.

    A second session sharing this box held the GPU at 100% during an early tile-cost run and
    put 200 ms outliers into numbers that should have been 103 ms. Every probe waits now.
    """
    deadline = time.time() + max_wait
    quiet_since: float | None = None
    while time.time() < deadline:
        if _util() <= threshold:
            if quiet_since is None:
                quiet_since = time.time()
            elif time.time() - quiet_since >= settle:
                return
        else:
            quiet_since = None
        time.sleep(0.5)
    raise SystemExit("GPU never went idle; refusing to measure against a contended GPU")
