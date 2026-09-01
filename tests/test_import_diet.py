"""Heavy SDKs stay off the app's import path (066 D3).

`openai` (~0.23s) and the google-auth stack (~0.11s) are needed only for a copilot turn and a
YouTube Connect/upload, so they import lazily behind their seams (`copilot/llm/openrouter.py`,
`exporters/youtube_api.py`). This guard flips red the moment anyone hoists one back to a
module top reachable from `shaderbox.ui`. Run in a subprocess: this test process's own
sys.modules is already polluted by other tests.
"""

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_HEAVY_SDKS = ("openai", "googleapiclient", "google_auth_oauthlib", "google.auth")


def test_importing_the_app_pulls_no_heavy_sdk() -> None:
    code = (
        "import sys\n"
        "import shaderbox.ui\n"
        f"print(','.join(m for m in {_HEAVY_SDKS!r} if m in sys.modules))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO_ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"child failed ({proc.returncode}):\n{proc.stderr}"
    loaded = proc.stdout.strip()
    assert loaded == "", f"heavy SDKs on the app import path: {loaded}"
