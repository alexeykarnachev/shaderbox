"""Dogfood rig (features 026/027): the headless copilot-engine harness + scenarios + runs.

`from scripts.dogfood import DogfoodHarness` resolves here. Importing this package eagerly runs
`harness.py`'s module-top env block (mkdtemp + SHADERBOX_DATA_DIR setdefault + MESA overrides +
the integrations write), so a resuming caller MUST set SHADERBOX_DATA_DIR in the process env on the
command line BEFORE `uv run` — assigning it in-script after import loses to the already-run setdefault.
"""

from scripts.dogfood.harness import DogfoodHarness

__all__ = ["DogfoodHarness"]
