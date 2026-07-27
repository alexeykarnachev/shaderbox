"""Dogfood rig (features 026/027): the headless copilot-engine harness + scenarios + runs.

`from scripts.dogfood import DogfoodHarness` resolves here, LAZILY (PEP 562): the harness module
is imported only when that name is actually read, so `import scripts.dogfood.analyze` /
`scripts.dogfood.judge` pull in no glfw/moderngl and mkdtemp no `runs/data-*` dir.

Importing `harness` runs its module-top env block (SHADERBOX_DATA_DIR + MESA overrides + the
integrations write), so a resuming caller MUST set SHADERBOX_DATA_DIR in the process env on the
command line BEFORE `uv run` — assigning it in-script after import is too late.
"""

import importlib
from typing import Any

_LAZY: tuple[str, ...] = ("DogfoodHarness",)
# No `__all__`: with the name resolved lazily it is not statically present in this module, and
# pyright then reports it as an unsupported dunder-all entry. `__dir__` is PEP 562's own answer for
# discoverability (tab-completion, `dir()`, `help()`) and costs no static-analysis lie.


def __dir__() -> list[str]:
    return sorted([*globals(), *_LAZY])


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module("scripts.dogfood.harness"), name)
