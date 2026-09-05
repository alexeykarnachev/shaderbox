"""Scripts the test suite cannot reach still import (079 final review).

`make gates` runs pytest over `tests/`, which imports the package but never the standalone
helpers under `.claude/skills/`, `scripts/` and `dogfood/`. Those are invoked by hand
(`uv run python .claude/skills/shader-lab/render_document.py ...`), so a package symbol they
depend on can be deleted with every gate green — which is exactly what happened to
`render_document.py` when `shaderbox.scripting.outputs` went away.

Imports only: running these needs a GL context, real credentials or a live LLM. The failure this
catches is the one a green suite hides.
"""

import ast
import importlib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_ROOTS: tuple[str, ...] = (".claude/skills", "scripts", "dogfood")


def _standalone_scripts() -> list[Path]:
    found: list[Path] = []
    for root in _ROOTS:
        for path in sorted((_ROOT / root).rglob("*.py")):
            if "__pycache__" not in path.parts:
                found.append(path)
    return found


def _package_imports(path: Path) -> set[str]:
    """Every `shaderbox.*` module the file imports, by static read.

    Static rather than executed: importing the script itself would run its module body, which
    for several of these opens a GL context or reads credentials.
    """
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return {m for m in modules if m.split(".")[0] == "shaderbox"}


def test_the_sweep_finds_the_scripts_it_is_meant_to_guard() -> None:
    # A glob that matches nothing passes every assertion below it.
    scripts = _standalone_scripts()
    assert len(scripts) >= 3, f"the sweep found only {[str(p) for p in scripts]}"


@pytest.mark.parametrize(
    "script", _standalone_scripts(), ids=lambda p: p.relative_to(_ROOT).as_posix()
)
def test_a_standalone_script_imports_a_package_module_that_exists(script: Path) -> None:
    # Falsifier: delete any `shaderbox` module one of these imports and this names the script
    # and the module, where the suite alone would stay green.
    for module in sorted(_package_imports(script)):
        importlib.import_module(module)
