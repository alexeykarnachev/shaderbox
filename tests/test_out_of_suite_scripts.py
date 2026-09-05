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
# Recorded model output kept as evidence, not this repo's code — several predate API changes on
# purpose, and they are gitignored, so including them makes the gate's domain differ per machine.
_NOT_OUR_CODE: tuple[str, ...] = ("runs", "__pycache__")


def _standalone_scripts() -> list[Path]:
    found: list[Path] = []
    for root in _ROOTS:
        for path in sorted((_ROOT / root).rglob("*.py")):
            if not any(part in _NOT_OUR_CODE for part in path.parts):
                found.append(path)
    return found


def _package_imports(path: Path) -> list[tuple[str, str | None]]:
    """Every `shaderbox.*` import the file makes, as (module, name-or-None), by static read.

    The NAME matters, not just the module: `render_document.py` broke by importing a symbol
    that had gone while its module stayed, and a module-only check calls that green. Static
    rather than executed, because importing the script itself would run its module body, which
    for several of these opens a GL context or reads credentials.
    """
    found: list[tuple[str, str | None]] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] == "shaderbox":
                found.extend((node.module, alias.name) for alias in node.names)
        elif isinstance(node, ast.Import):
            found.extend(
                (alias.name, None)
                for alias in node.names
                if alias.name.split(".")[0] == "shaderbox"
            )
    return found


def test_the_sweep_finds_the_scripts_it_is_meant_to_guard() -> None:
    # A glob that matches nothing passes every assertion below it.
    scripts = _standalone_scripts()
    assert len(scripts) >= 3, f"the sweep found only {[str(p) for p in scripts]}"


@pytest.mark.parametrize(
    "script", _standalone_scripts(), ids=lambda p: p.relative_to(_ROOT).as_posix()
)
def test_a_standalone_script_imports_names_that_exist(script: Path) -> None:
    # Falsifier: delete a module one of these imports, OR delete a SYMBOL from a module they
    # import while the module stays — this names the script, the module and the missing name,
    # where the suite alone would stay green.
    for module, name in sorted(
        _package_imports(script), key=lambda pair: (pair[0], pair[1] or "")
    ):
        imported = importlib.import_module(module)
        if name is not None and not hasattr(imported, name):
            # A submodule reached as an attribute (`from shaderbox import scripting`) is not an
            # attribute until it is imported in its own right.
            importlib.import_module(f"{module}.{name}")
