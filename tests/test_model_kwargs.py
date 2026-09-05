"""No call site passes a field its model does not declare (079 final review).

A pydantic model whose `extra` is unset DROPS an unknown key in silence. The persisted models
must stay that way — `extra="forbid"` would make an older `document.json` fail to load, which
the persistence posture forbids — so the guard belongs on the CALL SITES instead.

Two of these had already shipped: `PassEntry(inputs=...)` in the smoke canary (072 removed
`inputs`, so the canary armed nothing and passed anyway) and `TargetConfig(persist=True)` in a
test whose name still promised "and persists". Both read as correct and did nothing.
"""

import ast
import importlib
import pkgutil
from pathlib import Path

from pydantic import BaseModel

import shaderbox

_ROOT = Path(shaderbox.__file__).resolve().parent.parent
_SEARCHED: tuple[str, ...] = ("shaderbox", "tests", "scripts", "dogfood")


def _models() -> dict[str, type[BaseModel]]:
    """Every pydantic model the package defines, by class name."""
    found: dict[str, type[BaseModel]] = {}
    for info in pkgutil.walk_packages(shaderbox.__path__, "shaderbox."):
        if "resources" in info.name:
            continue
        try:
            module = importlib.import_module(info.name)
        except Exception:  # a module needing a GL context or a display
            continue
        for value in vars(module).values():
            if (
                isinstance(value, type)
                and issubclass(value, BaseModel)
                and value is not BaseModel
                and value.__module__.startswith("shaderbox")
            ):
                found[value.__name__] = value
    return found


def _construction_sites() -> list[tuple[Path, str, str, int]]:
    """(file, model name, keyword, line) for every `Model(field=...)` call we can read."""
    models = _models()
    sites: list[tuple[Path, str, str, int]] = []
    for root in _SEARCHED:
        for path in sorted((_ROOT / root).rglob("*.py")):
            if "__pycache__" in path.parts or "runs" in path.parts:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = node.func.id if isinstance(node.func, ast.Name) else None
                if name is None or name not in models:
                    continue
                for keyword in node.keywords:
                    if keyword.arg is not None:
                        sites.append((path, name, keyword.arg, node.lineno))
    return sites


def test_the_sweep_finds_construction_sites_to_check() -> None:
    # A walk that matches nothing passes the assertion below it.
    sites = _construction_sites()
    assert len(sites) >= 20, f"the sweep found only {len(sites)} keyword arguments"


def test_no_call_site_names_a_field_its_model_dropped() -> None:
    # Falsifier: pass a field a model does not declare — `PassEntry(inputs={})` or
    # `TargetConfig(persist=True)` — and this names the file, the model and the keyword.
    models = _models()
    unknown = [
        f"{path.relative_to(_ROOT)}:{line} — {model}({keyword}=...) is not a field of {model}"
        for path, model, keyword, line in _construction_sites()
        if keyword not in models[model].model_fields
    ]
    assert not unknown, "pydantic drops these in silence:\n  " + "\n  ".join(unknown)


def test_the_walk_reaches_the_models_that_persist() -> None:
    # The check above can only see models this walk imports. These are the ones whose `extra`
    # is unset — the persisted shapes, which must stay loadable from an older file and so
    # cannot take `extra="forbid"`. They are exactly the models a dropped keyword can hide in.
    reachable = _models()
    for name in (
        "PassEntry",
        "PassGraph",
        "TargetConfig",
        "RenderPreset",
        "UIDocument",
        "UIDocumentState",
        "UIUniform",
        "EditorSettings",
    ):
        assert name in reachable, f"the walk cannot see {name}; the check has a hole"
        assert reachable[name].model_config.get("extra") != "forbid", (
            f"{name} now forbids extras — pydantic raises there, so this walk is no longer "
            "what protects it and the list above should shrink"
        )
