"""Every persisted store survives a malformed file.

The expensive defect class here is not a broken loader — it is a CORRECT fix applied to some
of the loaders. `model_salvage` was written to stop one bad key wiping a file, then had two
callers when it needed more; `drop_invalid` was written to descend and did not. Both were
found by an audit, which is the slow way.

So this enumerates the persisted stores from a single roster and drives each one against the
same corruption battery. A NEW store is added to the roster (the test names it if you forget,
because the roster is checked against the modules that actually read JSON), and it inherits
every case below — no one has to remember what "the persistence rules" were.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from shaderbox.integrations import IntegrationsStore
from shaderbox.shader_lib.favorites import ShaderLibFavoritesStore
from shaderbox.shader_lib.tags import ShaderLibTagsStore
from shaderbox.ui_models import UIAppState

# The malformed shapes a real file takes: hand-edited, half-written by a crash, or left by an
# older build whose schema has since moved.
_CORRUPTIONS: list[tuple[str, str]] = [
    ("truncated", '{"a": '),
    ("not-json", "not json at all {"),
    ("empty", ""),
    ("a list", "[1, 2, 3]"),
    ("a scalar", "42"),
    ("null", "null"),
    ("nulled values", '{"telegram": null, "favorites": null, "tags": null}'),
    ("wrong-typed values", '{"telegram": 7, "favorites": 7, "global_target_fps": "x"}'),
    ("retired keys", '{"a_retired_key": true, "another": {"nested": 1}}'),
]


def _load_app_state(path: Path) -> object:
    return UIAppState.load(path)


def _load_integrations(path: Path) -> object:
    _ = path  # reads app_data_dir()/integrations.json, isolated by the fixture
    return IntegrationsStore.load()


def _load_tags(path: Path) -> object:
    _ = path
    return ShaderLibTagsStore.load()


def _load_favorites(path: Path) -> object:
    _ = path
    return ShaderLibFavoritesStore.load()


# name -> (on-disk filename, loader). The loader must never raise and never return None.
_STORES: dict[str, tuple[str, Callable[[Path], object]]] = {
    "app_state": ("app_state.json", _load_app_state),
    "integrations": ("integrations.json", _load_integrations),
    "shader_lib_tags": ("shader_lib_tags.json", _load_tags),
    "shader_lib_favorites": ("shader_lib_favorites.json", _load_favorites),
}


@pytest.mark.parametrize("store", sorted(_STORES))
@pytest.mark.parametrize(
    "label,content", _CORRUPTIONS, ids=[c[0] for c in _CORRUPTIONS]
)
def test_a_malformed_file_never_takes_the_app_down(
    monkeypatch: Any, tmp_path: Path, store: str, label: str, content: str
) -> None:
    # These loaders run from ProjectSession.__init__, unguarded. Raising here is not a
    # degraded load — it is an app that will not start, naming no file.
    monkeypatch.setenv("SHADERBOX_DATA_DIR", str(tmp_path))
    filename, loader = _STORES[store]
    path = tmp_path / filename
    path.write_text(content)

    loaded = loader(path)

    assert loaded is not None, f"{store} returned None on a {label} file"


@pytest.mark.parametrize("store", sorted(_STORES))
def test_an_absent_file_loads_defaults(
    monkeypatch: Any, tmp_path: Path, store: str
) -> None:
    monkeypatch.setenv("SHADERBOX_DATA_DIR", str(tmp_path))
    filename, loader = _STORES[store]
    assert loader(tmp_path / filename) is not None


def test_the_roster_covers_every_module_that_loads_a_persisted_store() -> None:
    # The completeness half: a new persisted store must join the roster above, or it silently
    # sits outside every case in this file — which is exactly how model_salvage ended up with
    # two callers when it needed four.
    pkg = Path(__file__).resolve().parent.parent / "shaderbox"
    rostered = {"ui_models.py", "integrations.py", "tags.py", "favorites.py"}
    # Modules that read JSON but are NOT user-facing persisted stores; each is excluded for a
    # stated reason, so the exclusion cannot quietly widen.
    exempt = {
        "model_salvage.py",  # the salvage helper itself
        "core.py",  # node.json, loaded per node with its own skip-and-warn path
        "seed.py",  # reads shipped resources, not user state
        "persistence.py",  # the copilot conversation, rebuilt from scratch when unreadable
        "checkpoint.py",  # per-turn rollback records, pruned and disposable
        "agent.py",  # parses LLM tool-call payloads, not a file
        "youtube_util.py",  # builds request bodies
        "youtube.py",  # OAuth client secrets, supplied by the user per-load
    }

    loaders = {
        path.name for path in pkg.rglob("*.py") if "json.load" in path.read_text()
    }
    unaccounted = loaders - rostered - exempt

    assert not unaccounted, (
        f"{sorted(unaccounted)} read JSON but are neither in the roster above nor exempt. "
        "Add the store to _STORES so it inherits the corruption battery, or list it in "
        "`exempt` with the reason it is not user state."
    )
