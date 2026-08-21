"""ShaderLibFavoritesStore — the sidecar JSON favourites store.

It lives in `app_data_dir()`, outside git and with no backup, and `ProjectSession.__init__`
loads it unguarded — so a malformed file used to raise `TypeError` straight out of startup
(the `except` clause caught only `OSError`/`JSONDecodeError`). A bad entry must cost that
entry, never the store and never the app's ability to start.
"""

import json
from pathlib import Path
from typing import Any

from shaderbox.shader_lib.favorites import ShaderLibFavoritesStore


def _isolate_app_data(monkeypatch: Any, tmp_path: Path) -> None:
    monkeypatch.setenv("SHADERBOX_DATA_DIR", str(tmp_path))


def test_round_trip(monkeypatch: Any, tmp_path: Path) -> None:
    _isolate_app_data(monkeypatch, tmp_path)
    store = ShaderLibFavoritesStore()
    store.toggle("SB_fbm")
    assert ShaderLibFavoritesStore.load().favorites == {"SB_fbm"}


def test_a_malformed_entry_costs_that_entry_not_the_store(
    monkeypatch: Any, tmp_path: Path
) -> None:
    _isolate_app_data(monkeypatch, tmp_path)
    (tmp_path / "shader_lib_favorites.json").write_text(
        json.dumps({"favorites": ["SB_keep", None, 7, "  "]})
    )

    assert ShaderLibFavoritesStore.load().favorites == {"SB_keep"}


def test_a_null_favorites_list_falls_back_without_raising(
    monkeypatch: Any, tmp_path: Path
) -> None:
    _isolate_app_data(monkeypatch, tmp_path)
    (tmp_path / "shader_lib_favorites.json").write_text(json.dumps({"favorites": None}))
    assert ShaderLibFavoritesStore.load().favorites == set()


def test_unreadable_json_returns_empty(monkeypatch: Any, tmp_path: Path) -> None:
    _isolate_app_data(monkeypatch, tmp_path)
    (tmp_path / "shader_lib_favorites.json").write_text("not valid json {")
    assert ShaderLibFavoritesStore.load().favorites == set()
