"""Unit tests for LibTagsStore — the sidecar JSON-backed tag store."""

import os
from pathlib import Path
from typing import Any

from shaderbox.lib_tags import LibTagsStore


def _isolate_app_data(monkeypatch: Any, tmp_path: Path) -> None:
    # Point app_data_dir() at a tmp dir for this test.
    monkeypatch.setenv("SHADERBOX_DATA_DIR", str(tmp_path))


def test_empty_store(monkeypatch: Any, tmp_path: Path) -> None:
    _isolate_app_data(monkeypatch, tmp_path)
    store = LibTagsStore.load()
    assert store.tags == {}
    assert store.all_tags() == frozenset()
    assert store.tags_for("SB_anything") == frozenset()


def test_add_and_query(monkeypatch: Any, tmp_path: Path) -> None:
    _isolate_app_data(monkeypatch, tmp_path)
    store = LibTagsStore()
    store.add("SB_hash", "noise")
    store.add("SB_hash", "hash")
    assert store.tags_for("SB_hash") == frozenset({"noise", "hash"})
    assert store.has_tag("SB_hash", "noise")
    assert not store.has_tag("SB_hash", "nope")


def test_remove_drops_function_when_empty(
    monkeypatch: Any, tmp_path: Path
) -> None:
    _isolate_app_data(monkeypatch, tmp_path)
    store = LibTagsStore()
    store.add("SB_x", "a")
    store.remove("SB_x", "a")
    assert store.tags_for("SB_x") == frozenset()
    assert "SB_x" not in store.tags  # cleaned up empty entry


def test_toggle(monkeypatch: Any, tmp_path: Path) -> None:
    _isolate_app_data(monkeypatch, tmp_path)
    store = LibTagsStore()
    store.toggle("SB_x", "color")
    assert store.has_tag("SB_x", "color")
    store.toggle("SB_x", "color")
    assert not store.has_tag("SB_x", "color")


def test_normalization_strips_hash_and_lowercases(
    monkeypatch: Any, tmp_path: Path
) -> None:
    _isolate_app_data(monkeypatch, tmp_path)
    store = LibTagsStore()
    store.add("SB_x", "#Noise")
    store.add("SB_x", "  PRNG  ")
    assert store.tags_for("SB_x") == frozenset({"noise", "prng"})


def test_all_tags_union(monkeypatch: Any, tmp_path: Path) -> None:
    _isolate_app_data(monkeypatch, tmp_path)
    store = LibTagsStore()
    store.add("SB_a", "x")
    store.add("SB_a", "y")
    store.add("SB_b", "y")
    store.add("SB_b", "z")
    assert store.all_tags() == frozenset({"x", "y", "z"})


def test_persistence_roundtrip(monkeypatch: Any, tmp_path: Path) -> None:
    _isolate_app_data(monkeypatch, tmp_path)
    store = LibTagsStore()
    store.add("SB_a", "noise")
    store.add("SB_b", "color")
    # Reload from disk: same content.
    reloaded = LibTagsStore.load()
    assert reloaded.tags_for("SB_a") == frozenset({"noise"})
    assert reloaded.tags_for("SB_b") == frozenset({"color"})


def test_unreadable_json_returns_empty(
    monkeypatch: Any, tmp_path: Path
) -> None:
    _isolate_app_data(monkeypatch, tmp_path)
    (tmp_path / "lib_tags.json").write_text("not valid json {")
    store = LibTagsStore.load()
    assert store.tags == {}


def test_empty_tag_is_dropped(monkeypatch: Any, tmp_path: Path) -> None:
    _isolate_app_data(monkeypatch, tmp_path)
    store = LibTagsStore()
    store.add("SB_x", "")
    store.add("SB_x", "   ")
    assert store.tags_for("SB_x") == frozenset()


# Ensure no lingering env var across tests (the monkeypatch above auto-undoes).
def test_env_isolation_works(monkeypatch: Any, tmp_path: Path) -> None:
    _isolate_app_data(monkeypatch, tmp_path)
    assert os.environ.get("SHADERBOX_DATA_DIR") == str(tmp_path)
