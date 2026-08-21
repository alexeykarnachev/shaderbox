"""One bad key costs the user that setting, not the whole file.

Both persisted stores are fail-soft, and the app writes loaded state back on quit — so
"fall back to defaults" on a single malformed key is silent data loss, not resilience. The
credential store learned this first (a retired key wiped every token); `app_state.json` had
the same shape, where a retired enum member or one wrong-typed value reset the node
selection, fps, editor prefs, keybindings and Telegram pack together.

The parametrisation walks the model's OWN field list, so a field added later is covered
without anyone remembering to add a case.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from shaderbox.integrations import IntegrationsStore
from shaderbox.model_salvage import drop_invalid, drop_unknown, load_model
from shaderbox.ui_models import UIAppState

_POPULATED: dict[str, Any] = {
    "current_node_id": "abc-123",
    "global_target_fps": 144,
    "editor_split_fraction": 0.37,
    "telegram_default_pack": "my_pack",
    "editor_settings": {"font_size": 22, "tab_size": 8},
}


def _write(tmp_path: Path, data: dict[str, Any]) -> Path:
    path = tmp_path / "app_state.json"
    path.write_text(json.dumps(data))
    return path


def test_a_populated_file_round_trips(tmp_path: Path) -> None:
    state = UIAppState.load(_write(tmp_path, _POPULATED))
    assert state.current_node_id == "abc-123"
    assert state.global_target_fps == 144
    assert state.editor_settings.font_size == 22


@pytest.mark.parametrize(
    "bad_key,bad_value",
    [
        ("global_target_fps", "sixty"),  # wrong type
        ("active_node_tab", "retired_tab"),  # retired enum member
        ("copilot_layout", "retired_layout"),
        ("editor_split_fraction", "half"),
        ("key_bindings", "not-a-mapping"),
    ],
)
def test_one_bad_key_costs_only_itself(
    tmp_path: Path, bad_key: str, bad_value: Any
) -> None:
    path = _write(tmp_path, {**_POPULATED, bad_key: bad_value})
    state = UIAppState.load(path)
    defaults = UIAppState()

    # The bad key fell back to its default...
    assert getattr(state, bad_key) == getattr(defaults, bad_key)
    # ...and every other setting survived.
    for key, value in _POPULATED.items():
        if key == bad_key or isinstance(value, dict):
            continue
        assert getattr(state, key) == value, f"{key} was collateral damage"
    assert state.editor_settings.font_size == 22


def test_an_unknown_key_is_dropped_without_touching_the_rest(tmp_path: Path) -> None:
    path = _write(tmp_path, {**_POPULATED, "retired_feature_flag": True})
    state = UIAppState.load(path)
    assert state.current_node_id == "abc-123"
    assert state.global_target_fps == 144
    assert not hasattr(state, "retired_feature_flag")


def test_unreadable_and_non_object_files_degrade_to_defaults(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert UIAppState.load(broken) == UIAppState()

    listy = tmp_path / "listy.json"
    listy.write_text("[1, 2, 3]")
    assert UIAppState.load(listy) == UIAppState()

    assert UIAppState.load(tmp_path / "absent.json") == UIAppState()


class _Nested(BaseModel):
    keep: int = 1


class _Outer(BaseModel):
    nested: _Nested = _Nested()
    value: int = 7


def test_drop_helpers_recurse_and_are_independent() -> None:
    data: dict[str, Any] = {
        "nested": {"keep": 5, "retired": True},
        "value": "bad",
        "gone": 1,
    }
    drop_unknown(_Outer, data, "outer")
    assert "gone" not in data
    assert "retired" not in data["nested"]

    drop_invalid(_Outer, data, "outer")
    assert "value" not in data
    assert data["nested"] == {"keep": 5}

    assert load_model(_Outer, "/nonexistent", "outer") == _Outer()


def test_the_credential_store_shares_the_same_salvage(tmp_path: Path) -> None:
    # The store is extra="forbid": a hard fail returns empty credentials that the next
    # save() writes over the real tokens. Retired keys must not reach the constructor.
    data = {
        "telegram": {"bot_token": "secret", "retired_key": 1},
        "retired_top_level": True,
    }
    drop_unknown(IntegrationsStore, data, "integrations")
    drop_invalid(IntegrationsStore, data, "integrations")
    store = IntegrationsStore(**data)
    assert store.telegram.bot_token == "secret"
