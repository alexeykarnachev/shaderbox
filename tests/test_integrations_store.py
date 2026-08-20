"""Credential survival across a retired field (feature 060). `IntegrationsStore` is extra="forbid",
so an unrecognized key in integrations.json used to hard-fail load() into all-empty defaults — and
App.save() then wrote those empties over the real OpenRouter / Telegram / YouTube tokens on quit.
A removed feature's leftover key (058's vision_* is the live example) must cost the user that
setting, never their credentials. Pure: no GL, no network; SHADERBOX_DATA_DIR redirects the store."""

import json
from pathlib import Path

import pytest

from shaderbox.integrations import IntegrationsStore

_KEY = "sk-or-fake-openrouter-key"
_BOT = "1234567890:AAfake_bot_token"


@pytest.fixture
def store_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SHADERBOX_DATA_DIR", str(tmp_path))
    return tmp_path / "integrations.json"


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=4), encoding="utf-8")


def test_retired_nested_key_does_not_wipe_credentials(store_path: Path) -> None:
    # The 058 shape: vision_* retired from CopilotIntegration, still present on an older machine.
    _write(
        store_path,
        {
            "copilot": {
                "openrouter_key": _KEY,
                "vision_enabled": True,
                "vision_model": "m",
            },
            "telegram": {"bot_token": _BOT},
        },
    )
    loaded = IntegrationsStore.load()
    assert loaded.copilot.openrouter_key == _KEY
    assert loaded.telegram.bot_token == _BOT

    # The save-on-quit that made the loss permanent must now round-trip the real values.
    loaded.save()
    after = json.loads(store_path.read_text(encoding="utf-8"))
    assert after["copilot"]["openrouter_key"] == _KEY
    assert after["telegram"]["bot_token"] == _BOT
    assert "vision_enabled" not in after["copilot"]  # retired key dropped, not carried


def test_retired_top_level_section_does_not_wipe_credentials(store_path: Path) -> None:
    _write(
        store_path, {"copilot": {"openrouter_key": _KEY}, "retired_section": {"a": 1}}
    )
    assert IntegrationsStore.load().copilot.openrouter_key == _KEY


def test_genuinely_bad_typed_value_still_falls_back(store_path: Path) -> None:
    # Pruning only removes UNKNOWN keys; a known field holding an unusable value is still a
    # fail-soft to defaults (the file is corrupt, not merely from another build).
    _write(store_path, {"copilot": "not-an-object"})
    assert IntegrationsStore.load().copilot.openrouter_key == ""


def test_missing_file_is_defaults(store_path: Path) -> None:
    assert not store_path.exists()
    assert IntegrationsStore.load().telegram.bot_token == ""


def test_retired_key_inside_a_pack_entry_does_not_wipe_credentials(
    store_path: Path,
) -> None:
    # `telegram.packs` is a list[PackEntry], and PackEntry forbids extras too — so the pruner has to
    # descend into LIST elements, not just nested dicts. Falsifier: drop the list branch from
    # _drop_unknown and the bot token below comes back empty.
    _write(
        store_path,
        {
            "copilot": {"openrouter_key": _KEY},
            "telegram": {
                "bot_token": _BOT,
                "packs": [{"set_name": "p1", "title": "T", "retired_field": 1}],
            },
        },
    )
    loaded = IntegrationsStore.load()
    assert loaded.telegram.bot_token == _BOT
    assert loaded.copilot.openrouter_key == _KEY
    assert [p.set_name for p in loaded.telegram.packs] == ["p1"]
