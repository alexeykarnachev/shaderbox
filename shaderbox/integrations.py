import json
import threading
from pathlib import Path
from typing import Self

from loguru import logger
from pydantic import BaseModel, ValidationError

from shaderbox.copilot.config import CopilotConfig, apply_user_limits
from shaderbox.model_salvage import drop_invalid, drop_unknown
from shaderbox.paths import app_data_dir

_STORE_FILE = "integrations.json"
# CopilotConfig is slotted, so its fields are only readable off an instance.
_COPILOT_DEFAULTS = CopilotConfig()
# Serializes save() across the render thread (Ctrl+S, disconnect) and exporter
# worker threads (token write on connect/refresh) — two interleaved json.dump
# writes would corrupt the file.
_SAVE_LOCK = threading.Lock()


def _file_path() -> Path:
    return app_data_dir() / _STORE_FILE


class PackEntry(BaseModel):
    title: str = ""
    set_name: str = ""

    model_config = {"extra": "forbid"}


class TelegramIntegration(BaseModel):
    bot_token: str = ""
    user_id: str = ""
    user_username: str = ""
    bot_username: str = ""
    packs: list[PackEntry] = []

    model_config = {"extra": "forbid"}

    def find_pack(self, set_name: str) -> PackEntry | None:
        for pack in self.packs:
            if pack.set_name == set_name:
                return pack
        return None


class YouTubeIntegration(BaseModel):
    client_id: str = ""
    client_secret: str = ""
    token_json: str = ""  # creds.to_json() — carries the refresh_token
    channel_title: str = ""  # whoami display (youtube.readonly)
    channel_id: str = ""  # the unambiguous "a real Connect happened" signal

    model_config = {"extra": "forbid"}


class CopilotIntegration(BaseModel):
    openrouter_key: str = ""
    model: str = "openai/gpt-5.1-codex-mini"  # OpenRouter "provider/model-id"
    # User-tunable agent limits (034 F12) — defaults sourced from CopilotConfig (the
    # single source of truth); applied onto the live config via apply_user_limits.
    max_iterations: int = _COPILOT_DEFAULTS.max_iterations
    max_input_tokens: int = _COPILOT_DEFAULTS.max_input_tokens
    max_tokens_per_turn: int = _COPILOT_DEFAULTS.max_tokens_per_turn
    max_edit_retries: int = _COPILOT_DEFAULTS.max_edit_retries
    max_compile_failures: int = _COPILOT_DEFAULTS.max_compile_failures
    clean_edit_soft_streak: int = _COPILOT_DEFAULTS.clean_edit_soft_streak
    clean_edit_hard_streak: int = _COPILOT_DEFAULTS.clean_edit_hard_streak
    noop_edit_soft_streak: int = _COPILOT_DEFAULTS.noop_edit_soft_streak
    noop_edit_hard_streak: int = _COPILOT_DEFAULTS.noop_edit_hard_streak
    auto_revert_after_failed_edits: int = (
        _COPILOT_DEFAULTS.auto_revert_after_failed_edits
    )
    turn_time_budget_s: int = _COPILOT_DEFAULTS.turn_time_budget_s

    model_config = {"extra": "forbid"}

    def apply_limits(self) -> None:
        # Push these persisted values onto the live COPILOT_CONFIG (startup + Settings edit).
        apply_user_limits(
            max_iterations=self.max_iterations,
            max_input_tokens=self.max_input_tokens,
            max_tokens_per_turn=self.max_tokens_per_turn,
            max_edit_retries=self.max_edit_retries,
            max_compile_failures=self.max_compile_failures,
            clean_edit_soft_streak=self.clean_edit_soft_streak,
            clean_edit_hard_streak=self.clean_edit_hard_streak,
            noop_edit_soft_streak=self.noop_edit_soft_streak,
            noop_edit_hard_streak=self.noop_edit_hard_streak,
            auto_revert_after_failed_edits=self.auto_revert_after_failed_edits,
            turn_time_budget_s=self.turn_time_budget_s,
        )


class IntegrationsStore(BaseModel):
    telegram: TelegramIntegration = TelegramIntegration()
    youtube: YouTubeIntegration = YouTubeIntegration()
    copilot: CopilotIntegration = CopilotIntegration()

    model_config = {"extra": "forbid"}

    @classmethod
    def load(cls) -> Self:
        path: Path = _file_path()
        if not path.exists():
            return cls()
        try:
            with path.open("r") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(
                f"Unreadable integrations.json ({e}); falling back to defaults"
            )
            return cls()
        if not isinstance(data, dict):
            logger.warning(
                "Malformed integrations.json (not an object); falling back to defaults"
            )
            return cls()
        # The store is extra="forbid": a hard fail here returns empty credentials that
        # the next save() writes over the real tokens.
        drop_unknown(cls, data, "integrations")
        drop_invalid(cls, data, "integrations")
        try:
            return cls(**data)
        except ValidationError as e:
            logger.warning(
                f"Incompatible integrations.json ({e}); falling back to defaults"
            )
            return cls()

    def save(self) -> None:
        path: Path = _file_path()
        with _SAVE_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w") as f:
                json.dump(self.model_dump(), f, indent=4)
                f.write("\n")
