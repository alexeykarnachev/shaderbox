import json
import threading
from pathlib import Path
from typing import Any, Self, get_args

from loguru import logger
from pydantic import BaseModel, ValidationError

from shaderbox.copilot.config import CopilotConfig, apply_user_limits
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
            auto_revert_after_failed_edits=self.auto_revert_after_failed_edits,
            turn_time_budget_s=self.turn_time_budget_s,
        )


def _drop_unknown(model: type[BaseModel], data: dict[str, Any], path: str) -> None:
    # Prune keys no field claims, recursing into nested models, BEFORE constructing: the store is
    # extra="forbid", and a hard fail here means load() returns empty credentials that the next
    # save() writes over the real ones. A retired field (a removed feature's key) must cost the user
    # that setting, never their tokens.
    for key in [k for k in data if k not in model.model_fields]:
        logger.warning(f"Ignoring unknown {path} key: {key}")
        data.pop(key)
    for key, field in model.model_fields.items():
        value = data.get(key)
        # `or (annotation,)` covers a bare nested model; get_args covers list[Model] (telegram.packs),
        # whose elements are their own extra="forbid" models.
        for nested in get_args(field.annotation) or (field.annotation,):
            if not (isinstance(nested, type) and issubclass(nested, BaseModel)):
                continue
            if isinstance(value, dict):
                _drop_unknown(nested, value, f"{path}.{key}")
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    if isinstance(item, dict):
                        _drop_unknown(nested, item, f"{path}.{key}[{index}]")


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
        _drop_unknown(cls, data, "integrations")
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
