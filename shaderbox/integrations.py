import json
import threading
from pathlib import Path
from typing import Self

from loguru import logger
from pydantic import BaseModel, ValidationError

from shaderbox.paths import app_data_dir

_STORE_FILE = "integrations.json"
# Serializes save() across the render thread (Ctrl+S, disconnect) and exporter
# worker threads (token write on connect/refresh) — two interleaved json.dump
# writes would corrupt the file.
_SAVE_LOCK = threading.Lock()


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


class IntegrationsStore(BaseModel):
    telegram: TelegramIntegration = TelegramIntegration()
    youtube: YouTubeIntegration = YouTubeIntegration()

    model_config = {"extra": "forbid"}

    @staticmethod
    def file_path() -> Path:
        return app_data_dir() / _STORE_FILE

    @classmethod
    def load(cls) -> Self:
        path: Path = cls.file_path()
        if not path.exists():
            return cls()
        try:
            with path.open("r") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.error(
                f"Unreadable integrations.json ({e}); falling back to defaults"
            )
            return cls()
        try:
            return cls(**data)
        except ValidationError as e:
            logger.error(
                f"Incompatible integrations.json ({e}); falling back to defaults"
            )
            return cls()

    def save(self) -> None:
        path: Path = self.file_path()
        with _SAVE_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w") as f:
                json.dump(self.model_dump(), f, indent=4)
