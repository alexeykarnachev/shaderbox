from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from shaderbox.media import MediaDetails
from shaderbox.ui_models import UIMessage


class ShareConfiguration(BaseModel):
    """Base configuration for a share provider"""

    provider_id: str
    display_name: str
    is_configured: bool = False
    settings: dict[str, Any] = {}


class ShareableMedia(BaseModel):
    """Represents media that can be shared to various platforms"""

    media_details: MediaDetails
    preview_texture_id: int | None = None
    log_message: UIMessage = UIMessage(text="Ready to configure", level="warning")

    model_config = {"arbitrary_types_allowed": True}

    def update_log(
        self, message: str, level: Literal["success", "warning", "error"] = "success"
    ) -> None:
        """Update the log message for this media"""
        self.log_message = UIMessage(text=message, level=level)


class ShareProvider(ABC):
    """Abstract base class for sharing providers (telegram, twitter, etc.)"""

    def __init__(self, config: ShareConfiguration):
        self.config = config

    @property
    @abstractmethod
    def provider_id(self) -> str:
        """Unique identifier for this provider"""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name for this provider"""
        pass

    @abstractmethod
    def get_configuration_fields(self) -> list[dict[str, Any]]:
        """Return list of configuration fields needed for this provider"""
        pass

    @abstractmethod
    def validate_configuration(self) -> bool:
        """Check if the current configuration is valid"""
        pass

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the provider with current configuration"""
        pass

    @abstractmethod
    async def fetch_existing_media(self) -> list[ShareableMedia]:
        """Fetch existing media from the platform"""
        pass

    @abstractmethod
    async def upload_media(self, media: ShareableMedia, file_path: Path) -> bool:
        """Upload media to the platform"""
        pass

    @abstractmethod
    async def delete_media(self, media: ShareableMedia) -> bool:
        """Delete media from the platform"""
        pass

    def create_new_media(self) -> ShareableMedia:
        """Create a new shareable media instance"""
        return ShareableMedia(
            media_details=MediaDetails(),
            log_message=UIMessage(
                text="New media - configure and render", level="warning"
            ),
        )


class ShareManager:
    """Manages multiple share providers and their configurations"""

    def __init__(self) -> None:
        self.providers: dict[str, ShareProvider] = {}
        self.active_provider_id: str | None = None
        self.media_list: list[ShareableMedia] = []
        self.selected_media_index: int = 0

    def register_provider(self, provider: ShareProvider) -> None:
        """Register a new share provider"""
        self.providers[provider.provider_id] = provider
        if self.active_provider_id is None:
            self.active_provider_id = provider.provider_id

    def get_active_provider(self) -> ShareProvider | None:
        """Get the currently active provider"""
        if self.active_provider_id:
            return self.providers.get(self.active_provider_id)
        return None

    def set_active_provider(self, provider_id: str) -> bool:
        """Set the active provider"""
        if provider_id in self.providers:
            self.active_provider_id = provider_id
            self.media_list.clear()
            self.selected_media_index = 0
            return True
        return False

    def get_provider_list(self) -> list[str]:
        """Get list of available provider IDs"""
        return list(self.providers.keys())

    async def refresh_media(self) -> None:
        """Refresh media list from active provider"""
        provider = self.get_active_provider()
        if provider:
            try:
                self.media_list = await provider.fetch_existing_media()
                # Add a new media slot at the beginning
                new_media = provider.create_new_media()
                self.media_list.insert(0, new_media)
            except Exception as e:
                # If fetch fails, just create a new media slot
                new_media = provider.create_new_media()
                new_media.update_log(f"Failed to fetch existing media: {e}", "error")
                self.media_list = [new_media]

    def get_selected_media(self) -> ShareableMedia | None:
        """Get the currently selected media"""
        if 0 <= self.selected_media_index < len(self.media_list):
            return self.media_list[self.selected_media_index]
        return None

    def add_new_media(self) -> None:
        """Add a new media slot"""
        provider = self.get_active_provider()
        if provider:
            new_media = provider.create_new_media()
            self.media_list.insert(0, new_media)
            self.selected_media_index = 0
