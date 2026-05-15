from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal, cast

import telegram as tg
from loguru import logger

from shaderbox.core import Canvas
from shaderbox.media import Image, MediaDetails, Video
from shaderbox.sharing import ShareableMedia, ShareConfiguration, ShareProvider
from shaderbox.ui_models import UIMessage


class TelegramShareableMedia:
    """Telegram-specific shareable media with sticker reference"""

    def __init__(
        self,
        media_dir: Path,
        sticker: tg.Sticker | None = None,
        media_details: MediaDetails | None = None,
        log_message: UIMessage | None = None,
    ) -> None:
        # Non-serializable attributes
        self.media_dir = media_dir
        self._sticker = sticker
        self.video: Video | None = None
        self.image: Image = Image.from_color((512, 512), (0.10, 0.24, 0.39))
        self.preview_canvas = Canvas()

        # ShareableMedia-like attributes (composition instead of inheritance)
        if media_details is None:
            media_details = MediaDetails(is_video=True)
            # Generate unique filename for rendered media
            media_file_name = (
                f"{hashlib.md5(str(id(self)).encode()).hexdigest()[:8]}.webm"
            )
            media_file_path = self.media_dir / media_file_name
            media_details.file_details.path = str(media_file_path)

        self.media_details = media_details
        self.log_message = log_message or UIMessage(
            text="New sticker - render and submit", level="warning"
        )

    def update_log(self, message: str, level: str = "success") -> None:
        """Update the log message for this media"""
        valid_level = cast(Literal["success", "warning", "error"], level)
        self.log_message = UIMessage(text=message, level=valid_level)

    async def load(self) -> TelegramShareableMedia:
        """Load sticker data from Telegram"""
        render_file_path = Path(self.media_details.file_details.path)

        if self._sticker is not None:
            bot = self._sticker.get_bot()

            if self._sticker.is_video:
                file_name = self._sticker.file_id + ".webm"
                file_path = self.media_dir / file_name

                file = await bot.get_file(self._sticker.file_id)
                await file.download_to_drive(file_path)

                self.video = Video(file_path)
                render_file_path = render_file_path.with_suffix(".webm")

            if self._sticker.thumbnail:
                file_name = self._sticker.thumbnail.file_id + ".webp"
                file_path = self.media_dir / file_name

                file = await bot.get_file(self._sticker.thumbnail.file_id)
                await file.download_to_drive(file_path)

                self.image = Image(file_path)
                render_file_path = render_file_path.with_suffix(".webp")

            if self.video:
                self.media_details = self.video.details
            else:
                self.media_details = self.image.details

        self.media_details.file_details.path = str(render_file_path)
        return self

    def update(self, t: float) -> None:
        """Update video frame if this is a video sticker"""
        if self.video:
            self.video.update(t)

    def get_thumbnail_texture(self) -> Any:
        """Get the thumbnail texture for display"""
        if self.video:
            return self.video.texture
        else:
            return self.image.texture

    def release(self) -> None:
        """Release resources"""
        self.preview_canvas.release()
        self.image.release()
        if self.video is not None:
            self.video.release()
            self.video = None


class TelegramShareProvider(ShareProvider):
    """Telegram share provider for sticker management"""

    def __init__(self, config: ShareConfiguration, media_dir: Path):
        super().__init__(config)
        self.media_dir = media_dir
        self._bot: tg.Bot | None = None

    @property
    def provider_id(self) -> str:
        return "telegram"

    @property
    def display_name(self) -> str:
        return "Telegram Stickers"

    def get_configuration_fields(self) -> list[dict[str, Any]]:
        """Return configuration fields needed for Telegram"""
        return [
            {
                "name": "bot_token",
                "label": "Bot Token",
                "type": "text",
                "required": True,
                "placeholder": "Enter your Telegram bot token",
            },
            {
                "name": "user_id",
                "label": "User ID",
                "type": "text",
                "required": True,
                "placeholder": "Your Telegram user ID",
            },
            {
                "name": "sticker_set_name",
                "label": "Sticker Set Name",
                "type": "text",
                "required": True,
                "placeholder": "Name of your sticker set",
            },
        ]

    def validate_configuration(self) -> bool:
        """Check if telegram configuration is valid"""
        settings = self.config.settings
        return all(
            settings.get(field)
            for field in ["bot_token", "user_id", "sticker_set_name"]
        )

    async def initialize(self) -> bool:
        """Initialize Telegram bot with current configuration"""
        if not self.validate_configuration():
            return False

        try:
            self._bot = tg.Bot(token=self.config.settings["bot_token"])
            await self._bot.initialize()
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {e}")
            return False

    async def fetch_existing_media(self) -> list[ShareableMedia]:
        """Fetch existing stickers from Telegram sticker set"""
        if not self._bot:
            raise Exception("Bot not initialized")

        try:
            sticker_set = await self._bot.get_sticker_set(
                name=self.config.settings["sticker_set_name"]
            )

            media_list: list[TelegramShareableMedia] = []
            for sticker in sticker_set.stickers:
                telegram_media = TelegramShareableMedia(
                    media_dir=self.media_dir, sticker=sticker
                )
                await telegram_media.load()
                media_list.append(telegram_media)

            # Return as list[ShareableMedia] for interface compatibility
            return cast(list[ShareableMedia], media_list)

        except Exception as e:
            if str(e) == "Stickerset_invalid":
                logger.info("Sticker set doesn't exist")
            else:
                logger.error(f"Failed to fetch stickers: {e}")
            return []

    async def upload_media(self, media: ShareableMedia, file_path: Path) -> bool:
        """Upload media as a sticker to Telegram"""
        if not self._bot:
            return False

        try:
            # Since we're using composition, media should have the required attributes
            if not hasattr(media, "_sticker") or not hasattr(media, "media_details"):
                return False

            telegram_media = media

            input_sticker = tg.InputSticker(
                sticker=file_path.read_bytes(),
                emoji_list=["🎨"],
                format=(
                    tg.constants.StickerFormat.VIDEO
                    if media.media_details.is_video
                    else tg.constants.StickerFormat.STATIC
                ),
            )

            user_id = int(self.config.settings["user_id"])
            sticker_set_name = f"test_by_{self._bot.username}"

            if telegram_media._sticker is not None:
                # Replace existing sticker
                await self._bot.replace_sticker_in_set(
                    user_id=user_id,
                    name=sticker_set_name,
                    old_sticker=telegram_media._sticker,
                    sticker=input_sticker,
                )
            else:
                # Add new sticker
                await self._bot.add_sticker_to_set(
                    user_id=user_id,
                    name=sticker_set_name,
                    sticker=input_sticker,
                )

            media.update_log("Successfully uploaded to Telegram!")
            return True

        except Exception as e:
            media.update_log(f"Failed to upload: {e}", "error")
            return False

    async def delete_media(self, media: ShareableMedia) -> bool:
        """Delete sticker from Telegram sticker set"""
        if not self._bot:
            return False

        try:
            # Check if media has the required telegram attributes
            if not hasattr(media, "_sticker") or not media._sticker:
                return False

            telegram_media = media

            await self._bot.delete_sticker_from_set(telegram_media._sticker)
            media.update_log("Successfully deleted from Telegram!")
            return True

        except Exception as e:
            media.update_log(f"Failed to delete: {e}", "error")
            return False

    def create_new_media(self) -> ShareableMedia:
        """Create a new telegram shareable media instance"""
        return cast(
            ShareableMedia,
            TelegramShareableMedia(
                media_dir=self.media_dir,
                sticker=None,
            ),
        )
