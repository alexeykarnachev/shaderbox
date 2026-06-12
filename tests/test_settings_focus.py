"""The generic settings field-focus contract (no GL, no imgui draw).

open_settings(focus=SettingsField.X) expands X's section + focuses its field. Exporters echo
their field key as a bare string (config_field) to stay below the popups layer — so the strings
MUST match the enum values, else a focus request silently matches nothing. This pins that contract.
"""

from shaderbox.exporters.telegram import TelegramExporter
from shaderbox.exporters.youtube import YouTubeExporter
from shaderbox.popups.settings import SettingsField


def test_exporter_config_fields_match_enum_values() -> None:
    # The cross-layer contract: each exporter's config_field string is a real SettingsField value.
    valid = {f.value for f in SettingsField}
    for cls in (TelegramExporter, YouTubeExporter):
        field = cls.config_field.fget(cls)  # property getter without instantiation
        assert field in valid, (
            f"{cls.__name__}.config_field={field!r} is not a SettingsField value "
            f"(drift would make focus={field!r} match nothing)"
        )


def test_telegram_field_is_token() -> None:
    assert TelegramExporter.config_field.fget(TelegramExporter) == (
        SettingsField.TELEGRAM_TOKEN
    )


def test_youtube_field_is_client() -> None:
    assert YouTubeExporter.config_field.fget(YouTubeExporter) == (
        SettingsField.YOUTUBE_CLIENT
    )
