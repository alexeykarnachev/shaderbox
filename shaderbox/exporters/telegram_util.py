import re
import unicodedata

_MAX_NAME_LEN = 64
_FALLBACK_STEM = "set"


def slugify_title(title: str) -> str:
    """Title → a Telegram-name-safe stem: ASCII [a-z0-9_], begins with a letter."""
    normalized: str = unicodedata.normalize("NFKD", title)
    ascii_only: str = normalized.encode("ascii", "ignore").decode("ascii").lower()
    underscored: str = re.sub(r"[^a-z0-9_]+", "_", ascii_only)
    collapsed: str = re.sub(r"_+", "_", underscored).strip("_")
    # Telegram requires the name begin with a letter.
    stem: str = re.sub(r"^[^a-z]+", "", collapsed)
    return stem or _FALLBACK_STEM


def derive_set_name(title: str, bot_username: str) -> str:
    """Derive the Telegram sticker-set short name from a human title + bot username.

    Telegram rule (PTB `_bot.py:6818-6821`): `[a-z0-9_]`, begins with a letter, no
    consecutive underscores, must end `_by_<bot_username>` (case-insensitive), 1-64
    chars. The fixed suffix must survive intact, so the stem is clamped, not the whole.
    """
    suffix: str = f"_by_{bot_username.lower()}"
    stem: str = slugify_title(title)
    max_stem: int = _MAX_NAME_LEN - len(suffix)
    if max_stem < 1:
        # Pathological: a bot username so long the suffix alone overflows. Clamp the
        # whole result; an invalid name here is a Telegram-side reject the user sees.
        return (stem + suffix)[:_MAX_NAME_LEN]
    stem = stem[:max_stem].rstrip("_") or _FALLBACK_STEM
    return f"{stem}{suffix}"
