import json
from typing import Any

from shaderbox.exporters.base import ExporterValueError

# Upload scope lets us insert videos; readonly lets us read the channel for the
# "Connected as {channel}" line. Persist + reload MUST use this exact list.
YOUTUBE_SCOPES: list[str] = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
]

DEFAULT_CATEGORY_ID = "22"  # People & Blogs
SHORT_MAX_DURATION_SEC = 60.0
SHORT_ASPECT: tuple[int, int] = (9, 16)
SHORT_LONGEST_EDGE = 1920
DEFAULT_FPS = 30
SHORTS_TAG = "#Shorts"

# Selectable Shorts render resolutions (9:16; the value is the longest edge = height). 1080p is the
# YouTube-recommended Shorts size; 1440p oversamples for crisper output, 720p is a fast preview.
SHORT_RES_PRESETS: list[tuple[str, int]] = [
    ("720p", 1280),
    ("1080p", 1920),
    ("1440p", 2560),
]
SHORT_RES_DEFAULT_IDX = 1  # 1080p

# A small, stable subset of YouTube's videoCategories (id -> label). The full set
# is region-dependent; these ids are valid everywhere and the user adjusts the rest
# in Studio. Keep "22" first (the default).
CATEGORY_CHOICES: list[tuple[str, str]] = [
    ("22", "People & Blogs"),
    ("24", "Entertainment"),
    ("28", "Science & Technology"),
    ("10", "Music"),
    ("20", "Gaming"),
    ("1", "Film & Animation"),
]


def parse_client_secret_json(raw: str) -> tuple[str, str]:
    """Parse a downloaded `client_secret.json` blob → (client_id, client_secret).

    Accepts only the Desktop-app shape (top-level `installed` key). A `web` client
    or malformed JSON raises `ExporterValueError` with actionable guidance — the
    caller surfaces it; it must never crash the frame.
    """
    text: str = raw.strip()
    if not text:
        raise ExporterValueError("Paste the contents of your client_secret.json.")
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError as e:
        raise ExporterValueError(f"Not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ExporterValueError("Expected a JSON object from client_secret.json.")
    if "web" in data and "installed" not in data:
        raise ExporterValueError(
            "This is a Web client. Create an OAuth client of type 'Desktop app' "
            "and paste that client_secret.json."
        )
    inner: Any = data.get("installed")
    if not isinstance(inner, dict):
        raise ExporterValueError(
            "Missing the 'installed' section — paste a Desktop-app client_secret.json."
        )
    client_id: Any = inner.get("client_id", "")
    client_secret: Any = inner.get("client_secret", "")
    if not client_id or not client_secret:
        raise ExporterValueError(
            "client_secret.json is missing client_id / client_secret."
        )
    return str(client_id), str(client_secret)


def build_client_config(client_id: str, client_secret: str) -> dict[str, Any]:
    """Assemble the `{"installed": {...}}` dict `InstalledAppFlow.from_client_config`
    expects, from a bare id + secret (used when the user pasted the pair, or after a
    paste was parsed into the store)."""
    return {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }
    }


def studio_edit_url(video_id: str) -> str:
    return f"https://studio.youtube.com/video/{video_id}/edit"


def decorate_short(title: str, description: str) -> tuple[str, str]:
    """Ensure `#Shorts` appears (in description), without duplicating it."""
    if SHORTS_TAG.lower() in title.lower() or SHORTS_TAG.lower() in description.lower():
        return title, description
    suffix: str = SHORTS_TAG if not description else f"{description}\n\n{SHORTS_TAG}"
    return title, suffix


def build_insert_body(
    title: str,
    description: str,
    tags: list[str],
    category_id: str,
    is_short: bool,
) -> dict[str, Any]:
    """The `videos.insert` body. privacyStatus is always 'private' (unverified app
    can only publish private; the user flips to public in Studio)."""
    if is_short:
        title, description = decorate_short(title, description)
    snippet: dict[str, Any] = {
        "title": title or "Untitled",
        "description": description,
        "categoryId": category_id or DEFAULT_CATEGORY_ID,
    }
    if tags:
        snippet["tags"] = tags
    return {
        "snippet": snippet,
        "status": {
            "privacyStatus": "private",
            "selfDeclaredMadeForKids": False,
        },
    }


def parse_tags(raw: str) -> list[str]:
    """Comma-separated tag string → cleaned list (drops blanks/whitespace)."""
    return [t.strip() for t in raw.split(",") if t.strip()]
