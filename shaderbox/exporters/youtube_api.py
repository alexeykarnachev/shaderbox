"""The google-SDK half of the YouTube exporter.

Every google auth/apiclient import lives here, and `youtube.py` imports this module inside
the worker functions that use it — so the SDK's import cost rides the first Connect/upload
on the worker thread, never the app start (066 D3). SDK exceptions never cross the seam:
each is mapped to one of the typed exceptions below, message already user-facing.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import google.auth.transport.requests
import google.oauth2.credentials
from google.auth.exceptions import GoogleAuthError, RefreshError
from google_auth_oauthlib.flow import InstalledAppFlow, WSGITimeoutError
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

from shaderbox.exporters.youtube_util import YOUTUBE_SCOPES, build_client_config


class YouTubeApiError(Exception):
    """A YouTube API failure, message already mapped for the user."""


class YouTubeAuthTimeout(Exception):
    """The browser authorization flow timed out."""


class YouTubeTokenRevoked(Exception):
    """The stored token is dead or revoked; message already mapped for the user."""


@dataclass
class ConnectResult:
    token_json: str
    channel_title: str
    channel_id: str


def map_google_error(e: Exception) -> str:
    text: str = str(e)
    low: str = text.lower()
    if "access_denied" in low:
        return "Authorization was denied — grant access on the consent screen."
    if "invalid_client" in low or "invalid_grant" in low or "unauthorized" in low:
        return "Client credentials rejected — re-check the pasted client_secret (Desktop client)."
    if "quotaexceeded" in low or "quota" in low:
        return "Daily YouTube upload quota reached (~100/day) — try again tomorrow."
    return f"YouTube API error: {text}"


def run_connect_flow(
    client_id: str, client_secret: str, timeout_seconds: int
) -> ConnectResult | None:
    """Run the browser OAuth flow and read the channel. None = the account has no channel."""
    try:
        config = build_client_config(client_id, client_secret)
        flow = InstalledAppFlow.from_client_config(config, YOUTUBE_SCOPES)
        creds = flow.run_local_server(port=0, timeout_seconds=timeout_seconds)
        youtube = build("youtube", "v3", credentials=creds)
        resp = youtube.channels().list(part="snippet", mine=True).execute()
    except WSGITimeoutError:
        raise YouTubeAuthTimeout from None
    except (GoogleAuthError, HttpError) as e:
        raise YouTubeApiError(map_google_error(e)) from e
    items = resp.get("items", [])
    if not items:
        return None
    channel = items[0]
    return ConnectResult(
        token_json=creds.to_json(),
        channel_title=channel["snippet"]["title"],
        channel_id=channel["id"],
    )


def upload_video(
    token_json: str,
    path: Path,
    body: dict[str, Any],
    on_token_refreshed: Callable[[str], None],
    on_progress: Callable[[float], None],
) -> str:
    """Resumable upload; returns the video id. `on_progress` gets the raw 0..1 fraction."""
    creds = _load_creds(token_json, on_token_refreshed)
    youtube = build("youtube", "v3", credentials=creds)
    media = MediaFileUpload(str(path), chunksize=-1, resumable=True)
    request = youtube.videos().insert(
        part="snippet,status", body=body, media_body=media
    )
    response: dict[str, Any] | None = None
    try:
        while response is None:
            status, response = request.next_chunk()
            if status is not None:
                on_progress(status.progress())
    except RefreshError as e:
        raise YouTubeTokenRevoked(map_google_error(e)) from e
    except HttpError as e:
        raise YouTubeApiError(map_google_error(e)) from e
    return response["id"]


def _load_creds(
    token_json: str, on_token_refreshed: Callable[[str], None]
) -> google.oauth2.credentials.Credentials:
    creds = google.oauth2.credentials.Credentials.from_authorized_user_info(
        json.loads(token_json), YOUTUBE_SCOPES
    )
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(google.auth.transport.requests.Request())
        except RefreshError as e:
            raise YouTubeTokenRevoked(map_google_error(e)) from e
        # Hand the refreshed token back immediately so a crash mid-upload can't lose it.
        on_token_refreshed(creds.to_json())
    return creds
