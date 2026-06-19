"""YouTubeExporter tests with Google mocked out — no GL, no network, no creds.

Drives the worker handlers (`_handle_connect`/`_handle_upload`) synchronously so
no thread is spawned; asserts the cross-thread events + store persistence.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from shaderbox.exporters.base import (
    AuthState,
    ExporterValueError,
    RenderedArtifact,
)
from shaderbox.exporters.integrations import IntegrationsStore
from shaderbox.exporters.youtube import YouTubeExporter, _AuthEvent, _ConnectEvent, _Job
from shaderbox.render_shape import RenderShape


def _video(tmp_path: Path, size: tuple[int, int] = (608, 1088)) -> RenderedArtifact:
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"\x00\x01\x02\x03")
    return RenderedArtifact(path=p, is_video=True, duration=4.0, size=size)


def _drain(exp: YouTubeExporter) -> list[Any]:
    events: list[Any] = []
    while True:
        try:
            events.append(exp._progress_queue.get_nowait())
        except Exception:
            break
    return events


# --------------------------------------------------------------------- connect
def test_connect_happy(tmp_path: Path) -> None:
    store = IntegrationsStore()
    store.youtube.client_id = "cid"
    store.youtube.client_secret = "secret"
    exp = YouTubeExporter()
    exp.set_integrations(store)

    fake_creds = MagicMock()
    fake_creds.to_json.return_value = '{"refresh_token": "r"}'
    fake_flow = MagicMock()
    fake_flow.run_local_server.return_value = fake_creds
    fake_service = MagicMock()
    fake_service.channels().list().execute.return_value = {
        "items": [{"id": "UC123", "snippet": {"title": "My Channel"}}]
    }

    with (
        patch(
            "shaderbox.exporters.integrations._file_path",
            return_value=tmp_path / "i.json",
        ),
        patch(
            "shaderbox.exporters.youtube.InstalledAppFlow.from_client_config",
            return_value=fake_flow,
        ),
        patch("shaderbox.exporters.youtube.build", return_value=fake_service),
    ):
        exp._handle_connect()

    assert store.youtube.token_json == '{"refresh_token": "r"}'
    assert store.youtube.channel_id == "UC123"
    assert store.youtube.channel_title == "My Channel"
    assert (tmp_path / "i.json").exists()  # persisted in the worker
    events = _drain(exp)
    assert any(isinstance(e, _ConnectEvent) and e.channel_id == "UC123" for e in events)


def test_connect_empty_channel_list(tmp_path: Path) -> None:
    store = IntegrationsStore()
    store.youtube.client_id = "cid"
    store.youtube.client_secret = "secret"
    exp = YouTubeExporter()
    exp.set_integrations(store)

    fake_flow = MagicMock()
    fake_flow.run_local_server.return_value = MagicMock()
    fake_service = MagicMock()
    fake_service.channels().list().execute.return_value = {"items": []}

    with (
        patch(
            "shaderbox.exporters.integrations._file_path",
            return_value=tmp_path / "i.json",
        ),
        patch(
            "shaderbox.exporters.youtube.InstalledAppFlow.from_client_config",
            return_value=fake_flow,
        ),
        patch("shaderbox.exporters.youtube.build", return_value=fake_service),
    ):
        exp._handle_connect()

    assert store.youtube.channel_id == ""  # not connected
    events = _drain(exp)
    err = [e for e in events if isinstance(e, _AuthEvent)]
    assert err and err[0].state == AuthState.ERROR
    assert "no YouTube channel" in err[0].message


# ---------------------------------------------------------------------- upload
def test_upload_happy_emits_studio_url(tmp_path: Path) -> None:
    store = IntegrationsStore()
    store.youtube.client_id = "cid"
    store.youtube.client_secret = "secret"
    store.youtube.channel_id = "UC123"
    store.youtube.token_json = json.dumps(
        {"refresh_token": "r", "client_id": "cid", "client_secret": "s"}
    )
    exp = YouTubeExporter()
    exp.set_integrations(store)

    fake_creds = MagicMock()
    fake_creds.expired = False
    fake_request = MagicMock()
    # First next_chunk: in-progress; second: done with an id.
    prog = MagicMock()
    prog.progress.return_value = 0.5
    fake_request.next_chunk.side_effect = [(prog, None), (None, {"id": "VID42"})]
    fake_service = MagicMock()
    fake_service.videos().insert.return_value = fake_request

    with (
        patch(
            "shaderbox.exporters.integrations._file_path",
            return_value=tmp_path / "i.json",
        ),
        patch(
            "shaderbox.exporters.youtube.google.oauth2.credentials.Credentials.from_authorized_user_info",
            return_value=fake_creds,
        ),
        patch("shaderbox.exporters.youtube.build", return_value=fake_service),
        patch("shaderbox.exporters.youtube.MediaFileUpload", MagicMock()),
    ):
        exp._handle_upload(
            _Job(kind="upload", artifact=_video(tmp_path), title="t", is_short=True)
        )

    assert (
        exp._render_state.last_studio_url
        == "https://studio.youtube.com/video/VID42/edit"
    )
    events = _drain(exp)
    terminal = [e for e in events if getattr(e, "is_terminal", False)]
    assert terminal and terminal[-1].url.endswith("/VID42/edit")
    assert not terminal[-1].is_error


def test_upload_refresh_error_emits_auth_error(tmp_path: Path) -> None:
    store = IntegrationsStore()
    store.youtube.channel_id = "UC123"
    store.youtube.token_json = json.dumps({"refresh_token": "r"})
    exp = YouTubeExporter()
    exp.set_integrations(store)

    from google.auth.exceptions import RefreshError

    fake_creds = MagicMock()
    fake_creds.expired = True
    fake_creds.refresh_token = "r"
    fake_creds.refresh.side_effect = RefreshError("token revoked")

    with (
        patch(
            "shaderbox.exporters.integrations._file_path",
            return_value=tmp_path / "i.json",
        ),
        patch(
            "shaderbox.exporters.youtube.google.oauth2.credentials.Credentials.from_authorized_user_info",
            return_value=fake_creds,
        ),
        patch("shaderbox.exporters.youtube.google.auth.transport.requests.Request"),
    ):
        # _handle_upload catches RefreshError, pushes _AuthEvent(ERROR), re-raises
        # ExporterError which _handle_job would convert to a terminal progress.
        exp._handle_job(_Job(kind="upload", artifact=_video(tmp_path)))

    events = _drain(exp)
    auth_errs = [
        e for e in events if isinstance(e, _AuthEvent) and e.state == AuthState.ERROR
    ]
    assert auth_errs, "expected an _AuthEvent(ERROR) on RefreshError"
    assert "Reconnect" in auth_errs[0].message


def test_size_gate_rejects_mismatched_artifact() -> None:
    # A long-form (landscape) artifact must not satisfy the Short gate, and vice
    # versa. _artifact_matches_shape compares artifact.size to resolve_dims of the
    # current shape's preset against the node's canvas.
    from shaderbox.render_preset import resolve_dims

    exp = YouTubeExporter()
    node_canvas: tuple[int, int] = (1920, 1080)

    class _Canvas:
        class texture:
            size = node_canvas

    class _Node:
        canvas = _Canvas()

    class _UINode:
        node = _Node()

    ui_node: Any = _UINode()

    exp._render_state.shape = RenderShape.SHORT_1080
    short_expected = resolve_dims(exp.render_preset(), node_canvas)
    exp._render_state.shape = RenderShape.NATIVE
    long_expected = resolve_dims(exp.render_preset(), node_canvas)
    assert short_expected != long_expected  # the two shapes resolve differently

    # Landscape artifact (matches long) while toggle is a Short -> gate fails.
    landscape = RenderedArtifact(
        path=Path("x.mp4"), is_video=True, duration=4.0, size=long_expected
    )
    exp._render_state.shape = RenderShape.SHORT_1080
    assert not exp._artifact_matches_shape(landscape, ui_node)
    # Same artifact while toggle is long-form -> gate passes.
    exp._render_state.shape = RenderShape.NATIVE
    assert exp._artifact_matches_shape(landscape, ui_node)
    # None artifact never passes.
    assert not exp._artifact_matches_shape(None, ui_node)


def test_upload_refreshes_expired_token(tmp_path: Path) -> None:
    store = IntegrationsStore()
    store.youtube.channel_id = "UC123"
    store.youtube.token_json = json.dumps({"refresh_token": "r"})
    exp = YouTubeExporter()
    exp.set_integrations(store)

    fake_creds = MagicMock()
    fake_creds.expired = True
    fake_creds.refresh_token = "r"
    fake_creds.to_json.return_value = '{"refreshed": true}'
    fake_request = MagicMock()
    fake_request.next_chunk.side_effect = [(None, {"id": "V"})]
    fake_service = MagicMock()
    fake_service.videos().insert.return_value = fake_request

    with (
        patch(
            "shaderbox.exporters.integrations._file_path",
            return_value=tmp_path / "i.json",
        ),
        patch(
            "shaderbox.exporters.youtube.google.oauth2.credentials.Credentials.from_authorized_user_info",
            return_value=fake_creds,
        ),
        patch("shaderbox.exporters.youtube.google.auth.transport.requests.Request"),
        patch("shaderbox.exporters.youtube.build", return_value=fake_service),
        patch("shaderbox.exporters.youtube.MediaFileUpload", MagicMock()),
    ):
        exp._handle_upload(_Job(kind="upload", artifact=_video(tmp_path)))

    fake_creds.refresh.assert_called_once()
    assert store.youtube.token_json == '{"refreshed": true}'  # persisted in worker


# ----------------------------------------------------------------- size gate
def test_prepare_rejects_non_video(tmp_path: Path) -> None:
    exp = YouTubeExporter()
    art = RenderedArtifact(
        path=tmp_path / "x.png", is_video=False, duration=0.0, size=(10, 10)
    )
    with pytest.raises(ExporterValueError, match="must be video"):
        exp.prepare(art, {})


def test_prepare_rejects_empty(tmp_path: Path) -> None:
    exp = YouTubeExporter()
    p = tmp_path / "empty.mp4"
    p.write_bytes(b"")
    art = RenderedArtifact(path=p, is_video=True, duration=1.0, size=(8, 8))
    with pytest.raises(ExporterValueError, match="empty"):
        exp.prepare(art, {})


# ----------------------------------------------------------------- lifecycle
def test_disconnect_clears_identity_keeps_client(tmp_path: Path) -> None:
    store = IntegrationsStore()
    store.youtube.client_id = "cid"
    store.youtube.client_secret = "secret"
    store.youtube.token_json = "tok"
    store.youtube.channel_id = "UC123"
    exp = YouTubeExporter()
    exp.set_integrations(store)
    with patch(
        "shaderbox.exporters.integrations._file_path", return_value=tmp_path / "i.json"
    ):
        exp.disconnect()
    assert store.youtube.token_json == ""
    assert store.youtube.channel_id == ""
    assert store.youtube.client_id == "cid"  # kept for one-click re-connect
    assert exp.auth_state == AuthState.UNCONFIGURED


def test_clear_credentials_wipes_client_key(tmp_path: Path) -> None:
    store = IntegrationsStore()
    store.youtube.client_id = "cid"
    store.youtube.client_secret = "secret"
    store.youtube.token_json = "tok"
    store.youtube.channel_id = "UC123"
    exp = YouTubeExporter()
    exp.set_integrations(store)
    with patch(
        "shaderbox.exporters.integrations._file_path", return_value=tmp_path / "i.json"
    ):
        exp.clear_credentials()
    # Unlike disconnect(), clear wipes the client key too (instructions reappear).
    assert store.youtube.client_id == ""
    assert store.youtube.client_secret == ""
    assert store.youtube.token_json == ""
    assert exp.auth_state == AuthState.UNCONFIGURED


def test_ingest_client_secret_valid_and_invalid(tmp_path: Path) -> None:
    store = IntegrationsStore()
    exp = YouTubeExporter()
    exp.set_integrations(store)
    good = json.dumps({"installed": {"client_id": "abc", "client_secret": "xyz"}})
    with patch(
        "shaderbox.exporters.integrations._file_path", return_value=tmp_path / "i.json"
    ):
        exp._ingest_client_secret(good)
    assert store.youtube.client_id == "abc"
    assert exp._render_state.paste_error == ""
    # A bad blob sets paste_error, does not raise, leaves prior key intact.
    exp._ingest_client_secret("{not json")
    assert exp._render_state.paste_error
    assert store.youtube.client_id == "abc"


def test_current_settings_empty() -> None:
    assert YouTubeExporter().current_settings() == {}


def test_rebind_clears_render_state() -> None:
    exp = YouTubeExporter()
    exp._render_state.title = "old project title"
    exp._render_state.shape = RenderShape.SHORT_1080
    exp._render_state.last_studio_url = "url"
    exp.rebind({})
    assert exp._render_state.title == ""
    assert exp._render_state.shape == RenderShape.NATIVE
    assert exp._render_state.last_studio_url == ""


def test_begin_auth_without_creds_errors() -> None:
    exp = YouTubeExporter()  # no client_id/secret
    exp.begin_auth()
    assert exp.auth_state == AuthState.ERROR


# ----------------------------------------------------------- store round-trip
def test_integrations_store_youtube_round_trip() -> None:
    store = IntegrationsStore()
    store.youtube.client_id = "cid"
    store.youtube.token_json = "tok"
    store.youtube.channel_id = "UC1"
    data = store.model_dump()
    reloaded = IntegrationsStore(**data)
    assert reloaded.youtube.client_id == "cid"
    assert reloaded.youtube.token_json == "tok"
    assert reloaded.youtube.channel_id == "UC1"
    # Telegram block still defaults cleanly alongside.
    assert reloaded.telegram.bot_token == ""
