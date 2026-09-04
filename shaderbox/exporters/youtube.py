from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from imgui_bundle import imgui, imgui_ctx
from imgui_bundle import portable_file_dialogs as pfd
from loguru import logger

from shaderbox.exporters.base import (
    AuthState,
    Exporter,
    ExporterError,
    ExporterStatus,
    ExporterValueError,
    ExportProgress,
    OutletUiDeps,
    RenderControl,
    RenderedArtifact,
)
from shaderbox.exporters.worker import DRAIN_TIMEOUT_SEC, ExporterWorker
from shaderbox.exporters.youtube_util import (
    CATEGORY_CHOICES,
    DEFAULT_CATEGORY_ID,
    DEFAULT_FPS,
    SHORT_MAX_DURATION_SEC,
    build_insert_body,
    parse_client_secret_json,
    parse_tags,
    studio_edit_url,
)
from shaderbox.integrations import IntegrationsStore, YouTubeIntegration
from shaderbox.render_preset import RenderPreset, resolve_dims
from shaderbox.render_shape import (
    MENU_SHAPES,
    SHAPE_TABLE,
    RenderShape,
    is_short,
    shape_to_preset,
)
from shaderbox.theme import COLOR, SIZE, SPACE
from shaderbox.ui_models import UIDocument
from shaderbox.ui_primitives import (
    NO_FOCUS,
    FieldFocus,
    button,
    caption_text,
    connection_status,
    danger_button,
    draw_link,
    focus_field,
    ghost_button,
    labeled_combo,
    labeled_drag_float,
    labeled_multiline_input,
    labeled_text_input,
    preview_box,
    primary_button,
    setup_steps,
    status_slot,
    unconnected_gate,
)
from shaderbox.util import pfd_block

_CONNECT_TIMEOUT_SEC = 180
_LONG_MAX_DURATION_SEC = 600.0
_DESC_INPUT_H = 60
_OPEN_SETTINGS_KEY = "open_settings"

# Both gate the buttons, or an upload could enqueue behind a connect blocked on
# run_local_server.
_BUSY_KINDS = frozenset({"connect", "upload"})


@dataclass
class _Job:
    kind: str
    artifact: RenderedArtifact | None = None
    title: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    category_id: str = DEFAULT_CATEGORY_ID
    is_short: bool = False


@dataclass
class _AuthEvent:
    state: AuthState
    message: str = ""


@dataclass
class _ConnectEvent:
    channel_title: str
    channel_id: str


@dataclass
class _RenderState:
    media_dir: Path | None = None
    shape: RenderShape = RenderShape.NATIVE
    title: str = ""
    description: str = ""
    tags_raw: str = ""
    category_id: str = DEFAULT_CATEGORY_ID
    last_studio_url: str = ""
    auth_state: AuthState = AuthState.UNCONFIGURED
    auth_message: str = ""
    last_progress: ExportProgress | None = None
    in_flight: bool = False
    paste_error: str = ""
    show_paste: bool = False  # paste-fallback textarea vs. file pick


def _read_client_secret_file(path: Path) -> str:
    try:
        return path.read_text()
    except OSError as e:
        raise ExporterValueError(f"Could not read {path.name}: {e}") from e


class YouTubeExporter(Exporter):
    def __init__(self) -> None:
        self._store = IntegrationsStore()
        self._render_state = _RenderState()
        self._worker: ExporterWorker[
            _Job, ExportProgress | _AuthEvent | _ConnectEvent
        ] = ExporterWorker("YouTubeExporter")

    @property
    def exporter_id(self) -> str:
        return "youtube"

    @property
    def display_name(self) -> str:
        return "YouTube"

    @property
    def auth_state(self) -> AuthState:
        return self._render_state.auth_state

    @property
    def _yt(self) -> YouTubeIntegration:
        return self._store.youtube

    def status(self) -> ExporterStatus:
        return ExporterStatus(
            auth_state=self._render_state.auth_state,
            auth_message=self._render_state.auth_message,
            last_progress=self._render_state.last_progress,
        )

    def set_integrations(self, store: IntegrationsStore) -> None:
        self._store = store

    def rebind(self, settings: dict[str, Any]) -> None:
        _ = settings
        # Restore auth from the persisted store (no network call); clear per-outlet
        # render state so a previous project's metadata never bleeds into this one.
        self._render_state.auth_state = (
            AuthState.AUTHED if self._has_identity() else AuthState.UNCONFIGURED
        )
        self._render_state.auth_message = ""
        self._render_state.shape = RenderShape.NATIVE
        self._render_state.title = ""
        self._render_state.description = ""
        self._render_state.tags_raw = ""
        self._render_state.category_id = DEFAULT_CATEGORY_ID
        self._render_state.last_studio_url = ""
        self._render_state.last_progress = None
        self._render_state.in_flight = False
        self._render_state.paste_error = ""

    def _has_identity(self) -> bool:
        return bool(self._yt.token_json and self._yt.channel_id)

    def set_media_dir(self, media_dir: Path) -> None:
        self._render_state.media_dir = media_dir

    def current_settings(self) -> dict[str, Any]:
        # Credentials are global (integrations.json); nothing persists per-project.
        return {}

    def begin_auth(self) -> None:
        if not (self._yt.client_id and self._yt.client_secret):
            self._render_state.auth_state = AuthState.ERROR
            self._render_state.auth_message = "Paste your client credentials first."
            return
        self._enqueue(_Job(kind="connect"))

    def disconnect(self) -> None:
        # Drop the OAuth token but keep the client key (re-Connect stays one click;
        # setup instructions stay hidden).
        self._yt.token_json = ""
        self._yt.channel_title = ""
        self._yt.channel_id = ""
        self._store.save()
        self._render_state.auth_state = AuthState.UNCONFIGURED
        self._render_state.auth_message = ""

    def clear_credentials(self) -> None:
        # Wipe everything incl. the client key — setup instructions reappear.
        self._yt.client_id = ""
        self._yt.client_secret = ""
        self.disconnect()
        self._render_state.paste_error = ""

    def _is_connected(self) -> bool:
        return (
            self._render_state.auth_state == AuthState.AUTHED and self._has_identity()
        )

    def current_is_short(self) -> bool:
        return is_short(self._render_state.shape)

    def current_shape(self) -> RenderShape:
        return self._render_state.shape

    def set_shape(self, shape: RenderShape) -> None:
        # Lets the copilot publish path drive the shape so preset + upload flag agree, then restore.
        self._render_state.shape = shape

    def render_preset(self) -> RenderPreset:
        return shape_to_preset(
            self._render_state.shape,
            is_video=True,
            fps=DEFAULT_FPS,
            container=".mp4",
            duration_max=SHORT_MAX_DURATION_SEC
            if is_short(self._render_state.shape)
            else None,
        )

    def build_render_extras(self, deps: OutletUiDeps) -> dict[str, Any]:
        return {_OPEN_SETTINGS_KEY: lambda: deps.open_settings(self.config_field)}

    # ---------------------------------------------------------------- Settings UI
    def _ingest_client_secret(self, raw: str) -> None:
        try:
            cid, secret = parse_client_secret_json(raw)
            self._yt.client_id = cid
            self._yt.client_secret = secret
            self._render_state.paste_error = ""
            self._render_state.show_paste = False
            self._store.save()
        except ExporterValueError as e:
            self._render_state.paste_error = str(e)

    def _pick_client_secret(self) -> None:
        results = pfd_block(
            pfd.open_file(
                "Select client_secret.json",
                default_path=".",
                filters=["JSON", "*.json"],
            )
        )
        if not results:
            return
        try:
            raw: str = _read_client_secret_file(Path(results[0]))
        except ExporterValueError as e:
            self._render_state.paste_error = str(e)
            return
        self._ingest_client_secret(raw)

    @property
    def config_field(self) -> str:
        # Matches popups.settings.SettingsField.YOUTUBE_CLIENT (a string, not an import — see the
        # telegram exporter's note on the layering).
        return "youtube.client"

    def draw_config_ui(self, focus: FieldFocus = NO_FOCUS) -> None:
        full_width: float = imgui.get_content_region_avail().x

        # Setup steps show only until a client key is loaded; reappear when cleared.
        have_key: bool = bool(self._yt.client_id)
        if not have_key:
            setup_steps(
                [
                    (
                        "1. Create a project at console.cloud.google.com:",
                        "console.cloud.google.com/projectcreate",
                    ),
                    (
                        "2. Enable the YouTube Data API v3:",
                        "console.cloud.google.com/apis/library/youtube.googleapis.com",
                    ),
                    (
                        "3. Configure OAuth (External) + add yourself as a Test user:",
                        "console.cloud.google.com/auth/audience",
                    ),
                    (
                        "4. Create an OAuth client of type 'Desktop app':",
                        "console.cloud.google.com/auth/clients",
                    ),
                    "5. Download its JSON and load it below.",
                    "On 'Google hasn't verified this app', click Advanced -> Continue.",
                    "Uploads are always private; flip to public in YouTube Studio.",
                ]
            )
            imgui.dummy(imgui.ImVec2(0, SPACE.SM))

        busy: bool = self._render_state.in_flight
        connected_now: bool = self._is_connected()

        # Three states:
        #   no key             -> Load / Paste (no Connect yet)
        #   key, not connected -> Client loaded + Connect + Clear (no Load)
        #   connected          -> just status + Disconnect (below)
        if not connected_now and not have_key:
            # The credential is loaded via a file pick, not typed — so a focus request lands
            # on the primary Load button, not the hidden paste box.
            with focus_field(focus):
                load_clicked = primary_button("Load client_secret.json...")
            if load_clicked:
                self._pick_client_secret()
            imgui.same_line()
            paste_label: str = (
                "Hide" if self._render_state.show_paste else "Paste instead"
            )
            if ghost_button(paste_label):
                self._render_state.show_paste = not self._render_state.show_paste

            if self._render_state.show_paste:
                imgui.set_next_item_width(full_width)
                changed, pasted = imgui.input_text_multiline(
                    "##yt_secret", "", size=imgui.ImVec2(full_width, _DESC_INPUT_H)
                )
                if changed and pasted.strip():
                    self._ingest_client_secret(pasted)

            if self._render_state.paste_error:
                imgui.text_colored(COLOR.STATE_ERROR, self._render_state.paste_error)
        elif not connected_now and have_key:
            caption_text(f"Client loaded: ...{self._yt.client_id[-28:]}")
            imgui.dummy(imgui.ImVec2(0, SPACE.XS))
            if busy:
                imgui.begin_disabled()
            if primary_button("Connect"):
                self.begin_auth()
            if busy:
                imgui.end_disabled()
            imgui.same_line()
            if danger_button("Clear credentials"):
                self.clear_credentials()

        if busy:
            imgui.text_colored(
                COLOR.STATE_WARN,
                "Waiting for authorization in your browser... (close it to cancel)",
            )

        connection_status(
            connected=connected_now,
            is_error=self._render_state.auth_state == AuthState.ERROR,
            message=self._render_state.auth_message,
            who=self._yt.channel_title,
            on_disconnect=self.disconnect if connected_now else None,
        )

    # ------------------------------------------------------------- render thread
    def update(self, current_document: UIDocument | None) -> None:
        _ = current_document
        while True:
            ev = self._worker.poll_event()
            if ev is None:
                break
            self._apply_event(ev)

    def _apply_event(self, ev: ExportProgress | _AuthEvent | _ConnectEvent) -> None:
        if isinstance(ev, _ConnectEvent):
            self._render_state.auth_state = AuthState.AUTHED
            self._render_state.auth_message = ""
            self._render_state.in_flight = False
        elif isinstance(ev, _AuthEvent):
            self._render_state.auth_state = ev.state
            self._render_state.auth_message = ev.message
            self._render_state.in_flight = False
        else:
            self._render_state.last_progress = ev
            if ev.is_terminal:
                self._render_state.in_flight = False

    def draw_target_panel(
        self,
        current_document: UIDocument | None,
        render_control: RenderControl,
    ) -> None:
        if not self._is_connected():
            extras = render_control.extras or {}
            unconnected_gate(
                "Not connected to YouTube.",
                "Connect your channel in Settings to upload.",
                "Set up credentials",
                extras.get(_OPEN_SETTINGS_KEY),
            )
            return

        # Preview left at the shared fixed size (always taller than the controls, so no
        # alignment math); controls stack on the right.
        preview_box(
            "yt_preview",
            render_control.preview_texture_glo,
            render_control.preview_size,
            float(SIZE.SHARE_PREVIEW_W),
            float(SIZE.SHARE_PREVIEW_H),
        )
        imgui.same_line()
        with imgui_ctx.begin_group():
            self._draw_controls(current_document, render_control)

    def _draw_controls(
        self, current_document: UIDocument | None, rc: RenderControl
    ) -> None:
        rs = self._render_state
        field_w: float = float(SIZE.NAME_INPUT_W)

        # Resolution: one combo over all shapes (native / short 9:16 / wide 16:9). The chosen shape
        # carries its group, so the duration cap + the Short upload flag below derive from it.
        short_now: bool = is_short(rs.shape)
        res_labels: list[str] = [SHAPE_TABLE[s].menu_label for s in MENU_SHAPES]
        cur_res: int = MENU_SHAPES.index(rs.shape)
        res_changed, new_res = labeled_combo("Resolution", cur_res, res_labels, field_w)
        if res_changed and 0 <= new_res < len(MENU_SHAPES):
            rs.shape = MENU_SHAPES[new_res]
            short_now = is_short(rs.shape)

        # Metadata fields (inline, no accordion).
        rs.title = labeled_text_input("Title", rs.title, field_w)
        rs.description = labeled_multiline_input(
            "Description", rs.description, field_w, _DESC_INPUT_H
        )
        rs.tags_raw = labeled_text_input("Tags (comma-separated)", rs.tags_raw, field_w)
        labels: list[str] = [label for _, label in CATEGORY_CHOICES]
        cur_idx: int = next(
            (i for i, (cid, _) in enumerate(CATEGORY_CHOICES) if cid == rs.category_id),
            0,
        )
        changed, new_idx = labeled_combo("Category", cur_idx, labels, field_w)
        if changed and 0 <= new_idx < len(CATEGORY_CHOICES):
            rs.category_id = CATEGORY_CHOICES[new_idx][0]

        v_max: float = SHORT_MAX_DURATION_SEC if short_now else _LONG_MAX_DURATION_SEC
        new_dur: float = labeled_drag_float(
            "Duration", rc.duration, 0.5, v_max, field_w
        )
        if new_dur != rc.duration:
            rc.set_duration(new_dur)

        artifact: RenderedArtifact | None = rc.artifact
        if artifact is not None and not artifact.path.exists():
            artifact = None

        if current_document is None:
            imgui.text_colored(COLOR.STATE_WARN, "Select a document to render.")
            return

        if button("Render"):
            rc.render()
        imgui.same_line()

        size_ok: bool = self._artifact_matches_shape(artifact, current_document)
        upload_enabled: bool = (
            artifact is not None
            and rc.artifact_is_fresh
            and not rs.in_flight
            and size_ok
        )
        if not upload_enabled:
            imgui.begin_disabled()
        if primary_button("Upload") and artifact is not None:
            self._enqueue(
                _Job(
                    kind="upload",
                    artifact=artifact,
                    title=rs.title,
                    description=rs.description,
                    tags=parse_tags(rs.tags_raw),
                    category_id=rs.category_id,
                    is_short=short_now,
                )
            )
        if not upload_enabled:
            imgui.end_disabled()

        mismatch: bool = (
            artifact is not None
            and rc.artifact_is_fresh
            and not rs.in_flight
            and not size_ok
        )
        # Fixed-height status row (always present, so nothing jumps). Spans the full
        # column width so text isn't clipped; the progress bar still sizes to field_w.
        slot_w: float = imgui.get_content_region_avail().x
        with status_slot("yt_status", slot_w):
            if rs.in_flight:
                prog: ExportProgress | None = rs.last_progress
                imgui.progress_bar(
                    prog.fraction if prog is not None else 0.0,
                    size_arg=(field_w, 0.0),
                    overlay=prog.message if prog is not None else "Working...",
                )
            elif rs.last_studio_url:
                # Short label, not the raw URL (too long for the slot); click opens + copies.
                draw_link("Open in YouTube Studio", url=rs.last_studio_url)
            elif mismatch:
                want: str = "Short" if short_now else "Long-form"
                imgui.text_colored(COLOR.STATE_INFO, f"Re-render for {want}.")
            else:
                caption_text("uploads land privately")

    def _artifact_matches_shape(
        self, artifact: RenderedArtifact | None, current_document: UIDocument
    ) -> bool:
        if artifact is None:
            return False
        expected: tuple[int, int] = resolve_dims(
            self.render_preset(),
            current_document.document.render_pass.canvas.texture.size,
        )
        return artifact.size == expected

    # -------------------------------------------------------------- worker thread
    def _enqueue(self, job: _Job) -> None:
        if self._worker.submit(job, self._worker_main) and job.kind in _BUSY_KINDS:
            self._render_state.in_flight = True

    def _worker_main(self) -> None:
        while True:
            job = self._worker.next_job()
            if job is None:
                return
            self._handle_job(job)

    def _handle_job(self, job: _Job) -> None:
        try:
            if job.kind == "connect":
                self._handle_connect()
            elif job.kind == "upload":
                if job.artifact is None:
                    raise ExporterError("Upload job missing artifact")
                self._handle_upload(job)
            else:
                logger.error(f"Unknown YouTubeExporter job kind: {job.kind}")
        except (ExporterError, ExporterValueError) as e:
            self._push_progress(
                ExportProgress(
                    message=str(e), fraction=0.0, is_terminal=True, is_error=True
                )
            )
        except Exception as e:
            logger.exception("YouTubeExporter job failed")
            self._push_progress(
                ExportProgress(
                    message=f"Internal error: {e}",
                    fraction=0.0,
                    is_terminal=True,
                    is_error=True,
                )
            )

    def _handle_connect(self) -> None:
        # The google SDK imports lazily behind this seam (066 D3) — first Connect pays it,
        # here on the worker thread, never the app start.
        from shaderbox.exporters import youtube_api

        try:
            result = youtube_api.run_connect_flow(
                self._yt.client_id,
                self._yt.client_secret,
                timeout_seconds=_CONNECT_TIMEOUT_SEC,
            )
        except youtube_api.YouTubeAuthTimeout:
            self._push_event(
                _AuthEvent(
                    state=AuthState.ERROR,
                    message="Browser authorization timed out - click Connect to retry.",
                )
            )
            return
        except youtube_api.YouTubeApiError as e:
            self._push_event(_AuthEvent(state=AuthState.ERROR, message=str(e)))
            return
        if result is None:
            self._push_event(
                _AuthEvent(
                    state=AuthState.ERROR,
                    message="This account has no YouTube channel. Create one, then Connect again.",
                )
            )
            return
        self._yt.token_json = result.token_json
        self._yt.channel_title = result.channel_title
        self._yt.channel_id = result.channel_id
        self._store.save()
        self._push_event(
            _ConnectEvent(
                channel_title=self._yt.channel_title,
                channel_id=self._yt.channel_id,
            )
        )

    def _handle_upload(self, job: _Job) -> None:
        from shaderbox.exporters import youtube_api

        assert job.artifact is not None
        self._push_progress(ExportProgress(message="Validating...", fraction=0.1))
        prepared: RenderedArtifact = self.prepare(job.artifact, {})

        if not self._yt.token_json:
            raise ExporterError("Not connected.")
        body = build_insert_body(
            job.title, job.description, job.tags, job.category_id, job.is_short
        )

        self._push_progress(ExportProgress(message="Uploading...", fraction=0.4))
        try:
            video_id: str = youtube_api.upload_video(
                token_json=self._yt.token_json,
                path=prepared.path,
                body=body,
                on_token_refreshed=self._persist_token,
                on_progress=lambda fraction: self._push_progress(
                    ExportProgress(
                        message="Uploading...", fraction=0.4 + 0.5 * fraction
                    )
                ),
            )
        except youtube_api.YouTubeTokenRevoked as e:
            raise self._revoked_error(e) from e
        except youtube_api.YouTubeApiError as e:
            raise ExporterError(str(e)) from e

        url: str = studio_edit_url(video_id)
        self._render_state.last_studio_url = url
        self._push_progress(
            ExportProgress(
                message="Uploaded (private). Open in Studio to publish.",
                fraction=1.0,
                is_terminal=True,
                url=url,
            )
        )

    def _revoked_error(self, e: Exception) -> ExporterError:
        # Dead/revoked token: reflect it in Settings (don't leave a stale "Connected")
        # and return the terminal error for the upload.
        self._push_event(
            _AuthEvent(
                state=AuthState.ERROR,
                message="Connection expired - Reconnect in Settings.",
            )
        )
        return ExporterError(str(e))

    def _persist_token(self, token_json: str) -> None:
        # Persist the refreshed token so a crash can't lose it.
        self._yt.token_json = token_json
        self._store.save()

    def prepare(
        self, artifact: RenderedArtifact, settings: dict[str, Any]
    ) -> RenderedArtifact:
        _ = settings
        if not artifact.is_video:
            raise ExporterValueError("YouTube uploads must be video.")
        if not artifact.path.exists() or artifact.path.stat().st_size == 0:
            raise ExporterValueError("Rendered file is empty or missing.")
        return artifact

    def is_connected(self) -> bool:
        return self._is_connected()

    def publish(self, artifact: RenderedArtifact, settings: dict[str, Any]) -> None:
        # `settings`: title, description, is_short. tags/category use the defaults.
        is_short: bool = bool(settings.get("is_short", False))
        self._enqueue(
            _Job(
                kind="upload",
                artifact=artifact,
                title=str(settings.get("title", "")),
                description=str(settings.get("description", "")),
                category_id=DEFAULT_CATEGORY_ID,
                is_short=is_short,
            )
        )

    def release(self) -> None:
        if self._worker.stop() is not None:
            logger.warning(
                f"YouTubeExporter worker did not exit within {DRAIN_TIMEOUT_SEC}s "
                "(a Connect may be blocked on the browser); abandoning."
            )

    def _push_progress(self, progress: ExportProgress) -> None:
        self._worker.push_progress(progress)

    def _push_event(self, event: _AuthEvent | _ConnectEvent) -> None:
        self._worker.push_event(event)
