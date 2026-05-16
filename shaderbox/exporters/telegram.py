import asyncio
import queue
import subprocess
import threading
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import telegram as tg
from imgui_bundle import imgui
from loguru import logger

from shaderbox.core import Canvas
from shaderbox.exporters.base import (
    AuthState,
    Exporter,
    ExporterError,
    ExporterStatus,
    ExporterValueError,
    ExportProgress,
    RenderedArtifact,
)
from shaderbox.media import Image, Video
from shaderbox.theme import COLOR, SIZE
from shaderbox.ui_models import UINode

_QUEUE_MAXSIZE = 128
_DRAIN_TIMEOUT_SEC = 5.0
_FFMPEG_TIMEOUT_SEC = 60
_PREVIEW_THUMB_HEIGHT = SIZE.TG_THUMB_H
_PREVIEW_CANVAS_WIDTH = SIZE.PREVIEW_W
_GRID_COLUMNS = SIZE.TG_GRID_COLS
_TG_VIDEO_MAX_BYTES = 256 * 1024
_TG_VIDEO_MAX_DIM = 512
_TG_VIDEO_MAX_DURATION_SEC = 3.0
_TG_VIDEO_MAX_FPS = 30

T = TypeVar("T")


@dataclass
class _Job:
    kind: str
    artifact: RenderedArtifact | None = None
    target_sticker_file_id: str | None = None


@dataclass
class _StickerSlot:
    file_id: str
    is_video: bool
    local_file_path: Path | None = None
    image: Image | None = None
    video: Video | None = None
    raw_sticker: tg.Sticker | None = None


@dataclass
class _AuthEvent:
    state: AuthState
    message: str = ""


@dataclass
class _StickerListEvent:
    slots: list[_StickerSlot]


_AUTH_SENTINEL = "__auth__"
_STOP_SENTINEL = "__stop__"


@dataclass
class _RenderState:
    media_dir: Path | None = None
    sticker_slots: list[_StickerSlot] = field(default_factory=list)
    selected_index: int = 0
    preview_canvas: Canvas | None = None
    auth_state: AuthState = AuthState.UNCONFIGURED
    auth_message: str = ""
    last_progress: ExportProgress | None = None
    in_flight: bool = False


class TelegramExporter(Exporter):
    def __init__(self) -> None:
        self._settings: dict[str, Any] = {}
        self._render_state = _RenderState()

        self._job_queue: queue.Queue[_Job | str] = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        self._progress_queue: queue.Queue[
            ExportProgress | _AuthEvent | _StickerListEvent
        ] = queue.Queue(maxsize=_QUEUE_MAXSIZE)

        self._worker: threading.Thread | None = None
        self._worker_loop: asyncio.AbstractEventLoop | None = None
        self._worker_lock = threading.Lock()
        self._current_async_task: asyncio.Task[Any] | None = None

    @property
    def exporter_id(self) -> str:
        return "telegram"

    @property
    def display_name(self) -> str:
        return "Telegram Stickers"

    @property
    def auth_state(self) -> AuthState:
        return self._render_state.auth_state

    def status(self) -> ExporterStatus:
        return ExporterStatus(
            auth_state=self._render_state.auth_state,
            auth_message=self._render_state.auth_message,
            last_progress=self._render_state.last_progress,
            in_flight=self._render_state.in_flight,
        )

    def rebind(self, settings: dict[str, Any]) -> None:
        self._settings = dict(settings)
        self._render_state.auth_state = AuthState.UNCONFIGURED
        self._render_state.auth_message = ""
        self._release_sticker_slots()
        self._render_state.sticker_slots = []
        self._render_state.selected_index = 0

    def set_media_dir(self, media_dir: Path) -> None:
        self._render_state.media_dir = media_dir

    def begin_auth(self) -> None:
        if not self._is_configured():
            self._render_state.auth_state = AuthState.ERROR
            self._render_state.auth_message = (
                "Configure bot_token, user_id, and sticker_set_name first"
            )
            return
        self._ensure_worker()
        self._enqueue(_Job(kind=_AUTH_SENTINEL))

    def current_settings(self) -> dict[str, Any]:
        return dict(self._settings)

    def draw_config_ui(self) -> None:
        bot_token: str = self._settings.get("bot_token", "")
        user_id: str = self._settings.get("user_id", "")
        sticker_set_name: str = self._settings.get("sticker_set_name", "")

        _, bot_token = imgui.input_text(
            "Bot token", bot_token, flags=imgui.InputTextFlags_.password
        )
        _, user_id = imgui.input_text(
            "User ID", user_id, flags=imgui.InputTextFlags_.chars_decimal
        )
        _, sticker_set_name = imgui.input_text(
            "Sticker set name",
            sticker_set_name,
            flags=imgui.InputTextFlags_.chars_no_blank,
        )

        self._settings["bot_token"] = bot_token
        self._settings["user_id"] = user_id
        self._settings["sticker_set_name"] = sticker_set_name

        if imgui.button("Authenticate"):
            self.begin_auth()

    def update(self, current_node: UINode | None) -> None:
        while True:
            try:
                ev = self._progress_queue.get_nowait()
            except queue.Empty:
                break
            self._apply_event(ev)

        if (
            current_node is not None
            and self._render_state.preview_canvas is not None
            and self._render_state.sticker_slots
            and 0
            <= self._render_state.selected_index
            < len(self._render_state.sticker_slots)
        ):
            self._render_state.preview_canvas.set_size(
                _adjust_size(
                    current_node.node.canvas.texture.size,
                    width=_PREVIEW_CANVAS_WIDTH,
                )
            )
            current_node.node.render(canvas=self._render_state.preview_canvas)

    def draw_target_panel(
        self,
        artifact: RenderedArtifact | None,
        current_node: UINode | None,
    ) -> None:
        _ = current_node

        if self._render_state.preview_canvas is None:
            self._render_state.preview_canvas = Canvas()

        state: AuthState = self._render_state.auth_state
        color: tuple[float, float, float, float] = {
            AuthState.AUTHED: COLOR.STATE_OK,
            AuthState.ERROR: COLOR.STATE_ERROR,
            AuthState.UNCONFIGURED: COLOR.STATE_WARN,
        }[state]
        imgui.text_colored(color, f"Auth: {state.value}")
        if self._render_state.auth_message:
            imgui.text_colored(color, self._render_state.auth_message)

        if state != AuthState.AUTHED:
            imgui.text_colored(
                COLOR.STATE_WARN, "Authenticate to load existing stickers."
            )
            return

        if imgui.button("Refresh stickers"):
            self._enqueue(_Job(kind="refresh"))

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        n_slots: int = len(self._render_state.sticker_slots)
        if n_slots == 0:
            imgui.text_colored(COLOR.FG_DIM, "No stickers in set.")
        else:
            imgui.text(f"{n_slots} sticker(s) in set:")
            for idx, slot in enumerate(self._render_state.sticker_slots):
                self._draw_sticker_button(idx, slot)
                if (idx + 1) % _GRID_COLUMNS != 0:
                    imgui.same_line()

            imgui.new_line()

            selected_slot: _StickerSlot = self._render_state.sticker_slots[
                self._render_state.selected_index
            ]
            preview: Canvas | None = self._render_state.preview_canvas
            if preview is not None:
                tex = preview.texture
                imgui.image(
                    imgui.ImTextureRef(tex.glo),
                    image_size=(tex.size[0], tex.size[1]),
                    uv0=(0, 1),
                    uv1=(1, 0),
                )

            if artifact is not None and not self._render_state.in_flight:
                if imgui.button("Replace selected"):
                    self._enqueue(
                        _Job(
                            kind="replace",
                            artifact=artifact,
                            target_sticker_file_id=selected_slot.file_id,
                        )
                    )
                imgui.same_line()
                if imgui.button("Delete selected"):
                    self._enqueue(
                        _Job(
                            kind="delete",
                            target_sticker_file_id=selected_slot.file_id,
                        )
                    )

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        if artifact is None:
            imgui.text_colored(COLOR.STATE_WARN, "Render an artifact first.")
        elif self._render_state.in_flight:
            prog: ExportProgress | None = self._render_state.last_progress
            if prog is not None:
                imgui.text(f"{prog.message} ({prog.fraction * 100:.0f}%)")
        else:
            if imgui.button("Add as new sticker"):
                self._enqueue(_Job(kind="add", artifact=artifact))

        if (
            self._render_state.last_progress is not None
            and not self._render_state.in_flight
        ):
            terminal: ExportProgress = self._render_state.last_progress
            terminal_color: tuple[float, float, float, float] = (
                COLOR.STATE_ERROR if terminal.is_error else COLOR.STATE_OK
            )
            imgui.text_colored(terminal_color, terminal.message)
            if terminal.url:
                imgui.text(terminal.url)

    def prepare(
        self, artifact: RenderedArtifact, settings: dict[str, Any]
    ) -> RenderedArtifact:
        _ = settings
        if not artifact.is_video:
            raise ExporterValueError("Telegram stickers must be video (.webm).")

        if artifact.duration > _TG_VIDEO_MAX_DURATION_SEC:
            logger.warning(
                f"Artifact duration {artifact.duration:.2f}s exceeds Telegram's "
                f"{_TG_VIDEO_MAX_DURATION_SEC}s cap; clipping."
            )

        out_path: Path = artifact.path.with_name(f"{artifact.path.stem}.prepared.webm")
        w, h = artifact.size
        if w >= h:
            scale: str = f"{_TG_VIDEO_MAX_DIM}:-2"
        else:
            scale = f"-2:{_TG_VIDEO_MAX_DIM}"

        cmd: list[str] = [
            "ffmpeg",
            "-y",
            "-i",
            str(artifact.path),
            "-t",
            str(_TG_VIDEO_MAX_DURATION_SEC),
            "-vf",
            f"scale={scale},fps={_TG_VIDEO_MAX_FPS}",
            "-c:v",
            "libvpx-vp9",
            "-b:v",
            "400k",
            "-crf",
            "32",
            "-an",
            str(out_path),
        ]
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=_FFMPEG_TIMEOUT_SEC,
            )
        except subprocess.CalledProcessError as e:
            raise ExporterValueError(
                f"ffmpeg re-encode failed: {e.stderr[-400:]}"
            ) from e
        except subprocess.TimeoutExpired as e:
            raise ExporterError(
                f"ffmpeg re-encode timed out after {_FFMPEG_TIMEOUT_SEC}s"
            ) from e

        size_bytes: int = out_path.stat().st_size
        if size_bytes > _TG_VIDEO_MAX_BYTES:
            raise ExporterValueError(
                f"Prepared file is {size_bytes // 1024} KB; Telegram cap is "
                f"{_TG_VIDEO_MAX_BYTES // 1024} KB."
            )

        return RenderedArtifact(
            path=out_path,
            is_video=True,
            duration=min(artifact.duration, _TG_VIDEO_MAX_DURATION_SEC),
            size=artifact.size,
        )

    def export(self, artifact: RenderedArtifact, settings: dict[str, Any]) -> None:
        self._run_async(self._do_export(artifact, settings))

    async def _do_export(
        self, artifact: RenderedArtifact, settings: dict[str, Any]
    ) -> None:
        async def op(bot: tg.Bot) -> None:
            input_sticker = tg.InputSticker(
                sticker=artifact.path.read_bytes(),
                emoji_list=["🎨"],
                format=tg.constants.StickerFormat.VIDEO,
            )
            await bot.add_sticker_to_set(
                user_id=int(settings["user_id"]),
                name=settings["sticker_set_name"],
                sticker=input_sticker,
            )

        await self._with_bot(settings["bot_token"], op)

    def release(self) -> None:
        with self._worker_lock:
            worker = self._worker
            loop = self._worker_loop
            task = self._current_async_task
            if worker is not None and worker.is_alive():
                if loop is not None and task is not None and not task.done():
                    loop.call_soon_threadsafe(task.cancel)
                self._job_queue.put(_STOP_SENTINEL)
                worker.join(timeout=_DRAIN_TIMEOUT_SEC)
                if worker.is_alive():
                    logger.warning(
                        "TelegramExporter worker did not exit within "
                        f"{_DRAIN_TIMEOUT_SEC}s; abandoning (uploads in flight "
                        "may leak file descriptors)."
                    )
                self._worker = None
                self._worker_loop = None
                self._current_async_task = None

        if self._render_state.preview_canvas is not None:
            self._render_state.preview_canvas.release()
            self._render_state.preview_canvas = None
        self._release_sticker_slots()
        self._render_state.sticker_slots = []

    def _release_sticker_slots(self) -> None:
        for slot in self._render_state.sticker_slots:
            if slot.image is not None:
                slot.image.release()
                slot.image = None
            if slot.video is not None:
                slot.video.release()
                slot.video = None

    def _is_configured(self) -> bool:
        return all(
            self._settings.get(k) for k in ("bot_token", "user_id", "sticker_set_name")
        )

    def _ensure_worker(self) -> None:
        with self._worker_lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._worker = threading.Thread(
                target=self._worker_main,
                name="TelegramExporter-worker",
                daemon=False,
            )
            self._worker.start()

    def _enqueue(self, job: _Job) -> None:
        self._ensure_worker()
        try:
            self._job_queue.put_nowait(job)
            self._render_state.in_flight = True
        except queue.Full:
            logger.error("TelegramExporter job queue full; dropping job")

    def _worker_main(self) -> None:
        loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._worker_loop = loop
        asyncio.set_event_loop(loop)
        try:
            while True:
                job = self._job_queue.get()
                if job == _STOP_SENTINEL:
                    return
                if isinstance(job, _Job):
                    self._handle_job(job)
        finally:
            try:
                loop.close()
            except Exception as e:
                logger.warning(f"Failed to close worker loop: {e}")

    def _run_async(self, coro: Coroutine[Any, Any, T]) -> T:
        loop = self._worker_loop
        if loop is None:
            raise ExporterError("Worker loop not initialized")
        task: asyncio.Task[T] = loop.create_task(coro)
        self._current_async_task = task
        try:
            return loop.run_until_complete(task)
        finally:
            self._current_async_task = None

    async def _with_bot(self, token: str, op: Callable[[tg.Bot], Awaitable[T]]) -> T:
        bot = tg.Bot(token=token)
        try:
            await bot.initialize()
            try:
                return await op(bot)
            except tg.error.TelegramError as e:
                raise ExporterError(f"Telegram API error: {e}") from e
        finally:
            await bot.shutdown()

    def _handle_job(self, job: _Job) -> None:
        try:
            if job.kind == _AUTH_SENTINEL:
                self._handle_auth()
            elif job.kind == "refresh":
                self._handle_refresh()
            elif job.kind == "add":
                if job.artifact is None:
                    raise ExporterError("Add job missing artifact")
                self._handle_add(job.artifact)
            elif job.kind == "replace":
                if job.artifact is None or job.target_sticker_file_id is None:
                    raise ExporterError("Replace job missing artifact or target")
                self._handle_replace(job.artifact, job.target_sticker_file_id)
            elif job.kind == "delete":
                if job.target_sticker_file_id is None:
                    raise ExporterError("Delete job missing target")
                self._handle_delete(job.target_sticker_file_id)
            else:
                logger.error(f"Unknown TelegramExporter job kind: {job.kind}")
        except asyncio.CancelledError:
            self._push_progress(
                ExportProgress(
                    message="Cancelled (app shutting down)",
                    fraction=0.0,
                    is_terminal=True,
                    is_error=True,
                )
            )
        except (ExporterError, ExporterValueError) as e:
            self._push_progress(
                ExportProgress(
                    message=str(e), fraction=0.0, is_terminal=True, is_error=True
                )
            )
        except Exception as e:
            logger.exception("TelegramExporter job failed")
            self._push_progress(
                ExportProgress(
                    message=f"Internal error: {e}",
                    fraction=0.0,
                    is_terminal=True,
                    is_error=True,
                )
            )

    def _handle_auth(self) -> None:
        try:
            self._run_async(self._do_auth_check())
            self._push_event(_AuthEvent(state=AuthState.AUTHED))
            self._handle_refresh()
        except (tg.error.TelegramError, ExporterError) as e:
            self._push_event(_AuthEvent(state=AuthState.ERROR, message=str(e)))

    async def _do_auth_check(self) -> None:
        async def op(bot: tg.Bot) -> None:
            await bot.get_me()

        await self._with_bot(self._settings["bot_token"], op)

    def _handle_refresh(self) -> None:
        slots: list[_StickerSlot] = self._run_async(self._do_refresh())
        self._push_event(_StickerListEvent(slots=slots))

    async def _do_refresh(self) -> list[_StickerSlot]:
        async def op(bot: tg.Bot) -> list[_StickerSlot]:
            slots: list[_StickerSlot] = []
            try:
                sticker_set = await bot.get_sticker_set(
                    name=self._settings["sticker_set_name"]
                )
            except tg.error.TelegramError as e:
                if "Stickerset_invalid" in str(e):
                    return []
                raise

            media_dir: Path | None = self._render_state.media_dir
            if media_dir is None:
                raise ExporterError("media_dir not set; call set_media_dir() first")
            media_dir.mkdir(parents=True, exist_ok=True)

            for sticker in sticker_set.stickers:
                ext: str = ".webm" if sticker.is_video else ".webp"
                local_path: Path = media_dir / f"{sticker.file_id}{ext}"
                if not local_path.exists():
                    file = await bot.get_file(sticker.file_id)
                    await file.download_to_drive(str(local_path))
                slots.append(
                    _StickerSlot(
                        file_id=sticker.file_id,
                        is_video=bool(sticker.is_video),
                        local_file_path=local_path,
                        raw_sticker=sticker,
                    )
                )
            return slots

        return await self._with_bot(self._settings["bot_token"], op)

    def _handle_add(self, artifact: RenderedArtifact) -> None:
        self._push_progress(
            ExportProgress(message="Preparing (ffmpeg)...", fraction=0.1)
        )
        try:
            prepared: RenderedArtifact = self.prepare(artifact, self._settings)
        except (ExporterError, ExporterValueError):
            self._cleanup_paths(artifact.path)
            raise
        try:
            self._push_progress(ExportProgress(message="Uploading...", fraction=0.5))
            self.export(prepared, self._settings)
            self._push_progress(
                ExportProgress(message="Sticker added", fraction=1.0, is_terminal=True)
            )
            self._handle_refresh()
        finally:
            self._cleanup_paths(artifact.path, prepared.path)

    def _handle_replace(self, artifact: RenderedArtifact, target_file_id: str) -> None:
        self._push_progress(
            ExportProgress(message="Preparing (ffmpeg)...", fraction=0.1)
        )
        try:
            prepared: RenderedArtifact = self.prepare(artifact, self._settings)
        except (ExporterError, ExporterValueError):
            self._cleanup_paths(artifact.path)
            raise
        try:
            self._push_progress(ExportProgress(message="Replacing...", fraction=0.5))
            self._run_async(self._do_replace(prepared, target_file_id))
            self._push_progress(
                ExportProgress(
                    message="Sticker replaced",
                    fraction=1.0,
                    is_terminal=True,
                )
            )
            self._handle_refresh()
        finally:
            self._cleanup_paths(artifact.path, prepared.path)

    async def _do_replace(
        self, artifact: RenderedArtifact, target_file_id: str
    ) -> None:
        async def op(bot: tg.Bot) -> None:
            input_sticker = tg.InputSticker(
                sticker=artifact.path.read_bytes(),
                emoji_list=["🎨"],
                format=tg.constants.StickerFormat.VIDEO,
            )
            await bot.replace_sticker_in_set(
                user_id=int(self._settings["user_id"]),
                name=self._settings["sticker_set_name"],
                old_sticker=target_file_id,
                sticker=input_sticker,
            )

        await self._with_bot(self._settings["bot_token"], op)

    def _handle_delete(self, target_file_id: str) -> None:
        self._push_progress(ExportProgress(message="Deleting...", fraction=0.5))
        self._run_async(self._do_delete(target_file_id))
        self._push_progress(
            ExportProgress(message="Sticker deleted", fraction=1.0, is_terminal=True)
        )
        self._handle_refresh()

    async def _do_delete(self, target_file_id: str) -> None:
        async def op(bot: tg.Bot) -> None:
            await bot.delete_sticker_from_set(target_file_id)

        await self._with_bot(self._settings["bot_token"], op)

    def _cleanup_paths(self, *paths: Path) -> None:
        for path in paths:
            try:
                if path.exists():
                    path.unlink()
            except OSError as e:
                logger.warning(f"Failed to cleanup {path}: {e}")

    def _push_progress(self, progress: ExportProgress) -> None:
        try:
            self._progress_queue.put_nowait(progress)
        except queue.Full:
            try:
                self._progress_queue.get_nowait()
                self._progress_queue.put_nowait(progress)
            except (queue.Empty, queue.Full):
                pass

    def _push_event(self, event: _AuthEvent | _StickerListEvent) -> None:
        try:
            self._progress_queue.put_nowait(event)
        except queue.Full:
            logger.warning("TelegramExporter progress queue full; event dropped")

    def _apply_event(self, ev: ExportProgress | _AuthEvent | _StickerListEvent) -> None:
        if isinstance(ev, _AuthEvent):
            self._render_state.auth_state = ev.state
            self._render_state.auth_message = ev.message
        elif isinstance(ev, _StickerListEvent):
            self._release_sticker_slots()
            self._render_state.sticker_slots = ev.slots
            self._render_state.selected_index = 0
        else:
            self._render_state.last_progress = ev
            if ev.is_terminal:
                self._render_state.in_flight = False

    def _draw_sticker_button(self, idx: int, slot: _StickerSlot) -> None:
        thumbnail: Image | Video | None = self._lazy_thumbnail(slot)
        is_selected: bool = idx == self._render_state.selected_index

        n_styles: int = 0
        if is_selected:
            highlight: tuple[float, float, float, float] = COLOR.ACCENT_PRIMARY
            imgui.push_style_color(imgui.Col_.button, highlight)
            imgui.push_style_color(imgui.Col_.button_hovered, highlight)
            imgui.push_style_color(imgui.Col_.button_active, highlight)
            n_styles += 3

        if thumbnail is not None:
            tex = thumbnail.texture
            h: int = _PREVIEW_THUMB_HEIGHT
            w: int = h * tex.size[0] // max(tex.size[1], 1)
            if imgui.image_button(
                f"##sticker_{idx}",
                imgui.ImTextureRef(tex.glo),
                image_size=(w, h),
                uv0=(0, 1),
                uv1=(1, 0),
            ):
                self._render_state.selected_index = idx
        else:
            if imgui.button(
                f"#{idx}", size=(_PREVIEW_THUMB_HEIGHT, _PREVIEW_THUMB_HEIGHT)
            ):
                self._render_state.selected_index = idx

        if n_styles:
            imgui.pop_style_color(n_styles)

    def _lazy_thumbnail(self, slot: _StickerSlot) -> Image | Video | None:
        if slot.local_file_path is None or not slot.local_file_path.exists():
            return None
        if slot.is_video:
            if slot.video is None:
                slot.video = Video(slot.local_file_path)
            return slot.video
        else:
            if slot.image is None:
                slot.image = Image(slot.local_file_path)
            return slot.image


def _adjust_size(
    size: tuple[int, int], width: int | None = None, height: int | None = None
) -> tuple[int, int]:
    w, h = size
    if width is not None:
        return (width, max(1, h * width // max(w, 1)))
    if height is not None:
        return (max(1, w * height // max(h, 1)), height)
    return size
