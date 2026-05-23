import asyncio
import queue
import subprocess
import threading
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import imageio_ffmpeg
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
from shaderbox.integrations import IntegrationsStore, PackEntry, TelegramIntegration
from shaderbox.media import Image, Video
from shaderbox.telegram_util import derive_set_name
from shaderbox.theme import COLOR
from shaderbox.ui_models import UINode

_QUEUE_MAXSIZE = 128
_DRAIN_TIMEOUT_SEC = 5.0
_FFMPEG_TIMEOUT_SEC = 60
_PREVIEW_THUMB_HEIGHT = 90
_PREVIEW_CANVAS_WIDTH = 200
_GRID_COLUMNS = 4
_TG_VIDEO_MAX_BYTES = 256 * 1024
_TG_VIDEO_MAX_DIM = 512
_TG_VIDEO_MAX_DURATION_SEC = 3.0
_TG_VIDEO_MAX_FPS = 30
_DEFAULT_PACK_TITLE = "ShaderBox"
# Telegram errors that mean "the linked user can't be acted on" — re-link guidance.
_USER_PROBLEM_MARKERS = (
    "USER_IS_BOT",
    "PEER_ID_INVALID",
    "user not found",
    "bot was blocked",
    "USER_ID_INVALID",
)

T = TypeVar("T")


@dataclass
class _Job:
    kind: str
    artifact: RenderedArtifact | None = None
    target_sticker_file_id: str | None = None
    pack_set_name: str = ""
    pack_title: str = ""
    emoji: str = "🎨"


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
class _LinkEvent:
    user_id: str
    user_username: str
    bot_username: str
    message: str = ""


@dataclass
class _StickerListEvent:
    slots: list[_StickerSlot]


_LINK_SENTINEL = "__link__"
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
    active_pack_set_name: str = ""
    new_pack_title: str = ""


class TelegramExporter(Exporter):
    def __init__(self) -> None:
        self._store = IntegrationsStore()
        self._render_state = _RenderState()

        self._job_queue: queue.Queue[_Job | str] = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        self._progress_queue: queue.Queue[
            ExportProgress | _AuthEvent | _LinkEvent | _StickerListEvent
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

    @property
    def _tg(self) -> TelegramIntegration:
        return self._store.telegram

    def status(self) -> ExporterStatus:
        return ExporterStatus(
            auth_state=self._render_state.auth_state,
            auth_message=self._render_state.auth_message,
            last_progress=self._render_state.last_progress,
            in_flight=self._render_state.in_flight,
        )

    def set_integrations(self, store: IntegrationsStore) -> None:
        self._store = store

    def rebind(self, settings: dict[str, Any]) -> None:
        _ = settings
        # Restore the connected state from the persisted store — no network call.
        # A stale token surfaces as an error at the next API call, not here.
        self._render_state.auth_state = (
            AuthState.AUTHED
            if self._has_persisted_identity()
            else AuthState.UNCONFIGURED
        )
        self._render_state.auth_message = ""
        self._release_sticker_slots()
        self._render_state.sticker_slots = []
        self._render_state.selected_index = 0

    def _has_persisted_identity(self) -> bool:
        return bool(self._tg.bot_token and self._tg.user_id and self._tg.bot_username)

    def set_media_dir(self, media_dir: Path) -> None:
        self._render_state.media_dir = media_dir

    def set_default_pack(self, set_name: str) -> None:
        self._render_state.active_pack_set_name = set_name
        if set_name and self._tg.find_pack(set_name) is None:
            # Orphan pointer: the API can't enumerate packs, so this is the only
            # evidence the pack exists — re-add it (title falls back to set_name).
            self._tg.packs.append(PackEntry(title=set_name, set_name=set_name))
        # Restored on launch: pull the pack's current stickers so the grid isn't
        # stale-empty until the user manually re-selects.
        if set_name and self._is_connected():
            self._enqueue(_Job(kind="refresh", pack_set_name=set_name))

    def current_default_pack(self) -> str:
        return self._render_state.active_pack_set_name

    def begin_auth(self) -> None:
        if not self._tg.bot_token:
            self._render_state.auth_state = AuthState.ERROR
            self._render_state.auth_message = "Enter a bot token first."
            return
        self._ensure_worker()
        self._enqueue(_Job(kind=_LINK_SENTINEL))

    def current_settings(self) -> dict[str, Any]:
        # Telegram persists nothing per-project; creds live in the global store.
        return {}

    # ---------------------------------------------------------------- Settings UI
    def draw_config_ui(self) -> None:
        full_width: float = imgui.get_content_region_avail().x

        imgui.text_colored(
            COLOR.FG_DIM, "1. In Telegram, message @BotFather -> /newbot"
        )
        imgui.text_colored(
            COLOR.FG_DIM, "   (a display name, then a username ending 'bot')."
        )
        imgui.text_colored(COLOR.FG_DIM, "2. Paste the token it gives you below.")
        imgui.text_colored(COLOR.FG_DIM, "3. Open YOUR new bot, press Start.")
        imgui.text_colored(COLOR.FG_DIM, "4. Click Connect.")
        imgui.spacing()

        imgui.text("Bot token:")
        imgui.set_next_item_width(full_width)
        _, bot_token = imgui.input_text(
            "##bot_token", self._tg.bot_token, flags=imgui.InputTextFlags_.password
        )
        self._tg.bot_token = bot_token

        if imgui.button("Connect", size=(full_width, 0)):
            self.begin_auth()

        state: AuthState = self._render_state.auth_state
        connected: bool = self._is_connected()
        color: tuple[float, float, float, float] = (
            COLOR.STATE_OK
            if connected
            else (COLOR.STATE_ERROR if state == AuthState.ERROR else COLOR.STATE_WARN)
        )
        if connected:
            who: str = (
                f"@{self._tg.user_username}"
                if self._tg.user_username
                else f"id {self._tg.user_id}"
            )
            imgui.text_colored(color, f"Connected as {who}")
        else:
            imgui.text_colored(color, "Not connected.")
        if self._render_state.auth_message:
            imgui.text_colored(color, self._render_state.auth_message)

    # ------------------------------------------------------------- render thread
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
        pending_emoji: str,
    ) -> None:
        _ = current_node
        if self._render_state.preview_canvas is None:
            self._render_state.preview_canvas = Canvas()

        if not self._is_connected():
            imgui.text_colored(COLOR.STATE_WARN, "Not connected to Telegram.")
            imgui.text_colored(
                COLOR.FG_DIM, "Open Settings (top bar) -> Integrations -> Telegram,"
            )
            imgui.text_colored(COLOR.FG_DIM, "paste your bot token and click Connect.")
            return

        self._draw_pack_controls()
        if not self._render_state.active_pack_set_name:
            return

        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        self._draw_sticker_grid(artifact, pending_emoji)
        self._draw_progress(artifact, pending_emoji)

    def _is_connected(self) -> bool:
        # bot_username is the unambiguous "a real Connect happened" signal — it is
        # only ever set by a successful link (get_me).
        return (
            self._render_state.auth_state == AuthState.AUTHED
            and self._has_persisted_identity()
        )

    def _draw_pack_controls(self) -> None:
        packs: list[PackEntry] = self._tg.packs
        full_width: float = imgui.get_content_region_avail().x

        imgui.text("Sticker pack:")
        if packs:
            labels: list[str] = [p.title for p in packs]
            active: str = self._render_state.active_pack_set_name
            current_idx: int = next(
                (i for i, p in enumerate(packs) if p.set_name == active), 0
            )
            imgui.set_next_item_width(full_width)
            changed, new_idx = imgui.combo("##pack", current_idx, labels)
            if changed and 0 <= new_idx < len(packs):
                self._select_pack(packs[new_idx].set_name)
        else:
            imgui.text_colored(COLOR.FG_DIM, "No packs yet — create one below.")

        imgui.spacing()
        imgui.text_colored(COLOR.FG_DIM, "New pack name:")
        imgui.set_next_item_width(full_width)
        _, self._render_state.new_pack_title = imgui.input_text(
            "##new_pack", self._render_state.new_pack_title
        )
        if imgui.button("Create pack", size=(full_width, 0)):
            self._create_pack(self._render_state.new_pack_title or _DEFAULT_PACK_TITLE)
            self._render_state.new_pack_title = ""

    def _select_pack(self, set_name: str) -> None:
        self._render_state.active_pack_set_name = set_name
        self._enqueue(_Job(kind="refresh", pack_set_name=set_name))

    def _create_pack(self, title: str) -> None:
        set_name: str = derive_set_name(title, self._tg.bot_username)
        if self._tg.find_pack(set_name) is not None:
            self._select_pack(set_name)
            return
        self._tg.packs.append(PackEntry(title=title, set_name=set_name))
        self._store.save()
        self._select_pack(set_name)

    def _draw_sticker_grid(
        self, artifact: RenderedArtifact | None, pending_emoji: str
    ) -> None:
        imgui.spacing()
        n_slots: int = len(self._render_state.sticker_slots)
        if n_slots == 0:
            imgui.text_colored(COLOR.FG_DIM, "No stickers in this pack yet.")
            return

        imgui.text(f"{n_slots} sticker(s):")
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
                        pack_set_name=self._render_state.active_pack_set_name,
                        emoji=pending_emoji,
                    )
                )
            imgui.same_line()
            if imgui.button("Delete selected"):
                self._enqueue(
                    _Job(
                        kind="delete",
                        target_sticker_file_id=selected_slot.file_id,
                        pack_set_name=self._render_state.active_pack_set_name,
                    )
                )

    def _draw_progress(
        self, artifact: RenderedArtifact | None, pending_emoji: str
    ) -> None:
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        full_width: float = imgui.get_content_region_avail().x

        if artifact is None:
            imgui.text_colored(
                COLOR.STATE_WARN, "Render a video on the left first, then add it here."
            )
        elif self._render_state.in_flight:
            prog: ExportProgress | None = self._render_state.last_progress
            msg: str = prog.message if prog is not None else "Working..."
            frac: float = prog.fraction if prog is not None else 0.0
            imgui.text(msg)
            imgui.progress_bar(frac, size_arg=(full_width, 0.0))
        else:
            pack: PackEntry | None = self._tg.find_pack(
                self._render_state.active_pack_set_name
            )
            title: str = pack.title if pack is not None else _DEFAULT_PACK_TITLE
            if imgui.button("Add as new sticker", size=(full_width, 0)):
                self._enqueue(
                    _Job(
                        kind="add",
                        artifact=artifact,
                        pack_set_name=self._render_state.active_pack_set_name,
                        pack_title=title,
                        emoji=pending_emoji,
                    )
                )

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
                imgui.set_next_item_width(imgui.get_content_region_avail().x)
                imgui.input_text(
                    "##pack_link", terminal.url, flags=imgui.InputTextFlags_.read_only
                )

    # -------------------------------------------------------------- worker thread
    def prepare(
        self, artifact: RenderedArtifact, settings: dict[str, Any]
    ) -> RenderedArtifact:
        _ = settings
        if not artifact.is_video:
            raise ExporterValueError("Telegram stickers must be video (.webm).")

        out_path: Path = artifact.path.with_name(f"{artifact.path.stem}.prepared.webm")
        w, h = artifact.size
        scale: str = f"{_TG_VIDEO_MAX_DIM}:-2" if w >= h else f"-2:{_TG_VIDEO_MAX_DIM}"

        cmd: list[str] = [
            imageio_ffmpeg.get_ffmpeg_exe(),
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
        _ = settings
        raise ExporterError("export() is dispatched per job kind via _handle_*")

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

    # Only upload ops gate the UI's "Add as new sticker" button; refresh/link are
    # background fetches whose results aren't terminal ExportProgress events.
    _UPLOAD_KINDS = frozenset({"add", "replace", "delete"})

    def _enqueue(self, job: _Job) -> None:
        self._ensure_worker()
        try:
            self._job_queue.put_nowait(job)
            if job.kind in self._UPLOAD_KINDS:
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

    async def _with_bot(self, op: Callable[[tg.Bot], Awaitable[T]]) -> T:
        bot = tg.Bot(token=self._tg.bot_token)
        try:
            await bot.initialize()
            try:
                return await op(bot)
            except tg.error.TelegramError as e:
                raise ExporterError(self._map_tg_error(e)) from e
        finally:
            await bot.shutdown()

    @staticmethod
    def _map_tg_error(e: tg.error.TelegramError) -> str:
        text: str = str(e)
        if any(marker in text for marker in _USER_PROBLEM_MARKERS):
            return "Open your bot, press Start, and Connect again."
        return f"Telegram API error: {text}"

    def _handle_job(self, job: _Job) -> None:
        try:
            if job.kind == _LINK_SENTINEL:
                self._handle_link()
            elif job.kind == "refresh":
                self._handle_refresh(job.pack_set_name)
            elif job.kind == "add":
                if job.artifact is None:
                    raise ExporterError("Add job missing artifact")
                self._handle_add(job)
            elif job.kind == "replace":
                if job.artifact is None or job.target_sticker_file_id is None:
                    raise ExporterError("Replace job missing artifact or target")
                self._handle_replace(job)
            elif job.kind == "delete":
                if job.target_sticker_file_id is None:
                    raise ExporterError("Delete job missing target")
                self._handle_delete(job.target_sticker_file_id, job.pack_set_name)
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

    def _handle_link(self) -> None:
        try:
            user_id, user_username, bot_username = self._run_async(self._do_link())
        except (tg.error.TelegramError, ExporterError) as e:
            self._push_event(_AuthEvent(state=AuthState.ERROR, message=str(e)))
            return
        if not user_id:
            self._push_event(
                _AuthEvent(
                    state=AuthState.ERROR,
                    message="No message received — open the bot, press Start, then Connect.",
                )
            )
            return
        self._push_event(
            _LinkEvent(
                user_id=user_id, user_username=user_username, bot_username=bot_username
            )
        )

    async def _do_link(self) -> tuple[str, str, str]:
        async def op(bot: tg.Bot) -> tuple[str, str, str]:
            me = await bot.get_me()
            bot_username: str = me.username or ""
            updates = await bot.get_updates(offset=-1, limit=1)
            for update in updates:
                user = update.effective_user
                if user is not None and not user.is_bot:
                    return (str(user.id), user.username or "", bot_username)
            return ("", "", bot_username)

        return await self._with_bot(op)

    def _handle_refresh(self, pack_set_name: str) -> None:
        slots: list[_StickerSlot] = self._run_async(self._do_refresh(pack_set_name))
        self._push_event(_StickerListEvent(slots=slots))

    async def _do_refresh(self, pack_set_name: str) -> list[_StickerSlot]:
        async def op(bot: tg.Bot) -> list[_StickerSlot]:
            slots: list[_StickerSlot] = []
            try:
                sticker_set = await bot.get_sticker_set(name=pack_set_name)
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

        return await self._with_bot(op)

    def _handle_add(self, job: _Job) -> None:
        assert job.artifact is not None
        self._push_progress(
            ExportProgress(message="Preparing (ffmpeg)...", fraction=0.1)
        )
        try:
            prepared: RenderedArtifact = self.prepare(job.artifact, {})
        except (ExporterError, ExporterValueError):
            self._cleanup_paths(job.artifact.path)
            raise
        try:
            self._push_progress(ExportProgress(message="Uploading...", fraction=0.5))
            self._run_async(
                self._do_add(prepared, job.pack_set_name, job.pack_title, job.emoji)
            )
            self._safe_refresh(job.pack_set_name)
            self._push_progress(
                ExportProgress(
                    message="Sticker added — sent to your Telegram.",
                    fraction=1.0,
                    is_terminal=True,
                    url=f"https://t.me/addstickers/{job.pack_set_name}",
                )
            )
        finally:
            self._cleanup_paths(job.artifact.path, prepared.path)

    async def _do_add(
        self,
        artifact: RenderedArtifact,
        pack_set_name: str,
        pack_title: str,
        emoji: str,
    ) -> None:
        async def op(bot: tg.Bot) -> None:
            input_sticker = tg.InputSticker(
                sticker=artifact.path.read_bytes(),
                emoji_list=[emoji],
                format=tg.constants.StickerFormat.VIDEO,
            )
            user_id = int(self._tg.user_id)
            try:
                await bot.get_sticker_set(name=pack_set_name)
                set_exists = True
            except tg.error.TelegramError as e:
                if "Stickerset_invalid" in str(e):
                    set_exists = False
                else:
                    raise
            if set_exists:
                await bot.add_sticker_to_set(
                    user_id=user_id, name=pack_set_name, sticker=input_sticker
                )
            else:
                await bot.create_new_sticker_set(
                    user_id=user_id,
                    name=pack_set_name,
                    title=pack_title,
                    stickers=[input_sticker],
                )
            await self._notify_user(bot, user_id, pack_set_name, pack_title)

        await self._with_bot(op)

    async def _notify_user(
        self, bot: tg.Bot, user_id: int, pack_set_name: str, pack_title: str
    ) -> None:
        # The user pressed Start, so the bot may DM them: deliver the just-added
        # sticker + the pack link so the pack is one tap away. Notification failure
        # must not fail the upload (the sticker is already in the set).
        link: str = f"https://t.me/addstickers/{pack_set_name}"
        try:
            sticker_set = await bot.get_sticker_set(name=pack_set_name)
            if sticker_set.stickers:
                await bot.send_sticker(
                    chat_id=user_id, sticker=sticker_set.stickers[-1].file_id
                )
            await bot.send_message(
                chat_id=user_id,
                text=f'Added a sticker to "{pack_title}".\nOpen the pack: {link}',
            )
        except tg.error.TelegramError as e:
            logger.warning(f"Could not DM the user the new sticker: {e}")

    def _handle_replace(self, job: _Job) -> None:
        assert job.artifact is not None and job.target_sticker_file_id is not None
        self._push_progress(
            ExportProgress(message="Preparing (ffmpeg)...", fraction=0.1)
        )
        try:
            prepared: RenderedArtifact = self.prepare(job.artifact, {})
        except (ExporterError, ExporterValueError):
            self._cleanup_paths(job.artifact.path)
            raise
        try:
            self._push_progress(ExportProgress(message="Replacing...", fraction=0.5))
            self._run_async(
                self._do_replace(
                    prepared, job.pack_set_name, job.target_sticker_file_id, job.emoji
                )
            )
            self._safe_refresh(job.pack_set_name)
            self._push_progress(
                ExportProgress(
                    message="Sticker replaced", fraction=1.0, is_terminal=True
                )
            )
        finally:
            self._cleanup_paths(job.artifact.path, prepared.path)

    async def _do_replace(
        self,
        artifact: RenderedArtifact,
        pack_set_name: str,
        target_file_id: str,
        emoji: str,
    ) -> None:
        async def op(bot: tg.Bot) -> None:
            input_sticker = tg.InputSticker(
                sticker=artifact.path.read_bytes(),
                emoji_list=[emoji],
                format=tg.constants.StickerFormat.VIDEO,
            )
            await bot.replace_sticker_in_set(
                user_id=int(self._tg.user_id),
                name=pack_set_name,
                old_sticker=target_file_id,
                sticker=input_sticker,
            )

        await self._with_bot(op)

    def _handle_delete(self, target_file_id: str, pack_set_name: str) -> None:
        self._push_progress(ExportProgress(message="Deleting...", fraction=0.5))
        self._run_async(self._do_delete(target_file_id))
        self._safe_refresh(pack_set_name)
        self._push_progress(
            ExportProgress(message="Sticker deleted", fraction=1.0, is_terminal=True)
        )

    def _safe_refresh(self, pack_set_name: str) -> None:
        # The mutating op already succeeded; a refresh failure must not surface as
        # a terminal error overwriting the success.
        try:
            self._handle_refresh(pack_set_name)
        except Exception as e:
            logger.warning(f"Sticker refresh after op failed: {e}")

    async def _do_delete(self, target_file_id: str) -> None:
        async def op(bot: tg.Bot) -> None:
            await bot.delete_sticker_from_set(target_file_id)

        await self._with_bot(op)

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

    def _push_event(self, event: _AuthEvent | _LinkEvent | _StickerListEvent) -> None:
        try:
            self._progress_queue.put_nowait(event)
        except queue.Full:
            logger.warning("TelegramExporter progress queue full; event dropped")

    def _apply_event(
        self, ev: ExportProgress | _AuthEvent | _LinkEvent | _StickerListEvent
    ) -> None:
        if isinstance(ev, _LinkEvent):
            self._tg.user_id = ev.user_id
            self._tg.user_username = ev.user_username
            self._tg.bot_username = ev.bot_username
            self._render_state.auth_state = AuthState.AUTHED
            self._render_state.auth_message = ""
            self._render_state.in_flight = False
            self._store.save()  # persist identity now, not just on Ctrl+S
        elif isinstance(ev, _AuthEvent):
            self._render_state.auth_state = ev.state
            self._render_state.auth_message = ev.message
            self._render_state.in_flight = False
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
