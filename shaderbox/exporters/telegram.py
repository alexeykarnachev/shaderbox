import asyncio
import queue
import subprocess
import threading
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import httpx
import imageio_ffmpeg
import telegram as tg
from imgui_bundle import imgui, imgui_ctx
from loguru import logger
from telegram.request import HTTPXRequest

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
from shaderbox.exporters.integrations import (
    IntegrationsStore,
    PackEntry,
    TelegramIntegration,
)
from shaderbox.exporters.telegram_util import derive_set_name
from shaderbox.media import Image, Video
from shaderbox.render_preset import FitPolicy, RenderPreset, ResolutionPolicy
from shaderbox.theme import COLOR, SIZE, SPACE, fade
from shaderbox.ui_models import UINode
from shaderbox.ui_primitives import (
    button,
    caption_text,
    connection_status,
    danger_button,
    draw_link,
    ghost_button,
    labeled_drag_float,
    labeled_text_input,
    preview_box,
    preview_cell,
    primary_button,
    setup_steps,
    status_slot,
    unconnected_gate,
)

_QUEUE_MAXSIZE = 128
_DRAIN_TIMEOUT_SEC = 5.0
_FFMPEG_TIMEOUT_SEC = 60
_PREVIEW_THUMB_HEIGHT = 110
_GRID_COLUMNS = 4
_CAROUSEL_ARROW_W = 18
_PACK_COMBO_W = 220
_TG_VIDEO_MAX_BYTES = 256 * 1024
_TG_VIDEO_MAX_DIM = 512
_TG_VIDEO_MAX_DURATION_SEC = 3.0
_TG_VIDEO_MAX_FPS = 30
_DEFAULT_PACK_TITLE = "ShaderBox"
_DEFAULT_NEW_STICKER_EMOJI = "🎨"
# Telegram errors that mean "the linked user can't be acted on" — re-link guidance.
_USER_PROBLEM_MARKERS = (
    "USER_IS_BOT",
    "PEER_ID_INVALID",
    "user not found",
    "bot was blocked",
    "USER_ID_INVALID",
)

T = TypeVar("T")


def _ipv4_request() -> HTTPXRequest:
    # Bind egress to IPv4. Telegram's AAAA resolves, but ptb's default HTTP/2
    # client picks the v6 address with no v4 fallback; on an IPv6-incapable
    # tunnel that fails the TLS handshake every time (EndOfStream).
    # Generous timeouts: ptb defaults are 5s read/connect, which a VPN/tunnel
    # routinely exceeds (ReadTimeout -> TimedOut); the sticker upload also needs a
    # long write window.
    transport = httpx.AsyncHTTPTransport(local_address="0.0.0.0")
    return HTTPXRequest(
        httpx_kwargs={"transport": transport},
        connect_timeout=30.0,
        read_timeout=30.0,
        write_timeout=30.0,
        media_write_timeout=120.0,
    )


def _map_tg_error(e: tg.error.TelegramError) -> str:
    text: str = str(e)
    if any(marker in text for marker in _USER_PROBLEM_MARKERS):
        return "Open your bot, press Start, and Connect again."
    return f"Telegram API error: {text}"


@dataclass
class EmojiControl:
    """Telegram-owned emoji affordances, delivered via `RenderControl.extras`.

    The share tab builds these (it owns `App`'s emoji font + picker); the
    generic render contract never names emoji.
    """

    emoji: str
    set_emoji: Callable[[str], None]
    open_emoji_picker: Callable[[Callable[[str], None]], None]
    # Draws `char` as a clickable emoji-font glyph button of the given side; -> clicked.
    emoji_button: Callable[[str, float], bool]


_EMOJI_EXTRA_KEY = "telegram_emoji"
_OPEN_SETTINGS_KEY = "open_settings"


@dataclass
class _Job:
    kind: str
    artifact: RenderedArtifact | None = None
    target_sticker_file_id: str | None = None
    pack_set_name: str = ""
    pack_title: str = ""
    emoji: str = _DEFAULT_NEW_STICKER_EMOJI


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
    auth_state: AuthState = AuthState.UNCONFIGURED
    auth_message: str = ""
    last_progress: ExportProgress | None = None
    in_flight: bool = False
    active_pack_set_name: str = ""
    new_pack_title: str = ""
    pack_delete_armed: bool = False
    pack_create_armed: bool = False
    sticker_delete_armed: str = ""  # file_id of the sticker pending delete-confirm
    carousel_offset: int = 0  # index of the first sticker shown in the grid row


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
            self._store.save()
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

    def disconnect(self) -> None:
        self._tg.bot_token = ""
        self._tg.user_id = ""
        self._tg.user_username = ""
        self._tg.bot_username = ""
        self._store.save()
        self._render_state.auth_state = AuthState.UNCONFIGURED
        self._render_state.auth_message = ""

    def current_settings(self) -> dict[str, Any]:
        # Telegram persists nothing per-project; creds live in the global store.
        return {}

    def render_preset(self) -> RenderPreset:
        return RenderPreset(
            is_video=True,
            container=".webm",
            fps=_TG_VIDEO_MAX_FPS,
            duration_max=_TG_VIDEO_MAX_DURATION_SEC,
            resolution_policy=ResolutionPolicy.LONGEST_EDGE,
            longest_edge=_TG_VIDEO_MAX_DIM,
            max_bytes=_TG_VIDEO_MAX_BYTES,
            fit=FitPolicy.RENDER_AT_TARGET,
        )

    def build_render_extras(self, deps: OutletUiDeps) -> dict[str, Any]:
        scratch: dict[str, Any] = deps.outlet_extra_state
        scratch.setdefault("pending_emoji", _DEFAULT_NEW_STICKER_EMOJI)

        def set_emoji(char: str) -> None:
            scratch["pending_emoji"] = char

        def emoji_button(char: str, side: float) -> bool:
            clicked: bool = imgui.button("##emoji_btn", size=(side, side))
            if imgui.is_item_hovered():
                imgui.set_tooltip("Click to change emoji")
            rmin = imgui.get_item_rect_min()
            rmax = imgui.get_item_rect_max()
            font = deps.glyph_font
            size_px: float = font.legacy_size
            imgui.push_font(font, size_px)
            glyph = imgui.calc_text_size(char)
            imgui.pop_font()
            pos = (
                (rmin.x + rmax.x) / 2 - glyph.x / 2,
                (rmin.y + rmax.y) / 2 - glyph.y / 2,
            )
            col = imgui.color_convert_float4_to_u32(COLOR.FG_PRIMARY)
            imgui.get_window_draw_list().add_text(font, size_px, pos, col, char)
            return clicked

        control = EmojiControl(
            emoji=scratch["pending_emoji"],
            set_emoji=set_emoji,
            open_emoji_picker=deps.open_glyph_picker,
            emoji_button=emoji_button,
        )
        return {
            _EMOJI_EXTRA_KEY: control,
            _OPEN_SETTINGS_KEY: deps.open_settings,
        }

    # ---------------------------------------------------------------- Settings UI
    def draw_config_ui(self) -> None:
        full_width: float = imgui.get_content_region_avail().x

        have_token: bool = bool(self._tg.bot_token)
        connected: bool = self._is_connected()

        # Setup steps show until actually connected — a pasted-but-not-connected
        # token is just text, the user still needs the instructions. Ghost/dim
        # wrapped text via the shared primitive (parity with YouTube).
        if not connected:
            setup_steps(
                [
                    "1. In Telegram, open @BotFather and send /newbot.",
                    "2. Follow the prompts to name your bot; copy the token it gives.",
                    "3. Open your new bot and press Start.",
                    "4. Paste the token below and click Connect.",
                ]
            )
            imgui.dummy(imgui.ImVec2(0, SPACE.SM))

        # Token field + Connect only while not connected; once connected just the
        # status + Disconnect (parity with YouTube — re-link is Disconnect -> Connect).
        if not connected:
            self._tg.bot_token = labeled_text_input(
                "Bot token", self._tg.bot_token, full_width, password=True
            )
            if primary_button("Connect"):
                self.begin_auth()
            if have_token:
                imgui.same_line()
                if danger_button("Clear token"):
                    self.disconnect()

        who: str = (
            f"@{self._tg.user_username}"
            if self._tg.user_username
            else f"id {self._tg.user_id}"
        )
        connection_status(
            connected=connected,
            is_error=self._render_state.auth_state == AuthState.ERROR,
            message=self._render_state.auth_message,
            who=who,
            on_disconnect=self.disconnect if connected else None,
        )

    # ------------------------------------------------------------- render thread
    def update(self, current_node: UINode | None) -> None:
        _ = current_node
        while True:
            try:
                ev = self._progress_queue.get_nowait()
            except queue.Empty:
                break
            self._apply_event(ev)

    def draw_target_panel(
        self,
        current_node: UINode | None,
        render_control: RenderControl,
    ) -> None:
        _ = current_node
        if not self._is_connected():
            extras = render_control.extras or {}
            unconnected_gate(
                "Not connected to Telegram.",
                "Connect a bot in Settings to share stickers.",
                "Set up token",
                extras.get(_OPEN_SETTINGS_KEY),
            )
            return

        self._draw_sticker_section(render_control, current_node is not None)

    def _is_connected(self) -> bool:
        # bot_username is the unambiguous "a real Connect happened" signal — it is
        # only ever set by a successful link (get_me).
        return (
            self._render_state.auth_state == AuthState.AUTHED
            and self._has_persisted_identity()
        )

    def _draw_pack_row(self) -> None:
        packs: list[PackEntry] = self._tg.packs
        active: str = self._render_state.active_pack_set_name
        combo_w: float = float(_PACK_COMBO_W)

        caption_text("Pack")
        if packs:
            labels: list[str] = [p.title for p in packs]
            current_idx: int = next(
                (i for i, p in enumerate(packs) if p.set_name == active), 0
            )
            imgui.set_next_item_width(combo_w)
            changed, new_idx = imgui.combo("##pack", current_idx, labels)
            if changed and 0 <= new_idx < len(packs):
                self._select_pack(packs[new_idx].set_name)
            if active:
                imgui.same_line()
                draw_link(
                    f"t.me/addstickers/{active}",
                    url=f"https://t.me/addstickers/{active}",
                )
        else:
            imgui.set_next_item_width(combo_w)
            imgui.text_colored(COLOR.FG_DIM, "no packs yet")

        # Management ghosts sit on their own row under the combo.
        if ghost_button("New pack"):
            self._render_state.pack_create_armed = (
                not self._render_state.pack_create_armed
            )
            self._render_state.pack_delete_armed = False
        if active and not self._render_state.in_flight:
            imgui.same_line()
            if danger_button("Delete pack"):
                self._render_state.pack_delete_armed = (
                    not self._render_state.pack_delete_armed
                )
                self._render_state.pack_create_armed = False

        if self._render_state.pack_create_armed:
            imgui.set_next_item_width(combo_w)
            _, self._render_state.new_pack_title = imgui.input_text(
                "##new_pack", self._render_state.new_pack_title
            )
            imgui.same_line()
            if primary_button("Create"):
                self._create_pack(
                    self._render_state.new_pack_title or _DEFAULT_PACK_TITLE
                )
                self._render_state.new_pack_title = ""
                self._render_state.pack_create_armed = False

        if self._render_state.pack_delete_armed and active:
            self._draw_delete_confirm(
                "Delete whole pack from Telegram?",
                lambda: self._delete_pack(active),
            )

    def _draw_delete_confirm(self, prompt: str, on_yes: "Callable[[], None]") -> None:
        imgui.push_style_color(imgui.Col_.child_bg, fade(COLOR.STATE_ERROR, 0.08))
        imgui.push_style_color(imgui.Col_.border, fade(COLOR.STATE_ERROR, 0.4))
        with imgui_ctx.begin_child(
            "##del_confirm",
            size=imgui.ImVec2(0, 0),
            child_flags=imgui.ChildFlags_.borders | imgui.ChildFlags_.auto_resize_y,
        ):
            imgui.align_text_to_frame_padding()
            imgui.text_colored(COLOR.STATE_ERROR, prompt)
            imgui.same_line()
            if danger_button("Delete"):
                self._render_state.pack_delete_armed = False
                on_yes()
            imgui.same_line()
            if ghost_button("Cancel"):
                self._render_state.pack_delete_armed = False
        imgui.pop_style_color(2)

    def _delete_pack(self, set_name: str) -> None:
        self._enqueue(_Job(kind="delete_pack", pack_set_name=set_name))
        # Remove locally now (render-side state); the worker deletes on Telegram.
        self._tg.packs = [p for p in self._tg.packs if p.set_name != set_name]
        self._store.save()
        new_active: str = self._tg.packs[0].set_name if self._tg.packs else ""
        if new_active:
            self._select_pack(new_active)
        else:
            self._render_state.active_pack_set_name = ""
            self._clear_grid()

    def _select_pack(self, set_name: str) -> None:
        self._render_state.active_pack_set_name = set_name
        # Clear the previous pack's stickers immediately so the grid never renders
        # the old pack's slots (with a stale selected_index) under the new pack.
        self._clear_grid()
        self._enqueue(_Job(kind="refresh", pack_set_name=set_name))

    def _clear_grid(self) -> None:
        self._release_sticker_slots()
        self._render_state.sticker_slots = []
        self._render_state.selected_index = 0
        self._render_state.sticker_delete_armed = ""
        self._render_state.carousel_offset = 0
        self._render_state.last_progress = None

    def _create_pack(self, title: str) -> None:
        set_name: str = derive_set_name(title, self._tg.bot_username)
        if self._tg.find_pack(set_name) is not None:
            self._select_pack(set_name)
            return
        self._tg.packs.append(PackEntry(title=title, set_name=set_name))
        self._store.save()
        self._select_pack(set_name)

    def _draw_sticker_section(self, rc: RenderControl, has_node: bool) -> None:
        artifact: RenderedArtifact | None = rc.artifact
        if artifact is not None and not artifact.path.exists():
            artifact = None

        emoji: EmojiControl = self._emoji(rc)

        def _overlay(_origin: imgui.ImVec2) -> None:
            if emoji.emoji_button(emoji.emoji, float(SIZE.ROW_HEIGHT)):
                emoji.open_emoji_picker(emoji.set_emoji)

        # Preview on the left (the shared fixed size — always taller than the
        # controls, no alignment math); controls + carousel stack top-down on the right.
        preview_box(
            "sticker_preview",
            rc.preview_texture_glo,
            rc.preview_size,
            float(SIZE.SHARE_PREVIEW_W),
            float(SIZE.SHARE_PREVIEW_H),
            overlay=_overlay,
        )
        imgui.same_line()
        with imgui_ctx.begin_group():
            self._draw_sticker_controls(rc, artifact, has_node)
            if self._render_state.active_pack_set_name:
                imgui.dummy(imgui.ImVec2(0, SPACE.SM))
                self._draw_sticker_grid_full(rc)

    def _emoji(self, rc: RenderControl) -> EmojiControl:
        extras = rc.extras or {}
        emoji = extras.get(_EMOJI_EXTRA_KEY)
        if not isinstance(emoji, EmojiControl):
            raise ExporterError("Telegram outlet missing its EmojiControl extra")
        return emoji

    def _draw_sticker_controls(
        self,
        rc: RenderControl,
        artifact: RenderedArtifact | None,
        has_node: bool,
    ) -> None:
        # Pack selector + management live here on the right (beside the preview),
        # so the preview can be large. Then duration + Render/Add below.
        self._draw_pack_row()
        imgui.dummy(imgui.ImVec2(0, SPACE.SM))

        new_dur: float = labeled_drag_float(
            "Duration",
            rc.duration,
            0.5,
            _TG_VIDEO_MAX_DURATION_SEC,
            float(SIZE.NAME_INPUT_W),
        )
        if new_dur != rc.duration:
            rc.set_duration(new_dur)

        imgui.dummy(imgui.ImVec2(0, SPACE.XS))
        row_w: float = float(SIZE.LABEL_W) + SPACE.MD + float(SIZE.NAME_INPUT_W)
        if not has_node:
            imgui.text_colored(COLOR.STATE_WARN, "Select a node to render.")
            return

        # Render (ordinary) + Add to pack (CTA) on one row; Add's right edge
        # aligns with the Duration drag's right edge.
        render_w: float = imgui.calc_text_size("Render").x + 2 * SPACE.MD
        add_w: float = row_w - render_w - imgui.get_style().item_spacing.x
        if button("Render", width=render_w):
            rc.render()
        imgui.same_line()
        self._draw_add_button(rc, artifact, add_w)

        # Constant-height status slot (one text line + one bar line) so the controls
        # never grow when uploading — otherwise the absolutely-positioned carousel
        # below would be overlapped. While uploading: message + progress bar; else:
        # render stats + an empty bar-height spacer (space reserved, no jitter).
        self._draw_status_slot(rc, artifact, row_w)

    def _draw_add_button(
        self, rc: RenderControl, artifact: RenderedArtifact | None, width: float
    ) -> None:
        active: str = self._render_state.active_pack_set_name
        enabled: bool = (
            artifact is not None and bool(active) and not self._render_state.in_flight
        )
        if not enabled:
            imgui.begin_disabled()
        if primary_button("Add to pack", width=width) and artifact is not None:
            pack: PackEntry | None = self._tg.find_pack(active)
            title: str = pack.title if pack is not None else _DEFAULT_PACK_TITLE
            self._enqueue(
                _Job(
                    kind="add",
                    artifact=artifact,
                    pack_set_name=active,
                    pack_title=title,
                    emoji=self._emoji(rc).emoji,
                )
            )
        if not enabled:
            imgui.end_disabled()

    def _draw_sticker_grid_full(self, rc: RenderControl) -> None:
        """A carousel row: [<] [cell x4] [>]. The arrows scroll a 4-wide window
        into the pack; they're always shown, disabled when there's nowhere to go.

        Each cell is its own child window so the corner overlays' absolute cursor
        moves stay inside the child (avoids imgui's SetCursorPos assert + jitter).
        """
        slots = self._render_state.sticker_slots
        cols: int = _GRID_COLUMNS
        cell: float = float(_PREVIEW_THUMB_HEIGHT)
        arrow_w: float = float(_CAROUSEL_ARROW_W)

        max_offset: int = max(0, len(slots) - cols)
        offset: int = max(0, min(self._render_state.carousel_offset, max_offset))
        self._render_state.carousel_offset = offset

        self._draw_carousel_arrow("<", offset > 0, cell, arrow_w, delta=-1)
        imgui.same_line()
        for col in range(cols):
            i: int = offset + col
            slot: _StickerSlot | None = slots[i] if i < len(slots) else None
            self._draw_grid_cell(rc, i, slot, cell)
            imgui.same_line()
        self._draw_carousel_arrow(">", offset < max_offset, cell, arrow_w, delta=1)
        imgui.new_line()

    def _draw_carousel_arrow(
        self, glyph: str, enabled: bool, height: float, width: float, delta: int
    ) -> None:
        if not enabled:
            imgui.begin_disabled()
        if imgui.button(f"{glyph}##carousel_{delta}", size=(width, height)):
            self._render_state.carousel_offset += delta
        if not enabled:
            imgui.end_disabled()

    def _draw_grid_cell(
        self, rc: RenderControl, idx: int, slot: "_StickerSlot | None", cell: float
    ) -> None:
        if slot is None:
            preview_cell(
                id_=f"sticker_{idx}",
                cell_w=cell,
                texture_glo=None,
                texture_size=(0, 0),
                selected=False,
                armed=False,
                bg_color=COLOR.BG_SURFACE,
            )
            return

        glo: int | None = None
        size: tuple[int, int] = (0, 0)
        thumbnail: Image | Video | None = self._lazy_thumbnail(slot)
        if thumbnail is not None:
            thumbnail.update(imgui.get_time())  # animate video stickers
            glo = thumbnail.texture.glo
            size = thumbnail.texture.size

        result = preview_cell(
            id_=f"sticker_{idx}",
            cell_w=cell,
            texture_glo=glo,
            texture_size=size,
            selected=idx == self._render_state.selected_index
            and not self._render_state.in_flight,
            armed=self._render_state.sticker_delete_armed == slot.file_id,
            border_color=COLOR.ACCENT_PRIMARY
            if idx == self._render_state.selected_index
            else None,
            bg_color=COLOR.BG_SURFACE,
            overlay=lambda side: self._draw_sticker_emoji(rc, slot, side),
        )
        if result.clicked:
            self._render_state.selected_index = idx
        if result.delete_armed:
            self._render_state.sticker_delete_armed = slot.file_id
        elif result.delete_confirmed:
            self._render_state.sticker_delete_armed = ""
            self._enqueue(
                _Job(
                    kind="delete",
                    target_sticker_file_id=slot.file_id,
                    pack_set_name=self._render_state.active_pack_set_name,
                )
            )
        elif result.delete_cancelled:
            self._render_state.sticker_delete_armed = ""

    def _draw_sticker_emoji(
        self, rc: RenderControl, slot: _StickerSlot, side: float
    ) -> None:
        emoji: EmojiControl = self._emoji(rc)
        cur_emoji: str = slot.raw_sticker.emoji or "" if slot.raw_sticker else ""
        if not emoji.emoji_button(cur_emoji, side):
            return
        file_id: str = slot.file_id
        pack: str = self._render_state.active_pack_set_name
        emoji.open_emoji_picker(
            lambda e: self._enqueue(
                _Job(
                    kind="set_emoji",
                    target_sticker_file_id=file_id,
                    pack_set_name=pack,
                    emoji=e,
                )
            )
        )

    def _draw_status_slot(
        self, rc: RenderControl, artifact: "RenderedArtifact | None", full_width: float
    ) -> None:
        _ = rc
        # Fixed-height status slot (shared primitive): progress bar w/ overlay while
        # uploading, else the render stats, else empty — same height always, so the
        # carousel below never shifts.
        with status_slot("tg_status", full_width):
            if self._render_state.in_flight:
                prog: ExportProgress | None = self._render_state.last_progress
                imgui.progress_bar(
                    prog.fraction if prog is not None else 0.0,
                    size_arg=(full_width, 0.0),
                    overlay=prog.message if prog is not None else "Working...",
                )
            elif artifact is not None:
                try:
                    size_kb: int = artifact.path.stat().st_size // 1024
                    imgui.text_colored(
                        COLOR.FG_DIM,
                        f"{artifact.size[0]}x{artifact.size[1]} · "
                        f"{artifact.duration:.1f}s · {size_kb} KB",
                    )
                except OSError:
                    pass

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

    # Mutating ops drive the in-flight gate; refresh/link are background fetches
    # whose results aren't terminal ExportProgress events.
    _UPLOAD_KINDS = frozenset({"add", "delete", "delete_pack", "set_emoji"})

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
        # Both pools need the IPv4 bind: get_updates() uses get_updates_request,
        # a separate pool that otherwise dials the dead AAAA (conventions.md
        # ## Known quirks / vpn-stack Gotcha #4).
        bot = tg.Bot(
            token=self._tg.bot_token,
            request=_ipv4_request(),
            get_updates_request=_ipv4_request(),
        )
        try:
            await bot.initialize()
            return await op(bot)
        except (tg.error.InvalidToken, tg.error.Forbidden) as e:
            # The persisted connection is no longer valid (token revoked / bot
            # blocked) → reflect it in the Settings panel instead of leaving a
            # green "Connected" lie. initialize() itself can raise InvalidToken.
            self._push_event(
                _AuthEvent(
                    state=AuthState.ERROR,
                    message="Connection invalid — re-Connect in Settings.",
                )
            )
            raise ExporterError(_map_tg_error(e)) from e
        except tg.error.TelegramError as e:
            # No exception= traceback: loguru's annotated frames would write the
            # bot token (Bot repr + request URL) to the log in cleartext. The
            # cause's type+message is enough to diagnose (e.g. httpx.ConnectError).
            cause: str = (
                f"{type(e.__cause__).__name__}: {e.__cause__}" if e.__cause__ else ""
            )
            logger.error(f"Telegram API call failed: {e!r} (cause: {cause})")
            raise ExporterError(_map_tg_error(e)) from e
        finally:
            await bot.shutdown()

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
            elif job.kind == "delete":
                if job.target_sticker_file_id is None:
                    raise ExporterError("Delete job missing target")
                self._handle_delete(job.target_sticker_file_id, job.pack_set_name)
            elif job.kind == "set_emoji":
                if job.target_sticker_file_id is None:
                    raise ExporterError("Set-emoji job missing target")
                self._handle_set_emoji(
                    job.target_sticker_file_id, job.emoji, job.pack_set_name
                )
            elif job.kind == "delete_pack":
                self._handle_delete_pack(job.pack_set_name)
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
                if "stickerset_invalid" in str(e).lower():
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
        prepared: RenderedArtifact = self.prepare(job.artifact, {})
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
            # Only the prepared (re-encoded) file is ours to delete; the input
            # artifact belongs to the share tab, which owns its lifecycle.
            self._cleanup_paths(prepared.path)

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
                if "stickerset_invalid" in str(e).lower():
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
        self,
        bot: tg.Bot,
        user_id: int,
        pack_set_name: str,
        pack_title: str,
    ) -> None:
        # The user pressed Start, so the bot may DM them the pack link (one tap to
        # open it). Notification failure must not fail the upload (the sticker is
        # already in the set).
        link: str = f"https://t.me/addstickers/{pack_set_name}"
        try:
            await bot.send_message(
                chat_id=user_id,
                text=f'Added a sticker to "{pack_title}".\nOpen the pack: {link}',
            )
        except tg.error.TelegramError as e:
            logger.warning(f"Could not DM the user: {e}")

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

    def _handle_set_emoji(
        self, target_file_id: str, emoji: str, pack_set_name: str
    ) -> None:
        self._push_progress(ExportProgress(message="Updating emoji...", fraction=0.5))
        self._run_async(self._do_set_emoji(target_file_id, emoji))
        self._safe_refresh(pack_set_name)
        self._push_progress(
            ExportProgress(message="Emoji updated", fraction=1.0, is_terminal=True)
        )

    async def _do_set_emoji(self, target_file_id: str, emoji: str) -> None:
        async def op(bot: tg.Bot) -> None:
            await bot.set_sticker_emoji_list(target_file_id, [emoji])

        await self._with_bot(op)

    def _handle_delete_pack(self, pack_set_name: str) -> None:
        self._push_progress(ExportProgress(message="Deleting pack...", fraction=0.5))
        self._run_async(self._do_delete_pack(pack_set_name))
        self._push_progress(
            ExportProgress(message="Pack deleted", fraction=1.0, is_terminal=True)
        )

    async def _do_delete_pack(self, pack_set_name: str) -> None:
        async def op(bot: tg.Bot) -> None:
            await bot.delete_sticker_set(name=pack_set_name)

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
            self._render_state.sticker_delete_armed = ""
        else:
            self._render_state.last_progress = ev
            if ev.is_terminal:
                self._render_state.in_flight = False

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
