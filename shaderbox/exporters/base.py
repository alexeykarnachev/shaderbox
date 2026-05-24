from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from shaderbox.integrations import IntegrationsStore
from shaderbox.render_preset import RenderPreset
from shaderbox.ui_models import UINode


@dataclass
class RenderControl:
    """The render-step hooks an outlet panel draws into its own UI.

    Lets each exporter own its full concise render+operations panel without
    importing `App`: the share tab supplies the mutable per-outlet state and
    the action callbacks.
    """

    emoji: str
    duration: float
    artifact: "RenderedArtifact | None"
    artifact_is_fresh: bool
    set_duration: Callable[[float], None]
    # Open the emoji picker; the picked emoji is delivered to `on_pick`.
    open_emoji_picker: Callable[[Callable[[str], None]], None]
    render: Callable[[], None]
    # Draws `char` as a clickable emoji-font glyph button of the given side; -> clicked.
    emoji_button: Callable[[str, float], bool]
    # Set `emoji` as the new-sticker emoji (the picker target for the New-sticker tile).
    set_emoji: Callable[[str], None]
    preview_texture_glo: int | None = None
    preview_size: tuple[int, int] = (0, 0)


class AuthState(Enum):
    UNCONFIGURED = "unconfigured"
    AUTHED = "authed"
    ERROR = "error"


class ExporterValueError(Exception):
    pass


class ExporterError(Exception):
    pass


@dataclass
class RenderedArtifact:
    path: Path
    is_video: bool
    duration: float
    size: tuple[int, int]


@dataclass
class ExportProgress:
    message: str
    fraction: float = 0.0
    is_terminal: bool = False
    is_error: bool = False
    url: str | None = None


@dataclass
class ExporterStatus:
    auth_state: AuthState = AuthState.UNCONFIGURED
    auth_message: str = ""
    last_progress: ExportProgress | None = None
    in_flight: bool = False


class Exporter(ABC):
    """Abstract base for an export target.

    Thread affinity:
      - render thread only:  exporter_id, display_name, auth_state,
                             is_available, unavailable_reason, begin_auth,
                             rebind, set_media_dir, set_integrations,
                             set_default_pack, current_default_pack, status,
                             current_settings, draw_config_ui,
                             draw_target_panel, update, release.
      - worker thread only:  prepare, export.

    Worker-thread methods MUST NOT construct or access `moderngl.*` (no
    `Image / Video / Canvas / Node / Texture`). The render-thread methods
    may; `RenderedArtifact` is GL-free precisely so it can cross the
    thread boundary as a value.
    """

    # Default-available; disabled stubs (YouTube/X) override to False. NOT
    # abstract — most exporters are simply available.
    @property
    def is_available(self) -> bool:
        return True

    @property
    def unavailable_reason(self) -> str:
        return ""

    # The render contract this outlet imposes (size/aspect/duration/fps/format/
    # byte caps). Drives the outlet's concise render controls AND prepare()'s
    # verification. Default is an unconstrained file export (the Render tab).
    def render_preset(self) -> RenderPreset:
        return RenderPreset()

    @property
    @abstractmethod
    def exporter_id(self) -> str: ...

    @property
    @abstractmethod
    def display_name(self) -> str: ...

    @property
    @abstractmethod
    def auth_state(self) -> AuthState: ...

    @abstractmethod
    def begin_auth(self) -> None: ...

    @abstractmethod
    def rebind(self, settings: dict[str, Any]) -> None: ...

    @abstractmethod
    def set_media_dir(self, media_dir: Path) -> None: ...

    @abstractmethod
    def set_integrations(self, store: IntegrationsStore) -> None: ...

    @abstractmethod
    def set_default_pack(self, set_name: str) -> None: ...

    @abstractmethod
    def current_default_pack(self) -> str: ...

    @abstractmethod
    def status(self) -> ExporterStatus: ...

    @abstractmethod
    def draw_config_ui(self) -> None: ...

    @abstractmethod
    def current_settings(self) -> dict[str, Any]: ...

    @abstractmethod
    def draw_target_panel(
        self,
        current_node: UINode | None,
        render_control: RenderControl,
    ) -> None: ...

    @abstractmethod
    def update(self, current_node: UINode | None) -> None: ...

    @abstractmethod
    def prepare(
        self, artifact: RenderedArtifact, settings: dict[str, Any]
    ) -> RenderedArtifact: ...

    @abstractmethod
    def export(self, artifact: RenderedArtifact, settings: dict[str, Any]) -> None: ...

    @abstractmethod
    def release(self) -> None: ...
