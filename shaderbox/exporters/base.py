from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from shaderbox.exporters.integrations import IntegrationsStore
from shaderbox.render_preset import RenderPreset
from shaderbox.ui_models import UINode


@dataclass
class RenderControl:
    """The render-step hooks an outlet panel draws into its own UI.

    Lets each exporter own its full concise render+operations panel without
    importing `App`: the share tab supplies the mutable per-outlet state and
    the action callbacks. Pure render plumbing — knows nothing of any
    exporter's domain concepts. Per-exporter extras (e.g. glyph-overlay
    affordances) ride in `extras`, an opaque bag only the owning exporter
    interprets.
    """

    duration: float
    artifact: "RenderedArtifact | None"
    artifact_is_fresh: bool
    set_duration: Callable[[float], None]
    render: Callable[[], None]
    preview_texture_glo: int | None = None
    preview_size: tuple[int, int] = (0, 0)
    extras: "Mapping[str, Any] | None" = None


@dataclass
class OutletUiDeps:
    """Generic app-level UI capabilities an exporter may need to build its
    per-panel extras, handed over without exposing `App`.

    `glyph_font` is an `imgui.ImFont` for symbol/glyph overlays; kept as `Any`
    so this generic module doesn't depend on imgui types. `open_glyph_picker`
    opens the symbol picker, delivering the chosen glyph to the given callback.
    `open_settings(focus)` opens the app's settings popup (for a panel that needs to
    send the user to configure credentials); `focus` is a SettingsField key that
    expands + focuses that field on open ("" = no specific field). `outlet_extra_state`
    is a free-form per-outlet scratch dict the exporter reads/writes for its own UI
    state — the generic outlet doesn't name it.
    """

    glyph_font: Any
    open_glyph_picker: Callable[[Callable[[str], None]], None]
    open_settings: Callable[[str], None]
    outlet_extra_state: dict[str, Any]


class AuthState(Enum):
    UNCONFIGURED = "unconfigured"
    LINKING = (
        "linking"  # a connect/link job is in flight (the copilot connect-await floor)
    )
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
                             rebind, set_media_dir, set_integrations, status,
                             publish, current_settings, draw_config_ui,
                             build_render_extras, draw_target_panel, update,
                             release.
      - any thread (GL-free scalar reads):  is_connected, render_preset.
                             The copilot worker's pre-gate cred check reads
                             these directly (no GL, no queue mutation — just
                             `_render_state` field reads under the GIL).
      - worker thread only:  prepare, and the _do_*/_handle_* job handlers.

    `publish` only ENQUEUES the upload job onto the exporter's own worker (a
    render-thread queue write); the actual upload runs later on that worker.
    Worker-thread methods MUST NOT construct or access `moderngl.*` (no
    `Image / Video / Canvas / Node / Texture`). The render-thread methods
    may; `RenderedArtifact` is GL-free precisely so it can cross the
    thread boundary as a value.
    """

    # Default-available; a not-yet-usable exporter overrides to False (the UI
    # gates on it). NOT abstract — most exporters are simply available.
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

    # Per-exporter UI extras carried in `RenderControl.extras`, built from
    # generic app capabilities. Default: no extras. An exporter that needs
    # app-level affordances (e.g. a glyph font + picker) overrides this.
    def build_render_extras(self, deps: "OutletUiDeps") -> "Mapping[str, Any] | None":
        _ = deps
        return None

    @property
    @abstractmethod
    def exporter_id(self) -> str: ...

    @property
    @abstractmethod
    def display_name(self) -> str: ...

    @property
    def config_field(self) -> str:
        # The SettingsField key (popups.settings.SettingsField) of this exporter's primary
        # credential input — drives open_settings(focus=...) expand+focus. "" = no such field.
        return ""

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
    def status(self) -> ExporterStatus: ...

    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def draw_config_ui(self, focus: bool = False) -> None: ...

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

    # Enqueue the upload job onto the exporter's own worker (a render-thread queue
    # write; the upload runs later on that worker). `settings` carries the per-exporter
    # job params (Telegram: pack/emoji; YouTube: title/description/is_short).
    @abstractmethod
    def publish(self, artifact: RenderedArtifact, settings: dict[str, Any]) -> None: ...

    @abstractmethod
    def release(self) -> None: ...
