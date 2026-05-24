from pathlib import Path
from typing import Any

from shaderbox.exporters.base import (
    AuthState,
    Exporter,
    ExporterStatus,
    RenderControl,
    RenderedArtifact,
)
from shaderbox.integrations import IntegrationsStore
from shaderbox.ui_models import UINode

_COMING_SOON = "Coming soon"


class _StubExporter(Exporter):
    """A registered-but-disabled integration. The UI gates on `is_available`,
    so the worker/draw methods are never reached; they no-op defensively."""

    @property
    def is_available(self) -> bool:
        return False

    @property
    def unavailable_reason(self) -> str:
        return _COMING_SOON

    @property
    def auth_state(self) -> AuthState:
        return AuthState.UNCONFIGURED

    def begin_auth(self) -> None: ...
    def rebind(self, settings: dict[str, Any]) -> None: ...
    def set_media_dir(self, media_dir: Path) -> None: ...
    def set_integrations(self, store: IntegrationsStore) -> None: ...

    def status(self) -> ExporterStatus:
        return ExporterStatus()

    def current_settings(self) -> dict[str, Any]:
        return {}

    def draw_config_ui(self) -> None: ...
    def draw_target_panel(
        self,
        current_node: UINode | None,
        render_control: RenderControl,
    ) -> None: ...
    def update(self, current_node: UINode | None) -> None: ...

    def prepare(
        self, artifact: RenderedArtifact, settings: dict[str, Any]
    ) -> RenderedArtifact:
        return artifact

    def export(self, artifact: RenderedArtifact, settings: dict[str, Any]) -> None: ...
    def release(self) -> None: ...


class YouTubeExporterStub(_StubExporter):
    @property
    def exporter_id(self) -> str:
        return "youtube"

    @property
    def display_name(self) -> str:
        return "YouTube"


class XExporterStub(_StubExporter):
    @property
    def exporter_id(self) -> str:
        return "x"

    @property
    def display_name(self) -> str:
        return "X (Twitter)"
