from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import imgui
from loguru import logger

from shaderbox.exporters.base import RenderedArtifact
from shaderbox.exporters.registry import ExporterRegistry
from shaderbox.media import MediaDetails
from shaderbox.ui_models import UINode


@dataclass
class TabState:
    scratch_dir: Path
    media_details: MediaDetails = field(
        default_factory=lambda: MediaDetails(is_video=True)
    )
    current_artifact: RenderedArtifact | None = None


def make_state(scratch_dir: Path) -> TabState:
    return TabState(scratch_dir=scratch_dir)


def update(
    state: TabState, registry: ExporterRegistry, current_node: UINode | None
) -> None:
    _ = state
    exporter = registry.get_active()
    if exporter is None:
        return
    exporter.update(current_node)


def draw(
    state: TabState,
    registry: ExporterRegistry,
    current_node: UINode | None,
    notifications_push: Callable[[str, tuple[float, float, float]], None],
) -> None:
    try:
        _draw_inner(state, registry, current_node)
    except Exception as e:
        logger.exception("Error in share tab")
        notifications_push(f"Error in share tab: {e!s}", (1.0, 0.0, 0.0))
        imgui.text_colored("An error occurred in the share tab.", *(1.0, 0.0, 0.0))
        imgui.text("Check the logs for more details.")


def _draw_inner(
    state: TabState,
    registry: ExporterRegistry,
    current_node: UINode | None,
) -> None:
    ids: list[str] = registry.ids()
    if not ids:
        imgui.text("No exporters registered.")
        return

    display_names: list[str] = []
    for eid in ids:
        exporter = registry.get(eid)
        assert exporter is not None, f"registry.ids() returned unknown id {eid}"
        display_names.append(exporter.display_name)

    current_idx: int = ids.index(registry.active_id) if registry.active_id in ids else 0
    changed, new_idx = imgui.combo("Exporter", current_idx, display_names)
    if changed and new_idx != current_idx:
        registry.set_active(ids[new_idx])

    exporter = registry.get_active()
    if exporter is None:
        imgui.text("No active exporter.")
        return

    imgui.spacing()
    imgui.separator()
    imgui.spacing()

    imgui.text_colored(f"Configuration for {exporter.display_name}", *(0.8, 0.8, 1.0))
    imgui.text_colored(
        "These settings are stored in the project's app_state.json.",
        *(1.0, 1.0, 0.0),
    )
    exporter.draw_config_ui()

    imgui.spacing()
    imgui.separator()
    imgui.spacing()

    available_width, available_height = imgui.get_content_region_available()
    left_width: float = 0.4 * available_width

    imgui.begin_child(
        "share_left", width=left_width, height=available_height, border=True
    )
    try:
        _draw_render_panel(state, current_node)
    finally:
        imgui.end_child()

    imgui.same_line()
    imgui.begin_child("share_right", border=True)
    try:
        exporter.draw_target_panel(state.current_artifact, current_node)
    finally:
        imgui.end_child()


def _draw_render_panel(state: TabState, current_node: UINode | None) -> None:
    imgui.text("Render output for export:")

    is_video: bool = imgui.checkbox("Video (.webm)", state.media_details.is_video)[1]
    if is_video != state.media_details.is_video:
        state.media_details.is_video = is_video

    state.media_details.fps = imgui.drag_int(
        "FPS", state.media_details.fps, min_value=10, max_value=60
    )[1]
    state.media_details.duration = imgui.drag_float(
        "Duration (s)",
        state.media_details.duration,
        change_speed=0.1,
        min_value=0.1,
        max_value=10.0,
    )[1]

    if current_node is None:
        imgui.text_colored("Select a node to render.", *(1.0, 1.0, 0.0))
        return

    canvas_size: tuple[int, int] = current_node.node.canvas.texture.size
    state.media_details.resolution_details.width = canvas_size[0]
    state.media_details.resolution_details.height = canvas_size[1]

    if imgui.button("Render", width=120):
        _render_into_state(state, current_node)

    artifact: RenderedArtifact | None = state.current_artifact
    if artifact is not None:
        imgui.spacing()
        imgui.text(f"Artifact: {artifact.path.name}")
        imgui.text(f"  size={artifact.size[0]}x{artifact.size[1]}")
        imgui.text(f"  duration={artifact.duration:.2f}s")
        if artifact.path.exists():
            imgui.text(f"  bytes={artifact.path.stat().st_size}")
        else:
            imgui.text("  (file missing)")


def _render_into_state(state: TabState, current_node: UINode) -> None:
    is_video: bool = state.media_details.is_video
    artifact_path: Path = _make_artifact_path(state.scratch_dir, is_video)
    state.media_details.file_details.path = str(artifact_path)
    try:
        rendered_details: MediaDetails = current_node.node.render_media(
            state.media_details
        )
    except Exception as e:
        logger.error(f"Failed to render artifact: {e}")
        if artifact_path.exists():
            try:
                artifact_path.unlink()
            except OSError as cleanup_err:
                logger.warning(f"Failed to cleanup partial render: {cleanup_err}")
        return

    new_artifact = RenderedArtifact(
        path=artifact_path,
        is_video=is_video,
        duration=rendered_details.duration,
        size=(
            rendered_details.resolution_details.width,
            rendered_details.resolution_details.height,
        ),
    )
    if state.current_artifact is not None and state.current_artifact.path.exists():
        try:
            state.current_artifact.path.unlink()
        except OSError as e:
            logger.warning(f"Failed to cleanup previous artifact: {e}")
    state.current_artifact = new_artifact


def _make_artifact_path(scratch_dir: Path, is_video: bool) -> Path:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    ext: str = ".webm" if is_video else ".png"
    return scratch_dir / f"{uuid4()}{ext}"
