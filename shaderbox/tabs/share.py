from pathlib import Path
from uuid import uuid4

from imgui_bundle import imgui, imgui_ctx
from loguru import logger

from shaderbox.app import App
from shaderbox.exporters.base import Exporter, RenderedArtifact
from shaderbox.media import MediaDetails
from shaderbox.tabs.share_state import TabState
from shaderbox.theme import COLOR, SIZE
from shaderbox.ui_models import UINode


def update(app: App) -> None:
    if app.share_tab_state is None:
        return
    exporter = app.exporter_registry.get_active()
    if exporter is None:
        return
    current_node = app.ui_nodes.get(app.current_node_id)
    exporter.update(current_node)


def draw(app: App) -> None:
    if app.share_tab_state is None:
        return
    try:
        _draw_inner(app)
    except Exception as e:
        logger.exception("Error in share tab")
        app.notifications.push(f"Error in share tab: {e!s}", COLOR.STATE_ERROR[:3])
        imgui.text_colored(COLOR.STATE_ERROR, "An error occurred in the share tab.")
        imgui.text("Check the logs for more details.")


def _draw_inner(app: App) -> None:
    assert app.share_tab_state is not None
    state = app.share_tab_state
    registry = app.exporter_registry
    current_node = app.ui_nodes.get(app.current_node_id)

    available: list[Exporter] = [e for e in registry.all() if e.is_available]
    if not available:
        imgui.text("No exporters available.")
        return

    if len(available) > 1:
        names: list[str] = [e.display_name for e in available]
        ids: list[str] = [e.exporter_id for e in available]
        current_idx: int = (
            ids.index(registry.active_id) if registry.active_id in ids else 0
        )
        changed, new_idx = imgui.combo("Exporter", current_idx, names)
        if changed and new_idx != current_idx:
            registry.set_active(ids[new_idx])

    exporter = registry.get_active()
    if exporter is None:
        imgui.text("No active exporter.")
        return

    imgui.text_colored(
        COLOR.STATE_INFO,
        f"{exporter.display_name} — configure credentials in Settings.",
    )
    imgui.spacing()
    imgui.separator()
    imgui.spacing()

    avail = imgui.get_content_region_avail()
    available_width = avail.x
    available_height = avail.y
    left_width: float = 0.4 * available_width

    with imgui_ctx.begin_child(
        "share_left",
        size=imgui.ImVec2(left_width, available_height),
        child_flags=imgui.ChildFlags_.borders,
    ):
        _draw_render_panel(app, state, current_node)

    # Pass the artifact only if its file still exists — a prior export/render may
    # have consumed it; acting on a missing path ffmpeg-fails.
    artifact: RenderedArtifact | None = state.current_artifact
    if artifact is not None and not artifact.path.exists():
        artifact = None

    imgui.same_line()
    with imgui_ctx.begin_child("share_right", child_flags=imgui.ChildFlags_.borders):
        exporter.draw_target_panel(artifact, current_node, state.pending_emoji)


def _draw_render_panel(app: App, state: TabState, current_node: UINode | None) -> None:
    imgui.text("Render output for export:")

    imgui.text("Emoji:")
    imgui.same_line()
    imgui.push_font(app.font_emoji, app.font_emoji.legacy_size)
    imgui.text(state.pending_emoji)
    imgui.pop_font()
    imgui.same_line()
    if imgui.button("Change emoji"):
        app.open_emoji_picker()

    is_video: bool = imgui.checkbox("Video (.webm)", state.media_details.is_video)[1]
    if is_video != state.media_details.is_video:
        state.media_details.is_video = is_video

    state.media_details.fps = imgui.drag_int(
        "FPS", state.media_details.fps, v_min=10, v_max=60
    )[1]
    state.media_details.duration = imgui.drag_float(
        "Duration (s)",
        state.media_details.duration,
        v_speed=0.1,
        v_min=0.1,
        v_max=10.0,
    )[1]

    if current_node is None:
        imgui.text_colored(COLOR.STATE_WARN, "Select a node to render.")
        return

    canvas_size: tuple[int, int] = current_node.node.canvas.texture.size
    state.media_details.resolution_details.width = canvas_size[0]
    state.media_details.resolution_details.height = canvas_size[1]

    if imgui.button("Render", size=(SIZE.BTN_MD_W, 0)):
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
