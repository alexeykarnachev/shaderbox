from imgui_bundle import imgui, imgui_ctx
from loguru import logger

from shaderbox.app import App
from shaderbox.exporters.base import Exporter, RenderControl, RenderedArtifact
from shaderbox.render_preset import RenderPreset
from shaderbox.tabs.share_state import OutletRenderState, TabState, render_for
from shaderbox.theme import COLOR, SPACE
from shaderbox.ui_models import UINode


def update(app: App) -> None:
    if app.share_tab_state is None:
        return
    current_node = app.ui_nodes.get(app.current_node_id)
    for exporter in app.exporter_registry.all():
        if not exporter.is_available:
            continue
        exporter.update(current_node)
        _surface_terminal_progress(app, exporter)


def _surface_terminal_progress(app: App, exporter: Exporter) -> None:
    assert app.share_tab_state is not None
    outlet = app.share_tab_state.outlet(exporter.exporter_id)
    progress = exporter.status().last_progress
    if (
        progress is None
        or not progress.is_terminal
        or progress is outlet.notified_progress
    ):
        return
    outlet.notified_progress = progress
    color = COLOR.STATE_ERROR if progress.is_error else COLOR.STATE_OK
    app.notifications.push(progress.message, color[:3])


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

    imgui.dummy(imgui.ImVec2(0, SPACE.SM))  # breathing room below the tab bar

    # Single outlet: no accordion (collapsing a list of one only adds a divider).
    if len(available) == 1:
        exporter = available[0]
        registry.set_active(exporter.exporter_id)
        with imgui_ctx.push_id(exporter.exporter_id):
            _draw_outlet(
                app, state, state.outlet(exporter.exporter_id), exporter, current_node
            )
        return

    # Accordion: one outlet open at a time, mirroring the one-popup-open invariant.
    for exporter in available:
        outlet: OutletRenderState = state.outlet(exporter.exporter_id)
        is_open: bool = registry.active_id == exporter.exporter_id

        imgui.set_next_item_open(is_open)
        header_open: bool = imgui.collapsing_header(
            f"{exporter.display_name}##outlet_{exporter.exporter_id}"
        )
        if header_open and not is_open:
            registry.set_active(exporter.exporter_id)
            is_open = True
        if not header_open:
            continue

        with imgui_ctx.push_id(exporter.exporter_id):
            _draw_outlet(app, state, outlet, exporter, current_node)


def _draw_outlet(
    app: App,
    state: TabState,
    outlet: OutletRenderState,
    exporter: Exporter,
    current_node: UINode | None,
) -> None:
    preset: RenderPreset = exporter.render_preset()

    artifact: RenderedArtifact | None = outlet.current_artifact
    if artifact is not None and not artifact.path.exists():
        outlet.set_artifact(None)
        artifact = None

    preview = outlet.preview_media()
    glo: int | None = None
    size: tuple[int, int] = (0, 0)
    if preview is not None:
        preview.update(imgui.get_time())
        glo = preview.texture.glo
        size = preview.texture.size

    def _do_render() -> None:
        _render(outlet, preset, current_node, state)

    def _set_duration(value: float) -> None:
        outlet.duration = value

    def _set_emoji(char: str) -> None:
        outlet.pending_emoji = char

    def _emoji_button(char: str, side: float) -> bool:
        clicked: bool = imgui.button("##emoji_btn", size=(side, side))
        if imgui.is_item_hovered():
            imgui.set_tooltip("Click to change emoji")
        rmin = imgui.get_item_rect_min()
        rmax = imgui.get_item_rect_max()
        font = app.font_emoji
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

    control = RenderControl(
        emoji=outlet.pending_emoji,
        duration=outlet.duration,
        artifact=artifact,
        artifact_is_fresh=outlet.artifact_is_fresh,
        set_duration=_set_duration,
        open_emoji_picker=app.open_emoji_picker,
        render=_do_render,
        emoji_button=_emoji_button,
        set_emoji=_set_emoji,
        preview_texture_glo=glo,
        preview_size=size,
    )
    exporter.draw_target_panel(current_node, control)


def _render(
    outlet: OutletRenderState,
    preset: RenderPreset,
    current_node: UINode | None,
    state: TabState,
) -> None:
    if current_node is None:
        return
    new_artifact: RenderedArtifact | None = render_for(
        current_node.node, preset, outlet.duration, state.scratch_dir
    )
    if new_artifact is None:
        return
    prev: RenderedArtifact | None = outlet.current_artifact
    outlet.set_artifact(new_artifact)
    if prev is not None and prev.path.exists():
        try:
            prev.path.unlink()
        except OSError as e:
            logger.warning(f"Failed to cleanup previous artifact: {e}")
