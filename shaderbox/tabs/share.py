from imgui_bundle import imgui, imgui_ctx
from loguru import logger

from shaderbox.app import App
from shaderbox.exporters.base import (
    Exporter,
    OutletUiDeps,
    RenderControl,
    RenderedArtifact,
)
from shaderbox.render_job import render_for
from shaderbox.render_preset import RenderPreset
from shaderbox.tabs.share_state import OutletRenderState, TabState
from shaderbox.theme import COLOR, SPACE
from shaderbox.ui_models import UIDocument


def update(app: App) -> None:
    if app.share_tab_state is None:
        return
    current_document = app.ui_documents.get(app.current_document_id)
    for exporter in app.exporter_registry.all():
        if not exporter.is_available:
            continue
        exporter.update(current_document)
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
    current_document = app.ui_documents.get(app.current_document_id)

    available: list[Exporter] = [e for e in registry.all() if e.is_available]
    if not available:
        imgui.text("No exporters available.")
        return

    imgui.dummy(imgui.ImVec2(0, SPACE.SM))  # breathing room below the tab bar

    # Single outlet: no accordion.
    if len(available) == 1:
        exporter = available[0]
        registry.set_active(exporter.exporter_id)
        with imgui_ctx.push_id(exporter.exporter_id):
            _draw_outlet(
                app,
                state,
                state.outlet(exporter.exporter_id),
                exporter,
                current_document,
            )
        return

    # Accordion: one outlet open at a time. Force-collapse only NON-active headers
    # each frame; leave the active one free to toggle, else it never shuts.
    for exporter in available:
        outlet: OutletRenderState = state.outlet(exporter.exporter_id)
        is_active: bool = registry.active_id == exporter.exporter_id

        if not is_active:
            imgui.set_next_item_open(False, imgui.Cond_.always)
        header_open: bool = imgui.collapsing_header(
            f"{exporter.display_name}##outlet_{exporter.exporter_id}"
        )
        if header_open and not is_active:
            registry.active_id = exporter.exporter_id
        elif not header_open and is_active:
            registry.active_id = ""
        if not header_open:
            continue

        with imgui_ctx.push_id(exporter.exporter_id):
            _draw_outlet(app, state, outlet, exporter, current_document)


def _draw_outlet(
    app: App,
    state: TabState,
    outlet: OutletRenderState,
    exporter: Exporter,
    current_document: UIDocument | None,
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
        # Defer the encode one frame so the "Rendering..." cue (and any modal) paints before the
        # synchronous encode freezes the loop — same path as the Render tab (tabs/render.py).
        # Capture the document ID, not the UIDocument: the run frame is a frame LATER, and a delete or
        # project switch in between releases that document's GL program, so a captured object would
        # encode a released document (a black artifact, published with no error).
        document_id = app.current_document_id

        def _run() -> None:
            target = app.ui_documents.get(document_id)
            if target is not None:
                _render(app, outlet, preset, target, state)

        app.render_defer.submit(_run)

    def _set_duration(value: float) -> None:
        outlet.duration = value

    deps = OutletUiDeps(
        glyph_font=app.font_emoji,
        open_glyph_picker=app.open_emoji_picker,
        open_settings=app.open_settings,
        outlet_extra_state=outlet.extra_state,
    )
    control = RenderControl(
        duration=outlet.duration,
        artifact=artifact,
        artifact_is_fresh=outlet.artifact_is_fresh,
        set_duration=_set_duration,
        render=_do_render,
        preview_texture_glo=glo,
        preview_size=size,
        extras=exporter.build_render_extras(deps),
    )
    exporter.draw_target_panel(current_document, control)


def _render(
    app: App,
    outlet: OutletRenderState,
    preset: RenderPreset,
    current_document: UIDocument | None,
    state: TabState,
) -> None:
    if current_document is None:
        return
    new_artifact: RenderedArtifact | None = render_for(
        current_document.document, preset, outlet.duration, state.scratch_dir
    )
    if new_artifact is None:
        # STALE-ARTIFACT GUARD, not just feedback: artifact_is_fresh is only ever set by
        # set_artifact, so returning here leaves the PREVIOUS render's True in place — and
        # the publish buttons gate on that flag. Silence here lets a user publish the old
        # artifact believing it is the one they just rendered.
        outlet.artifact_is_fresh = False
        app.notifications.push(
            "Render failed — nothing to publish", COLOR.STATE_ERROR[:3]
        )
        return
    prev: RenderedArtifact | None = outlet.current_artifact
    outlet.set_artifact(new_artifact)
    if prev is not None and prev.path.exists():
        try:
            prev.path.unlink()
        except OSError as e:
            logger.warning(f"Failed to cleanup previous artifact: {e}")
