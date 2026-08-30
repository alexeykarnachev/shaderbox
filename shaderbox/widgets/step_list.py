"""The Steps section of the node panel: one row per render step (064).

READ-ONLY over the chain's structure. The shader is the only place a step is declared or
configured, so this section shows what the engine derived and never writes back into it.
Editing a step means editing the shader, which the Shader entry-point opens in one click.

Its whole window onto the engine is `Node.step_views()` -- a list of value objects. The
ping-pong pair, the front-buffer index and the per-step programs stay private, so either
side can be rewritten without touching the other.
"""

from imgui_bundle import imgui

from shaderbox.app import App
from shaderbox.core import StepView
from shaderbox.theme import COLOR, SPACE
from shaderbox.ui_primitives import (
    caption_text,
    ghost_button,
    preview_cell,
    small_caption,
)

_THUMB_W: float = 56.0


def _format_line(view: StepView) -> str:
    """The dim facts line: what this step's target IS."""
    bits = [f"{view.size[0]}x{view.size[1]}", view.dtype]
    bits.append("linear" if view.filter_linear else "nearest")
    if view.wrap:
        bits.append("repeat")
    if view.persist:
        bits.append("persist")
    return " · ".join(bits)


def _reads_line(view: StepView) -> str:
    """The dim wiring line: what this step READS, which is the chain's shape in text."""
    parts = list(view.reads)
    if view.reads_self:
        parts.append("itself (last frame)")
    if not parts:
        return "reads: nothing"
    return "reads: " + ", ".join(parts)


def draw_step_list(app: App) -> None:
    """Draw the Steps section, or nothing at all when the node declares no steps.

    A node with no steps is the overwhelming majority, and it must see no trace of the
    feature -- not even an empty header.
    """
    ui_node = app.ui_nodes.get(app.current_node_id)
    if ui_node is None:
        return
    views = ui_node.node.step_views()
    if not views:
        return

    # Frozen mid-copilot-turn like every other panel control: a click here retargets the
    # preview, and the turn may be reloading the node underneath it.
    imgui.begin_disabled(app.copilot_turn_active)
    small_caption(app.font_12, f"Steps ({len(views)})")

    for view in views:
        _draw_step_row(app, view)

    if app.viewed_step:
        imgui.dummy((0, SPACE.XS))
        if ghost_button("show the node's output##steps_unpin"):
            app.viewed_step = ""
        if imgui.is_item_hovered():
            imgui.set_tooltip("Stop showing a step and return to the finished frame")

    imgui.end_disabled()


def _draw_step_row(app: App, view: StepView) -> None:
    is_viewed = app.viewed_step == view.name

    imgui.begin_group()
    result = preview_cell(
        id_=f"step_{view.name}",
        cell_w=_THUMB_W,
        texture_glo=view.texture_glo,
        texture_size=view.size,
        selected=is_viewed,
        armed=False,
        border_color=COLOR.SELECT if is_viewed else None,
    )
    if result.clicked:
        # Click to pin, click again to unpin -- the same gesture both ways, so there is
        # no separate "stop showing" target to hunt for on the row.
        app.viewed_step = "" if is_viewed else view.name

    imgui.same_line()
    imgui.begin_group()
    label = f"{view.order_index + 1}  {view.name}"
    if is_viewed:
        imgui.text_colored(COLOR.SELECT, label)
    else:
        imgui.text_colored(COLOR.FG_PRIMARY, label)
    caption_text(_format_line(view))
    caption_text(_reads_line(view))
    imgui.end_group()
    imgui.end_group()

    if imgui.is_item_hovered():
        imgui.set_tooltip(
            f"`{view.sampler}` in the shader.\n"
            f"{'Showing this step — click to go back.' if is_viewed else 'Click to show this step in the preview.'}"
        )
    imgui.dummy((0, SPACE.XS))
