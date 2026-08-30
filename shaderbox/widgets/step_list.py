"""The Steps section of the node panel: one row per render step (064).

The chain's STRUCTURE comes from the shader -- which steps exist, and what each one reads
-- so this section never writes back into GLSL text. A step's TARGET is node state, so its
size, format and filtering ARE edited here: the shader says what the steps are, the panel
says how each one is set up, and neither can contradict the other.

Its whole window onto the engine is `Node.step_views()` -- a list of value objects. The
ping-pong pair, the front-buffer index and the per-step programs stay private, so either
side can be rewritten without touching the other.
"""

from imgui_bundle import imgui

from shaderbox.app import App
from shaderbox.core import StepView
from shaderbox.step_spec import DTYPES, StepConfig
from shaderbox.theme import COLOR, SPACE
from shaderbox.ui_models import UINode
from shaderbox.ui_primitives import (
    caption_text,
    ghost_button,
    labeled_combo,
    preview_cell,
    small_caption,
)

_THUMB_W: float = 56.0
_COMBO_W: float = 68.0

_SCALES: tuple[tuple[str, float], ...] = (
    ("full", 1.0),
    ("1/2", 0.5),
    ("1/4", 0.25),
    ("1/8", 0.125),
    ("1/16", 0.0625),
    ("1/32", 0.03125),
)
_FILTERS = ("linear", "nearest")
_WRAPS = ("clamp", "repeat")


def _scale_index(scale: float) -> int:
    """The nearest preset to `scale` — a value set elsewhere snaps to what it is closest to."""
    return min(range(len(_SCALES)), key=lambda i: abs(_SCALES[i][1] - scale))


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

    # Frozen mid-copilot-turn like every other panel control: an edit here recompiles the
    # node, and the turn may be reloading it underneath.
    imgui.begin_disabled(app.copilot_turn_active)
    small_caption(app.font_12, f"Steps ({len(views)})")

    for view in views:
        _draw_step_row(app, ui_node, view)

    if app.viewed_step:
        imgui.dummy((0, SPACE.XS))
        if ghost_button("show the node's output##steps_unpin"):
            app.viewed_step = ""
        if imgui.is_item_hovered():
            imgui.set_tooltip("Stop showing a step and return to the finished frame")

    imgui.end_disabled()


def _apply_config(ui_node: UINode, name: str, config: StepConfig) -> None:
    """Write a step's configuration back to node state and rebuild the chain.

    Both halves matter: `ui_state` is what gets saved, and `node.step_configs` is what the
    engine reads on its next compile.
    """
    ui_node.ui_state.step_configs[name] = config
    ui_node.node.step_configs[name] = config
    node = ui_node.node
    node.invalidate()
    node.compile()


def _draw_step_row(app: App, ui_node: UINode, view: StepView) -> None:
    is_viewed = app.viewed_step == view.name
    config = ui_node.node.step_configs.get(view.name, StepConfig())

    imgui.push_id(f"step_{view.name}")
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
    imgui.text_colored(COLOR.SELECT if is_viewed else COLOR.FG_PRIMARY, label)
    caption_text(f"{view.size[0]}x{view.size[1]}  ·  {_reads_line(view)}")

    changed_scale, scale_idx = labeled_combo(
        "size", _scale_index(config.scale), [name for name, _ in _SCALES], _COMBO_W
    )
    imgui.same_line()
    changed_dtype, dtype_idx = labeled_combo(
        "format", DTYPES.index(config.dtype), list(DTYPES), _COMBO_W
    )
    imgui.same_line()
    changed_filter, filter_idx = labeled_combo(
        "filter", 0 if config.filter_linear else 1, list(_FILTERS), _COMBO_W
    )
    imgui.same_line()
    changed_wrap, wrap_idx = labeled_combo(
        "edge", 1 if config.wrap else 0, list(_WRAPS), _COMBO_W
    )

    imgui.end_group()
    imgui.end_group()

    if changed_scale or changed_dtype or changed_filter or changed_wrap:
        _apply_config(
            ui_node,
            view.name,
            StepConfig(
                scale=_SCALES[scale_idx][1],
                size=config.size,
                dtype=DTYPES[dtype_idx],
                filter_linear=filter_idx == 0,
                wrap=wrap_idx == 1,
                persist=config.persist,
            ),
        )

    imgui.pop_id()
    imgui.dummy((0, SPACE.SM))
