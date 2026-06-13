from uuid import uuid4

from imgui_bundle import imgui, imgui_ctx
from OpenGL.GL import GL_SAMPLER_2D

from shaderbox.app import App
from shaderbox.glyph_tables import TABLE_UNIFORMS
from shaderbox.theme import COLOR, SIZE, SPACE
from shaderbox.ui_models import (
    UIUniform,
    UniformSortKey,
    load_node_from_dir,
    sort_uniform_hashes,
)
from shaderbox.ui_primitives import (
    button,
    caption_text,
    confirm_delete_popup,
    danger_button,
    ghost_button,
    notice_strip,
    reset_state_button,
    small_caption,
)
from shaderbox.util import format_auto_value, get_resolution_str, get_uniform_hash
from shaderbox.widgets.uniform import draw_ui_uniform, uniform_name_label

_MAX_BRAIN_SOFT = (
    3  # cap the soft-key list so a typo-spewing brain can't blow the panel height
)


def _section_break() -> None:
    imgui.spacing()
    imgui.separator()
    imgui.spacing()


def _draw_auto_row(app: App, uniforms: list[UIUniform]) -> None:
    # Engine-driven uniforms (u_time / u_aspect / u_resolution): one inline row,
    # names participate in the code<->panel hover/jump bridge; value is read-only.
    ui_node = app.ui_nodes[app.current_node_id]
    imgui.push_font(app.font_12, app.font_12.legacy_size)
    for i, u in enumerate(uniforms):
        if i > 0:
            imgui.same_line(spacing=float(SPACE.LG))
        name_w = imgui.calc_text_size(u.name).x
        uniform_name_label(
            app, u.name, name_w, text_color=COLOR.STATE_INFO, accent=COLOR.STATE_INFO
        )
        imgui.same_line(spacing=float(SPACE.MD))
        value = ui_node.node.uniform_values.get(u.name)
        imgui.text_colored(COLOR.FG_DIM, format_auto_value(value))
    imgui.pop_font()


def draw(app: App) -> None:
    if not (ui_node := app.ui_nodes.get(app.current_node_id)):
        return

    imgui.spacing()

    standard_resolutions = [
        (1080, 1920),
        (960, 1280),
        (1080, 1080),
        (1280, 960),
        (1920, 1080),
        (3440, 1440),
    ]

    cw, ch = ui_node.node.canvas.texture.size
    current_size: tuple[int, int] = (cw, ch)

    uniform_resolutions = []
    matching_uniforms = []
    uniform_sizes = set()
    for uniform in ui_node.node.get_active_uniforms():
        if getattr(uniform, "gl_type", None) == GL_SAMPLER_2D:
            value = ui_node.node.uniform_values[uniform.name]

            w, h = value.texture.size
            if (w, h) == current_size:
                matching_uniforms.append(uniform.name)
            else:
                uniform_resolutions.append((w, h, uniform.name))
                uniform_sizes.add((w, h))

    current_name = ", ".join(matching_uniforms) if matching_uniforms else None
    resolution_items = [get_resolution_str(None, *current_size)]
    resolution_sizes: list[tuple[int, int]] = [current_size]
    for w, h, name in uniform_resolutions:
        resolution_items.append(get_resolution_str(name, w, h))
        resolution_sizes.append((w, h))
    for w, h in standard_resolutions:
        if (w, h) != current_size and (w, h) not in uniform_sizes:
            resolution_items.append(get_resolution_str(None, w, h))
            resolution_sizes.append((w, h))

    node_ui_state = app.current_node_ui_state_or_default

    combo_offset = SIZE.NAME_INPUT_W + SPACE.XL
    resolution_label = "Resolution"
    if current_name:
        resolution_label += f" ({current_name})"

    small_caption(app.font_12, "Node name")
    imgui.same_line(combo_offset)
    small_caption(app.font_12, resolution_label)

    imgui.set_next_item_width(SIZE.NAME_INPUT_W)
    ui_node.ui_state.ui_name = imgui.input_text_with_hint(
        "##node_name", "node name", ui_node.ui_state.ui_name
    )[1]

    imgui.same_line(combo_offset)
    imgui.set_next_item_width(SIZE.RES_COMBO_W)
    new_res_idx = imgui.combo("##resolution", 0, resolution_items)[1]
    if new_res_idx != 0:
        w, h = resolution_sizes[new_res_idx]
        ui_node.node.canvas.set_size((w, h))
        app.notifications.push(
            f"Canvas resolution changed: {resolution_items[new_res_idx]}"
        )

    imgui.same_line()
    if ghost_button("...##node_actions"):
        imgui.open_popup("node_actions_popup")
    with imgui_ctx.begin_popup("node_actions_popup") as popup:
        if popup and imgui.selectable("Save as template", False)[0]:
            dir = app.save_ui_node(
                ui_node,
                root_dir=app.node_templates_dir,
                dir_name=str(uuid4()),
            )
            app.ui_node_templates[dir.name] = load_node_from_dir(dir)
            app.notifications.push("New template created")
        if popup:
            has_brain = app.session.get_brain_status(app.current_node_id) is not None
            if imgui.menu_item_simple(
                "New node-brain script",
                enabled=not has_brain and not app.copilot_turn_active,
            ) and not (has_brain or app.copilot_turn_active):
                app.create_script_for(app.current_node_id, None)

    _section_break()

    ui_uniforms = node_ui_state.ui_uniforms

    active_uniform_hashes = []
    auto_hashes = []
    for uniform in ui_node.node.get_active_uniforms():
        if (
            uniform.name in TABLE_UNIFORMS
        ):  # engine glyph tables — pure machinery, no row
            continue
        hash = get_uniform_hash(uniform)
        if hash not in ui_uniforms:
            ui_uniforms[hash] = UIUniform.from_uniform(uniform)
        ui_uniforms[hash].snap_input_type()
        if ui_uniforms[hash].input_type == "auto":
            auto_hashes.append(hash)
        else:
            active_uniform_hashes.append(hash)

    sort_keys: list[UniformSortKey] = ["code", "name", "type"]
    imgui.set_next_item_width(SIZE.SORT_COMBO_W)
    if imgui.begin_combo(
        "##uniform_sort_key", f"Sort by: {node_ui_state.uniform_sort_key}"
    ):
        for key in sort_keys:
            if imgui.selectable(key, key == node_ui_state.uniform_sort_key)[0]:
                node_ui_state.uniform_sort_key = key
        imgui.end_combo()

    imgui.same_line()
    arrow = "v" if node_ui_state.uniform_sort_desc else "^"
    if button(f"{arrow}##uniform_sort_dir", width=SIZE.BTN_SM_H):
        node_ui_state.uniform_sort_desc = not node_ui_state.uniform_sort_desc

    if auto_hashes:
        imgui.same_line(spacing=float(SPACE.XL))
        _draw_auto_row(app, [ui_uniforms[h] for h in auto_hashes])

    imgui.dummy((0, SPACE.MD))

    _draw_brain_strip(app)

    small_caption(app.font_12, "Right-click a uniform for script actions")
    imgui.dummy((0, SPACE.SM))

    sorted_hashes = sort_uniform_hashes(
        active_uniform_hashes,
        ui_uniforms,
        node_ui_state.uniform_sort_key,
        node_ui_state.uniform_sort_desc,
    )

    # nav_flattened: Tab/arrows reach the sliders without an Enter/Esc window boundary.
    with imgui_ctx.begin_child(
        "ui_uniforms", child_flags=imgui.ChildFlags_.nav_flattened
    ):
        for hash in sorted_hashes:
            draw_ui_uniform(app, ui_uniforms[hash])
            imgui.dummy((0, SPACE.SM))


def _draw_brain_strip(app: App) -> None:
    # The node-level node-brain banner (feature 042): the ONLY home for a brain's compile error
    # (it drives zero rows) + its homeless soft-key errors (typo/orphan/engine-owned keys that
    # name no uniform row). Drawn only when the node has a script.py; a real uniform the brain
    # drives surfaces the brain chip on its own row, not here.
    status = app.session.get_brain_status(app.current_node_id)
    if status is None:
        return
    if status.sentinel_error is not None:
        # A broken brain drives ZERO rows, so this strip is its ONLY home — Detach must be reachable
        # here (the exact case a user wants to remove it).
        notice_strip(
            "brain_sentinel",
            status.sentinel_error.message,
            tone="error",
            line=status.sentinel_error.line,
            on_click=lambda: app.open_script_file(app.current_node_id, "script.py"),
        )
    else:
        small_caption(
            app.font_12, f"node-brain  ·  drives {status.driven_count} uniforms"
        )
        imgui.same_line()
        if reset_state_button("brain_reset"):
            app.session.reset_script(app.current_node_id, "script.py")
        if imgui.is_item_hovered():
            imgui.set_tooltip("Restart the node-brain (re-run __init__)")
    imgui.begin_disabled(app.copilot_turn_active)
    if danger_button("Detach node-brain##brain_detach"):
        imgui.open_popup("brain_detach_confirm")
    imgui.end_disabled()
    confirm_delete_popup(
        "brain_detach_confirm",
        "Detach the node-brain? It drives all its uniforms.",
        lambda: app.detach_script(app.current_node_id, "script.py"),
    )
    for key, err in status.soft_errors[:_MAX_BRAIN_SOFT]:
        notice_strip(f"brain_soft_{key}", f"{key}: {err.message}", tone="warn")
    extra = len(status.soft_errors) - _MAX_BRAIN_SOFT
    if extra > 0:
        caption_text(f"+{extra} more brain key issues")
    imgui.dummy((0, SPACE.MD))
