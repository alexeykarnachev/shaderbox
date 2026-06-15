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
    ghost_button,
    play_stop_toggle,
    small_caption,
)
from shaderbox.util import format_auto_value, get_resolution_str, get_uniform_hash
from shaderbox.widgets.uniform import draw_ui_uniform, uniform_name_label


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

    imgui.dummy((0, SPACE.MD))
    _draw_entry_points(app)

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


# The Shader/Script label column: a tick gutter + the widest label, so both `open` buttons align.
_ENTRY_TICK_W = float(SPACE.MD)
_ENTRY_LABEL_W = 64.0


def _entry_row_label(active: bool, label: str) -> None:
    # One entry-point row label: an inset accent tick (049) marking the editor's active tab, then the
    # label in a fixed-width column so the `open` buttons line up across rows. align_text_to_frame_
    # padding centres the text on the button's row height (the font mix used to float it high). The tick
    # is a draw-list line (presence/colour only, never size — /imgui-ui §3) drawn over the gutter.
    imgui.align_text_to_frame_padding()
    pos = imgui.get_cursor_screen_pos()
    if active:
        h = imgui.get_frame_height()
        col = imgui.color_convert_float4_to_u32(COLOR.ACCENT_PRIMARY)
        imgui.get_window_draw_list().add_line(
            (pos.x, pos.y + 2.0), (pos.x, pos.y + h - 2.0), col, 2.0
        )
    imgui.dummy((_ENTRY_TICK_W, 0))
    imgui.same_line()
    imgui.text_colored(COLOR.FG_DIM, label)
    imgui.same_line(_ENTRY_TICK_W + _ENTRY_LABEL_W)


def _draw_entry_points(app: App) -> None:
    # The node's two entry-points (049): SHADER (GPU) and SCRIPT (CPU brain), each with an `open`
    # action that summons its tab into the editor (the node panel is "about this node"; the tab bar is
    # the editor's own state — `open` is a summoner, not a duplicate). The whole-node PLAY/STOP toggle
    # lives on the Script row (its true owner — it freezes/resumes the script's driven uniforms; the
    # brain keeps ticking). An accent tick marks whichever entry-point is the editor's active tab.
    # Frozen mid-copilot-turn (a write races the reload).
    node_id = app.current_node_id
    present = app.session.has_script(node_id)
    error = present and app.session.script_has_error(node_id)
    active = app.active_tab
    shader_active = (
        active is not None and active.kind == "shader" and active.node_id == node_id
    )
    script_active = (
        active is not None and active.kind == "script" and active.node_id == node_id
    )

    imgui.begin_disabled(app.copilot_turn_active)
    small_caption(app.font_12, "Entry points")

    _entry_row_label(shader_active, "Shader")
    if ghost_button("open##entry_shader"):
        app.ensure_shader_tab(node_id)
    if imgui.is_item_hovered():
        imgui.set_tooltip("Open the fragment shader (GPU)")

    _entry_row_label(script_active, "Script")
    open_tooltip = (
        "Node script error — click to open and fix"
        if error
        else "Open the node script (drives many uniforms)"
        if present
        else "Create + open a node script (drives many uniforms)"
    )
    open_color = COLOR.STATE_ERROR if error else COLOR.FG_SECONDARY
    if ghost_button("open##entry_script", text_color=open_color):
        app.open_script_for(node_id)
    if imgui.is_item_hovered():
        imgui.set_tooltip(open_tooltip)
    if present:
        imgui.same_line()
        playing = not app.current_node_ui_state_or_default.all_stopped
        if play_stop_toggle(
            "node",
            playing,
            tooltip="Stop the whole script (freeze all uniforms)"
            if playing
            else "Resume the whole script",
        ):
            app.set_node_all_stopped(node_id, playing)
    imgui.end_disabled()
