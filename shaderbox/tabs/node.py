from uuid import uuid4

import pyperclip
from imgui_bundle import imgui, imgui_ctx
from OpenGL.GL import GL_SAMPLER_2D

from shaderbox.app import App
from shaderbox.theme import COLOR, SIZE
from shaderbox.ui_models import UIUniform, load_node_from_dir
from shaderbox.ui_utils import get_resolution_str, get_uniform_hash
from shaderbox.widgets.uniform import (
    draw_selected_ui_uniform_settings,
    draw_ui_uniform,
)


def draw(app: App) -> None:
    if not (ui_node := app.ui_nodes.get(app.current_node_id)):
        return

    full_file_path = app.nodes_dir / ui_node.id / "shader.frag.glsl"
    local_file_path = full_file_path.relative_to(app.project_dir)

    imgui.push_style_color(imgui.Col_.text, COLOR.FG_DIM)
    if imgui.selectable(str(local_file_path), False)[0]:
        pyperclip.copy(str(full_file_path))
        app.notifications.push("Copied to clipboard!")
    imgui.pop_style_color()

    imgui.spacing()
    ui_node.ui_state.ui_name = imgui.input_text("Name", ui_node.ui_state.ui_name)[1]
    imgui.spacing()

    if imgui.button("Edit code", size=(SIZE.BTN_SM_W, 0)):
        app.edit_current_node_fs_file()

    imgui.same_line()
    if imgui.button("Open dir", size=(SIZE.BTN_SM_W, 0)):
        app.open_current_node_dir()

    imgui.same_line()
    if imgui.button("Save as template"):
        dir = app.save_ui_node(
            ui_node,
            root_dir=app.node_templates_dir,
            dir_name=str(uuid4()),
        )
        app.ui_node_templates[dir.name] = load_node_from_dir(dir)
        app.notifications.push("New template created")

    imgui.new_line()
    imgui.separator()
    imgui.spacing()

    standard_resolutions = [
        (1080, 1920),
        (960, 1280),
        (1080, 1080),
        (1280, 960),
        (1920, 1080),
        (3440, 1440),
    ]

    uniform_resolutions = []
    matching_uniforms = []
    uniform_sizes = set()
    for uniform in ui_node.node.get_active_uniforms():
        if getattr(uniform, "gl_type", None) == GL_SAMPLER_2D:  # type: ignore
            value = ui_node.node.uniform_values[uniform.name]

            w, h = value.texture.size
            if (w, h) == ui_node.node.canvas.texture.size:
                matching_uniforms.append(uniform.name)
            else:
                uniform_resolutions.append((w, h, uniform.name))
                uniform_sizes.add((w, h))

    resolution_items = []

    for w, h, name in uniform_resolutions:
        resolution_items.append(get_resolution_str(name, w, h))

    for w, h in standard_resolutions:
        if (w, h) != ui_node.node.canvas.texture.size and (
            w,
            h,
        ) not in uniform_sizes:
            resolution_items.append(get_resolution_str(None, w, h))

    imgui.text(
        "Current resolution: "
        + get_resolution_str(None, *ui_node.node.canvas.texture.size)
    )

    imgui.spacing()

    if matching_uniforms:
        imgui.same_line()
        imgui.text_colored(COLOR.FG_DIM, "(" + ", ".join(matching_uniforms) + ")")

    node_ui_state = app.current_node_ui_state_or_default
    node_ui_state.resolution_idx = imgui.combo(
        "##resolution_idx", node_ui_state.resolution_idx, resolution_items
    )[1]

    imgui.same_line()
    if imgui.button("Apply##resolution"):
        resolution_str = resolution_items[node_ui_state.resolution_idx]
        w, h = map(
            int,
            resolution_str.split(" | ")[0].split("x"),
        )
        ui_node.node.canvas.set_size((w, h))
        app.notifications.push(f"Canvas resolution changed: {resolution_str}")

    imgui.new_line()
    imgui.separator()
    imgui.spacing()

    ui_uniforms = node_ui_state.ui_uniforms

    active_uniform_hashes = []
    for uniform in ui_node.node.get_active_uniforms():
        active_uniform_hashes.append(get_uniform_hash(uniform))
        hash = get_uniform_hash(uniform)
        if hash not in ui_uniforms:
            ui_uniforms[hash] = UIUniform.from_uniform(uniform)

    with imgui_ctx.begin_child(
        "ui_uniforms",
        size=imgui.ImVec2(imgui.get_content_region_avail().x // 2, 0),
    ):
        imgui.push_style_color(imgui.Col_.separator, COLOR.BG_FRAME)
        for hash in active_uniform_hashes:
            draw_ui_uniform(app, ui_uniforms[hash])
            imgui.spacing()
            imgui.separator()
        imgui.pop_style_color()

    if node_ui_state.selected_uniform_name:
        imgui.same_line()
        with imgui_ctx.begin_child(
            "selected_uniform_settings", child_flags=imgui.ChildFlags_.borders
        ):
            draw_selected_ui_uniform_settings(app)
