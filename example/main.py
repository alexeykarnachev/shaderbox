from pathlib import Path

from shaderbox.renderer import Renderer
from shaderbox.utils import scale_size

_THIS_DIR = Path(__file__).parent

if __name__ == "__main__":
    main_fs = _THIS_DIR / "shaders" / "main.glsl"

    renderer = Renderer(is_headless=False)

    photo_texture = renderer.load_texture("photo.jpeg")
    depth_texture = renderer.load_texture("depth.png")
    output_size = scale_size(photo_texture.size, 600)

    main_node = renderer.create_node(
        name="Main",
        fs_source=main_fs,
        output_size="u_photo_texture",
        uniforms={
            "u_photo_texture": photo_texture,
            "u_depth_texture": depth_texture,
            "u_time": lambda: renderer.render_time,
        },
    )

    try:
        renderer.run_editor(60)
    finally:
        renderer.cleanup()
