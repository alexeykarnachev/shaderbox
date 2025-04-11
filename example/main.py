from pathlib import Path

from shaderbox.renderer import Renderer
from shaderbox.utils import scale_size

_THIS_DIR = Path(__file__).parent

if __name__ == "__main__":
    parallax_fs = _THIS_DIR / "shaders" / "parallax.glsl"
    bright_pass_fs = _THIS_DIR / "shaders" / "bright_pass.glsl"

    downscale_fs = """
    #version 460
    in vec2 vs_uv;
    out vec4 fs_color;
    uniform sampler2D u_input_texture;
    void main() {
        fs_color = texture(u_input_texture, vs_uv);
    }
    """

    blur_fs = """
    #version 460
    in vec2 vs_uv;
    out vec4 fs_color;
    uniform sampler2D u_blur_input;
    uniform vec2 u_pixel_size;
    void main() {
        vec4 color = vec4(0.0);
        for (int i = -1; i <= 1; i++)
            for (int j = -1; j <= 1; j++)
                color += texture(u_blur_input, vs_uv + vec2(float(i), float(j)) * u_pixel_size);
        fs_color = color / 9.0;
    }
    """

    outline_fs = _THIS_DIR / "shaders" / "outline.glsl"

    combine_fs = """
    #version 460
    in vec2 vs_uv;
    out vec4 fs_color;
    uniform sampler2D u_base_image;
    uniform sampler2D u_bloom_pass1;
    uniform sampler2D u_bloom_pass2;
    uniform sampler2D u_outline_pass;
    void main() {
        vec4 base = texture(u_base_image, vs_uv);
        vec4 bloom1 = texture(u_bloom_pass1, vs_uv);
        vec4 bloom2 = texture(u_bloom_pass2, vs_uv);
        vec4 outline = texture(u_outline_pass, vs_uv);
        vec3 color = base.rgb + bloom1.rgb + bloom2.rgb;
        color = outline.r * color * vec3(1.0, 0.8, 0.7) + 1.0 * color;
        fs_color = vec4(color, 1.0);
    }
    """

    renderer = Renderer(is_headless=False)

    photo_texture = renderer.load_texture("photo.jpeg")
    depth_texture = renderer.load_texture("depth.png")
    output_size = scale_size(photo_texture.size, 400)

    parallax_node = renderer.create_node(
        name="Parallax",
        fs_source=parallax_fs,
        output_size="u_base_texture",
        uniforms={
            "u_base_texture": photo_texture,
            "u_depth_map": depth_texture,
            "u_parallax_amount": 0.02,
            "u_focal_length": 1480.0,
            "u_time": lambda: renderer.render_time,
        },
    )

    bright_pass_node = renderer.create_node(
        name="Bright pass",
        fs_source=bright_pass_fs,
        output_size="u_source_texture",
        uniforms={
            "u_source_texture": parallax_node,
            "u_threshold": 0.5,
        },
    )

    downscale_node1 = renderer.create_node(
        name="Downscale 1",
        fs_source=downscale_fs,
        output_size=("u_input_texture", 0.5),
        uniforms={
            "u_input_texture": bright_pass_node,
        },
    )

    blur_node1 = renderer.create_node(
        name="Blur 1",
        fs_source=blur_fs,
        output_size=("u_blur_input", 1.0),
        uniforms={
            "u_blur_input": downscale_node1,
            "u_pixel_size": lambda: (
                1.0 / blur_node1.get_output_size()[0],
                1.0 / blur_node1.get_output_size()[1],
            ),
        },
    )

    downscale_node2 = renderer.create_node(
        name="Downscale 2",
        fs_source=downscale_fs,
        output_size="u_input_texture",
        uniforms={
            "u_input_texture": blur_node1,
        },
    )

    blur_node2 = renderer.create_node(
        name="Blur 2",
        fs_source=blur_fs,
        output_size="u_blur_input",
        uniforms={
            "u_blur_input": downscale_node2,
            "u_pixel_size": lambda: (
                1.0 / blur_node2.get_output_size()[0],
                1.0 / blur_node2.get_output_size()[1],
            ),
        },
    )

    outline_node = renderer.create_node(
        name="Outline",
        fs_source=outline_fs,
        output_size="u_outline_source",
        uniforms={
            "u_outline_source": parallax_node,
            "u_pixel_size": lambda: (
                1.0 / outline_node.get_output_size()[0],
                1.0 / outline_node.get_output_size()[1],
            ),
            "u_outline_thickness": 8.0,
        },
    )

    combine_node = renderer.create_node(
        name="Combine",
        fs_source=combine_fs,
        output_size="u_base_image",
        uniforms={
            "u_base_image": parallax_node,
            "u_bloom_pass1": blur_node1,
            "u_bloom_pass2": blur_node2,
            "u_outline_pass": outline_node,
        },
    )

    try:
        renderer.run_editor(60)
    finally:
        renderer.cleanup()
