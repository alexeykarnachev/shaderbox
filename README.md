# ShaderBox

A modern OpenGL-based rendering framework for creating GPU shaders and effects.

## Development

```bash
# Install with uv
uv install -e .

# Install dev dependencies and pre-commit hooks
uv install --dev
pre-commit install
```

## Usage Example

```python
from shaderbox.renderer import Renderer

# Initialize renderer
renderer = Renderer()

# Load textures
base_texture = renderer.load_texture("image.jpg")

# Create a shader node
shader_node = renderer.create_node(
    name="Effect",
    fs_source="""
    #version 460
    in vec2 vs_uv;
    out vec4 fs_color;
    uniform sampler2D u_texture;
    uniform float u_time;

    void main() {
        vec2 uv = vs_uv;
        vec4 color = texture(u_texture, uv);
        float vignette = 1.0 - length(uv - 0.5) * 1.5;
        color.rgb *= vignette;
        fs_color = color;
    }
    """,
    output_size=(800, 600),
    uniforms={
        "u_texture": base_texture,
        "u_time": lambda: renderer.render_time,
    }
)

# Render to screen
renderer.render_to_screen(shader_node, 60)

# Or save as image/animation
renderer.render_image(shader_node, "output.png")
renderer.render_gif(shader_node, duration=2.0, fps=30, file_path="output.gif")
```
