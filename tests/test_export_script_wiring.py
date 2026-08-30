"""The export script tick reaches the pass that actually draws.

`ScriptEngine.tick_export` is covered directly, and `render_media` is covered for ENTERING
`export_isolation` — but nothing observed the receiver the isolation factory hands the engine.
Pointing it at any other object leaves the export silently unscripted: the file is written, at the
right size, from whatever uniform values happened to be there. Both gates stayed green under that
mutation, so this drives a real `ProjectSession` export and reads the pixel back.
"""

from pathlib import Path
from typing import Any

from PIL import Image as PILImage

from shaderbox.media import MediaDetails
from shaderbox.paths import DOCUMENT_SCRIPT_BASENAME

_SRC = """#version 460 core
in vec2 vs_uv;
uniform float u_level;
out vec4 fs_color;
void main() { fs_color = vec4(u_level, 0.0, 0.0, 1.0); }
"""

# Frame-driven, so an export starting cold at frame 0 produces a value the live path never holds.
_SCRIPT = (
    "class Behavior(ScriptBehavior):\n"
    "    def __init__(self):\n"
    "        self.n = 0\n"
    "    def update(self, ctx):\n"
    "        self.n += 1\n"
    "        return {'u_level': min(1.0, self.n / 500.0)}\n"
)


def test_an_exported_frame_carries_the_scripted_value(app: Any, tmp_path: Path) -> None:
    document_id = app.current_document_id
    document = app.ui_documents[document_id].document
    document.render_pass.release_program(_SRC)
    document.render_pass.source.path.write_text(_SRC)
    document.render_pass.compile()
    assert document.render_pass.compile_unit.errors == []

    scripts_dir = app.session.paths.scripts_dir_for(document_id)
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / DOCUMENT_SCRIPT_BASENAME).write_text(_SCRIPT)
    app.session.reload_scripts()

    # Warm the live instance well past what one cold export tick can reach.
    for frame in range(400):
        app.session.tick([document_id], frame / 60, 1 / 60, frame)
    live_level = document.render_pass.uniform_values["u_level"]
    assert live_level > 0.5

    out = tmp_path / "exported.png"
    details = MediaDetails(is_video=False, duration=1.0)
    details.file_details.path = str(out)
    width, height = document.render_pass.canvas.texture.size
    details.resolution_details.width = width
    details.resolution_details.height = height
    document.render_media(details)

    red = PILImage.open(out).convert("RGBA").getpixel((0, 0))[0]
    # One cold tick => u_level == 1/500, i.e. ~0.5/255 of red. An unscripted export would show
    # whatever the uniform last held; a live-warmed one would be bright.
    assert red <= 2, f"export ignored the script or inherited live state (red={red})"
    assert red != round(live_level * 255)
