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
from shaderbox.pass_graph import PassEntry, PassGraph
from shaderbox.paths import DOCUMENT_SCRIPT_BASENAME
from shaderbox.scripting import EXPORT_MOUSE

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


_SEED_SRC = """#version 460 core
in vec2 vs_uv;
uniform float u_level;
out vec4 fs_color;
void main() { fs_color = vec4(u_level, 0.0, 0.0, 1.0); }
"""
_OUT_SRC = """#version 460 core
in vec2 vs_uv;
uniform sampler2D u_seed;
out vec4 fs_color;
void main() { fs_color = vec4(texture(u_seed, vs_uv).r, 0.0, 0.0, 1.0); }
"""
# The value goes to the NON-output pass only, and the output pass merely samples it — so the
# exported pixel is non-black ONLY if the export routed to `seed`.
_PASS_BLOCK_SCRIPT = (
    "class Behavior(ScriptBehavior):\n"
    "    def update(self, ctx):\n"
    "        return {'seed': {'u_level': 1.0}}\n"
)


def test_an_export_drives_a_non_output_pass(app: Any, tmp_path: Path) -> None:
    # A pass block addressing a NON-output pass must drive it during an export, not only live: the
    # export ticks a fresh instance through the same routing. Falsifier: route an export to the
    # output pass alone — the exported pixel stays black.
    document_id = app.current_document_id
    document = app.ui_documents[document_id].document
    app.session.add_pass(document_id, "seed")
    seed = document.passes["seed"]
    seed.release_program(_SEED_SRC)
    seed.source.path.write_text(_SEED_SRC)
    seed.compile()
    out_pass = document.passes[document.graph.output_pass]
    out_pass.release_program(_OUT_SRC)
    out_pass.source.path.write_text(_OUT_SRC)
    out_pass.compile()
    assert seed.compile_unit.errors == [] and out_pass.compile_unit.errors == []
    output = document.graph.output_pass
    assert output is not None
    document.graph = PassGraph(
        output=output,
        passes={
            "seed": PassEntry(),
            output: PassEntry(inputs={"u_seed": "seed"}),
        },
    )

    scripts_dir = app.session.paths.scripts_dir_for(document_id)
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / DOCUMENT_SCRIPT_BASENAME).write_text(_PASS_BLOCK_SCRIPT)
    app.session.reload_scripts()

    out = tmp_path / "two_pass.png"
    details = MediaDetails(is_video=False, duration=1.0)
    details.file_details.path = str(out)
    width, height = out_pass.canvas.texture.size
    details.resolution_details.width = width
    details.resolution_details.height = height
    document.render_media(details)

    red = PILImage.open(out).convert("RGBA").getpixel((0, 0))[0]
    assert red > 200, f"the export did not drive the non-output pass (red={red})"


def test_export_mouse_is_down_false_and_prev_equals_current(
    app: Any, tmp_path: Path
) -> None:
    # An export is DETERMINISTIC: a script gated on the button paints nothing, and one reading
    # prev-to-current sees a zero-length capsule rather than a jump from the origin. Falsifier:
    # default down=True, or wire the live cursor into the export context — both halves go red.
    assert EXPORT_MOUSE.down is False
    assert (EXPORT_MOUSE.prev_x, EXPORT_MOUSE.prev_y) == (
        EXPORT_MOUSE.x,
        EXPORT_MOUSE.y,
    )

    document_id = app.current_document_id
    document = app.ui_documents[document_id].document
    document.render_pass.release_program(_SRC)
    document.render_pass.source.path.write_text(_SRC)
    document.render_pass.compile()

    scripts_dir = app.session.paths.scripts_dir_for(document_id)
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / DOCUMENT_SCRIPT_BASENAME).write_text(
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, ctx):\n"
        "        return {'u_level': 1.0} if ctx.mouse.down else {}\n"
    )
    app.session.reload_scripts()

    out = tmp_path / "gated.png"
    details = MediaDetails(is_video=False, duration=1.0)
    details.file_details.path = str(out)
    width, height = document.render_pass.canvas.texture.size
    details.resolution_details.width = width
    details.resolution_details.height = height
    document.render_media(details)

    red = PILImage.open(out).convert("RGBA").getpixel((0, 0))[0]
    assert red <= 2, f"a down-gated script painted in an export (red={red})"
