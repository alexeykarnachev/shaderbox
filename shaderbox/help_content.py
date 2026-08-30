"""In-app help content as DATA (feature 055).

Sections are built at call time so the two generated ones — engine uniforms and keyboard shortcuts —
read their facts from the code that owns them (`ENGINE_DRIVEN_UNIFORMS`, `COMMAND_SPECS`) instead of
a hand-typed copy that rots. Imports no imgui and no `App`, so the content is unit-testable without
a window; `core` comes along for the uniform set (its module-top GL imports cost collection time,
not a context).

Body prose uses the markdown-lite vocabulary `ui_primitives.markdown_text` understands (`**bold**`,
`` `code` ``); a `snippet` is GLSL the panel renders as a code block and can insert at the caret.
"""

from dataclasses import dataclass

from shaderbox.commands import (
    CATEGORY_ORDER,
    COMMAND_SPECS,
    chord_to_str,
)
from shaderbox.core import ENGINE_DRIVEN_UNIFORMS
from shaderbox.glyph_tables import TABLE_UNIFORMS


@dataclass(frozen=True)
class HelpSection:
    key: str
    title: str
    body: str
    snippet: str = ""
    # Whether `snippet` is GLSL the user can drop into a shader. False for a snippet that is
    # displayed-only (the shortcuts table is plain text — inserting it would never compile).
    insertable: bool = True


# The user-facing engine uniforms: GLSL type + what the engine writes into it each frame. Keys must
# cover ENGINE_DRIVEN_UNIFORMS minus the glyph tables (engine machinery, never hand-declared) —
# tests/test_help_content.py fails if a new builtin lands without a doc entry here.
ENGINE_UNIFORM_DOCS: dict[str, tuple[str, str]] = {
    "u_time": ("float", "seconds since launch — the animation clock"),
    "u_aspect": ("float", "canvas width / height"),
    "u_resolution": ("vec2", "canvas size in pixels"),
}


def user_facing_engine_uniforms() -> set[str]:
    return set(ENGINE_DRIVEN_UNIFORMS) - set(TABLE_UNIFORMS)


def _engine_uniform_section() -> HelpSection:
    names = sorted(user_facing_engine_uniforms())
    rows = "\n".join(
        f"uniform {ENGINE_UNIFORM_DOCS[n][0]} {n};".ljust(30)
        + f"// {ENGINE_UNIFORM_DOCS[n][1]}"
        for n in names
        if n in ENGINE_UNIFORM_DOCS
    )
    return HelpSection(
        key="engine_uniforms",
        title="Engine uniforms",
        body=(
            "Declare any of these and ShaderBox writes it every frame. They get no slider — the "
            "engine owns them. Declare only the ones you use; an unused uniform is compiled away.\n"
            "\n"
            "Everything else you declare becomes a control in the Node tab instead."
        ),
        snippet=rows,
    )


def _shortcuts_section() -> HelpSection:
    lines: list[str] = []
    for category in CATEGORY_ORDER:
        specs = [s for s in COMMAND_SPECS if s.category == category and s.default_chord]
        if not specs:
            continue
        lines.append(f"{category.value}")
        width = max(len(s.label) for s in specs)
        for spec in specs:
            lines.append(
                f"  {spec.label.ljust(width)}   {chord_to_str(spec.default_chord)}"
            )
        lines.append("")
    return HelpSection(
        key="shortcuts",
        title="Keyboard shortcuts",
        body=(
            "Defaults — every one is rebindable in **Settings**. The floating cheatsheet "
            "(`Alt+/`) shows only the chords valid right now."
        ),
        snippet="\n".join(lines).rstrip(),
        insertable=False,
    )


def help_sections() -> list[HelpSection]:
    return [
        HelpSection(
            key="shader_skeleton",
            title="A ShaderBox shader",
            body=(
                "Every node is one **fragment shader**. ShaderBox draws a full-screen quad and runs "
                "your `main()` once per pixel.\n"
                "\n"
                "Three things are fixed: the `#version` line (required — nothing is injected for "
                "you), the `vs_uv` input, and a single `vec4` output. `vs_uv` runs 0..1 across the "
                "canvas; the output name is up to you (`fs_color` is just what the examples call "
                "it) since a lone fragment output always binds to location 0.\n"
                "\n"
                "Save with `Ctrl+S` and the render updates instantly."
            ),
            snippet=(
                "#version 460 core\n"
                "\n"
                "in vec2 vs_uv;          // 0..1 across the canvas\n"
                "out vec4 fs_color;      // one vec4 out; the name is yours\n"
                "\n"
                "void main() {\n"
                "    fs_color = vec4(vs_uv, 0.0, 1.0);\n"
                "}"
            ),
        ),
        _engine_uniform_section(),
        HelpSection(
            key="your_uniforms",
            title="Your uniforms become controls",
            body=(
                "Declare a uniform the engine does not own and it shows up in the **Node** tab as a "
                "control, picked from the type: a drag for numbers, a text field for a glyph array, "
                "an image slot for a `sampler2D`. Name a `vec3`/`vec4` so it ends in `color` and it "
                "gets a colour picker — otherwise switch the control by hand on its row.\n"
                "\n"
                "Give it a value in the declaration and that becomes its default. Tuned values are "
                "saved with the node, so a shader reopens exactly as you left it.\n"
                "\n"
                "For a value that needs memory between frames (a physics step, an integrator), a "
                "node can carry a Python script that drives its uniforms — see the Script entry "
                "point in the Node tab."
            ),
            snippet=(
                "uniform float u_radius = 0.35;                     // drag\n"
                "uniform vec3  u_tint_color = vec3(1.0, 0.6, 0.2);  // colour picker\n"
                "uniform sampler2D u_image;                         // image slot"
            ),
        ),
        HelpSection(
            key="render_steps",
            title="Several draws in one node",
            body=(
                "Some effects need more than one draw: a blur chain, bloom, radiance cascades, "
                "trails, a simulation. Declare a **step** by riding a comment on the sampler that "
                "reads it, and write its body as `void step_<name>(out vec4 o)`.\n"
                "\n"
                "ShaderBox works out the order from who reads whom, so you never list it. A step "
                "that reads **itself** gets its own previous frame — that is feedback, and there "
                "is no second buffer to manage.\n"
                "\n"
                "Options after `step`, all optional: `scale: N` or `size: WxH` for the target "
                "size; `f1`/`f2`/`f4` for precision (**`f2` is the default** — 8-bit saturates "
                "immediately when you accumulate); `linear`/`nearest`; `clamp`/`repeat`.\n"
                "\n"
                "A typo in the rider is an error, not a silent miss: a step whose body is missing, "
                "an unknown option, or a marker that merely looks like `step` all stop the compile "
                "and point at the line."
            ),
            snippet=(
                "#version 460 core\n"
                "\n"
                "in vec2 vs_uv;\n"
                "out vec4 fs_color;\n"
                "\n"
                "uniform sampler2D u_bright;  // step, scale: 0.5, f2\n"
                "uniform sampler2D u_trail;   // step, f2\n"
                "uniform float u_threshold;\n"
                "uniform float u_fade;\n"
                "\n"
                "void step_bright(out vec4 o) {\n"
                "    vec3 c = vec3(vs_uv, 0.5);\n"
                "    o = vec4(max(c - u_threshold, 0.0), 1.0);\n"
                "}\n"
                "\n"
                "void step_trail(out vec4 o) {\n"
                "    // reads ITSELF: last frame, handed over automatically\n"
                "    vec4 prev = texture(u_trail, vs_uv) * u_fade;\n"
                "    o = max(texture(u_bright, vs_uv), prev);\n"
                "}\n"
                "\n"
                "void main() {\n"
                "    fs_color = vec4(texture(u_trail, vs_uv).rgb, 1.0);\n"
                "}"
            ),
        ),
        HelpSection(
            key="shader_library",
            title="The SB_ library",
            body=(
                "Call any `SB_*` helper directly — **no `#include`**. On compile ShaderBox scans "
                "your source for `SB_` names, pulls in the ones you used plus everything they "
                "depend on, and splices them above your code.\n"
                "\n"
                "Press `Ctrl+P` to browse the library with a live preview of each function's source, "
                "and insert a name at the caret. The library is yours to edit: add your own helpers, "
                "and restore the shipped set any time from **Settings**.\n"
                "\n"
                "The layers are `SB_sd_*` (signed distance — negative inside), `SB_op_*` "
                "(combine/transform distances), and renderers like `SB_fill` / `SB_glow` that turn "
                "a distance into pixels."
            ),
            snippet=(
                "vec2  p    = SB_center_uv(vs_uv, u_aspect);   // centered, aspect-corrected\n"
                "float d    = SB_sd_circle(p, 0.3);            // distance: negative inside\n"
                "float mask = SB_fill_aa(d);                   // distance -> antialiased mask\n"
                "fs_color   = vec4(vec3(mask), 1.0);"
            ),
        ),
        _shortcuts_section(),
    ]
