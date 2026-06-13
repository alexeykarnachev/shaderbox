"""Feature-040 determinism verification (the required dogfood check, decision 4).

Drives the headless ProjectSession (the harness EGL path) on a project with a hand-placed scripted
uniform, renders the node at two distinct `t` values, and asserts the rendered pixels DIFFER (the
scripted uniform animated through the tick->render seam). Then it breaks the script and confirms the
frame still renders (frozen, non-crashing). NOT an LLM scenario — v1 scripts are hand-placed (the
copilot write-behavior tool is feature 043); this is a deterministic harness assertion.

Run: `uv run python scripts/dogfood/verify_script_engine.py` (exit 0 = pass).
"""

import json
import sys
from pathlib import Path

from PIL import Image

from scripts.dogfood.harness import DogfoodHarness

_SHADER = """#version 460 core
in vec2 vs_uv;
uniform float u_wave;
out vec4 fs_color;
void main() { fs_color = vec4(vec3(u_wave), 1.0); }
"""


def _seed_scripted_node(project_dir: Path, body: str) -> None:
    node = project_dir / "nodes" / "scripted"
    node.mkdir(parents=True, exist_ok=True)
    (node / "shader.frag.glsl").write_text(_SHADER, encoding="utf-8")
    (node / "node.json").write_text(
        json.dumps(
            {
                "canvas_size": [256, 256],
                "uniforms": {},
                "ui_state": {"ui_name": "Scripted Wave", "description": "040 check"},
            }
        ),
        encoding="utf-8",
    )
    scripts = node / "scripts"
    scripts.mkdir(exist_ok=True)
    (scripts / "u_wave.py").write_text(body, encoding="utf-8")


def _mean_luma(path: str) -> float:
    img = Image.open(path).convert("L")
    px = list(img.getdata())
    return sum(px) / len(px)


def main() -> int:
    h = DogfoodHarness.create(seed_templates=False)
    _seed_scripted_node(h.project_dir, "out.set(0.5 + 0.45 * sin(ctx.t))")
    h.session.load(h.project_dir)
    h.session.set_current_node_id("scripted")

    driven = h.session.get_script_driven_uniforms("scripted")
    assert driven == {"u_wave"}, f"expected u_wave script-driven, got {driven}"

    # Render at two t values where sin(t) differs: t=0 -> 0.5, t=pi/2 -> ~0.95.
    p_a = h.render_at(0.0, "scripted")
    p_b = h.render_at(1.5708, "scripted")
    luma_a, luma_b = _mean_luma(p_a), _mean_luma(p_b)
    print(f"  luma @t=0: {luma_a:.1f}  @t=pi/2: {luma_b:.1f}")
    assert abs(luma_a - luma_b) > 20, "scripted uniform did NOT animate between two t values"

    # Same t -> same value (ctx.t-pure determinism).
    p_a2 = h.render_at(0.0, "scripted")
    assert abs(_mean_luma(p_a) - _mean_luma(p_a2)) < 0.5, "ctx.t-pure render not deterministic"

    # Break the script -> the frame must still render (frozen, non-crashing) + record an error.
    _seed_scripted_node(h.project_dir, "out.set(1.0 / 0.0)")  # ZeroDivisionError at runtime
    h.session.reload_scripts()
    p_broken = h.render_at(3.0, "scripted")
    assert p_broken, "a broken script crashed the render instead of freezing"
    assert ("scripted", "u_wave") in h.session.script_engine.errors, "no ScriptError recorded"

    print("  PASS: animated across t, deterministic at fixed t, broken script froze + recorded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
