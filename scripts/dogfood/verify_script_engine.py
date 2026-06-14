"""Feature-041 script-engine verification (the required dogfood check).

Drives the headless ProjectSession (the harness EGL path) on a project with a hand-placed class-form
scripted uniform, renders the node at two distinct `t` values, and asserts the rendered pixels DIFFER
(the scripted behavior animated through the tick->render seam). Then it (a) confirms ctx.t-pure
determinism, (b) proves EXPORT-STATE ISOLATION — a live-warmed integrator's export frame matches a
cold-start render via the `exporting` seam, NOT the warmed live value — and (c) breaks the script and
confirms the frame still renders (frozen, non-crashing). NOT an LLM scenario — scripts are hand-placed
(the copilot write-behavior tool is feature 043); this is a deterministic harness assertion.

Run: `uv run python scripts/dogfood/verify_script_engine.py` (exit 0 = pass).
"""

import contextlib
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
    # One script per node (048): the node-brain `script.py` drives u_wave via a dict return. Binds by
    # existence (no activate step).
    (scripts / "script.py").write_text(body, encoding="utf-8")


def _mean_luma(path: str) -> float:
    img = Image.open(path).convert("L")
    px = list(img.getdata())
    return sum(px) / len(px)


# A ctx.t-pure brain: animates with t, identical at the same t (the determinism guarantee).
_PURE = (
    "import math\n"
    "from shaderbox.scripting import ScriptBehavior, Ctx\n\n"
    "class Behavior(ScriptBehavior):\n"
    "    def update(self, ctx: Ctx) -> dict:\n"
    "        return {'u_wave': 0.5 + 0.45 * math.sin(ctx.t)}\n"
)
# A STATEFUL integrator: only possible with per-instance state — used for export isolation.
_INTEGRATOR = (
    "from shaderbox.scripting import ScriptBehavior, Ctx\n\n"
    "class Behavior(ScriptBehavior):\n"
    "    def __init__(self) -> None:\n"
    "        self.v = 0.0\n"
    "    def update(self, ctx: Ctx) -> dict:\n"
    "        self.v += ctx.dt\n"
    "        return {'u_wave': self.v % 1.0}\n"
)
# A broken body: a runtime error every tick -> freeze last-good, never crash.
_BROKEN = (
    "from shaderbox.scripting import ScriptBehavior, Ctx\n\n"
    "class Behavior(ScriptBehavior):\n"
    "    def update(self, ctx: Ctx) -> dict:\n"
    "        return {'u_wave': 1.0 / 0.0}\n"
)


def main() -> int:
    h = DogfoodHarness.create(seed_templates=False)
    _seed_scripted_node(h.project_dir, _PURE)
    h.session.load(h.project_dir)
    h.session.set_current_node_id("scripted")
    # One script per node (048): the brain binds by existence — no activate step. The driven set is
    # the brain's LAST-TICK keys (dynamic), so tick once before reading it (the live frame order:
    # tick -> read; the engine knows nothing it drove until the first tick).
    h.session.reload_scripts()
    h.session.tick(["scripted"], t=0.0, dt=1 / 60, frame=0)

    driven = h.session.get_script_driven_uniforms("scripted")
    assert driven == {"u_wave"}, f"expected u_wave script-driven, got {driven}"

    # Render at two t values where sin(t) differs: t=0 -> 0.5, t=pi/2 -> ~0.95.
    p_a = h.render_at(0.0, "scripted")
    p_b = h.render_at(1.5708, "scripted")
    luma_a, luma_b = _mean_luma(p_a), _mean_luma(p_b)
    print(f"  luma @t=0: {luma_a:.1f}  @t=pi/2: {luma_b:.1f}")
    assert (
        abs(luma_a - luma_b) > 20
    ), "scripted uniform did NOT animate between two t values"

    # Same t -> same value (ctx.t-pure determinism).
    p_a2 = h.render_at(0.0, "scripted")
    assert (
        abs(_mean_luma(p_a) - _mean_luma(p_a2)) < 0.5
    ), "ctx.t-pure render not deterministic"

    # Export-state isolation: warm a stateful integrator on the LIVE instance, then export — the
    # export frame must match a cold-start export, NOT the warmed live value (the fresh-per-export
    # instance seam, feature 041 decision 11).
    _seed_scripted_node(h.project_dir, _INTEGRATOR)
    h.session.reload_scripts()
    p_cold = h.export_at(0.0, "scripted")
    luma_cold = _mean_luma(p_cold)
    for i in range(120):  # warm the LIVE instance well past the ramp's wrap
        h.session.tick(["scripted"], i / 60.0, 1.0 / 60.0, i)
    live_wave = h.session.ui_nodes["scripted"].node.uniform_values["u_wave"]
    p_export = h.export_at(0.0, "scripted")
    luma_export = _mean_luma(p_export)
    print(
        f"  export isolation: cold {luma_cold:.1f}  live-warmed value {live_wave:.3f}  "
        f"export {luma_export:.1f}"
    )
    assert (
        abs(luma_cold - luma_export) < 0.5
    ), "export inherited live state (no fresh-per-export instance)"

    # Post-load hook wiring: a node's export_isolation must be wired (not the bare nullcontext) after
    # reload_scripts — this is what covers a node inserted AFTER load (copilot create / revert).
    node = h.session.ui_nodes["scripted"].node
    assert (
        node.export_isolation is not contextlib.nullcontext
    ), "node export_isolation not wired (post-load insertions would export frozen)"

    # drop_node frees engine state: delete + assert the binding + its registry entry are gone.
    h.session.script_engine.drop_node("scripted")
    assert h.session.script_engine.script_driven_uniforms("scripted") == set()
    # re-resolve so the rest of the check (broken-script) has a live binding again.
    h.session.reload_scripts()

    # Break the script -> the frame must still render (frozen, non-crashing) + record an error. The
    # raw throw fires inside update() before the dict is built, so it is a behavior-level error under
    # the sentinel key (node, "script.py") — not a per-key (node, "u_wave") shape error (048: one
    # object, one coherent failure).
    _seed_scripted_node(h.project_dir, _BROKEN)
    h.session.reload_scripts()
    p_broken = h.render_at(3.0, "scripted")
    assert p_broken, "a broken script crashed the render instead of freezing"
    assert (
        "scripted",
        "script.py",
    ) in h.session.script_engine.errors, "no ScriptError recorded"

    print(
        "  PASS: animated across t, deterministic at fixed t, export-isolated, broken script froze"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
