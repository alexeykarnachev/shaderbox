"""shader-lab render helper: render one node dir headless on EGL — a still PNG at a
chosen t, OR an MP4 clip. Lets the agent SEE its shader (the copilot is render-blind),
and produce the MP4 deliverable for offscreen sessions (no app / no display).

    # still frame at t, for the agent to eyeball:
    uv run python .claude/skills/shader-lab/render_node.py image <node_dir> <out.png> [--t 0.0] [--size 512]

    # MP4 clip (offscreen mode — iPad/Pi can't play WebM, so MP4 is mandatory):
    uv run python .claude/skills/shader-lab/render_node.py video <node_dir> <out.mp4> [--seconds 10] [--fps 30] [--size 512]

Runs against the live shader lib (so SB_* resolve) on a standalone EGL context — no
App, no glfw window. Video goes through Node.render_media, so a node script ticks a
FRESH per-export instance (export-isolation), same as a real export.
"""

import argparse
import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "4.6")
os.environ.setdefault("MESA_GLSL_VERSION_OVERRIDE", "460")

import moderngl

from shaderbox.core import Node
from shaderbox.media import FileDetails, MediaDetails, ResolutionDetails, texture_to_pil
from shaderbox.paths import shader_lib_root
from shaderbox.scripting.outputs import normalize_output
from shaderbox.shader_lib import ShaderLibIndex, set_active
from shaderbox.uniform_coerce import coerce_uniform_value


@dataclass
class _Ctx:
    t: float
    dt: float
    frame: int


def _load(node_dir: str) -> Node:
    ctx: moderngl.Context = moderngl.create_standalone_context(backend="egl")  # type: ignore[arg-type]
    set_active(ShaderLibIndex.build(shader_lib_root()))
    node, _ = Node.load_from_dir(node_dir, gl=ctx)
    return node


def _load_behavior(node_dir: str) -> object | None:
    script_path = Path(node_dir) / "scripts" / "script.py"
    if not script_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("_node_script", str(script_path))
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Behavior()


def _apply_driven(node: Node, driven: dict[str, object]) -> None:
    # Coerce each script output into the program-ready shape exactly as the live engine does
    # (normalize the typed output, then chunk a vecN[M]/array per the live Uniform) — core.render
    # writes uniform_values straight to the program without coercing, so this must mirror it.
    if not node.program:
        node.compile()
    uniforms = {u.name: u for u in node.get_active_uniforms()}
    for name, val in driven.items():
        normalized = normalize_output(val)
        uniform = uniforms.get(name)
        if uniform is None:
            node.uniform_values[name] = normalized
            continue
        coerced = coerce_uniform_value(normalized, uniform)
        node.uniform_values[name] = normalized if coerced is None else coerced


def _apply_script_at(node: Node, node_dir: str, t: float) -> None:
    # For a STILL frame: step a fresh Behavior 0 -> t so its state (random walks etc.) matches
    # what the live engine would have at t, then feed the driven uniforms in.
    beh = _load_behavior(node_dir)
    if beh is None:
        return
    dt = 1.0 / 60.0
    driven: dict[str, object] = {}
    for i in range(max(1, int(t / dt))):
        driven = beh.update(_Ctx(t=i * dt, dt=dt, frame=i))
    _apply_driven(node, driven)


def render_image(node_dir: str, out_path: str, t: float, size: int) -> None:
    node = _load(node_dir)
    node.canvas.set_size((size, size))
    _apply_script_at(node, node_dir, t)
    node.render(u_time=t, canvas=node.canvas)
    texture_to_pil(node.canvas.texture).save(out_path)
    print(f"wrote {out_path} ({size}x{size}, t={t})")


def render_video(
    node_dir: str, out_path: str, seconds: float, fps: int, size: int
) -> None:
    if not out_path.endswith(".mp4"):
        raise SystemExit("offscreen deliverable must be .mp4 (WebM won't play on iPad)")
    node = _load(node_dir)
    node.canvas.set_size((size, size))

    # A bare Node has no ProjectSession, so nothing wires on_pre_render — without it render_media's
    # per-frame loop never ticks the script and the sim stays frozen at its __init__ state. Wire a
    # minimal hook: one fresh Behavior ticked per export frame (mirrors export-isolation: a clean
    # instance per render, ticked 0 -> duration).
    beh = _load_behavior(node_dir)
    if beh is not None:

        def _pre_render(t: float, dt: float, frame: int) -> None:
            _apply_driven(node, beh.update(_Ctx(t=t, dt=dt, frame=frame)))

        node.on_pre_render = _pre_render

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    details = MediaDetails(
        is_video=True,
        file_details=FileDetails(path=out_path),
        resolution_details=ResolutionDetails(width=size, height=size),
        fps=fps,
        duration=seconds,
    )
    node.render_media(details)
    print(f"wrote {out_path} ({size}x{size}, {seconds}s @ {fps}fps, mp4)")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)

    pi = sub.add_parser("image")
    pi.add_argument("node_dir")
    pi.add_argument("out_path")
    pi.add_argument("--t", type=float, default=0.0)
    pi.add_argument("--size", type=int, default=512)

    pv = sub.add_parser("video")
    pv.add_argument("node_dir")
    pv.add_argument("out_path")
    pv.add_argument("--seconds", type=float, default=10.0)
    pv.add_argument("--fps", type=int, default=30)
    pv.add_argument("--size", type=int, default=512)

    a = p.parse_args()
    if a.mode == "image":
        render_image(a.node_dir, a.out_path, a.t, a.size)
    else:
        render_video(a.node_dir, a.out_path, a.seconds, a.fps, a.size)


if __name__ == "__main__":
    main()
