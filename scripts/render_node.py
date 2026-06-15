"""Throwaway: render one node dir to a PNG at a chosen t, headless on EGL.

    uv run python scripts/render_node.py <node_dir> <out.png> [t] [size]

Lets me (the agent) author a shader by hand, render, and LOOK at the result — the
loop the copilot itself can't do (it's render-blind). Not a fixture; delete when done.
"""

import os
import sys

os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "4.6")
os.environ.setdefault("MESA_GLSL_VERSION_OVERRIDE", "460")

import moderngl

from shaderbox.core import Node
from shaderbox.media import texture_to_pil
from shaderbox.paths import shader_lib_root
from shaderbox.shader_lib import ShaderLibIndex, set_active


def main() -> None:
    node_dir: str = sys.argv[1]
    out_path: str = sys.argv[2]
    t: float = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    size: int = int(sys.argv[4]) if len(sys.argv) > 4 else 512

    ctx: moderngl.Context = moderngl.create_standalone_context(backend="egl")  # type: ignore[arg-type]
    set_active(ShaderLibIndex.build(shader_lib_root()))

    node, _ = Node.load_from_dir(node_dir, gl=ctx)
    node.canvas.set_size((size, size))

    # If the node has a script, step a fresh Behavior from 0 -> t so its random
    # walks (wind/flicker) reach the same state the live engine would, then feed
    # the driven uniforms into the render. Lets me SEE the scripted behaviour.
    script_path = os.path.join(node_dir, "scripts", "script.py")
    if os.path.exists(script_path):
        import importlib.util
        from dataclasses import dataclass

        spec = importlib.util.spec_from_file_location("_node_script", script_path)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]

        @dataclass
        class _Ctx:
            t: float
            dt: float
            frame: int

        beh = mod.Behavior()
        dt = 1.0 / 60.0
        frames = max(1, int(t / dt))
        driven: dict = {}
        for i in range(frames):
            driven = beh.update(_Ctx(t=i * dt, dt=dt, frame=i))
        for name, val in driven.items():
            node.uniform_values[name] = val

    node.render(u_time=t, canvas=node.canvas)
    texture_to_pil(node.canvas.texture).save(out_path)
    print(f"wrote {out_path} ({size}x{size}, t={t})")


if __name__ == "__main__":
    main()
