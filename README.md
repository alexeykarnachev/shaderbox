# ShaderBox

**A desktop playground for GLSL shaders.** Write a fragment shader, get an instant control panel for
every uniform in it, and watch the render change as you drag.

![Night City - a raymarched city rendered live in ShaderBox](./docs/screenshots/hero_night_city.png)

*The scene above is one shader file, shipped as an example. Every part of it - camera, building
density, window warmth, traffic, fog - is a slider ShaderBox built by reading the code.*

[**Download for Linux or Windows on itch.io**](https://where-is-your-keyboard.itch.io/shaderbox)

## The idea

Writing a shader usually means hard-coding a constant, recompiling, squinting, changing it,
recompiling again. ShaderBox removes that loop:

```glsl
uniform float u_glow = 0.8;      // a drag appears
uniform vec3  u_tint_color;      // a color picker appears
uniform sampler2D u_image;       // an image slot appears
```

Declare a uniform, save, and the control is there - nothing to register, no UI code to write.
`Ctrl+S` recompiles in place, so the picture updates while you are still looking at it.

![The editor, the live render and the uniforms panel ShaderBox built from the shader](./docs/screenshots/editor_uniforms.png)

## What else is in the box

- **Examples to pull apart.** `Alt+E` opens a gallery of complete shaders - the city above, a fire,
  a flock, a 2D global-illumination pipeline. Open a copy and take it apart.
- **Multi-pass graphs.** A document can hold several passes, each its own shader and render target.
  Name a sampler `u_<pass>` and it reads that pass; `u_prev` reads its own previous frame, and a
  pass can run several iterations a frame. `Alt+G` shows the whole document as a node graph you can
  rewire by dragging.

![Radiance Cascades as a pass graph: paint, seed, jump flood, distance field, cascade, composite](./docs/screenshots/pass_graph.png)

- **A shader library.** Call `SB_fbm(...)` or `SB_sd_circle(...)` straight from your code, no
  `#include` - SDF shapes and operators, noise, glow, and a full text stack are already there.
- **Python on top of GLSL.** A document can carry a script that drives its uniforms every frame, for
  the things a stateless shader cannot do - a physics step, an integrator. Return a numpy array for a
  `sampler2D` and the pass samples it as a texture that frame.
- **Thousands of entities, one draw call.** Declare a per-entity field as `flat in` and a pass
  switches from one fragment shader to an instanced quad per entity - a script steers the swarm in
  numpy, the GPU draws all of it in a single call.
- **Your own native code.** When numpy is not fast enough, build a shared library in C, Odin or
  anything that exports C, and call it from the script with `native("name")`. It runs in its own
  process with shared buffers, so a crash is a script error rather than a dead app, and a rebuild is
  picked up while the document plays. The header and an example per language ship with the app.

![Entity Flock: 20000 entities simulated in numpy and drawn as instanced quads](./docs/screenshots/script_instancing.png)

- **An AI copilot** (optional, bring your own OpenRouter key). It edits your shader, the engine
  recompiles in-process, and the compile errors go straight back to the model - so it fixes its own
  GLSL instead of handing you something broken. Destructive actions wait for a confirm; any turn
  reverts in one click.
- **Export.** An image, a video, a Telegram sticker set, or straight to YouTube.
- **Keyboard-first.** Rebindable chords, a command palette, `F1` for in-app help.

## Install

Grab a build from [itch.io](https://where-is-your-keyboard.itch.io/shaderbox), unzip it and run
`run.bat` (Windows) or `./run.sh` (Linux). Nothing to install first: the launcher fetches
[uv](https://docs.astral.sh/uv/), a Python runtime and the dependencies on the first start (about
500 MB, once). It needs OpenGL 4.6, or Mesa's software renderer, which the Linux launcher configures.

From a clone, with uv:

```
uv sync --no-dev
uv run --no-dev python ./shaderbox/ui.py
```

## Built with

Python 3.12, [moderngl](https://github.com/moderngl/moderngl), [glfw](https://www.glfw.org/), and
[Dear ImGui](https://github.com/pthom/imgui_bundle) via imgui-bundle.

## License

GPL-3.0-or-later - use it, study it, share it, modify it; distributed derivatives stay under the same
license with source available. See [LICENSE](./LICENSE).
