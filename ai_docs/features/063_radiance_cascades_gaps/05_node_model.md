# Audit: node/project model, scripting, library, performance (current state)

Sources: code reads of `project_session.py`, `ui_models.py`, `paths.py`,
`copilot/checkpoint.py`, `scripting/*`, `shader_lib/*`, `app.py`, `ui.py`, `watch.py`.

## The node model

A node is **one directory** `<project>/nodes/<uuid4>/`; the dir name IS the id (no id field
inside `node.json`). At most five things live in it:

| Path | Contents |
|---|---|
| `node.json` | metadata (required) |
| `shader.frag.glsl` | the fragment source (required) |
| `media/<uniform>.<ext>` | image/video bound to a sampler |
| `textures/<uniform>.bin` | raw texture bytes — **no instance exists on disk anywhere** |
| `scripts/script.py` | the node's behaviour script (lazily created) |

`node.json` has exactly three top-level keys: `canvas_size`, `uniforms`, `ui_state`.

**There is no graph.** "Node" is a naming legacy. Searches for
`graph|edge|link|connection|upstream|downstream|dag|topolog|parent_node|child_node|node_input|node_output`
return only false positives — the GLSL *include* call graph inside `shader_lib/`, resolution
math (`longest_edge`), Telegram/YouTube account linking, imgui layout. `render_to_texture|node_texture|pipeline`
-> zero node-relevant hits.

The only way one node's output reaches another today is **through a file**: render to
mp4/png, then bind that file to a sampler. The binding is a `file_path` string carrying no
node id and creating no tracked dependency.

## Every node is already live in GL

This is load-bearing for cost estimates. `ProjectSession.load` loads **every** node dir, and
each load ends in `node.render()` — compiling a program and allocating a canvas texture + FBO.
It also loads **all 5 shipped examples** into GL simultaneously. `is_render_all_nodes`
(default True) gates *rendering*, never *loading*; even with it off, `frame_idx == 0` forces
one render of every node so every program compiles.

Per-node resources: canvas texture (RGBA8 at full `canvas_size`), FBO, program, VBO, VAO,
plus every bound media texture and **open video capture**.

At the `1280x960` both dev nodes use, that is ~4.9 MB per node just for the canvas. So a
6-level cascade chain adds no new *kind* of resource — only more of what the app already
holds routinely.

## canvas_size

**There is no in-memory `canvas_size` field.** The canvas texture IS the store; every reader
goes through `node.canvas.texture.size`, and `UINode.save` serializes
`list(self.node.canvas.texture.size)`. `Canvas.set_size` releases and reallocates
texture + FBO (no recompile), early-returning when unchanged.

Two writers only: the Resolution combo in `tabs/node.py`, and `CopilotBackend.set_canvas_size`
(clamped 16..4096). The UI path does not save; the copilot path saves immediately.

**The preview is not at canvas resolution.** The current node renders twice per frame — once
into the shared `app.preview_canvas` scaled to `SIZE.PREVIEW_W`, once into its own canvas for
the grid thumbnail. Since `u_resolution`/`u_aspect` derive from the passed canvas, a
resolution-dependent shader already shows *different output* in the preview than in the
thumbnail or export.

## Scripting: the ceiling is measured, and it is low

Entry point is a class (`ScriptBehavior` subclass with `update(self, ctx) -> dict`), mapping
uniform name -> value. State persists on `self` across frames; a file edit rebuilds the
instance, so **state resets on every edit**. `export_isolation` swaps in a fresh behaviour per
export from a cached source string.

`Vec2/3/4` and `Array` are **pure Python** (`outputs.py` imports only `math`); `_Vec`
subclasses `tuple` and every op reallocates. Measured on this box, CPython 3.12:

```
65536 Vec2 add+mul  : 110.1 ms   (~1.7 us per (a+b)*k pair)
Array(flat 65536)   :  17.7 ms
coercion scan+convert of 65536 : 10.6 ms
```

A 60 fps frame budget is 16.7 ms total. **A per-cell sim over 256x256 (65536 cells) is not
feasible** — the Vec math alone is ~6.6x the whole frame budget before any logic. The
realistic ceiling is hundreds to low thousands of elements. The engine is designed for that:
"A value that is a pure function of `ctx.t` usually belongs in the shader instead."

Uniform arrays work and have **no ShaderBox-side cap** (`array_length` is read straight off
the live `moderngl.Uniform`); the limit is the driver's uniform-component budget and the
Python cost above. Numeric arrays are **exact-length, no padding** ("padding a data array is
silent corruption"); only text arrays pad.

**A script cannot write a texture, touch GL, read the rendered output, trigger a render, or
influence render order.** `is_scriptable` structurally excludes samplers and uniform blocks.
Caveat: this is a *boundary, not a sandbox* — `__builtins__` is real, so a determined script
can `import moderngl` itself. That posture is deliberate and recorded ("No sandbox (a personal
IDE; locked posture)").

## The shader library

**There is no `#include`.** Resolution is by scanning for `SB_*` identifiers, then splicing
transitively-closed function bodies as a preamble, with `#line` directives remapping driver
errors back to (file, line). Cycles are caught (`library cycle involving 'X'`); duplicate
splices are structurally impossible (a `visited` set). An unknown `SB_*` call gets a
did-you-mean via `difflib`.

*(Stale-doc defect found: `paths.py::shader_lib_root`'s comment still claims
`every node's #include "name" resolves against this dir`. No such mechanism exists.)*

**26 public `SB_*` functions across 10 files.** Coverage relevant to a GI workload:

- **Exists:** 2D SDFs (circle, box, segment), six `SB_op_*` combinators, three renderers
  (`SB_fill`, `SB_fill_aa` via `fwidth`, `SB_glow`), hashes (`SB_hash21/31/22`), value noise,
  fBm (8-octave cap), `SB_domain_warp`, `SB_center_uv`, `SB_rotate`, and the segment-glyph
  text stack.
- **Absent, all verified by search:** 3D SDFs (zero — the only `vec3` in the library is a hash
  parameter), **raymarching of any kind** (no march loop, no `map()` convention, no normal
  estimation, no AO, no soft shadows), noise families beyond value noise (no simplex/perlin/
  worley/voronoi/curl), **blur/convolution** (no separable blur, no kernel helpers),
  **colour-space conversion** (no rgb2hsv/oklab/srgb/gamma — none at all), **tonemapping**
  (no aces/reinhard), and **texture-sampling helpers** — `grep -ril texture` over the library
  returns *nothing*; no lib function takes a `sampler2D`.

The one shipped example that raymarches ("Night City") defines its own private `sdBox(vec3)`,
`hash(vec3)`, and `map` inline — none of it is in the library.

**Shader contract:** `#version 460 core` (user-written, not enforced — no GL version hints are
requested anywhere; the context is the driver default). `in vec2 vs_uv` in [0,1], y-up; one
`vec4` out, name free. **Nothing is injected** — engine-driven uniforms must still be declared
by hand or the compile fails. **MRT is not supported**: `color_attachments=[self.texture]` is
the only attachment site in the repo, no `layout(location=N)` anywhere, no depth/stencil.

## Frame loop and per-frame cost

`run()` is: `update_and_draw()`, then `time.sleep(max(0, 1/target_fps - elapsed))`. **No vsync
control anywhere** (`swap_interval` -> zero hits). Target FPS is a setting (default 60, 30..240).
An EMA FPS counter exists as a clickable chip on the preview (`fps_overlay`); there is no
frame-time graph, no GPU timer queries, no profiler.

Per-frame disk I/O, all polling (no filesystem-notify):

- **Shader-lib sweep** — a recursive `**/*.glsl` glob + N `lstat`s over the lib root, every
  frame, unconditionally. A change invalidates *every* lib-using node, forcing recompiles.
- **Node-dir sync** — `iterdir()` + ~4 stat-class syscalls per node dir per frame. (Its own
  comment says "one glob + a stat per dir"; it is `is_dir` + two `is_file` + one `lstat`.)
  Skipped while a copilot turn is in flight.
- **Per-node shader mtime** — `exists()` + `lstat()` for the root *and every resolved lib
  source*. A node pulling 5 lib files costs 10 syscalls/frame.
- **Per-node script poll** — note the ordering: `read_text()` runs **before** the mtime
  comparison, so a scripted node re-reads its whole `script.py` from disk every frame.

**Live `u_time` is `time.monotonic()`.** Both live render sites call `render()` bare, so the
default fires. This contradicts `core.py`'s own comment ("the caller passes an explicit u_time
on every real render path (the live loop, ...)"). It is therefore a *different clock* from the
`glfw.get_time()` used for script `dt` in the same frame. Export is deterministic
(`i / details.fps`); the probe defaults to `t=0.0`.

**No pause, no scrub, no timeline** (`pause|scrub|timeline|time_offset|freeze_time` -> only a
video-seek comment and copilot turn-stop prose). The nearest thing is per-uniform script
freezing, which freezes *script-driven values*, not `u_time`.

## Checkpointing — the codebase's own definition of node state

`CopilotBackend._capture_node` captures exactly `UINode.save()` output + `scripts/script.py`.
So the repo's answer to "what is the full mutable state of a node" is: `node.json` +
`shader.frag.glsl` + `media/` + `textures/` + the script, serialized from the **live** node
(not the possibly-stale on-disk dir, because `set_uniform` writes only in-memory values).
Plus `pre_switch_node_id` — **which node was current is checkpointed state**.

Any new per-node state (extra passes, extra buffers) must be reachable from `UINode.save` or
it silently escapes both persistence and copilot revert.
