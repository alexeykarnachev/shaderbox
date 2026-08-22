# Audit: the render pipeline (current state)

Source: code read of `core.py`, `project_session.py`, `ui.py`, `app.py`, `render_preset.py`,
`media.py`, plus `moderngl` 5.12.0's installed source for defaults.

**One-line verdict: ShaderBox is a single-pass, single-draw, 8-bit, stateless-per-frame
renderer.** One node = one RGBA8 canvas = one fullscreen-quad draw preceded by a clear.

## The draw path

The entire draw is the last three statements of `Node.render`:

```python
canvas.fbo.use()
self._gl.clear()
self.vao.render()
```

`vao.render()` appears **exactly once in the whole repo** — one `glDrawArrays` of 6 vertices
(two triangles over NDC `[-1,1]^2`, from `FULLSCREEN_QUAD_VERTICES`). Everything preceding it
is CPU-side uniform marshalling.

`clear()` is moderngl's default — transparent black, full viewport, **unconditional, every
frame**. It destroys the previous contents before the fragment shader runs.

Absent: depth/stencil attachment (`Canvas._init` passes `color_attachments=[self.texture]`
only), blend-state config, scissor, viewport override, instancing, multi-draw.

## Two facts that make multipass cheaper than it looks

**1. `Node.render` already takes a caller-supplied canvas.** The signature is
`render(self, u_time=None, canvas=None)` with `canvas = canvas or self.canvas`. Rendering a
node into an arbitrary target is a supported, *exercised* operation — not new surface.

**2. The current node already renders twice per frame.** In `ui.update_and_draw`, step 4
renders it into `app.preview_canvas` (an App-owned `Canvas` sized to `SIZE.PREVIEW_W`), and
step 5 renders it again into its own `node.canvas`. Two independent full draws with
independent clears. So "more than one draw per frame" is already the status quo; what is
missing is *sequencing between draws with the output of one feeding the next*.

## Canvas / texture model

```python
self.texture = self._gl.texture(size or DEFAULT_CANVAS_SIZE, 4)
self.fbo = self._gl.framebuffer(color_attachments=[self.texture])
```

`dtype` is not passed, so moderngl's default `"f1"` applies -> **RGBA8, 8-bit unsigned
normalized**. Corroborated by the readback path (`np.frombuffer(..., dtype=np.uint8)`) and
`texture_to_pil`'s `PILImage.frombytes("RGBA", ...)`.

**No float/f2/f4/HDR texture path exists anywhere.** All four `.texture(` construction sites
in the repo (`Canvas._init`, `Image.texture`, `Video._upload_frame`,
`Node.load_from_dir`'s textures branch) are 8-bit. No `internal_format`, no `renderbuffer`, no
`samples` argument anywhere in the tree. The only `f4` present is a *numpy* dtype for the VBO.

`DEFAULT_CANVAS_SIZE` is `(64, 64)`; real sizes come from `node.json`. `Canvas.set_size`
releases and reallocates (early-returns when unchanged).

## No graph between nodes

`ProjectSession.ui_nodes` is a flat `dict[str, UINode]` sorted by `st_ctime` — a display
order, nothing more. Each `Node` owns exactly one `Canvas`. **Despite the name, a "Node" is a
standalone shader document, not a graph node.**

Searches establishing the negative:
- `\.canvas\.texture` filtered of `.size`/`.glo`/`texture_to_pil`/`.read()`/`set_size` ->
  **zero hits**. Every consumer reads dimensions, the GL handle for imgui display, or pixels
  for CPU readback. **Nothing binds a canvas texture as a uniform value.**
- Every writer into a sampler slot (`uniform_values[...] =`) sources from disk, the default
  image, or a file dialog. None sources from another node.
- `multi.?pass|dag|topolog|dependenc|eval.?order|upstream|downstream|node_graph` -> only
  false positives (the *include* resolver's text-flattening order; `duplicate_node`'s local
  variable).

## Sampler binding — closed and enforced

Exactly two accepted types at the binding site: `MediaWithTexture` (concrete subclasses
`Image`, `Video` — no others) and a raw `moderngl.Texture`. Anything else raises. Units are
assigned sequentially from 0 in `get_active_uniforms()` order.

The raw-`Texture` branch is a **self-consistent but unreachable-in-practice round-trip**: the
only producer is `Node.load_from_dir`'s `textures/*.bin` branch, whose files are only written
by the symmetric `UINode.save` branch, which is only reachable if a raw `Texture` was already
in `uniform_values` — and no live path puts one there. No `.bin` files exist under
`projects/`. The path is plumbed; it has no producer.

Note `MediaWithTexture` gets the temporal hook `value.update(render_time)` (where `Video`
seeks/decodes to the shader clock). A raw `Texture` gets no update.

## Per-frame ordering (live loop)

1. `copilot.drain_bridge()` — may run marshalled GL ops.
2. `session.sync_nodes_from_disk()` + per-node mtime `reload_node_if_changed`.
3. `session.tick(...)` — the script engine writes into `uniform_values` **before** any render.
4. Preview render of the current node into `app.preview_canvas`.
5. Node render block: by default **only the current node** renders into its own canvas (plus
   frame 0, which warms every node so grid thumbnails aren't blank); `is_render_all_nodes`
   renders all; the Examples popup renders example nodes instead.
6. imgui frame, screen clear, `imgui_renderer.render`, `glfw.swap_buffers`.
7. After the swap: `gl.finish()` then deferred encodes fire.

## No feedback of any kind

Searches: `ping.?pong|feedback|double.?buffer|swap_texture|prev_frame|previous_frame|back_buffer|u_prev|last_frame`
over py/glsl/md -> every "feedback" hit is UI/agent feedback; every `last_frame` is
`Video._last_frame_idx`. `u_prev|u_back|u_self|iChannel|backbuffer` over `.glsl` -> **zero
hits**. `copy_framebuffer|blit|texture\.write` -> the only texture write is
`Video._upload_frame`'s CPU video upload. No FBO-to-texture copy, no blit.

Structurally it cannot exist today: `render()` clears the target immediately before drawing,
and the only sampler-bindable values are file-backed. **All temporal state in ShaderBox is
CPU-side** — `u_time` and the script engine's uniform writes.

## Export

Differs from live in *target* and *clock*, **not in pass structure** — still exactly one
`fbo.use(); clear(); vao.render()` per exported frame.

- `FitPolicy.RENDER_AT_TARGET` allocates a temporary `Canvas` at resolved dims (aligned to
  `VIDEO_RESOLUTION_ALIGNMENT = 16`), released in a `finally`. Otherwise the node's own canvas
  is the target and resizing happens on the CPU (PIL resize / ffmpeg `-s`).
- Export passes a deterministic clock (`i / details.fps`); live falls back to
  `time.monotonic()`.
- Export fires `on_pre_render` itself per frame; the live path fires it once via
  `session.tick` (firing it in `render()` would double-tick).
- `render_media` enters `export_isolation()` so a stateful script starts cold.

Absent: accumulation buffer, supersample/downsample pass, multisample resolve, temporal
accumulation, higher-precision export target. The one post-process
(`Video.apply_temporal_smoothing`) shells out to ffmpeg `tmix` on an already-encoded file.
