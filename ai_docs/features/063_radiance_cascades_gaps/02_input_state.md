# Audit: interactive input (current state)

Source: code read of `shaderbox/scripting/context.py`, `shaderbox/ui.py::_draw_app_panel`,
`shaderbox/core.py`, `shaderbox/app.py`, plus the negatives' searches.

RC's demo is, mechanically, a **painting app**: drag to paint light and occluders, with
strokes interpolated from the previous point. That makes input a first-class requirement,
not a nicety.

## What exists

**`ctx.mouse` in scripts** — `scripting/context.py::MouseState`:

```python
@dataclass(frozen=True)
class MouseState:
    x: float = 0.5
    y: float = 0.5
```

Normalized `0..1` over the **displayed preview rect**, y-up (GLSL convention; the preview
draws uv-flipped and `item_normalized_mouse(..., flip_y=True)` flips back). Scale-correct: it
normalizes against the displayed rect, so a 1080p canvas shown at 400px still maps edge to
edge.

**Per-tick timing** — `EngineContext(t, dt, frame, mouse)`. Live: `t` is `glfw.get_time()`
(wall clock since glfw init, NOT since project open), `dt` is wall-clock frame delta. Export:
fixed-step `i/fps`; still-image export passes `dt=0.0, frame=0`.

**Export determinism** — `EXPORT_MOUSE = MouseState(0.5, 0.5)` is frozen in export and in the
headless probe, and the copilot prompt teaches it: "drive motion from ctx.t, never ctx.mouse".

## What does not exist

**No `u_mouse` shader uniform.** The engine writes exactly `u_time`, `u_aspect`,
`u_resolution` plus the program-resident glyph tables (`core.py::ENGINE_DRIVEN_UNIFORMS`).
A shader reaches the cursor only if a user script wires `ctx.mouse` into an ordinary uniform.

This was a deliberate cut, recorded in `042_script_ui.md`: an auto-fed `u_mouse` was
considered and rejected as "a SECOND write-path for the same cursor value". **A spec that
wants `u_mouse` must engage with that decision, not silently reverse it.**

**No buttons, drag, or wheel.** `MouseState.__dataclass_fields__` is exactly `{x, y}` — no
down/up, no click, no drag delta, no previous position, no `inside` flag. Also a recorded 042
cut: "v1 `ctx.mouse` carries position only".

**Hover is latched, not gated.** `ui.py::_draw_app_panel` updates `app.script_mouse` only when
the hit-test reports inside; otherwise it leaves the last in-bounds value in place forever. A
script cannot distinguish "hovering here" from "parked here after the cursor left".

**The preview is display-only.** It draws via `imgui.image_with_bg`, which submits no
interactive imgui item — no ID, no hover/active state. A click on the render reaches nothing.
The cursor-over-preview path feeds `ctx.mouse` and nothing else. (The node-grid thumbnails
ARE clickable, but only for select/delete; no in-thumbnail position is computed.)

**One-frame lag by construction.** `session.tick(...)` runs near the top of `update_and_draw`;
the preview and its hit-test are drawn later in `_draw_app_panel`.

**No keyboard surface.** `shaderbox/scripting/` imports no imgui and no glfw (the deliberate
headless-core boundary). `EngineContext` has four fields and none is a key state. The only
glfw keyboard hook in the app is `App._install_escape_filter`, which exists to swallow
jobless Escape.

**No painting into a texture, anywhere.** Sampler uniforms accept only file-backed
`Image`/`Video` or a raw `moderngl.Texture`; `Node.render` raises on anything else. There is
no CPU-writable scratch texture, no feedback buffer, no previous-frame texture. Searches
behind this negative: `paint|brush|canvas_draw|scribble|stroke` over `shaderbox/` returns only
the SDF glyph stroke table, imgui draw-list chrome, and "paints the cue" prose about the
render overlay; `feedback|ping_pong|prev_frame|u_prev` returns zero hits.

## Bearing on RC

| RC needs | Today |
|---|---|
| Drag to paint light/occluders | no button state, no click on preview |
| Strokes interpolated from previous point | no previous position, no drag delta |
| Erase mode (a modifier or second button) | no button state at all |
| Persistent accumulating draw canvas | no writable texture of any kind |
| Cursor reaching the shader | only via a hand-wired script uniform |
