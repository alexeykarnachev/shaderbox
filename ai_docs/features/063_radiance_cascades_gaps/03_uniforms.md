# Audit: the uniform system (current state)

Source: code read of `core.py`, `ui_models.py`, `widgets/uniform.py`, `tabs/node.py`,
`uniform_coerce.py`, `scripting/`, `glyph_tables.py`.

## The load-bearing find

**The raw-texture binding path already exists and is already exercised.**
`Node.render`'s sampler branch accepts `MediaWithTexture` *or* a bare `moderngl.Texture`:

```python
if isinstance(value, MediaWithTexture):
    value.update(render_time); texture = value.texture
elif isinstance(value, moderngl.Texture):
    texture = value
else:
    raise ValueError(...)
```

and `Node.load_from_dir` reconstructs one from `textures/<name>.bin` (a `file_path` whose
parent dir is `textures` rather than `media`), with `UINode.save` writing that side too.

So "bind a texture that is not a file" is **plumbed end to end and has no producer** — every
UI and copilot bind path produces a `MediaWithTexture`. Handing a sampler another node's
`canvas.texture` is a smaller change than "samplers accept files only" would imply.

## Type coverage

No explicit GLSL-type -> widget table exists. Dispatch is on `(gl_type, dimension,
array_length)` triples in `UIUniform.valid_input_types`, then an if/elif chain in
`draw_ui_uniform` over seven input types: `texture, buffer, array, color, text, drag, auto`.

Editable: `float/int/uint` (drag), `vec2..vec4` (drag or colour), `sampler2D` (load button),
`float[N]` and `uint[N]` (comma-separated text; `uint[N]` also gets the glyph-text editor),
UBO (a "Randomize" button over opaque bytes).

Read-only (`auto`, displayed but not editable): any array whose base type is neither
`GL_FLOAT` nor `GL_UNSIGNED_INT` — so **`int[N]`, `ivec2[N]`, `vec3[N]` are display-only**.
The array editor is flat: no per-row editing for a `vec3[8]`, though the *coercion* layer
(script/copilot path) does chunk flat lists into rows.

Entirely unmodelled — `grep` for `GL_FLOAT_MAT|GL_SAMPLER_CUBE|GL_SAMPLER_3D|GL_BOOL|GL_IMAGE|SAMPLER_2D_ARRAY`
returns nothing: **`mat2/3/4`, `sampler3D`, `samplerCube`, `sampler2DArray`, `bool`, image/storage
types.** They are not rejected either — they fall through to `auto`, get seeded from
`uniform.value`, and are written verbatim; a moderngl failure is caught and the cached value
popped.

## Engine-driven uniforms

Exactly five names (`core.py::ENGINE_DRIVEN_UNIFORMS`): `u_time`, `u_aspect`, `u_resolution`,
plus the program-resident `SBT_SPANS`/`SBT_STROKES`.

Adding a per-frame one touches **two** places: the frozenset literal, and a new
`elif uniform.name == ...` branch in `Node.render()`. Everything downstream keys off the
frozenset automatically — `seed_uniform_values` skips it, `UINode.save` never persists it,
`valid_input_types` forces the read-only `auto` row, the script engine treats it as
engine-owned. There is no declarative registry, so the `render()` branch is unavoidable.

Absent: `u_mouse`, `u_frame`, `u_dt`, `u_date`, `u_channelN`.

## sampler2D specifics

Every sampler is always bound to something; "unbound" is an identity test
(`media.py::is_default_image` against `resources/textures/default.jpeg`), not a null state.

Binds come from: the `Load` button (native `portable_file_dialogs` picker, filtered to
`MEDIA_EXTENSIONS` = png/jpg/jpeg/bmp/webp + mp4/webm/mov), or the copilot's
`bind_media`/`unbind_media`.

Absent: node-to-node binding (no producer, though the path exists — see above), webcam, URL,
procedural source, clipboard, drag-and-drop (`drop_callback|begin_drag_drop` -> zero hits),
thumbnail-click-to-load.

Texture units are assigned positionally by a counter in `render()`, in
`get_active_uniforms()` order.

## Arrays and UBOs

Arrays are detected by `array_length > 1` and edited as a flat comma list, parsed under
`suppress(Exception)` and truncated to the cap. Exact-length, no padding for data arrays
("padding a data array is silent corruption").

UBOs are **opaque bytes** — `is_ubo` locks the input type to `buffer`, whose only control is
a "Randomize" button filling random float32s, plus a byte-size caption. No member
introspection: no field names, no types, no per-field rows. UBOs are excluded from scripting.

`TABLE_UNIFORMS` (`glyph_tables.py`) is a `dict[str, bytes]` with two entries, written into
any program declaring the name during `Node.compile()`. Mechanically a third entry would work;
structurally it does not generalize — hardcoded imports in three modules, no registration API,
no way for a user shader or lib file to contribute, no per-node scoping.

## Persistence

`UINode.save` rebuilds `meta["uniforms"]` from the **live program**, skipping engine-driven
names. Scalars and lists go inline; a UBO goes base64; a `MediaWithTexture` is **copied into
`<node>/media/`** and referenced by relative path; a raw `moderngl.Texture` goes to
`<node>/textures/<name>.bin`. Unsupported types are dropped with a warning.

An unbound sampler is deliberately **not persisted** (and stale files are unlinked), so
`seed_uniform_values` re-establishes the default on load. A bound sampler survives reload, but
the node owns a *copy* — the link to the user's original file does not survive.

Two GC passes (stale UI rows, orphaned assets) run only when `node.program is not None`; when
it is None, `meta["uniforms"]` is copied forward verbatim from the existing `node.json` rather
than written empty.

## Scripting relationship

A script's `update(ctx) -> dict` is fanned into `node.uniform_values` before `Node.render()`
reads them, fired via `Node.on_pre_render`. Writable: `int`/`float`, `Vec2/3/4` (real vector
math), `Array` (flat or auto-flattened rows, exact length), `Text`.

**A script cannot write a texture.** `grep Texture|Image|Video` over `shaderbox/scripting/*.py`
-> zero matches, and the gate is structural:

```python
def is_scriptable(uniform) -> TypeGuard[moderngl.Uniform]:
    return (not isinstance(uniform, moderngl.UniformBlock)
            and getattr(uniform, "gl_type", None) != GL_SAMPLER_2D
            and hasattr(uniform, "dimension") and hasattr(uniform, "array_length"))
```

Both UBOs and sampler2Ds are non-scriptable and absent from the generated stub.
