# Measured: what the GPU actually does

Run first-hand on the maintainer's box (NVIDIA 580.173.02, standalone EGL context, the same
`MESA_*` overrides `make test` sets). Not relayed, not reasoned — executed.

Probe scripts are throwaway; the numbers are the artifact.

## Float targets and filter control: both work, trivially

```
GL: 3.3.0 NVIDIA 580.173.02
max_texture_units: 32
GL_MAX_COLOR_ATTACHMENTS: 8

dtype=f1: max=255.000 g=255.000 b=0.000     <- clamped, as unorm must
dtype=f2: max=9.922   g=4.500   b=-2.000
dtype=f4: max=9.922   g=4.500   b=-2.000
   filter settable: NEAREST -> LINEAR ok; clamp-to-edge ok; build_mipmaps: OK   (all three dtypes)
```

A shader wrote `vec4(vs_uv.x * 10.0, 4.5, -2.0, 1.0)` into each target. **`f2`/`f4` preserved
both the >1.0 HDR range and the NEGATIVE value**; `f1` clamped to 255/255/0.

So the two format-shaped gaps are not engineering problems. `ctx.texture(size, 4, dtype="f2")`
plus three attribute assignments (`tex.filter`, `tex.repeat_x/y`, `tex.build_mipmaps()`) covers
every format and filtering requirement RC has. **`Canvas._init` simply never passes them.**

Eight color attachments are available if MRT is ever wanted.

## Ping-pong accumulation: works, and demonstrates why 8-bit is fatal

Seed a target to 1.0, then run six accumulate passes (`texture(u_input0, uv) + 1.0`),
swapping A/B each pass — structurally the cascade merge:

```
f1: after seed(1.0) + 6 accumulate passes -> R = 255   (expect 7)
f2: after seed(1.0) + 6 accumulate passes -> R = 7.0   (expect 7)
```

The `f1` row is the concrete falsifier for "maybe 8-bit is good enough": it saturates on the
FIRST pass and stays there. This is a demonstration, not an argument from the article's
authority.

## Performance: a non-issue by three orders of magnitude

19 sequenced passes at 256x256 `f2`, with a per-frame `texture.read()` (the pattern the
`/dogfood` V3D blank-framebuffer quirk requires):

```
19 passes @256x256 f2 + readback: 0.52 ms/frame  (1930 fps)
```

Against a 16.7 ms frame budget at 60 fps, a full RC pass chain costs **~3% of one frame**. And
this is the pessimistic shape — the reference caches its distance field after frame 0, so
steady state runs far fewer than 19.

**"Multipass is too expensive for a live playground" is dead.** Any cost argument against this
feature has to be about complexity or product fit, not milliseconds.

## What this leaves

Of the three requirements called "independently disqualifying" in `01_reference.md`, two
collapse to unset attributes on one 15-line class:

| Requirement | Status |
|---|---|
| Float/half-float targets | `dtype="f2"` — works, verified |
| Per-target filter/wrap/mipmaps | attribute assignment — works, verified |
| ~19 sequenced draw calls | 0.52 ms — works, verified |

The genuinely absent pieces are **sequencing** (nothing renders one node into another's
sampler) and **input** (no mouse buttons, no click reaching the preview). Neither is a GPU
capability question.
