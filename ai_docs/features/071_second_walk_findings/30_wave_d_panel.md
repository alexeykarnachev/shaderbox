# 071 W-D — Uniforms panel sampler rows, dormant tiles (#10 #5 + D2 D9)

Parent: `01_spec.md § W-D`. Size: small (three package files, one test module), one post-impl
reviewer, no pre-impl.

## What landed

- **The resolver is the document's.** `Document.sampler_source(pass_name, uniform)` answers
  from `effective_graph()`, the same resolution the renderer binds with (069 D9): the stored
  edge or the name default when it names a pass that exists; an explicit `""`, a stale name, a
  pass that declares no such sampler and a pass that does not exist all answer None, which is
  the black the renderer seeds.
- **The row has three states** (`widgets/uniform.py`, the `texture` branch). A pass source: a
  live thumbnail of that pass's canvas captioned with the pass name, no `Load`, no resolution
  line; wiring stays the gear's. A user-bound texture: today's row (`Load`, resolution,
  thumbnail, video filters). Otherwise: `Load`, the caption "unwired", and a black swatch the
  size of a thumbnail, which is what the shader reads. The seeded default image is never drawn:
  it was only ever a placeholder in the slot, since `Document.render` seeds every sampler that is
  not user-bound with black before binding the edges. The precedence matches the renderer's: an
  explicit edge on a user-bound sampler binds the pass, so the row shows the pass.
- **Dormant tiles** (D2) landed during the walk (`STALE_TINT`); no change here. The value is
  tuned from the maintainer's look in the manual check.

The caption of the pass name is a variable, so `tests/test_ui_prose_budget.py` carries it in
`_UNMEASURABLE` with its reason; "unwired" is one word against the caption budget.

## Tests

`tests/test_uniform_panel.py`: the shipped Radiance Cascades example, brought online the way
the live loop's sweep does it, then `composite`'s two samplers resolve to `cascade` and `paint`
(the walk's case), an explicit `""` and a stale name resolve to None, and so do a pass without
the sampler and a pass that does not exist. The draw itself is covered by `make smoke`'s frame
loop and the manual check.

## Manual verification (the maintainer, in the app)

1. Radiance Cascades, `composite` selected: the `u_cascade` and `u_paint` rows show the cascade
   and paint pictures with their names and no Load button. Falsifier: the dog.
2. A new pass with `uniform sampler2D u_tex;` and no pass called `tex`: the row shows Load,
   "unwired" and a black square. Load an image: the row shows it with its resolution.
3. The gear's Reads section and the panel row always agree on the source.
4. Dormant tiles in the strip read as dimmed pictures at the current tint; if the value is off,
   name a direction.
