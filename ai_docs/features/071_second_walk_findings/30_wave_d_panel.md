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
the sampler and a pass that does not exist; with a stored edge removed the name rule still
answers, which only the effective graph knows; and a self-reading sampler's texture is the
feedback history, a different object from the live canvas. `make smoke`'s frame loop draws the
pass-source branch on the shipped example (74 rows a run); the unwired and user-bound branches
are drawn by the manual check.

Two things landed differently from the spec's sketch, both on purpose: the row calls
`document.sampler_source` itself, once per sampler row, instead of `tabs/document.py` resolving
once per draw and passing a `source` in (GL-free dict work over a handful of passes, and the
knowledge stays in one place); and the gear and the row are two call sites over the same
resolution (`effective_inputs`) rather than one shared call, verified to agree in every state.

## Review history

**Post-impl (one reviewer, opus, code correctness + spec fidelity).** Verdict FIX, taken in a
fix-up: (1) the test never exercised the name-default branch, since every edge in the shipped
graph is stored explicitly -- resolving from `document.graph` instead of the effective graph
stayed green; one assertion with a stored edge removed now goes red under that mutation. (2) A
self-reading sampler's row drew the pass's LIVE canvas while the renderer binds the feedback
history (measured: two textures, different content); `Document.input_texture` now answers with
what the renderer binds, pinned on `cascade`'s `u_prev`. (3) The smoke-coverage claim was wider
than the branch smoke reaches; trimmed above. Cleared by the reviewer, with probes: resolver and
renderer agree in every state (stored edge, stale name, explicit `""`, name default, user-bound
with and without an edge, an uncompiled pass), the gear agrees with the row in every state,
no GL lifetime issue (every render completes before `imgui.new_frame`), the swatch goes through
the theme, the unwired and user-bound rows are the same height so binding a texture does not
shift the layout, "unwired" is measured by the prose gate and the registered caption's reason
is accurate, and the module's standalone context is rebound by the App's `init_context`.

## Manual verification (the maintainer, in the app)

1. Radiance Cascades, `composite` selected: the `u_cascade` and `u_paint` rows show the cascade
   and paint pictures with their names and no Load button. Falsifier: the dog.
2. A new pass with `uniform sampler2D u_tex;` and no pass called `tex`: the row shows Load,
   "unwired" and a black square. Load an image: the row shows it with its resolution.
3. The gear's Reads section and the panel row always agree on the source.
4. Dormant tiles in the strip read as dimmed pictures at the current tint; if the value is off,
   name a direction.
