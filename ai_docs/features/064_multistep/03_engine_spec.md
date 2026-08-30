# 064 — The engine seam: implementation spec

**Status: PLAN-LOCKED (maintainer delegated the review: "I will not review the spec manually, but
you should implement it for the sake of a robust process").** Anchor: `00_scenario.md` R1-R10.
Seam decision + why: `02_decision.md`.

Scope: **the engine only.** No new panel, no new tab, no graph. The authoring surface (proposal B's
list or D's contact sheet) is a separate feature, decided later with a working cascade on screen.

## What ships

A node can declare extra render steps in its shader. The engine compiles one program per step from
the one source, evaluates them in dependency order into per-step targets, and binds each step's
output to the sampler that names it. Everything else — files, errors, hot-reload, persistence,
copilot, export — keeps working unchanged, because a step chain is still one directory and one
`.frag.glsl`.

## D-decisions

**D1. A step is declared by a sampler uniform plus a `// step` rider.**

```glsl
uniform sampler2D u_blur;   // step, scale: 0.5, f2, linear
```

The rider configures the declaration it rides on, so the two cannot desync — the survey's named
hybrid, and the failure mode KodeLife documents (a declaration drifting from its parameter) is
structurally impossible. The step's body is `void step_blur(out vec4 o)` in the same file.
`00_scenario.md` R1.

Rider grammar, all optional after the `step` marker:

| Token | Meaning | Default |
|---|---|---|
| `scale: N` | target size = `max(1, round(canvas * N))` | `1.0` |
| `size: WxH` | absolute target size | — |
| `f1` / `f2` / `f4` | target dtype | **`f2`** |
| `linear` / `nearest` | filter | `linear` |
| `clamp` / `repeat` | wrap | `clamp` |
| `persist` | target survives a recompile | off |

**`f2` is the default, not `f1`.** 8-bit is measured fatal for accumulation (063: `f1` saturated at
255 on the first pass where `f2` reached exactly 7.0), and R7 exists because of that measurement.
The safe value is the default and `f1` is the opt-in. Proposal D reached the same conclusion
independently. `clamp` inverts moderngl's `repeat_x/y = True`, which is wrong for a feedback border.

**D2. A malformed rider is a loud error, never a silent wrong picture.** This is the one defect
every judge flagged in proposal A: `// stp, scale: 0.5` would fall through to an ordinary texture
uniform bound to the shipped default image, so a picture appears and it is the wrong one — the
silent-corruption class `17_direction.md` bans. So:

- A `//` rider on a `sampler2D` whose first token is within edit distance 1 of `step` and is not
  `step` -> a synthetic `ShaderError` on the declaration's line.
- An unrecognised token after a valid `step` marker -> likewise.
- A `step` marker with no matching `void step_<name>(out vec4)` body -> likewise.
- A `step_<name>` body with no matching declared step -> likewise.

All four route through `CompileUnit.errors`, so they land in the existing error strip with
click-to-jump and cost no new UI. The parser is GL-free and unit-testable without a context.

**D3. One program per step, compiled from ONE source by aliasing `main`.** `resolve_usage` runs
once; the flattened text is compiled N+1 times. A fragment shader needs exactly one `main`, so for a
step variant the engine injects, after the header and before the body:

```glsl
#define main sb_user_main
```

and appends, after the body:

```glsl
#undef main
out vec4 sb_step_out;
void main() { step_<name>(sb_step_out); }
```

The final variant injects nothing and compiles the source as-is.

**The engine never modifies user text — it only brackets it.** Three shapes were probed on a real
context; all three compile, and this one was chosen because it neither edits the user's source (a
textual `void main()` rename breaks on unusual formatting) nor duplicates the body inside an `#if`
(which doubles the error-line mapping). The C preprocessor substitutes whole tokens only, verified
adversarially: a shader declaring `u_main_scale` and `domain_warp` compiles unchanged in both
variants.

Measured, and it is the evidence for D4: the final variant introspects `['u_blur','u_gain']` while
the step variant introspects `['u_radius']` — **disjoint sets**, so the union is mandatory, not an
optimisation. The `#line` anchor (`d1781b5`) keeps error lines exact under the injection.

Why not N files: `node.source` and `node.compile_unit` are singular at ~15 call sites each (six in
`copilot/backend.py`), `watch.py` hardcodes `sources[0]` as the root, and `copilot/address.py` has
no slot for a step. One file keeps all of it, and the copilot authors a whole chain with **zero new
tools** because `write_shader` already writes that file. See `02_decision.md`.

**D4. The uniform panel shows the UNION across variants, and writes go to every variant declaring
the name.** Measured: each variant exposes only its own branch's uniforms, so "the live program" is
undefined with N programs. `UINode.save` prunes UI rows against the live program's uniform set, so
without the union a tuned value on any non-final step is **silently deleted on save** — the
data-loss class commit `b25d9e3` was written to prevent. This is the largest correctness risk in
the feature.

Consequence made explicit: a uniform of the same name in two steps is ONE row driving both. That is
the desired behaviour (a shared `u_ray_count` across eight cascade levels is the ergonomic win the
whole feature exists for) and it must be stated, because it means a step cannot have a private
uniform that merely shares a name.

**D5. Evaluation is a topological sort with memoization; each step renders at most once per frame.**
Both prior designs in the maintainer's own repos got exactly one half of this: the deleted DAG had
correct pull-recursion order but no memoization (a diamond re-renders a shared ancestor per
consumer); freska had per-node targets but iterated an `unordered_map` under its own
`// TODO: this is incorrect!`, lagging an N-step chain by N-1 frames nondeterministically. Neither
got both.

The edge set is derived from which `u_*` step samplers each `step_*` body reads, resolved on the
flattened source so `SB_*` lib splicing is already applied.

**D6. A step reading itself is implicit ping-pong.** The engine double-buffers that step and hands
the previous frame's texture; the user never names a pair. The survey's single strongest
convergence ("if ShaderBox adopts one idea from this survey, this is the one"). A self-edge
contributes NO ordering constraint and is excluded from the cycle check — the subtle trap proposal C
correctly identified. R5.

**D7. A non-self cycle is a loud error, not a hang.** Reported per-step through `CompileUnit.errors`
like any other compile failure.

**D8. Ordering is asserted where it can be seen.** The resolved order is exposed on `Node` and
covered by tests. freska's ordering bug was invisible precisely because nothing displayed the order;
an ordering nothing checks is an ordering nobody checks.

**D9. Step targets are transient by default and never serialized.** `persist` survives a recompile,
not a reload. Persisting float buffers means reviving the raw-`Texture` write path for megabytes of
transient state per save, which `14_ergonomics.md` measured as a defect in its own right, and it
would stop `node.json` being small app-written derived state.

Copilot revert therefore clears every step target. That is correct rather than a shrug: a reverted
shader reading a buffer accumulated under the un-reverted shader is a state that never legitimately
existed — the silent-corruption class again. Three of the four proposals reached this independently.

**D10. Export starts cold.** `render_media` already brackets every export in `export_isolation()`,
sited once "so no export caller can forget to isolate", and the script engine is deliberately
re-instantiated fresh per export. Step targets obey the same discipline or the same node exported
twice differs by how long the app has been open. A warm-up control belongs with the UI feature, not
here.

## D11-D16: precautions from the blast-radius review

A pre-impl reviewer swept every call site depending on the assumptions this spec changes, and
measured the resource ceiling. Six findings became decisions; each names the site that forced it.

**D11. Step samplers NEVER enter `uniform_values`, enforced by ONE predicate on `Node`.**
This is where D4's union and D9's ban on serialization meet, and the spec did not name the meeting
point. Verified: `core.py::_default_uniform_value` hands any `GL_SAMPLER_2D` an
`Image(DEFAULT_IMAGE_FILE_PATH)`, and `ui_models.py::UINode.save` writes any `moderngl.Texture` in
`uniform_values` to `textures/<name>.bin` recording size/components/dtype — a real, tested path
(`test_raw_texture_round_trip.py`). So without a guard: a 512x512 `f2` target is ~1 MB per step per
save, reloaded next session as a frozen stale frame bound to the sampler, with no error anywhere.

The same predicate is consulted by `seed_uniform_values`, the `save` serialization loop, the
`live_rows` prune, `tabs/node.py`'s row + resolution loops (which would otherwise `AttributeError`
on `value.texture.size` for a bare `Texture`, and offer a "Load media" button that overwrites the
chain wiring), and `copilot/backend.py`'s `bind_media` sampler lists. It also removes a
double-release: `release()` calls `try_to_release` on every `uniform_values` entry, and
`invalidate()` would free the same target again.

**D12. Step targets are sized off `self.canvas`, never off a passed canvas.** `ui.py:214` is the
ONLY external-canvas caller in shipped code (`render(canvas=app.preview_canvas)`) and it renders at
`adjust_size(..., width=200)` — a different resolution. Sizing off the passed canvas would
thrash-reallocate every target between 200px and full every frame. The consequence, accepted and
stated: **the small preview is a downscale of a full-resolution chain**, which is correct for a
scale-dependent blur/cascade chain and is what the node-grid thumbnail already shows.
`render_media`'s preset branch (`core.py:561`) takes the same rule.

**D13. The ping-pong swap is tied to a FRAME, not to a `render()` call.** The current node renders
twice per frame (`ui.py:214` into the preview canvas, then `ui.py:247/250` into its own), and the
copilot probe renders twice back-to-back (`backend.py:1701/1709`). Swapping per call would make the
current node's feedback advance at 2x every other node's — so a decay constant tuned while a node is
selected evolves at half the rate once it is not, and toggling "Render all" changes it again. Worse,
the probe's second frame would carry the first's accumulation, so `_MOTION_EPS` reports ANIMATES for
a static chain and the no-op detector can never fire — both feed the model's beliefs directly.

**D14. A step-parse error refuses the whole compile.** `compile()` already preserves the last-good
program on failure, so the picture keeps rendering while the error strip shows the problem. The
dangerous inverse: compiling a single variant with step samplers left as ordinary textures binds
them to the default image — the silent wrong picture D2 exists to forbid.

**D15. Test the silent regressions, not just the loud ones.** Three existing tests keep passing while
the behaviour underneath them breaks, so each needs a twin in this wave:
`test_gl_lifetime_guards.py` asserts only the canvas is freed (N step targets can leak with it
green — and the count must be asserted, or dropping variant N-1 passes);
`test_uniform_row_pruning.py::test_every_surviving_row_names_a_live_uniform` is a subset check with
no lower bound, so the union makes it pass MORE easily while the data loss D4 prevents stays
invisible; `test_render_for.py::test_render_media_preset_none_byte_identical` is the natural D10
falsifier but its fixture is step-free, so it must be parametrized over a chain node.
`test_raw_texture_round_trip.py` needs a negative twin: a step sampler must NOT write a
`textures/*.bin`.

**D16. State the measured ceiling.** 063's "19 passes at 0.52 ms" is ONE node's chain and does not
extrapolate. Measured on this box: **20 nodes x 15 steps at 512x512 `f2` = 7.28 ms/frame** (44% of a
16.6 ms budget, GPU work alone) and **629 MB of VRAM** — before ping-pong doubles the self-reading
targets. **VRAM is the real ceiling, and `f2` is what makes it steep.** Recompiling 15 variants costs
57.8 ms against 1.5 ms for one, so every Ctrl+S on a deep chain is a ~60 ms main-thread stall, paid
again per node by the copilot's opportunistic `compile()` calls. Not a blocker for this wave; a
bound on total step-target allocation is a real follow-up, and the existing stale-mark makes
"don't render every node's full chain" the honest default.

## Out of scope for this wave

- **Any UI.** No strip, no rows, no chips. The panel change is D4's union only, which is a change to
  which uniforms the EXISTING rows enumerate, not a new surface.
- **The live HDR view transform.** Design-independent prerequisite (`02_decision.md`); float
  intermediates preview white until it lands. Owed, tracked, not this wave.
- **MRT**, inter-node wiring, per-object simulation, 3D — out per `00_scenario.md`.

## Files touched

- `shaderbox/step_spec.py` (new) — the rider parser + step model. GL-free, no `App` import, so no
  cycle-from-types pressure.
- `shaderbox/core.py` — `Node`: per-step programs, targets, order, ping-pong; `compile` builds N
  variants; `render` evaluates the chain; `release` frees step targets.
- `shaderbox/ui_models.py` — `UINode.save` enumerates the union (D4).
- `tests/test_step_spec.py`, `tests/test_step_chain.py` (new).

## Verification

Each check fails for exactly one reason.

1. Rider parsing: every token, defaults, and each of D2's four malformed cases -> one error each.
2. A 2-step chain renders B-reads-A, not A alone. Falsifier: the output equals step A's.
3. Order is topological on a diamond, and the shared ancestor renders ONCE (memoization).
4. A self-read gets last frame's content; the value advances across frames and does not read its
   own current write.
5. A non-self cycle reports an error and does not hang.
6. `f2` accumulation exceeds 1.0. Falsifier: it clamps, proving the target is 8-bit.
7. Union: a uniform declared only in a non-final step survives a save/load round-trip.
8. Mutation: reverting memoization, the topological order, or the union must each turn the suite
   red.
