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

**D3. One program per step, compiled from ONE source via `#define`.** `resolve_usage` runs once;
the flattened text is compiled N+1 times with `#define SB_STEP <i>` injected after the header. The
step bodies and `main()` are guarded so each variant contains one entry point. Verified working:
variants compile and introspect independently, and the `#line` anchor (`d1781b5`) keeps error lines
exact under injection.

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
