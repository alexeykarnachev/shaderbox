# Stress test: what survives attack, what breaks

The fifth reviewer attacked the proof along seven axes with running experiments (scratchpad
`exp1`-`exp11`, all re-runnable). Baseline reproduced; `make test` green (792 passed), so every
finding below is untested behaviour, not a broken tree.

**It corrects two earlier claims — one in the approach's favour, one against.**

## What PASSES, several bit-exactly

- **Canvas resize — PASS.** `512 -> 256 -> 1024x768 -> 97x53 -> 1x1 -> 2048 -> 512`: no crash, no
  script error, no corruption. uv-space sampling really is resolution-independent. The
  `u_resolution` vs fixed-RES desync is a **stretch**, not corruption.
- **Multiple nodes — PASS, bit-exact.** Two cascade nodes interleaved for 30 frames vs a solo
  control: `IDENTICAL`, max delta 0. No texture-unit collision, no context bleed.
- **Mid-pass exceptions — PASS.** A raise with an FBO bound and viewport 512 leaves the next
  node CLEAN, because the next `fbo.use()` restores binding + viewport.
- **Export pixels — correct.** 640x480 export mean 23.85 vs live 23.82, stable over 20
  consecutive exports.

**Correction in the approach's favour:** the `sys._getframe()` walk is **more robust than I
claimed**. It resolves the correct node in every path (live, `dry_run`, `tick_export`), and even
when an *outer* frame shadows `node` with the wrong one — it hits `_tick_script`'s parameter at
depth 2 first. The fragility is real but narrower than "works by luck": it breaks only if that
specific parameter is renamed.

## GL state leakage is exactly ONE bit

Isolated directly: `moderngl.Framebuffer.use()` resets viewport, scissor-test and colour-mask,
but **BLEND survives it**.

| leaked state | victim node |
|---|---|
| raise mid-chain (FBO bound, viewport 512) | CLEAN |
| `enable(BLEND)` / raw `glEnable(GL_BLEND)` | **CORRUPTED — red_frac 1.0 -> 0.0** |
| scissor, colour-mask, viewport, depth, cull, stencil | CLEAN |

One-line epilogue (`ctx.disable(moderngl.BLEND)`), leak surface precisely bounded.

## The leak, measured properly — and a correction against my framing

Counting **live GL names** via `glIsTexture`/`glIsFramebuffer`/`glIsProgram`, not moderngl's
`glo` counter:

| edits | default `gc_mode=None` | `gc_mode="auto"` |
|---|---|---|
| 0 | 3 tex / 3 fbo / 2 prog | 3 / 3 / 2 |
| 20 | 43 / 43 / 22 | 17 / 17 / 9 |
| 50 | **103 / 103 / 52 ~ 206 MiB** | 21 / 21 / 11 |

`auto` **does** cover FBOs and programs, not just textures.

**Correction against my "one-line fix, done" framing:** `auto` is not a clean flat line — it
leaves a residual (21 vs the ideal 3), because the Behavior's VAO<->program<->buffer graph is
**cyclic**, so refcounting cannot free it and release waits for a generational GC pass.
Diagnosed by falsifier: adding `gc.collect()` per edit pins it at exactly 3, and `gc.garbage` is
empty. So it is a **bounded lag, not a leak** — but the fix is less clean than I said.

Correctness held across all 50 edits: canvas mean stable 23.82, zero errors.

## The two hard failures on ordinary user actions

**Revert/reload destroys live GL, and does NOT self-heal.** `revert.py::_reload_node_in_place`
calls `old.node.release()`, which `try_to_release()`s every value in `uniform_values` — including
the script-owned texture the still-live instance owns:

```
after node.release(): inst.tex[0].read() RAISED AttributeError: 'InvalidObject' ...
engine still holds the SAME instance after revert-style reload? True   <- mtime unchanged
post-revert render mean: 135.5   (correct = 23.8)
--- 100 more frames --- still broken, mean 135.5
--- eng.reset() --- no errors, mean 23.819
```

**Permanently broken until a manual script restart**, because `reload()` keys on mtime and revert
does not change it. The on-screen error (`'InvalidObject' has no attribute 'use'`) points nowhere
near the cause. Same hazard on `_load_one_node_from_disk` — the **external node.json watcher**,
reachable by any outside edit — and on delete/sync.

**Save crashes — on the path every Ctrl+S and every copilot checkpoint takes.** Confirms
`13_reliability.md` defects 3/6 from a third angle, and adds the consequence: fixing the mkdir
makes it write **2 MB of binary garbage per save** and then hard-crash on load. Downstream,
`snapshot_node` catches it and files the node under `failed_nodes` — **every copilot turn
touching a cascade node is silently un-revertable**, and the manual path toasts "Save failed"
while the tuning is silently not persisted.

Both share one root: **the node's release/save/load paths assume it owns everything in
`uniform_values`.**

## dry_run: the violation, quantified

Copilot-realistic sequence — warm live to t=1.98, `dry_run` at t=0, re-render at the **same**
`u_time` with **no** tick between:

```
live t=1.98 canvas mean:      22.840
after dry_run, SAME u_time:   23.829
DELTA: max 230/255, mean 8.8, 65927 / 262144 px changed (25%)
=> promise VIOLATED
```

**Not fixable by a small change.** Sink-redirecting the returned dict *is* the isolation
mechanism, and the sampler deliberately bypasses it (`_binding_reject` refuses sampler keys, so
it can never travel through the dict). Real isolation needs a sanctioned node handle that
respects the sink, or dry-running against a copy of the node.

Note the subtlety: the canvas texture read immediately after is byte-identical (the dry run does
not draw the node) — **the very next render diverges.** The corruption is invisible: no error,
no log, just a wrong picture.

## Verdict

**Not reliable enough to build real work on as it stands — but the gap is four fixes, three
small.**

The GPU half is genuinely solid under attack: performance, correctness, resize, multi-node
isolation and mid-pass recovery all pass, several bit-exactly. "The capability is already there"
holds up. What breaks is **every path where the engine touches the node's lifecycle.**

*Acceptable with discipline:* the hot-reload leak (one line, bounded GC residual), BLEND (one
line, one bit), export sampler theft (self-heals on the next live tick), canvas/RES desync
(visual, documented).

*Genuinely dangerous:* save/checkpoint crash (fires on Ctrl+S; highest blast radius; small fix),
revert/reload destroying live GL (no self-heal, misleading error; small fix, but must be paired
with the save fix — same root), and `dry_run` isolation (**not** small; it is the copilot's
feedback channel).

**Do not point the copilot at a cascade node** until the dry_run and save issues are addressed —
that pairing gives an agent a lying preview *and* a broken undo simultaneously, which is the
worst possible combination.

**The honest reframe of `12_it_already_works.md`'s closing:** the caveats it lists as "real and
unresolved, but not arguments against playing with it now" are right in spirit, but two of them
(save, revert) are **hard failures on ordinary user actions**, not theoretical contract
breaches. That belongs in the headline, not the caveat paragraph.
