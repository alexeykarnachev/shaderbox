# Reliability: the script route is a good probe and a bad way to work

An architectural review of the script-driven multipass approach found **three verified defects
the earlier docs missed**, and one **inverted** feedback loop. `12_it_already_works.md` is still
correct that no architecture is needed; it under-counts the cost. Two of the three were
reproduced first-hand here before being written down.

The distinction that matters: the proof demonstrated **the engine cannot stop you**. The docs
then read that as **the engine supports you**. Those are different claims.

## Defect 1 — silent use-after-free on the ordinary hot-reload path (REPRODUCED)

`Node.release()` frees every value in `uniform_values`, and `util.try_to_release` is
**duck-typed** (`getattr(value, "release", None)`), so a script-owned `moderngl.Texture` parked
there gets freed by the node:

```
script tex glo before: 2
after node.release(), script tex: 2
USE-AFTER-FREE: script drew into released FBO with NO error raised
```

The behaviour instance still holds `self.tex` / `self.fbo` and draws into them on the next
tick. **Nothing raises.** `Node.release()` is not exotic — the comment beside it names the
callers: "every reload (the file watcher, a revert, a project switch)".

The trigger is worse than it first looks: `ScriptEngine.reload` rebuilds the behaviour only
when **`script.py`'s** mtime changes. Edit the **shader** and the node frees the texture while
the script keeps its stale instance. Nothing resets the mismatch.

Two contracts are written down, enforced, and contradictory — `core.py`'s "The node OWNS its
uniform values" versus the script's ownership of `self.tex`. That is the exact class
`conventions.md` legislates against: *"Structural impossibility over guard-piles... redesign so
the unsafe outcome can't be EXPRESSED."*

## Defect 2 — `dry_run`'s isolation guarantee is already false

`ScriptEngine.dry_run` promises the live node is "byte-identical afterward", implemented by
routing writes into a `values_sink`. **The sink only catches writes made through the return
dict.** A side-effecting `node.uniform_values[name] = tex` bypasses it — the script writes the
live dict directly.

Measured by the reviewer: the live node's sampler was swapped to the probe's throwaway texture,
and the probe reported `driven: set()` — the engine did not know a uniform was touched. The
probe's texture is untracked, so **every `write_script` leaks one.**

This is a **checker that silently narrows its own domain** (global rule 5): it promises
isolation over *what the script did* and delivers it over *what the script returned*. Same
code, different promise, reports clean forever.

## Defect 3 — saving such a node throws (VERIFIED BY READING)

`ui_models.py::UINode.save` *does* have a raw-`Texture` branch writing
`textures/<name>.bin` — but **only `dir` is mkdir'd**; nothing creates `textures/`. So the
branch raises `FileNotFoundError`. Consistent with `04_render_pipeline.md`'s finding that the
path "has no producer": **it has never executed.**

And the load side is broken independently: `Node.load_from_dir`'s textures branch passes
**no `dtype`**, so moderngl defaults to `f1`. An `f2` texture writes 32 KB and reads back as
8-bit expecting 16 KB. The round-trip is broken on both sides.

Because the copilot checkpoint is defined as "a full `save_ui_node` serialize of the LIVE node"
and `snapshot_node` is best-effort try/except, the failure is **swallowed** into `failed_nodes`
and surfaces as "Could not restore X (no snapshot)". **Revert silently stops working on exactly
the nodes doing the most interesting work.**

## Defect 4 — the copilot's feedback is not degraded, it is INVERTED

`CopilotBackend._render_facts_for` calls `node.render(...)` and **never ticks the script**
(`on_pre_render` fires only from the export loops; `core.py` says so explicitly: "NEVER from
render()"). For a normal script this is fine — sampled values are merged in first. For a GL
script, the sampler holds whatever the last live tick left there, so `probe_render` returns a
**plausible, non-blank, stale** frame. Headless (dogfood) renders the default image.

That is strictly worse than a blank frame, which the agent is trained to read as failure.

`_motion_verdict` compounds it: a GL script returns `{}` or a couple of scalars, so the agent
is told *"drives 0 uniforms... Nothing animates and every uniform stays manual"* about a script
running six cascade passes. With `_script_broken_streak`, the agent re-edits, is told the same,
and after N rounds **force-restore throws away working code.**

## Defect 5 — export leaks a full resource set each time

`export_isolation` builds a fresh behaviour per export, so `_setup()` runs again: a second full
GL resource set allocated, and the isolation context drops the behaviour on the floor in
`finally`. Ten exports, ten leaked cascade chains.

## The ratchet — why "decide later" does not stay open

- **The `sys._getframe()` walk couples user scripts to a private parameter name** at a fixed
  stack depth. Rename `_tick_script`'s `node` parameter and every such script breaks
  **silently** (the proof does `if n is not None:`). The engine's refactoring freedom becomes
  hostage to user scripts, and nothing records this.
- **The natural fix for defect 1 is a per-call-site guard** — precisely the guard-pile the law
  forbids. Ownership is exactly the invariant that must be lifted onto the entity.
- **Nothing marks these nodes.** A year on, some nodes are secretly multi-pass, distinguishable
  only by grepping `script.py` for `moderngl.get_context()`. Every future change to
  `Node.release`, `UINode.save`, the checkpoint, or export must ask "does a script own GL
  here?" and **cannot answer**. That is the definition of an unofficial second API: not that it
  exists, but that the rest of the codebase must accommodate it without being able to detect
  it.

Fair to the approach: this is **not** a violation of the no-sandbox posture, which is locked
and deliberate. The trap is not that scripts have power — it is that the engine makes
ownership claims over `uniform_values` that a powerful script silently breaks.

## The minimum principled fix — four changes, no node graph

**1. `ctx.gc_mode = "auto"`** at context creation. Owed regardless; every GL-touching script
leaks on every save today.

**2. Put `node` on `EngineContext`.** Kills the `_getframe()` walk and the frozen-parameter
coupling. (The 041 precedent for cutting `ctx.state` does not apply: that was speculative with
no consumer; this has a working one committed beside the doc.)

**3. A sanctioned script-owned-texture declaration — the load-bearing one.** The whole cluster
(use-after-free, `dry_run` violation, save crash, export leak, checkpoint failure) reduces to
one missing fact: **who owns this texture.** Lift it onto the entity per the funnel law —
e.g. `ctx.node.set_script_texture(name, tex)` recording the name in `script_owned: set[str]`.
Then each site is correct by construction: `Node.release` skips them; `UINode.save` skips them
(they are derived, regenerated next tick); `_tick_script` routes them to `values_sink` under
`dry_run`; the behaviour gets a `release()` that export isolation calls in `finally`.
**One concept, five call-sites.**

**4. Make `probe_render` honest.** Either tick the script before the probe, or — cheaper — stamp
`render: STALE (script-owned sampler not ticked)` on the facts line. Given (3) the node knows.
An honest "I cannot see this" is worth far more to the agent than a confident stale frame.

Also worth fixing in the same wave, both independent of this feature: the `textures/` mkdir +
the missing `dtype` on load, and the blend-state epilogue.

## If working this way BEFORE (3) lands

- **One node only** — a second engages the ratchet.
- **Never save or export it**; expect Revert to be dead on it.
- **Set `gc_mode` first**, or every iteration leaks and the session degrades in a way that looks
  like a shader bug.
- **Turn the copilot off for it** — not "be careful", off. Force-restore will eat working code.

That is four ungated rules to remember on every interaction — and `conventions.md` opens by
noting a rule that isn't enforced drifts, while the global rule is blunter: **a rule with no
gate is a wish.** Four ungated rules on a solo project over a year is not a working posture.

**Bottom line: the proof was worth building and its conclusion stands — no architecture is
needed. But the honest fix list is four small changes, not two, and item (3) is what converts
this from clever to safe.** It is still far cheaper than a node graph, and unlike the
discipline list, it holds without anyone remembering it.

## Second review: two further defects, and the timing argument

An independent reviewer arguing the opposite case (build the engine feature NOW) confirmed the
above and added two defects. Both reproduced first-hand here.

**Defect 6 — the texture round-trip is impossible even with the mkdir fixed.**

```
f2 8x8x4 -> save writes 512 bytes
reconstruct -> Error: data size mismatch 512 != 256
```

`save` writes raw `f2` bytes; `load_from_dir` reconstructs with no `dtype`, so `f1`. **The
project cannot be reopened.** Defect 3 is a crash on save; this is unrecoverable data loss one
layer beneath it. And since `checkpoint.py::snapshot_node` calls the same `save_ui_node` and
**swallows the exception**, the copilot's Revert cannot restore such a node and reports
"could not restore" only after the damage.

**Defect 7 — export corrupts the live preview.** `ProjectSession._make_export_isolation`'s
`finally` restores **only `on_pre_render`**. The fresh export behaviour has already written its
own texture into the live `node.uniform_values`, and nothing restores the previous value. After
any export the live preview reads a buffer the export owned and abandoned — and with
`gc_mode=None` those resources are never freed (GL names monotonic `[1,2,3,4,5,6]` vs
`[1,1,1,1,1,1]` under `"auto"`).

**The copilot probe measured, not argued.** On a node whose pass output demonstrably ramps with
`t`: `copilot motion probe |diff| = 0.0 -> verdict: STATIC`. With `cache_key` set,
`_render_facts_for` additionally emits *"this mutation changed NOTHING on screen... dead code,
the wrong node/target..."* — a confident, specific, wrong diagnosis. Roughly **12 of 31 tools
degraded, 7 actively lying.**

**The engine already forbids this in writing.** `prompt.py` states the contract verbatim:
**"A script writes VALUES only."** `_binding_reject` refuses sampler keys on purpose. The route
exists *only* because it evades that check via a side effect. This is not a grey area the docs
overlooked — it is a documented prohibition circumvented through a hole.

**The timing argument — the no-migration rule cuts toward building.** `CLAUDE.md`: "NO
backward-compatibility / migration code, EVER", whose sanctioned fix is hand-editing
`projects/dev/`. But a script-driven multipass node is **not hand-fixable** into a declarative
pass chain — its behaviour is imperative Python GL calls; there is no field to edit. Every such
node must be rewritten from scratch when a real feature lands, **and per defects 3+6 there is
no on-disk artifact to rewrite from** — only the script source, whose GLSL is a Python string.

**The deferral names a learning objective the activity cannot serve.** The open seam question
is "declaration in shader source vs `node.json`". A `script.py` **has no UI seam, generates no
controls, and bypasses `node.json` entirely**. Months of scripting would end with exactly the
four options `00_findings.md` already lists and no new evidence to choose between them.

Set against that, the reviewer concedes honestly: this is not a weekend. `dev_flow.md` triages
it as a feature AND high-blast-radius (upper-end review, spec-fidelity auditor, mandatory
sanitization sweep), the copilot needs a pass-aware probe or it inherits the same lying
verdict, and the seam decision is a real judgement call — `node.json` is app-written derived
state today, and making it hand-authored genuinely changes what the file means.

## Three fixes owed NOW, regardless of the verdict

Each is a latent defect in today's codebase, independent of multipass:

1. `ctx.gc_mode = "auto"` at context creation — every GL-touching script leaks on every save.
2. The `textures/` mkdir in `ui_models.py::save` — that branch crashes on first contact.
3. The missing `dtype` in `core.py::load_from_dir`'s texture reconstruction — the round-trip
   is broken for any non-`f1` texture.

## Revised bottom line

"Play with it now, decide later" is **too generous by one category**. Defects 3, 6 and 7 are
not unresolved concerns — they are a crash on save, unrecoverable data loss, and live-state
corruption after export. Playing with it now means **building things you cannot save, reload,
revert, or export**, in order to learn something the exercise does not teach.
