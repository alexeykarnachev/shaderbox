# 062 — final design hunt

**Status:** done (2026-08-21)

## Goal

Answer one question with evidence: **is the codebase ready for new feature work?** 060 audited the
module layout (null result), 061 swept contracts and persistence (14 fixes). This is the third and
LAST audit wave — its other job was to establish whether auditing had hit diminishing returns.

Method: a 12-agent workflow — 8 sonnet finders, each anchored to a distinct concrete artifact and
required to demonstrate by execution, then an opus skeptic per finding who had to reproduce it
first-hand and defaulted to `real: false`, then an opus readiness judge. 19 raw findings, **16
confirmed, 3 rejected**; the skeptic RAISED severity on 2 and lowered it on 1 — a calibrated swarm,
not a padded one. Every finding was then re-verified by me before any code changed.

Every finder was told what 060/061 had already settled and instructed to drop anything ruled on
there, and told that "I found nothing" was a valid, valued answer. That framing is why the yield is
signal: **zero architectural findings, third wave running.**

## The verdict

**READY TO PROCEED**, after the fixes below (about one day's work, all landed here).

Not "the codebase is bad" — the opposite. The architecture is settled and untouched for a third
consecutive audit; the import graph is acyclic (114 modules, 420 edges, zero cycles). Every confirmed
defect was in LEAF code: a loader, a closure, an indent calculation, an index clamp.

## The finding underneath the findings

Three of the confirmed defects were the SAME class 061 had already fixed, at sites 061 did not sweep:

- `model_salvage` existed, was correct, and had **two callers when it needed four**.
- `render.py` carried a comment explaining a capture hazard that `share.py` still had.
- `test_node_dir_layout` guarded **two of the node dir's three** basenames while advertising the
  class as closed.

So the wave-2 fixes were right but not swept to completion. That is grep-shaped work, not discovery —
which is the argument for stopping. **The countermeasure is built here** (see the completeness
battery), not deferred to a fourth wave.

## Fixed

Each carries a mutation-tested guard.

1. **[CRITICAL] `drop_invalid` did not recurse — one bad nested value wiped the credentials.**
   In the module written to prevent exactly that, whose docstring states the rule. `drop_unknown`
   recursed; its twin did not, so a nested block was validated whole and one malformed pack row
   rejected the entire `telegram` block — `{}` — which the next quit wrote over the real token.
   Reproduced first-hand. **This was code written earlier the same day in 061.**
2. **[CRITICAL] The shader-lib sidecar stores crashed the app at startup.** `ShaderLibTagsStore` /
   `ShaderLibFavoritesStore.load()` caught only `(OSError, JSONDecodeError)`, so a `null` or `int`
   row raised `TypeError` out of `ProjectSession.__init__` — the app would not start, naming no file.
   These live in `app_data_dir()`, outside git, with no backup. Now salvaged per row.
3. **[CRITICAL, found by the new battery] `IntegrationsStore.load` crashed on a non-dict file.**
   Not from the swarm — the completeness battery written in step 10 found it on its first run, at a
   site nobody had swept. Guarded at the shared helpers so every caller inherits it.
4. **[HIGH] Video seek death-spiral.** `Video.update` demanded an exact +1 frame gap and took a
   random-access seek for anything else, including a +2. Measured on the shipped example: a seek is
   **4.68ms vs 0.09ms** for a forward grab — 52x. Below the video's own fps the seek rate goes from
   ~0% to ~100% in a cliff, and it is self-reinforcing (the seek lengthens the frame, the longer
   frame widens the gap, the wider gap forces the next seek). Bounded forward-grab: **15.6x faster,
   seeks 200 -> 7** over the same workload, pixels byte-identical at every gap on both sides of the
   bound.
5. **[HIGH] The Share tab captured the node OBJECT across a frame boundary.** `render.py` captures
   the node ID and re-resolves, with a comment explaining why; `share.py` was the missed half, so a
   delete or project switch in the deferred frame encoded a released node — a black artifact,
   published with no error.
6. **[HIGH] `splice_script` double-indented the first line**, so the copilot wrote syntactically
   invalid Python on the exact path the indent-forgiving fallback exists to serve. Its existing guard
   was VACUOUS: a substring assertion that the broken 24-space output also satisfies. **The first fix
   was wrong too** — my tightened assertion caught it (stripping the shift left the replacement's own
   indent); the rule is that the replacement's first line contributes no leading whitespace.
7. **[HIGH] Closing a tab left of the active one silently switched the editor to a different file.**
   `active_tab_index` was only CLAMPED, which keeps the index valid but not the identity, so
   `flush_current_editor()` then flushed the wrong tab on Ctrl+S and on quit. Both removal paths now
   re-anchor on the tab's identity through one helper.
8. **[HIGH, found while fixing #4] `Video.texture` ignored `grab()`'s return value.** A capture at
   end-of-stream retrieved `None` and crashed inside `cvtColor`, taking the whole node load with it.
   Forward-decoding makes that position reachable, so it went from latent to a **~25% flake** — which
   is how it surfaced. (I had seen this error once earlier and written it off as noise; that was
   wrong, and the stress run is what corrected it.)
9. **[MED] Render/Share failures were invisible.** A failed render logged to a file the user never
   opens while the "Rendering..." cue cleared normally. Worse on the Share side:
   `artifact_is_fresh` is set ONLY by `set_artifact`, so the early return left the previous render's
   `True` in place — and the publish buttons gate on that flag, so a user could publish a stale
   artifact believing it was the new one. Now toasts and clears the flag.
10. **[MED] The `script.py` basename was re-spelled at 6 sites** beside a module-private constant,
    including the copilot's revert and checkpoint paths. Moved to `paths.py` with its siblings.
11. **[MED] Five guards over correct code that no test observed.** Each survived a mutation with 700+
    tests green: `Node.release` freeing the canvas, `invalidate()` freeing program/vbo/vao,
    `Canvas.set_size` freeing the old texture, its same-size no-op guard (without it, a per-frame
    call reallocates the canvas every frame), and `snapshot_script`'s first-touch-wins (breaking it
    makes Revert restore the copilot's mid-turn draft over the user's pre-turn script, reporting
    success). Finding the right release probe took a measurement: `.glo` keeps its stale id, but
    `mglo` becomes `InvalidObject` for every object type.

## The countermeasure: a completeness battery, not a rule

`tests/test_persistence_completeness.py` drives EVERY persisted store against one corruption battery
(truncated, non-JSON, empty, list, scalar, null, nulled values, wrong types, retired keys), and a
roster check enumerates the modules that read JSON and fails if one is neither rostered nor
explicitly exempted with a stated reason. A new store inherits every case without anyone remembering
what "the persistence rules" were.

It found defect #3 on its first run. That is the whole argument: **the class of defect this wave
surfaced is what a completeness test catches for free, forever** — prose asks you to remember to
check; a test asks by itself.

## What came back CLEAN (the null result)

Checked with evidence and found sound — recorded so a future wave does not re-litigate:

- **Architecture.** Third independent audit, third no-reorg. Import graph acyclic. Closed.
- **GL resource lifetime**, everywhere it is exercised: node release, hot-reload, canvas resize,
  project switch (4 switches, flat at 56 objects), media bind/unbind, export including the FAILURE
  path, double-release. The 060 uniform-release fix was independently confirmed load-bearing.
- **The copilot loop.** All 6 brake mutations went red on well-named tests; gate protocol,
  Stop-during-gate, worker reuse, checkpoint orphan sweep, tool error boundary (no secret leak) and
  credential redaction (detects PARTIAL weakening) all verified.
- **~30 documented invariants** mutation-tested and holding.
- **The frame loop**: flat in node count with Render-all off; the alarming-looking per-frame
  glob/stat traffic is ~2ms of a 119ms frame.
- **Zero skipped tests**, `make check` clean, 6 `type: ignore` all on the sanctioned allowlist.
- **Rejected by the skeptic:** the mid-turn project-switch race (guarded by `_copilot_busy_blocked`,
  already ruled on in 022 amendment A1) — the best-argued finding of the wave, and wrong.

## Explicitly NOT done

- **`exporters/telegram.py` worker-half split.** The shared worker machinery is already extracted;
  the rest is a file-size argument. **Trigger:** unchanged.
- **`backend.py`'s size.** Re-derived independently a third time; the 060 verdict holds.
- **Dependency bumps.** Not rot-fixing; each needs a manual render/copilot check.
- **The degenerate-frame freeze during an in-flight copilot turn.** Real, but needs an external
  `git checkout` landing inside an LLM turn. **Trigger:** next edit to `ui.py`'s degenerate-frame path.
- **Delete-arm persistence.** LOW, cosmetic.

## Is a fourth wave worth it? No.

Zero architectural findings in three waves. Every defect now in leaf code. Five of the confirmed
findings were missing TESTS over correct code, and three were missed sweep sites from the previous
wave. That shape says the discovery phase is over: what remains is mechanical, and the completeness
battery now catches it automatically. **Go build features.**

Suite: 713 -> **792 passing**, `make check` + `make smoke` green.
