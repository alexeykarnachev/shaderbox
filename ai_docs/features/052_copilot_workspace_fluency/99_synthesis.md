# 99 — Synthesis: locked decisions, build order, open questions

Cross-cutting home for the umbrella. Per-slice detail lives in the slice docs; this is what binds them
and what the maintainer signs off at plan-lock.

## Goal (umbrella)

Make the copilot literate in the ShaderBox workspace (textures, canvas, node files) and give it ONE
safe, user-in-the-loop channel to the user's filesystem — closing the texture-blindness gap, the
file-manipulation gap, and the no-import gap, all without violating the actor model (no raw
shell/path access; the user picks every incoming file).

## Cross-cutting locked decisions

1. **Every incoming-file op routes through the shared user-file-pick primitive** (`pick_user_file`,
   defined in slice 2) — a **GATE-family** affordance (NOT a `bridge.run_on_main` op; round-1
   correction — a bridge op would freeze the frame loop for the unbounded dialog and hit the 5s
   timeout). The worker blocks on a file gate; the UI opens `pfd` and polls it across LIVE frames (the
   `widgets/uniform.py:246` pattern); the abs path is **consumed engine-side and never returned to the
   handler** (structural corollary-1, not a redaction the handler must remember). This is the
   credential-gate discipline generalized (`conventions.md` — the `GateKind.CREDENTIAL` bullet is the
   parent, and it too keeps the UI live). `bind_media` (slice 2) and `import_node` (slice 4) both use
   it — ONE primitive, two consumers.

2. **New capabilities feed facts back on the EXISTING channel** (the working-set rows + the tool
   result), never a new inspect tool the model must remember to call (skill §4). A bind shows up as a
   working-set row change; a canvas resize shows in the header. The model SEES the effect.

3. **Tool-count discipline is enforced by slice 0** — the 7 new tools ride the lazy catalogue, so the
   eager per-turn core stays lean. No new tool is eager unless it is per-turn-hot (none of these are).

4. **Media binds are FULLY revertable FOR FREE — no checkpoint code change** (slice 2 decision 3,
   round-1 correction). The existing full-`UINode.save` snapshot already serializes `media/`/`textures/`
   (`ui_models.py:289`) and restore is a whole-dir replace (`revert.py:18`), so media reverts already.
   The ONLY requirement is that every new mutating tool REGISTER with the checkpoint container
   (`_capture_node` on first touch; `duplicate_node`/`import_node` also `mark_created`) — a
   `30_turn_rollback.md` decision-2 obligation, named in each slice. `30_turn_rollback.md` decision-9's
   *justification* is updated (media is now touchable), but its mechanism is unchanged.

5. **The sandbox prompt line is updated, not contradicted.** The copilot still has no shell / no raw
   path / no arbitrary-disk read. It gains: seeing textures/canvas, manipulating node & lib files by
   handle, and opening a USER-driven picker to bring a file in. The "no filesystem beyond the tools"
   line becomes precise about the picker being a user-consented tool.

## Build order (the slicing map)

Independent lanes — the maintainer cuts tasks from these:

```
Lane A (infra):     Slice 0  lazy tool catalogue        [independent, land first]
Lane B (read):      Slice 1  awareness                  [independent, zero risk]
Lane C (headline):  Slice 2  bind_media + pick primitive [needs 0 for lazy, 1 for the visible result]
Lane D (files):     Slice 3  rename/dup/canvas/lib CRUD  [needs 0; each tool an independent task]
Lane E (import):    Slice 4  import_node                 [needs 0 + 2's pick primitive]
```

Recommended first cut: **Slice 1 (awareness) + Slice 0 (lazy)** — both independent, both derisk the
rest. Then **Slice 2** (the headline). Then **3 + 4** in any order.

## Files touched (rollup)

- `copilot/capabilities.py` — `WorkingSetView.canvas: str = ""` (defaulted); `MediaBindResult` (path
  consumed engine-side, NOT a `UserFilePick` with an `_abs_path` field); Protocol methods
  `pick_user_file`, `bind_media`, `unbind_media`, `rename_node`, `duplicate_node`, `set_canvas_size`,
  `delete_lib_file`, `import_node` (8 new Protocol methods = 7 model-callable tools + the shared
  `pick_user_file` primitive; `rename_lib_file` CUT).
- `copilot/backend.py` — all implementations + `_format_uniforms` sampler branch (default-vs-bound
  detection, NO path) + working-view `canvas`; every mutating tool calls `_capture_node`
  (duplicate/import also `mark_created`).
- `copilot/gate.py` + the gate-draw site (`widgets/copilot_chat.py`) — a `GateKind.FILE`: worker-block
  half + the UI `pfd`-poll-across-frames half.
- `copilot/tools/` — new `media.py` (+ maybe `node_ops.py`); extend `registry.build_registry`;
  `eager=False` on the new + the 7 cold integration tools (6 Telegram + 1 YouTube).
- `copilot/agent.py` + `copilot/tools/registry.py` — the lazy load path + SORTED tool serialization
  (slice 0).
- `copilot/prompt.py` — AWARENESS + MEDIA + NODES/import prompt notes; the lazy catalogue block; the
  updated sandbox line.
- **NO `copilot/checkpoint.py`/`app.py` media-scope change** (round-1: media revert already works via
  the full-node snapshot). `delete_lib_file` reuses the existing `snapshot_lib`/`_revert_lib_file`.
  Update `30_turn_rollback.md` decision-9 *justification* only.
- `tests/_caps.py` — a `_FakeCaps` field + `minimal_caps` default for EACH of the 8 new capability
  methods (else pyright Protocol conformance fails — `make check` red). MANDATORY, easy to forget.
- `tests/test_path_redaction.py` (new) — the automated corollary-1 guard (mirrors
  `test_credential_redaction.py`); `tests/test_working_set.py` — new defaulted-field constructors.
- `constants.py` — reuse `MEDIA_EXTENSIONS`; add `GLSL_EXTENSIONS`.
- Docs: `roadmap.md` row + banner; `conventions.md ## Design decisions` — a new bullet for the
  user-file-pick primitive (the credential-gate channel generalized to a path); update the parked
  `bind_media` (feature 020 `20_ui_ux_polish.md`) + `delete_lib_file` (`17_gate_ui.md`) entries as
  un-parked.

## Manual verification (umbrella invariants)

- **The corollary-1 guarantee, once:** across slices 2 + 4, no absolute path ever appears in the tool
  trace or the LLM history — only basenames. This is the load-bearing safety property of the whole
  "system interaction" theme; assert it with a grep in each slice's verification.
- **The rebuilt-every-step invariant holds for the new facts:** a bind / canvas resize / rename is
  reflected in the NEXT working-set build, never a cached string.
- Per-slice falsifiers are in each slice doc (each check fails for exactly one reason).

## Locked resolutions (maintainer: "full, solid solutions always — no half-features", 2026-07-02; amended after round-1 review)

- **O-1 → FULL, achieved for free.** A media bind is fully revertable — and the round-1 review found the
  existing full-node snapshot ALREADY captures/restores media, so "full" needs NO new checkpoint code,
  only the `_capture_node` registration every mutating tool owes (decision 4 above). Same outcome the
  maintainer asked for, less code.
- **O-2 → mechanism (a).** Explicit model-visible `load_tools(names)` meta-tool + a catalogue block.
  Auto-inject-by-intent rejected (hidden heuristic). (Serialization must SORT — round-1 fix, slice 0.)
- **O-3 → FULL umbrella.** Build all slices, no phased half-ship. The build-order lanes are the
  TASK-SLICING map, not a scope-reduction. Recommended sequencing stays 0 + 1 first (infra +
  zero-risk read) purely for de-risking.
- **O-4 → IN, re-scoped.** `unbind_media` ships as **"reset to default image"** (round-1: there is no
  empty sampler state — `core.py:333`). A coherent complete op, not the impossible "empty slot". One
  reviewer dissented (defer as low-demand); kept per the completeness mandate (Review history).
- **O-5 → CUT (reversed from "IN" after round-1 review).** `rename_lib_file` is dropped: it is
  DERIVABLE (`write_shader` new + `delete_lib_file` old) and had no revert path — shipping it would be
  the exact half-feature "no half-features" forbids. Cutting it HONORS the mandate rather than
  violating it. (See Review history for the reversal rationale.)

## Review history

### Round 1 (pre-implementation, 2026-07-02) — 3 adversarial reviewers, all code-anchored, all PARTIAL

Three reviewers (correctness-vs-code, verification-vs-blast-radius, devil's-advocate-vs-actor-model),
spawned in parallel, each anchored to the real source (not the spec's self-reasoning). All returned
PARTIAL with code-cited findings; two premises the spec was built on were REFUTED. Applied fixes:

- **[CRITICAL, 2 reviewers] "(no media bound)" state does not exist** — `core.py:333` defaults every
  sampler to `Image(_DEFAULT_IMAGE_FILE_PATH)`. Fixed: slice-1 detects default-vs-bound by comparing
  the source path; slice-2 `unbind_media` re-scoped to "reset to default" (O-4).
- **[CRITICAL, 2 reviewers] checkpoint already captures/restores media** — the planned
  `checkpoint.py`/`app.py` expansion was redundant; the real (unnamed) requirement is `_capture_node`
  registration. Fixed: decision-4/O-1 rewritten; the redundant work removed from Files touched.
- **[HIGH, 2 reviewers] the pick primitive was the wrong shape** — a `bridge.run_on_main` op freezes
  the frame loop for the unbounded dialog + hits the 5s bridge timeout. Fixed: re-shaped as
  GATE-family (worker blocks on a file gate; UI polls `pfd` across live frames), abs path consumed
  engine-side (structural corollary-1). `UserFilePick` → `MediaBindResult` (no `_abs_path` field, which
  leaks via repr/asdict into the trace).
- **[HIGH] `tests/_caps.py` omitted** — 8 new Protocol methods break `_FakeCaps` → pyright fails.
  Added to every slice's Files touched.
- **[HIGH → decision reversal] `rename_lib_file` cut (O-5 IN → CUT).** Derivable (write+delete) + no
  revert path = a half-feature. **Reversal rationale:** the maintainer's "no half-features" mandate
  means "each shipped capability fully works", NOT "complete every CRUD symmetry"; cutting a
  derivable, revert-less tool honors the mandate. Two reviewers independently recommended the cut.
- **[MEDIUM] corollary-1 leak via the working-set row** — the bound media's `details.file_details.path`
  is the full abs path. Fixed: slice-1 renders dims/kind only; added an automated `test_path_redaction.py`
  (the credential channel has one; the path channel now does too).
- **[MEDIUM] lazy-load "sorted" claim was false + the mid-turn cache-bust unmeasured.** Fixed: slice-0
  sorts the tool union + asserts byte-stability; added a warm-turn bust measurement; recorded the
  no-orphaned-pair invariant as verified-safe.
- **[LOW] wiring gaps** — `duplicate_node`/`import_node` need `mark_created`; `create_node` is
  positional-only with a required `template`; `delete_lib_file` revert re-creates on an absent path
  (impl-verify + falsifier). All named in the slice docs.

**Reviewer dissent kept (documented, per `dev_flow.md` step-4 "main agent decides"):** `unbind_media`
kept (one reviewer wanted it deferred as low-demand) — a re-scoped "reset to default" is a complete op
and the maintainer prefers completeness; its tool-count cost is paid by lazy loading.

**Not re-litigated (PASS from all three):** the awareness slice as corollary-2 exemplar; the
single-project / feedback / shared-pool / bundle-import deferrals as correct scoping (not half-features);
the backend-does-not-import-App boundary; content-addressed-edit and `<kind>:`-prefix rules honored.

### Round 2 (pre-implementation, 2026-07-02) — 2 code-anchored reviewers, both NOT-CONVERGED

Re-spawned against the patched spec. Confirmed round-1 fixes 2/4/5 correct (checkpoint-is-free, the
`_caps` obligation, the `rename_lib_file` cut — favorites/tags key on function name not path, so
write+delete truly composes). But found TWO round-1 fixes were themselves unsound (both code-cited by
BOTH reviewers — real, not late-round fabrication):

- **[CRITICAL] the slice-1 "(no media bound)" detector was broken for disk-loaded nodes.** `ui_models.save`
  persists EVERY sampler (incl. default) to `media/<uniform>`, and `core.py:181` reloads it from that
  path — so `details.file_details.path` is never `_DEFAULT_IMAGE_FILE_PATH` after a reload; default
  samplers would read as bound and unbind wouldn't round-trip. **Fixed:** move the fix to the SAVE
  boundary — `ui_models.save` SKIPS a still-default sampler (path==default is reliable in-memory at
  save time); load's `seed_uniform_values` re-seeds the default, so detection round-trips (verified
  against `core.py` load+seed). Flagged the app-wide save-path blast radius + its own falsifier.
- **[HIGH] the gate-family primitive would DEADLOCK as framed.** `GateKind.CREDENTIAL` is a PRE-execute
  gate (`agent.py` yields `AgentGateOpened` before `execute`; `session.py::take_pending` is the only
  drive site). The FILE gate is raised INSIDE `execute` (the handler calls `pick_user_file`), where the
  worker yields nothing → the existing drive never fires. **Fixed:** the FILE gate needs a NEW
  independent main-thread per-frame poll (not the `AgentGateOpened` path); the `GateRequest` carries
  `node_id`+`uniform` so main does the load+bind; the `pfd` cross-frame poll is NEW code (`uniform.py:246`
  is the BLOCKING `pfd_block`, corrected); added late-pick-on-cancelled-gate reconciliation.

Also fixed: tool-count drift (canonical = **7 model-callable tools + `load_tools` meta + `pick_user_file`
primitive = 8 new Protocol methods**; 7 cold integration tools); added the three missing falsifiers
(UI-stays-live-during-dialog, unbind round-trip, import_node revert).

**Confirmed-safe (no change):** the `bind_media` `_capture_node` revert falsifier genuinely proves the
wiring; turn-Stop wakes a gate-blocked worker via `cancel_all` (no indefinite hang).

### Round 3 (pre-implementation, 2026-07-02) — focused convergence check — CONVERGED

Re-verified the two round-2 fixes against the code. Both CONFIRMED-SOUND, zero new
implementation-breaking issues:
- **Detector (save-skip):** `ui_models.save` calls `seed_uniform_values()` before the save loop, so a
  never-bound sampler reliably holds `Image(_DEFAULT_IMAGE_FILE_PATH)` in-memory (the skip-check is
  correct); load re-seeds via `render()`→`compile()`→`seed_uniform_values()` STRICTLY AFTER the
  metadata loop, so a skipped sampler round-trips to `path == _DEFAULT_IMAGE_FILE_PATH`. No downstream
  consumer requires every sampler in `node.json`. `projects/dev` has ZERO sampler/media nodes → no
  stale-media migration to hand-fix.
- **FILE gate:** confirmed `take_pending()` has exactly ONE drive site (`session.py:195`, off the
  pre-execute `AgentGateOpened` yield), so a mid-`execute` gate genuinely needs the new per-frame poll;
  it is placeable in the existing main-thread `pump_events` drain; `cancel_all` wakes the block and the
  `answer()` `if pending is None: return` guard drops a late pick by construction.
- Non-blocking impl note (already in the gate.py files-touched): `GateRequest` gains `node_id`+`uniform`,
  `GateResponse` carries back the path-free `MediaBindResult` — straightforward additions.

**Convergence reached** (`review-agent-loop`: a late round finding nothing new is the closing
condition). Spec is ready to slice into tasks.
