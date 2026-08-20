# 060 — rot audit & architectural reorg

**Status:** done (2026-08-20)

## Goal

The project sat untouched for a while. Establish, with evidence, what had rotted and whether the
module/package layout is still right — then fix what the audit proved.

Method: a 10-agent read-only swarm in one wave (5 architecture lanes, 5 rot lanes; 6 opus / 4 sonnet
by whether judgement or retrieval was the deliverable). **Every finding was re-verified first-hand
before any code changed** — which mattered: three agent claims were wrong or incomplete, and one of
my own starting hypotheses was wrong too (see *Review history*).

## The architectural answer: no reorg

The headline question was "should we re-organize modules/classes/sub-packages?" The answer, from
three independent lanes, is **no** — and that null result is the finding:

- The intra-package import graph is **acyclic** (Tarjan over an AST-built edge set; the one SCC was
  an analysis artifact, confirmed clean at runtime).
- Zero `if TYPE_CHECKING`, zero inline function-body imports, zero `# noqa` in the package. The
  no-cycle conventions are holding structurally, not by discipline.
- `App` is confined to the UI layer — all 23 import sites are modules the module map sanctions.
- `app.py` (1377 L) and `ui_primitives.py` (1230 L) were both examined for a split and both **kept**:
  `app.py`'s residue is one ordered `__init__` sequence plus session forwarders whose tab/nav/popup
  clusters all read the same focus one-shots; `ui_primitives.py` is 47 stateless free functions whose
  call web crosses every candidate boundary (a split would make two private helpers public
  cross-module API and add imports without removing coupling). `conventions.md` had already ruled on
  `app.py`; the independent re-derivation agreed.
- Three hypothesized groupings (`render/`, `text/`, `ui/`) were each checked against the real graph
  and each dissolved.

Only three moves earned their churn, each deleting a real boundary violation:

1. **`copilot/edit_match.py`** (new) — the 8 pure text-matching/splicing functions (contiguous
   `backend.py:227-384`, AST-verified module-level and `self`-free) that are the Python half of what
   `glsl_lex.py` does for GLSL. Three test files were already reaching into `backend.py` to import
   them privately — the strongest evidence they wanted a public home. `backend.py` 2335 -> 2169.
2. **`render_job.py`** (new, top level) — `render_to`/`render_for`/`preset_ext` lifted out of
   `tabs/share_state.py`. These are GL-free, imgui-free render plumbing that the HEADLESS copilot
   core and the dogfood harness both call; the copilot reaching into a UI tab-state module was the
   only `copilot -> tabs` edge in the codebase. It is now gone.
3. **`integrations.py`** (moved out of `exporters/`) — `IntegrationsStore` holds Telegram + YouTube +
   **copilot** credentials, so it is a global credential store, not an exporter; it was the only
   `exporters -> copilot` package edge.

## Rot found and fixed

Ordered by consequence. Each fix carries a guard that was **mutation-tested** — reintroduce the bug
and the named test goes red.

1. **[CRITICAL] Silent credential wipe.** `IntegrationsStore` is `extra="forbid"`, and `load()`
   fail-softed a `ValidationError` to all-empty defaults; `App.save()` on quit then wrote those
   empties over the real tokens. Any retired key triggers it — including the `vision_enabled` key
   feature 058 removed, which `todo.md` explicitly warns may still sit on a dev box. Reproduced
   end-to-end. The sibling `UIAppState.load` already drops unknown keys before constructing; the
   store holding the *credentials* never got that treatment. Fixed with a recursive pruner that
   logs each dropped key by path. (The maintainer's real file was checked: no stale keys, creds safe.)
   Guard: `tests/test_integrations_store.py`.
2. **[HIGH] The copilot was told `ivec3` uniforms are `vec3`.** Two type-label producers fed the same
   consumer; `backend._uniform_type_label` collapsed every non-`GL_UNSIGNED_INT` type to `"float"`,
   so the project map advertised int uniforms as float ones and `set_uniform`'s reject message named
   the wrong type. Deleted in favour of `uniform_coerce.gl_type_label` (extended to carry the
   block/sampler branches). Guard: `tests/test_uniform_arrays.py`.
3. **[HIGH] `make test` had been red for four releases, and could not say so.**
   `test_revert_executor.py` was marked `pytest.mark.forked`; the `app` fixture leaves an open X11
   display socket, and `fork()` hands that socket to the child — two processes on one Xlib connection
   kills it (`XIO: fatal IO error`), so every test after the first `app` module died. The crash also
   took out pytest's own summary (`-rf` and `--junitxml` both produced nothing), which is *why* it
   stayed invisible. The marker was obsolete: the `glUseProgram(0)` bleed it defended against is
   suppressed at source in `core.py::release_program`, and `git merge-base` confirms that suppress
   landed *after* the marker. Removed -> **passing, order-independent, and faster**.
   A `[tool.pytest.ini_options]` block was added (there was none at all): `-ra`, `testpaths`,
   `xfail_strict`. **Correction from post-impl review:** the "and it could not report it" half of
   this finding was WRONG about the Makefile. A reviewer reconstructed the original recipe and
   showed pytest prints its summary under `-q` anyway and make already exited non-zero; my canary
   "verified" a wrapper that changed nothing. The `set -e` + `|| exit 1` wrapper has been reverted.
   What DID hide the failure is real and unchanged: the fork crash killed pytest before it could
   print any summary at all (`-rf` and `--junitxml` both produced nothing).
4. **[HIGH] The dogfood coverage metric was blind to 8 of its own tools.** `REACHABLE_TOOLS` was a
   hand-listed 15-name tuple used as the coverage DENOMINATOR, and its comment only justified
   excluding the 9 telegram/youtube/publish tools. The other 8 (`bind_media`, `unbind_media`,
   `rename_node`, `set_canvas_size`, `duplicate_node`, `import_node`, `delete_lib_file`,
   `load_tools`) were simply absent, so coverage read `N/15` forever and could never report a gap on
   a feature-052 tool. This is the `CLAUDE.md` "checker that narrows its own domain" family. Fixed by
   INVERTING the default: `REACHABLE = CANONICAL - _UNREACHABLE_IN_HARNESS`, so a new tool defaults
   INTO the denominator. Guard pins the exclusion set: `tests/test_tool_registry.py`.
5. **[MED] Unbounded ffmpeg inside the imgui frame.** `apply_temporal_smoothing` ran
   `subprocess.run` with no timeout from the "Apply" button; a hang froze the app, and a non-zero
   exit escaped `update_and_draw` and skipped `save()`/`release()` at `ui.py`, losing app state and
   the copilot conversation. Now bounded + typed (`MediaError`), reported as a toast. The sibling
   call site in `telegram.py` already did this correctly.
6. **[MED] Video export orphaned ffmpeg and left a truncated file.** No `try/finally` around the
   writer: a mid-export `render()` raise skipped `writer.close()` and left a partial `.mp4` that
   looks like a finished export. Now closes the pipe and unlinks the partial on the error path.
7. **[MED] Gate lost-wakeup could latch the copilot permanently.** `ask()` checked `_shutdown`, THEN
   published its slot; a `cancel_all()` landing between the two swept two empty slots and released
   nothing, and the worker blocked on an event nobody would set. On a project switch `_ensure_worker`
   then saw the thread alive and every later turn was queued and silently never ran. **Two rounds:**
   the first fix (publish + check under one lock) was caught by BOTH post-impl reviewers as
   insufficient — it closes the race only when `_shutdown` latches, i.e. `reusable=False`, while the
   Stop button and reset-conversation both call `cancel_all(reusable=True)` where nothing catches a
   late publisher. The landed fix adds a `_generation` counter bumped by every sweep: a worker
   samples it before building its request and re-checks under the lock, so a cancel that landed in
   between cancels this request too. Guards drive the exact interleaving for BOTH `ask` and
   `ask_file` (a plain thread race never lands in the window) and fail in 5s rather than hanging.
8. **[MED] Dead code drained.** `DEFAULT_DURATION`, `SIZE.CHEATSHEET_W`,
   `_ResolveState.flattened_lines` (never read AND never written), and `ToolRegistry.specs_for`
   (superseded by `assemble_specs`, which additionally sorts for prefix-cache stability — following
   the stale docs to `specs_for` would have reintroduced a cache-busting bug). The three
   `DEFAULT_TEMPORAL_*` constants were UNWIRED rather than dead — every consumer duplicated their
   literals — so they were wired up instead of deleted.
9. **[MED] The headless-core layering claim was false.** `import shaderbox.project_session` pulled in
   both imgui_bundle and glfw via four chains, contradicting `dev_flow.md` and `conventions.md`. Two
   chains were cheap and are now closed: the nav enums moved to `ui_regions.py` (so the persisted
   model layer no longer drags in the imgui-evaluating command table), and `core.py`'s single
   `glfw.get_time()` fallback became `time.monotonic()`. The other two are structural
   (`shader_lib/file_ops.py` genuinely owns UI state; `copilot/backend.py` genuinely drives the
   exporters) — purging them is a large refactor with real risk and no user-visible payoff, so the
   DOCS were corrected instead to state the invariant that actually holds and that the harness
   depends on: **no imgui context and no glfw window is created at import** (verified).

10. **[MED] `Node.release()` leaked every uniform-held resource.** It freed the program and canvas
   but never touched `uniform_values`, which owns the `Image`/`Video` bound to each sampler (a
   texture each, plus an open `cv2.VideoCapture` for a video), the default `Image`, and the
   uniform-block `Buffer`. One leak per reload — and the file watcher reloads on every external
   `node.json` touch, plus one per revert and per project switch (which also leaked the examples).
   Verified nothing shares those objects (`load_from_dir` and `duplicate_node` both build fresh
   ones), so releasing them is safe. Guard: `test_release_frees_uniform_held_resources`.

## Doc harness fact-check

A lane checked 118/118 module-map claims, 61/61 `conventions.md` claims and 60/60 roadmap `Spec:`
pointers. Corrected: the eager-tool list (all 6 telegram tools + `set_youtube_credentials` are LAZY,
while `probe_render` + the script trio are eager and were unlisted); `App.restore_checkpoint` ->
`RevertExecutor.restore_checkpoint`; "feature 030" (never existed) -> `020_copilot_agent/30_*`; the
"four modal popups" (five since 055 shipped HELP); `_plain_text_spans` (never existed) and its
now-false "exact-match / whitespace-tolerance is unsafe" rationale; a broken section anchor; the
`/review-agent-loop` skill (it is `agent-spawn-discipline`) cited twice as the canonical pointer; the
"Shader Library" menu (it is "Library"); the lab-worktree note (per-machine, may not exist); the
Makefile's `smoke` comment (it uses a throwaway tmp project, not `projects/dev/`). Eleven modules had
no module-map row at all and now do.

**Release-line correction:** the banner claimed v0.21.0 was last live on itch; `master` is tagged
**v0.25.0**. v0.22.0's commit records "itch upload deferred" and no later commit records an upload,
so the banner now states both facts rather than guessing which is public.

## Out of scope (deferred, with triggers)

- **`exporters/telegram.py` worker-half split** (1283 L -> ~780 + a `telegram_api.py`). Real, but the
  reviewing agent's supporting detail was partly fabricated (it claimed three `# ----` banners "already
  mark the seam" — the file has none) and the coupling is slightly wider than reported (`_with_bot`
  also touches `self._push_event`). Not free, and this wave had better-evidenced work.
  **Trigger:** next time a worker-thread bug is fixed in `telegram.py`, or when `youtube.py` grows its
  own copy of the same machinery.
- **Exporter worker-thread machinery duplicated** across `telegram.py` and `youtube.py`
  (`_ensure_worker`/`_enqueue`/`_push_progress`/`_push_event`/teardown, plus two sentinels declared
  twice). `conventions.md` documents this teardown as a contract with a known hang mode, and BOTH
  copies set `daemon=False`. **Trigger:** the same split above — do them together.
- **5 wired-but-unfalsifiable copilot brakes.** `bulk_gate_threshold`,
  `copilot_working_set_max_nodes`, `auto_revert_after_failed_edits`, `clean_edit_hard_streak` and
  `_MOTION_EPS` each have a real reader, but the suite stays fully green when the brake is disabled.
  (Contrast: `max_iterations`, `max_edit_retries`, `max_compile_failures` and 6 others DO go red.)
  **Trigger:** before plan-locking any feature that changes a brake's semantics — add the falsifier
  first. Cheapest durable form is one parametrized test walking each config field.
- **Two node-handle resolvers with different ambiguity rules.** `tools/shader.py::_resolve_node_name`
  takes the first prefix hit (and matches in both directions) where `backend._copilot_resolve_node_id`
  requires a UNIQUE prefix; the loose one names the node in the destructive-delete confirmation.
  **Trigger:** first user report of a delete gate naming the wrong node, or the next edit to either.
- **On-disk node filenames re-spelled beside their own constants** — `"node.json"` has no constant at
  all (5 sites); `sync_nodes_from_disk`'s half-written-node guard uses its own literals, so a rename
  would break the typed loader loudly and the watcher's guard silently. **Trigger:** next change to
  the node-dir layout. Fix shape: `ProjectPaths` gains `node_json_for`/`shader_for`/`script_for`.
- **Dependency bumps.** `opencv-python-headless` (4.13 -> 5.0) and `openai` (2.41 -> 3.3) are each a
  major behind with unbounded `>=` floors; `imgui-bundle` 1.92.801 -> 1.92.900 is held deliberately
  (the `imgui-ui` skill §8 carries ~20 version-conditional workarounds that CI cannot check).
  Upgrades are not rot-fixing and each needs a manual render/copilot check. **Trigger:** when
  something actually needs one, or at the next ship — take the routine tier as one sweep.
- **`backend.py` remains 2169 L / 29 injected callbacks.** The class is a wide facade over one
  coherent state (the read, edit, script and node-creation paths all write the same bookkeeping, and
  the `CopilotCapabilities` Protocol requires all 43 methods on one object), so a split is a
  deliberate decision, not a drive-by. **Trigger:** when a 5th distinct domain wants in.

## Files touched

New: `shaderbox/copilot/edit_match.py`, `shaderbox/render_job.py`, `shaderbox/ui_regions.py`,
`tests/test_integrations_store.py`. Moved: `exporters/integrations.py` -> `integrations.py`.
Modified: `copilot/{backend,gate,tools/registry}.py`, `core.py`, `media.py`, `uniform_coerce.py`,
`constants.py`, `theme.py`, `ui_models.py`, `app.py`, `ui.py`, `commands.py`,
`shader_lib/resolver.py`, `tabs/{share,share_state}.py`, `widgets/{media_ops,node_grid}.py`,
`exporters/*`, `project_session.py`, `scripts/{smoke,token_probe,dogfood/{analyze,harness}}.py`,
`Makefile`, `pyproject.toml`, 6 test files, and the doc harness.

## Manual verification

Not required for this wave — no user-visible UI change landed. The one behavioural change a user
could see is the smoothing-failure toast (was: an app crash that skipped the shutdown save).
Automated gates cover the rest: `make check` green, `make test` **670 passed** (was 6 failed + 1
error) and order-independent, `make smoke` OK (200 frames, 6 nodes).

**Verified-by-falsifier** (each mutation-tested — bug reintroduced, named test goes red, restored):
credential wipe -> `test_integrations_store.py` (3 tests, incl. the `packs` list case); `ivec3`
label -> `test_int_family_labels_are_not_collapsed_to_float`; coverage denominator ->
`test_dogfood_coverage_denominator_holds_every_tool_but_the_named_exclusions`; gate race ->
`test_ask_racing_cancel_all_never_blocks_forever` AND
`test_ask_file_racing_cancel_all_never_blocks_forever` (both fail in 5s, neither hangs); resource
leak -> `test_release_frees_uniform_held_resources`; suite pollution -> re-adding the `forked`
marker reproduces 6 failed + 1 error.

**Knowingly unguarded** (stated, not hidden): the ffmpeg timeout and the video-writer cleanup have
NO automated coverage — a reviewer reverted both and the suite stayed green. Both need a real ffmpeg
failure to exercise; the honest status is "fixed, untested". **Trigger:** if either regresses, the
fix is a fake-ffmpeg fixture, not another manual check.

## Review history

Corrections made to agent findings before acting — recorded because the pattern repeats:

- **My own first hypothesis was wrong.** I attributed the test pollution to a module-scoped `gl_ctx`
  fixture calling `ctx.release()`. Removing that release did NOT fix it; the standalone-context
  modules are innocent. The test lane re-derived it independently and landed on the same real cause I
  did on re-testing (the `app` fixture's X socket + `fork()`).
- **The fat-module lane fabricated a supporting detail** — "three `# ----` banners already mark the
  seam" in `telegram.py`; the file has zero. Its coupling list was also incomplete. The split it
  proposed is still real, but was deferred rather than acted on from a bad citation.
- **The dead-code lane mislabelled `theme.py`'s `SYN_*` block as unwired rot.** Feature 006 §5 records
  it as a DELIBERATE decision: imgui-bundle's `TextEditor.Palette` is read-only from Python, so the
  custom palette cannot ship. Re-checked against the installed 1.92.801 — still only `.get`, no
  setter — so the tokens correctly stay; only the misleading "palette wiring" comment was corrected.
- **The copilot lane undercounted** the invisible dogfood tools (7; it is 8 — `load_tools` too).
- **Post-impl review caught my own gate fix as insufficient, twice over.** Both reviewers
  independently built a falsifier proving the first fix left the `reusable=True` path (the Stop
  button) still able to latch the copilot permanently — the exact defect class this wave was
  convened to kill, shipped inside the fix for it. A third gap (`ask_file` untested) came from the
  guard-audit lane. All three are closed above.
- **Post-impl review also proved one of my findings partly FALSE.** I claimed `make test` "could not
  report" its failures and added a Makefile wrapper for it; the reviewer reconstructed the original
  recipe and showed it already named the failed tests and exited non-zero. The wrapper was reverted
  and finding 3 corrected. Lesson: I verified the canary against the NEW Makefile only — a control
  run against the OLD one would have caught it immediately.
- **The topology lane's `integrations.py` move reverses feature 017 §11**, whose stated premise
  ("4 of its 5 importers are under `exporters/`") still holds today, 4-vs-2. Kept anyway: a
  package-level dependency inversion outweighs a weak majority-by-count, and the store is not an
  exporter. Recorded here because it is a genuine trade-off, not a clear win.

## Open questions for the user

None — the maintainer delegated the full flow end-to-end with no approval gates.
