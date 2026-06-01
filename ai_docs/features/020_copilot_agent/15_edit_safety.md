# 020 · 15 — copilot edit-safety: editor lock + source-freshness guard

A mid-sized safety feature with two halves that share one concern — **the copilot must only ever edit
the actual current shader source, never a stale or wrong-target snapshot.** Grounded in a deliberate
two-agent research sweep (lifecycle/guard-surfaces + freshness design-space); the findings are folded
into the Design decisions below.

## Background — the exposure (verified, not assumed)

The copilot worker reads the shader (`get_current_shader` → `_copilot_current_shader_view`, a main-thread
bridge round-trip) and later edits it (`edit_shader`/`replace_lines`/`insert_after` →
`_copilot_apply_*`, separate main-thread round-trips). The worker blocks on the LLM *between* those two
round-trips, while the main frame loop keeps running. Four things can move the source in that gap, and
**none is guarded today**:

1. **User types into the editor mid-turn.** The editor is fully editable while the copilot thinks. The
   `App.copilot_turn_active` flag exists and is reconciled every frame (`ui.py`) — but is **read by
   nothing**. The `# locks the editor read-only (§11)` comments in `app.py` prove the lock was always
   intended; it was scaffolded and never wired.
2. **User Ctrl+S mid-turn** → `App.save` → `flush_current_editor` → `Node.release_program` (+ `UINode.save`
   writes the file). Not gated.
3. **The mtime watcher reloads the file mid-turn.** `ui.py::_reload_if_changed` runs EVERY frame,
   main-thread, unconditionally; an external disk change (another editor, `git checkout`, file-sync)
   calls `release_program(new_text)` and replaces `node.source` under the agent. **The editor lock does
   NOT close this** — the lock governs the in-app TextEditor, not the disk.
4. **User switches the active node mid-turn** (grid mouse-click is reachable even while the chat is
   focused; also create/delete/palette). The caps closures read `self.current_node_id` LIVE at edit
   time, so the agent's next edit lands on the NEW node — a silent **wrong-file** edit, the scariest
   vector.

Plus a cross-turn case the lock can't help with: the user edits+saves while the copilot is IDLE, then
sends a turn, and the agent edits without re-reading (it reasons from training-data recall of a shape it
never read this turn). The "read before editing" rule is **prompt-only / soft** today
(`prompt.py` + the tool descriptions); nothing enforces it.

---

## Goal

Two halves, one feature:

**A — Editor lock (close the in-app in-turn vectors 1, 2, partly 4).** While a copilot turn is in
flight, the code editor is read-only and Ctrl+S is a no-op; the active node cannot be switched and nodes
cannot be created/deleted. Wires the dead `copilot_turn_active` flag through to real enforcement.

**B — Source-freshness guard (close vectors 3, 4-detection, and the cross-turn case).** The agent can
only apply an edit to a shader whose `(node_id, content)` is unchanged since it last `get_current_shader`
this turn. A mismatch (or a never-read this turn) rejects the edit, mutates nothing, and tells the agent
to re-read — distinguishing "you switched nodes" from "the source changed under you". This must exist
**independent of A**, because the every-frame mtime watcher (vector 3) is outside the lock's reach.

The two halves are complementary, not redundant: A prevents the in-app mutations; B is the backstop for
what A can't reach (disk reloads) plus the cross-turn case, and it subsumes the soft read-first rule
(never-read = no stamp = reject).

---

## Out of scope (each with a trigger)
- **A bespoke "the file keeps changing on disk" giveup.** If an external process churns the file every
  frame, each stale-reject correctly fires and the turn re-reads; the existing `max_iterations` ceiling
  bounds the loop (it surfaces as the normal max-iterations `AgentError`). **Trigger:** a trace shows a
  real turn burning iterations on a churning file — then add a distinct giveup.
- **Locking lib-file editing during a turn.** The copilot only edits the current NODE shader (the caps
  closures key on `current_node_id`, never a lib path). The editor lock applies to whatever session is
  on screen (node or lib) since the user shouldn't edit *anything* while a turn runs, but the freshness
  guard is node-only. **Trigger:** the copilot gains a lib-file editing tool (a Slice-2 deferral).
- **Deferring/queuing a mid-turn disk reload to apply after the turn.** The research noted the watcher
  *could* defer the root-reload while `copilot_turn_active`. Rejected for this feature: the freshness
  guard already makes a mid-turn reload SAFE (the next edit rejects), and deferring adds watcher state +
  a "apply on turn end" path. The reload happening mid-turn is harmless given B. **Trigger:** a maintainer
  finds the mid-turn reload's editor `set_text` visibly disruptive (it would flash the user's external
  change into the read-only editor mid-turn).
- **Persisting the locked state across a crash / unclean turn end.** `in_flight` is reset in
  `_finish_turn` on every terminal event (done/error/cancelled); a worker that never returns is the
  pre-existing "thinking forever" problem, out of scope here. **Trigger:** the worker-hang case is
  addressed (it has its own deferral).

---

## Design decisions
*(numbered, lock-in only; the research-grounded ones cite the finding)*

### Half A — editor lock

1. **The lock is read from `app.copilot_turn_active` every frame in `tabs/code.py::draw`, on the active
   session.** Right before `editor.render(...)`, call `editor.set_read_only_enabled(app.copilot_turn_active)`.
   `TextEditor.set_read_only_enabled(bool)` is the real upstream API (verified present in imgui-bundle
   1.92.801). Set it every frame (it latches, but the active session can change between frames, and the
   flag flips at turn boundaries) on the ONE session `code.py::draw` resolves per frame — only the active
   session renders/is editable, so the others need no lock. **Why there's no editable gap (corrected
   per review — the mechanism is the latched flag, not the reconcile order):** `copilot_turn_active` is
   set True SYNCHRONOUSLY in `copilot_send` (which fires from the chat widget draw, AFTER `code_tab.draw`
   already ran that frame — but the user just clicked Send in the chat, so the editor cannot also receive
   a keystroke that same frame). From the next frame on, the flag stays True for the whole `in_flight`
   span (the per-frame reconcile reads `in_flight`, set True in `enqueue_turn`). At turn END the reconcile
   DOES run before `code_tab.draw` (`pump_events`→`_finish_turn` sets `in_flight=False`, the reconcile
   reads it, then the editor draws unlocked) — so the end is gapless too. Note a copilot edit lands via
   `drain_bridge` (early in the frame) calling `sync_editor_from_disk`→`editor.set_text` — that's a
   PROGRAMMATIC write, which `set_read_only_enabled` does not block, so the copilot's own edits are
   unaffected by the lock (correct).

2. **Ctrl+S is gated: `App.save` early-returns while `copilot_turn_active`.** A read-only editor stays
   clean (the `copilot_send` flush drained it at turn start), so `flush_current_editor` is a no-op during
   the turn — but `App.save` also writes the node file via `UINode.save`, a separate write that could race
   the copilot's own `write_text`. Early-returning `App.save` (with a brief notification "editor locked
   while the assistant is working") is the explicit, robust guard. The SAVE command stays bound; it just
   no-ops with feedback during a turn.

3. **Node selection + create/delete + project-open are frozen during a turn (resolves OQ2 — robust
   default: prevent, don't just detect).** Gate the FOUR node verbs `App.select_node` / `delete_node` /
   `create_node_from_selected_template` (and `open_project`) with an early-return + notification while
   `copilot_turn_active`. **Do NOT gate `set_current_node_id` alone** (review BLOCKER): `delete_node`
   and `create_node_from_selected_template` do their destructive/creative work and only THEN call
   `set_current_node_id` to repoint — gating the setter alone would leave `current_node_id` dangling at a
   deleted/missing id (→ a later `KeyError`). Gate the verbs, not the chokepoint. `open_project` is added
   because it reaches `_init`→`_seed_starter_node`→`set_current_node_id` and `reset_conversation`; the
   review flagged that `reset_conversation` cancels the agent LOOP but an already-dispatched `run_on_main`
   closure could still execute against half-swapped state — gating `open_project` during a turn closes
   that. (`_seed_starter_node` itself is boot-only inside `_init`; not separately reachable.)
   Rationale: a mid-turn switch is the silent WRONG-FILE vector (reachable by grid mouse-click even while
   the chat is focused). The freshness guard's identity-half (B) stays as defense-in-depth.
   **UX (review SHOULD-FIX):** the repo's established "busy" idiom is `imgui.begin_disabled()` around the
   widget (exporters, render, node_creator all use it; the copilot chat disables Send while `in_flight`).
   The verb-gate is MORE robust (catches palette/menu/hotkey paths the visual can't), but less
   discoverable. So ALSO wrap the node-grid cells (`widgets/node_grid.py::draw_node_preview_grid`) in
   `begin_disabled()` while `copilot_turn_active` for the visual cue — verb-gate for correctness, the
   disabled visual for affordance. **OQ2 flagged** — alternative (allow switching, let B reject) is
   simpler but wastes a tool call and is worse UX.

### Half B — source-freshness guard

4. **The "last read" revision lives App-side as `(node_id, content_hash)`, written/read only inside the
   bridge `_on_main` closures (research axis A → option b).** A single field on `App`:
   `self._copilot_read_revision: tuple[str, bytes] | None` (the hash is a `bytes` digest, D5 — NOT an
   `int`). `_copilot_current_shader_view`'s `_on_main` STAMPS it on every read; the mutating `_on_main`s
   COMPARE against it.
   Because every access is inside a `run_on_main` closure, the field is touched only on the main thread —
   automatically serialized with the mtime watcher and node switches, no lock needed. The copilot package
   never sees the field (leaf-seam intact). Rejected: loop-local in `run_turn` (fights the registry's
   tool-agnostic seam — the agent would have to know shader reads produce a token shader mutates consume);
   token-in-tool-I/O (the model can forget/fake/reuse-stale it — the prompt even tells it to reuse read
   results).

5. **The token is `(node_id, hash(node.source.text))` (research axis B).** Node IDENTITY catches the
   switch vector; content HASH catches the disk-reload + cross-turn vectors. `ShaderSource` is a frozen
   dataclass whose `mtime` is NOT reliable for in-memory edits (`release_program` keeps the stale mtime),
   and a monotonic epoch would need every writer to remember to bump it — so the content hash (derived
   from the bytes that matter, impossible to desync) is the correct key. Use a stable digest of the
   UTF-8 bytes (`hashlib.blake2b(text.encode(), digest_size=16).digest()`) rather than builtin `hash()`
   — reads as obviously-correct and is salt-free. Collision risk is non-existent for shader-sized text;
   state it and move on. The revision is computed by a MODULE-LEVEL free function
   `_shader_revision(node_id: str, text: str) -> tuple[str, bytes]` in `app.py` (alongside `_edit_result`
   / `_number_lines` / `_splice` — the existing free-function cluster), NOT an `App` method (it uses no
   `self` → the no-`@staticmethod` / self-less-method rule in `conventions.md ## Code rules`). Do NOT
   pattern-match on the pre-existing `App._create_dir_if_needed` `@staticmethod` — that's a known
   violation, not a model.

6. **The check is enforced inside the mutating `_on_main`s BEFORE any `ui_nodes` access, distinguishing
   two drift kinds.** ORDER IS LOAD-BEARING: the freshness check runs **before** the existing first line
   `node = self.ui_nodes[self.current_node_id].node`. B must work even when A is absent, and in the
   B-without-A scenario `current_node_id` may be `""` or a just-deleted id — the unguarded dict index
   would `KeyError` instead of producing the clean reject. So the closure first binds `nid =
   self.current_node_id`, then checks the stamp:
   - stamp is `None` (never read this turn) → reject: "call get_current_shader before editing."
   - `nid` not in `self.ui_nodes`, OR stamp `node_id` != `nid` → identity reject: "you read shader <X>
     but the active shader is now <Y> (or none) — call get_current_shader again."
   - same node, hash differs → content reject: "this shader's source changed since you read it — call
     get_current_shader again and re-match."
   ONLY after the stamp passes does the closure dereference `self.ui_nodes[nid].node`. Reject = mutate
   nothing → `EditResult(matches=0, stale=True, stale_reason=<specific message>)`. After a SUCCESSFUL
   apply, the closure re-stamps `_shader_revision(nid, new_text)` so the agent can chain edits without an
   intervening read (matches today's behavior). Multi-span `replace_all` re-stamps on the whole new
   source string (correct — the entire post-edit source is the key).

7. **The stale-reject is surfaced via a typed marker so the retry cap ignores it (resolves OQ1 — robust
   default: do NOT count it).** `EditResult` gains `stale: bool = False` and `stale_reason: str = ""`
   (both leaf primitives). The App handler builds the SPECIFIC reason (D6, where the `node_id`s are
   known); the tool layer, on `result.stale`, returns `ok=False` with `result.stale_reason` as the
   message AND `payload={"stale": True}`. `run_turn`'s consecutive-failed-edit counter changes from
   `is_mutating(name) and not ok` to ALSO require the result is not stale. **BLOCKER fix (from review):**
   `payload` is `None` on a malformed-args / unknown-tool / handler-exception result (`registry.execute`
   returns `(False, msg, None)`), so the predicate MUST null-guard: `is_mutating(tc.name) and not ok and
   not (payload or {}).get("stale")` — `payload.get(...)` on a bare `None` would `AttributeError` and
   crash the worker on the first malformed mutating call. A stale-reject is a benign "re-read and
   continue", not the convergence failure the cap names. The agent must not parse the human string, so
   the marker rides the existing `payload` channel (parallel to `payload["errors"]`). **OQ1 flagged** —
   cost is one boolean on the agent-visible payload; the fallback ("let it count") is acceptable since a
   re-read (non-mutating tool) resets the counter anyway.

8. **The stamp resets to `None` at each turn boundary** (in `App.copilot_send`, where
   `copilot_turn_active` is already set, or at worker turn-start). This forces a re-read at the start of
   every turn — catching the cross-turn case (user edited+saved while idle) and making the soft
   read-first rule a hard check (never-read this turn = None stamp = reject, D6 first bullet).

9. **The read-first prompt text STAYS** (complementary, not redundant). The hard check (D8) guarantees
   it, but the prompt line saves a wasted round-trip by getting the model to read first on its own. No
   prompt rewrite beyond what already landed in `14`.

---

## Files touched
- **`shaderbox/tabs/code.py`** (`draw`): `editor.set_read_only_enabled(app.copilot_turn_active)` before
  `render` on the active session (A1).
- **`shaderbox/app.py`:**
  - `save` early-return + notification while `copilot_turn_active` (A2).
  - the FOUR verbs `select_node` / `delete_node` / `create_node_from_selected_template` / `open_project`
    early-return + notification while `copilot_turn_active` (A3) — NOT `set_current_node_id` alone.
  - new field `_copilot_read_revision: tuple[str, bytes] | None` (B4); a module-level free function
    `_shader_revision(node_id: str, text: str) -> tuple[str, bytes]` (B5) — NOT an App method.
  - `_copilot_current_shader_view._on_main` stamps the revision (B4).
  - `_copilot_apply_shader_edit._on_main` + `_copilot_apply_line_edit._on_main`: the freshness check
    BEFORE the `self.ui_nodes[...]` deref (B6), re-stamp after success (B6).
  - `copilot_send`: reset the stamp to `None` (B8).
- **`shaderbox/widgets/node_grid.py`** (`draw_node_preview_grid`): `begin_disabled()` around the cells
  while `copilot_turn_active` — the visual affordance for the verb-gate (A3).
- **`shaderbox/copilot/capabilities.py`** (`EditResult`): add `stale: bool = False` + `stale_reason: str
  = ""` (B7) — both leaf primitives, defaulted (existing call sites unaffected).
- **`shaderbox/copilot/tools/shader.py`**: the mutating handlers, on `result.stale`, return
  `(False, result.stale_reason, {"stale": True})` (B7). The App computes the SPECIFIC reason (B6); the
  tool layer just carries it (mirrors how `hint` flows). The stale payload is `{"stale": True}` (no
  `errors` key — the chat card / `_apply_event` read only `ok`/`name`, so that's safe).
- **`shaderbox/copilot/agent.py`** (`run_turn`): the cap predicate becomes `is_mutating(tc.name) and not
  ok and not (payload or {}).get("stale")` — the `(payload or {})` null-guard is MANDATORY (payload is
  `None` on malformed-args/unknown-tool/exception; see B7 BLOCKER fix). `payload` IS in scope at the
  counter (verified by review: `ok, msg, payload = registry.execute(...)` binds it in the same loop
  iteration where the counter increments).
- **Tests:** a new `tests/test_edit_safety.py` (or extend `test_copilot_loop.py`) — the freshness fake +
  the scenarios in Manual verification. The editor-lock half (A) is GL/imgui-bound (UN-headless), verified
  by the maintainer in-app + a smoke pass.
- **Docs:** `roadmap.md` (020 row + banner); this spec; a `todo.md` entry retirement if the cross-project
  trace-bleed deferral's worker-quiesce overlaps (it does NOT — that's 022; leave it).

## Manual verification
- `make check` + `make smoke` green.
- **Freshness unit tests (the real proof — drive `run_turn` / the fake caps):**
  - *never-read reject:* a mutating tool as the FIRST action this turn (no get_current_shader) → stale
    reject, `payload["stale"]`, mutates nothing, does NOT count toward the cap.
  - *content drift:* read → (fake mutates the source out-of-band) → edit → stale reject with the
    "source changed" reason; then read again → edit → success.
  - *identity drift:* read node A → (fake switches current_node_id to B) → edit → stale reject with the
    "you switched nodes" reason.
  - *chain after success:* read → edit (success, re-stamps) → second edit WITHOUT a read → succeeds
    (the re-stamp kept the revision current).
  - *cap interaction:* N consecutive stale-rejects interleaved with reads do NOT trip
    `max_edit_retries` (each read resets); a genuine non-matching edit still counts as before.
  - *turn reset:* a stamp from a prior turn does not satisfy a new turn's first edit (reset to None).
  - *malformed-args null-guard (review BLOCKER regression):* a mutating tool called with MALFORMED args
    (→ `registry.execute` returns `payload=None`) must NOT crash the cap predicate — the turn proceeds
    and the failed call counts normally (it's not stale). This is the `(payload or {})` guard.
- **Editor-lock (maintainer, UN-headless):** start a copilot turn; confirm (1) the editor is read-only
  (typing does nothing, the caret/edits are blocked), (2) Ctrl+S shows the "locked while working"
  notification and does not save, (3) clicking another node in the grid does nothing (with a
  notification), (4) on turn end the editor is editable again immediately. Hand to maintainer with the
  exact steps.
- **Live freshness (maintainer):** with a turn idle, externally edit the node's `shader.frag.glsl` on
  disk (so the mtime watcher reloads it), then send a turn that edits without the agent re-reading — OR
  rely on the unit tests (the live repro is fiddly). The unit tests own this guarantee.

## Open questions for the user (resolved with robust defaults — flagged for review)
- **OQ1 — does a stale-reject count toward the edit-retry cap?** Default taken: **NO** (D7) — add the
  `payload["stale"]` marker so `run_turn` excludes it. Cost: one boolean on the agent-visible payload
  contract. If you'd rather not widen it, the fallback is "let it count" (the re-read usually resets the
  counter anyway). **Confirm or override.**
- **OQ2 — freeze node-switch during a turn, or just detect it at edit time?** Default taken: **freeze**
  (D3) — block select/create/delete with a notification; the token's identity-half stays as backstop.
  Alternative: allow switching, let B reject the edit (simpler, worse UX). **Confirm or override.**
- **OQ3 — churning-source giveup?** Default taken: **none** — rely on `max_iterations` (Out-of-scope,
  trigger-gated). **Confirm or override.**

## Review history
- **Plan-lock deferred (maintainer AFK, authorized autonomous progress via the dev flow).** The three
  OQs are resolved with robust defaults per `dev_flow.md ## Mid-flight escalation` rule 3, flagged above
  for review on return.
- **Pre-impl review (1 agent, adversarial) → FIX-THEN-SHIP; all findings folded in.** Two BLOCKERs:
  (1) the cap predicate `payload.get("stale")` crashes on `payload=None` (malformed-args/unknown-tool/
  exception) → MUST be `(payload or {}).get("stale")` (D7, Files-touched, + a regression test); (2) the
  freshness check must run BEFORE the `self.ui_nodes[self.current_node_id]` deref or it `KeyError`s in
  the B-without-A scenario it backstops → D6 ordering pinned. SHOULD-FIXes: A1's stated mechanism was
  partly wrong (the gapless-ness comes from the latched flag set in `copilot_send`, not the reconcile
  order — A1 re-justified); `_shader_revision` must be a module-level free function, not an App method
  (no-`@staticmethod` rule, D5); the `tuple[str,int]` vs `tuple[str,bytes]` contradiction resolved to
  `bytes` (D4/D5); `set_current_node_id`-alone gating is WRONG (delete/create do their work then
  repoint) → gate the four VERBS incl. `open_project` (D3); add the `begin_disabled` visual on the node
  grid to match the repo's busy idiom (D3); malformed-args regression test added. Reviewer confirmed:
  `payload` in scope at the counter; `set_current_node_id` is the single chokepoint but must NOT be the
  gate; the mtime watcher is genuinely outside A's reach (B is necessary, not redundant); the new
  `EditResult` fields are leaf-seam-safe; the freshness fake harness extends `test_copilot_loop._fake_caps`
  cleanly. No redesign — all spec-pinning.
- **Implemented** in one diff (Half B then Half A then tests): `_shader_revision` (module free fn) +
  `_copilot_freshness_reject` + stamp/check/re-stamp/reset; the editor `set_read_only_enabled` lock + the
  five verb-gates (`_copilot_busy_blocked`) + the node-grid `begin_disabled`; `EditResult.stale`/
  `stale_reason`; the `_stale_result` tool routing; the null-guarded cap predicate. One impl fix: the
  `begin_disabled` is `imgui.begin_disabled()`/`end_disabled()` (paired), NOT `imgui_ctx.begin_disabled`
  (no such context manager — pyright caught it).
- **Post-impl review (1 agent, adversarial) → FIX-THEN-SHIP; the one finding closed.** Zero
  correctness/race bugs; both pre-impl BLOCKERs (check-before-deref, null-guarded predicate) verified
  correctly implemented; the editor lock can't latch stuck (the early-returns that skip the
  `set_read_only` call are frames where the editor isn't rendered); `begin_disabled` balanced (no
  `continue`/`return` in the loop); re-stamp consistent with the next read (`release_program` doesn't
  transform the text). The SHOULD-FIX (coverage gap: no test hit the REAL `_shader_revision` /
  `_copilot_freshness_reject` — all rested on the fake mirroring them) is CLOSED: added pure unit tests
  of both real functions (all branches incl. the node-absent short-circuit). 144 tests green.
