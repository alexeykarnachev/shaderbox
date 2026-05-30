# 08 — Editor auto-flush: hook vs. manual, the right mechanism

Research only. Feeds (a) a near-term standalone auto-save change and (b) the copilot spec's
"dirty-editor policy" open question (`99_synthesis.md` OQ1 / `07_phasing_risk_spec.md` R11).
No production code here.

---

## Recommendation (TL;DR)

**Build a single generalized hook, not N manual calls. The ONE trigger is a focus-falling-edge
detector wired into the existing focus-tracking in `tabs/code.py`.**

Concretely:

1. Generalize `flush_current_editor` → a path-keyed `flush_session(path)` (today it only flushes
   the *current* editor and re-derives the node from `current_node_id`; see the bug note in §3).
   Keep `flush_current_editor()` as a thin wrapper.
2. In `tabs/code.py`, where `app.editor_focused` is recomputed each frame (code.py:226), detect the
   **True→False transition** (`was_focused and not now_focused`) and call
   `app.flush_session(prev_path)` on that edge. This is the single choke-point: *every* way the
   editor can lose focus (node switch, lib switch, Esc, Ctrl+Tab region cycle, popup/modal open,
   clicking the grid/panel, window-focus loss) routes through this one recompute.
3. Keep the two **lifecycle** flushes that do NOT go through a focus edge as explicit calls — they
   are not focus transitions and there is no frame in which the edge would fire:
   - **Shutdown**: `ui.py:68` `app.save()` already flushes. Leave it (belt-and-suspenders; see §4).
   - **Project open / `_init`**: `app._init` calls `release()` (app.py:640) and `ui_nodes.clear()`
     (app.py:647) — sessions are about to be dropped. Add one `flush_all_dirty_sessions()` at the
     **top of `_init`** (before `release()`), OR route project-open through `save()` first. (Today
     `open_project` does NOT flush — a live data-loss gap independent of the copilot. See §2.)
4. For the **copilot's future file-writes**: the agent's shader-edit tool calls `flush_session(path)`
   for the target path *immediately before* it writes, regardless of focus. This is the one case the
   focus-edge does NOT cover (the editor can be focused *while* the agent writes — §6).

Why a hook and not manual calls: the focus-loss surface is **not** a small enumerable set of call
sites. It is "any frame where `is_window_focused` flips false," and that flip is produced by code
all over (`select_node`, `_set_region`, `_handle_escape`, every `open_*` popup, raw mouse clicks
into other regions, OS-level window deactivation). Manual calls would have to be sprinkled at ~8
sites today and every future site that can move focus — a missed site is **silent data loss**. The
focus edge is the *single physical event* all those sites already converge on. (§3 compares both
shapes in full.)

---

## 1. Focus-loss / inconsistency transition enumeration

Every transition where the editor's in-memory buffer could diverge from disk and then get clobbered
or lost. For each: is there unsaved-edit risk, and does it pass through a single choke-point?

| # | Transition | Trigger site | Unsaved-edit risk | Passes through focus-edge? |
|---|---|---|---|---|
| 1 | **Switch node** (grid click / palette) | `select_node` (app.py:418) → `set_current_node_id` (app.py:628). Sets `editor_defocus_requested = True` when grid-owned (app.py:424). | Old node's session stays in `editor_sessions` (never popped on switch) → not *lost*, but stays dirty + uncompiled until re-selected. The mtime watcher won't touch it (disk unchanged). | **Yes** — `current_editor_path` changes, the prior pane loses focus. |
| 2 | **Switch lib file / node↔lib** | `open_shader_lib_file` (app.py:733), `show_node_editor` (app.py:749) | Same as #1 — prior session retained, dirty, uncompiled. | **Yes** — `current_editor_path` flips, focus recompute sees the new pane. |
| 3 | **Region cycle (Ctrl+Tab) / tab jump** | `cycle_region`→`_set_region` (app.py:404); leaving EDITOR sets `editor_defocus_requested = True` (app.py:416) | Buffer stays dirty in the (still-current) session; no clobber, but inconsistent vs preview until save. | **Yes** — `set_window_focus(None)` in code.py:235 drops focus → recompute flips false. |
| 4 | **Esc** | `_handle_escape` (hotkeys.py:64) sets `editor_defocus_requested = True` | Same as #3. | **Yes** — same defocus path, code.py:234-238. |
| 5 | **Popup/modal opens** (settings, node-creator, emoji, lib-picker, palette) | `open_settings`/`open_node_creator`/… (app.py:454-485); `open_palette` (app.py:365) | Editor is **not drawn** while a popup is open (code.py:136-139, the render() FPE quirk) and `editor_focused` is forced False there. The session is untouched. | **Yes (with a caveat)** — code.py:138 sets `app.editor_focused = False` on the early-return frame, so the edge fires. BUT `current_editor_path` is still valid then, so the flush targets the right session. See §4 gotcha "popup early-return." |
| 6 | **Click into grid / panel** (no chord) | raw mouse; `active_region` corrected from live focus (ui.py:261-265) | Buffer dirty, no clobber. | **Yes** — `is_window_focused` flips false on the click frame. |
| 7 | **OS window deactivation** (alt-tab away) | glfw focus loss | Buffer dirty; risk only if an external process writes the file while away (the copilot, or the user's `$EDITOR`). | **Yes** — `is_window_focused(child_windows)` returns false when the whole window is unfocused. |
| 8 | **App shutdown** | `run()` loop exit → `app.save()` (ui.py:68) | `save()` already calls `flush_current_editor` (app.py:976). Covered today. | **No** — last frame already drew; no further focus recompute. Keep the explicit `save()`. |
| 9 | **Project open** | `open_project`→`_init` (app.py:1013/639); `_init` calls `release()` + `ui_nodes.clear()` | **Live gap today**: `open_project` does NOT flush. Dirty edits in the outgoing project are lost when `_init` clears `ui_nodes`. Sessions dict is *not* cleared (see §4 "stale sessions"), but the nodes they referenced are gone. | **No** — `_init` is a synchronous teardown, no intervening frame. Needs an explicit flush. |
| 10 | **mtime watcher reload (the clobber asymmetry)** | `_reload_if_changed` (ui.py:72) | The actual data-loss bug the copilot research flagged. **Node-root branch (i==0, ui.py:85-98): unconditional `set_text` via `sync_editor_from_disk` (app.py:848) — clobbers unsaved edits with NO diff.** Lib branch (ui.py:108-119) diffs first (`if session.editor.get_text() != new_text`) and preserves undo when texts match. | **No** — this is the *external-write* path, the mirror of the copilot problem. Auto-flush-on-focus-loss DISSOLVES it for the common case (§6). |

**Verdict on the enumeration:** transitions 1-7 are all instances of *one event* — the per-frame
focus recompute flipping False. They do **not** want N manual calls; they want one edge detector.
Transitions 8-10 are NOT focus transitions (no intervening frame / external writer) and stay as
targeted explicit flushes.

---

## 2. Is there a single focus-loss signal in imgui-bundle?

**There is no `is_focused()` on the TextEditor widget** — confirmed in the stub
(`imgui_color_text_edit.pyi`: `set_focus()` exists at line 299, no getter) and in the in-tree note
"The editor exposes no is-focused getter, so read imgui's real focus state for this pane"
(code.py:225). The widget also offers no `is_deactivated`-style edge.

**But the app already has the signal it needs**, computed every frame:

```
# tabs/code.py:226
app.editor_focused = imgui.is_window_focused(imgui.FocusedFlags_.child_windows)
```

This reads the focus state of the editor's child window (the TextEditor renders its own focusable
inner window inside the `code_editor` child). It is the canonical focus proxy the rest of the app
relies on — `hotkeys.py:34` gates EDITOR-scope commands on it, the dispatcher uses it, the dim
overlay uses it (code.py:203).

`is_item_deactivated_after_edit()` is **not usable here** — the TextEditor is not a standard imgui
item (it doesn't go through `ItemAdd`/`is_item_*`); it's a self-contained child window. That whole
family of getters returns garbage for it. The child-window focus query is the only reliable signal.

**The falling edge is trivial to derive** and there is a natural place for it: `editor_focused` is
recomputed once per frame at code.py:226. Track the previous frame's value (or the previous
`current_editor_path`) and fire on `prev_focused and not now_focused`:

```
# sketch, NOT production
prev_focused = app.editor_focused          # value from LAST frame, before recompute
prev_path = app._last_focused_editor_path  # path we were editing last frame
app.editor_focused = imgui.is_window_focused(...)   # existing line 226
if prev_focused and not app.editor_focused and prev_path is not None:
    app.flush_session(prev_path)            # the ONE auto-flush trigger
app._last_focused_editor_path = current_path if app.editor_focused else prev_path
```

**Critical subtlety — flush the PREVIOUS path, not the current one.** On a node-switch frame
(transition #1/#2), `current_editor_path` has *already* moved to the new node by the time the focus
recompute runs (the switch happens in the hotkey/click handler earlier in the same frame). So the
edge detector must flush the path it was editing *last* frame, not `current_editor_path`. This is
why `flush_current_editor` (which always reads `current_*`) is insufficient and must be generalized
to `flush_session(path)` (§3). The `_last_focused_editor_path` latch carries that path across the
switch.

**Three existing flags — could the flush hang off them instead?** `editor_focused`,
`editor_was_ever_focused`, `editor_defocus_requested`:
- `editor_focused` — the live proxy. The edge detector IS this flag's falling edge. ✅ correct anchor.
- `editor_defocus_requested` (app.py:222) — set by Esc / region-leave / node-switch-while-grid
  (app.py:416/424, hotkeys.py:64), consumed in code.py:234. Covers transitions 1,3,4 but **NOT**
  5,6,7 (popup-open, raw click, OS deactivation don't set it). So it's a *subset* of the focus-edge.
  Hanging the flush off `editor_defocus_requested` alone would miss the click/popup/OS cases →
  partial coverage → exactly the "missed site" failure mode. Use the focus edge, not this flag.
- `editor_was_ever_focused` (app.py:219) — sticky "user typed here" bit; orthogonal to flushing.

So: **the falling edge of `editor_focused` is the single signal.** It's already computed; the hook
is ~4 lines around the existing line 226.

---

## 3. The two design shapes, compared

### Shape A — generalized auto-flush hook (RECOMMENDED)

**Mechanism:** focus-falling-edge detector around code.py:226, calling a generalized
`flush_session(path)`. One logical trigger. Plus the 2 lifecycle flushes (shutdown — already there;
project-open — one new call) and 1 copilot pre-write flush.

- **Call sites:** 1 (the edge) + 2 lifecycle + 1 copilot = effectively **one** for the focus surface.
- **Failure mode if a transition is added later:** none — a new way to lose focus automatically
  routes through the same recompute. Zero-maintenance for the whole 1-7 class.
- **Interaction with `_reload_if_changed` clobber asymmetry:** auto-flush means the editor is clean
  *before* it loses focus, so by the time any external writer (or the watcher) reads the file, disk
  == buffer. The unconditional node-root `set_text` (ui.py:88-92) then re-sets identical text — a
  no-op clobber (undo index resets but the text is unchanged). The asymmetry's *teeth* (losing
  divergent unsaved edits) are gone for the focus-loss case. (The asymmetry still bites if an
  external write lands *while the editor is focused* — §6.)
- **User-visible behavior change:** yes — leaving the editor now compiles + persists. Generally
  desirable (the "(unsaved)" tag in code.py:117 mostly disappears; the preview always reflects what
  you last looked at). Gotchas in §4.

### Shape B — manual `flush_current_editor()` at each transition site

**Mechanism:** add `flush_current_editor()` (or `flush_session(prev)`) calls in `select_node`,
`_set_region`, `open_shader_lib_file`, `show_node_editor`, each `open_*` popup opener,
`_handle_escape`, `_init`, and the copilot tool.

- **Call sites:** ~8 today, growing with every new focus-moving feature.
- **Failure mode if a site is missed:** **silent data loss** — the user's edits in the outgoing
  session are dropped (node-switch retains the session, but it stays uncompiled; project-open drops
  it entirely). No error, no notification — the worst kind of bug. The transitions are exactly the
  kind that get added casually (a new popup, a new nav command) by someone who doesn't know the
  flush contract.
- **The `flush_current_editor`-reads-current trap:** transitions 1/2 change `current_editor_path`
  *before* a manual call in the new handler would run, so a naive `flush_current_editor()` placed
  *after* the switch flushes the WRONG (new) session. Each manual site would have to flush *before*
  mutating current — easy to get backwards. The hook sidesteps this with the `prev_path` latch.
- **Maintainability:** poor. The contract "every focus-mover must flush first" is invisible and
  unenforceable; it's the canonical scattered-invariant smell.

### The `flush_current_editor` bug that both shapes expose

`flush_current_editor` (app.py:815-837) computes `node = self.ui_nodes[node_id].node`
(app.py:819-820) **unconditionally**, then branches on `session.source.path == node.source.path`.
If `current_node_id` is `""` or absent from `ui_nodes`, line 820 raises `KeyError`. Today it's
guarded only indirectly (callers tend to have a current node). The generalized `flush_session(path)`
should:
- take the path explicitly (so the edge can flush `prev_path`),
- look up whether that path is a node-root (find the owning node) or a lib file, independent of
  `current_node_id`,
- write to disk + reset `saved_undo`, and for a node-root, `release_program(text)` + `render()` as
  today (app.py:823-826).

This refactor is a prerequisite for Shape A and is good hygiene regardless. **Recommend Shape A.**

---

## 4. Is auto-save-on-every-focus-loss desirable? Adversarial pass

### Broken-intermediate-edit auto-compiled
You type half a shader, click the grid → it auto-saves + recompiles → compile fails. **This is
fine:** `core.py` preserves the last-good program on compile failure (core.py:226-229, the
feature-013 invariant) — the preview stays bright, the error strip shows the diagnostics. No black
screen, no crash. The only change vs. today: the error now persists across the focus loss instead of
vanishing because the buffer was never flushed. That's arguably *more* honest (the file on disk now
matches what you see). **Not a blocker.**

### Undo history
`flush` only updates `saved_undo` (app.py:837); it does **not** touch the undo stack. The
`TextEditor`'s undo history survives a flush — you can still Ctrl+Z after an auto-save. The dirty
tag (`get_undo_index() != saved_undo`, app.py:805) correctly goes clean. ✅ No regression.

### Write churn / mtime thrash → watcher feedback loop?
Every auto-flush writes the file → bumps mtime → `_reload_if_changed` fires next frame. **Does this
loop?** No:
- **Node-root (ui.py:85-98):** flush calls `release_program(text)` which sets `source.text` but
  **does not bump `source.mtime`** (core.py:191 keeps the old mtime). So next frame the watcher sees
  `disk_mtime != src.mtime` (disk was just written) → enters the root branch → reads disk →
  `sync_editor_from_disk` sets identical text + resets `saved_undo` → updates `source.mtime` to
  disk's. One extra recompile, then `disk_mtime == src.mtime` → quiescent. **One wasted recompile
  per focus-loss, not a loop.** (Could be eliminated by having `flush_session` set `source.mtime`
  after writing, but that's an optimization, not a correctness need.)
- **Lib (ui.py:108-119):** flush writes disk; watcher's lib branch diffs (`get_text() != new_text`)
  → texts match → **no `set_text`, undo preserved**, just bumps mtime + invalidates dependents.
  Clean. The lib branch was *designed* for exactly the "user saved in-app" case (ui.py:101-104
  comment). ✅
- The flush is gated on `is_current_editor_dirty()` (app.py:817) — a focus loss with no edits writes
  nothing. No churn when nothing changed. ✅

So: at most one recompile per *dirty* focus-loss. Recompiles are already the steady-state cost of
the hot-reload model; one per focus-loss is negligible. **Not a blocker.**

### When would always-auto-save surprise or annoy?
1. **Lib files shared across nodes.** Auto-saving a lib file on focus-loss invalidates *every*
   dependent node (ui.py:145-147 / the lib branch). If the user was mid-experiment in a lib helper
   and clicks away, all nodes recompile against the half-edit. Mitigated by last-good-program (still
   bright), but the error strip lights up on unrelated nodes. **Acceptable but worth a note** — it's
   the same blast radius as an explicit Ctrl+S on a lib file today.
2. **"Scratchpad" edits the user intended to throw away.** Today you can type junk, *not* save, and
   switch away — the junk dies with the session (or is recoverable). With auto-save it's committed to
   disk. Counter: the undo stack survives (above), and there's a `trash_dir` for nodes but **not**
   for shader text. A user who relied on "didn't save = didn't happen" loses that. This is the one
   genuine behavior loss. Judgment: the maintainer explicitly *wants* always-save; the scratchpad
   pattern is niche and undo covers most of it. **Note it; don't let it block.**
3. **No "dirty" indicator anymore.** The `(unsaved)` tag (code.py:117) becomes near-vestigial (only
   shows while actively typing, between edits and the next focus-loss). Fine, arguably cleaner.

**Adversarial conclusion:** the strongest case against is the scratchpad-throwaway loss (#2). It does
*not* change the recommendation — the maintainer asked for always-save, undo mitigates it, and the
mechanism (hook) is right regardless of how aggressive the policy is. If the maintainer later wants
to soften it, the *policy* (when to flush) lives in one place (the edge condition); the *mechanism*
doesn't change.

### Belt-and-suspenders: keep shutdown `save()` and add project-open flush
The focus-edge does not fire on shutdown (no further frame) or on `_init` (synchronous teardown).
Keep `app.save()` in `run()` (ui.py:68) and add a flush at the top of `_init` (before `release()`,
app.py:640). These are the two non-focus lifecycle holes; they are explicit by necessity, not
scatter.

### Stale-sessions caveat (pre-existing, flagged for the hook author)
`editor_sessions` is keyed by path and **never pruned on node-switch** — only on delete
(app.py:1043), lib-trash (app.py:878), or rename (app.py:892). `_init` clears `ui_nodes` but does
**not** clear `editor_sessions` (app.py:639-661). So after a project-open, stale sessions for the old
project's paths linger. A blanket `flush_all_dirty_sessions()` that iterates `editor_sessions` must
tolerate paths whose node no longer exists (write-to-disk is still valid for a path that exists on
disk; skip ones that don't). The project-open flush should run *before* `release()`/`clear()` while
`ui_nodes` is still intact, so node-root flushes can still find their node + recompile. (Or: only
flush, don't recompile, during teardown — the project's about to be torn down anyway.)

---

## 5. (covered inline above — focus signal finding is §2)

---

## 6. The copilot connection: does auto-save-on-focus-loss dissolve the clobber problem?

**The copilot research's open question** (`99_synthesis.md` OQ1, `07_phasing_risk_spec.md` R11,
`06_glsl_domain.md` §7): when the agent writes a `.frag.glsl` while the user has unsaved editor
edits, the node-root reload branch (`_reload_if_changed` ui.py:85-92 → `sync_editor_from_disk`
app.py:848) does an **unconditional `set_text`** — it clobbers the user's unsaved edits with no
diff. (The lib branch is safe; it diffs first. The asymmetry is the whole hazard.)

**Does always-auto-save-on-focus-loss dissolve it? Mostly yes, with one residual race.**

- **Common case — agent writes while the editor is NOT focused (user is reading/chatting):** the
  editor lost focus when the user moved to the chat panel → auto-flush already ran → buffer == disk.
  The agent then writes; the watcher's node-root branch re-`set_text`s the agent's new content
  (which is now the source of truth — the user had no pending edits to lose). **No clobber. The
  hazard is gone.** This is the dominant interaction pattern (you talk to the copilot *instead of*
  typing in the editor), so auto-save-on-focus-loss removes the teeth from R11 for the realistic
  workflow. ✅

- **Residual race — agent writes while the editor IS focused:** if the user is actively typing in
  the editor *and* the agent writes the same file in the same window (e.g. a fast agent, or the user
  kicked off a turn then clicked back into the editor), the focus edge has NOT fired (editor still
  focused) → buffer is dirty → the watcher's unconditional `set_text` clobbers the live edits.
  Auto-save-on-focus-loss does **not** cover this. **The copilot's shader-edit tool must still guard
  this case** — it should `flush_session(path)` *before writing* is **wrong** here (it would flush
  the user's edits, *then* the agent overwrites them — the user's work is on disk but immediately
  replaced). The correct copilot-side policy for the focused-editor race is the original OQ1
  decision: **refuse-if-dirty-and-focused** (tell the LLM "user is editing this file, ask them to
  save/pause") OR snapshot-and-merge. Auto-save shrinks this to a *rare* race but does not eliminate
  it.

**Net for the copilot spec:**
- Auto-save-on-focus-loss **dissolves R11 for the common case** (agent writes while user is in the
  chat/elsewhere). The OQ1 "dirty-editor policy" no longer needs a heavyweight answer for the 95%
  path — it's just "disk is already clean, the write lands, the watcher re-syncs."
- The **residual** is the focused-editor race. The copilot tool should still check
  `is_current_editor_dirty()` on the target path *and* whether the editor is currently focused on it,
  and if both → refuse + report to the LLM (the swarm's lean-(a) answer survives, but now as a rare
  edge guard, not the primary mechanism). It should NOT blindly flush-then-write (loses the user's
  edits to an immediate overwrite).
- This lets the spec **downgrade R11 from "Med/Med, lock in Phase 3" to a small edge guard** and
  removes auto-flush from the copilot's critical path: the standalone auto-save change (this doc)
  ships first and independently; the copilot inherits a mostly-clean world.

---

## 7. Open questions for the maintainer

1. **Auto-save aggressiveness — policy, not mechanism.** Confirm the policy is "flush on *every*
   focus-loss while dirty" (the maximal reading of your ask). The scratchpad-throwaway loss (§4 #2)
   is the only real cost; undo mitigates it. If you'd rather soften (e.g. only flush on node/lib
   *switch*, not on every click-away), that's a one-line change to the edge condition — the hook is
   the same. Which do you want?
2. **Lib-file auto-save blast radius (§4 #1).** Auto-saving a lib helper on click-away recompiles
   *all* dependent nodes immediately. Acceptable? (It's the same as an explicit Ctrl+S on a lib
   today, just triggered more often.)
3. **Project-open flush (transition #9, a live gap today).** `open_project` currently loses unsaved
   edits silently. Fold the fix into this change (flush-all before `_init` teardown)? It's the same
   data-loss class and the same `flush_session` primitive.
4. **`flush_current_editor` KeyError hardening (§3).** The generalization to `flush_session(path)`
   fixes the latent `current_node_id == ""` KeyError (app.py:820) as a side effect. Confirm it's in
   scope for the standalone change (it should be — it's a prerequisite for the `prev_path` flush).
5. **Copilot focused-editor race (§6 residual).** Confirm the copilot tool's policy for "user is
   actively editing the file the agent wants to write": refuse-and-report (lean-(a)) vs. a
   snapshot/merge. This is now a *small edge guard*, not the primary mechanism — but it still needs a
   decision in the copilot spec. NOT flush-then-write (loses the user's edits).

---

## Appendix — load-bearing citations

- `flush_current_editor` only flushes current; re-derives node from `current_node_id`, unconditional
  `ui_nodes[node_id]` → latent KeyError: app.py:815-837 (esp. 819-820).
- Dirty check: `get_undo_index() != saved_undo`: app.py:801-805.
- Non-creating session lookup: `get_current_session_if_exists`: app.py:807-813.
- Focus proxy recompute (the hook anchor): tabs/code.py:226; "no is-focused getter" note: code.py:225.
- Defocus consumption: tabs/code.py:234-238.
- Popup early-return forces `editor_focused = False`: tabs/code.py:136-139.
- `editor_defocus_requested` is a *subset* of focus-loss (set at app.py:416/424, hotkeys.py:64).
- Node-switch defocus: select_node app.py:418-426; set_current_node_id app.py:628-634.
- Lib switch: open_shader_lib_file app.py:733-747; show_node_editor app.py:749-754.
- Region cycle / Esc defocus: _set_region app.py:404-416; _handle_escape hotkeys.py:45-66.
- Shutdown save (already flushes): ui.py:68 → app.save app.py:975-996.
- Project open / _init teardown (no flush — gap): open_project app.py:1013-1023; _init app.py:639-661.
- **mtime clobber asymmetry**: node-root unconditional set_text ui.py:85-98 / sync_editor_from_disk
  app.py:839-849; lib diff-first-preserve-undo ui.py:108-119.
- Last-good-program invariant (broken edit stays bright): core.py:225-265 (esp. 226-229).
- `release_program` does NOT bump mtime (why the watcher quiesces, not loops): core.py:189-192.
- Stale sessions not pruned on switch/_init: pop only on delete app.py:1043 / trash 878 / rename 892.
- TextEditor has no focus getter, has set_focus + change-callback: imgui_color_text_edit.pyi:299, 493.
- Copilot R11 / OQ1 (the problem this feeds): 07_phasing_risk_spec.md:126,254,391;
  99_synthesis.md:164,206-209; 06_glsl_domain.md:469-470,560-563.
- conventions.md: "Inline editor state lives on App; disk is the source of truth; app.save() flushes
  the dirty editor before [save]": conventions.md:91-92.
