# 086 — the source lock is a MODE you set, not a state you discover

The maintainer, on 085's three-state lock:

> "explicitly locked" and "locked by default" had to stop being the same value — i don't like this...
> too implicit.

Then, after sketching a six-answer gate card and rejecting it mid-sentence:

> Let's do the following: by default the lock is OFF, but on the first edit the agent will ask the
> permission with the following options: allow once (this turn), allow this session, or allow always
> in this project, disallow once, disallow ... wait. no, this is shit as well.. Let's think more...

The design question that resolved it, and his answer: **is the lock a mode you set, or a question you
answer? — "yes, let's do A"**, a mode, with the selector where the icon is now.

## Goal

One control, three positions, all visible, any of them one click away. What the copilot does with a
source edit is a setting you can read off the screen and change directly — never a state you infer
from a padlock's color, and never a horizon you have to choose while a blocked agent waits.

| Position | A source edit |
|---|---|
| **Allow** | runs, asks nothing |
| **Ask** | opens the gate; one deny answers the turn (085 D1, kept) |
| **Deny** | declined, asks nothing |

Persisted per project in `app_state.json`, so a project answers this question once.

## Out of scope

- **A per-project override that is not the mode itself.** The mode IS the persistence; there is no
  second "remember this" affordance anywhere. *Trigger: the maintainer asks for a per-document or
  per-tool exception.*
- **Scope choices on the gate card.** 085's three answers collapse to two (see D3). The card asks
  about THIS call; the horizon is the mode, set outside a blocking prompt. *Trigger: he asks for
  "allow for this session" as a distinct answer again — that would be re-opening D3.*
- **Widening the locked tool roster.** 083 D6's `locks_source` set is untouched. *Trigger: a new
  mutating, ungated tool appears (the enumeration test fires on its own).*

## Design decisions

Numbered, locked.

### D1 — 085's `ASK`/`ARMED` split is REPLACED, not renamed.

085 D2 made the lock three-valued to separate "he armed it" from "it defaults armed", because the
maintainer's sentence about an explicit lock could not otherwise be honored against a bool that was
`True` by default. He has now rejected that separation by name: it is implicit, because the two
states look alike and he never chose the one he was in.

The three values survive; what changes is **who sets them and how you reach them**. `ARMED` becomes
`DENY` and is reachable by one click instead of only by a cycle that skips it. `OFF` becomes `ALLOW`.
`ASK` keeps both its behavior and its default status (D2) — what it loses is being the state you
arrive at without choosing.

**So the implicitness is gone by construction, not by relabelling**: every position is written on
screen, and the one you are in is the one you (or this project, once) selected.

### D2 — the default is ASK, and 083 D5's protection is KEPT.

An earlier draft of this spec defaulted to ALLOW, reading "by default the lock is OFF" from the
maintainer's abandoned sketch. That was selective: the same sentence continues **"but on the first
edit the agent will ask the permission"**, so the state he was describing — off, yet still asking on
the first edit — is what this spec calls `ASK`. He confirmed directly when the reading was put to
him: *"yes, Ask is the default."*

**Persistence, not the default, is what answers his complaint.** A persisted `ASK` is neither
implicit (it is written on screen and one click from any other position) nor a per-session guard he
must rediscover (it survives restarts per project). So the feature needs nothing from a permissive
default, and 083 D5's reasoning — a copilot that edits unasked is the thing to protect against —
stands untouched.

**This also matters for projects that already exist.** No `app_state.json` on disk carries this key,
so every one of them takes the default; shipping ALLOW would have silently switched the copilot to
editing unasked in projects that ask today. `ASK` keeps their behavior exactly as it is.

### D3 — the gate card drops to TWO answers, because the mode owns the horizon.

085 shipped three (`SESSION` / `ONCE` / `DENY`). "Allow for this session" only existed because there
was no way to say "stop asking" outside the card; the segmented control is now that way, and a better
one — it persists, and it does not require an agent to be blocked before you can reach it.

So the card asks the one question a card should ask, about the call in front of you:

- **Allow** — this call runs. The mode is unchanged.
- **Deny** — this call does not run, **and no further source call asks this turn** (085 D1's latch,
  kept exactly).

`LockAnswer` therefore loses `SESSION` and becomes a two-member enum. **`ONCE` is renamed `ALLOW`**:
with no session-wide sibling, "once" was contrasting with something that no longer exists, and the
button already said "Allow once" only to distinguish it from the answer being removed.

**Why not keep a session answer anyway.** It would be a fourth way to reach a state the control
already shows, and the two would drift the moment one persists and the other does not — the "two
parallel things that must stay in lockstep" smell `conventions.md` names. One writer, one home.

### D4 — the control is ONE chip that cycles, matching the uniform panel's input-type selector.

Radio buttons were offered and rejected: three circles plus three labels cost ~200px of a bar that
is already rationing width, and they read as a settings form dropped into a toolbar. A three-chip
segmented group was then built and rejected by the maintainer on sight — *"i don't like these large
3 chips. Looks like a shit. Let's do a single chip with rotating option (like we use for the
uniforms)"*.

So it is one fixed-width chip showing where the setting IS, advancing on click:
`ui_primitives.cycle_chip`, the drawn seam matching `draw_input_type_selector`'s shape in the
uniform panel. The caller owns the ordering and does the advancing.

**This does not reintroduce 085's implicitness**, which is worth stating because the complaint that
started this feature was about a cycling control. What made the padlock implicit was that its
positions were *unnamed* — a colour told you which of three states you were in, and `ASK` could not
be reached by clicking at all. The chip names its position in words, every position is on the cycle,
and the mode is persisted rather than reset per session. The cost that remains is the honest one: a
cycle shows where you are, not where you could go, which is the trade for a control that costs 64px
instead of 140.

`lock_icon_button` is DELETED, along with `SourceLock.variant` and `SourceLock.toggled` — the glyph
index and the two-position toggle exist only to serve it.

### D4a — `must_confirm` KEEPS its `is not OFF` shape; only the refuse condition moves.

The tempting change is to narrow `must_confirm` to `is ASK`, on the reasoning that DENY no longer
asks. **It would ship a Deny mode that allows everything**, and the suite would be green.

The refuse branch lives INSIDE `if registry.must_confirm(tc.name):`, and `registry.execute` sits
outside it. A `must_confirm` that answers False for DENY therefore does not reach the refusal — it
falls through the whole block to the execute call. `tests/test_brake_falsifiers.py` already pins this
in as many words ("a must_confirm that went False here would route the call straight past the gate
block and RUN the edit the lock exists to stop"), which is why **that assert is not relaxed by this
feature**; it gains a `DENY` row instead.

So the predicate keeps meaning *does this call need the user's permission at all* — which DENY does,
in the sense the loop cares about: it must not run unexamined. 085 D2a and the promoted bullet in
`conventions.md` already say this; the narrowing would have contradicted both.

### D5 — the labels are `Allow` / `Ask` / `Deny`.

All three name what happens to an edit, so they are parallel and readable as one row. The rejected
set was `Free / Ask / Off`, where two words are about permission and the third is about being asked.

Tooltip on the group, one per chip, within the five-word `set_tooltip` budget
(`tests/test_ui_prose_budget.py`, and a clause joiner fails it outright):

| Chip | Tooltip |
|---|---|
| Allow | `Edits run without asking` |
| Ask | `Asks before each edit` |
| Deny | `Declines every edit` |

### D5a — the top bar's width floor is DERIVED, not hand-summed.

The floor's literal (`BTN_SM_H + USAGE_BARS_W + 2*BTN_SM_W + 4*SPACE.LG` = 307) budgeted for ONE
icon while TWO had shipped since 083, and nothing caught the drift — the row survived on the gauge's
headroom. A floor that enumerates what it is flooring cannot go stale the next time the bar gains a
control; a hand-summed one already did. With one chip the floor is 355px against a 524px panel.

**`USAGE_BARS_W` 64 → 48 and `COPILOT_W` 504 → 524** are the maintainer's own sizing (*"we can make
the session progress bar a little bit smaller ... and the copilot window itself a little bit
wider"*), asked for while the three-chip group was crowding the row. They are KEPT as preferences,
but they are **no longer load-bearing**: the floor fits either way (371px with a 64px gauge), so a
future reader must not treat the layout as depending on them.

### D6 — persistence rides `UIAppState`, the per-project surface that already holds the sibling pref.

`UIAppState` is loaded from and saved to `app_state.json` per project
(`project_session.py`, `app.py`), and already carries `copilot_layout` — the same kind of
persisted copilot UI preference. A new `copilot_source_lock: SourceLock = SourceLock.ASK` field
goes beside it (D2).

**The live copy stays where it is, and `set_source_lock` stays a TWO-field writer.** 083 D7a's split
(`ChatState` for the main thread, `ToolRegistry` for the worker) is untouched. The persisted copy is
mirrored where `copilot_layout`'s is — in `App.save`, reading the live `ChatState` value — rather
than written from `set_source_lock`.

**That placement is the threading argument, not a style choice.** `set_source_lock` has a worker-side
caller today (`_unlock_source_from_worker`), and while this feature deletes it (D3 removes the answer
that calls it), a persisted pydantic field written from the worker while `App.save` reads it is a
race that would exist only by coincidence of that deletion. Mirroring at save keeps the persisted
copy main-thread-only **by shape**: no future worker path can reach it, because `set_source_lock`
does not touch it.

**A project switch re-seeds from the incoming project's file**, which is the whole point of
per-project persistence: `reset_conversation` currently re-seeds from a freshly-built `ChatState`,
and must instead take the value the switched-to project carries.

**No migration.** A pre-086 `app_state.json` has no such key and pydantic supplies the default, which
is the repo's no-backcompat rule working as intended rather than an exception to it.

### D6a — `reset_conversation` re-seeds from `app_state`, which serves BOTH its callers correctly.

`reset_conversation` is reached two ways and they want opposite things: a **project switch** must take
the incoming project's mode, while **Clear** (the chat's own button) must leave the current project's
mode alone. Getting this wrong makes Clear silently reset the user's setting, which is a regression no
amount of lock testing would surface.

Seeding from `self.app_state` rather than from a freshly-built `ChatState()` satisfies both **without
a branch**: on a switch, `App._init` has already loaded the incoming project's `app_state` ~50 lines
before it calls `reset_conversation`, so the value is the new project's; on Clear, `app_state` is
untouched, so the value is the one the user set. A parameter passed at the switch site would satisfy
only the first and hand Clear the default.

**The ordering is a real dependency, not an observation**: it holds because the load precedes the
reset in `_init`. If that ever inverts, a switch silently carries the outgoing project's mode into the
incoming one, which is why verification names it.

## Files touched

| File | Change |
|---|---|
| `shaderbox/copilot/gate.py` | `SourceLock` members renamed `ALLOW`/`ASK`/`DENY`; `variant` + `toggled` deleted; `LockAnswer` drops `SESSION`, renames `ONCE` → `ALLOW`; `GateResponse.lock_answer`'s "one of the three" comment restates the now |
| `shaderbox/copilot/state.py` | `ChatState.source_lock` stays `ASK`; the module docstring's cross-thread exception (it justifies itself by naming the deleted session-unlock path) and the field comment (which names a two-field writer) both restate the now |
| `shaderbox/copilot/tools/registry.py` | the member rename only; `must_confirm` keeps `is not OFF` (D4a) |
| `shaderbox/copilot/agent.py` | the refuse condition reads `DENY` rather than `ARMED`; `release_lock` / `unlock_source` and its `SESSION` branch deleted — which changes `run_turn`'s SIGNATURE, so its three call sites (`session.py` and two in the lock tests) drop the keyword |
| `shaderbox/copilot/session.py` | `_unlock_source_from_worker` deleted with its answer; `reset_conversation` seeds from `app_state` (D6a); `_LOCK_OUTCOMES` loses its SESSION row and `ONCE`'s value becomes plain `Yes`; `answer_gate_lock`'s docstring names two answers |
| `shaderbox/ui_models.py` | `UIAppState.copilot_source_lock` |
| `shaderbox/project_session.py` / `shaderbox/app.py` | seed the session's lock from the loaded project state |
| `shaderbox/ui_primitives.py` | `lock_mode_chips`; `lock_icon_button` deleted |
| `shaderbox/widgets/copilot_chat.py` | the top bar draws the chips, `gauge_w` and the `min_w` floor are recomputed (D5a); `_draw_lock_choices` drops to two buttons |
| `shaderbox/theme.py` | `USAGE_BARS_W` 64 → 48, `COPILOT_W` 504 → 524 (D5a) |
| `tests/test_copilot_source_lock.py` | the renames, the two-answer card, the persistence round-trip, the project-switch re-seed |
| `tests/test_brake_falsifiers.py` | the renames; a `DENY` row on the `must_confirm` test (NOT a relaxation — see D4a); `test_a_bare_registry_is_unlocked` rewritten, since under the renames it would assert a default equals itself |
| `tests/test_summary_consumption.py` | its one `set_source_lock` call takes the renamed member |
| `scripts/dogfood/harness.py` + the `/dogfood` skill | `auto_approve_gates` answers a SOURCE_LOCK gate with `answer_gate(True)`, which sends no `lock_answer`; the flow it documents changes with the two-answer card |

## Verification

Each falsifiable, each failing for one reason. **The 085 invariants that survive are re-run under the
new names**, since a rename is exactly when a behavior quietly changes — and 085's own history is that
seven of thirteen attempted breaks initially passed.

**New behavior:**

1. **ALLOW runs the edit and opens no gate.** *Falsifier: treat ALLOW as ASK → a gate opens.*
2. **ASK opens exactly one gate and the approved edit lands.** *Falsifier: treat ASK as ALLOW → no
   gate; as DENY → no edit.*
3. **DENY opens no gate and runs no edit.** *Falsifier: narrow `must_confirm` to `is ASK` (D4a's
   trap) → the call skips the whole gate block and the edit RUNS, which this catches and the
   suite otherwise would not.*
4. **One deny still answers the turn** (085 D1). Two edits in two batches, one DENY: one gate, zero
   edits. *Falsifier: drop the latch → two gates.*
5. **DENY does not swallow a tool with its own gate**, in BOTH directions — the armed half (DENY +
   `render_image` → one CONFIRM gate) and the latch half (a denied edit, then `render_image` → its
   own gate). 085's review called this guard the one that cost the most to find, and each half fails
   independently. *Falsifier: drop the `locks_source` guard → the corresponding half goes red.*

**Persistence:**

6. **The mode round-trips through `app_state.json`.** Write DENY, save, load into a fresh
   `UIAppState`, read DENY. Lives beside the existing app-state tests, NOT in the copilot lock file —
   that file imports no `UIAppState` and builds no project dir. *Falsifier: omit the field → the load
   returns the default.*
7. **A project switch takes the incoming project's mode** (D6a). *Falsifier: seed from a fresh
   `ChatState()` → the switched-to project shows the outgoing mode or the default.*
8. **Clear does NOT reset the mode** (D6a's other caller). Set DENY, clear the chat, still DENY.
   *Falsifier: seed from anything but `app_state` → Clear silently reverts the user's setting.*
9. **A fresh project ASKS** (D2, pinned so a later default flip is loud and so the reasoning that
   chose it is not re-derived). *Falsifier: change the default → red, with the reason in the message.*
10. **The live copies never disagree** (083 D7a). After a set, `ChatState` and `ToolRegistry` read the
    same value at construction, on an explicit set, and after a reset. *Falsifier: write either
    directly → the agreement assert fires.*

**Deletions — a removed mechanism must leave nothing behind:**

11. **`SESSION` is gone from the vocabulary.** `LockAnswer` has exactly two members, AND
    `run_turn`'s signature no longer accepts `unlock_source`. The member count alone is not enough:
    `release_lock` would sit wired-but-dead and green, so the signature assert
    (`"unlock_source" not in inspect.signature(run_turn).parameters`) is what makes the deletion
    decidable. *Falsifier: leave the parameter → red even with the enum trimmed.*
12. **`test_a_bare_registry_is_unlocked` still discriminates.** Under the renames a bare registry
    defaults `ALLOW` and a session defaults `ASK`, so the test must assert THAT DIFFERENCE rather than
    a default equalling itself — which is what a naive rename produces, and which is a gate that
    reads as enforcement while enforcing nothing. *Falsifier: default the bare registry to `ASK` →
    red (today's naive rename would stay green).*

**085 invariants this feature rewires and must not break** (each already has a test; each is re-run,
not rewritten):

13. **A declined call still returns a tool result** — the provider-400 invariant, invisible to the
    fake client and held only by the recorded-request assert.
14. **A refused call reaches the turn ledger** — else the next turn is told the copilot did nothing.
15. **The decline message is one template on both paths.**

## Review history

One pre-implementation reviewer, anchored to the maintainer's verbatim words rather than to this
spec. Verdict PARTIAL; two findings were blockers and both are fixed above.

- **The ALLOW default was read out of half a retracted sentence.** The reviewer traced the same
  sentence's second clause ("but on the first edit the agent will ask") and showed the spec's
  justification argued about persistence while concluding about permissiveness — and that it would
  have silently changed behavior for every project already on disk. Put to the maintainer, who chose
  `ASK`. Now D2.
- **`must_confirm` narrowing to `is ASK` would have shipped a Deny mode that runs every edit**, since
  the refuse branch sits inside that predicate and `execute` sits outside it. The spec's own
  Files-touched rows contradicted each other, and it separately instructed relaxing the one assert
  that guards this. Now D4a, with the predicate unchanged.
- The top-bar floor overflows by ~60px with a chip group, and was ALREADY stale (budgeting one icon
  for the two that ship). Now D5a, with the floor derived rather than hand-summed.
- `reset_conversation`'s two callers want opposite things, and Clear would have reset the mode. Now
  D6a.
- D6's "third copy is just like `copilot_layout`" was true only as a side effect of deleting the
  worker-side caller. The persisted copy now mirrors at `App.save`, which makes it main-thread-only
  by shape.
- Six unlisted sites and four stale comments added to Files touched; the verification section
  rewritten (twelve steps from eight, plus three carried-forward 085 invariants).

**Checked and rejected:** `_chip_row` is not the helper D4 needs — it renders non-interactive labels
with `+N` overflow, not a selector. `lock_mode_chips` is still warranted.

## Open questions for the user

None blocking. Two defaults were taken on his behalf and remain cheap to reverse:

- **The labels** are `Allow / Ask / Deny` per D5; he was offered `Free / Ask / Locked` too and
  answered "yes, go" without picking. If the other set is wanted, D5 is a three-string change.
- **Per-project persistence is ON** per D6, which is what distinguished design A from design B in the
  question he answered.

A third — the shipped default — was raised WITH him rather than assumed, after a review found the
first draft had read half a sentence. He answered `ASK` (D2). The lesson is filed at D2 rather than
here: a default that applies to every project already on disk is not a detail to infer.
