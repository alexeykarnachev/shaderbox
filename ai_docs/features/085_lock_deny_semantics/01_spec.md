# 085 — the source lock's deny is a turn-wide answer, and an armed lock does not ask

The maintainer's seventh walk, one finding, in his words:

> when I deny the shader editing, the assistant keeps asking (in the same turn!). If we denied
> once, we should deny all other attempts this turn. From the users perspective I don't even care
> how many editions the assistant does. For me — it is just a simple single deny. Also, if we
> explicitly set the lock ON, I think we shouldn't even ask the user to allow the edit, since the
> user explicitly disallow it.

Both halves reverse a decision 083 locked deliberately. That makes the reconciliation the first
job of this spec, not a footnote.

## Goal

One user gesture answers one question. Today a "Deny" answers one CALL, so a turn where the model
attempts five edits asks five times, and the icon the user set to "locked" is a request to be
asked rather than an answer.

Two changes, and they are the same change seen from the two directions the user can approach it:

1. **DENY latches for the rest of the turn.** After one Deny, every further source-locked call in
   that turn is declined without asking. The next turn starts fresh.
2. **A lock the user ARMED deliberately declines without asking at all.** The icon becomes a real
   switch: armed = "no, and don't ask me"; the session's default lock keeps asking, because that
   is the only thing that makes a default-on lock usable.

## Out of scope

- **Persisting either state across restarts.** 083 D5 made the lock per-session on purpose and this
  does not touch that; a new session starts at the default lock with no latch. *Trigger: the
  maintainer asks for a project- or app-level "never edit my source" setting.*
- **A per-turn latch on ALLOW.** 083 D5's argument against widening one approval to calls the user
  never saw stands untouched — it is an argument about approvals, and this feature only changes what
  a REFUSAL covers. `SESSION` remains the only answer that stops the asking in the allow direction.
  *Trigger: the maintainer reports answering "Allow once" repeatedly in one turn.*
- **Telling the model about the lock in the system prompt.** 083 D8 keeps the lock out of the prompt
  and this keeps it out: the decline message on the tool channel is the whole signal. *Trigger: a
  trace shows a model burning a turn re-attempting after a latched deny — see D4's brake.*

## Design decisions

Numbered, locked.

### D1 — 083 D5's per-call model is REVERSED for DENY, on the maintainer's own re-decision.

083 D5 considered exactly this and rejected it:

> a per-turn latch on "deny" would suppress a question about a DIFFERENT edit he might well approve

and closed with the honest consequence — "a locked session in which the model attempts five edits
and the user answers 'allow once' five times asks five times. That is the user asking to be asked."

The maintainer has now run that session and overruled the premise: *"From the users perspective I
don't even care how many editions the assistant does. For me — it is just a simple single deny."*
The rejected cost was that a latched deny hides a later edit he might approve; his answer is that
he does not model the turn as a sequence of separable edits at all. One ask, one answer.

This is recorded as a reversal rather than a refinement so the next reader does not re-derive D5's
argument and re-flip it. **The user-level model of a turn is atomic: he asked for one thing, the
copilot's several tool calls are its business, and his single "no" is about the thing he asked for.**

### D2 — the lock is TWO states, not one bool: `ASK` and `ARMED`.

The second half ("if we explicitly set the lock ON we shouldn't even ask") cannot be implemented
against the current `source_locked: bool`, because that bool is `True` in every fresh session by
083 D5. Reading it as "the user explicitly disallowed" would mean the copilot silently refuses
every edit until the user finds the icon and toggles it twice — the default session would be one in
which the copilot cannot work and never says why.

So what the maintainer calls "explicitly set the lock ON" is a state the current model cannot
express: the difference between *the default guard* and *a switch the user reached out and flipped*.
The lock becomes a three-valued enum on `ChatState`, `SourceLock`:

| Value | Set by | A source call does |
|---|---|---|
| `OFF` | the user unlocking | runs, asks nothing |
| `ASK` | the session default; "Allow once"/"Deny" leaving it | opens the three-answer gate |
| `ARMED` | the user clicking the icon INTO the locked state | declines, asks nothing |

`ASK` is what 083 shipped as `True` and keeps every one of its behaviors. `ARMED` is new and is the
only state that answers the maintainer's second sentence. The icon cycles `OFF → ARMED → OFF`: a
user reaching for the lock control wants the switch, and `ASK` is not a destination he ever picks —
it is where a session starts and where "Allow once" and "Deny" leave it.

**Why an enum and not a second bool.** Two bools admit `armed and not locked`, a state with no
meaning that every reader would then have to normalize. The three values are mutually exclusive by
construction, and `must_confirm` reads one field.

### D2a — `must_confirm` stays a BOOL; the loop, not the registry, picks ask-vs-refuse.

Three outcomes now exist (ask / refuse silently / run) and the tempting move is a three-valued
`must_confirm`. It is the wrong split, for a reason that decides itself: **the registry is SESSION
state and the latch is TURN state**, so a three-valued `must_confirm` would be answering half the
question — it can see `ARMED` and can never see the latch, leaving the loop to re-decide anyway.

So `must_confirm(name)` keeps its bool and its meaning widens by one word: *does this call need the
user's permission at all* — `requires_gate(name) or (lock is not OFF and locks_source(name))`. The
loop then asks the second question itself:

    if registry.must_confirm(tc.name):
        refuse = registry.locks_source(tc.name) and (
            registry.source_lock is SourceLock.ARMED or deny_latched
        )
        if refuse:
            <decline without asking>
        else:
            <the existing gate block, unchanged>

**The `locks_source` guard is load-bearing, not decoration.** `must_confirm` is True for two
different reasons — the lock, and a tool's own ALWAYS policy — and only the first is the lock's
business. Without the guard an armed lock silently swallows `render_image`, `delete_document` and
every publish: tools the user confirms separately, which would go dead with no card saying why. A
user arms this to stop EDITS. (Held by
`test_the_lock_never_reaches_a_tool_that_does_not_write_source`, whose two halves — armed, and
latched — each fail on their own.)

This keeps 083 D7's funnel intact — one call site, one blocking hop — and it keeps the bool shape
that `tests/test_brake_falsifiers.py` asserts against (`assert registry.must_confirm("edit_shader")`
/ `assert not registry.must_confirm("read_shader")`; a truthy enum member would pass the negative
only by accident of its value).

**This is a REFINEMENT of 083 D7, and it is named rather than left implicit** because D7's premise
was the maintainer's own plan-lock constraint ("use our generalized gating machinery, not just
ad-hoc work-arounds"). What D7 forbade was a parallel BLOCKING branch beside the funnel; the
refuse path does not block, does not touch `GateChannel`, and sits inside the funnel's own `if`.

### D2b — 083 D7a's two-field single-writer split SURVIVES the enum, and is load-bearing.

Stated because a reader implementing D2 sees two fields holding one value and reasonably reaches
for the tidy-up. Do not: `state.source_lock` is read on the MAIN thread (the icon,
`copilot_chat.py`), `registry.source_lock` on the WORKER (the tool loop), and they exist apart for
exactly that reason. `CopilotSession.set_source_lock` stays the ONE writer, and the two seeding
sites (`session.py.__init__` and `reset_conversation`) keep calling it.

The enum does not weaken the thread argument — a `StrEnum` assignment is a single attribute store,
atomic under the GIL exactly as the bool was. What it DOES invalidate is the written justification:
`state.py`'s module docstring says the write is safe because it is "a bare bool", which stops being
true of the code as it is. That comment is rewritten in the same wave, per the repo's rule that a
comment states the now.

### D3 — the DENY latch lives on the TURN, in the agent loop, not on the session.

`run_turn` already owns every other turn-scoped brake (`no_action_retried`, `compile_nudge_sent`,
`consecutive_failed_edits`, `loaded_tools`) as a local rebuilt per turn. The deny latch is exactly
that shape — a local `bool` set when a `SOURCE_LOCK` gate answers `DENY`, read before the next
`must_confirm` ask — and putting it there makes "the next turn starts fresh" true by construction
rather than by a reset somebody must remember, the same reasoning 083 D5 used to put the lock on
`ChatState`.

It must NOT go on `ChatState` or the registry: both outlive the turn, so a latch there would need
an explicit clear at turn start, and the clear is the thing that gets forgotten (`dev_flow.md` §7's
"an unwired mechanism counts as ABSENT" is the same failure at one remove).

**The latch is DENY-only, and the discriminator is `resp.lock_answer is LockAnswer.DENY` — never
`not resp.approved`.** The two are not the same set: a plain CONFIRM `No` on `delete_document` also
lands in `not resp.approved`, and latching there would let a refused deletion silence the source
lock for the rest of the turn. `ONCE` and `SESSION` do not latch either; `SESSION` already stops the
asking by unlocking, and `ONCE` leaves the lock where it was, which is the state D1 does not touch.

**The silent-refuse path carries the same three obligations the live decline carries**, and one of
them is not cosmetic: it MUST append a `_tool_message` for the call. An assistant message whose
`tool_call_id` has no matching tool result makes the provider 400 the next stream — the existing
decline path says so in a comment, and a refuse path that only records a card and a ledger line
would break the turn rather than decline a call.

### D4 — a latched or armed decline returns the SAME message a live decline returns, plus one turn-scoped clause.

083 D8 settled that a declined lock reuses the existing decline message. That holds: the model reads
the same `error: user declined — the <tool> did NOT happen.` on every path, so nothing about the
lock's internals reaches the token stream (the actor model's §2 — the model is blind to our flags,
and telling it about them buys nothing it can act on).

The existing message already ends `do not retry it this turn`, which is the right instruction and
is now BACKED by a mechanism rather than being a request the model may ignore. That is the actual
shape of the fix in actor-model terms: 083 shipped a conscience plea (`do not retry`) with no
enforcement; this feature makes the engine enforce it, which is the "facts as data, not conscience"
rule applied to a brake instead of a prompt line.

**No new message, no new tool result vocabulary.** A distinct "you already tried this" line would be
a second thing to keep in sync for no behavior the model can take differently — it cannot un-decline,
and the only useful action (stop and tell the user) is what the existing sentence already asks for.

### D5 — the decline still costs a card, and the card says which answer produced it.

A silent decline that leaves no trace in the chat would make an `ARMED` session look like a copilot
that ignores instructions. So a refused call yields an `AgentToolCard` and its ledger entry — what
disappears is the blocking gate card, not the record.

**This is deliberately NOT parity with a live decline, which yields no card.** The two differ
because their visible residue differs: answering `Deny` already leaves the gate card standing in the
chat with the outcome on it, so a tool card would say the same thing twice. A refusal has no card of
its own, so without one it would leave nothing at all. The rule is one visible trace per refused
call, not one mechanism per path.

The gate card for the ANSWERED gate keeps its outcome text (083's `_LOCK_OUTCOMES`). Calls declined
by the latch behind it produce ordinary failed tool cards, which is what they are.

**The ledger entry is the load-bearing half, not the card.** `ran.record` is what reaches the turn
summary the NEXT turn reads: drop it and the model is told the copilot did nothing at all, and the
document address a "do the same to C" follow-up needs goes with it. The live-decline path records
it, so nothing about the code looks wrong when the refuse path does not.

### D6 — the icon's tooltip and the gate card's copy change with the states.

The tooltip today reads `Asks before changing` / `Changes freely`. With three states it reads:

| State | Tooltip |
|---|---|
| `OFF` | `Changes freely` |
| `ASK` | `Asks before changing` |
| `ARMED` | `Declines changes` |

`ASK` is reachable only as the session default and after `ONCE`/`DENY`, so its tooltip is unchanged
from what 083 shipped.

The `Deny` button on the gate card is the surface where D1's widening must be legible, because a
user who reads it as "deny this one" and gets a turn-wide refusal has been surprised by his own
feature. The label stays **`Deny`** — it sits in a three-button row whose widths are already tight —
and a hover tooltip `Denies further changes this turn` carries the scope.

**Every string here is measured by `tests/test_ui_prose_budget.py`**, whose `set_tooltip` budget is
five words and whose `_CLAUSE_JOINERS` rejects `;`, ` — ` and ` -- ` outright. The copy above is
written to that gate rather than exempted into its allowlist: a tooltip needing a subordinate clause
is documentation, and three states are distinguishable without one.

**The three-way tooltip is written as a NESTED CONDITIONAL EXPRESSION, not a dict or a `match`.**
The gate's `_score` reads `Constant`, `JoinedStr`, `IfExp`, `List` and `Tuple`, scoring an `IfExp`
as the worst of its branches; every other shape scores `UNMEASURABLE` and would have to buy a
written `_UNMEASURABLE` exemption. A shape that keeps the strings measurable is strictly better than
one that needs an entry explaining why they cannot be.

**The glyph carries the third state in COLOR, and the color is `STATE_ERROR`, not the accent.**
`lock_icon_button` already draws two glyphs (closed shackle / open shackle) and colors them
`ACCENT_PRIMARY` vs `FG_DIM`. `ARMED` keeps the closed-shackle glyph — it IS locked — and takes
`COLOR.STATE_ERROR`: a state that refuses is the same visual class as `danger_button`, and
`ACCENT_PRIMARY` is swappable at runtime (`theme.apply_theme` reassigns it), so a distinction built
on the accent stops being a distinction under some presets. Width does not change with state, which
is the property the existing docstring is protecting.

## Files touched

| File | Change |
|---|---|
| `shaderbox/copilot/state.py` | `source_locked: bool` → `source_lock: SourceLock` (new `StrEnum`, default `ASK`); the module docstring's "bare bool (atomic under the GIL)" justification and the field comment both restate the now |
| `shaderbox/copilot/tools/registry.py` | `source_locked: bool` → the same enum (a bare registry stays `OFF`, D2b); `must_confirm` keeps its bool per D2a |
| `shaderbox/copilot/agent.py` | the turn-local deny latch beside the other turn-scoped locals; the refuse-without-asking path with its `_tool_message`; `build_gate` unchanged (it reads `requires_gate` + `locks_source`, never the lock state) |
| `shaderbox/copilot/session.py` | `set_source_locked(bool)` → `set_source_lock(SourceLock)`, still the one writer; its docstring; `_unlock_source_from_worker`; the two seeding sites; `answer_gate_lock` unchanged |
| `shaderbox/widgets/copilot_chat.py` | the icon reads three states; tooltips; the `Deny` tooltip |
| `shaderbox/ui_primitives.py` | `lock_icon_button` takes the state, not a bool (an armed lock draws distinctly from an asking one); its docstring names two states and must name three |
| `tests/test_copilot_source_lock.py` | the latch's falsifier, the armed path, and the unchanged-`ASK` regressions |
| `tests/test_brake_falsifiers.py` | the `must_confirm` / `requires_gate` split assertions follow the enum, plus the `ARMED` row (`must_confirm` answers differently there, so the two-state test becomes three) |
| `tests/test_summary_consumption.py` | its one `set_source_locked(False)` becomes the enum call |

## Verification

Each falsifiable, each failing for one reason.

1. **The latch declines the second call without a gate.** A turn scripted with two `edit_shader`
   calls, lock `ASK`, one `DENY` answer: exactly ONE `AgentGateOpened`, ZERO edits reaching the
   backend. The two calls go in **two separate response batches**, which is where a latch wrongly
   scoped per-batch would hide (one batch would pass either way). *Falsifier: remove the latch read
   → two gates open and the count assert reports the surplus (the existing pump answers past its
   list with DENY specifically so this fails rather than hangs).*
2. **The latch is turn-scoped.** Two successive `run_turn` calls on one registry, each with one
   edit, one DENY answer in the first: the SECOND turn opens its own gate. *Falsifier: hoist the
   latch to the registry → the second turn asks nothing and the assert goes red.* **Needs a small
   harness extension**: `_run` builds a fresh registry and runs exactly one turn, so this step adds
   a two-turn helper beside it. A helper, not a redesign — named here so it is not discovered at
   implementation time.
3. **`ARMED` opens no gate and runs no edit.** *Falsifier: treat `ARMED` as `ASK` → a gate opens.*
   Needs `_run`'s `locked: bool` parameter to take the enum.
4. **`ASK` still asks per call when the answer is `ONCE`.** The 083 regression: two calls, two
   `ONCE` answers, two gates, two edits. *Falsifier: latch on every answer instead of `DENY` → the
   second gate never opens.*
5. **`SESSION` still unlocks and stops asking.** 083's existing test, carried forward on the enum.
   *Falsifier: latch on `SESSION` too → the gate count still reads 1 and passes, while the lock
   identity assert (`registry.source_lock is SourceLock.OFF`, today's `assert not source_locked`)
   goes red. The identity check is what discriminates, which is why the step needs both asserts.*
6. **A declined call still returns a tool result.** Every call declined by the latch or by `ARMED`
   yields its `AgentToolCard` and appends a tool message for its `tool_call_id`. *Falsifier: drop
   the `_tool_message` append → the card count still matches and this assert goes red. The real
   failure it stands for (a provider 400 on the next stream) is invisible to the fake client, so
   nothing else in the suite would catch it.*
7. **The lock domain check follows the enum, in the two halves it splits into.**
   `test_the_source_lock_domain_is_enumerated_not_asserted` and
   `test_a_locked_tool_never_carries_a_second_prompt` carry forward **untouched** — both read
   `d.locks_source` off `ToolDefinition` and never the lock state, so D2 changing the lock's type
   cannot reach them. `test_the_lock_widens_must_confirm_and_leaves_requires_gate_alone` and
   `test_a_bare_registry_is_unlocked` DO move: the first assigns `registry.source_locked = True` and
   asserts on `must_confirm`'s bool (it gains the `ARMED` row, since that is where `must_confirm`
   still answers True while the loop refuses), the second becomes an identity check against
   `SourceLock.OFF`.

## Review history

One pre-implementation reviewer, anchored to the maintainer's verbatim TODO paragraph rather than to
this spec alone. Verdict PARTIAL; every finding accepted and folded in above:

- The two D6 tooltips tripped `test_ui_prose_budget.py` — one on the clause-joiner assert, one at
  eight words against a five-word budget. Found independently while drafting and already fixed; the
  reviewer added the shape requirement (a nested `IfExp` stays measurable, a dict or `match` does not).
- Verification step 6 claimed `test_brake_falsifiers.py` passes untouched. **False for two of its
  four lock tests** — one assigns `registry.source_locked = True` and asserts on `must_confirm`'s
  bool. Now split into the two halves as step 7.
- "`must_confirm` gains a third answer" was one clause in a table row. Promoted to D2a, with the
  bool kept: a three-valued `must_confirm` can see `ARMED` but never the turn latch, so it would
  answer half the question and the loop would re-decide anyway.
- The refuse path's `_tool_message` obligation and the `LockAnswer.DENY`-not-`not approved`
  discriminator were both unstated. Added to D3, with the provider-400 consequence named.
- 083 D7a went unmentioned while being relied on. Stated as D2b.
- `tests/test_summary_consumption.py` and four stale-comment sites were missing from Files touched.

Nothing was rejected. No finding contested the design; all seven were gaps in what the spec SAID
versus what the implementation would have to do.

**Two post-implementation rounds followed, and their detail is in the commit bodies** (`12cc5d7`
and `9ee2c4c`) rather than restated here, per `dev_flow.md`'s "spec or the commit message". The
shape worth carrying forward: neither round found a behavioral defect, and both found gates that
passed whether or not they worked. Eleven breaks were tried across the feature and SEVEN initially
passed — every one a test that read as enforcement and enforced nothing. The two that most deserve
remembering are in D2a and D5 above, because in both cases the spec had already named the
obligation and no assert held it.

A third round verified the fixes by mutation and returned PASS.

## Open questions for the user

None blocking — the maintainer is afk and both halves are unambiguous in his message. One judgment
call made on his behalf and flagged here: **D2's three-state lock** is more than his sentence asked
for, and it exists because his sentence is unimplementable against a bool that is `True` by default.
If he wants the simpler reading — the default lock itself refuses silently — that is a one-line
change to D2's table (`ASK` disappears, the default becomes `ARMED`), at the cost of a fresh session
in which the copilot cannot edit anything until he unlocks it.
