# 083 — Sixth walk findings

The maintainer's sixth walk over the running app, six findings. Four land here; two are editor-repo
work (the vim status band and insert-mode Escape both live inside `libeditor`, not in ShaderBox).

Source: the maintainer's `../TODO` batch, verbatim below each workstream.

---

## Goal

Six findings from one walk, grouped by the surface they touch:

- **W-A — the vim keymap says nothing about what it is waiting for.** In NORMAL mode a half-typed
  phrase (`3d`, an armed `<leader>`) is invisible; the status band shows the mode and the ruler and
  no pending keys. **EDITOR REPO.**
- **W-B — Escape with the completion popup open closes the popup and stays in INSERT.** One Escape
  should do both, as vim does. **EDITOR REPO.**
- **W-C — the uniforms panel's texture preview has no border**, so a texture with `alpha = 0`
  has no visible edge. Give it the pass tile's border and the pass tile's picture size.
- **W-D — the uniforms panel has no pass selector**, and it has outgrown the Document tab. It gets
  its own top-level tab, and a pass selector on it.
- **W-E — the copilot modifies code when the maintainer only asked a question.** A per-session lock,
  a toggle icon in the chat's top bar, and a three-answer gate on the first modification attempt.
- **W-F — the shipped default copilot model becomes `openai/gpt-5.6-luna`.**

---

## Out of scope

- **A rebinding UI for `<leader>` sequences.** Unbuilt by maintainer decision (roadmap banner,
  `commands.py`'s `LEADER_BINDINGS` comment); W-A only DISPLAYS what is pending, it does not let
  anyone change what a key does. Trigger: the maintainer asks for a rebinding surface.
- **Locking anything but source.** W-E's lock covers the tools that write shader, script and pass
  SOURCE (see D6 for the exact set). Publishing and credential tools keep their existing ALWAYS
  gate, which already asks every time. Trigger: the maintainer reports the copilot publishing or
  deleting something he did not want.
- **Persisting the lock across sessions.** D5 makes it per-session with a fixed default. Trigger:
  the maintainer says the default is wrong for him.
- **Changing the on-disk copilot model.** W-F changes the DEFAULT a fresh install gets;
  `integrations.json` already holds `openai/gpt-5.6-luna` on this box, and a persisted value wins
  over the default by design (fail-soft per-key persistence). No migration — none is wanted.

---

## Design decisions

Numbered, locked. Open questions are separate, below.

### D1 — W-A and W-B are editor-repo work; ShaderBox's half of W-A is display only.

`hotkeys.py::_drain_editor_input` hands Escape to `editor.key(KeyCode.ESCAPE)` unconditionally and does nothing
else with it — no `complete_cancel()`, no mode check after. The completion popup is drawn by the
library inside its own primitive stream, and the status band (mode badge, ruler, the `:`/`/`/`?`
line) is library chrome too (`tabs/code.py::draw_chrome`: "The mode badge, the ruler and the
`:`/`/`/`?` line are drawn by the library inside the editor rect"). So neither behavior is
reachable from this repo: both are `libeditor` changes, re-vendored per the seven-file procedure in
`conventions.md`.

**W-B's cause is located**, in the editor repo at `src/keymap.odin`: `editor_key_insert`'s Escape
case ends the insert session at lines 634-670 (mode to `.Normal`, discard the auto-indent, commit
the repeat, break undo, step the caret left), but an OPEN POPUP is intercepted thirty lines earlier
at 490-493 — `autocomplete_clear(&e.autocomplete); return .Consumed` — which never falls through to
that block. That early return is the whole bug, and vim's behavior is to do both.

**The phrase is the RAW TYPED KEYS, not a rendering of `Pending`** (measured by the editor session
against nvim on a real tty, after this spec was locked). nvim's `showcmd` shows the literal
keystrokes in every case with no per-state formatting: `d`, `3`, `3d`, `f`, `g`, `z`, `r`, `y`, `c`,
`2d3`, `"a`. `2d3` is the one that settles it — count, operator, second count, in typed order — and
`"a` is not representable at all, since a register prefix never enters `Pending`. So there is no
mapping from `Pending`'s variants to a rendering, and inventing one would be a vocabulary nvim does
not have. The armed leader stays the one deliberate substitution (`<leader>` for a key that is
invisible when it is a space), which is what the maintainer asked for.

**W-A's data exists and is internal.** `src/mode.odin`'s `Pending` struct holds `count`,
`operator`, `op_count` and `awaiting` (find-char, a `g`/`z` prefix, an operator's motion, `r`'s
target); `src/bindings.odin`'s `Bindings.armed` holds the armed leader. `ed_pending` reduces all of
it to one bool via `editor_pending` and is the only export over any of it — it cannot render `3d`,
only that something is pending. It also has **zero call sites** in this repo today.

### D2 — the pending phrase is drawn by the LIBRARY in its own status band, not by the host.

The alternative — a new `ed_pending_text` export the host reads and draws in `draw_chrome` — puts
the phrase in a different place from the mode badge and the ruler it belongs beside, and makes the
host re-implement a status-band layout the library already owns. The library draws the band; the
pending phrase is a field in that band. Two `ChromeFlag` members that already exist
(`STATUS_SHOWS_MODE`, `STATUS_SHOWS_RULER`) establish the shape: what the band shows is a chrome
flag. Revisit if the maintainer wants the phrase somewhere other than the status band.

Consequence for this repo, and how it was resolved. The risk was an unwired mechanism:
`App._apply_editor_settings_to` sets only `ChromeFlag.LINE_NUMBERS`, and `ed_set_style` replaces chrome wholesale
(`s.chrome = ed.chrome_for(style)`), so a flag ShaderBox never sets would ship dead and pass every
test. The editor session settled it by DEFAULT rather than by a host call: the new flag is true in
`chrome_for(.Vim)` (where `status_shows_mode` and `status_shows_ruler` already are, and where vim's
own `showcmd` defaults on) and false for the standard keymap, which has no pending phrase — so the
flag exists for a host that wants it OFF. ShaderBox's existing `set_style` therefore delivers it and
**no new `set_chrome_flag` call is needed**.

What DOES remain on this side: the `ChromeFlag` IntEnum in `editor/ffi.py` widens by one member
in the SAME commit as the binary copy, per the re-vendor law that values are appended and never
renumbered. The new member appends at 5, after `STATUS_SHOWS_RULER` at 4.

### D2a — the re-vendor is the last step, from a COMMITTED sha, and it carries a behavior note.

`shaderbox/resources/editor/VERSION` holds `dd58aa9`. The re-vendor happens after the editor
session lands W-A and W-B and commits — never from a dirty tree (`conventions.md`: "rebuilds from a
COMMITTED editor-repo sha, never a dirty tree").

**The target sha is not `dd58aa9`'s successor.** Two commits already landed there that change vim
keymap BEHAVIOR without touching the ABI: six measured divergences from nvim closed (a counted `g$`
curswant rule, a blank-line `daw` register that was charwise where nvim reports `V`, `I` on a
whitespace-only line), and an `ed_feed_aborted` latch — `ed_key` never cleared the flag, so one
failed `f` motion made every later key report aborted. That one cost this repo nothing: `_SIG`
declares `ed_feed_aborted` and **no ShaderBox code calls it**, which is worth stating because it is
the kind of latent bug a vendored-binary bump silently inherits.

So the re-vendor writes up what BEHAVIOR changed, not just a new sha — six nvim divergences closed
is a roadmap line, not a silent binary swap. The export count moves off the 106 the roadmap banner
states; it is taken from the editor session's `nm -D` delta at the sha, never predicted here.

**Expect one new export, not zero.** The phrase needs its own buffer in the library rather than
riding the existing `.`-repeat recorder, settled there by driving the dispatcher key by key: that
recorder deliberately DROPS normal-mode count digits (so `.` can substitute a new count — with the
digits kept, `3x2.` replayed `23x`), and it resets at the START of a phrase rather than the end, so
after a completed `x` it still reads `x` with nothing pending. A display reading it would show the
count-less phrase and then keep showing the last finished command forever. Both behaviors are right
for what the recorder is for and wrong for this. Since a host cannot render a buffer it cannot read,
and `ed_pending` collapses everything to one bool, a getter is the expectation — so the banner's
"106 exports at `dd58aa9`" gets a new number, not a re-confirmation.

### D3 — the uniform texture preview becomes a `preview_cell`, not an `imgui.image` with a border pushed.

`widgets/uniform.py`'s three texture surfaces (`_thumb_size` + the bound-media branch,
`_draw_pass_source`, `_draw_black_swatch`) each draw a raw `imgui.image` / draw-list rect with no
border at all. The pass tile's border is not a color it pushes — it is `preview_cell`'s
`begin_child(child_flags=borders)` picking up the theme's `Col_.border` (`COLOR.BORDER`, `#3c3836`)
at the global `style.child_border_size = 1.0`. Reproducing "the same gray-ish color, the same
thickness" by hand at three call sites is three copies of one primitive's behavior; calling the
primitive is one. This also settles the size: `preview_cell(cell_w=...)` gives the same picture
area the strip does, so "the same size" is structural rather than a matched constant.

`preview_cell` serves a uniform row at `selected=False`, `armed=False`, `footer=""`, `chips=None`,
`overlay=None`: the delete-cross, the gear overlay, the footer, the chip row and the delete confirm
all sit behind `if selected` / `if footer` / `if chips is not None` branches that do not run. What
IS unconditional is the full-cell `selectable` inside `preview_cell`, which on the strip means
"make this pass the output" and on a uniform row would mean nothing.

**The selectable stays, and its click is given a job.** Suppressing it (a `clickable=False`
parameter, or a second `preview_frame` primitive sharing the body) buys a shape nobody can see and
costs either a flag on a primitive or a second primitive to keep in step. Instead the uniform
preview's click does the obvious thing for the row it sits on: for a sampler reading a PASS
(`_draw_pass_source`), it selects that pass — the same verb the strip's tile has, on a picture of
the same pass, so the two agree rather than one being inert. For a bound texture and for the empty
swatch there is no pass to select, and the click is ignored.

Nav-keyboard is OFF in this app (imgui skill §9), so the extra `selectable` adds no Tab stop; the
concern that applies under nav-on does not apply here.

### D4 — the uniform preview passes `SIZE.PASS_THUMB`; there is no new size token.

**Measured in a real imgui frame, not derived on paper** — the first draft of this decision got it
wrong twice and both errors are recorded here so the number is not re-guessed.

`begin_child(112, …, borders)` reports a content region of **96 x 96**; `begin_child(96, …)` reports
80 x 80. In `preview_cell`, `cell_h = cell_w + footer_h + chips_h` and then
`img_h = avail.y - footer_h - chips_h`, so the text rows are added below a picture whose height is
`cell_w` less the child's window padding. The footer and chip heights CANCEL: a pass tile's picture
is `PASS_THUMB - 2 * SPACE.MD` = **96 x 96**, whatever the footer and chips measure.

Two corrections that follow:

- The first draft said ~67 by subtracting the footer and chips from `cell_w`. `preview_cell` does
  not do that. **96 is the number.**
- The first draft concluded the pass picture is smaller than `SIZE.THUMB_SM = 90` and that a
  uniform cell therefore needs a NEW token holding 96. Both are wrong: 96 > 90, so the pass picture
  is marginally LARGER; and since a footerless cell's picture is also `cell_w - 16`, passing
  `cell_w = SIZE.PASS_THUMB` lands on exactly 96. **The uniform cell passes the pass strip's own
  constant. No new token, and "the same size" is the same symbol rather than a matched number.**

What the maintainer reads as "a little bit smaller" is the drawn image, not the box.
`preview_cell` letterboxes (`scale = min(avail.x/tw, img_h/th)`), so a 16:9 texture in a 96-square
draws 96x54 — visibly shorter than the same texture at `_thumb_size`'s full 90 of HEIGHT, which is
what the uniforms panel shows today. Matching the element delivers the perception he described.

**The cell becomes SQUARE, and that is a real visual change to call out.** `_thumb_size`
(`uniform.py::_thumb_size`, deleted by this wave) scaled width from a fixed height, so a 16:9 bound video currently draws as a
160x90 strip; under `preview_cell` it becomes a 96 square with letterbox bars. Nobody asked for
that specifically — it arrives with "basically the same element". It is accepted because it is what
makes the rows align into a column instead of running ragged at every texture's aspect, and because
`_draw_black_swatch` is ALREADY a square: after this, all three texture
surfaces agree instead of two disagreeing. **Flagged for the maintainer's eye** — it is a layout
call that cannot be judged headless (imgui skill §0), and it is the one thing in W-C he did not
literally ask for.

### D5 — the lock is per-session, defaults to LOCKED, and lives on `ChatState`.

The maintainer's words: "which will lock the code modification in each session by default". So the
default is locked, and "session" is the chat session — the thing `reset_conversation` rebuilds and
`Clear` empties. `ChatState` is the per-session state object and is reconstructed whole
(`self.state = ChatState()`) on reset, so a field there is per-session by construction rather than
by a reset call somebody must remember to add. It is NOT persisted: `ConversationStore` would carry
it across restarts, which contradicts "in each session by default".

Three answers, matching the maintainer's three: **allow for this session** (unlocks for the rest of
the session), **allow once** (this call only, stays locked), **deny** (this call does not run, stays
locked). The unlocked state asks nothing.

**"On the first modification attempt" is per CALL, not per turn.** The maintainer's phrasing
describes the common case — the first attempt is when he is asked — but a turn can carry several
source calls, so the spec has to say what the second one does. Each answer decides only its own
call, because the alternative is worse in both directions: a per-turn latch on "allow once" would
silently widen one approval to cover calls he never saw, and a per-turn latch on "deny" would
suppress a question about a DIFFERENT edit he might well approve. So "allow this session" is the
only answer that stops the asking, which is exactly its name and the icon it corresponds to.

The practical consequence is honest and worth stating: a locked session in which the model attempts
five edits and the user answers "allow once" five times asks five times. That is the user asking to
be asked. If it grates, the answer he already has is "allow this session" or the unlock icon —
which is why the three answers are the three he specified, with no fourth "allow for this turn".

### D6 — the lock covers SOURCE-mutating tools, defined by a field on `ToolDefinition`, not by a name list.

`ToolDefinition` already carries `mutating` and `is_edit`. Neither is the right set: `mutating` is
27 tools wide and includes publishing and credentials (which have their own ALWAYS gate), while
`is_edit` is narrower than the ask — it is deliberately just the shader/script edit+write pairs, and
`add_pass` / `set_pass` / `set_uniform` / `create_document` also change the maintainer's project
without being "edits" in that sense.

So the lock reads a new explicit field, `locks_source: bool`, on `ToolDefinition`. A name list in
the agent loop would go stale the first time a tool is added — a field on the definition makes each
new tool declare its own answer at the site where the tool is written.

The set, enumerated from the live registry rather than written by hand. The mutating-and-ungated set
is exactly 15 tools; the roster is that set minus the three excluded below.

**Locked (12):** `edit_shader`, `write_shader`, `set_uniform`, `create_document`, `write_script`,
`edit_script`, `add_pass`, `set_pass`, `rename_document`, `set_canvas_size`, `duplicate_document`,
`unbind_media` — every one runs today with no confirmation whatsoever, which is precisely the
maintainer's complaint.

**Excluded, each with its reason:**
- `telegram_connect` — mutating and ungated, but touches no project source.
- `bind_media` and `import_document` — **both already block on a FILE gate as their first act**
  (`CopilotBackend.bind_media` / `.import_document`, both `GateKind.FILE`), so the user is already stopped and
  shown a native picker before anything changes. Locking them would put a three-button card in
  front of a file dialog: two blocking prompts for one call, which is the double-ask this decision
  exists to avoid. Cancelling the picker already declines the action.
- The ALWAYS-gated tools (`delete_document`, `delete_pass`, `delete_lib_file`, `render_image`,
  `render_video`, the four publish/pack tools, the two credential tools) — they confirm every time
  already.
- The read-only tools (`read_shader`, `read_script`, `read_lib`, `grep`, `probe_render`,
  `switch_document`, `list_telegram_packs`, `load_tools`) — nothing to lock.

**The check enumerates the domain rather than asserting an implication.** An implication
(`locks_source ⇒ mutating`) cannot catch the drift this decision exists to prevent: a NEW mutating,
ungated tool that simply forgets the flag. So the test computes the set from the registry and
asserts equality against the roster, with the three exclusions named in the test beside their
reason:

    {d.name for d in defs if d.mutating and d.gate_policy is NONE} - EXCLUDED == locked_names

and separately that no locked tool carries `gate_policy is ALWAYS` **or** `gate_kind is FILE` —
the second disjunct is what would have caught `bind_media`, and an ALWAYS-only check would have
passed while the double-ask shipped.

### D7 — the lock rides the EXISTING gate machinery end to end; no second path, no ad-hoc branch.

**This is the maintainer's explicit constraint** (plan-lock answer 2: "use our generalized gating
machinery, not just ad-hoc work-arounds"), so it is a premise, not a preference. Every hop the lock
uses is a hop that already exists:

| Hop | Existing mechanism the lock uses |
|---|---|
| decide whether to ask | `registry.requires_gate(name)` — widened, not bypassed |
| phrase the question | `build_gate(registry, name, args)` — a new branch beside CREDENTIAL/CONFIG |
| block the worker | `GateChannel.ask()` on the `_pending` slot — unchanged |
| tell the UI | the `AgentGateOpened` yield + `pump_events` — unchanged |
| draw the card | `_draw_pending_action`'s kind dispatch — a new branch beside the three |
| answer + unblock | `session.answer_gate_*` → `gate.answer(GateResponse(...))` — unchanged |
| cancel / Stop / reset | `cancel_all`'s generation sweep — unchanged, inherited free |

`agent.py`'s tool loop has exactly ONE place a tool blocks on the user
(`if registry.requires_gate(tc.name): ... gate.ask(req)`). The lock's question goes THROUGH that
same block. A parallel `if session_locked: ...` beside it would be exactly the ad-hoc branch the
maintainer ruled out, and would also be a second thing to cancel, to drain on Stop, and to reconcile
with the generation sweep — three bugs the existing funnel has already paid for.

**But `requires_gate` is asked TWO different questions today, and only one of them is "do we ask?"**
Pre-implementation review found the second caller: `_RunLog.summary_lines` in `agent.py`,
where `requires_gate` means **"is this action irreversible?"** — its result decides whether a ledger
line carries its identity verbatim and uncapped into the NL turn summary, so a "continue" after a
cutoff never re-does a publish. Widening that one method would reclassify every source edit in a
locked session as irreversible, pushing uncapped identities into history and inverting the soft cap
that exists to stop a many-call turn bloating it. The method's two readers want two different facts
and have been sharing one name.

So the widening SPLITS them rather than overloading either:

- `requires_gate(name)` — unchanged meaning and unchanged body: this tool's own `gate_policy` is
  ALWAYS. `summary_lines` keeps calling exactly this, so the irreversible classification is
  untouched by the lock and the token-cost behavior does not move.
- `must_confirm(name)` — NEW, the tool loop's question: `requires_gate(name) or (self.source_locked
  and tool.locks_source)`. Only `agent.py`'s gate block calls it.

This is still one funnel — the loop has one call site and one blocking hop — while leaving the
irreversibility question alone. Naming the two facts apart is what makes the funnel safe to widen;
had they stayed one method, the lock would have silently changed what the model reads back.

**`must_confirm` needs the lock state, which the registry does not hold.** The registry is a
per-session object built at session construction (`CopilotSession.__init__`), so the lock is a field ON it
(`ToolRegistry.source_locked`, default False), written by the one seam below. The alternative —
passing the flag into the method at the call site — spreads the decision to every caller and leaves
the tests free to forget it.

**The registry field defaults UNLOCKED even though the session defaults LOCKED.** A bare
`build_registry(caps)` is what the tests construct, and `test_gating_is_a_two_state_decision`
asserts a freshly built registry gates `delete_document` and not `read_shader`. The locked default
is a property of a chat SESSION, applied by `CopilotSession.__init__` through the one writer in
D7a — not a property of a registry in isolation. This keeps the two facts from being conflated
again: the registry holds a state, the session decides its initial value.

`GateResponse` gains one field for the three-way answer rather than overloading `approved`: a
`SOURCE_LOCK` answer of "allow this session" must both approve THIS call and unlock the session,
and folding that into a bool loses the difference from "allow once".

### D7a — the lock has ONE writer, and the toggle and the gate answer both go through it.

**Why two fields rather than one.** The conventions' drift rule says to lift a fact onto one entity
rather than keep two copies in lockstep, and the honest answer for why that is not done here is
THREADING: `ChatState` is main-thread-only by construction (its module docstring pins it as
"Written ONLY by session.pump_events on the main thread, read ONLY by the UI on the main thread ->
single-writer, no lock"), while the gate decision is read on the WORKER thread inside the tool loop.
Collapsing them to one field would put the chat's single-writer state on a cross-thread read path
and give up the invariant that makes `ChatState` lock-free. Two fields with one writer is the
cheaper trade, and V4 exists precisely because it IS a trade.

Two surfaces set the lock (the top-bar icon, and an "allow this session" answer) and one reads it
(`must_confirm`, on the worker). That is the shape a divergence bug lives in, so both writers call
one method,
`CopilotSession.set_source_locked(bool)`, which writes `ChatState.source_locked` (what the icon
draws) and `registry.source_locked` (what the gate reads) together. Nothing else assigns either
field. `reset_conversation` rebuilds `ChatState` and must re-seed the registry through the same
method, or a cleared chat would show a locked icon over an unlocked registry — the exact divergence
this decision exists to prevent, and the one the verification below breaks on purpose.

### D8 — a declined lock returns the existing decline message; an approved one runs unchanged.

The decline path already appends a truthful tool result ("error: user declined — the {name} did NOT
happen. Tell the user it was not done; do not retry it this turn."). That message is exactly right
for a lock denial, and it is the message the actor-model rules want: the fact rides the channel the
model already reads, and the model is told plainly not to retry. No new prompt rule, no standing
"ask before editing" instruction — a conscience plea a cheap model ignores (the copilot skill's
fact-vs-conscience rule). Nothing about the lock enters the system prompt.

### D9 — the lock icon is a drawn glyph in `ui_primitives.py`, beside the layout icon.

The chat's top bar is `[layout icon] [context gauge ...] [Clear][Close]` with one arithmetic owner
for the widths. The lock goes immediately after the layout icon, as
`lock_icon_button(id_, locked, side)` built on the private `_glyph_button` helper the other five
icons share — a padlock drawn from a rect and an arc, shackle up when unlocked. No font
dependency, per the no-icon-font rule. The gauge width arithmetic subtracts one more
`icon_side + SPACE.MD`, in the same expression that already owns it.

### D10 — Uniforms becomes a fourth `DocumentTab` member, and the pass selector is a combo on it.

`DocumentTab` is a three-member StrEnum (`DOCUMENT` / `RENDER` / `SHARE`) driving `_NODE_TABS` in
`ui.py`, persisted on `UIAppState.active_document_tab`, and reachable by `Ctrl+1/2/3`
(`CommandId.FOCUS_TAB_*`). Adding `UNIFORMS` is a member, a `_NODE_TABS` row, a new module
`tabs/uniforms.py`, a `CommandId.FOCUS_TAB_UNIFORMS` and a `Ctrl+4` binding.

The persisted enum is salvage-protected: a value no member claims is dropped by `drop_invalid` and
falls back to the field default, so an `app_state.json` written by this build and read by an older
one loses the tab choice and nothing else. Adding a member is safe in the other direction by
construction.

**Uniforms sits SECOND in the bar** (maintainer's call): `Document, Uniforms, Render, Share`, so
`Ctrl+2` becomes Uniforms and Render/Share renumber to `Ctrl+3`/`Ctrl+4`. The bar order is
`_NODE_TABS`'s list order; the chord is the `CommandSpec` row in `commands.py`. Renumbering is a
change to those two rows, not to the enum's member order — but `DocumentTab` is a StrEnum persisted
by VALUE, so member order carries no meaning on disk and the renumber costs nothing there.

The pass selector is a `grouped_combo`/`labeled_combo` row at the top of the new tab listing the
document's passes in `strip_order`. It writes the panel pass.

### D11 — the pass selector drives a real field; `panel_pass` keeps its tab-follows behavior as the default.

Today `App.panel_pass(document_id)` DERIVES the panel pass each frame: the active shader tab's pass
when that tab belongs to this document, else the output pass. There is no stored choice, which is
exactly why the maintainer cannot tweak a pass that is neither open nor the output.

So a stored override joins it: `UIDocumentState` gains a `panel_pass: str` (empty = follow the
active tab, the behavior above). The combo writes it; opening a pass's shader tab clears it back to
empty, so the common path — click a tile, edit its uniforms — keeps working with no extra state to
notice. A stored name that no longer names a pass falls through to the derived answer rather than
erroring, which is also what a per-key salvage would leave behind after a rename.

**The lazy-row trap does not apply, but a sibling of it does.** `UIDocumentState` is eager — a field
default on `UIDocument`, built per document there and explicitly in
`load_ui_document` — so there is no path where a `UIDocument` exists without one.

(An earlier draft justified this by pointing at `_ui_uniform_for`'s `setdefault`. **That symbol does
not exist in the tree** — row creation is still inline in the draw loop (now `tabs/uniforms.py::draw`)
(`if hash not in ui_uniforms: ui_uniforms[hash] = UIUniform.from_uniform(uniform)`), which is exactly
the lazy shape. `conventions.md`'s lazy-row bullet claims the eager-create fix shipped and names that
function; it is stale about its own remedy. **Corrected in this wave** under the docs-are-living rule —
the bullet keeps its law and loses the false claim about a symbol that is not there. The law itself
still holds and still applies to `ui_uniforms`; only its "already fixed" sentence was wrong.)

What IS live: `App.current_document_ui_state_or_default` returns a **throwaway
`UIDocumentState()`** when no document is selected, and the uniform block being moved reads its
state through exactly that accessor. Writes through it are silently
discarded. `uniform_sort_key` and `uniform_sort_desc` already ride that path harmlessly — with no
document there is nothing to sort. A `panel_pass` write must NOT: the selector's write goes through
an explicit `App` method that resolves the real `ui_documents[id].ui_state` and no-ops on a missing
document, rather than assigning onto whatever the `or_default` property handed back. The combo is
not drawn at all when there is no document, so the no-op branch is unreachable by clicking — it
exists so a headless or programmatic caller cannot write into a discarded object.

### D12 — the Document tab keeps the pass strip; only the uniforms move.

The strip is where a pass is CHOSEN and where its picture is; the uniforms tab is where one pass's
values are TUNED. Moving the strip too would leave the Document tab holding a name field, a canvas
size and a Reset button, and would put the strip's six verbs a tab away from the document they
belong to.

### D13 — W-F is a one-line default change, and the fresh-install path is the only one it reaches.

`CopilotIntegration.model` defaults to `tencent/hy4-preview`. It becomes `openai/gpt-5.6-luna`. A
persisted `integrations.json` keeps whatever it holds — this box already holds
`openai/gpt-5.6-luna`, so the observable change here is nil and the change is for a fresh install.
Said plainly rather than claimed as a behavior change.

---

## Files touched

**W-C / W-D (uniforms):**
- `shaderbox/ui_regions.py` — `DocumentTab.UNIFORMS`.
- `shaderbox/tabs/uniforms.py` — NEW: the tab body (pass selector + the uniform list moved here).
- `shaderbox/tabs/document.py` — the uniforms block leaves; the strip and the document fields stay.
- `shaderbox/widgets/uniform.py` — the three texture surfaces move to `preview_cell`.
- `shaderbox/ui.py` — the `_NODE_TABS` row.
- `shaderbox/ui_models.py` — `UIDocumentState.panel_pass`.
- `shaderbox/app.py` — `panel_pass` reads the override; `focus_document_tab` wiring; the command callback.
- `shaderbox/commands.py` — `CommandId.FOCUS_TAB_UNIFORMS` + the `Ctrl+4` row.

**W-E (copilot lock):**
- `shaderbox/copilot/gate.py` — `GateKind.SOURCE_LOCK`; the three-way answer field on `GateResponse`.
- `shaderbox/copilot/tools/base.py` — `locks_source` on `ToolDefinition`.
- `shaderbox/copilot/tools/registry.py` — `locks_source(name)`.
- `shaderbox/copilot/tools/*.py` — the flag on the twelve locked tools (D6).
- `shaderbox/copilot/agent.py` — the gate condition + `build_gate`'s lock branch + the session-unlock effect.
- `shaderbox/copilot/state.py` — `ChatState.source_locked`.
- `shaderbox/copilot/session.py` — the three-answer `answer_gate_lock`.
- `shaderbox/widgets/copilot_chat.py` — the top-bar icon + the three-button card.
- `shaderbox/ui_primitives.py` — `lock_icon_button`.

**W-F:**
- `shaderbox/integrations.py` — one default.

**Tests + scripts (the three below were missed in the first draft and found by review):**
- `tests/test_brake_falsifiers.py` — the roster checks (V5), beside the existing gate-discriminates
  test whose `requires_gate` call D7 deliberately leaves alone.
- `tests/test_command_registry_coverage.py` — `test_every_command_id_has_a_handler` asserts
  `set(app.command_callbacks) == set(CommandId)`, so it goes RED the moment `FOCUS_TAB_UNIFORMS`
  exists without its callback. A free gate; nothing to add there beyond letting it do its job.
- `scripts/smoke.py` — V7's per-member loop. Today line 270 focuses RENDER only and line 146
  asserts membership; the loop over `list(DocumentTab)` is NEW code, not an existing check.
- `tests/test_ui_prose_budget.py` — the two `_UNMEASURABLE` keys follow their sites: the renamed
  `_draw_texture_preview`, and `_draw_auto_block`'s move to `tabs/uniforms.py`. The allowlist is
  keyed by `(file, function)`, so a moved or renamed scored site fails it until the key follows —
  which is the gate working, not maintenance noise.
- `tests/test_region_system_is_gone.py` — `_table_callees`' docstring names the three `_NODE_TABS`
  entries and calls them "the only route from `document_settings` to the uniform sliders". After
  W-D that route is `tabs/uniforms.py::draw`; the AST walk still works, the sentence does not.
- New: the locked-session agent-loop tests (V1-V3), the one-writer test (V4), the `DocumentTab`
  round-trip + salvage test (V6), the panel-pass override test (V8).

**Editor repo (W-A / W-B), then re-vendored here:**
- `shaderbox/resources/editor/` — the seven files + `VERSION`.
- `shaderbox/editor/ffi.py` — any new export's `_SIG` row, AND the `ChromeFlag` IntEnum
  if the band's new field arrives as a flag. Re-vendor law: values are appended,
  never renumbered, so a host enum mapping them widens in the SAME commit as the copy.

---

## Verification — each check fails for exactly one reason, and each names its falsifier

The maintainer runs the app; a step that asks him to go and look is not a gate. These are checks the
repo runs. Every gate below is DONE only once the break has been tried and the check named it.

**V1 — a locked session gates a source tool** (`locks_source` + lock ⇒ the funnel asks).
Drive `run_turn` headless with a stub LLM emitting one `edit_shader` call, the registry locked, and
a `GateChannel` answered "deny". Assert the edit did NOT reach the capability and the tool message
is the decline line. *Falsifier:* unlock the registry — the edit lands and the test goes red. Break
to confirm: flip `locks_source` off on `edit_shader` and watch V1 fail.

**V2 — an unlocked session does not gate.** Same harness, registry unlocked, gate answered never.
Assert the edit landed and `take_pending()` is empty. *Falsifier:* a `requires_gate` that ignores
the lock state and asks always would hang or fail here. This is the check that keeps the widened
`requires_gate` from becoming "gate everything".

**V3 — "allow this session" unlocks, "allow once" does not.** Two turns, two `edit_shader` calls.
Answering `ALLOW_SESSION` on the first ⇒ the second never opens a gate. Answering `ALLOW_ONCE` ⇒ it
does. *Falsifier:* collapsing the three answers into `approved: bool` makes these two runs
identical, so the test cannot pass under the design D7 rejects.

**V4 — the lock has one writer (D7a's divergence). THIS ONE WAS SPECIFIED AND THEN NOT WRITTEN,
and the gap it left shipped a live defect** — recorded here because that sequence is the lesson,
not the bug. A fresh `CopilotSession` built `ChatState()` (locked) and `build_registry()`
(unlocked) and seeded neither, so a session that was never reset drew a closed padlock over a
registry that confirmed nothing: the icon said "asks before changing" and the copilot edited
freely. That is precisely the divergence D7a exists to prevent, arriving through the one door the
re-seed did not cover. Two independent post-implementation reviewers found it; the full suite,
`make gates` and the 200-frame GL smoke were all green with it live.

Worse, **V4 as originally worded would not have caught it** — it tested `set_source_locked(False)`
then `reset_conversation()`, never construction. So the check that was missing was also the check
that was mis-specified. The shipped form asserts agreement at all THREE lifecycle points that reach
the pair by different routes: construction (two defaults meeting), an explicit toggle, and a reset.
Each break was tried and named its own row. After
`set_source_locked(False)` then `reset_conversation()`, assert `ChatState.source_locked` and
`registry.source_locked` AGREE. *Falsifier — the break to try:* delete the re-seed from
`reset_conversation` and confirm V4 goes red; a cleared chat then draws an unlocked icon over a
locked registry (or the reverse), which no other test in the suite would notice.

**V5 — the `locks_source` roster is closed.** Extends `tests/test_brake_falsifiers.py`, which
already pins that gating discriminates (`delete_document` gated, `read_shader` not) and whose
`test_gating_is_a_two_state_decision` calls `registry.requires_gate` directly — so widening that
method changes what that existing test measures, and the widened decision needs the lock state
named there rather than left implicit.

Per D6 the check asserts **set EQUALITY against the registry**, not an implication: an implication
passes while a new mutating, ungated tool silently omits the flag, which is the drift the field
exists to prevent and the domain-narrowing class the conventions call the most expensive bug family.
*Three falsifiers, each a different hole:* set `locks_source=True` on `read_shader` (a non-mutating
tool joins — fails); on `delete_pass` (an ALWAYS-gated tool joins — fails the double-ask half); and
**add a new mutating, ungated tool without the flag** (fails the equality, which an implication-only
check would let through). The third break is the one that matters and the reason the check is an
equality.

**V6 — `DocumentTab.UNIFORMS` round-trips, and an unknown value costs only the tab.** Save a
`UIAppState` with `active_document_tab=UNIFORMS` and reload it; then hand the loader
`{"active_document_tab": "not_a_tab"}` and assert the model loads with the DEFAULT tab and every
other field intact. *Falsifier:* a loader that raises, or one that resets the whole model, fails the
second half. (This is `drop_invalid`'s contract; the test pins that it covers the new member.)

**V7 — every `DocumentTab` member actually becomes the visible tab.** The first draft of this check
was theater and review caught it; the corrected form matters more than the original.

Two things made it unfalsifiable. First, **the selection lands the frame AFTER the focus** —
`_draw_document_settings`'s own comment says "set_selected drives the tab the frame after a Ctrl+digit jump" — so
a focus-and-assert inside one frame reads the PREVIOUS tab and passes regardless. Second, "assert
the loop covers `list(DocumentTab)`" is a claim about the test's own loop, not about the app: it
cannot go red for a broken tab.

The check instead drives the real frame loop: for each member of `list(DocumentTab)`, call
`focus_document_tab(member)`, run **at least two** frames, then assert
`app.active_document_tab == member` — the value `_draw_document_settings` commits after its tab loop
from the tab that actually drew, not the one that was requested. That reads the CONSUMER (which tab
imgui selected) rather than the producer (what we asked for), and it fails for exactly one reason.
*Falsifier — the break to try:* add the `DocumentTab.UNIFORMS` member WITHOUT its `_NODE_TABS` row.
`focus_document_tab` then sets a target no tab item matches, no tab claims `visible_tab`, and the
assert fails on that member alone. Restore the row and it passes. The same break under the original
wording passed silently, which is why it is the one recorded here.

This is NEW code in `scripts/smoke.py`: line 270 focuses `RENDER` once today and line 146 only
asserts membership in the enum. Neither is the loop.

**V8 — the panel-pass override.** Set the override to a non-output pass with no editor tab open for
it, and assert `App.panel_pass` returns THAT pass; set it to a name no pass carries and assert it
falls through to the derived answer rather than raising. *Falsifier:* the current derive-only
implementation fails the first half — which is the whole point of W-D.

**The test writes through `app.ui_documents[id].ui_state`, never through
`current_document_ui_state_or_default`** — the latter hands back a throwaway when no document is
current, so a test written against it could pass or fail for a reason that has
nothing to do with the override. Same rule as the production write seam above; naming it here keeps
the test from verifying the accessor instead of the feature.

**Not gated here, handed to the maintainer** (no display on this box, §0 of the imgui skill): how
the bordered uniform preview and the four-tab bar actually LOOK, and whether the lock icon reads as
a lock. Headless confirms they draw without asserting; it cannot confirm they look right.

**W-A / W-B carry their own falsifiers in the editor repo** (its own test discipline): an Escape in
INSERT with the popup open leaves NORMAL mode, and a pending `3d` renders in the band. Neither is
checkable from this repo until the re-vendor, and the vendored-ABI test (`tests/test_editor_ffi.py`)
gates only the signature table.

---

## Open questions for the user

All three answered at plan-lock; kept here as the record of what was asked and decided.

1. **Where the Uniforms tab sits.** → **SECOND**, beside Document; Render and Share renumber to
   `Ctrl+3`/`Ctrl+4`. Folded into D10.

2. **What the lock covers.** → the content tools that ask nothing today (twelve, after review
   removed `bind_media` and `import_document` — see D6); the already-ALWAYS-gated
   deletes and publishes stay as they are rather than asking twice. Folded into D6. Answered with
   the constraint that it use the generalized gating machinery — now D7's premise.

3. **The pending-phrase vocabulary.** → an armed leader renders as `<leader>`, counts and operators
   as typed (`3d`). Folded into D2. **Refined after the lock by measurement:** the editor session
   drove nvim on a real tty and found `showcmd` renders the RAW TYPED KEYS with no per-state
   formatting at all (`2d3`, `"a`), so the phrase accumulates keystrokes rather than formatting the
   `Pending` struct — which could not express `"a` anyway. The maintainer's answer is unchanged and
   `<leader>` remains the one substitution; what changed is that there is no vocabulary to invent.

**`switch_document` stays unlocked** — it changes what the app shows and writes nothing.

---

## Review history

Two pre-implementation reviewers, run in parallel against different anchors: one against the
maintainer's verbatim `../TODO` (does the spec deliver what was asked), one against this repo's own
conventions and skills. Both returned PARTIAL and converged INDEPENDENTLY on the same two blocking
defects, which is the signal worth recording — agreement from two different anchors is evidence in a
way agreement from two readings of my own text is not.

**Accepted and fixed (7):**

1. **D4's number was wrong and its conclusion inverted.** I had ~67px for the pass tile's picture;
   both reviewers derived 96, and I confirmed it by measuring `begin_child` in a real imgui frame
   (112 -> 96 content, 96 -> 80). The error was subtracting the footer and chip rows a second time
   after `preview_cell` had already added and removed them. The inversion mattered: at 96 the pass
   picture is LARGER than `THUMB_SM = 90`, which removed the premise for the new size token — the
   uniform cell now passes `SIZE.PASS_THUMB` itself.
2. **D7's widening hit a second consumer.** `_RunLog.summary_lines` reads `requires_gate` to mean "is this
   irreversible?" for the NL turn-summary ledger. Widening it would have pushed every source edit in
   a locked session into the uncapped verbatim-identity branch of persisted history. Split into
   `requires_gate` (unchanged, ledger) and `must_confirm` (new, the loop).
3. **`bind_media` and `import_document` already raise a FILE gate** (`CopilotBackend.bind_media` / `.import_document`), so
   locking them meant a card in front of a file dialog. Both dropped from the roster; the roster is
   12, not 14. The check widened to reject `gate_kind is FILE` too — an ALWAYS-only check would have
   passed while the double-ask shipped.
4. **V5 asserted an implication, not coverage** — it could not catch a new mutating, ungated tool
   that forgets the flag, which is the drift the field exists to prevent. Now a set equality against
   the registry with the exclusions named.
5. **V7 could pass while broken.** `_draw_document_settings`'s own comment says the selection lands the frame
   AFTER the request, so a same-frame focus-and-assert reads the previous tab; and "assert the loop
   covers `list(DocumentTab)`" was a claim about the test, not the app. Now two frames, asserting the
   committed `active_document_tab` — the consumer, not the producer.
6. **D11 cited `_ui_uniform_for`, which does not exist.** I took it from `conventions.md`, which is
   itself stale about its own remedy; row creation is still inline in the uniform draw loop.
   The conventions bullet gets corrected in this wave.
7. **Three missing Files-touched entries** (`test_command_registry_coverage.py`,
   `test_region_system_is_gone.py`, `scripts/smoke.py`) and a "seventeen source tools" count that
   contradicted D6's own roster.

**Considered and not adopted (2), with reasons:**

- **Suppress `preview_cell`'s unconditional `selectable`** (a `clickable=False` flag, or a sibling
  `preview_frame`). Both reviewers flagged the dead click target; one proposed a parameter. Rejected
  in favor of giving the click a job — on a pass-source row it selects that pass, the same verb the
  strip's tile carries on a picture of the same pass. A flag on the primitive or a second primitive
  to keep in step both cost more than the behavior is worth. The related Tab-ring concern does not
  apply: nav-keyboard is off in this app (imgui skill §9).
- **D7a keeps two copies of the lock rather than one field.** One reviewer read this as accepting a
  drift the conventions say to remove structurally, and asked why one field cannot serve both. The
  reason is real and is now stated in D7a: the UI thread reads one and the worker reads the other,
  so a single field would be cross-thread state on a path that is deliberately single-writer. The
  two-copy shape stays, with one writer and V4 breaking the divergence on purpose.

**Not verified by either reviewer, verified by me instead:** every editor-repo claim (D1, D2a) sits
outside this repository, and both reviewers correctly labelled those relayed rather than confirmed.
I opened `src/keymap.odin` and `src/mode.odin` directly and confirmed the popup intercept at 487-493,
the insert-end block at 634-670, the `Pending` fields and `Bindings.armed`.


---

## Review history — post-implementation

Three reviewers in parallel against different anchors: code correctness (concurrency and
lifecycle), conventions and architecture, and spec fidelity against the maintainer's verbatim
`../TODO`. All three returned PARTIAL. **Two independently found the same blocking defect**, which
is the finding worth trusting most.

**Blocking, fixed:**

1. **A fresh session drew a locked icon over an unlocked registry.** `CopilotSession.__init__`
   built `ChatState()` (locked) and `build_registry()` (unlocked) and seeded neither. Probed and
   reproduced before fixing: raw construction gave icon `True`, registry `False`,
   `must_confirm("edit_shader")` `False` — the copilot would edit freely while the padlock said it
   would ask. One line in `__init__` through the one writer. See V4 above for why the check that
   would have caught it was both missing AND mis-worded.

   The live app was saved by accident, which is its own finding: `App.copilot` is a property over
   `self.session` assigned before `_init`, so `hasattr(self, "copilot")` is True on the first init
   and `reset_conversation()` re-seeded — despite the comment there stating it is "Guarded for the
   first _init". A correct invariant held by a call whose own comment says it does not run is not
   an invariant, and the headless drivers (dogfood) do not take that path at all.

2. **Fixing it broke `test_summary_consumption.py`**, correctly: it builds a session directly and
   calls `set_uniform`, which now confirms. Unlocked explicitly in that test, since its subject is
   the NL turn-summary and there is no UI to answer. A new check pins the consequence — a headless
   session locks like any other, deliberately, because a headless default of "unlocked" would make
   the one place nobody is watching the one place the copilot edits freely.

**Accepted and fixed (5):**

3. **Three orphaned symbols** the uniforms move left behind: `_section_break` in
   `tabs/document.py`, and `SIZE.THUMB_SM` / `COLOR.BLACK` in `theme.py`, whose only consumers were
   the deleted `_thumb_size` and `_draw_black_swatch`. Ruff cannot see a dead private function or an
   unused dataclass field, so nothing flagged them. Defined-but-never-read is this repo's named most
   expensive class, and it landed in a commit about removing a stale claim.
4. **Two comments narrated history** rather than the code as it is (`_draw_texture_preview`'s
   docstring saying what the raw `imgui.image` "never gave it" and what "the old fixed-height thumb"
   did; `reset_conversation`'s "without this the two diverge"). Rewritten to state the current
   invariant. The rule is `conventions.md ## Code rules`, and the history belongs here and in the
   commit message, which is where it now is.
5. **Function-body imports** in `tests/test_uniforms_tab.py`, hoisted to module top.
6. **The gate pump busy-spun** ~600k iterations per idle half-second; a 1ms sleep on the empty
   branch cut the file's runtime from 0.95s to 0.44s.
7. **Three spec inconsistencies**: an open-question line still saying "14 content tools" after D6
   went to 12, a Files-touched entry for a `theme.py` token D4 explicitly killed, and
   `tests/test_ui_prose_budget.py` missing from Files-touched.

**Hardened beyond the findings (1):** `scripts/smoke.py`'s Examples-browser frames were fixed
numbers while the tab sweep derives from `list(DocumentTab)` and grows with it — safe at four
members, colliding at five. Both frames now derive from the sweep's end, so the collision cannot
occur rather than being noticed later.

**Considered and not adopted (1):** `state.py`'s single-writer docstring is now inaccurate, since
the worker writes `source_locked` through the session writer. One reviewer suggested restructuring
the callback into an event hop; rejected — the worker's very next tool call gates on that value, and
an event hop would leave it reading stale state for the rest of the turn. The write is a bare bool
and the UI only picks a glyph from it. The docstring now names the exception, its reason, and the
limit (do not extend it to a field the UI computes layout from).
