# 091 pre-implementation review round 3 — verification & blast radius (closure)

Reviewer role: close round 2 item by item, judge the items that changed, and decide whether the
20-item `## Verification` list catches a wrong implementation of every locked decision. Read-only —
nothing under `shaderbox/`, `tests/` or the spec was edited, and no command rewrote the working tree
(`git status --porcelain` shows only the sibling reviewer's own report). Every claim below is a file
read with line numbers or a command with its output.

## Verdict: **PARTIAL** — one item, one replacement clause

Round 2 closed completely: all thirteen rewrites landed, eight of them verbatim, and the two dead
gates (old 5(ii), old 7) are now live gates — **measured, both directions**. Nineteen of the twenty
items catch their decision.

**Item 11(b) is red on the correct implementation.** D8's tuple contains `aqua_n`, and `aqua_n` IS
an accent ACTIVE (`theme.py:97`, `_ACCENTS["aqua"] = (aqua_b, aqua_n, …)`), which 11(b) requires the
tuple to be disjoint from. D8 says both halves in one sentence — *"`aqua_n` is the aqua accent's
ACTIVE colour, not its primary — allowed"* and then *"the assert … pins that no tint equals an
accent primary, an accent ACTIVE, …"*. Measured:

```
purple_n   in 11(b) set: False  SYN hits: []
green_b    in 11(b) set: False  SYN hits: ['SYN_BUILTIN', 'SYN_SCRIPT_UNIFORM', 'SYN_STRING']
yellow_n   in 11(b) set: False  SYN hits: []
aqua_n     in 11(b) set: True   SYN hits: []
```

This is not a preference: the gate fires on the implementation D8 specifies. And the palette cannot
supply a fourth tint under 11(b) as written — after the enumerated set (9 hues) and D8's own stated
exclusions, exactly **three** chromatic hues remain free:

```
accent PRIMARIES: ['aqua_b', 'blue_b', 'orange_b', 'yellow_b']
accent ACTIVES  : ['aqua_n', 'blue_n', 'orange_b', 'orange_n']
SELECT = purple_b   TAG = blue_b   FAVS = yellow_b
STATE_*: ERROR=red_b  INFO=blue_n  OK=aqua_b  WARN=yellow_b

free of that set (chromatic only): red_n, green_n, green_b, yellow_n, purple_n
minus D8's exclusions (red_n = the error border, green_n = 2 degrees from green_b)
            → green_b, yellow_n, purple_n
```

`orange_n` is the orange accent's ACTIVE, so it is not the escape hatch round 2's report suggested.
Either D8 drops to three tints, or 11(b) stops asserting disjointness from the accent ACTIVES and
says why an active is allowed where a primary is not.

**Replacement text — pick one, they are mutually exclusive:**

> **(A) three tints.** D8: *"`COLOR.GROUP_TINTS`, three hues: `purple_n`, `green_b`, `yellow_n`.
> Three and not four because the palette has no fourth that is free: every remaining chromatic hue
> is an accent primary (`aqua_b`, `blue_b`, `orange_b`, `yellow_b`), an accent ACTIVE (`aqua_n`,
> `blue_n`, `orange_n`), a `STATE_*` (`red_b`, `blue_n`, `aqua_b`, `yellow_b`), `SELECT`
> (`purple_b`), `TAG` (`blue_b`) or `FAVS` (`yellow_b`) — measured; `green_n` sits 2 degrees of hue
> from `green_b` and `red_n` reads as the red error border these tiles can also carry."* Item 11(a)'s
> literals become `% 3`, and 11(b) stands unchanged.
>
> **(B) four tints, `aqua_n` kept, the assert narrowed.** D8: *"`aqua_n` is the aqua accent's ACTIVE
> colour. An ACTIVE is a pressed-state fill, never an outline or a chrome edge, so a group outline
> cannot merge with it — the assert therefore pins the tints against accent PRIMARIES, every
> `STATE_*` hue, `SELECT`, `TAG` and `FAVS`, and NOT against the actives."* Item 11(b): *"…disjoint
> from the accent primaries, every `STATE_*` hue, `SELECT`, `TAG` and `FAVS` (NOT the accent actives
> — `aqua_n` is the aqua active, allowed because an active is a pressed fill and a tint is an
> outline), and has no duplicate."*

Everything else in the list is a live gate. §C carries no new findings.

| Item | Verdict |
| --- | --- |
| 5 (both scenarios), 7, 8, 9, 15, 17, 18, 19, 20 | **CATCHES** — each measured below |
| 11 | **FAILS AS WRITTEN** — red on D8's own tuple |
| 1, 2, 3, 4, 6, 10, 12, 13, 14, 16 | CATCHES, unchanged from round 2 |

---

## A. Round 2, closed item by item

| Round-2 ask | The new spec text | Status |
| --- | --- | --- |
| **5(ii)**: falsifier unreachable (`UIDocument.save` compiles program-less passes first) | 5: *"(ii) a host pass whose shader is BROKEN named in a handover is rejected with a message naming it (D6), rather than silently dropped by the save's carry-forward of its disk rows."* | **CLOSED.** The rewrite is taken and tightened — it now asserts the REJECTION, which is D6's own statement, instead of the dropped row. Reachable: measured, a `grade` with a syntax error survives its own save program-less with `errors` set and its `document.json` rows empty (§B-5). |
| **7**: dead gate, same cause; and D5's parenthetical carried the false reason | 7: *"Falsifier: re-key the merged row by the new pass name (a hash nothing computes) and the prune drops it; `get_uniform_hash` is name-and-shape only, so the row must be copied under its own key. (Skipping the copies' compile is NOT a falsifier: `UIDocument.save` compiles a program-less pass itself before it prunes, measured in review round 2.)"* D5: *"compile it (as `add_pass` does, so its uniforms are live for the panel on the next frame; `UIDocument.save` compiles a program-less pass itself before pruning, so the merged rows do not depend on this)"* | **CLOSED, both halves.** The new falsifier FIRES — measured both directions (§B-7): the wrong-keyed row is popped by the prune and absent from disk; the right-keyed row reaches disk as `drag`. D5's parenthetical now states the true reason and keeps the instruction. |
| **8**: missing the spurious-entry-point assert | 8: *"and the broken pass is NOT among the entry points offered (D3). Falsifiers: … treat its empty wiring as a root and `entry_points` answers `[\"blur\", \"scene\"]` for a bloom whose `blur` is broken, growing the dialog a spurious row."* | **CLOSED**, with the measured `["blur", "scene"]` inline. D3 gained the matching design clause (*"it is not offered as an entry point (the dialog would otherwise grow a spurious row for it)"*). |
| **9**: the home has no `app` fixture | 9: *"rename keeps it and delete drops it through the session verbs (`tests/test_pass_verbs.py`, the `app` fixture); a reload reads it and a `graph.json` without the key loads as `\"\"` (`tests/test_graph_persistence.py`, which has no `app` fixture and drives the loader directly)"* | **CLOSED** — the split, not the private-import workaround. Re-verified: `grep -n "def test_" tests/test_graph_persistence.py` → 17 tests, none taking `app`; its tools are `_write_document` (`:65`) + `load_document_from_dir`, which is exactly what clauses 3–4 need. |
| **11(b)**: the hand-enumerated set lets `green_b` (`SYN_*`) through | 11(b) widened to *"the accent primaries, the accent actives, every `STATE_*` hue, `SELECT`, `TAG` and `FAVS"*, plus *"The set is the tile-and-outline context on purpose: the editor's syntax tokens (`green_b` is `SYN_BUILTIN`) never share a surface with the strip, so they are not in it."* | **CLOSED as asked** — the `SYN_*` question is answered as a stated design call, which is the right resolution and better than the allowlist-sweep this reviewer proposed. **But the widening to accent ACTIVES introduced the new failure above.** |
| **15's hotkeys substring** | 15: *"AND `hotkeys.py`'s Escape dispatch names `close_import_passes` (a substring assert on the source, the shape `test_escape_is_owned_by_an_open_name_input_at_the_dispatch` uses, since a second frame-driving App in one process hits the torn-down font atlas). Falsifier: rely on the `hotkeys.py` fallthrough; the funnel test passes either way."* | **CLOSED**, taken whole including the reason. Precedent re-read at `tests/test_project_management.py:278-281` — same mechanism (`Path("shaderbox/hotkeys.py").read_text()` + an `in source` assert) and the same docstring reason. |
| **17's focus caveat** | 17: *"Falsifier: compute the plan on selection only. This holds only because the group field is not auto-focused (D10): a focused `input_text` writes its own buffer back over the external write on the next frame, measured, and the test would then green a stale-rejection implementation."* D10: *"No field takes keyboard focus on open or on selection (an `input_text` that was given focus writes its own buffer back over an external write on the next frame, which would defeat verification 17 and any programmatic prefill); the group field is focused by a click."* | **CLOSED**, and better than asked. Round 2 offered "state which of two routes"; the spec instead made the no-auto-focus a DESIGN constraint in D10 and the item cites it, so the ambiguity is gone rather than documented. |
| **18's falsifier** (1px → 8px) | 18: *"Falsifier: drop `ChildFlags_.borders` without `always_use_window_padding` and grouped pictures sit 8px off their neighbours with 16px more room (measured: content `(8, 8)` / 152x164 bordered, `(0, 0)` / 168x180 plain)."* | **CLOSED.** The number is corrected, the cause is the flag rather than a pad, and the measurement is inline. |
| **19's by-name stamp and the window** | 19: *"stamp a group onto `paint` and `df` BY NAME (`session.set_pass_group`), two passes that are non-adjacent in Radiance Cascades' `strip_order` whether or not it has compiled (the order is name-sorted before and topological after, and the frame-48 switch away from the document closes the window)"* | **CLOSED**, and the claim is true: measured `pre = ['cascade','composite','df','jfa','paint','seed']`, `post = ['paint','seed','jfa','df','cascade','composite']`, `paint`/`df` distance **2 pre-compile, 3 post-compile** — non-adjacent in both, as the item asserts. The frame-48 window is named. |
| **the new item 20** (D11 had no gate) | 20 added, quoted in full in §B-20 | **CLOSED.** It covers both halves of D11's two predicates and names `_tick_frame_state`'s gate specifically, which is the half round 2 said nothing reads. |
| **D11's coverage** | D11 rewritten as two positive predicates (`examples_planned`, `import_project_tab`) on the render chain AND on `_tick_frame_state`'s `tick_documents` gate, with the reason the negation is wrong | **CLOSED** — and item 20's falsifier targets the `_tick_frame_state` gate, the site D11's last paragraph says is load-bearing. |
| **the chordless command** | D10: *"a palette command `IMPORT_PASSES` (chordless, palette only; a `COMMAND_SPECS` row AND an `app.command_callbacks` handler, both gated by `test_command_registry_coverage.py`)"* | **CLOSED.** Confirmed against the third test: `test_every_bound_spec_reaches_the_help_shortcuts` (`tests/test_command_registry_coverage.py:20-23`) loops `if spec.default_chord:`, so a chordless spec skips it and no help line is owed. The red run round 2 predicted is now predicted away. |
| **the `hotkeys.py` touch** | Files touched: *"`shaderbox/hotkeys.py` — the `IMPORT_PASSES` Escape branch"*; D10: *"Escape reaches `close_import_passes` through its own branch in `hotkeys.py`, the way `PASS_SETTINGS` does; the bare `popup_state = CLOSED` fallthrough would leave the draft populated."* | **CLOSED**, with item 15's source assert as the gate. The insertion point re-read at `hotkeys.py:376-388` — the new `elif` sits beside the `PASS_SETTINGS` branch in the same chain, and `escape_has_job()` already returns True on `any_popup_open()`, so no upstream gating change. |

---

## B. The changed items judged

### Item 5 — **CATCHES, both scenarios**

**Scenario 2's (i) — the brief's question, answered by measurement: route A compiles, and a test
can build the program-less `grade` with no save.** `Pass(...)` directly + `document.passes[name] =`
+ `graph.with_passes` is the route, and it leaves the pass program-less:

```
program after Pass(...)              = False   (i.e. program is None)
still program-less in doc            = True
effective_wiring UNCOMPILED          = {'main': {}, 'grade': {}}
after compile, wiring                = {'main': {}, 'grade': {'u_main': 'main'}}   errors []
```

That is the discriminator the item needs and nothing else supplies: `add_pass` (`project_session.py:901-923`)
compiles at `:916` and saves at `:922`, and every other verb ends in `save_ui_document`, whose first
act is the compile loop (`ui_models.py:409-412`). So *"the test must build `grade` without a save in
between"* is satisfiable exactly one way, and item 5(i) is right to spell the constraint out. The
item does not name the route, which is the one clause worth adding — not a gap in what the item
asserts, but the implementer has to rediscover that `add_pass` is the wrong tool:

> 5(i) … *(the program-less `grade` is built with `Pass(gl=…, source=ShaderSource.load(path),
> canvas_size=…, target=…)` written straight into `document.passes` plus a `graph.with_passes`
> entry — never `add_pass`, which compiles at `project_session.py:916` and saves at `:922`;
> measured: that route leaves `effective_wiring()` at `{'main': {}, 'grade': {}}` and a compile
> turns it into `{'grade': {'u_main': 'main'}}`.)*

**Scenario 2's (ii) is now reachable and the falsifier is real.** A `grade` whose shader has a
syntax error survives its own save with no program:

```
BROKEN after save: program-less: True   errors: True
BROKEN wiring after save: {'main': {}, 'grade': {}}
graph.json output: grade
doc.json grade uniform_values: {}
```

`save` ran its compile loop, the compile failed, and the `program is None` branch at
`ui_models.py:484-488` carried the (empty) disk rows forward — exactly D6's stated hazard. So a
handover onto `grade` would be silently dropped, and the item asserting the REJECTION is the right
gate. D6 and item 5(ii) now say the same thing, which is what round 2 asked for.

### Item 7 — **CATCHES** (was the dead gate)

The new falsifier fires, measured both directions on a real `app` with a compiled extra pass
carrying `uniform float u_amount`:

```
right key: 233662036186801150087314693372063320716
wrong key:  53504612896050506898259832067202961652   (md5 of "bloom_x_u_amount_1_1_35678")
wrong key survived in memory: False
rows on disk: [37154164850612245702881750831795076973]     # the starter's own row only
--- the correct key, same setup ---
right key on disk: drag
```

The mechanism is `live_rows` at `ui_models.py:462-471`: it is built from `get_uniform_hash(u)` over
the live programs (`util.py:78-85`, keyed `f"{name}_{array_length}_{dimension}_{gl_type}"` — no pass,
no document), and any row keyed otherwise is not in that set and is popped in the same save that
wrote it. So "re-key by the new pass name" is red and "copy under its own key" is green. Live gate.

### Item 8 — **CATCHES**

Both falsifiers are decidable and the second is the one round 2 asked for. Re-confirmed that
`Document.load_from_dir` loads and defers rather than refusing (`document.py:860-892`, docstring:
*"A pass file that cannot be read costs THAT pass, never the document"*; it catches `OSError` only
and never calls `compile()`), so the fixture is one `copytree` plus one bad line, and the
`entry_points` assert reads `['blur', 'scene']` against the healthy `['scene']`.

### Item 9 — **CATCHES, home now correct**

The split is right by fixture. `tests/test_graph_persistence.py` has 17 tests, none taking `app`
(`grep -n "def test_"`), and `_write_document` at `:65` is the loader-driving tool clauses 3–4 need;
`tests/test_pass_verbs.py` has the `app` fixture and drives rename/delete through the verbs, which
clauses 1–2 need. The falsifier (`_graph_renamed` rebuilt from a fresh `PassEntry()`) is the real bug
class and unchanged.

### Item 11 — **FAILS AS WRITTEN.** See the verdict. Item 11(a) is sound:

```
bloom    crc32%4 = 3   hash%4 = 2
radiance crc32%4 = 2   hash%4 = 0
fx       crc32%4 = 0   hash%4 = 3
```

Three literal pins, all three wrong under `hash()` in this process. (Under `% 3` after fix (A) the
same reasoning holds; the literals change.)

### Item 15 — **CATCHES**

Both the funnel and the branch. The source-substring half is the precedent's exact shape
(`tests/test_project_management.py:278-281`) and its stated reason still holds — the font atlas is
process-global, so a second frame-driving App cannot be the gate. `tests/test_pass_draft.py` +
`test_command_registry_coverage.py` + the three pure files run green today (41 passed, 1.40s).

### Item 17 — **CATCHES**

The caveat is now a design constraint in D10 rather than a note on the test, which removes the
failure round 2 measured. The rig transfers: `_gear_sizes` (`tests/test_pass_settings_layout.py:23-46`)
is `imgui.new_frame()` / draw / `imgui.end_frame()` in a loop, and `draft.rejection` lives on the
draft so no monkeypatched recorder is needed. The falsifier "compute the plan on selection only"
goes red on frame two given D10's no-auto-focus rule. The one place the rule has to hold at
implementation time is `popups/import_passes.py` — `pass_settings.py:78-80` shows the sibling modal
doing the opposite (`if draft.needs_focus: imgui.set_keyboard_focus_here(0)`), so the copy-paste
hazard is real and D10 names it.

### Item 18 — **CATCHES**

The numbers are right and the ingredients are measurable from a bare frame
(`imgui.get_cursor_screen_pos()` / `get_content_region_avail()` inside the child, the way
`test_pass_settings_layout.py:84-94` measures text widths). Round 2's measurement — `(8,8)`/152x152
bordered vs `(0,0)`/168x168, and `always_use_window_padding` restoring byte-identity — is what the
item now asserts.

### Item 19 — **CATCHES**

The by-name stamp is correct and the `paint`/`df` choice is safe in both orders (measured above).
Two further confirmations: only ONE shipped example is multi-pass (`77a84d27-…`, Radiance Cascades),
so frame 42's `next((i for i, u in app.ui_documents.items() if len(u.document.passes) > 1), "")`
(`scripts/smoke.py:256-266`) cannot pick anything else, and the smoke's own `script_document` has one
pass (`_seed_tmp_project`, `:104-128`); and the smoke is green today —
`uv run python scripts/smoke.py` → `EXIT=0`, `smoke: OK (200 frames, 7 documents)`, about four
seconds. `set_pass_group` saves, so the stamp rewrites `graph.json` inside the smoke's throwaway tmp
project, never the tracked resources.

### Item 20 — **CATCHES**, and the brief's question is yes

> 20. **The planned set under the dialog** (`tests/test_render_decoupling_loop.py`'s shape, which
> sets `popup_state` directly and counts renders): with `IMPORT_PASSES` open on the Examples tab,
> the examples render and the project's non-current documents do not; on the project tab,
> `tick_documents` holds the render-all documents and the pending-first election runs. Falsifier:
> leave `_tick_frame_state`'s gate untouched and the project-tab case renders only the current
> document, so a never-rendered card stays black.

**`_count_renders` can count project documents as well as examples.** Its signature is
`_count_renders(app, monkeypatch, documents: dict[str, Any])` (`tests/test_render_decoupling_loop.py:54-79`)
and it builds `by_object = {id(ui_document.document): document_id for …}` from whatever dict it is
handed, then patches `render` on `type(sample)` — the Document class, shared by examples and project
documents alike. The file already calls it **both** ways: `:153` with `app.ui_documents` and `:394`
with `app.ui_document_examples`. So no new helper is needed for either half.

The project-tab half is also buildable: the fixture seeds the starter only, and
`tests/conftest.py:36-44`'s `seed_extra_document(app, new_id)` is already imported at the top of that
module (`:29`) for exactly this. `app.app_state.is_render_all_documents` defaults to `True`
(`ui_models.py:247`), so the render-all clause is live without a flag flip. `tests/test_render_decoupling_loop.py`
is green today (26 passed, 4.76s).

One clause the item would be sharper with, since the second document is not there by default:

> 20. … *(the project tab's case needs a second project document: `seed_extra_document(app, "extra")`,
> already imported in that module; `is_render_all_documents` defaults True, so no flag flip. Count
> with the existing `_count_renders`, handed `app.ui_documents` for the project tab and
> `app.ui_document_examples` for the Examples tab — it takes any document dict and patches `render`
> on the shared Document class, and the file already calls it both ways at `:153` and `:394`.)*

That is a convenience, not a gap: the item as written is decidable and its falsifier fires.

---

## C. New findings

**None.** Every candidate was re-read against the item that would cover it and each was already
covered:

- *Does item 20's project-tab clause depend on a flag the fixture leaves off?* No —
  `is_render_all_documents: bool = True` (`ui_models.py:247`).
- *Does frame 42 of the smoke risk picking a different multi-pass document now that the project is
  seeded with every example?* No — one example is multi-pass, measured.
- *Does the chordless command trip `test_every_bound_spec_reaches_the_help_shortcuts`?* No, it
  guards on `if spec.default_chord:` (`tests/test_command_registry_coverage.py:20-23`), and D10 now
  says chordless.
- *Does item 5's route-A construction need a `graph` entry to be visible to `effective_wiring`?*
  It needs both the `passes` write and the `with_passes` entry; the measurement above did both and
  the item's phrasing ("gives the host a second pass") covers it.

---

## False trails

Round 1's and round 2's are settled per the brief and were not re-checked. New to this round:

- **`_count_renders` being examples-only.** It is not — it takes any `dict[str, UIDocument]` and the
  same file already passes both `app.ui_documents` and `app.ui_document_examples`. Item 20 needs no
  new counting helper.
- **Needing `add_pass` to build item 5(i)'s program-less host pass.** Not usable —
  `project_session.py:901-923` compiles (`:916`) and saves (`:922`). The direct `Pass(...)` +
  `passes[name] =` + `with_passes` route is the one that works, measured.
- **`orange_n` as the fourth tint once `aqua_n` is dropped.** It is the orange accent's ACTIVE
  (`_ACCENTS["orange"] = (orange_b, orange_n, …)`, `theme.py:98`), so 11(b) rejects it for the same
  reason. Round 2's report named it as free; that was against the then-narrower assert. The only
  chromatic hues free of 11(b)'s widened set are `red_n`, `green_n`, `green_b`, `yellow_n`,
  `purple_n`.
- **`test_graph_persistence.py` gaining an `app` fixture since round 2.** It has not — 17 tests,
  none taking `app`. Item 9's split is still the right resolution.
