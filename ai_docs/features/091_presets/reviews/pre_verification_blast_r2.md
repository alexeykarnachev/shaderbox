# 091 pre-implementation review round 2 — verification & blast radius

Reviewer role: does the revised `## Verification` list (19 items) catch a wrong implementation of
each locked decision; are round 1's findings closed; what do the two new "Files touched" entries
touch. Read-only — nothing under `shaderbox/`, `tests/` or the spec was edited, and no command
rewrote the working tree. Every claim below is a file read with line numbers or a command with its
output.

**Note on the anchor.** The spec was revised again while this review was being written (D1 now
moves `PASS_NAME_RE` to `pass_graph.py`; D7 carries the 8px / `always_use_window_padding`
correction; D8 dropped to four tints over the `purple_b`/`COLOR.SELECT` collision; item 11's
falsifier changed to match). Three findings this reviewer reached independently had therefore
already landed by the time the report was filed, and they are recorded below as **landed
independently** rather than as asks. Everything judged below is against the spec as it reads now.

## Verdict: **PARTIAL**

Round 1 closed cleanly: all eleven old items and all sixteen B/C findings are addressed, eight of
them by new numbered items. Fifteen of the nineteen items catch their decision. **Four need
rewriting**, two of them because the gate does not fire — it passes whether or not the
implementation is right, which is the failure round 1 called out for old item 7 and which has
recurred in a new place.

| Item | Verdict | One-line reason |
| --- | --- | --- |
| 5 (second scenario) | **WEAK — falsifier (ii) is unreachable** | `UIDocument.save` compiles every program-less pass at `ui_models.py:409-412`, seventy lines BEFORE the `program is None` branch at `:484-488`, so no verb can leave `grade` program-less. Measured. |
| 7 (merged row) | **DEAD GATE** | Its falsifier ("skip `compile()` and the prune drops the row") does not fire, for the same reason. Measured: a never-compiled pass's merged row reaches disk intact. D5 carries the same wrong claim. |
| 11 (group tints) | CATCHES, **one hole in the hand-enumerated list** | D8's new `green_b` IS `SYN_BUILTIN` / `SYN_SCRIPT_UNIFORM` / `SYN_STRING`, and 11(b)'s enumerated token set does not include `SYN_*`. Measured: `green_b in <11(b)'s set>` is False. |
| 17 (per-frame replan) | **WEAK — transfers only for an UNFOCUSED field** | Measured: once `set_keyboard_focus_here` has run on an `input_text`, an externally-set buffer is clobbered back to imgui's state on the NEXT frame, `needs_focus` already False. An unfocused field keeps the write across three frames. |
| 19 (smoke) | **WEAK — a five-frame window and a compile-dependent order** | Frame 48 switches away from `multi`, so the group draws frames 43–47; and RC's `strip_order` differs pre- and post-compile, so a stamp picked by index names a different pair than intended. Both measured. |
| 8 (non-compiling source) | CATCHES, **missing its sharpest assert** | Runnable — `load_document_from_dir` loads a broken dir and defers (measured). But a broken pass becomes a SPURIOUS ENTRY POINT, growing the dialog a row, and nothing reads that. |
| 1, 2, 3, 4, 6, 9, 10, 12, 13, 14, 15, 16, 18 | CATCHES | See §B; item 9's HOME is wrong though its verdict stands. |

Plus **one decision with no verification item at all**: D11's planned-render-set predicate (§B,
D11).

---

## A. Round 1, closed item by item

Old numbering → new numbering, with the closing text quoted.

### Old items 1–11

| Old | New | Status | The text that closes it |
| --- | --- | --- | --- |
| 1 `entry_points` WEAK | **1** | **CLOSED** | New 1: *"a wiring whose ONLY edge is a self-read (`{"acc": {"u_prev": "acc"}}`) answers `["acc"]` … The self-read case must be its own wiring: in both real shapes every self-reader also reads a sibling, so the bug is invisible there."* Taken verbatim including the reason. |
| 2 `plan_import` CATCHES | **3** | **CLOSED** (carried, plus the note) | New 3(a): *"including the self-read written as `PassSource("bloom_trail")`"* — round 1 asked for exactly that, since `u_prev` self-resolves today and is the one sampler that works either way. |
| 3 `import_passes` e2e MISSING FIXTURE | **5** | **CLOSED**, all four defects | 3a (no shipped Bloom Chain) → new 5: *"the five-pass bloom fixture copied into `tmp_path` and loaded with `load_document_from_dir` (as `test_lazy_compile.py` and `test_default_wiring.py` do; it is not in `app.ui_documents`)"*. 3b → *"the four copied files beside the host's own"*. 3c → *"`bloom_bright.u_scene`, `bloom_composite.u_blur`"*, the 069 W-D spellings. 3d → the parenthetical above. |
| 4 rendered import WEAK | **4** | **CLOSED** | New 4: *"a host whose `main` renders a known constant … the output canvas's red is the value the bundle produces from that constant"* plus *("Above the starter's black" would not do: the starter is UV Mango, a gradient.)* |
| 5 group survives verbs CATCHES | **9** | **CLOSED as a verdict, but the HOME is now wrong** | New 9 keeps the falsifier (*"`_graph_renamed` rebuilt from a fresh `PassEntry()`"*) and the home `tests/test_graph_persistence.py`. That file has **no `app`-fixture test** (`grep -n "def test_"` → 16 tests, none taking `app`), so "rename keeps it, delete drops it" cannot be driven through the real verbs there. See §B-9. |
| 6 `group_runs` CATCHES | **10** | **CLOSED** | Unchanged, same home, same falsifier. `tests/test_pass_strip_layout.py` has zero fixtures and imports only `theme` + `tiles_per_row` (read; confirmed). |
| 7 theme invariant FAIL AS WRITTEN | **11** | **CLOSED** | Both halves taken. 11(a) pins by value at crc32 literals; 11(b) is the disjointness sweep; and D8 states the reason outright: *"the GATE is a pure test over the tuple (verification 7), because an import-time assert cannot be tripped from a test without rewriting `theme.py` on disk."* (D8's cross-reference says "verification 7"; the item is **11** — one-word fix.) |
| 8 copilot table CATCHES | **12** | **CLOSED** | Unchanged. D9 folded round 1's implementer note: *"APPENDED to the positional-only signature … (the five positional test call sites lengthen; an inserted parameter would break them)"*. Verified: `grep -c "set_pass(" tests/test_copilot_pass_tools.py` → 5, each nine positional args. |
| 9 command registered CATCHES (two gates) | **13** | **CLOSED** | New 13 names both halves: *"fails on a `CommandId` with no `COMMAND_SPECS` row and, separately, on one with no `app.command_callbacks` handler."* |
| 10 modal state resets CATCHES + the second reset | **15** | **CLOSED** | New 15 carries both resets: *"select a second source: `handovers` is empty and the group buffer is the second slug; re-pick a host pass: `handovers` is that pass's readers"*. |
| 11 smoke WEAK | **19** | **CLOSED as text, WEAK as a check** | New 19 takes the rewrite: *"stamp a group onto two NON-adjacent members in `strip_order` … today no smoke document carries a group, so the path runs zero times."* Two mechanical problems remain — §B-19. |

### B-0 through B-10

| Finding | Closed by | Status |
| --- | --- | --- |
| **B-0** D10's draw wiring is already gated | **item 14** | **CLOSED.** New 14: *"goes red on a `PopupState` member whose draw is not called in `ui.py` (7 called == `len(PopupState) - 1` today)."* Re-verified by running the same AST walk: 7 imported, 7 called, `len(PopupState) - 1 == 7`. D10 names the gate inline too. |
| **B-1** group survives `sync_documents_from_disk` | recorded as a false trail | **CLOSED.** Review history: *"`load_graph` rides the new field"*. Round 1 itself downgraded this to no-gap. |
| **B-2** import while the copilot is mid-turn | **item 16** + D10 | **CLOSED.** D10: *"`open_import_passes` refuses while `app.copilot_turn_active`, through `_copilot_busy_blocked`, since the palette route does not pass through the strip's `begin_disabled`."* `_copilot_busy_blocked` read at `app.py:919-927` — notifies and returns True, so both of item 16's asserts are readable. |
| **B-3** the D4 rejection recomputed each frame | **item 17** + D10 + `ImportDraft.rejection` | **CLOSED as text, WEAK as a check** — §B-17. D10 says *"recomputed each frame the dialog draws (never cached on selection)"*, and `ImportDraft` carries *"`rejection: str` … stored so a test can read what the button read"* — exactly the seam round 1 asked for, and better than it asked for (no monkeypatched recorder needed). |
| **B-4** the `ImportDraft` reset on Escape | **item 15** + `## Files touched` | **CLOSED.** `shaderbox/hotkeys.py` is in Files touched (*"the `IMPORT_PASSES` Escape branch"*), D10 gives the reason (*"the bare `popup_state = CLOSED` fallthrough would leave the draft populated"*), and item 15's falsifier is *"rely on the `hotkeys.py` fallthrough and the draft survives Escape."* One gap in how it is gated — §C. |
| **B-5** handover rows surviving `UIDocument.save` | **item 5's second scenario** + D6 | **HALF CLOSED.** Falsifier (i) is real and D6 states it with the measurement. Falsifier (ii) is **unreachable** — §B-5. |
| **B-6** a source pass that failed to compile | **item 8** + D3 + `ImportResult.notes` | **CLOSED**, one assert missing — §B-8. `ImportResult.notes: tuple[str, ...]` is now a named field, so the notification is testable rather than a toast. |
| **B-7** the source is in the read-only resources dir | **item 6** | **CLOSED, and improved.** Round 1 said "mtime + content"; the spec dropped mtime for *"content of `graph.json`, `document.json` and every `passes/*.glsl`"*. That matters — §B-6. |
| **B-8** the `ui_uniforms` merge | **item 7** + D5 | **CLOSED as text, DEAD as a gate** — §B-7. D5 carries the precedence rule verbatim; the falsifier does not fire, and D5 repeats the same wrong claim. |
| **B-9** a non-contiguous group | **item 10** + D7 | **CLOSED.** D7: *"by adjacency and never by name, so a group split by an outside pass is two runs."* |
| **B-10** `group_slug` | **item 2** | **CLOSED.** New 2 promotes it to a numbered item with round 1's falsifier. Confirmed: `re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$").match("2d")` is None, `("g_2d")` matches. D2 added a fourth input (`""` → `preset`) and the 168px/21-char reason for the first word. |

### C-1 through C-5

Settled per the brief; not redone. Two of the three things round 1 flagged "for the spec, not
verification" did land: D1's closing sentence (*"Two sites compare against `PassEntry()`'s field
defaults … and must not be read as 'is this entry default' once `group` exists"*) and D9's
create-mode row (*"In create mode the row edits `draft.entry.group` and `create_pass_from_draft`
applies it through `set_pass_group` after `add_pass`"*). D7 took C-3's three corrections; the
compensation's SIZE has since been corrected in D7 too (§B-18).

---

## B. The 19 items judged

### Item 1 — `entry_points` — CATCHES

Round 1's rewrite taken whole, including the reason the self-read needs its own wiring. Pure, in a
zero-fixture home (`grep -c fixture tests/test_pass_graph.py` → 0). The three pure test files run
together in 0.59s (`uv run pytest tests/test_theme.py tests/test_pass_strip_layout.py
tests/test_pass_graph.py -q` → 35 passed), so items 1, 2, 10 and 11 cost nothing.

### Item 2 — `group_slug` — CATCHES

New this round. Four inputs, pure, and the falsifier is decidable:

```
$ uv run python -c "import re; RE=re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$'); print([(c, bool(RE.match(c))) for c in ['bloom','radiance','2d','g_2d','preset']])"
[('bloom', True), ('radiance', True), ('2d', False), ('g_2d', True), ('preset', True)]
```

A `g_`-less implementation makes `2D SDF` reject at plan time over a name the user never typed.

**Landed independently.** This reviewer's draft flagged a layering problem — `_PASS_NAME_RE` lived
at `project_session.py:124` while `pass_graph.py` imports neither `re` nor any name pattern
(`grep -n "^import\|^from\|NAME_RE\|re\." shaderbox/pass_graph.py` → only `collections.abc`,
`dataclasses`, `typing`, `pydantic`), so D4's *"imports `pass_graph` only"* and D4's rejection on
*"a group name failing `PASS_NAME_RE`"* could not both hold. D1 now says exactly that and moves the
regex: *"That regex moves from `project_session.py` to `pass_graph.py` and loses its underscore …
both sit at or below `pass_graph` in the import order, so it cannot stay private to a module that
imports them."* Closed.

### Item 3 — `plan_import` — CATCHES

Eight lettered sub-cases, each decidable from `ImportPlan`'s five fields, in a new pure file. No
fixture. Round 1 verified (a)'s falsifier by measurement (`wired_pass(AutoSource(), "u_bright",
"bloom_blur", …)` is `None`) and D4 now carries that measurement inline, so the item and the design
text agree.

### Item 4 — A rendered import reads its bundle — CATCHES, one mechanical mismatch

The home is right: `tests/test_document_graph.py`'s `gl_ctx` is a module-scoped
`moderngl.create_standalone_context()` (`:71-84`, with the one-recipe-per-process warning) and
`_document` (`:87-113`) builds a document from raw GLSL strings, compiling each pass and asserting
no errors. Cheaper than `app` and the right home for a constructed host.

**The mismatch:** `import_passes` is a `ProjectSession` method whose first line is
`self.ui_documents.get(document_id)` and which writes through
`self.paths.pass_shader_for(document_id, name)` — every one of the six existing verbs does
(`project_session.py:901`, `:926`, `:947`, `:978`, `:1015`, `:1027`). A `_document`-built `Document`
is in no session and has no project dir, so it cannot be the host of a `session.import_passes`
call. Item 4 therefore either drives the plan's execution by hand (build the passes, apply
`ImportPlan.sources` as `PassSource` rows, render) or moves to a registered document. Say which, in
one clause, or the implementer discovers it at the call site:

> 4. … *(the host is built with `_document` and the plan applied by hand — `session.import_passes`
> needs a document registered in `ui_documents` with a project dir, which a `_document` host is
> not; this item checks that the PLAN's rows render, which is the half `test_pass_verbs.py`'s
> end-to-end cannot see in pixels).*

### Item 5 — `import_passes` end to end — CATCHES (scenario 1), **WEAK (scenario 2)**

**Scenario 1 is sound.** `tests/test_pass_verbs.py` has the `app` fixture and `_reload` at `:72-73`
(`load_document_from_dir(app.session.paths.documents_dir / document_id)`), and rename/delete are
verb-driven there (`:110`, `:149`, `:171`).

**Scenario 2's verbs — the brief's question, answered.** Yes, the starter allows it, through two
existing verbs plus one direct source write (there is no verb for a pass's shader TEXT; the editor
writes the file and a test writes the text):

1. `app.session.add_pass(document_id, "grade")` — `project_session.py:901`.
2. `document.passes["grade"].release_program(GRADE_SRC)` to give it a `u_main` sampler.
3. `app.session.set_output_pass(document_id, "grade")` — `project_session.py:978`.

Measured end to end:

```
starter passes: ['main'] output: main
add_pass -> ''
grade program after add_pass: True
grade program after release_program: False
set_output_pass -> ''
output now: grade
grade program AFTER the save inside set_output_pass: True
wiring (after compile): {'main': {}, 'grade': {'u_main': 'main'}}
```

**That second-to-last line is the defect.** Falsifier (ii) says *"leave `grade` program-less at save
time and `UIDocument.save` carries its disk rows forward, dropping the handover."* It cannot be
arranged. `UIDocument.save` compiles every program-less pass at the TOP of the method
(`ui_models.py:409-412`):

```python
for render_pass in self.document.passes.values():
    if render_pass.program is None:
        render_pass.compile()
```

The `program is None` branch that carries disk rows forward is at `:484-488`, seventy lines later.
So the only pass reaching it is one whose **source does not compile** — a broken shader, not a
never-rendered one. Every session verb ends in `save_ui_document` → `save()`, so "never rendered"
does not survive a single verb call.

> **Rewrite of scenario 2's falsifier (ii).** *"(ii) give `grade` a shader that does NOT compile:
> `UIDocument.save`'s own compile loop (`ui_models.py:409-412`) leaves its program None, the
> `program is None` branch (`:484-488`) carries its disk rows forward, and the handover is dropped
> with no error — which is why D6 rejects a handover onto a host pass that does not compile, so the
> test asserts that REJECTION by message rather than the dropped row. A never-rendered host pass is
> not a reachable state for (ii): `save` compiles every program-less pass before it reads one, and
> every verb saves (measured: `add_pass` → `release_program` → `set_output_pass` leaves
> `grade.program` not None)."*

D6's own sentence (*"a host pass whose compile fails cannot take a handover (its rows would be
carried forward from disk by `UIDocument.save`'s `program is None` branch …)"*) is the accurate
statement; scenario 2's (ii) contradicts it. The design text is right and the verification text is
what drifted.

### Item 6 — the source stays untouched — CATCHES

The brief's three questions, by measurement.

**Where do the examples live for the fixture?** In the **real tracked resources directory** — the
`app` fixture does not copy them. `conftest.py:26-34`'s `seed_tmp_project` copies only
`DOCUMENT_EXAMPLES_DIR / STARTER_EXAMPLE_ID` into `tmp_path`; the other five arrive through
`ProjectSession._load` → `load_documents_from_dir(self._document_examples_dir)`
(`project_session.py:506-508`), and `App` passes `document_examples_dir=DOCUMENT_EXAMPLES_DIR`
(`app.py:366-368`), which is `RESOURCES_DIR / "document_examples"` (`constants.py:8`). Measured
through a real App, `document_dir_of(u.document)` per example:

```
53724dbd RESOURCES ['document.json', 'graph.json', 'passes']
73ea2431 RESOURCES ['document.json', 'graph.json', 'media', 'passes']
f90f5ff9 RESOURCES ['document.json', 'graph.json', 'passes']
0b0d16bb RESOURCES ['document.json', 'graph.json', 'passes']
8d454b7b RESOURCES ['document.json', 'graph.json', 'passes']
77a84d27 RESOURCES ['document.json', 'graph.json', 'passes']
```

**Does the fixture copy them or point at the resources dir?** Points. So item 6's falsifier — a
`save_ui_document` on the source — really does dirty the tracked tree, which is what makes the
check worth having.

**Is byte-identity runnable?** Yes, and content-only is the right spelling, which the spec already
uses. Building a real App over all six examples and compiling RC's six passes leaves them
untouched:

```
before=b6b8c70e9da61c8c3e5f6a51b14aa30c  -
after =b6b8c70e9da61c8c3e5f6a51b14aa30c  -
CONTENT IDENTICAL
```

(`find shaderbox/resources/document_examples -type f -exec md5sum {} \; | sort | md5sum`, and
`git status --porcelain shaderbox/resources/` clean.)

**One hazard the item should name.** Content-only is load-bearing, not a style choice: the examples'
`document.json` mtimes are already a day newer than their `graph.json` while git reports clean
(`stat -c %Y` on all six → `document.json` 1789141814 / 1789143220 vs `graph.json` 1788967842), so
an mtime assert would be red from the first run. And this is the one test in the suite that reads
the live working tree, so a failure leaves it dirty — the assert fires after the damage.

> 6. **The source stays untouched** (`tests/test_pass_verbs.py`): … byte-identical afterwards
>    (CONTENT of `graph.json`, `document.json` and every `passes/*.glsl`; **never mtime** — the
>    shipped examples' `document.json` mtimes already differ from their `graph.json` while git
>    reports clean, so an mtime assert is red from the first run). The examples are the REAL
>    tracked `shaderbox/resources/document_examples/`: the `app` fixture copies only the starter
>    into `tmp_path` and loads the other five from resources (`conftest.py:26-34` +
>    `project_session.py:506-508` + `app.py:367`), so this is the one test that reads the working
>    tree and a failure leaves it dirty. Falsifier: unchanged.

### Item 7 — a merged row survives the save — **DEAD GATE**

The brief asks for the source uniform, the input type, and whether the prune would drop it without
the compile. The first two are straightforward; the third is **no**, and that kills the falsifier.

**The uniform and input type a test would set.** Declare `uniform float u_amount;` on a source
pass, compile it to get the live `moderngl.Uniform`, take `get_uniform_hash(u)` (`util.py:78-85`,
keyed by `f"{name}_{array_length}_{dimension}_{gl_type}"` — name and shape, no pass, no document,
which is why no re-keying is needed across the rename), and set

```python
ui_state.ui_uniforms[h] = UIUniform(name="u_amount", gl_type=…, dimension=1,
                                    array_length=1, input_type="drag")
```

`"drag"` is a non-default `UIUniformInputType` (the default is `"auto"`, `ui_models.py:73`), so its
survival is one equality assert after reload.

**Would the prune drop it without the compile? No.** Measured directly — a document with a
deliberately never-compiled pass carrying `u_amount`, a merged row for it, then one `save`:

```
copied program before save: False
row set, hash 233662036186801150087314693372063320716 -> input_type drag
copied program AFTER save: True
row survived in memory: True name='u_amount' … input_type='drag'
rows on disk: {'233662036186801150087314693372063320716': 'drag'}
```

The caller never compiled the copy, `save` compiled it (`ui_models.py:409-412`), `live_rows`
(`:465-471`) saw its `u_amount`, and the row reached disk. So the test passes under both
implementations — the "a gate that passes whether or not it works" shape round 1 flagged for old
item 7.

> **Rewrite.** *7. **A merged row survives the save** (`tests/test_pass_verbs.py`): give the source a
> `uniform float u_amount` with `input_type="drag"` in its `ui_state.ui_uniforms` (keyed by
> `get_uniform_hash`, which is name-and-shape only — `util.py:78-85` — so the copy needs no
> re-keying), import, reload, assert the host's row for that hash has `input_type == "drag"`.
> Falsifier: **skip the `ui_uniforms` merge** and the reloaded row is absent or back to `"auto"`,
> so the imported pass loses its input type and range.*
>
> *Not "skip `compile()` on the copied passes": `UIDocument.save` compiles every program-less pass
> at `ui_models.py:409-412` before `live_rows` is built at `:465-471`, so the prune never sees an
> uncompiled copy and that falsifier cannot fire (measured — a never-compiled pass's merged row
> reaches disk intact). D5's "compile it exactly as `add_pass` does" stays for the reason
> `add_pass` has it, but the GATE for the merge is the merge's own absence.*

**D5 carries the same wrong claim** and should lose it: *"build a `Pass` from that path with the
source entry's target and **compile it** (exactly as `add_pass` does; a never-compiled copy has no
live uniforms, so `UIDocument.save`'s `ui_uniforms` prune would drop every merged row in the same
save that wrote it)"*. The parenthetical's reason is false; the instruction is still right.

### Item 8 — a source pass that does not compile — CATCHES, one assert missing

**The brief's question: does `load_document_from_dir` refuse such a dir, or load and defer?** It
loads and defers, by design. `Document.load_from_dir` (`document.py:877-892`) builds one `Pass` per
shader FILE from `ShaderSource.load` and catches only `OSError` ("Skipping unreadable pass"); it
never calls `compile()`. Compiles are lazy (066 D1), and the docstring at `:863-865` states the
posture: *"A pass file that cannot be read costs THAT pass, never the document."* Measured on a
deliberately broken copy of the bloom fixture:

```
LOADED OK. passes: ['blur', 'bright', 'composite', 'scene', 'trail']
blur program is None -> True          (every pass: lazy)
blur after compile: program False errors True
bright after compile: program True errors False
effective_wiring: {'blur': {}, 'bright': {'u_scene': 'scene'},
                   'composite': {'u_blur': 'blur', 'u_scene': 'scene', 'u_trail': 'trail'},
                   'scene': {}, 'trail': {'u_prev': 'trail', 'u_scene': 'scene'}}
```

So the item is runnable, the fixture is one `shutil.copytree` plus one appended bad line, and the
`ImportResult.notes` assert is decidable.

**What the item misses.** That wiring has `'blur': {}`, and under D3's rule `blur` is now a ROOT:

```
entry_points under D3 rule: ['blur', 'scene']     # broken blur
healthy:                    ['scene']
```

A broken source pass therefore **grows the dialog an extra entry-point row**, offering a
substitution decision for a pass that is merely broken. That is a user-visible consequence of D3's
"contributes its explicit rows only", it is free to assert in the same test, and nothing in the 19
items reads it.

> Add to item 8: *"and the plan sees TWO entry points, not one — a pass whose compile failed
> contributes no edges, so it reads as a root (measured: the broken bloom fixture's wiring is
> `{'blur': {}, …}` and `entry_points` answers `['blur', 'scene']` against the healthy
> `['scene']`). Assert the dialog's row set, so the degraded case is visible rather than silently
> changing what the user is asked."*

### Item 9 — the group survives every existing verb — CATCHES, wrong home

Verdict unchanged from round 1: the falsifier is the real bug class and the code is already immune
for a documented reason (`_graph_renamed` at `project_session.py:143-150` re-keys the existing
entry object; `with_passes`' docstring states the principle).

**But `tests/test_graph_persistence.py` cannot host the first two clauses.** It has no `app`
fixture — `grep -n "def test_"` gives 16 tests, every one taking `tmp_path` and/or its own
`gl_ctx`, none taking `app` — and `rename_pass` / `delete_pass` are `ProjectSession` methods
needing a registered document. The file's own tools are `_write_document` + `load_document_from_dir`
+ `UIDocument.save`, which cover the third and fourth clauses (*"a reload reads it, a `graph.json`
without the key loads as `""`"*) perfectly.

Two ways out, both workable:

- **Split.** Clauses 3–4 stay in `test_graph_persistence.py`; clauses 1–2 move to
  `tests/test_pass_verbs.py`, where rename/delete already run through the verbs (`:110`, `:149`,
  `:171`) and where item 5 already lives.
- **Call the helpers directly.** `_graph_renamed` and `_graph_without` import cleanly from
  `shaderbox.project_session`, and the repo has ample precedent for importing privates into tests
  (`test_completion.py:22` → `hotkeys._is_lookup_key`; `test_anchored_note.py:15` →
  `ui_primitives._ellipsize`; six more). That keeps the item in one file but tests the helper
  rather than the verb.

> 9. **The group survives every existing verb**: rename keeps it and delete drops it
>    (`tests/test_pass_verbs.py`, the `app` fixture — `test_graph_persistence.py` has no `app`
>    fixture and the two verbs are `ProjectSession` methods); a reload reads it and a `graph.json`
>    without the key loads as `""` (`tests/test_graph_persistence.py`'s `_write_document` +
>    `load_document_from_dir`). Falsifier: `_graph_renamed` rebuilt from a fresh `PassEntry()`.

### Item 10 — `group_runs` — CATCHES

Unchanged. `tests/test_pass_strip_layout.py` imports only `theme.SIZE/SPACE` and
`widgets.pass_list.tiles_per_row`, has zero fixtures, and already pins `tiles_per_row` against two
named falsifiers — the exact style `group_runs` wants.

### Item 11 — group tints — CATCHES, **one hole in the hand-enumerated list**

**Landed independently, half of it.** This reviewer's draft reported that D8's then-current five
tints collided twice: `purple_b` IS `COLOR.SELECT` (`theme.py:161`) and `blue_n` IS
`COLOR.STATE_INFO`. D8 has since dropped to **four** (`purple_n`, `green_b`, `yellow_n`, `aqua_n`),
names both collisions as the reason, and item 11(b) now enumerates *"the accent primaries, the
accent actives, every `STATE_*` hue, `SELECT`, `TAG` and `FAVS"*` with `purple_b` as the falsifier.
Verified against the code: the list is right, `aqua_n` is indeed the aqua accent's ACTIVE and not a
primary, and `_accent_primaries` does enumerate element [0] only:

```
yellow [['yellow_b'], ['orange_b'], (…, 0.18)]
aqua   [['aqua_b'],   ['aqua_n'],   (…, 0.18)]
orange [['orange_b'], ['orange_n'], (…, 0.18)]
blue   [['blue_b'],   ['blue_n'],   (…, 0.22)]
```

(a) holds. Two runs in two processes:

```
bloom crc32%4 = 3  ;  hash%4 = 3 then 0       # crc32 identical, hash() moves
radiance crc32%4 = 2
fx crc32%4 = 0
```

Note `% 4` weakens (a) slightly: three pinned names under `hash()` go green when all three happen
to land, about 1 run in 64 rather than round 1's 1 in 216. Still red on essentially every run, as
the item claims.

**The remaining hole.** `green_b` is not free — it is three syntax-highlight tokens, and 11(b)'s
enumeration does not cover `SYN_*`:

```
purple_n -> SYN hits: []
green_b  -> SYN hits: ['SYN_BUILTIN', 'SYN_SCRIPT_UNIFORM', 'SYN_STRING']
yellow_n -> SYN hits: []
aqua_n   -> SYN hits: []

green_b in <11(b)'s enumerated token set>: False
```

So the gate passes on a tint that is the editor's string/builtin colour. Whether that collision
MATTERS is a design call (the editor and the strip never share a surface, unlike `SELECT`'s nested
outlines), but a hand-enumerated allowlist is the shape this repo has been bitten by before —
`ui_models.py:580-586` records `_reset_out_of_range_values` growing an allowlist one field at a
time, and the fix there was to enumerate the model instead. The same fix applies:

> 11(b). `set(GROUP_TINTS)` is disjoint from **every 4-tuple colour token on `COLOR`** except the
>    `GROUP_TINTS` entry itself, and from the accent actives (which `_accent_primaries` does not
>    carry — it enumerates element [0] of each preset only, verified), and has no duplicate.
>    Enumerate the tokens rather than listing them by hand: `{getattr(COLOR, k) for k in dir(COLOR)
>    if not k.startswith("_") and isinstance(getattr(COLOR, k), tuple)}`, which is how a new token
>    added tomorrow joins the sweep instead of being forgotten. Falsifier: `purple_b`
>    (`COLOR.SELECT`), which a check over accent primaries and state hues alone lets through — and
>    `green_b` (`SYN_BUILTIN` / `SYN_SCRIPT_UNIFORM` / `SYN_STRING`), which even the enumerated
>    `STATE_*`/`SELECT`/`TAG`/`FAVS` list lets through, so D8 either drops it for `orange_n` or
>    states that a syntax hue is allowed and why.
>
> Free hues measured against the whole `COLOR` surface: `red_n`, `green_n`, `yellow_n`,
> `purple_n`, `aqua_n`, `orange_n` — six, of which D8's stated exclusions (`green_n` too close to
> `green_b`, `red_n` reading as the error border) leave four: `yellow_n`, `purple_n`, `aqua_n`,
> `orange_n`. That is a four-tint list with no collision at all.

### Items 12, 13 — CATCHES

Item 12 fits `tests/test_copilot_pass_tools.py` exactly: five `backend.set_pass(...)` sites, each
nine positional args (`:49`, `:55`, `:76`, `:80`, `:85`), asserts on `res.table` by substring
(`"glow [output]: runs 12, target f4 x0.5, linear" in res.table`, `:41`) and on `res.ok` / `.error`.
`_pass_table` (`backend.py:1279-1299`) builds one f-string row per pass, so D9's `, group <name>`
suffix is a one-line change read by a one-line assert.

Item 13 is confirmed in §C.

### Items 14, 15, 16 — CATCHES

Item 14 is confirmed in §C. Items 15 and 16 fit `tests/test_pass_draft.py` (read end to end, 48
lines), which drives real App methods (`open_add_pass`, `close_pass_settings`,
`create_pass_from_draft`) and reads `app.pass_draft` fields directly — no frames, no monkeypatch.
Every sub-case of 15 and 16 is that shape. Green today (6 passed with
`test_command_registry_coverage.py`).

One inherited gap in 15: *"Cancel and reopen: the initial state"*. Cancel is a BUTTON inside the
dialog body, not an App method, so a frame-free test cannot press it. If Cancel routes through
`close_import_passes` the two clauses are one test — say so, or drop the Cancel clause as covered
by the funnel.

### Item 17 — the plan is recomputed per frame — **WEAK**

The brief asks whether `draft.rejection` is readable after pumping frames the way
`tests/test_pass_settings_layout.py` pumps them. Two separate answers.

**The pumping mechanism transfers, and more cheaply than the gear's.** `_gear_sizes`
(`test_pass_settings_layout.py:23-46`) is `imgui.new_frame()` /
`pass_settings.draw_pass_settings(app)` / `imgui.end_frame()` in a loop, with
`pass_settings._draw_body` monkeypatched **only because** the thing measured
(`imgui.get_window_size()`) is readable only inside the frame. `draft.rejection` is stored ON THE
DRAFT, so it outlives `end_frame` and needs no recorder — D10's *"stored so a test can read what the
button read"* is exactly that seam. And `modal_window` (`ui_primitives.py:341-342`) opens the popup
itself (`if not imgui.is_popup_open(label): imgui.open_popup(label)`), so the body draws without
input injection. Rig confirmed working: `uv run pytest tests/test_pass_settings_layout.py -q` → 3
passed.

**But item 17's second step — "set the buffer to `ok` without changing the source" — depends on
focus, and fails under the natural UI choice.** Measured on the existing create-mode draft, whose
name field calls `set_keyboard_focus_here` (`popups/pass_settings.py:78-80`). A spy over
`_draw_draft` recorded what the body saw per frame:

```
body saw name_buf per frame: ['', '', '', '', 'CHANGED_BETWEEN_FRAMES', '']
draft after pumping: ''
```

The body ran all six frames, saw the externally-written value once, and imgui overwrote it on the
next frame — with `needs_focus` already False, so this is not a one-shot:

```
after 6 frames, name_buf= '' needs_focus= False
  frame 0: name_buf= ''      # after setting name_buf = "EXT" between frames
  frame 1: name_buf= ''
  frame 2: name_buf= ''
```

The contrast, on the gear's EDIT-mode name field, which has no `set_keyboard_focus_here`:

```
after pump, name_buf = 'main'
  frame 0: name_buf = 'EXTERNALLY_SET'
  frame 1: name_buf = 'EXTERNALLY_SET'
  frame 2: name_buf = 'EXTERNALLY_SET'
```

So an **unfocused** `input_text` keeps an externally-written buffer; a field imgui has taken
keyboard focus of owns its state and clobbers the write every frame thereafter. D10 says the group
field is prefilled on selection; if it is also FOCUSED on selection — the natural choice, and the
one `PassDraft.needs_focus` already makes for the sibling modal — item 17's second step silently
does nothing and the test goes green on a stale-rejection implementation too.

> **Rewrite.** *17. **The plan is recomputed per frame** (`tests/test_pass_settings_layout.py`'s
> pumped-frame shape — `imgui.new_frame()` / `draw_import_passes(app)` / `imgui.end_frame()`; no
> monkeypatched recorder, since `draft.rejection` outlives the frame): open the dialog with group
> `2bad`, pump a frame, `draft.rejection` names the group; set the buffer to `ok` WITHOUT changing
> the source, pump one frame, `draft.rejection` is empty.*
>
> ***The group input must not hold imgui keyboard focus when the buffer is written*** *— a focused
> `input_text` rewrites the buffer from imgui's own state on the next frame and the external write
> is erased (measured both ways: the gear's create-mode field, which calls
> `set_keyboard_focus_here`, loses the write on the very next frame with `needs_focus` already
> False; the gear's unfocused edit-mode field keeps it across three frames). So either the dialog
> does not auto-focus the group field, or the test drives the rejection through the SUBSTITUTION
> combo instead — state which, because the green test and the broken test look identical.
> Falsifier: compute the plan on selection only and the second frame still reports the stale
> rejection.*

### Item 18 — the unbordered tile does not move its contents — CATCHES

The brief asks whether the screen position of a `preview_cell`'s image is measurable. Directly, no:
the image goes onto the draw list as `dl.add_image(..., (ix, iy), ...)`
(`ui_primitives.py:1287-1294`) and `PreviewCellResult` (`:1168-1172`) carries only four click
booleans. But `ix`/`iy` are computed from exactly two live reads inside the child —
`origin = imgui.get_cursor_screen_pos()` and `avail = imgui.get_content_region_avail()`
(`:1268-1269`) — and **both are measurable from a bare frame**, the way
`test_pass_settings_layout.py:84-94` measures text widths inside `new_frame()` + `begin("rig")`. So
the item is runnable; it just has to measure the two ingredients rather than the `add_image` call.

**Landed independently.** This reviewer measured the shift and it was **8px on each axis, not the
1px** both round 1 and the then-current D7 claimed:

```
BORDERED   (origin-dx, origin-dy, avail.x, avail.y) = (8.0, 8.0, 152.0, 152.0)
UNBORDERED (origin-dx, origin-dy, avail.x, avail.y) = (0.0, 0.0, 168.0, 168.0)
style child_border_size = 1.0 ; style window_padding = (8.0, 8.0)   # SPACE.MD, theme.py:310/:403
```

and the fix is one flag the library documents (`imgui/__init__.pyi:3356-3358`:
*"always_use_window_padding … Pad with style.WindowPadding even if no border are drawn"*), while
pushing `StyleVar_.window_padding` does nothing:

```
borderless + push_style_var(padding 8)   (0.0, 0.0, 168.0, 168.0)   <- no effect
borderless + always_use_window_padding   (8.0, 8.0, 152.0, 152.0)   <- identical to bordered
```

D7 now carries exactly this (*"`ChildFlags_.borders` also enables `WindowPadding` … dropping it
alone moves the cell's content from `(8, 8)` to `(0, 0)` and grows `avail` from 152x164 to 168x180
… So `bordered=False` passes `ChildFlags_.always_use_window_padding` instead … and measures
byte-identical to `borders`"*). Closed.

**One remaining wording fix in item 18 itself**, which still says the old number: *"Falsifier: drop
`ChildFlags_.borders` with no compensating pad and grouped pictures sit 1px off their neighbours."*
It is 8px, and the fix is a flag rather than a pad.

> 18. … Falsifier: drop `ChildFlags_.borders` without `ChildFlags_.always_use_window_padding` and
>     the content origin moves by `style.window_padding` — **8px**, not the 1px
>     `child_border_size` — so grouped pictures, footers and chip rows all sit 8px off their
>     ungrouped neighbours.

A second consequence worth one clause in D7: with `bordered=border is not None`
(`pass_list.py:102-108` sets `border` for the output and for errors), a grouped OUTPUT tile keeps
the real border and its padding while its grouped siblings get the padding via the flag — so both
branches must land on the same content origin. That is what item 18 asserts, so it is covered once
the item compares the two branches rather than bordered-vs-unbordered in the abstract.

### Item 19 — smoke — WEAK

The brief's three questions.

**Where does the group stamp go?** Frame 42 is the only place that selects a multi-pass document
(`scripts/smoke.py:256-270`), so the stamp goes immediately after
`app.set_current_document_id(multi)` at `:267`. **But frame 48 switches away** (`:273-276`:
`app.set_current_document_id(canary_id)`), so "keep them for the rest of the loop" draws the group
for frames 43–47 — five frames, not 150. Enough to execute the path; the item should say so, rather
than implying the group is live at frame 199.

**Does RC's `strip_order` have two non-adjacent members available to stamp?** Yes — and the order
depends on compile state, which is the trap. Replaying the smoke's own seed and its frame-42
selection:

```
frame-42 multi = 77a84d27-…          (Radiance Cascades, the only multi-pass example)
passes: ['cascade', 'composite', 'df', 'jfa', 'paint', 'seed']
compiled before any render? all False                   (066 D1, lazy)
wiring pre-compile:  {'cascade': {}, 'composite': {}, 'df': {}, 'jfa': {}, 'paint': {}, 'seed': {}}
strip_order pre-compile:  ['cascade', 'composite', 'df', 'jfa', 'paint', 'seed']   (alphabetical fallback)
strip_order post-compile: ['paint', 'seed', 'jfa', 'df', 'cascade', 'composite']   (topological)
```

Six tiles either way, so two non-adjacent members always exist. But a stamp written as `order[0]`
and `order[2]` names `cascade`+`df` uncompiled and `paint`+`jfa` compiled — and at frame 42 RC has
only just become current, so which one depends on whether the strip drew it first. `strip_order`
falls back to `sorted(known - set(order))` for passes `plan_passes` leaves out
(`pass_graph.py:432-441`), which is every pass when the wiring is empty.

**`set_pass_group` availability.** It does not exist yet (`grep -n "def set_pass_"
shaderbox/project_session.py` → `set_pass_target`, `set_pass_iterations` only); as D9's seventh
verb it will save, so the stamp rewrites RC's `graph.json` inside the smoke's
`tempfile.TemporaryDirectory` (`_seed_tmp_project`, `:103-128`) — correct and harmless, never the
tracked resources.

The smoke is green today: `uv run python scripts/smoke.py` → `EXIT=0`,
`smoke: OK (200 frames, 7 documents)`, about four seconds.

> **Rewrite.** *19. **Smoke** (`scripts/smoke.py`): at frame 42, beside the existing multi-pass
> selection, stamp the group on two NON-ADJACENT members **by name**
> (`session.set_pass_group(multi, "paint", …)` and `…, "jfa", …`) — never by index into
> `strip_order`: RC's order is `['cascade', 'composite', 'df', 'jfa', 'paint', 'seed']` uncompiled
> (the `sorted()` fallback, since an uncompiled document's wiring is empty) and `['paint', 'seed',
> 'jfa', 'df', 'cascade', 'composite']` compiled, and at frame 42 it has only just become current,
> so an index names a different pair depending on when the strip first drew it (both orders
> measured). The group then draws for frames 43–47, until frame 48 returns to the feedback canary —
> five frames, enough for the outline, its label, the split-run path and the unbordered member
> tiles to execute on the parent draw list. Falsifier: `add_rect` inside a tile's child window;
> today no smoke document carries a group, so the path runs zero times.*

### D11 — no verification item at all

D11 rewires `examples_open` into *"the Examples popup, or the import dialog with its Examples tab
active"*, and `planned` / `planned_documents` / `current_planned` all follow it
(`ui.py:275-284`, consumers at `:299-325`). Nothing in the 19 items reads it.

The nearest existing cover is
`tests/test_render_decoupling_loop.py::test_a_throttled_example_renders_less_often_than_a_cheap_one`
(`:370-409`), which sets `app.popup_state = PopupState.EXAMPLES` directly and counts renders. It
exercises the EXAMPLES half and says nothing about the IMPORT_PASSES-with-Examples-tab half — the
state the predicate gains. Its own comment names the class: *"The popup's set is an ALTERNATIVE to
`tick_documents`, so an interval computed for an example id has to be read by the EXAMPLES branch or
it is read by nothing at all."* The failure mode is the one D11 exists to prevent: the dialog's
Examples grid shows six tiles of a never-rendered texture, which looks like a slow first frame.

> Add: ***20. The dialog's Examples tab renders the examples*** *(`tests/test_render_decoupling_loop.py`'s
> render-counting shape): with `popup_state = IMPORT_PASSES` and the Examples tab active, the
> examples are in `planned_documents` and at least one renders per frame; with the project tab
> active, `planned_documents` is the ordinary set. Falsifier: leave `examples_open` as
> `popup_state == PopupState.EXAMPLES` and the dialog's grid shows six never-rendered textures
> while the predicate test for the Examples MODAL stays green.*

---

## C. Blast radius — delta only

### `scripts/smoke.py` — new to Files touched

**What the touch consists of.** Two or three lines at frame 42, after `:267`
(`app.set_current_document_id(multi)`): two `session.set_pass_group(multi, <name>, "smoke_group")`
calls on named non-adjacent members. Nothing else in the file changes — frame 42's selection exists
precisely to draw the six-tile strip (`:255-270`), and the group is the only missing ingredient.

**What test covers it.** Nothing but the smoke itself, by construction. `make gates` runs
check → test → smoke, and the smoke's contract is "200 frames with no exception"; a skipped smoke
(no display) reports as skipped, not a pass, so the group path is only covered when the smoke
actually runs. The pure halves of D7's drawing decision are covered elsewhere — item 10
(`group_runs` adjacency) and item 18 (the unbordered cell's geometry) — so what the smoke adds is
only "the `add_rect` on the parent draw list does not assert mid-frame", which is the
`/imgui-ui` `SetCursorPos`/jitter class, and exactly what a frame loop can prove and a unit test
cannot.

Two constraints for the implementer: name the members rather than indexing `strip_order` (§B-19),
and note that `set_pass_group` saves, so the stamp rewrites `graph.json` inside the smoke's
throwaway tmp project (`:103-128`).

### `shaderbox/hotkeys.py` — new to Files touched

**What the touch consists of.** One `elif` in `_handle_escape`'s popup chain (`hotkeys.py:376-388`),
beside the existing `PASS_SETTINGS` branch:

```python
if app.popup_state == PopupState.PASS_SETTINGS:
    app.close_pass_settings()
elif app.popup_state == PopupState.PROJECTS and app.projects_input_owns_esc():
    pass
elif (app.popup_state != PopupState.SHADER_LIB_PICKER or not inline_input_owns_esc(app)):
    app.popup_state = PopupState.CLOSED
```

The new branch calls `app.close_import_passes()` so the draft is dropped instead of falling through
to the bare `popup_state = CLOSED`. No gating change upstream: `escape_has_job()`
(`app.py:604-612`) returns True on `any_popup_open()`, which the new member satisfies for free.

**What test covers it.** **Nothing structurally, and the repo already knows why.** The precedent is
`tests/test_project_management.py::test_escape_is_owned_by_an_open_name_input_at_the_dispatch`
(`:255-287`), added for exactly this class — its docstring says *"Falsifier: delete the PROJECTS
branch from `_handle_escape` … The predicate test above passes either way, which is what let that
branch survive as dead code through a mutation round."* And its mechanism is a **substring assert on
the source** (`:278-281`):

```python
source = Path("shaderbox/hotkeys.py").read_text(encoding="utf-8")
assert "projects_input_owns_esc()" in source, (
    "the ownership predicate must be CONSULTED in the Esc dispatch, not merely defined"
)
```

because a second frame-driving App in one process hits a torn-down imgui font atlas — a
process-global GL limit, not a feature limit, as its docstring explains.

Item 15 tests the FUNNEL (`app.close_import_passes()` resets the draft) and its falsifier names the
fallthrough — but a funnel test passes whether or not `hotkeys.py` calls it. That is the same gap
the precedent closed.

> Add to item 15: *"and the BRANCH, not only the funnel: assert `"close_import_passes()"` appears in
> `shaderbox/hotkeys.py`, the way
> `test_project_management.py::test_escape_is_owned_by_an_open_name_input_at_the_dispatch` (`:278-281`)
> asserts its own predicate is consulted — for the reason that test's docstring gives (a second
> frame-driving App in one process hits the process-global imgui font atlas). Falsifier: define
> `close_import_passes` and never call it from the Esc dispatch; the funnel test stays green and
> Escape leaves the draft populated."*

### Item 14 — `test_every_popup_state_has_a_draw_call` — confirmed as the spec describes

Read at `tests/test_project_management.py:635-668`. It `ast.parse`s `shaderbox/ui.py`, collects
names imported from any `shaderbox.popups*` module, collects the subset appearing as an `ast.Call`
with an `ast.Name` func, and asserts both `imported - called == []` and
`len(called) == len(PopupState) - 1`. Its docstring names the bug class (*"A state with no draw
call is INVISIBLE at runtime — the popup mutex suppresses every render"*) and the reason it parses
rather than substring-matches (an earlier version was satisfied by a call inside a comment).

Re-ran the same AST walk against today's tree:

```
imported 7 ['draw_emoji_picker', 'draw_examples', 'draw_help', 'draw_lib_picker',
            'draw_pass_settings', 'draw_projects', 'draw_settings']
called 7
len(PopupState)-1 = 7
members ['CLOSED', 'EXAMPLES', 'HELP', 'SETTINGS', 'PASS_SETTINGS', 'EMOJI_PICKER',
         'SHADER_LIB_PICKER', 'PROJECTS']
```

So `PopupState.IMPORT_PASSES` makes the second assert `7 != 8` and the suite is red until
`draw_import_passes` is both imported and CALLED in `ui.py`. Item 14's parenthetical is exact.

What it does NOT gate, which D10 correctly handles in prose: the module's own early-return guard.
`ui.py` has an `if/elif` chain and no dispatch table, so a missing guard means the dialog draws
under every other modal; D10 says *"with the module's own early-return guard"*, and the AST test
cannot see it.

### Item 13 — `test_command_registry_coverage.py` — confirmed as the spec describes

Read end to end (24 lines). Two independent strict set-equalities, exactly as the item says:

```python
def test_every_command_id_has_a_spec() -> None:
    assert set(SPEC_BY_ID) == set(CommandId)

def test_every_command_id_has_a_handler(app: Any) -> None:
    assert set(app.command_callbacks) == set(CommandId)
```

Both green today. So `CommandId.IMPORT_PASSES` needs both a `COMMAND_SPECS` row and an
`app.command_callbacks` entry, either alone red — matching D10's *"a `COMMAND_SPECS` row AND an
`app.command_callbacks` handler, both gated"*.

**One thing neither the item nor D10 mentions.** The file's third test,
`test_every_bound_spec_reaches_the_help_shortcuts` (`:20-23`), asserts `spec.label in snippet` for
every spec that sets a `default_chord`. A chordless `IMPORT_PASSES` skips it; a CHORDED one needs a
help line too, or the suite goes red for a reason the spec does not predict. One clause in D10
("palette-only, no default chord" or "chorded, and the help line comes with it") saves a red run.

---

## False trails

Probed this round, turned out fine — so nobody re-spends the time.

- **The `app` fixture writing the tracked examples dir at startup.** It does not. An App built over
  all six examples with RC's six passes compiled leaves
  `find shaderbox/resources/document_examples -type f -exec md5sum {} \; | sort | md5sum`
  byte-identical (`b6b8c70e…` before and after) and `git status --porcelain shaderbox/resources/`
  clean. Item 6's baseline is stable. (Their `document.json` mtimes ARE newer than their
  `graph.json`, which is why content-only is the right spelling — but the content never moves.)
- **`modal_window` needing a frame-driven click to open.** It does not: `ui_primitives.py:341-342`
  calls `imgui.open_popup(label)` itself when the popup is not open, so a pumped-frame test reaches
  the body with no input injection. Demonstrated by `test_pass_settings_layout.py` passing (3
  passed) — `_gear_sizes` never clicks anything.
- **`load_document_from_dir` living in `document.py`.** It is in `shaderbox/ui_models.py:591`
  (`Document.load_from_dir` is the `document.py` classmethod it wraps). Items 5 and round 1 name it
  unqualified; both existing callers import it from `ui_models` (`test_lazy_compile.py:22`,
  `test_default_wiring.py:32`). No spec change needed, just don't guess the module.
- **`document_dir_of` being in `paths.py`.** It is in `shaderbox/document.py:1092`; the one
  production caller imports it from there (`copilot/backend.py:88`). D5 and the out-of-scope bullet
  both name it without a module.
- **Importing `project_session`'s private graph helpers from a test.** Allowed by precedent —
  `_graph_renamed` and `_graph_without` import cleanly, and tests already import privates
  (`test_completion.py:22` → `hotkeys._is_lookup_key`; `test_anchored_note.py:15` →
  `ui_primitives._ellipsize`; `test_brake_falsifiers.py:12`, `test_content_editing.py:21`, and
  more). So item 9's verb half has a second way out besides moving file (§B-9).
- **`test_pass_graph.py` / `test_pass_strip_layout.py` / `test_theme.py` being GL-bound.** All three
  are pure: `grep -c fixture tests/test_pass_graph.py` → 0, and the three run together in 0.59s
  (35 passed). Items 1, 2, 10 and 11 cost nothing.
- **The smoke being red or slow today.** `uv run python scripts/smoke.py` → `EXIT=0`,
  `smoke: OK (200 frames, 7 documents)`, about four seconds. Frame 42 picks Radiance Cascades, the
  only multi-pass example among the six seeded.
- **`test_copilot_pass_tools.py` needing more than a mechanical widening for D9.** Five
  `backend.set_pass(...)` sites, nine positional args each; `_pass_table`
  (`backend.py:1279-1299`) builds one f-string row per pass, so the `, group <name>` suffix is a
  one-line change read by a one-line substring assert. Mechanical, not a hole.
- **`D8`'s `aqua_n` being an accent primary.** It is not — it is the aqua preset's ACTIVE colour
  (`_ACCENTS["aqua"] = (aqua_b, aqua_n, …)`), exactly as D8 says, and `_accent_primaries`
  enumerates element [0] only. D8's reason for naming actives explicitly in the assert is correct.
