# 069 — pre-impl review 1: correctness & design vs conventions

Reviewer role: correctness & design (per `dev_flow.md ## Feature flow` step 4). Read-only.
Anchor artifacts: `01_spec.md`, `00_findings.md` (the maintainer's verbatim "Reported" column),
`01_design_scripting.md`, `conventions.md`, `CLAUDE.md`, and the code as it is on `dev` at
`1767483`.

Every claim below names the file I opened or the command I ran. Where I cite a line number it is
the line as it stands today; per the repo's own "no raw line numbers in docs" rule I also name the
symbol, so the citation survives an edit.

---

## Task 1 — decisions vs conventions

### Verdict: **PARTIAL**

Ten of the twelve locked decisions and six of the eight workstreams sit cleanly inside the
conventions. Three items silently contradict a convention or a hard rule, and two supersede one
without naming the doc update. The enumeration follows.

#### Summary table

| Item | Convention / hard rule it touches | Status |
|---|---|---|
| D1 (word budget) | `conventions.md ## Code rules` — "UI authoring rules live in the `/imgui-ui` skill, not here" | **(a) honours** |
| D2 (no manual JSON) | none directly | (a) |
| D3 (nested-dict script) | `conventions.md ## Design decisions` — "The CPU-script engine … a document has ONE STATEFUL-class script"; 065 D12 | **(b) supersedes, doc update named** |
| D4 (019 removed) | `conventions.md ## Design decisions` — "App-wide keyboard nav is region-confined (`nav_enable_keyboard` ON)" | **(b) supersedes, doc update named** |
| D5 (keymap Setting) | `conventions.md ## Design decisions` — the code-editor entry's own revisit trigger | **(c) silently contradicts — see C1** |
| D6 (vim furniture inside the rect) | `/imgui-ui` skill; no conventions entry | (a) |
| D7 (one owner per chord cell) | none existing; creates one | (a) |
| D8 (tutorial template) | none | (a) |
| D9 (`u_<pass>` naming) | `conventions.md ## Design decisions` — "An unfilled pass input reads BLACK" (closed-set wiring) | (a) — see F5 for a factual error in the mapping |
| D10 (add pass activates) | none | (a) |
| D11 (commit on deactivate) | `/imgui-ui` § 7.5 | (a) — spec names the skill edit |
| D12 (graph view) | 065 § "the graph is edited as a list … a canvas UI is a separate feature" | **(b) supersedes; 065 doc update named for D12-the-script-decision, NOT for this one — see C2** |
| W-A | funnel rule; `Document.set_canvas_size` | (a) — a textbook application |
| W-B | `## Code rules` "no `@staticmethod`"; theme token rule | (a) |
| W-C | draw-once invariant (`assert_plan_invariants`); memoization rule | (a) design-wise, **but see Task 3(a)** |
| W-D | closed-set wiring; "Two parallel name-keyed dicts … are a drift smell" | (a) |
| W-E | region-confinement entry; `SELECT`-hue assertion entry | (b) — deletion named |
| W-F | "Inline editor state lives on `App` … one libeditor instance per opened FILE" + `## Known quirks` vendored-binary entry | **(c) — see C3** |
| W-G | script-engine entry; NO-migration rule; headless-boundary rule | (b) for the entry, (a) for no-migration |
| W-H | none (docs only) | (a) |

#### The three that silently contradict

**C1 — D5 fires the code-editor entry's own revisit trigger, and the spec does not update it.**

`conventions.md`, the entry beginning "**Inline editor state lives on `App`; disk is the source of
truth; one libeditor instance per opened FILE.**", ends:

> Revisit if a tab needs durable per-tab state beyond its open files (e.g. persisting the open-tab
> set across restart), a 4th editable `kind` lands, **or a non-modal keymap ships editor-side.**

D5 is exactly "a non-modal keymap ships editor-side" (`01_spec.md`, D5: "The editor keymap (vim /
standard) is a global Setting"). The trigger fires, and the entry's own statement — that the editor
is "the maintainer's own vim-modal library" — becomes false the moment the standard keymap is
selectable. The spec's `## Files touched` names `conventions.md` for four things ("scripting entry,
region entry, the naming rule, the word-budget rule pointer") and **not** this entry. Per
`dev_flow.md` step 2's pre-flight ("re-read `conventions.md` … if one contradicts the request, halt
and reconcile"), this is an unreconciled contradiction, not a deliberate supersession.

Fix: add the code-editor entry to W-E's or W-F's `conventions.md` edit list, rewording "vim-modal
library" to "modal-or-non-modal, keymap chosen by `EditorSettings.keymap`", and either retire the
now-fired trigger or restate it.

**C2 — D12 supersedes 065's "a canvas UI is a separate feature" without the doc line.**

`ai_docs/features/065_pass_graph/01_spec.md` records the deliberate exclusion (the finding ledger
quotes it at #19: "065 deliberately left a spatial graph view out … 'the graph is edited as a list …
a canvas UI is a separate feature'"). The spec's `## Files touched` says `roadmap.md` gets
"065 → done with the D12 supersession noted" — but that D12 is *065's script-per-pass decision*
(superseded by 069 D3), a different decision from the graph-view exclusion. Two different 065 items
are being superseded and only one gets a line.

W-G names it correctly for the script half: "065 D12 and 068 D7 get 'superseded by 069' lines".
Nothing names the graph-view half. Note the spec's own Open question 3 asks whether W-D lands in
this feature at all — so the doc line cannot be written until that is answered, which makes the
omission a real ordering gap rather than a typo.

**C3 — W-F re-vendors the binary but does not name the `## Known quirks` entry that documents it.**

`conventions.md ## Design decisions`, code-editor entry, last sentence: "The vendored binary +
rebuild procedure live in `## Known quirks`." W-F's first bullet re-vendors `libeditor.so` from
upstream HEAD and updates `resources/editor/VERSION`. I confirmed the current vendored version is
`e7db554ddfc46f143dc96ca4456c8212fbe8e381` (`cat shaderbox/resources/editor/VERSION`) and that
upstream HEAD is `68def59` (`git log --oneline -5` in the editor repo). W-F's file list
says `resources/editor/*` and does not name `conventions.md`. If the quirks entry quotes the version
or the procedure, it goes stale in this wave.

#### Where the spec honours a rule well (worth recording, so a later wave does not undo it)

- **The NO-migration rule.** `conventions.md`: "NO backward-compatibility / migration code, EVER …
  when you see a migration/back-compat proposal, DELETE it, don't implement it." W-G states it
  explicitly: "`projects/dev` scripts hand-edited to the new shape (NO migration code)". Verified
  the sandbox: `find projects/dev -name script.py` returns four files, all under
  `projects/dev/trash/` — no live document script exists, so the hand-edit is a no-op. Harmless, and
  the clause is right to be there anyway.
- **The funnel rule.** `conventions.md`: "A cross-cutting guarantee is enforced at the single
  FUNNEL, not per-caller." W-A routes the Resolution combo through `Document.set_canvas_size` and
  moves the copilot's clamp to "one clamp, shared". I confirmed the bypass is real: `tabs/document.py`
  calls `ui_document.document.render_pass.canvas.set_size((w, h))` directly, while
  `Document.set_canvas_size`'s own docstring names this exact bug ("A caller that resized
  `render_pass.canvas` directly — which is what the copilot's set_canvas_size did — left this field
  stale"). This is the second instance of the same bug at a sibling site, which the convention names
  as the trigger to move to the funnel. Correct call.
- **`ui_primitives` / `theme` ownership** (CLAUDE.md hard rule). W-A: "Two greys from `theme.py`, no
  literal colours at the call site." W-B routes the auto-size through `ui_primitives.modal_window`.
  Both correct.
- **A rule with no gate is a wish.** D1 ships with the `ast` word-count test in the same commit as
  the cut (W-B "Gate"), which is the repo's own stated discipline. Good.

#### One structural note on the spec's own shape

`dev_flow.md ## Feature flow` step 2 lists the minimum spec sections: *Goal* / *Out of scope* /
**Design decisions** / *Files touched* / *Manual verification* / *Open questions for the user*. The
spec's header is `## Locked decisions (from the walk — constraints, not options)`. Content-wise it is
the Design-decisions section (numbered, lock-in only, open questions separate), so this is cosmetic —
but the flow says "keep the headers", so rename it or note the alias.

---

## Task 2 — blast radius

### Verdict: **PARTIAL**

Every deleted or reshaped symbol was enumerated with `grep -rn` across `shaderbox/`, `tests/` and
`ai_docs/`. The spec's Files-touched lists cover the large majority. **Eleven misses**, listed
below with the reference that is not covered.

#### W-E — keyboard regions

Enumerated symbols: `ActiveRegion`, `active_region_outline`, `region_focus_pending`, `CYCLE_REGION`,
`no_nav_inputs`, `nav_enable_keyboard`, `active_region`, `cycle_region`, `_set_region`,
`focus_move_in_flight`, `region_derive_allowed`, `region_outline_visible`,
`_yield_editor_to_region`, `_focus_or_add_tab`, `FOCUS_TAB_*`, plus `nav_flatten` / `nav_flattened`
(which only exist because `nav_enable_keyboard` is on).

W-E names: `app.py`, `ui.py`, `ui_regions.py`, `ui_primitives.py`, `widgets/document_grid.py`,
`widgets/copilot_chat.py`, `commands.py`, `hotkeys.py`, `ui_models.py`, `popups/settings.py`,
`editor/ffi.py`, `conventions.md`, tests. Every code file carrying a region symbol is named. Good.

**MISS E1 — `shaderbox/popups/examples.py`** passes `nav_flatten=True` to `preview_cell`
(`popups/examples.py:101`). `nav_flatten` sets `ChildFlags_.nav_flattened`, whose only purpose is to
let imgui keyboard-nav cross a tile border — dead the moment `nav_enable_keyboard` goes off. Not in
W-E's file list.

**MISS E2 — `shaderbox/tabs/document.py`** carries `child_flags=imgui.ChildFlags_.nav_flattened |
imgui.ChildFlags_.auto_resize_y` with the comment "nav_flattened: Tab/arrows reach the sliders
without an Enter/Esc window boundary." That comment becomes a false statement about the code (the
`## Code rules` bar: a comment "states what's non-obvious about the code as it is NOW"). W-E does not
name `tabs/document.py`. (W-A and W-B both do, for other reasons — so the file will be open; but the
spec does not say the nav comment is part of the removal, and a wave that does not know to look will
leave it.)

**MISS E3 — `shaderbox/ui_primitives.py::preview_cell`'s `nav_flatten` parameter and its docstring
paragraph** ("`nav_flatten` lets keyboard-nav cross the per-tile child border so a grid traverses as
one ring; the click target is a `selectable` (a nav stop, unlike an `invisible_button`) …"). W-E
names `ui_primitives.py` for `active_region_outline`, so the file is open — but the spec's bullet
enumerates six named deletions and this parameter is not among them, and it is a public parameter
with three call sites (`document_grid.py` ×2, `examples.py`, `pass_list.py` implicitly by default).
Worth naming so the wave decides deliberately whether the param goes or stays inert.

**MISS E4 — `shaderbox/ui.py`'s third `no_nav_inputs`.** W-E says to delete "every `no_nav_inputs`
flag whose only reason was confinement (the copilot window's stays if it guards something else —
check `copilot_chat.py:46` comment)". There are **four** `no_nav_inputs` sites in `shaderbox/`, not
the three the ledger's #24 row lists: `ui.py` (three: the editor child, one at line 553, and the
panel child) and `copilot_chat.py` (one). The one at `ui.py:553` is unaccounted for in both the
ledger and the spec — it is an unconditional flag, not a `panel_active ? none : no_nav_inputs`
ternary, so it is a different case from the two confinement sites and needs its own decision. W-E
names `ui.py`, so the file is open; the count is what is wrong, and a checker that narrows its own
domain is the failure mode the repo's own rules single out.

**MISS E5 — `ai_docs/features/067_custom_editor.md`** documents both the `_VIM_RESERVED_CHORDS` set
("The FULL reserved set is `_VIM_RESERVED_CHORDS` = d u f b e y r o w n p h j") and the
`no_nav_inputs` flag on the editor child, and states the vim-only routing that D5/D7 replace. W-E
makes `_VIM_RESERVED_CHORDS` "per-keymap data in `hotkeys.py`" and D5 adds a second keymap. 067 is
not in any Files-touched list; the spec gives 019 a "removed by 069" banner and gives 067 nothing.

#### W-G — scripting

Enumerated: `EngineNode`, `ScriptEngine.tick`, `tick_export`, `dry_run`, `script_stub_for`,
`MouseState`, `EXPORT_MOUSE`, `stopped_uniforms`.

W-G names: `scripting/engine.py`, `scripting/context.py`, `scripting/behavior.py`,
`project_session.py`, `ui.py`, `commands.py`, `app.py`, `copilot/prompt.py`,
`copilot/capabilities.py`, `help_content.py`, `conventions.md`, `projects/dev`, tests.

**MISS G1 — `shaderbox/scripting/api_doc.py`.** It derives the copilot's SCRIPT API prompt block
from the live types:

```python
_MOUSE_FIELDS: str = ", ".join(f"`{n}`" for n in MouseState.__dataclass_fields__)
_EXPORT_MOUSE_AT: str = f"{EXPORT_MOUSE.x:g},{EXPORT_MOUSE.y:g}"
```

W-G adds `down`, `prev_x`, `prev_y` to `MouseState`, so the generated block changes. The spec says
"the copilot SCRIPT API prompt block regenerates (`tests/test_script_api_doc.py` pins it)" — it names
the *test* that pins the output and not the *module that produces it*. `api_doc.py` also carries the
prose that must describe the nested-dict return contract (D3), which is a hand edit, not a
regeneration.

**MISS G2 — `shaderbox/copilot/prompt_context.py`.** It is the sole importer of the generated block:
`from shaderbox.scripting.api_doc import script_api_summary`. W-G names `copilot/prompt.py` instead.
Both may need touching (prompt.py carries the pass block D9 changes), but prompt_context.py is the
one on the script path and it is unnamed.

**MISS G3 — `shaderbox/scripting/__init__.py`.** It re-exports `EngineNode` in both the import list
and `__all__`. W-G retires the `EngineNode` protocol ("retired for a `Document`-shaped one"), so the
package's public surface changes. Not named.

**MISS G4 — `ScriptEngine.tick_export`.** W-G's engine bullet names `tick`, `dry_run` and
`script_stub_for`. `tick_export` has the identical `(document_id, document: EngineNode, ctx, …)`
shape and is called from `project_session.py`'s `_export_pre_render` closure with
`document.render_pass` — the exact "output pass only" binding D3 removes. The spec says
"`dry_run` / export through the same routing", which covers the intent, but the third public method
that takes an `EngineNode` is never named, and the export path is the one 068 D7's retraction
already burned on.

**MISS G5 — the `stopped` key change's UI consumers.** The design note pins "Stopped-uniform keys
become `(pass, name)`; today `stopped_uniforms` is a set of names … and would freeze a name on every
pass", and W-G repeats it ("`stopped` keyed `(pass, name)`"). The consumers of the flat form are:
`ui_models.py::UIDocumentState.stopped_uniforms` (named by W-E, not W-G),
`project_session.py::_stopped_for` / `is_uniform_stopped` / `set_uniform_stopped` /
`set_document_all_stopped` (project_session.py is named), and — not named anywhere —
`shaderbox/widgets/uniform.py`, which draws the per-row play/stop button that calls those setters,
and the five shipped `document.json` files that persist `"stopped_uniforms": []`. Those five are:

```
shaderbox/resources/document_examples/{73ea2431…, 8d454b7b…, 53724dbd…, f90f5ff9…, 0b0d16bb…}/document.json
```

An `extra='forbid'` persisted model whose field changes shape makes every one of those a load
failure — which is the *correct* loud behaviour per the persistence-evolution posture, but the spec
must say the five files are hand-edited in the same wave (the sandbox rule's shipped-resource twin).
Neither `widgets/uniform.py` nor the example `document.json`s appear in any Files-touched list.

#### W-D — graph representation and naming

**MISS D1 — `tests/test_pass_verbs.py` imports the symbol W-D deletes.** Line 28:
`from shaderbox.widgets.pass_list import _strip_order`, used by
`test_the_strip_order_is_topological_and_independent_of_the_output`. W-D says
"`widgets/pass_list.py` (deleted or reduced to the add-pass row)" and its file list ends with a bare
"tests". `_strip_order` is the topological-order function the new `pass_graph_view.py` needs anyway
(W-D moves rank computation into `pass_graph.py` "as a pure function, tested"), so the test moves
with it — but the spec does not say which module owns it afterwards, and a bare "tests" does not
flag that an existing test's import breaks.

**MISS D2 — `shaderbox/tabs/document.py` is the only caller of `pass_list.draw`.** Line 19 imports
the module, line 221 calls `pass_list.draw(app, document_id, lambda name: …)`. Replacing the strip
with `pass_graph_view.py` changes that call site. W-D's file list names `widgets/pass_graph_view.py`,
`widgets/pass_list.py`, `pass_graph.py`, `project_session.py`, examples, `help_content.py`,
`copilot/prompt.py`, tests — **not** `tabs/document.py`. (W-A and W-B name it, so it will be open in
earlier waves; W-D is wave 7 and the file list is what a wave-7 implementer reads.)

**MISS D3 — the `u_light_*` / `u_glow_*` prefix collision in unrelated examples.** A grep-driven
rename of `u_light` → `u_cascade` across `shaderbox/resources/` would corrupt two single-pass
examples that have nothing to do with the graph naming rule:

```
8d454b7b…/passes/main.frag.glsl:56  uniform float u_light_ambient = 0.55;
8d454b7b…/passes/main.frag.glsl:57  uniform float u_light_sky_key = 0.45;
8d454b7b…/passes/main.frag.glsl:58  uniform float u_light_moon_key = 0.40;
8d454b7b…/passes/main.frag.glsl:59  uniform vec3  u_light_cool_color = …;
8d454b7b…/passes/main.frag.glsl:60  uniform vec3  u_light_warm_color = …;
0b0d16bb…/passes/main.frag.glsl:12  uniform float u_glow_strength = 0.79;
0b0d16bb…/passes/main.frag.glsl:13  uniform float u_glow_radius = 1.73;
```

`u_light` has 8 hits in that shader and 10 in its `document.json` (persisted uniform values keyed by
name); `u_glow` has 4 and 4. None is a sampler and none is an input edge — D9 governs *input
uniforms* only. The spec does not scope the rename to samplers, and a whole-word-anchored rename is
the fix. Recording it because it is the kind of thing a rename wave discovers at `make gates` time,
and the persisted `document.json` values would be the silent half.

#### The symbols with no miss

- `PASS_SETTINGS_W` / `PASS_SETTINGS_H` — three sites total (`theme.py` ×2, `popups/pass_settings.py`
  ×1). W-B names both files. Zero test references. **Clean.**
- `preview_cell` — nine sites; the three consumers outside `pass_list.py` are `document_grid.py`,
  `popups/examples.py`, `exporters/telegram.py`. W-D reuses `preview_cell` at a smaller size rather
  than changing its signature, so the other three are untouched by construction. **Clean** (the
  `nav_flatten` param is a W-E matter, filed above as E3).
- `_VIM_RESERVED_CHORDS` — two code sites, both in `hotkeys.py`, which W-E names. **Clean** in code;
  the 067 doc reference is filed as E5.
- `CYCLE_REGION`, `region_focus_pending`, `_yield_editor_to_region`, `region_derive_allowed`,
  `region_outline_visible`, `focus_move_in_flight`, `cycle_region`, `_set_region` — every code site
  is in `app.py`, `ui.py`, `ui_regions.py`, `widgets/document_grid.py`, `hotkeys.py`,
  `commands.py`, all named. Only `tests/test_pass_editor_wiring.py` references them, and W-E names
  that test by name. **Clean.**
- The example input names in `tests/` — 17 test files matched a grep for `u_src`/`u_glow`/etc., but
  inspecting each shows every hit is a *local fixture* string, not an assertion about a shipped
  example (`test_pass_graph.py` builds a Bloom-shaped graph from literals; `test_pass_verbs.py` uses
  `u_src` in its own two-pass fixture). `test_radiance_cascades_example.py` and
  `test_example_library.py` contain **zero** hits for any of the six names. So D9's rename does not
  break an api-lock test, and the spec's "the api-lock tests for examples updated" is a smaller job
  than it sounds — possibly empty. **Clean, and the spec over-scopes rather than under-scopes.**

---

## Task 3 — feasibility of the load-bearing claims

### Verdict: **PARTIAL** — (a) needs a signature the spec does not name; (b) mostly confirmed with
one wrong premise; (c) misses call sites; (d) confirmed with one over-claim; (e) confirmed sound.

### (a) W-C first render — **the claim is not expressible with the current signature**

Read: `shaderbox/ui.py` (the `_tick_frame_state` render-set block and the `update_and_draw` render
block), `shaderbox/document.py::Document.render`, `shaderbox/pass_graph.py::evaluation_order` /
`plan_for_output`.

`Document.render` is:

```python
def render(self, u_time: float | None = None, canvas: Canvas | None = None) -> None:
```

Its body reads the output from the graph and nothing else:

```python
output = self.graph.output_pass
if output is None or output not in self.passes:
    self._graph_errors = plan_passes(self.graph)[1]
    return
planned, self._graph_errors = plan_for_output(self.graph, output)
```

So **there is no way to ask the current `Document.render` to draw a chain other than the output's.**
W-C says: "`ui.py`'s frame gate gains 'at most one not-yet-drawn pass per frame, drawn with its own
chain via `evaluation_order(graph, pass)`'". `evaluation_order(graph, target)` does take an arbitrary
target — I read it, and `plan_for_output` is already parameterised — so the *planning* half is free.
The *drawing* half is not: `ui.py` calls `ui_document.document.render()` with no target, and the
`canvas`/`u_time` parameters do not reach `output`.

What must change, minimally: `Document.render` gains a `target: str | None = None` parameter that
substitutes for `self.graph.output_pass` in the `plan_for_output` call and in the two places `output`
is compared to `name` (the scale exemption "the OUTPUT keeps full size", and the `target = canvas if
(name == output and last) else None` external-canvas rule). Both comparisons are load-bearing: a
first-render pass drawn as a target must size by its own `scale` like an intermediate, and must never
receive an external `canvas`. Neither follows automatically from swapping the planner's target.

The spec's Files-touched for W-C **does** name `document.py`, so the file is open — but the bullet
describes the change as living in "`ui.py`'s frame gate", which is only half of it, and the spec
nowhere states that `Document.render` grows a parameter. That is the kind of unnamed signature change
that gets discovered at impl time.

Two further observations, both in the spec's favour:

- `Pass.first_render_done` (W-C's new flag) mirrors an existing `Document.first_render_done`
  (`document.py`, set inside `render()` when `canvas is None`; read at five sites in `ui.py`,
  `popups/examples.py`, `widgets/document_grid.py` and pinned by `tests/test_lazy_compile.py`). So
  the pattern is proven one level up and the naming is consistent. Good.
- The draw-once invariant W-C promises to keep is genuinely enforced on the drawing path:
  `evaluation_order`'s docstring says "This is the function the renderer calls, so it asserts the
  plan itself rather than trusting a test to have done it", and `assert_plan_invariants` asserts
  `len(plan.order) == len(set(plan.order))`. Because each first-render pass is drawn in its **own**
  `render` call with its own plan, the invariant holds per call and cannot catch a pass drawn twice
  across two calls in one frame. W-C's test ("the steady state draws only the output chain") is the
  right falsifier for that gap and the spec names it. Good.

### (b) W-F — three of the four issues are real gaps; one premise is wrong

Read (via a delegated read-only investigation of the editor repo (alexeykarnachev/editor), whose quotes I list
with their file:line so they are checkable): `ffi/README.md` § Chrome and § Line markers,
`ffi/ffi.odin`, `src/theme.odin`, `src/marker.odin`, `src/keymap_normal.odin`, `src/keymap_visual.odin`,
`src/keymap.odin`, `src/register.odin`, `src/behavior_test.odin`, `docs/standard_keymap.md`,
`docs/vim_coverage.md`, and `git log`.

**Issue (1) — visual-mode `p`/`P` replaces the selection. REAL GAP, confirmed.**
`src/keymap.odin:264` routes `.Normal, .Visual, .Visual_Line` all into `editor_key_normal`.
`src/keymap_normal.odin:307` handles `'p','P'` by calling `register_paste(e, c == 'P', n)` with **no
mode check** — while the same function does special-case visual mode at five other points.
`src/keymap_visual.odin` contains no `'p'`/`'P'` case at all. `src/register.odin`'s `register_paste`
reads only `e.core.cursor` and `e.register.mode` — it never touches `e.visual_anchor` or calls
`visual_range`. `docs/vim_coverage.md` lists `p`/`P` only under normal-mode paste. Filing this is
correct.

**Issue (2) — export the gutter/status emission through the ABI. REAL GAP, confirmed.**
`chrome_emit_gutter` is defined at `src/chrome_emit.odin:50` and is **not** exported: zero hits in
`ffi/ffi.odin`. `ffi/README.md § Chrome` states it outright: "**`ed_layout` draws none of this
furniture** — no gutter, no status line. A host compositing the editor into its own frame renders
those in its own visual language … So this is mostly a QUERY surface." Only the flag setters/getters,
`ed_set_tab_width`/`ed_tab_width`, `ed_set_number_width`, `ed_set_filler_glyph`/`ed_filler_glyph`
and `ed_gutter_cells` cross. Filing this is correct, and the spec is right that ShaderBox does not
wait on it (W-F's host `_draw_gutter` fallback).

**Issue (3) — a marker or theme slot that overrides text colour. REAL GAP, but the spec's framing
is half wrong.** The theme has no `Error` slot: `src/theme.odin`'s `Theme_Slot` enum is
`Background, Text, Caret, Caret_Insert, Selection, Gutter_Text, Gutter_Current, Filler, Status_Bg,
Status_Text, Status_Accent, Popup_Panel, Popup_Text, Popup_Selected, Syntax_1..Syntax_7, Whitespace,
Bracket_Match` — 22 slots, zero matches for "error". And `ffi/README.md § Line markers` confirms the
fill "draws behind the text as a `Background` primitive". So vim's `hi Error` (fg replaces syntax
colour) is genuinely inexpressible. **The half that is wrong:** the ledger's #14 row says the marker
gutter mark is "currently passed the same" colour, implying the host cannot colour it separately — but
`ed_add_marker` takes **two** independent RGBAs, a line fill `(fr,fg,fb,fa)` and a gutter mark
`(gr,gg,gb,ga)`, plus a `gutter_glyph`. W-F's host-side fix ("the marker's gutter mark … draws in
`STATE_ERROR` in the gutter") is therefore already fully supported by today's ABI and needs no
upstream change — which is what W-F says, so the plan is right even though the ledger's reasoning
about *why* is loose. Worth noting only so the issue text filed upstream does not claim the ABI
cannot carry a gutter colour; it can.

**Issue (4) — markers anchored to text. REAL GAP, confirmed.** `src/marker.odin` stores
`Marker :: struct { line: int, … }` and its own doc comment says "A marker is HOST data, not editor
state. It survives no edit and means nothing to a motion". `ffi/README.md § Line markers`: "Lines are
BUFFER lines, 0-based … a host rebuilding from diagnostics will name a line an edit has already
removed." There is no update-in-place API. Filing this is correct, and W-F's host-side fix (no line
markers while `is_tab_dirty`) is the right stopgap given that.

**The standard keymap at upstream HEAD — CONFIRMED, and it changes one of W-E's premises.**
`git log --oneline -5` in the editor repo → HEAD is `68def59`, matching the spec's "68def59 at writing".
`git log --oneline -- docs/standard_keymap.md` → two commits (`7ae414c` "Add the standard keymap",
`942ccf5`), so the file exists at HEAD. The vendored version is `e7db554` (from
`shaderbox/resources/editor/VERSION`), which also exists — so the spec's "`e7db554..68def59`" range
is accurate.

`ed_set_style` and `ed_style` are both exported (`ffi.odin:894` and `ffi.odin:906`), so W-E's "the
lib side already has the switch at the ABI" holds. And `shaderbox/editor/ffi.py` binds **neither** —
I grepped: it binds `ed_set_chrome_flag` and `ed_gutter_cells` only. So W-E's "`editor/ffi.py` binds
`ed_set_style` / `ed_style`" is a genuine, correctly-identified addition. W-F additionally needs
`ed_filler_glyph` (exported at `ffi.odin:861`) — the spec names it. All three bindings check out.

**The one premise that does not hold:** W-E lists the known chord moves as "`NEW_DOCUMENT` off
Ctrl+N, `TOGGLE_DOCUMENT_PLAY` off Ctrl+Space (both collide with standard completion)". The
standard keymap's actual Ctrl surface, from `docs/standard_keymap.md`, is:

> `Ctrl+Left`/`Ctrl+Right`, `Ctrl+Home`/`Ctrl+End`, `Ctrl+Backspace`/`Ctrl+Delete`, `Ctrl+A`
> (select all), `Ctrl+Z` (undo), `Ctrl+Y` / `Ctrl+Shift+Z` (redo), `Ctrl+Space` / `Ctrl+N`
> (completion), `Shift+Tab`.

and the doc closes: "Every other key with Ctrl, Alt or Super held — Ctrl+X, Ctrl+C and Ctrl+V among
them — returns false from `ed_key` and is the host's."

So the standard keymap claims **`Ctrl+A`, `Ctrl+Z` and `Ctrl+Y`** on top of the two the spec names.
Cross-checking `shaderbox/commands.py` and `shaderbox/hotkeys.py`: none of those three has an app
binding today (no `K.a` / `K.z` / `K.y` chord in `COMMAND_SPECS`; `hotkeys.py` wires only Ctrl+C/X/V
against the system clipboard, which the editor explicitly returns to the host). So the collision set
is exactly the two the spec names, and the audit's rule ("the focused editor owns every chord its
ACTIVE keymap lists") resolves the other three for free. **The spec's conclusion is right; its
enumeration is a subset of the real keymap surface and the audit table (W-E's `02_keybindings.md`)
must be built from the doc, not from this bullet.** The spec already says the test loads the lists
"from the vendored `VERSION`'s docs (copied under `resources/editor/`) rather than retyped", which is
the correct mechanism — good design, and it makes the incomplete bullet harmless as long as the
implementer builds the table from the file.

**The gutter picture W-F promises is pinned upstream, exactly as claimed.**
`src/behavior_test.odin`, `test_relative_numbers_show_distance_and_an_absolute_cursor_line`, carries
the measured nvim picture in a comment (cursor on line 4 of 6: `3 2 1` above, absolute `4`
left-aligned on the cursor row, `1 2` below) and asserts `cursor_row_x < other_x`. W-F's description
("distance on other rows, absolute + left-aligned on the cursor row") matches it. Good.

### (c) W-G — the nested-dict contract's call sites

Read: `shaderbox/scripting/engine.py` in full, `shaderbox/project_session.py`'s tick / export /
script paths.

Every call site whose signature or argument shape changes under D3, and whether the spec names it:

| Call site | What changes | Spec names it? |
|---|---|---|
| `ScriptEngine.tick(document_id, document: EngineNode, ctx, stopped: frozenset[str])` | `document` becomes `Document`-shaped; `stopped` becomes `(pass, name)`-keyed | **Yes** (W-G names it verbatim) |
| `ScriptEngine.tick_export(document_id, document: EngineNode, ctx, behavior)` | same `document` reshape | **No** — see MISS G4 |
| `ScriptEngine.dry_run(document_id, document: EngineNode, sample_times, fps)` | same reshape; `ScriptProbe.driven` / `orphan_keys` become pass-qualified | **Yes** ("`dry_run` / export through the same routing") |
| `ScriptEngine.reload(document_id, scripts_dir, document: EngineNode)` | same reshape — it reads `document` only to hand it on, but the parameter type is `EngineNode` | **No** |
| `ScriptEngine._tick_script(...)` (private, 11 params) | the routing rewrite lands here | implied by "engine" |
| `ProjectSession.tick` → `self.script_engine.tick(document_id, ui_document.document.render_pass, …)` | passes `.render_pass` today — **this is the exact line 068 D7 was retracted over**; becomes `.document` | **Yes** (project_session.py named) |
| `ProjectSession._export_pre_render` closure → `tick_export(document_id, document.render_pass, …)` | same | file named, method not (G4) |
| `ProjectSession.reload_scripts` → `reload(document_id, scripts_dir_for(id), ui_document.document.render_pass)` | same | file named |
| `ProjectSession.write_script_source` → `reload(…, .render_pass)` then `dry_run(…, .render_pass, …)` | same, twice | file named |
| `ProjectSession._scriptable_uniforms_for(document_id)` | today returns the OUTPUT pass's uniforms (`…document.render_pass.get_active_uniforms()`); must become per-pass to feed the nested stub | **No** — and this is the function `script_stub_for` consumes |
| `ProjectSession._stopped_for(document_id) -> frozenset[str]` | returns `frozenset[tuple[str, str]]` | file named |
| `ProjectSession.is_uniform_stopped` / `set_uniform_stopped` / `set_document_all_stopped` | all name-keyed; need a pass | file named |
| `ProjectSession.uniform_is_driven` / `get_script_driven_uniforms` | `script_driven_uniforms(document_id) -> set[str]` becomes pass-qualified; consumed by the copilot's `set_uniform` reject | file named; **`copilot/backend.py` is not** |
| `script_stub_for(uniforms: Iterable[moderngl.Uniform])` | becomes `script_stub_for(document)` per D3 | **Yes** |
| `ui.py::_tick_frame_state` → `app.session.tick(tick_documents, now, dt, frame, mouse=…)` | unchanged shape, but `MouseState` gains fields | **Yes** |

**MISS G6 — `shaderbox/copilot/backend.py`.** It holds `_get_script_driven_uniforms` as an injected
callable and calls it at four sites; one of them gates `set_uniform` ("if name in
`self._get_script_driven_uniforms(document_id)`"). Under D3 a uniform name is no longer unique across
passes, so that membership test becomes ambiguous — a `u_time_scale` driven on `paint` would reject a
`set_uniform` on `composite`'s. W-G names `copilot/capabilities.py` ("read/write_script unchanged in
shape") and `copilot/prompt.py`, not `backend.py`. This is a behaviour change hiding inside a
"unchanged in shape" claim.

**MISS G7 — `ProjectSession._scriptable_uniforms_for`.** Not named, and it is the function that
decides what the stub lists. D3's stub is "one commented block per pass, each listing that pass's
scriptable uniforms"; today the function reaches for `.render_pass` only. Same class of defect as the
one 068 D7 was retracted over, in the sibling function.

**Test blast radius, for the wave's planning:** `tests/test_script_engine.py` has ~60 `eng.tick(...)`
call sites, `tests/test_script_engine_gl.py` has 7 (all passing `document.render_pass`),
`tests/test_script_dry_run.py` ~15, plus `test_export_script_wiring.py`, `test_copilot_script_tools.py`,
`test_script_driven_reject.py` and `test_script_api_doc.py`. The spec's bare "tests" is honest about
the shape but does not convey that this is the largest single test rewrite in the feature. Given
W-G is flagged high-blast-radius with a spec-fidelity auditor, that is acceptable — but the wave
should budget for it.

### (d) W-E — the chord collisions, both sides quoted

Read: `shaderbox/commands.py`, `shaderbox/hotkeys.py`, and `the editor repo (alexeykarnachev/editor) docs/standard_keymap.md`.

**Ctrl+N — CONFIRMED.**
App side, `shaderbox/commands.py`:
```python
CommandId.NEW_DOCUMENT, "New document", _chord(K.n, K.mod_ctrl), C.DOCUMENT
```
Standard keymap, `docs/standard_keymap.md` § "Selection, undo and completion":
> | `Ctrl+Space`, `Ctrl+N` | Opens the completion popup on the word before the caret; with it open, moves to the next candidate. |

**Ctrl+Space — CONFIRMED.**
App side, `shaderbox/commands.py`:
```python
CommandId.TOGGLE_DOCUMENT_PLAY,
…
_chord(K.space, K.mod_ctrl),
```
Standard keymap: the same row quoted above.

**Ctrl+Y — NOT an app collision.** There is no `K.y` chord in `commands.py` (grepped: zero hits for
`K.y` in a `_chord(` call). `Ctrl+Y` appears app-side only inside `_VIM_RESERVED_CHORDS`
(`hotkeys.py`: `frozenset("dufbeyrownphj")` — the `y` is vim's Ctrl+Y scroll). The standard keymap
claims it for redo:
> | `Ctrl+Y`, `Ctrl+Shift+Z` | Redo. |
So it is a **vim ↔ standard** cell in the audit table, resolved by "the focused editor owns every
chord its ACTIVE keymap lists" with no app move needed. The spec does **not** claim Ctrl+Y as a
collision — its "Known moves" list is exactly Ctrl+N and Ctrl+Space — so the spec is correct here.
Recorded because the audit table must still carry the cell.

Also confirmed: `hotkeys.py`'s clipboard handler wires Ctrl+C/X/V host-side and the standard keymap
explicitly returns those to the host ("Ctrl+X, Ctrl+C and Ctrl+V among them — returns false from
`ed_key` and is the host's"), so that trio needs no carve-out. The one existing carve-out,
NORMAL-mode Ctrl+W, is documented in `hotkeys.py`'s own comment and D7 ("decided by a generic rule,
not per-chord carve-outs") explicitly targets it — good, and the spec should expect the audit to
either justify or retire it.

### (e) W-D — does any single pass read two inputs from the same source? **No. The spec is right.**

Read both `graph.json` files in full and every `uniform sampler2D` declaration in Bloom's shaders.

Radiance Cascades (`77a84d27…`), per-pass inputs:
```
paint     {}
seed      {u_scene: paint}
jfa       {u_seed: seed,  u_prev: jfa}
df        {u_jfa: jfa}
cascade   {u_scene: paint, u_df: df,   u_prev: cascade}
composite {u_light: cascade, u_scene: paint}
```
Bloom Chain (`1c4f8a20…`):
```
scene     {}
bright    {u_src: scene}
blur      {u_src: bright}
trail     {u_src: scene,  u_prev: trail}
composite {u_lit: scene,  u_glow: blur, u_trail: trail}
```

No pass's `inputs` dict has two keys mapping to the same value. The nearest approaches are
`cascade` (three distinct sources: `paint`, `df`, itself) and Bloom's `composite` (three distinct:
`scene`, `blur`, `trail`). Feedback (`u_prev: self`) is D9's stated exception and never doubles with
another read of the same pass. **D9's naming rule is expressible on both shipped examples.**

**But the spec's own W-D sentence about this is factually wrong on the Bloom mapping.** It says:

> Bloom Chain's `u_src`/`u_lit`/`u_glow`/`u_trail` → `u_scene`/`u_scene`… by source pass — note
> Bloom's `bright` and `trail` both read `scene`, so two samplers named `u_scene` in different
> passes is fine

The observation about `bright` and `trail` is correct (both read `scene`, in different passes, so
two `u_scene` samplers coexist harmlessly). The mapping is not. Under D9 the correct Bloom renames
are:

| pass | today | under D9 |
|---|---|---|
| `bright` | `u_src ← scene` | `u_scene` |
| `blur` | `u_src ← bright` | **`u_bright`** (not `u_scene`) |
| `trail` | `u_src ← scene` | `u_scene` |
| `trail` | `u_prev ← trail` | `u_prev` (feedback, unchanged) |
| `composite` | `u_lit ← scene` | `u_scene` |
| `composite` | `u_glow ← blur` | **`u_blur`** (not `u_glow`) |
| `composite` | `u_trail ← trail` | `u_trail` (already matches) |

So `u_src` maps to **two different** new names depending on the pass, and `u_glow` becomes `u_blur`.
The spec's "→ `u_scene`/`u_scene`…" reads as if one name absorbs the lot. Since D9's whole payoff is
that the wire name is derivable, getting the derivation wrong in the spec that states the rule is
worth fixing before the wave runs. Shader-side sites to change, from grep:
`bright.frag.glsl` (2), `blur.frag.glsl` (3), `trail.frag.glsl` (2), `composite.frag.glsl` (`u_lit`
2, `u_glow` 2, `u_trail` 3), plus the `graph.json` keys.

---

## Task 4 — order

### Verdict: **PARTIAL** — three of the seven stated dependencies are real and cited in code; four
are preference. Three waves can run in parallel.

The spec's heading claims "**Order (dependencies, not preference)**". Taking each link:

**Real dependencies, forced by code or by an artifact:**

1. **W-F → W-E's keymap half.** Real and the spec states it correctly ("The keymap setting's ABI
   half waits for W-F's re-vendor"). Verified: the vendored `VERSION` is `e7db554`;
   `docs/standard_keymap.md` first lands at `7ae414c`, after it. `ed_set_style` exists at HEAD but
   the vendored `.so` predates the standard keymap's implementation, so binding it against
   `e7db554` would set a style the binary does not have. The spec's own hedge ("land the setting
   last within W-E or first within W-F") is the right resolution.
2. **W-D → W-H.** Real. The tutorial's wiring lines quote uniform names (`tutorial_body.html` has 9
   hits for `u_scene` and 5 for `u_light`), and W-H's `build_tutorial.py` generates the pass cards
   **from `graph.json`**. Renaming after the tutorial is generated would make the generated cards
   disagree with the prose. The spec states this.
3. **W-A → W-H.** Real. W-H's "Before you start" step says "512×512 via the new preset", which does
   not exist until W-A ships it (confirmed: `tabs/document.py`'s `resolution_items` is a fixed list
   with no 512 entry, and the ledger's #1 verifies the same).

**Stated as dependencies but actually preference:**

4. **W-C first ("small, unblocks the walk itself").** This is a value judgement, not a dependency.
   Nothing in W-A/W-B/W-D/W-E/W-F/W-G reads anything W-C creates. The one coupling is that W-C's
   `ADD_PASS` chord comes "from W-E's audit" — which makes W-C **depend on W-E**, the reverse of the
   stated order. The spec says "`ADD_PASS` (chord from W-E's audit; Alt+A candidate)" and Open
   question 4 leaves the chord to the maintainer. So W-C wave 1 must either ship with a provisional
   chord and revisit, or ship without the chord and add it in W-E. **This is the one ordering
   inconsistency inside the spec itself.**
5. **W-B third ("independent; early so every later UI wave is written under D1").** The spec admits
   it: "independent". Sequencing it early is a good discipline argument — the prose gate exists
   before the waves that write prose — but it is preference, and the spec's heading over-claims.
6. **W-E → W-F ("because the status line and the host's key routing both hang on which focus
   notions remain").** Partly real. W-F's status line (D6) draws inside the editor rect and needs
   to know whether the editor is a focus stop — which W-E preserves explicitly ("The editor's focus
   stop and the copilot focus gate stay"). Since W-E *keeps* the editor focus stop, W-F's status
   line does not actually change under it. The real coupling is the other direction (#1 above).
   Preference, defensible.
7. **W-G → W-H ("so the paint step can be written against it").** Real **only if** the tutorial's
   paint step becomes mouse-driven, which the spec flags as optional ("Tutorial's paint step **may**
   then be mouse-driven"). If it stays analytic (068 D6's shipped shape), W-H does not depend on
   W-G. Conditional dependency, correctly hedged in the wording but presented as fixed in the order
   list.

**What can run in parallel:**

- **W-B is fully independent** of every other wave (its own admission; its files are
  `popups/pass_settings.py`, `widgets/pass_list.py`, `tabs/document.py`, `ui_primitives.py`,
  `theme.py`, one new test). It shares `popups/pass_settings.py` with W-C and
  `tabs/document.py` with W-A, so parallel means merge conflicts, not logical conflicts.
- **W-E and W-G are independent of each other** and of W-A/W-B: their file sets intersect only at
  `app.py`, `ui.py` and `commands.py` — every wave touches those three, so file-level serialisation
  is unavoidable regardless of order. Logically, nothing in W-G reads a region symbol and nothing in
  W-E reads the script engine.
- **W-F is independent of everything except the keymap-setting half of W-E.**

So the genuine partial order is:

```
W-A ─┐
W-B ─┼──> W-H
W-C ─┤          (W-C also needs W-E's chord decision)
W-D ─┘
W-E ──> (keymap half) ──> after W-F's re-vendor
W-F
W-G ──> W-H  (only if the paint step becomes mouse-driven)
```

The eight-wave serial order is a **legitimate scheduling choice for a solo maintainer** — one wave,
one commit series, one review pair, no merge conflicts on the three shared files. That is a good
reason. It is just not "dependencies, not preference", and the heading should say so, because a
future reader treating the order as forced will not notice that W-C's chord depends backwards on
W-E.

---

## False trails

Things I probed that turned out fine, recorded so a later reviewer does not re-probe them.

- **`preview_cell`'s other three consumers.** I expected W-D's "no sublines" node to force a
  signature change rippling into `document_grid.py`, `popups/examples.py` and
  `exporters/telegram.py`. It does not: `sublines: Sequence[str] = ()` is already defaulted and the
  new graph view simply omits it. Call-site change only.
- **Example api-lock tests breaking under D9.** A grep for the six uniform names hit 17 test files,
  which looked like a large W-D blast radius. Inspecting each: every hit is a local fixture literal.
  `test_radiance_cascades_example.py` and `test_example_library.py` contain **zero** hits for any of
  the six. The rename does not break a single test assertion about a shipped example.
- **`projects/dev` script migration.** W-G's "`projects/dev` scripts hand-edited to the new shape"
  looked like it might collide with the no-migration rule. It does not — it *is* the sanctioned
  form. And `find projects/dev -name script.py` shows the only four are under `trash/`, so there is
  nothing live to hand-edit.
- **`Ctrl+Y` as an app collision.** My brief listed it alongside Ctrl+N and Ctrl+Space. It is not
  one: no app command binds it, and the spec does not claim it does. Vim↔standard cell only.
- **`todo.md` triggers.** `dev_flow.md` step 2's pre-flight requires grepping `todo.md` by `Trigger`
  before drafting. The file has exactly one live entry, `[VERIFY] Live-only UI checks, unverified on
  this box`, whose trigger is "next `make run` on a machine with a display". Nothing in 069 fires it
  (though W-A/W-B/W-C/W-D/W-F all add live-only UI checks to its pile — worth a mention in the wave
  that runs on a display, not a spec change). The spec's "`todo.md` untouched (frozen)" is correct.
- **`modal_window` auto-size.** W-B asks for it and I checked whether it already exists.
  `ui_primitives.py::modal_window(label, size, flags=0, fixed_size=False)` has no auto-size mode —
  it seeds `size` with `first_use_ever` or forces it with `always`. So W-B's parenthetical
  "(`modal_window` auto-size)" is an honest new capability, correctly named in the file list.
- **`Document.first_render_done` colliding with W-C's new `Pass.first_render_done`.** Different
  objects, no shadowing; the document-level flag is set inside `render()` only when `canvas is None`,
  and the five readers are all document-scoped. The parallel naming is a feature, not a hazard.
- **`assert_plan_invariants` blocking W-C's per-pass render.** I expected the draw-once assert to
  fire when a first-render pass and the output chain share an ancestor across two `render` calls in
  one frame. It cannot: the assert runs per-plan inside `evaluation_order`, and each call builds its
  own plan. The invariant genuinely does not cover cross-call duplication — which is why W-C's
  proposed test ("the steady state draws only the output chain") is the right guard, and the spec
  already names it.

---

## Recommendation

**Do not proceed to implementation until C1, C2, the W-D Bloom mapping (Task 3(e)) and the W-C
`Document.render` signature (Task 3(a)) are resolved in the spec.** The eleven Files-touched misses
in Task 2 are lower-stakes — each is a file an implementer would find at `make gates` time — but
G5 (the five shipped `document.json`s carrying `stopped_uniforms`) and G6 (`copilot/backend.py`'s
name-keyed driven check) are behaviour changes hiding under "unchanged in shape", and D3 in W-D
(the `u_light_*` prefix collision) is a silent data-corruption risk in a rename wave.

The spec's design judgement is sound throughout — the redesign-not-patch rule is genuinely applied,
the gates are shipped with the sweeps that need them, and the two prior decisions it supersedes are
named. The defects here are enumeration defects, not design defects.
