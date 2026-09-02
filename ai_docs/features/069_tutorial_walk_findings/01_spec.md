# 069 — the 068 walk: findings into a plan

The maintainer walked the 068 Radiance Cascades tutorial in the app and filed 37 findings
(`00_findings.md`, each verified against the code; the scripting design is in
`01_design_scripting.md`). This spec groups them into eight workstreams, fixes the order, and locks
the decisions the maintainer made during the walk. **Nothing here is an ad-hoc patch: where a
finding needs a redesign, the redesign is the fix** (maintainer's rule for this session).

Size: **high-blast-radius** — it removes a subsystem (019 keyboard regions), changes the script
contract, re-vendors the editor, rewrites a tutorial, and touches `conventions.md`. It lands as
eight waves in the order of § Order, each wave its own commit series with its own pre/post review
per `dev_flow.md`. One spec, because the waves share decisions and the tutorial rewrite depends on
four of them.

---

## Goal

After this feature, a reader can build the Radiance Cascades example from the tutorial alone, in
the app, with no step that names a control that does not exist, no crash, and no console-only
diagnostic — and the app surfaces the walk exposed (pass settings, the strip, the editor chrome,
keyboard ownership, scripting across passes) are redesigned where the walk showed the design was
wrong, not patched.

## Out of scope (each with a trigger)

- **Cross-document composition.** Unchanged from 065. Trigger: unchanged.
- **The graph view of the pass strip (#19 option A).** Maintainer's call: a separate feature
  (070), not this one. This feature ships #19's option B — the strip loses its sublines (the
  truncated `u_x <- y` text), keeps name + thumbnail. Trigger: 070 opens once 069 lands; its
  direction is fixed (thumbnails as nodes in evaluation order, edges labelled by pass name under
  D9, feedback as a loop mark, read-only, draw-list only — not `imgui_node_editor`).
- **Keys in the script context** (`ctx.keys`). Trigger: a second script-side use for keyboard
  input after the clear-canvas command exists (finding #23 option B).
- **Absolute per-pass sizes** (a pass larger than the canvas). Trigger: an effect that needs a
  supersampled intermediate; today `scale <= 1.0` stands (#4).
- **Windows `libeditor.dll`.** Still owed from 067; the re-vendor here is Linux `.so` only.
  Trigger: next `/ship` on a Windows host.
- **A standard-keymap gutter/status design of its own.** W-F ships vim's furniture; under the
  standard keymap the status line shows a caret readout only. Trigger: the maintainer uses the
  standard keymap daily and wants more.

## Locked decisions (from the walk — constraints, not options)

- **D1. UI strings have a word budget** (`.claude/skills/imgui-ui/SKILL.md § 2`): label 1-2 words;
  icon tooltip = the control's name; `help_marker` one clause, <= 8 words, only where the label is
  ambiguous; empty state <= 4 words; derived values in the control, never the label. Enforced by a
  test (W-B).
- **D2. No manual JSON editing is a workaround.** Every state the tutorial needs is reachable in
  the UI (#1, #3).
- **D3. Scripting across passes: one document script; `update` returns
  `{pass: {uniform: value}}`; a non-dict value under a bare key broadcasts to every pass declaring
  that uniform; a pass block wins over a broadcast; unknown pass / uniform = strip error.** The
  value-type dispatch rests on one invariant the engine asserts: no uniform value is ever a
  `dict` (`coerce_one` accepts scalars, tuples, `Vec*`, `Array`, text — never a mapping).
  (`01_design_scripting.md § Decision: B1` + `§ Bare keys broadcast`.) Supersedes 065 D12 (per-pass
  files) and lifts 068 D7's retraction.
- **D4. The active-region system (019) is removed whole.** No outline, no region cycling, no
  imgui keyboard nav. The editor's focus stop and the copilot focus gate stay (#24).
- **D5. The editor keymap (vim / standard) is a global Setting**, one control in the Settings
  popup's Editor section, applied to every open editor session (#13).
- **D6. Vim furniture is drawn INSIDE the editor rect** — status line (mode, `line:col`, the
  command line) and gutter (relative numbers, `~` filler) — in the editor's own visual language,
  never on the host's bottom bar (#11, #12).
- **D7. Every keybinding has exactly one owner per (app, vim-focused, standard-focused) cell**,
  decided by a generic rule, not per-chord carve-outs (#26).
- **D8. Tutorial pass steps follow one template**, pass cards and pass code are GENERATED from the
  shipped example by `build_tutorial.py` (#27, #31, #34, #35).
- **D9. Input uniforms are named after the pass they read, `u_<pass>`; feedback is `u_prev`.**
  Applied to both examples and the tutorial; the gear default-wires by name (#37).
- **D10. Add pass activates the pass** (opens its tab, makes it the output), then opens the gear
  (#28). **D11. Inline inputs commit on deactivate-after-edit**, Enter is a shortcut (#18) — rule
  filed into the imgui skill § 7.5 with the wave.
- **D12. The pass strip loses its wiring sublines now; the graph view is feature 070.** The
  strip shows name + thumbnail only (wiring is the gear's, and 070's). Maintainer's call.

Open questions for the maintainer are in the last section; none of the above is one.

---

## Workstreams

Each lists: findings folded, the shape, the files, what pins it (test or check).

### W-A — Canvas size and the viewer (#1 #2 #3 #4 #21)

- The Document tab's Resolution combo routes through `Document.set_canvas_size` (the funnel;
  `tabs/document.py:111` bypasses it today — root of #2/#3). Test: after a UI resize, every
  non-output pass's `target_size(document.canvas_size)` follows on the next render.
- Resolution control redesign: today's combo is fed a literal index `0` every frame
  (`tabs/document.py:109` — it has no selection state, it is a menu) and its list is built from
  `render_pass` (the last output-keyed row in a panel that otherwise uses `panel_pass` /
  `document`). It becomes one control with state: a `W × H` pair of `input_int`s showing
  `document.canvas_size` (commit on deactivate-after-edit, D11; clamped to the copilot's
  existing range from `copilot/backend.py:135`, moved to one shared constant), plus a presets
  menu beside it (squares 256 / 512 / 1024 / 2048, the named video shapes, and any bound
  texture's size) that writes the pair. Both paths call `Document.set_canvas_size`.
- Gear: the output pass's size slider is disabled with the help text it already has (#4).
- Viewer: checkerboard behind the preview + 1px border (#21). Two greys from `theme.py`, no
  literal colours at the call site.
- Files: `tabs/document.py`, `popups/pass_settings.py`, `ui.py` (preview), `theme.py`,
  `document.py` (no change; the funnel exists), `copilot/backend.py` (clamp moves to a shared
  constant), tests.

### W-B — Prose diet, gear layout, engine-uniform block (#5 #7 #10 #32 + D1)

- Every `help_marker(` / `set_tooltip(` / `separator_text(` / empty-state literal cut to budget.
  `_FORMATS` tooltips become one clause each. The gear's Reads header tooltip goes; the empty state
  is "no sampler2D uniforms".
- Gear popup sizes to content (`modal_window` with auto-resize or a computed height), no
  scrollbar. The size row: label `size`, slider format `%.0f%% · WxH` (derived value in the
  control, D1).
- Engine-driven uniforms: a vertical block under the sort row, one per line, fixed name column,
  outside the sorted list (#32).
- **Gate:** a test walks `shaderbox/` with `ast` over every call to `help_marker`,
  `set_tooltip`, `separator_text`, `label_row`, `row_label`, and `text_colored` with a
  `COLOR.FG_DIM` first arg (the empty-state idiom). Census at writing, by AST (grep undercounts multi-line calls): `help_marker` 7 sites,
  `set_tooltip` 19, `separator_text` 10, `text_colored`+`FG_DIM` 31; round 3's throwaway
  implementation measured 62 sites, 26 in the allowlist, 11 flagged including all four ledger
  strings. The test pins the allowlist, not a site count. It scores `Constant`
  strings AND `JoinedStr` (each `FormattedValue` counts one word — `pass_settings.py:128`'s "(?)"
  tooltip is an implicit concatenation with an f-string and parses as `JoinedStr`), asserts word
  counts against D1, and asserts a `label_row` / `row_label` label carries no `FormattedValue`
  at all (a label is fixed text; `pass_settings.py:178`'s `f"size ({w}, {h})"` is the overflow
  #7 reported and the gate must refuse it). A `Name` argument is resolved to its assignment in
  the enclosing function when that is a string; otherwise the site is listed in a pinned
  allowlist in the test (the `ui_primitives` shared helpers that forward a caller's text — their
  callers are the measured sites), so an unmeasured site is a deliberate entry, not a blind spot.
  Ships in the same commit as the cut. Runtime-built text (the vim status line, notifications)
  stays outside it, and the UI waves' reviews cover those by eye.
- Files: `popups/pass_settings.py`, `widgets/pass_list.py`, `tabs/document.py`, `ui_primitives.py`
  (`modal_window` auto-size), `theme.py` (`PASS_SETTINGS_H` goes), `tests/test_ui_prose_budget.py`.

### W-C — Pass verbs: crash, commit, activate, hotkeys, first render (#9 #17 #18 #25 #28 #36)

- Rename crash: `_draw_name` returns `renamed: bool`; `_draw_body` returns early on a rename so
  the frame never indexes `document.passes` with a dead name. Test drives `_draw_body` through a
  rename headlessly (the imgui-context test rig from `/imgui-ui § 0`).
- Commit on `is_item_deactivated_after_edit()` + Enter, and on Close/Escape with a pending edit.
  Never per keystroke (rename moves a file and rewrites edges). Same rule applied to
  `_draw_add_input` and every § 7.5 inline input; the skill's § 7.5 rewritten.
- Add pass activates (D10): after `add_pass`, call the tile-click path (open tab + set output),
  then `open_pass_settings`. Copilot has no pass tools — nothing to mirror.
- Commands: `OPEN_PASS_SETTINGS` (Alt+P) and `ADD_PASS` (Alt+A) — provisional per the closed
  questions; W-E's audit may move them, W-C does not wait for it. Dispatch to `open_pass_settings(panel_pass(...).name)` / `pass_add.open`. The
  chord-uniqueness test and the generated Help shortcuts table pick them up.
- First render of every pass (#36 option A): `Document.render` gains `target: str | None = None`.
  Decisions, each a branch in `document.py::render` that reads `output` today: (1) `target` (or
  the graph output when None) feeds the early-out guard, `plan_for_output`, and the cycle
  fallback `order = [output]` — all three, not two; (2) a target draws its WHOLE ancestor chain
  (a pass alone would sample black inputs and show a wrong picture); ONLY a target render skips
  passes already drawn this frame — `Pass.drawn_frame: int`, set by every `render` against the
  document's `_frame` from `begin_frame` — so the sweep never redraws what the output chain drew
  and an iterated feedback pass never advances twice. Output renders never skip: the preview
  render (`ui.py:265`) and the own-canvas render (`ui.py:301`) both draw the chain each frame
  today and keep doing so (a skip there would blank every thumbnail). Target renders are issued
  only by the main frame gate over ticked documents, where `_frame` is defined; the examples
  popup (`ui.py:298`) renders whole documents and never a target; (3) a target pass sizes by its own scale and never receives the external `canvas` (the
  two `name == output` comparisons use the graph output, unchanged); (4) `Pass.first_render_done`
  is set by `render` on every pass it draws; `Document.first_render_done` keeps its meaning
  (output drawn, `canvas is None and target is None`). `ui.py`'s frame gate draws at most one
  pass with `first_render_done == False` per frame via `render(target=name)`. Export is
  untouched: `target` is not threaded through `_render_image` / `_render_media_into`, by
  intent. The stale wash then means "was live, is no longer". Test: every pass drawn exactly
  once across the first N frames, and exactly the output chain per frame afterwards.
  Test: after load, N frames later every pass has rendered once; the steady state draws only the
  output chain (the draw-once invariant assert in `pass_graph.py` stays the guard).
- Files: `popups/pass_settings.py`, `widgets/pass_list.py`, `commands.py`, `app.py`, `ui.py`,
  `core.py` (`Pass.first_render_done`), `document.py`, tests.

### W-D — Naming, default wiring, strip tune (#19 #37 + D9 D12)

- Naming: rename inputs in both shipped examples' `graph.json` + shaders. Radiance Cascades:
  `u_scene`→`u_paint` (on `seed`, `cascade`, `composite`), `u_light`→`u_cascade` (on
  `composite`). Bloom Chain, by source pass: `bright.u_src`→`u_scene`, `blur.u_src`→`u_bright`,
  `trail.u_src`→`u_scene`, `composite.u_lit`→`u_scene`, `composite.u_glow`→`u_blur`,
  `composite.u_trail` stays. Two passes each naming a `u_scene` is fine (different programs);
  inside ONE pass two inputs from the same source would collide under D9, and none exists in
  either example (verified from both `graph.json`s). The rename is per-file and exact-token,
  limited to the two multi-pass examples: unrelated single-pass examples declare `u_light_*`
  (`8d454b7b…/passes/main.frag.glsl:56-60`) and `u_glow_*` (`0b0d16bb…:12-13`) uniforms whose
  values persist in their `document.json`; a prefix-blind replace corrupts them. Verify by
  loading every example headlessly after the rename (the resolve-clean example test). Rule stated in the Help panel's pass section and the copilot prompt's
  pass block; the api-lock tests for examples updated.
- Default wiring by name is a RESOLUTION rule, not a stored edge. Two facts force that: the
  hot-reload seam (`watch.py::_reload_pass_if_changed`) has no graph and no compile — it
  `invalidate()`s and the samplers are unknowable there (forcing a compile inverts 066 D1); and
  `PassGraph.with_input(…, "")` deletes the key, so an explicit "(none)" is byte-identical to
  never-wired and any rule keyed on absence would re-wire against the user's choice. So: (1)
  `with_input(name, "")` STORES the empty string — absent key = never decided, `""` = explicitly
  none (`graph.json` gains `""` values; the two multi-pass examples are hand-checked); (2) the
  effective input of a sampler with an ABSENT key is `u_<x>` → pass `<x>` when such a pass
  exists (`u_prev` → the pass itself), else black — computed in one pure function
  `pass_graph.py::effective_inputs(entry, samplers, passes)` (GL-free: it takes sampler NAMES)
  that `Document.render`, the planner (`plan_for_output` must see auto edges to order the draw
  and to detect cycles), the gear, and the strip all call. The sampler names come from
  `Pass.get_active_uniforms()` of COMPILED passes only — `Document` gathers them before planning;
  an uncompiled pass contributes no auto edges until its first render, which W-C's sweep
  guarantees within N frames (and opening its gear compiles it). So the draw order may change
  between frame 1 and frame N as passes come online — one frame from black, the same as any
  lazy first render — and 066 D1 (no eager compile) holds; compiling every pass at the first
  render is rejected; (3) a sampler whose `uniform_values` entry is a user-bound texture
  (`MediaWithTexture`) is never auto-wired — `core.py:373`'s `inputs.get(name,
  uniform_values.get(name))` would otherwise let a pass named `image` silently replace the PNG
  in the `Media Input` example (`73ea2431…`); (4) the gear's combo has three kinds of item — "auto: x" / "auto: none" (absent key; picking
  it DELETES the key via a new `PassGraph.without_input`), "(none)" (stores `""`), and each
  pass by name (stores it) — so the three stored states display distinctly (today's two-item
  combo shows `(none)` for both absent and `""`, because `"" in choices` is false and the index
  falls to 0), and the copilot's `edit_shader` path gets the
  same behaviour with no code, since resolution happens at render. Tests: `effective_inputs`
  over (absent, "", explicit) × (pass exists, not) × (media-bound); the planner sees the auto
  edge; declaring `u_df` in a fresh pass beside `df` renders `df`'s texture without touching the
  gear; a stored `""` stays black across a reload.
- Strip tune (D12, the maintainer's "at least tune the current visuals"): `_draw_pass_tile`
  passes no `sublines` (the truncated `u_x <- y` text and the arrow go); the "has compile errors"
  subline becomes the error border it already has; the card is then thumbnail + one footer line,
  and the strip's spacing is re-measured against that shorter card (`SPACE.MD` between cards, no
  slack below the footer). `SIZE.PASS_THUMB` stays 112 — the maintainer chose that number in 065
  round two. Everything structural is 070.
- Files: `widgets/pass_list.py` (`_strip_order` stays; `tests/test_pass_verbs.py` imports it),
  `project_session.py` (default wiring), both examples' `graph.json` + shaders, `help_content.py`,
  `copilot/prompt.py`, tests.

### W-E — Keyboard ownership: regions out, keymap setting, the audit (#13 #24 #26 + D4 D5 D7)

- Remove 019: `nav_enable_keyboard` off; delete `ActiveRegion`, `active_region_outline` and its
  six call sites, the App region state and methods (`cycle_region`, `_set_region`,
  `focus_move_in_flight`, `region_derive_allowed`, `region_outline_visible`,
  `_yield_editor_to_region`), the derive blocks, `CYCLE_REGION`, and all FOUR `no_nav_inputs`
  sites (`ui.py`: editor child, the copilot bar child at `:553`, the panel child;
  `copilot_chat.py:53`) — with nav off the flag is inert everywhere, so all four go.
  **SUPERSEDED: the flags STAY, all five of them** (the census was four, and the fifth is
  `document_grid.py`). imgui runs basic Tab traversal regardless of `nav_enable_keyboard`, so
  the flag is not region machinery — only the region CONDITION goes. Measurement, ruling and
  the count are in `50_wave_e_keyboard.md § Design decisions item 1`. The
  `nav_flatten` machinery goes with it: `preview_cell`'s `nav_flatten` parameter and docstring
  paragraph (`ui_primitives.py`), its call sites (`widgets/document_grid.py` ×2,
  `popups/examples.py:101`), and `tabs/document.py`'s `ChildFlags_.nav_flattened` plus its
  now-false comment. `FOCUS_TAB_*` keep selecting the tab. `_focus_or_add_tab` becomes
  plain "open + focus editor". `tests/test_pass_editor_wiring.py:173` rewritten. Docs: 019 gets a
  "removed by 069" banner; the `conventions.md` "App-wide keyboard nav is region-confined" entry deleted; the `SELECT`-hue assertion keeps
  its own justification or goes (decide at impl by reading `theme.py:193`);
  `067_custom_editor.md` gets a "keymap routing superseded by 069" note where it states the
  vim-only `_VIM_RESERVED_CHORDS` routing and the editor child's `no_nav_inputs`; the
  `conventions.md` code-editor entry ("Inline editor state lives on `App` … one libeditor instance
  per opened FILE") has its revisit trigger fired by D5 ("a non-modal keymap ships editor-side")
  — rewrite its "vim-modal library" sentence to "vim-modal or standard, per `EditorSettings.keymap`".
- Keymap setting (D5): `EditorSettings.keymap: Literal["vim", "standard"] = "vim"`; one combo in
  Settings → Editor; `editor/ffi.py` binds `ed_set_style` / `ed_style`; applied in
  `_apply_editor_settings_to`. Depends on W-F's re-vendor (the standard keymap is upstream since
  `e7db554..68def59`).
- The audit (D7): one table in this spec's `02_keybindings.md` — every app chord × {app
  unfocused, vim focused, standard focused}, winner per cell. Rule: the focused editor owns every
  chord its ACTIVE keymap lists (read from the editor repo's `docs/vim_coverage.md` /
  `docs/standard_keymap.md` at the vendored commit — they are not in this repo until W-F copies
  them under `resources/editor/`); the app owns the rest; app chords that must work inside the editor live on
  Alt or F-keys. Known moves: `NEW_DOCUMENT` off Ctrl+N, `TOGGLE_DOCUMENT_PLAY` off Ctrl+Space
  (both collide with standard completion); the standard keymap also owns Ctrl+A / Ctrl+Z /
  Ctrl+Y / Ctrl+Shift+Z, which no app chord uses today — the table records the editor as owner
  so nothing lands on them later. The vim-reserved set becomes per-keymap data in `hotkeys.py`. Test: app chords disjoint from both keymaps' chord lists, lists loaded from the
  vendored `VERSION`'s docs (copied under `resources/editor/`) rather than retyped.
- Files: `app.py`, `ui.py`, `ui_regions.py` (`ActiveRegion` out, `DocumentTab` stays),
  `ui_primitives.py`, `widgets/document_grid.py`, `widgets/copilot_chat.py`, `popups/examples.py`,
  `tabs/document.py`, `commands.py`, `hotkeys.py`, `ui_models.py`, `popups/settings.py`,
  `editor/ffi.py`, `conventions.md` (two entries), `067_custom_editor.md`, tests.

### W-F — Editor chrome and lib issues (#11 #12 #14 #15 #16 + D6)

- Re-vendor `libeditor.so` from upstream HEAD (`68def59` at writing; rebuild from a clean tree per
  067 § 13, update `resources/editor/VERSION`, and the `conventions.md ## Known quirks` entry
  that owns the vendored version + rebuild procedure). Brings the standard keymap for W-E.
- **Issues to file on `alexeykarnachev/editor`** (none exist; `gh issue list` is empty), each
  with the repro from the ledger: (1) visual-mode `p`/`P` replaces the selection (#16 — a real
  bug, ShaderBox-side nothing); (2) export gutter + status-line emission through the ABI, or a
  per-row query, so a host draws vim's exact furniture without re-deriving `chrome_emit_gutter`
  (#12, #11); (3) a way for a marker to override the TEXT colour on its line (the marker already carries an
  independent gutter RGBA and a background fill — what is missing is the foreground, vim's
  highlight-group semantics) (#14);
  (4) markers anchored to text so edits shift them (#15). ShaderBox does not wait on (2)–(4).
- Status line inside the editor rect (D6): `tabs/code.py` reserves one cell row at the bottom of
  the editor image (layout height − `cell_h`), draws a `STATUS_BG` strip on the cell grid: mode
  badge + `line:col` left, the command line replacing them while open, ruler right; under the
  standard keymap a caret readout only. The bottom bar keeps file path / compiled / Open dir /
  Copilot.
- Gutter (D6): set `ChromeFlag.RELATIVE_NUMBERS` under vim; host `_draw_gutter` draws the picture
  `behavior_test.odin:338` pins (distance on other rows, absolute + left-aligned on the cursor
  row) and `~` on rows past the buffer end (`ed_filler_glyph`); absolute numbers, no filler under
  standard. Replaced by the ABI emitter when issue (2) lands.
- Error lines (#14 host side): no whole-line fill; the marker's gutter mark (already passed, never
  drawn) draws in `STATE_ERROR` in the gutter + a 2px left bar on the row; the red tab tint stays.
- Stale markers (#15): the marker fingerprint includes `is_tab_dirty`; while dirty, no line
  markers; the error strip stays with a dim "(stale until save)" suffix — one clause, D1.
- Files: `resources/editor/*`, `tabs/code.py`, `editor/ffi.py` (bind `ed_set_style`,
  `ed_style`, `ed_filler_glyph`), `ui.py` (bottom bar), `theme.py` (status tokens if missing),
  tests for the gutter label function (pure: `(line, current, count) -> (text, left_aligned)`).

### W-G — Scripting across passes, mouse, clear (#22 #23 #29 #30 + D3)

- Engine (D3, `01_design_scripting.md`): `ScriptEngine.tick` / `tick_export` / `dry_run` /
  `reload` take the `Document` (every caller in `project_session.py` — `tick`, the export pre-render closure,
  `write_script_source`'s reload + dry_run — stops passing `.render_pass`), route per the decision,
  orphan-check per pass; `stopped` keyed `(pass, name)`; `script_stub_for(document)` emits the
  nested skeleton with one commented block per pass, fed by `_scriptable_uniforms_for` reshaped to
  per-pass (today it reads `.render_pass` only — the sibling of the 068 D7 defect). `EngineNode`
  protocol retired for a `Document`-shaped one (passes + their uniforms).
- Driven / stopped state goes pass-qualified end to end: `UIDocumentState.stopped_uniforms`
  becomes a list of `[pass, name]` pairs (the five shipped examples' `document.json` files that
  persist `"stopped_uniforms": []` are hand-edited, NO migration code); `is_uniform_stopped` /
  `set_uniform_stopped` / `set_document_all_stopped` / `uniform_is_driven` /
  `get_script_driven_uniforms` take a pass — and so do their `App` wrappers
  (`App.set_uniform_stopped` / `set_document_all_stopped`, `app.py:1414-1435`) and the panel's
  call site (`tabs/document.py:253`); `widgets/uniform.py`'s play/stop button passes the panel
  pass; the `EngineNode` `stopped: frozenset[str]` parameter type and `DocumentScripts.last_driven`
  / `last_skipped` become `(pass, name)`-keyed; the live tick (`project_session.py:598`) is the
  fourth `.render_pass` seam and the one whose protocol type changes. In `copilot/backend.py` the
  site this genuinely fixes is `_pass_views` (`:740`), which loops every pass against one
  document-scoped driven set; the `set_uniform` gate (`:889`) resolves against `render_pass`
  only and stays name-keyed within that pass. The engine's soft error for an orphan records the
  PASS it named (a new field on the error, not only the key), so a shader tab can show the
  errors that name its pass. Persisted shape: `stopped_uniforms: list[StoppedKey]` with
  `StoppedKey(pass: str, name: str)` a `BaseModel` — element-level salvage, and a stale
  `list[str]` fails `validate_assignment` and drops to `[]` under the existing `drop_invalid`
  policy (`model_salvage` needs no change). Seven shipped examples exist; five persist
  `stopped_uniforms` and are hand-edited; the two that do not are exactly the multi-pass ones.
  First launch after W-G logs one salvage line per stale `projects/dev` document — expected, and
  the verification list says so.
- The SCRIPT API prompt block: `scripting/api_doc.py` generates it from `MouseState`'s dataclass
  fields and the stub, `copilot/prompt_context.py` is its importer — both change;
  `scripting/__init__.py` re-exports follow. Tests: this is the largest test rewrite in the
  feature (`test_script_engine.py` ~60 tick sites, `test_script_engine_gl.py`,
  `test_script_dry_run.py`, `test_export_script_wiring.py`, `test_copilot_script_tools.py`,
  `test_script_driven_reject.py`, `test_script_api_doc.py`) — budget for it. The console `logger.warning` for orphans
  goes; the strip already shows soft errors on the script tab (`tabs/code.py:130`) — keep, and
  add the same strip on a SHADER tab whose pass the script names wrongly (today only the script
  tab shows them).
- `MouseState` gains `down: bool` and `prev_x`/`prev_y`; `ui.py` fills them from
  `is_mouse_down(0)` inside the existing hit-test; `EXPORT_MOUSE` keeps `down=False`,
  prev = current. **No builtin `u_mouse` uniform** (#22's open half, default picked): the mouse
  reaches shaders through the script only, so the tutorial's paint step is written against D3 —
  which restores 068's lost "stress the scripting path" goal (068 D7 lifted). Trigger to add a
  builtin: a second mouse-driven example that needs no other script state.
- `CommandId.RESET_FEEDBACK` ("Clear canvas") → `Document.reset_feedback` on the current
  document; chord from the audit; a small ghost button by the preview.
- Docs: the `conventions.md` scripting entry ("A broken script is error-as-data" … "PLAY/STOP is
  document-scoped + name-keyed") rewritten for the nested contract and the `(pass, name)` key; 065 D12 and
  068 D7 get "superseded by 069" lines; the copilot SCRIPT API prompt block regenerates
  (`tests/test_script_api_doc.py` pins it); the Help panel's script section updated.
- Files: `scripting/engine.py`, `scripting/context.py`, `scripting/behavior.py`,
  `scripting/api_doc.py`, `scripting/__init__.py`, `project_session.py`, `ui_models.py`,
  `widgets/uniform.py`, `ui.py`, `commands.py`, `app.py`, `copilot/prompt.py`,
  `copilot/prompt_context.py`, `copilot/backend.py`, `copilot/capabilities.py` (read/write_script
  unchanged in shape), `help_content.py`, `conventions.md`, the five example `document.json`s,
  `projects/dev` scripts + `document.json`s hand-edited to the new shape (NO migration code), tests.

### W-H — The tutorial rewrite (#1 #6 #8 #20 #27 #31 #33 #34 #35 + D8 D9)

- Template per pass step: heading = pass name (subtitle = concept); pass card (name · reads · size
  · format · smooth · repeat · runs, "default" where it holds); what it produces + picture; the
  COMPLETE shader; explanation. Concept sections (naive GI, sphere marching, the cascade idea, the
  merge) become unnumbered interludes; steps 1–6 are the six passes.
- `build_tutorial.py` generates every pass card from the example's `graph.json` and splices each
  pass's code from `passes/<name>.frag.glsl`; `tutorial_body.html` holds prose + `{{CARD:x}}` /
  `{{CODE:x}}` markers. A test builds the tutorial and asserts every pass in the example has a
  card and a code block, and no `{{` survives.
- Text fixes: "Before you start" says 512×512 via the new preset (W-A), rename `main` → `paint`
  (no "Add"), defaults stated once; history/CORRECTION callouts move to the 068 spec.
- Script mentions (#20): every sentence in the tutorial and the Help panel that mentions the
  document script says that `Ctrl+R` (or Script → open) creates it; the build test asserts each
  occurrence of "script" in `tutorial_body.html` is within a sentence carrying `Ctrl+R` or is
  inside a code block.
- JFA run count (#6): the example's `jfa.frag.glsl` derives its offset from the canvas —
  `offset = exp2(ceil(log2(max(u_resolution.x, u_resolution.y))) - 1.0 - u_pass_iteration)` and
  returns its input unchanged when `offset < 1.0` — so any run count >= `ceil(log2(max side))`
  is correct at every preset; the shipped `iterations` becomes 11 (correct through 2048, the
  largest W-A preset) and the "panel warns" comment goes. The tutorial's card shows runs 11 and
  the interlude explains the formula; the "resize changes the answer" paragraph becomes "resize
  past 2048 and add a run". The engine stays out of it (it cannot know a JFA from a cascade).
- Files: `ai_docs/features/068_radiance_cascades/{build_tutorial.py, tutorial_body.html,
  tutorial.html}`, the example's `jfa.frag.glsl`, `tests/test_tutorial_build.py`.

---

## Order (three forced edges, the rest a solo-maintainer scheduling choice)

Forced: W-F's re-vendor before W-E's keymap setting and audit; W-D's naming before W-H; W-A's
preset before W-H. Everything else could run in parallel (W-B, W-E, W-G, W-F share only
`app.py` / `ui.py` / `commands.py`); serial waves are chosen so one review pair and one commit
series own each change and the three shared files never merge.

1. **W-C** (crash + commit + activate + hotkeys + first render) — small, unblocks the walk itself.
2. **W-A** (canvas funnel + presets + viewer) — the tutorial's first step depends on it.
3. **W-B** (prose diet + gate) — independent; early so every later UI wave is written under D1.
4. **W-F** (re-vendor, editor chrome, lib issues filed) — the re-vendor brings the standard
   keymap and the keymap docs W-E's audit and keymap setting read from. The status line itself
   depends on nothing in W-E (verified: it reads the editor's focus stop, which stays).
5. **W-E** (regions out, audit, keymap setting) — after the re-vendor. Independent of W-F's chrome
   work; the two may run in parallel once the `.so` is vendored.
6. **W-G** (scripting + mouse + clear) — before W-H so the paint step can be written against it.
7. **W-D** (naming + default wiring + strip tune) — before W-H (the tutorial's wiring lines change
   under D9).
8. **W-H** (tutorial) — last. It generates cards and code from the example files W-D rewrites
   (naming) and W-G's script contract, and the resolution preset from W-A. It is the verification
   of everything above: the maintainer walks it again.

Each wave: pre-impl review 1 (correctness & design vs this spec), post-impl review 2 (code
correctness; spec fidelity). W-E and W-G are high-blast-radius: add a spec-fidelity auditor.

## Files touched

Per workstream above. Cross-cutting: `conventions.md` (scripting entry, region entry, the
naming rule, the word-budget rule pointer), `roadmap.md` (069 row + banner; 019 → superseded;
065 → done with the D12 supersession noted), `.claude/skills/imgui-ui/SKILL.md` (§ 7.5 commit
rule; § 2 already carries D1), `todo.md` untouched (frozen).

## Manual verification (the maintainer, in the app)

- W-A: pick 512×512 from the combo; open a pass gear: size shows 512×512 at 100%; add a pass at
  50%: its tile is 256×256 on the next frame; a fully transparent output shows the checkerboard.
- W-B: no popup scrolls; every `(?)` reads in one glance; the engine-uniform block fits at the
  narrowest panel width.
- W-C: rename a pass by typing and clicking elsewhere — it renames, no crash; add a pass — its tab
  opens, the viewer shows it, the gear is open; Alt+P opens the active pass's gear; reopen the app
  on a six-pass document — every tile shows a picture within a second.
- W-D: the strip shows name + thumbnail, nothing truncated; declare `uniform sampler2D u_df;` in
  a new pass — it is wired.
- W-E: no yellow outline anywhere; arrows do nothing outside the editor; Ctrl+N no longer makes a
  document while the standard keymap is focused (whatever the audit moves it to works
  everywhere); switching keymap in Settings switches every open tab.
- W-F: the status line and gutter look like nvim with `number relativenumber`; an error line is
  readable; edit above an error — markers vanish until save; visual `p` replaces (after the
  re-vendor carrying the fix).
- W-G: a script returning `{"paint": {...}, "u_time_scale": 0.5}` drives both; LMB paints,
  hover does not; the stroke is continuous; Clear canvas empties it. The first launch logs one
  salvage line per `projects/dev` document whose `stopped_uniforms` predates the pair shape —
  expected once, gone after the hand-edit.
- W-H: the maintainer walks the tutorial end-to-end; every step produces its picture.

## Open questions — closed by the maintainer ("pick the defaults")

1. **#20** ("Add scripts/script.py to the document") — the sentence was not found in the tutorial,
   Help, or any tooltip. Default: W-H adds a build-time check that the tutorial and Help contain
   no "add … script" instruction and that every mention of the script says `Ctrl+R` creates it;
   if the maintainer meets the sentence again, its location goes to the W-H commit.
2. **#36** — assumed to be the branch rule (passes beside/after the output never draw). W-C's
   first-render-every-pass covers it; if black tiles persist with the LAST pass as output, that is
   a new finding with a repro.
3. **W-D scope** — graph view is feature 070 (see Out of scope). D12 reduced to the strip tune.
4. **Chords** — provisional defaults, the W-E audit may move them: `ADD_PASS` = Alt+A;
   `OPEN_PASS_SETTINGS` = Alt+P; `NEW_DOCUMENT` moves Ctrl+N → Ctrl+Shift+N;
   `TOGGLE_DOCUMENT_PLAY` moves Ctrl+Space → F5; `RESET_FEEDBACK` ("Clear canvas") = F6.
   F-keys and Alt are untouched by both keymaps (`docs/vim_coverage.md`, `docs/standard_keymap.md`).

## Review history

**Closing condition.** Three rounds, five reviewer runs, all on opus (judgement was the
deliverable: spec against code). Round 3 was narrow and still returned four real defects inside
the two bullets round 2 had forced a redesign of; the sentences that closed them (W-C's
target-only skip, W-D's compiled-passes fixpoint and three-state combo) have had no reviewer of
their own. That is deliberate: each wave's pre-implementation review re-reads its own bullets
against the code before a line is written (`dev_flow.md` step 4), and W-C / W-D are the first
two such reviews to run. Findings in the ledger are the external anchor throughout (the
maintainer's verbatim words); every citation in this spec was opened by a reviewer at least once.

**Round 3, design reviewer, narrow** (`reviews/round3_design.md`): closure PASS (24/24),
implementability PARTIAL (4 defects, 2 preferences), gate PARTIAL (ran as a script: all four
target strings flagged; census was wrong). Folded: the once-per-frame skip is scoped to target
renders only (unscoped it blanked every thumbnail); the planner's sampler names come from
compiled passes with W-C's sweep as the fixpoint, no eager compile; the gear gets a three-state
combo and `PassGraph.without_input`; `drawn_frame` is defined only where target renders are
issued; the `text_colored` census corrected to 31 by AST. The two preferences (naming of
`drawn_frame`; whether the allowlist lives in the test or a data file) left as written.

**Round 2, design reviewer** (`reviews/round2_design.md`): W-C PARTIAL, W-G PARTIAL, W-D
FAIL, W-B gate FAIL, D9 wiring PARTIAL, five fresh items. All accepted. W-D's default wiring is
REDESIGNED as a render-time resolution rule with `""` as explicit none (the stored-edge shape
could not survive an explicit un-wire or reach the hot-reload seam); W-B's gate walks f-strings
and the label helpers with a pinned allowlist; W-C's `render(target)` states all four decisions
and the once-per-frame skip; W-G names the four missed consumers, corrects the copilot
motivation to `_pass_views`, fixes the example count (seven, five with the field), picks
`StoppedKey` as a model, and adds the orphan's pass to the error; W-A's resolution control gains
state and reads the document. Media-bound samplers are excluded from auto-wiring.

**Round 2, coverage reviewer** (`reviews/round2_coverage.md`): round-1 closure PASS (22/22 by
quoted text), coverage PARTIAL (#19's "tune" half), citations PASS (24/24). Folded: the strip tune
now re-measures the card and says why the thumb size stays; D3 states the no-dict-value invariant;
`ScriptEngine.reload` named; the prose gate's literal-only reach stated; the two `conventions.md`
line cites that this feature itself would rot are now entry names.

**Round 1, design reviewer** (`reviews/round1_design.md`): conventions PARTIAL (3), blast radius
PARTIAL (11 misses), feasibility PARTIAL, order PARTIAL. All accepted and folded: `Document.render`
grows `target` (W-C); four `no_nav_inputs` sites not three, the `nav_flatten` machinery, 067 and
the code-editor convention entry (W-E); the Known-quirks vendoring entry and a precise issue (3)
(W-F); `tick_export`, `_scriptable_uniforms_for`, `api_doc.py`, `prompt_context.py`, `backend.py`'s
driven gate, `widgets/uniform.py`, the persisted `stopped_uniforms` in five `document.json`s, the
test budget (W-G); exact-token rename with the `u_light_*` / `u_glow_*` prefix trap (W-D); the
order heading now says which edges are forced. Ctrl+Y noted as editor-owned, not an app collision.

**Round 1, coverage reviewer** (`reviews/round1_coverage.md`): coverage PARTIAL (34/37),
citations PASS (35/35 opened and true), consistency PARTIAL. Accepted and folded: #6's run-count
half (now derived in-shader, W-H); #22's builtin-uniform half (decided: none, W-G); #20's check
restated positively (W-H); the Bloom rename mapping corrected (W-D); the order's false W-F→W-E
dependency removed (W-F now precedes W-E; the two may run in parallel after the re-vendor);
W-H's dependency on W-D stated as the naming half. Nothing rejected.
