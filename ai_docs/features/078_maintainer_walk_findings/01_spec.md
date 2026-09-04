# 078 — the maintainer's walk findings: ledger, decisions, tasks

`00_raw_findings.md` is the maintainer's verbatim stream from walking the app on 2026-09-04.
This file refines it: one entry per finding with what the code actually does (each verified
against the tree at `20bcfaf` before filing), the decisions the stream already made, the one
design question the stream asked for (editor intelligence), and the tasks. Nothing is fixed in
this session; the fixes are the sessions after plan-lock.

Size: **high-blast-radius** — one new module family (editor intelligence), two library seams
(completion kinds, identifier classes), a formatter dependency, and seven host-only fixes.
Each workstream is its own commit series with its own pre/post review per `dev_flow.md`.

Two repos, two sessions, as in 073: the editor-library items went to the editor session
(`editor-c8`, message of this session) as a bugs compilation with the harness ask; it pings
back with a commit sha and this session re-vendors per `conventions.md ## Known quirks`. A
second message follows plan-lock, for the two seams W-A needs.

---

## Goal

After this feature: the code panel knows the document it is editing — completion and `K`
answer from the pass's own declarations, the engine's builtins, the script's symbols and the
uniforms the script returns, and the samplers another pass could feed; each candidate is
colored by what it is, and the same classes color the identifiers in the text; a formatter
sits on a chord for both languages; an error line under the cursor lights its row in the strip
and `F8` pops the message; Reset lives on the Document tab; a new pass is made in the settings
modal; the sampler-source list reads as groups; a settings field you were sent to is visibly
marked; a copilot edit lands at the right indent; and nothing shows a checkerboard while the
copilot thinks. The editor library keeps vim's cursor after `dd`, `u` and `<Right>`.

## Findings (verified ledger)

Classes as in 073: **DEFECT** / **UX** / **ENGINE**; `editor lib` marks what the editor
session owns. Quotes are the maintainer's words from `00_raw_findings.md`.

| # | Class | Reported | What the code does (verified) |
|---|---|---|---|
| 1 | UX (design) | "the default Docstring is mostly noise, we need to implement a proper "shift+k" on python symbols, including our local stuff like Ctx" | The stub a new script gets (`scripting/engine.py::script_stub_for`) carries `_INIT_DOC` and a twelve-line `_UPDATE_DOC` as method docstrings, which is the noise. `K` on a script tab does what it does on a shader tab: `tabs/code.py::_consume_lookup_request` calls `completion.symbol_doc(word, app.shader_lib_index.functions)` with no look at `tab.kind`, so over `Ctx`, `self`, `math` or a local name it finds nothing (or a GLSL builtin of the same spelling). The doc source for the API already exists: `scripting/api_doc.py` renders `_CTX_GLOSS` (every `EngineContext` field with its meaning) and the `Vec*` / `Array` / `Text` members for the copilot prompt, and `behavior.py::_build_globals` is the exact namespace a script resolves against (`ScriptBehavior`, `Ctx`, `MouseState`, `Vec2/3/4`, `Array`, `Text`, plus real builtins and imports). |
| 2 | UX (design) | "We should do autocomplete for the local symbols as well ... fully support python auto-complete. ... maybe there is something like pre-compiled python symbols table? I don't want to integrate the full-blown language server" | The script tab's only provider is `completion.py`'s `python` entry: `keyword.kwlist`, nothing else. No buffer symbols, no `ctx.` members, no `math.`, no `self.` attributes. The library's own buffer-word source is suppressed at session creation (host is the sole source). Options in `## Open questions` Q1. |
| 3 | UX (design) | "I want to integrate auto-formatting option (ctrl+shift+I) for glsl and python" | No format command exists (`commands.py` has none). `ruff` is a dev dependency only (`pyproject.toml`), so the shipped app has no formatter; no GLSL formatter is on the box. The bundle is a source tree installed per the install guide, so a runtime dependency ships by `uv add`. Options in Q2. |
| 4 | UX (design) | "we can parse the ast and find all the return statements and extract the uniforms we return and suggest them as autocomplete in glsl" | Nothing reads the script for completion. The engine knows the returned keys only at RUN time (`ScriptEngine` after `update` ran); a static read of `update`'s `return {...}` (bare keys and pass blocks, the stub's two forms) is an `ast` walk over the buffer text, and a literal value's shape gives the GLSL type (`0.5` → `float`, `Vec3(...)` → `vec3`, `Array([...])` → `float[]`) exactly as `engine.py::_stub_kind` maps the other way. |
| 5 | UX (design) | "red-ish for special words and builtin types (continue, if, vec3, etc), blue-ish for builtin uniforms (u_time, u_aspect ...), green for script uniforms" | The completion popup is drawn by the library from bare strings (`Editor.complete_push(word)`); the `Completion` item has no kind or color, and the popup's rows read one slot (`POPUP_TEXT` / `POPUP_SELECTED`). A per-item color is a library seam (`editor lib`). |
| 6 | DEFECT | "under the u_aspect uniform I start to type "uniform vec4 u_" and it suggests me "u_time" ... why not u_aspect?" / "I'm typing "vec3 color = u_" and it suggests me "u_time" again. Only this fucking u_time" | The `u_` candidates come from `CompletionContext.pass_uniforms = tuple(edited.uniform_values)` (`tabs/code.py::_completion_context`), and `Pass.uniform_values` is the GL program's ACTIVE uniform set: `seed_uniform_values` fills it from `get_active_uniforms()` and `render()` writes the engine-driven ones into it (`core.py:540`). A declared uniform the shader body never reads is optimised out by the GLSL compiler and is not active, so `u_aspect` never appears; `u_time` is read, so it does. The source is wrong in kind: it lags compilation, drops unused declarations, and never sees the script's names. The `builtin uniforms` provider does not fire on `uniform vec4 u_` because its pattern is `\buniform\s+\w*$` (the type word breaks it), which is why the second form suggested nothing better either. Fix is the intelligence index of W-A (the buffer's declarations are the source; the GL active set never is). |
| 7 | ENGINE, editor lib | "I perform "dd" the line disappears, good. But the cursor appears at the beginning of the next line, but not at the same position as it was before." | Reproduced on the vendored `f738744` through `Editor.feed`: `j$dd` from column 17 lands at column 0. The editor's own oracle is nvim with `'startofline'` off (its `vim_coverage.md` says so for `G`/`gg`), so the column should be kept and clamped. Sent to the editor session. |
| 8 | ENGINE, editor lib + host check | "I undo this line deletion "u" and the line appears. But the current line (gray-ish strip) stays at the second line" | The strip is host-drawn: `tabs/code.py::_apply_markers` adds a cursor-line marker from `editor.get_current_cursor_position().line` each frame, fingerprinted so a change redraws. Headlessly the library's cursor LINE after `u` is the restored line in every case tried (last line, middle line, undo from another line), so the library is not the cause of a stale strip. What is not settled: the column vim puts the cursor at after `u` (asked of the editor session), and whether the strip is stale at the display for a reason the headless path cannot show (W-C's display check). |
| 9 | ENGINE, editor lib | "when I hold "->" or "l" and the cursor is at the end of the line, I don't jump to the next line" | Reproduced for insert-mode `<Right>`: on `ab\ncd`, `i` then `ed_key(RIGHT)` three times gives (0,1), (0,2), (1,0); nvim's default `'whichwrap'` is `b,s`, so it stops at column 2. Normal-mode `l`, `ed_key(CHAR,'l')` and `ed_key(RIGHT)` at `$` all stay put headlessly, so the `l` half is not reproduced; the editor session checks the repeat path. Sent. |
| 10 | ENGINE, editor lib | "Why don't they catch this simple case? ... ask it to implement more robust end-to-end tests" | The editor repo's `fuzz_walk.py` has two oracles, crash and undo-to-root, neither looking at the cursor; `e2e_cases.tsv` is 137 hand-written cases. A wrong cursor after an ordinary operator is invisible to all of it. Asked for: a differential fuzzer against the nvim oracle comparing text, cursor and mode after every step, mutation-tested against #7 and #9. Sent. |
| 11 | UX | "when the cursor is hovering the errored line it would be useful to highlight this specific error on the errors panel ... if we press "f8" - jump to the next error, we also show the popup explicitly; otherwise ... we only highlight the error at the bottom" | Today the strip's rows (`_draw_error_strip`) are plain selectables; nothing ties the caret's line to a row. The other direction exists for uniforms only: a hovered uniform-panel row marks its declaration line (`widgets/uniform.py:75` sets `app.editor_hover_line`). `F8` is `CommandId.JUMP_NEXT_ERROR` and moves the caret; the error marker carries `tooltip=message` for the mouse. The popup primitive exists: `anchored_note` at `_note_anchor`, what `K` draws. Decision D3. |
| 12 | UX | "Reset button ... Let's better put it inside the Document tab, at the top, where we keep Document name and Canvas ... this is about a document, not about documentS" | 073 D5 put `Reset` on the documents grid's `New document` / `Render all` row (`widgets/document_grid.py:62-70`). The Document tab's top row is `tabs/document.py::draw`: `Document name` input at `SIZE.NAME_INPUT_W`, then `Canvas` W x H fields at `combo_offset`, inside a `begin_disabled(app.copilot_turn_active)`. Decision D4, superseding 073 D5. |
| 13 | UX | "let's just open the pass settings modal, just add another button "Create" along side with the "Close" button ... we already open the settings modal when we confirm the name, so obviously we can combine these steps" | `widgets/pass_list.py::_draw_add_input` draws an `InlineInput` stretched to the panel width with an `x` ghost button, commits on Enter / deactivate, calls `session.add_pass`, then `app.pick_pass` and `app.open_pass_settings(name)`. The modal (`popups/pass_settings.py`) edits a LIVE pass: the name row renames through `session.rename_pass`, target and runs write through `session.set_pass_target` / `set_pass_iterations` on every change, and the only button is `Close`. `session.add_pass` writes `PASS_STUB` to disk, compiles, and adds a default `PassEntry`. Decision D5; the modal's create mode is Q4. |
| 14 | UX (design) | "provide autocomplete suggestion for a possible sampler2D which could be wired from another passes" | The name rule is in `pass_graph.py::_auto_source`: `u_<pass>` auto-wires to the pass of that name, `u_prev` to the consumer's own previous frame (069 D9). So for a pass being edited, every other pass name and `prev` yield one declaration candidate `uniform sampler2D u_<name>;` that wires itself on save. No provider offers it. Part of W-A. |
| 15 | UX | "the options looks confusing, this is just a plain list ... maybe we colorize them / group them?" | `widgets/uniform.py:290`: `imgui.combo` over `[f"auto ({auto or 'none'})", "none", *passes, "file..."]`. An `imgui.combo` cannot color or group rows; `begin_combo` + `selectable` per row can, with section captions. Decision D6. |
| 16 | UX (design) | "colorize the builtin uniforms names in glsl (blue-ish?) and maybe distinguish them from the passes sampler uniforms ... when I change the uniform source to a texture, the color should change back to the normal uniform" | Colouring is the library lexer's (`Language.GLSL`): identifiers are one class, and the host only sets the palette for `SYNTAX_1..7` (`theme.py:566-572`). `theme.py:192 SYN_UNIFORM` exists and maps to no slot — a dead token. No seam lets the host classify an identifier. The classification the maintainer describes is host knowledge: engine uniforms are `ENGINE_UNIFORM_DOCS`, a sampler's source is `document.sampler_source(pass, name)` (a pass, or a texture / none). A library seam is needed (`editor lib`): Q3. |
| 17 | UX | "the settings window opens, but I need several seconds to find the input field visually. Let's highlight it when we get transfered from the copilot widget. (the same for the telegram key and for the youtube key probably)" | `app.open_settings(focus=SettingsField.COPILOT_KEY)` (from `widgets/copilot_chat.py:157`; exporters pass their `config_field` the same way) sets `app.settings_focus`; `popups/settings.py` force-opens the owning tree node and `ui_primitives.focus_field` gives the NEXT item keyboard focus plus `set_scroll_here_y`, one-shot. A password field with keyboard focus shows a caret and nothing else, which is the "several seconds". The shared primitive is the one place to fix (class, not instance). Decision D7. |
| 18 | DEFECT (offset verified; color not reproduced) | "when the copilot is changing my shader for some reason the code color completely fucks up... the copilot inserts its first line shifted to the right. When I shift it back, the code color restores" | Offset: the GLSL edit matcher is token-based — `copilot/glsl_lex.py::token_match` returns the span from the first token to the last, so on `    vec3 color = vec3(1.0);` the span starts at column 4, after the indent (measured: `(18, 41)`), and `edit_match.splice` puts `new_str` there verbatim; a `new_str` that carries its own leading indent (what a model types when it copies the block) is doubled on its first line and only there. Colour: not reproduced headlessly — after `set_text` with a tab- or space-shifted first line the lexer classes are right (`ed_class_at` after `layout`), and `set_text` bumps the revision so the redraw gate repaints. It needs a display reproduction with a real copilot edit (W-I). |
| 19 | DEFECT (not reproduced) | "when sending a message to the copilot ... the main rendering canvas render the checkerboard instead of the black alpha channel. Something gets reset when agent is thinking or what?" | The viewer always draws the quiet checker (`ui.py:647 _draw_canvas_backdrop` with `checker_texture`) under `image_with_bg` of the output texture, so a checker showing means the output texture went transparent, not that a backdrop was switched (the loud checker is only the `COLOR_ALPHA` channel view). The live loop keeps rendering `tick_documents` during a turn (`ui.py:294`, no `in_flight` gate); the copilot's own render (`copilot/backend.py:157 document.render(u_time=t, target=target)`) draws into the same pass canvases at its sample time. Which of these leaves alpha at 0 needs the display (W-J). |

## Locked decisions (from the stream — constraints, not options)

- **D1. Editor intelligence is designed first and partitioned; no stubs.** Maintainer: "first
  design a good architecture which we can easily maintain and generalize ... You either
  implement a proper solution or don't slap these half-way done things". W-A ships as a whole:
  index, completion, lookup, colors. A partial provider (the `u_time` shape) is out. The GL
  active-uniform set is never a completion source; the buffer's declarations are.
- **D2. Candidate and identifier colors by kind:** language words and types one color
  (red-ish), engine builtins another (blue-ish), script-returned uniforms a third (green-ish),
  pass samplers distinguished from plain uniforms. Exact tokens tuned at the display. A
  sampler's class follows its LIVE source: wired to a pass → pass sampler; a file or none →
  plain uniform.
- **D3. Errors: the caret on an error line highlights that row in the strip; `F8` also pops
  the message at the caret; a caret alone pops nothing.** Maintainer's words in #11.
- **D4. Reset moves to the Document tab's top row (with `Document name` / `Canvas`).**
  Supersedes 073 D5. `F6` and the tooltip stay. Maintainer: "this is about a document, not
  about documentS".
- **D5. `add pass` opens the pass settings modal with a `Create` button beside `Close`; the
  inline name field goes.** Maintainer's words in #13.
- **D6. The sampler-source list is grouped and colored, not a flat list.** Groups: the two
  rules (`auto (…)`, `none`), the passes (colored as pass samplers, D2's hue), `file...`.
  Look tuned at the display.
- **D7. A settings field reached through `open_settings(focus=…)` is visibly marked for a
  moment, at the shared primitive** (every consumer: copilot key, Telegram, YouTube).
- **D8. The editor-library items go to the editor session as a batch with the harness ask**,
  never approximated in the host (standing rule since 069). Sent this session.
- **D9. Formatting is a registered command on `Ctrl+Shift+I` for shader, lib and script
  tabs.** The backend per language is Q2.
- **Standing:** fixes at the class; no compat code; `make gates` green before "done"; UI work
  through `ui_primitives.py` / `theme.py`.

## The design: editor intelligence (W-A)

The one architecture the stream asks for. Everything the editor "understands" about the
document lives in one place and is read three ways.

```
shaderbox/intel/
  symbols.py     Symbol(name, kind, signature, doc, insert_text)   SymbolKind enum
  glsl.py        GlslIndex.build(text, engine_docs, lib_index, script_returns, wirable, sampler_sources)
  script.py      ScriptIndex.build(text, api_gloss, exec_globals) + returned_uniforms(text)
  document.py    DocumentIntel: per (document, tab) index, rebuilt when its fingerprint changes
```

- **`SymbolKind`** is the whole vocabulary, one enum: `GLSL_KEYWORD`, `GLSL_TYPE`,
  `GLSL_BUILTIN`, `LIB_FUNCTION`, `ENGINE_UNIFORM`, `PASS_UNIFORM` (declared in the buffer,
  plain), `PASS_SAMPLER` (declared and wired to a pass), `WIRABLE_SAMPLER` (not declared, could
  be), `SCRIPT_UNIFORM` (returned by the script, declared or not), `BUFFER_SYMBOL` (a function
  or local the buffer defines), `PY_KEYWORD`, `PY_BUILTIN`, `PY_API` (`Ctx`, `Vec3`, …),
  `PY_MEMBER` (`ctx.t`, `math.sin`, `self.x`), `PY_LOCAL`. Colour is a function of kind in
  `theme.py` (one table), so the popup, the text and the sampler-source list agree by
  construction (D2), and the checker-narrowing test enumerates the enum: every kind has a
  color and a provider.
- **Sources, each a pure function of its inputs, GL-free:** the buffer text (declarations,
  functions, locals — a small lexer-level scan, not a parser); `ENGINE_UNIFORM_DOCS`;
  `shader_lib_index`; `returned_uniforms(script_text)` (an `ast` walk of `update`'s `return`,
  bare keys and pass blocks, literal shapes → GLSL types); the pass names for
  `WIRABLE_SAMPLER` via `_auto_source`'s rule; `document.sampler_source` for
  `PASS_SAMPLER` vs `PASS_UNIFORM`.
- **Rebuild trigger:** a fingerprint of cheap inputs — the editor revision, the script file's
  revision, the graph's pass set and wiring, each sampler's source kind. Computed per frame,
  the index rebuilt only on change, so a source switch recolors next frame (D2's robustness).
- **Three readers:** `completion.py` keeps its provider table but every provider yields
  `Symbol`s from the index (context predicates stay: `uniform\s+(\w+\s+)?$` offers the
  declarations, an identifier prefix offers by kind, `ctx.` / `self.` / `math.` offer members);
  `K` is `index.lookup(word)`; the color feed is `index.classes()` → the library seam (Q3).
- **Contracts pinned by tests:** the `u_aspect` case (declared, unused → offered); the
  `uniform vec4 u_` case (a type between → still the declarations provider); a script
  returning `{"u_speed": 0.5}` → `uniform float u_speed;` offered in the shader with kind
  `SCRIPT_UNIFORM`; a second pass `paint` → `uniform sampler2D u_paint;` offered as
  `WIRABLE_SAMPLER`; a sampler re-sourced to a file flips `PASS_SAMPLER` → `PASS_UNIFORM`;
  `K` over `Ctx` on the script tab yields the gloss; the enum-domain test.

Python's own completion (Q1) plugs in as `ScriptIndex`'s member source; the module boundary
does not move whichever option wins.

## Open questions for the user

**Q1. Python completion depth** (finding 2). Three shapes:

- *(a) Own index, no dependency.* `ast` over the buffer for defs / classes / `self.*`
  assignments / names in scope; members for the known static types only: `ctx.` from
  `_CTX_GLOSS`, `self.` from the class body, `math.` from `dir(math)` at runtime, `Vec3(...)`
  members from `api_doc`. `K` reads the gloss or `__doc__`. Cheap, in-frame, deterministic.
  Blind to anything it does not know statically (`x = ctx.mouse` then `x.` completes nothing).
- *(b) `jedi` in-process.* A library, not a server: real completion and docstrings for
  arbitrary Python including `ctx: Ctx` through the stub's real import line
  (`from shaderbox.scripting import ScriptBehavior, Ctx`, installed package). One dependency
  (`jedi` + `parso`, pure Python). First call warms typeshed (about a second); later calls tens
  of ms — run on a worker thread, result latched next frame, the popup is host-pushed anyway.
  Our gloss still overlays `ctx` fields (jedi shows the field's type, the gloss says what it
  means). Not in the venv today.
- *(c) A language server (pyright / pylsp).* Out by the maintainer's own words.

Recommendation: **(b) for members and `K`, (a)'s `ast` walk for the script → GLSL uniform
extraction** (jedi does not do that). (a) alone is what "half-done" looks like once someone
writes `x = ctx.mouse`.

**Q2. Formatter backends** (finding 3; the bundle installs a source tree, so a runtime
dependency ships by `uv add`):

- Python: *`ruff format` via subprocess* (the repo's own formatter, a binary wheel, ~20 ms a
  call) or *`black` in-process* (`black.format_str`, pure API, no subprocess). Either is fine;
  `ruff` keeps one formatter in the project.
- GLSL: *`clang-format` from its pip wheel via subprocess* with a C style (handles GLSL well;
  the only mature option), or *an own minimal formatter* (indent + brace + spacing normaliser —
  a rabbit hole that would be finding 6's shape again).

Recommendation: **`ruff` for Python, the `clang-format` wheel for GLSL**, both behind one
`Formatter` protocol keyed by tab kind, applied as ONE undo step through `set_text` with the
caret kept by line. Chord `Ctrl+Shift+I`: registry chords are struck only when the editor
consumes them (`hotkeys.py:68-76`), and the library binds no Ctrl+Shift chord, so it is free.

**Q3. The two library seams** (findings 5 and 16), to go to the editor session after lock:

- Completion item color: a `kind` (or a color slot index) on `Completion`, drawn per row.
- Identifier classes: *(i) a host-fed word table* `ed_set_word_class(word, slot)` /
  `ed_clear_word_classes` — the lexer colors an identifier found in the table with that slot;
  the host re-feeds when the index changes. Simple, robust, no per-layout callback.
  *(ii) per-line span overlays* from the host each revision. *(iii) a classify callback* per
  identifier per layout. Recommendation: **(i)**, with new slots `IDENT_CLASS_1..4`.

**Q4. The modal's create mode** (finding 13, D5):

- *(a) Pending draft.* `add pass` opens the modal on a draft (`name` empty and focused, target
  and runs on a draft `PassEntry`); `Create` calls `session.add_pass` then applies the draft's
  target / runs; `Close` discards. Exactly the words; the modal's controls write to the draft
  instead of the session while in create mode.
- *(b) Create-then-edit.* `add pass` creates `pass_N` at once and opens the modal on it with
  the name selected; `Close` keeps it. Cheaper; leaves a pass behind on a change of mind.

Recommendation: **(a)**.

**Q5. Stub docstrings** (finding 1): drop `_INIT_DOC` / `_UPDATE_DOC` from the stub entirely
(the `K` gloss and the Help panel carry the semantics), or keep one line each? Recommendation:
drop them; the commented uniform blocks stay.

## Out of scope (each with a trigger)

- **Hover-triggered popups (mouse dwell over a symbol).** `K` stays the trigger. Trigger: the
  maintainer asks after using `K` on Python.
- **A full Python type inference of our own.** Q1(b) covers it or nothing does. Trigger: a
  completion miss the maintainer names after (b) ships.
- **Formatting on save.** The chord only. Trigger: the maintainer asks for it.
- **Docs for GLSL builtins under `K` beyond `glsl_docs.py`.** Already served (`BUILTINS`).
- **The `l` half of finding 9** if the editor session cannot reproduce it. Trigger: the
  maintainer sees it again and names the mode.
- **Windows `libeditor.dll`.** Still owed from 067. Trigger: next `/ship` on a Windows host.

## Workstreams (tasks)

### W-A — Editor intelligence (#1 #2 #4 #5 #6 #14 #16, D1 D2; Q1 Q3)

1. `shaderbox/intel/`: `symbols.py`, `glsl.py`, `script.py`, `document.py` as designed; the
   color table in `theme.py` keyed by `SymbolKind`; the enum-domain test.
2. `completion.py` re-based on the index (providers yield `Symbol`s; the `uniform` predicate
   admits a type word); `_completion_context` loses `pass_uniforms`; the `u_aspect` and
   `u_time` tests.
3. `K` on both tab kinds through `index.lookup`; the `Ctx` gloss test.
4. Python members per Q1; the script → GLSL `returned_uniforms` walk with its shape → type
   table pinned against `_stub_kind`'s.
5. After the editor session lands the seams: bind `Completion.kind` and the word-class table
   in `editor/ffi.py`, feed both from the index; delete the dead `SYN_UNIFORM`.
6. The stub docstrings per Q5.

### W-B — Formatter (#3, D9; Q2)

`shaderbox/formatting.py` (a `Formatter` per tab kind), `CommandId.FORMAT_BUFFER` on
`Ctrl+Shift+I`, one undo step, caret kept by line, an error toast when the backend rejects the
text (a syntax error formats nothing). Tests: a known-ugly GLSL and Python sample round-trip to
a pinned result; a broken sample leaves the buffer untouched.

### W-C — Editor library re-vendor (#7 #8 #9 #10, D8)

When `editor-c8` pings: rebuild from the sha, copy the vendored set, `abi_probe.py`,
`vim_coverage.md` refreshed; the strip-after-`u` check at the display (#8's host half). Wave
file `10_wave_c_editor.md` records what came back.

### W-D — Code panel errors (#11, D3)

`_draw_error_strip` highlights the row whose line is the caret's (`fade(ACCENT_PRIMARY)` ground
as the hover mark uses); `JUMP_NEXT_ERROR` sets `app.editor_error_note = (line, message)` drawn
by `anchored_note` at the caret through `_note_anchor`, dismissed as `K`'s popup is. Tests: the
row index for a caret line; the note is set by the command and cleared by a key.

### W-E — Reset on the Document tab (#12, D4)

The ghost button leaves `document_grid.py` and joins `tabs/document.py::draw`'s first row
after the Canvas fields, same handler / tooltip / chord. 073's spec gets a superseded note on
D5.

### W-F — Add pass through the modal (#13, D5; Q4)

`InlineInput` for passes deleted; `App.open_pass_create()` opens the modal in create mode;
`_draw_body` draws `Create` beside `Close` in that mode; `pick_pass` + close on success, the
name error as a toast. Tests: create mode commits nothing until `Create`; a duplicate name
stays in the modal.

### W-G — Sampler-source list (#15, D6)

`widgets/uniform.py`'s combo becomes `begin_combo` with three groups and per-row colors from
the D2 table; the current source shown with its color on the closed combo. Test: the row
order and the group captions.

### W-H — Settings focus mark (#17, D7)

`focus_field` gains the mark: the focused item's frame drawn with `ACCENT_PRIMARY` border for
~1.5 s after the jump (`app.settings_focus_marked_until`), fading; consumers unchanged. Test:
the mark expires.

### W-I — Copilot edit indent (#18)

`edit_match.splice` (GLSL path): when the span starts after the line's leading whitespace and
`new_str`'s first line carries its own indent, drop that indent from the first line (the
matched line's indent is kept). Test: the doubled-indent case. Then the color half at the
display with a real copilot edit; if it survives the indent fix, root-cause it there.

### W-J — Checkerboard while the copilot thinks (#19)

Reproduce at the display with a black-opaque shader; instrument which render leaves the output
alpha at 0 (the live tick or `backend.py:157`); fix at the cause. Test after the cause is
known.

### W-K — Sanitize

`/sanitize`: `conventions.md` gains the intel module's boundary (a GL-free leaf read by the
code panel) and the formatter dependency note; roadmap row + banner; cold-context check.

## Order

1. **W-E, W-H, W-I (indent half), W-D** — small, host-only, visible on the next launch.
2. **W-F, W-G** — host-only, medium.
3. **W-A steps 1-4** — the design, after plan-lock on Q1 and Q3; then the second message to
   the editor session (Q3 seams).
4. **W-B** — after Q2.
5. **W-C** when the editor session pings; **W-A step 5** on the seams it lands.
6. **W-J, W-I color half** — at the display.
7. **W-K.**

## Files touched

`shaderbox/intel/*` (new), `shaderbox/completion.py`, `shaderbox/formatting.py` (new),
`shaderbox/tabs/code.py`, `shaderbox/tabs/document.py`, `shaderbox/widgets/document_grid.py`,
`shaderbox/widgets/pass_list.py`, `shaderbox/widgets/uniform.py`, `shaderbox/popups/pass_settings.py`,
`shaderbox/popups/settings.py`, `shaderbox/ui_primitives.py`, `shaderbox/theme.py`,
`shaderbox/commands.py`, `shaderbox/app.py`, `shaderbox/scripting/engine.py`,
`shaderbox/copilot/edit_match.py`, `shaderbox/editor/ffi.py` + `shaderbox/resources/editor/*`
(W-C, W-A step 5), `pyproject.toml` (W-B, Q1b), tests beside each.

## Manual verification (the maintainer, in the app)

1. Declare `uniform float u_aspect;` unused, type `u_` on a new line: `u_aspect` is offered,
   colored as a plain uniform; type `uniform vec4 u_`: the builtin declarations are offered.
2. In the script return `{"u_speed": 0.5}`; in the shader type `u_sp`: `uniform float
   u_speed;` is offered in the script-uniform color; accept it and the identifier in the
   text carries that color.
3. With a second pass `paint`, type `uniform sampler2D u_` in the other pass: `u_paint` and
   `u_prev` are offered as wirable samplers; after save the panel row shows `auto (paint)`
   and the name in the text is colored as a pass sampler; switch its source to a file and it
   turns plain within a frame.
4. On the script tab, `ctx.` lists the context fields; `K` over `Ctx` and over `ctx.t` shows
   the gloss; a local `def` and a `self.x` complete.
5. `Ctrl+Shift+I` on an unindented shader and on a script formats each in one undo step.
6. The caret on an error line lights that row in the strip; `F8` moves and pops the message;
   any key closes it.
7. Reset sits on the Document tab's first row; `F6` still resets.
8. `add pass` opens the modal with an empty focused name; `Create` makes the pass and picks
   it; `Close` makes nothing.
9. The sampler-source list shows three groups, passes colored.
10. The copilot widget's key button opens Settings with the key field visibly marked.
11. A copilot edit lands at the block's own indent; the colors survive it.
12. Sending a copilot message never shows the checker under an opaque render.
13. Editor: `dd` keeps the column; `<Right>` in insert mode stops at the line end; the cursor
    line strip follows `u`.
