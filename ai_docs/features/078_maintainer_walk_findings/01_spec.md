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
| 2 | UX (design) | "We should do autocomplete for the local symbols as well ... fully support python auto-complete. ... maybe there is something like pre-compiled python symbols table? I don't want to integrate the full-blown language server" | The script tab's only provider is `completion.py`'s `python` entry: `keyword.kwlist`, nothing else. No buffer symbols, no `ctx.` members, no `math.`, no `self.` attributes. The library's own buffer-word source is suppressed at session creation (host is the sole source). Resolved by D10. |
| 3 | UX (design) | "I want to integrate auto-formatting option (ctrl+shift+I) for glsl and python" | No format command exists (`commands.py` has none). `ruff` is a dev dependency only (`pyproject.toml`), so the shipped app has no formatter; no GLSL formatter is on the box. The bundle is a source tree installed per the install guide, so a runtime dependency ships by `uv add`. Resolved by D11. |
| 4 | UX (design) | "we can parse the ast and find all the return statements and extract the uniforms we return and suggest them as autocomplete in glsl" | Nothing reads the script for completion. The engine knows the returned keys only at RUN time (`ScriptEngine` after `update` ran); a static read of `update`'s `return {...}` (bare keys and pass blocks, the stub's two forms) is an `ast` walk over the buffer text, and a literal value's shape gives the GLSL type (`0.5` → `float`, `Vec3(...)` → `vec3`, `Array([...])` → `float[]`) exactly as `engine.py::_stub_kind` maps the other way. |
| 5 | UX (design) | "red-ish for special words and builtin types (continue, if, vec3, etc), blue-ish for builtin uniforms (u_time, u_aspect ...), green for script uniforms" | The completion popup is drawn by the library from bare strings (`Editor.complete_push(word)`); the `Completion` item has no kind or color, and the popup's rows read one slot (`POPUP_TEXT` / `POPUP_SELECTED`). A per-item color is a library seam (`editor lib`). |
| 6 | DEFECT | "under the u_aspect uniform I start to type "uniform vec4 u_" and it suggests me "u_time" ... why not u_aspect?" / "I'm typing "vec3 color = u_" and it suggests me "u_time" again. Only this fucking u_time" | The `u_` candidates come from `CompletionContext.pass_uniforms = tuple(edited.uniform_values)` (`tabs/code.py::_completion_context`), and `Pass.uniform_values` is the GL program's ACTIVE uniform set: `seed_uniform_values` fills it from `get_active_uniforms()` and `render()` writes the engine-driven ones into it (`core.py:540`). A declared uniform the shader body never reads is optimised out by the GLSL compiler and is not active, so `u_aspect` never appears; `u_time` is read, so it does. The source is wrong in kind: it lags compilation, drops unused declarations, and never sees the script's names. The `builtin uniforms` provider does not fire on `uniform vec4 u_` because its pattern is `\buniform\s+\w*$` (the type word breaks it), which is why the second form suggested nothing better either. Fix is the intelligence index of W-A (the buffer's declarations are the source; the GL active set never is). |
| 7 | ENGINE, editor lib | "I perform "dd" the line disappears, good. But the cursor appears at the beginning of the next line, but not at the same position as it was before." | Reproduced on the vendored `f738744` through `Editor.feed`: `j$dd` from column 17 lands at column 0. The editor's own oracle is nvim with `'startofline'` off (its `vim_coverage.md` says so for `G`/`gg`), so the column should be kept and clamped. Sent to the editor session. |
| 8 | ENGINE, editor lib + host check | "I undo this line deletion "u" and the line appears. But the current line (gray-ish strip) stays at the second line" | The strip is host-drawn: `tabs/code.py::_apply_markers` adds a cursor-line marker from `editor.get_current_cursor_position().line` each frame, fingerprinted so a change redraws. Headlessly the library's cursor LINE after `u` is the restored line in every case tried (last line, middle line, undo from another line), so the library is not the cause of a stale strip. What is not settled: the column vim puts the cursor at after `u` (asked of the editor session), and whether the strip is stale at the display for a reason the headless path cannot show (W-C's display check). | **Cause found (the formatter's display check, same class): the library MOVES a marker with the text it marks (`ffi/README.md`: "where the marked code is NOW"), and the host re-pushed its line markers only when the cursor line, errors or hover changed — `dd` then `u` leaves the cursor line equal and the band where the deleted line was; a whole-buffer replace drags it to the end. `_apply_markers`'s fingerprint now carries the buffer revision; `tests/test_marker_follow.py` pins both shapes.**
| 9 | ENGINE, editor lib | "when I hold "->" or "l" and the cursor is at the end of the line, I don't jump to the next line" | Reproduced for insert-mode `<Right>`: on `ab\ncd`, `i` then `ed_key(RIGHT)` three times gives (0,1), (0,2), (1,0); nvim's default `'whichwrap'` is `b,s`, so it stops at column 2. Normal-mode `l`, `ed_key(CHAR,'l')` and `ed_key(RIGHT)` at `$` all stay put headlessly, so the `l` half is not reproduced; the editor session checks the repeat path. Sent. |
| 10 | ENGINE, editor lib | "Why don't they catch this simple case? ... ask it to implement more robust end-to-end tests" | The editor repo's `fuzz_walk.py` has two oracles, crash and undo-to-root, neither looking at the cursor; `e2e_cases.tsv` is 137 hand-written cases. A wrong cursor after an ordinary operator is invisible to all of it. Asked for: a differential fuzzer against the nvim oracle comparing text, cursor and mode after every step, mutation-tested against #7 and #9. Sent. |
| 11 | UX | "when the cursor is hovering the errored line it would be useful to highlight this specific error on the errors panel ... if we press "f8" - jump to the next error, we also show the popup explicitly; otherwise ... we only highlight the error at the bottom" | Today the strip's rows (`_draw_error_strip`) are plain selectables; nothing ties the caret's line to a row. The other direction exists for uniforms only: a hovered uniform-panel row marks its declaration line (`widgets/uniform.py:75` sets `app.editor_hover_line`). `F8` is `CommandId.JUMP_NEXT_ERROR` and moves the caret; the error marker carries `tooltip=message` for the mouse. The popup primitive exists: `anchored_note` at `_note_anchor`, what `K` draws. Decision D3. |
| 12 | UX | "Reset button ... Let's better put it inside the Document tab, at the top, where we keep Document name and Canvas ... this is about a document, not about documentS" | 073 D5 put `Reset` on the documents grid's `New document` / `Render all` row (`widgets/document_grid.py:62-70`). The Document tab's top row is `tabs/document.py::draw`: `Document name` input at `SIZE.NAME_INPUT_W`, then `Canvas` W x H fields at `combo_offset`, inside a `begin_disabled(app.copilot_turn_active)`. Decision D4, superseding 073 D5. |
| 13 | UX | "let's just open the pass settings modal, just add another button "Create" along side with the "Close" button ... we already open the settings modal when we confirm the name, so obviously we can combine these steps" | `widgets/pass_list.py::_draw_add_input` draws an `InlineInput` stretched to the panel width with an `x` ghost button, commits on Enter / deactivate, calls `session.add_pass`, then `app.pick_pass` and `app.open_pass_settings(name)`. The modal (`popups/pass_settings.py`) edits a LIVE pass: the name row renames through `session.rename_pass`, target and runs write through `session.set_pass_target` / `set_pass_iterations` on every change, and the only button is `Close`. `session.add_pass` writes `PASS_STUB` to disk, compiles, and adds a default `PassEntry`. Decision D5; the modal's create mode is D13. |
| 14 | UX (design) | "provide autocomplete suggestion for a possible sampler2D which could be wired from another passes" | The name rule is in `pass_graph.py::_auto_source`: `u_<pass>` auto-wires to the pass of that name, `u_prev` to the consumer's own previous frame (069 D9). So for a pass being edited, every other pass name and `prev` yield one declaration candidate `uniform sampler2D u_<name>;` that wires itself on save. No provider offers it. Part of W-A. |
| 15 | UX | "the options looks confusing, this is just a plain list ... maybe we colorize them / group them?" | `widgets/uniform.py:290`: `imgui.combo` over `[f"auto ({auto or 'none'})", "none", *passes, "file..."]`. An `imgui.combo` cannot color or group rows; `begin_combo` + `selectable` per row can, with section captions. Decision D6. |
| 16 | UX (design) | "colorize the builtin uniforms names in glsl (blue-ish?) and maybe distinguish them from the passes sampler uniforms ... when I change the uniform source to a texture, the color should change back to the normal uniform" | Colouring is the library lexer's (`Language.GLSL`): identifiers are one class, and the host only sets the palette for `SYNTAX_1..7` (`theme.py:566-572`). `theme.py:192 SYN_UNIFORM` exists and maps to no slot — a dead token. No seam lets the host classify an identifier. The classification the maintainer describes is host knowledge: engine uniforms are `ENGINE_UNIFORM_DOCS`, a sampler's source is `document.sampler_source(pass, name)` (a pass, or a texture / none). A library seam is needed (`editor lib`): D12. |
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
  tabs.** The backends are D11.
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

Python's own completion (D10) plugs in as the script side's member source; the module boundary
does not move.

**Amended after the pre-implementation review (round 1, PARTIAL; `## Review history`):**

- The declarations provider's predicate is `\buniform\s+(?:\w+\s+)?\w*$` — it fires on
  `uniform `, `uniform u_`, `uniform vec4 u_` and `uniform sampler2D u_`, and not on
  `vec3 color = u_` (an identifier-prefix site) nor past the name. Pinned as that table.
- `PASS_SAMPLER` vs `PASS_UNIFORM` is decided from the buffer's `sampler2D` declarations
  joined to `pass_graph.wired_pass` with the sampler's VALUE (`Pass.uniform_values`, a dict; a
  declared sampler with no value yet is `AutoSource`) and the graph's pass names — never from
  `document.sampler_source` / `effective_wiring`, which read the compiled program's sampler set
  and carry finding 6's two defects (lag, dropped declarations) into the classification.
- The script text on a shader tab: the live script buffer when `app.editor_sessions` holds the
  script path (fingerprint: its revision), else the engine's cached copy
  (`ScriptEngine.cached_source`, fingerprint: its mtime). The pinned test exercises the live
  buffer.
- The script → GLSL type table, over the whole literal domain: `0.5` / `-2.0` → `float`;
  `0` / `3` → `int` (`bool` → `bool`); `Vec2/3/4(...)` → `vec2/3/4`; `Array([...])` →
  `float[N]` with N the literal list's length (`[x] * k` folded); `Array(<anything else>)`,
  `Text(...)`, a str, and any non-literal → a NAME-ONLY candidate, never a guessed declaration.
  This is not `_stub_kind`'s inverse: `_stub_kind` reads a compiled uniform's `gl_type` and
  lengths, the walk reads a literal, and they differ exactly there.
- `intel/`'s GLSL half (`symbols`, `glsl`, `script`, `index`, `document`) and `completion.py`
  are GL-free and gated in a fresh interpreter: `ENGINE_UNIFORM_TYPES` /
  `ENGINE_DRIVEN_UNIFORMS` moved to the leaf `shaderbox/engine_uniforms.py` (core re-imports
  them). The Python half (`python`, `worker`) reads the API glosses from
  `shaderbox.scripting.api_doc`, and that package's `__init__` re-exports the engine, so it is
  App-free and context-free but loads `moderngl` by import; the gate names the halves.
- jedi: constructed with `jedi.InterpreterEnvironment` (in-process, no child interpreter;
  measured faster cold and warm, and safe under concurrent use where the default corrupts);
  exactly ONE thread calls it — a worker the App owns, whose first queued job on a script tab's
  open is the warm-up — and the frame thread never does; every request carries the editor
  revision and cursor it was computed for and a result is dropped unless both are still
  current.
- Text colors and the library's slots (measured: the GLSL lexer emits classes 1 keyword —
  types included — 4 number, 6 builtin and functions; 7 is free, a host class fills only plain
  identifiers): engine uniform → slot 7 (`SYN_UNIFORM`, blue); script uniform and library
  function → slot 6 (builtin green, the same token); pass sampler → slot 8, requested from the
  editor session with a spare 9 (`Theme.syntax` widened on this measurement). `SYNTAX_7` stops
  mapping `SYN_TYPE`, which the GLSL lexer never emitted.

## Answered at plan-lock (2026-09-04)

The five questions the first draft carried, with the maintainer's answers folded into
decisions. The options considered stay in git history (`6ddfdaa`).

- **D10 (Q1). Python completion and `K` use `jedi` in-process**, one pure-Python runtime
  dependency, run on a worker thread with the result latched next frame; the `ctx` gloss from
  `api_doc.py` overlays jedi's answer for the API. The script -> GLSL uniform extraction is an
  own `ast` walk of `update`'s `return`. Maintainer: "let's use the library then".
- **D11 (Q2). Formatting is `ruff format --line-length 88` for Python and `clang-format` with
  `{BasedOnStyle: LLVM, IndentWidth: 4, TabWidth: 4, UseTab: Never}` for GLSL** -- the exact
  arguments the maintainer's nvim `conform` config passes, with no `.clang-format` file on the
  box, so the fallback style IS the setting. Both ship with ShaderBox as runtime dependencies
  (`ruff` and the `clang-format` pip wheel, each a binary wheel, called by subprocess); a
  formatter absent at runtime is a defect, never a silent no-op.
- **D12 (Q3). The two library seams are the editor session's call**, not the maintainer's: he
  asked what the question meant, and it is an implementation choice inside the vendored
  library. This session sends the need (a per-item color in the completion popup, a way for
  the host to say which identifiers are engine uniforms / pass samplers / script uniforms so
  the lexer colors them, following a source change within a frame) and the recommendation (a
  host-fed word-to-slot table); the editor session decides the API.
- **D13 (Q4). Pending draft.** `add pass` opens the modal on a draft; `Create` makes the pass
  with the draft's name, target and runs; `Close` makes nothing. Maintainer: "if we close it
  nothing gets created".
- **D14 (Q5). The stub docstrings stay and are rewritten dense**: what the method is, when it
  runs (`__init__` once at start and on reload; `update` every drawn frame), what `update`
  returns (bare key -> every pass declaring it; pass block -> that pass, wins over a bare key;
  a returned uniform plays, an omitted one stays manual), a line each, the way a good
  docstring reads. The commented uniform blocks stay.

## Out of scope (each with a trigger)

- **Hover-triggered popups (mouse dwell over a symbol).** `K` stays the trigger. Trigger: the
  maintainer asks after using `K` on Python.
- **A full Python type inference of our own.** D10's jedi covers it or nothing does. Trigger: a
  completion miss the maintainer names after (b) ships.
- **Formatting on save.** The chord only. Trigger: the maintainer asks for it.
- **Docs for GLSL builtins under `K` beyond `glsl_docs.py`.** Already served (`BUILTINS`).
- **The `l` half of finding 9** if the editor session cannot reproduce it. Trigger: the
  maintainer sees it again and names the mode.
- **Windows `libeditor.dll`.** Still owed from 067. Trigger: next `/ship` on a Windows host.

## Workstreams (tasks)

### W-A — Editor intelligence (#1 #2 #4 #5 #6 #14 #16, D1 D2; D10 D12)

1. `shaderbox/intel/`: `symbols.py`, `glsl.py`, `script.py`, `document.py` as designed; the
   color table in `theme.py` keyed by `SymbolKind`; the enum-domain test.
2. `completion.py` re-based on the index (providers yield `Symbol`s; the `uniform` predicate
   admits a type word); `_completion_context` loses `pass_uniforms`; the `u_aspect` and
   `u_time` tests.
3. `K` on both tab kinds through `index.lookup`; the `Ctx` gloss test.
4. Python members per D10; the script → GLSL `returned_uniforms` walk with its shape → type
   table pinned against `_stub_kind`'s.
5. After the editor session lands the seams: bind `Completion.kind` and the word-class table
   in `editor/ffi.py`, feed both from the index; delete the dead `SYN_UNIFORM`.
6. The stub docstrings per D14.

*(Landed, steps 1-6: `shaderbox/intel/` — `symbols` (the enum), `glsl` (the buffer read as
text), `script` (`update`'s returns by `ast`), `python` (jedi, in-process environment, the
engine's glosses and the injected API overlaid), `worker` (the one jedi thread; newest request
per kind wins; results stamped with path, revision and cursor), `index` (`build_glsl_index`
over a `GlslContext` of explicit, GL-free inputs), `document` (the per-path fingerprint cache).
`completion.py` re-based: providers yield `Symbol`s; the declarations site regex from the
review; declaration inserts shaped to the site (`float u_time;` after `uniform `, `u_time;`
after `uniform float `); the identifier site never offers a declaration; a Python member site
opens with an empty prefix. `tabs/code.py` builds the index on every frame's cheap fingerprint
(revision, script revision or mtime, pass set, sampler source kinds, lib index identity),
re-feeds `ed_set_word_class` on a rebuild, pushes candidates with `ed_complete_push_class`, and
answers `K` from the index or through the worker. `theme.kind_color` and `theme.kind_slot` are
the two tables; slots 7 and 8 are host classes (engine uniform, pass sampler), script
uniforms and library functions share the builtin green. `engine_uniforms.py` is the GL-free
home of the engine tables. Tests: the sources, the index, the completion policy (the
maintainer's two lines pinned), the panel through the real driver (colors in the text, `ctx.`
members and `K` through the worker), the GL-free gate in a fresh interpreter. After the
post-implementation round (below): every kind has a popup slot (keywords and types red, the
builtin green shared by builtins, library and script uniforms, engine blue, samplers aqua)
while the TEXT feed stays the four host classes; the API gloss wins for an injected name in
completion and under `K` whether the stub imports it or not, and an empty answer opens no
note; the declaration site after a typed type filters by that type and after a bare
`uniform` never offers a name it offers whole; the cache is per editor handle; the script's
text is read on a rebuild only; the lib index stamps with a counter; a dropped worker answer
releases its latch; the warm-up starts when the script tab opens. Display checks owed: items
1-4 of `## Manual verification`.)*

### W-B — Formatter (#3, D9, D11)

`shaderbox/formatting.py` (a `Formatter` per tab kind), `CommandId.FORMAT_BUFFER` on
`Ctrl+Shift+I`, one undo step, caret kept by line, an error toast when the backend rejects the
text (a syntax error formats nothing). Tests: a known-ugly GLSL and Python sample round-trip to
a pinned result; a broken sample leaves the buffer untouched.

*(Landed: `shaderbox/formatting.py`, a formatter per tab kind over subprocess — the venv's
`ruff format --line-length 88` for scripts, the `clang-format` wheel with the nvim fallback
style for shaders and lib files; `CommandId.FORMAT_BUFFER` on `Ctrl+Shift+I`, editor scope,
free in both vim modes (measured: the library consumes neither); `App.format_current_editor`
applies the result as ONE host edit (select all + `replace_selection`; `set_text` is not
undoable) with the caret kept on its line, and toasts the formatter's first complaint line on a
syntax error. `jedi`, `clang-format` and `ruff` are runtime dependencies now. Pinned by
`tests/test_formatting.py`, including the undo step.)*

### W-C — Editor library re-vendor (#7 #8 #9 #10, D8)

When `editor-c8` pings: rebuild from the sha, copy the vendored set, `abi_probe.py`,
`vim_coverage.md` refreshed; the strip-after-`u` check at the display (#8's host half). Wave
file `10_wave_c_editor.md` records what came back.

### W-D — Code panel errors (#11, D3)

`_draw_error_strip` highlights the row whose line is the caret's (`fade(ACCENT_PRIMARY)` ground
as the hover mark uses); `JUMP_NEXT_ERROR` sets `app.editor_error_note = (line, message)` drawn
by `anchored_note` at the caret through `_note_anchor`, dismissed as `K`'s popup is. Tests: the
row index for a caret line; the note is set by the command and cleared by a key.

*(Landed: the strip's caret row draws selected through `shader_errors.error_at_line`; `F8`
walks the strip's own list (`App.editor_errors`, what the panel last drew) and latches
`editor_error_note_pending`, which the panel turns into the `K` note at the caret on the
frame the jump lands. Display check owed: the row and the note.)*

### W-E — Reset on the Document tab (#12, D4)

The ghost button leaves `document_grid.py` and joins `tabs/document.py::draw`'s first row
after the Canvas fields, same handler / tooltip / chord. 073's spec gets a superseded note on
D5.

*(Landed: after the Canvas presets combo on the first row, outside the copilot-turn
`begin_disabled` as before; `document_grid.py` loses it and its command imports.)*

### W-F — Add pass through the modal (#13, D5, D13)

`InlineInput` for passes deleted; `App.open_pass_create()` opens the modal in create mode;
`_draw_body` draws `Create` beside `Close` in that mode; `pick_pass` + close on success, the
name error as a toast. Tests: create mode commits nothing until `Create`; a duplicate name
stays in the modal.

*(Landed: `PassDraft` (`ui_models.py`) held on `App.pass_draft`; `add pass` and `Alt+A` open
the modal in create mode; the target and runs controls draw over a `PassEntry` and return it,
so edit mode writes through the session's verbs and create mode into the draft; `Create` (or
Enter in the name field) calls `App.create_pass_from_draft`, `Cancel` and Escape reach
`close_pass_settings`, which drops the draft. The inline `InlineInput` for passes is gone.
Pinned by `tests/test_pass_draft.py`. Display check owed.)*

### W-G — Sampler-source list (#15, D6)

`widgets/uniform.py`'s combo becomes `begin_combo` with three groups and per-row colors from
the D2 table; the current source shown with its color on the closed combo. Test: the row
order and the group captions.

*(Landed: `ui_primitives.grouped_combo` — captioned groups of colored rows, the closed
control in the current row's color; the source row uses three groups: the two rules, the
passes in `COLOR.SYN_PASS_SAMPLER`, `file...`. `SYN_SCRIPT_UNIFORM` / `SYN_PASS_SAMPLER` are
the first two D2 tokens; W-A's kind table reads them. Display check owed: the look.)*

### W-H — Settings focus mark (#17, D7)

`focus_field` gains the mark: the focused item's frame drawn with `ACCENT_PRIMARY` border for
~1.5 s after the jump (`app.settings_focus_marked_until`), fading; consumers unchanged. Test:
the mark expires.

*(Landed: `FieldFocus(keyboard, mark)` replaces the bool; `focus_field` is a context manager
wrapping the one widget, drawing an accent outline after it at `mark` alpha; the popup
computes the alpha from `App.settings_mark`'s deadline, fading over `SETTINGS_MARK_S`; every
exporter's `draw_config_ui` takes the same object. Display check owed: the outline's look.)*

### W-I — Copilot edit indent (#18)

`edit_match.splice` (GLSL path): when the span starts after the line's leading whitespace and
`new_str`'s first line carries its own indent, drop that indent from the first line (the
matched line's indent is kept). Test: the doubled-indent case. Then the color half at the
display with a real copilot edit; if it survives the indent fix, root-cause it there.

*(Landed, indent half: `edit_match.splice` drops the replacement's first-line indent when
only whitespace precedes the span on its line, the rule `splice_script` already applied;
pinned by `test_splice_drops_the_replacement_indent_where_the_line_already_has_it`. The
color half waits for the display.)*

### W-J — Checkerboard while the copilot thinks (#19)

Reproduce at the display with a black-opaque shader; instrument which render leaves the output
alpha at 0 (the live tick or `backend.py:157`); fix at the cause. Test after the cause is
known.

### W-K — Sanitize

`/sanitize`: `conventions.md` gains the intel module's boundary (a GL-free leaf read by the
code panel) and the formatter dependency note; roadmap row + banner; cold-context check.

## Review history

- **W-A pre-implementation, round 1, one opus reviewer anchored on `00_raw_findings.md`, the
  design section and the landed editor seam (`011_symbol_classes`), every claim probed.**
  Verdict PARTIAL: the architecture and the finding-6 diagnosis right; seven findings, all
  accepted and folded into the design amendments above — the predicate that did not fire on
  the maintainer's line, `sampler_source` reading the GL program, the unnamed script source
  on a shader tab, the incomplete type table, the GL-free claim failing through
  `help_content`, jedi's one-thread rule and `InterpreterEnvironment`, the slot budget (three
  colors, one free slot; the widening was already requested). Unverified concerns it listed
  and this session's answer: the selected popup row keeps the selection color by the
  library's design (fine); the 50-candidate cap is met by ordering the document's own symbols
  before the language's; the full re-push per rebuild is ~0.1 ms by the editor's measurement;
  the glyph-table uniforms (`SBT_*`) are engine-driven and get no kind (they are never typed).

- **W-A post-implementation, round 1, two opus reviewers in parallel (code correctness;
  spec fidelity and architecture), anchored on `00_raw_findings.md` and the editor's own seam
  document, every claim probed.** Both PARTIAL. Code: a dropped jedi answer silenced its
  site (the latch never released); a recreated editor handle kept the cached index and got no
  class feed; `uniform u_` offered bare names beside the whole declarations, and a bare one
  landed a typeless `uniform u_time`; eleven kinds reached the popup as class 0; the lib
  index was stamped by an object id; one assertion was `... or True`. Fidelity: `K` over
  `Ctx` returned "statement Ctx" with no doc and the note opened anyway; the API gloss was
  skipped for names the stub imports; the panel's sampler bridge was untested and dropped an
  unknown value in the wrong direction; the script text was copied every frame for a stamp;
  the declaration regex existed twice; manual-verification item 1 contradicted the typed-type
  filter. All accepted and fixed in the wave; the accepted deviation from the design is that
  `uniform vec4 u_` offers nothing unless a vec4 is missing (item 1 rewritten to say so).
  The architecture note that `_glsl_index_for`'s App-to-context translation sits in a draw
  module is recorded, not acted on.

## Order

1. **W-E, W-H, W-I (indent half), W-D** — small, host-only, visible on the next launch.
2. **W-F, W-G** — host-only, medium.
3. **W-A steps 1-4** — the design, then the second message to
   the editor session (D12 seams).
4. **W-B**.
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
(W-C, W-A step 5), `pyproject.toml` (W-B, D10), tests beside each.

## Manual verification (the maintainer, in the app)

1. Declare `uniform float u_gain;` unused, type `u_` on a new line: `u_gain` is offered
   with the buffer's other names (`u_time`, `u_aspect`), never only `u_time`; `uniform `
   offers whole declarations (`float u_aspect;`, `sampler2D u_prev;`); `uniform float u_`
   offers `u_aspect;`; `uniform vec4 u_` offers nothing when no vec4 is missing (his own
   words: "Why do I even see the uniform suggestion here at all?").
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
