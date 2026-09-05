# 079 — the fifth walk: findings into a plan

The maintainer ran 078's display checks on 2026-09-05 and filed fourteen findings
(`00_raw_findings.md`, verbatim). This spec has one entry per finding with what the code
actually does (each verified against `7f38d0b` before filing), the decisions the stream already
made, the open questions for plan-lock, and the tasks. One deliverable was made in the filing
session because the maintainer asked to see it before anything changes: the button-style
candidates page, `button_styles.html` beside this file (open it in a browser).

Size: **medium** — host-only polish over the 078 surfaces plus one editor-library ask (the
output variable's class). Each workstream is its own commit with pre/post review per
`dev_flow.md` where it changes a design; the small ones are direct.

---

## Goal

After this feature: a `K` or `F8` note appears cleanly, shows a uniform's value and a sampler's
picture; candidates sort by color then name; the output variable reads orange; the script API's
docs read as documentation; typing `1.` offers nothing; a script key the shader does not declare
yet is a normal state, not an error; the error strip sits under the status bar; the source list
is `none` / the passes / `file...`; the stub is already formatted; the copilot's dim never shows
the checker; and the low-emphasis buttons and the Document tab's top row have the shape the
maintainer picks from the candidates page.

## Findings (verified ledger)

Classes as in 073: **DEFECT** / **UX** / **ENGINE**; `editor lib` marks what the editor session
owns. Quotes are the maintainer's words from `00_raw_findings.md`.

| # | Class | Reported | What the code does (verified) |
|---|---|---|---|
| 1 | DEFECT | "when triggering help popup (shift+K) and then triggering on some other symbol, the first frame blinks with the first popup ... The same flickering happens with error popups (f8) .. looks like some general issue." | Both notes are `ui_primitives.anchored_note`: an imgui window with a FIXED id (`##lookup`) and `always_auto_resize`. imgui sizes an auto-resize window from the previous frame's content, so on the frame the text changes the new note is drawn inside the old note's size (clipped or padded) and settles a frame later; that is the blink. A fresh id per note would instead HIDE the window for its first frame (imgui measures an appearing auto-resize window before showing it), a different blink. Same primitive, same cause for `F8`. Fix: measure the note (`calc_text_size` of the title and of the body at the wrap width) and `set_next_window_size` explicitly; no auto-resize, no first-frame lag. Any image the note carries (finding 3) is measured the same way. |
| 2 | UX | "sort suggestions by their type (color) -- when possible, otherwise alphabetically" | `GlslIndex.words` is in build order (the buffer's declared uniforms, undeclared engine/script/wirable names, buffer declarations, plain words, library, language) and `offer` keeps provider order; nothing sorts. A rank per `SymbolKind` beside `kind_color` / `kind_slot` in `theme.py`, then the name, is the whole change; the declarations site sorts the same way. Order is Q1. |
| 3 | UX | "show the current uniform value inside the hint popup (make sure that long array-like uniforms correctly stripped and don't overflow). Also, if it's possible and easy, let's visualize a texture inside sampler2D help snippets ... probably it is not possible without modifying the editor api." | It is possible without the editor: the note is host imgui (`anchored_note`), not the library's popup. The value is `Pass.uniform_values[name]` through `util.format_auto_value`, which the hover tooltip in `tabs/code.py::draw` already shows for a live uniform; the texture is `document.input_texture(consumer, source)` for a pass-sourced sampler (what the panel row's thumbnail draws, `widgets/uniform.py::_draw_pass_source`) or the media's `.texture` for a file. `anchored_note` takes a title and a body only; it grows a value line (ellipsized at `SIZE.NOTE_W` through `_ellipsize`) and an optional image. The candidate-doc note beside the popup is the same primitive and gets the same. |
| 4 | UX | "color output shader variable name into orange-ish color ... Keep this generalizable across different themes though." | Measured on the vendored lexer: `gl_FragColor` lexes as BUILTIN (class 6, the builtin green); a user's `out vec4 fragColor;` name is plain (0). A host class fills only identifiers the lexer left plain (`011_symbol_classes`), so `gl_FragColor` cannot be recolored from the host today: an `editor lib` ask, small — either the host table wins over a builtin identifier or `gl_*` variables leave the builtin list. The user's `out` names are the host's: `intel/glsl.py` scans `out <type> <name>;`, a kind `OUTPUT_VARIABLE` takes slot 9 (spare) and a theme token from the orange palette entry (`_P["orange_b"]`, the gruvbox orange every accent swap keeps; "generalizable" is met by the token, not a hex). |
| 5 | UX | "I still don't like the docstrings in python code ... all these ";" in one line, full mess ... ctx.dt - the help popup is empty. ctx.frame -- is empty. ctx.mouse -- aweful help string ... The same is for "Ctx" itself, one short sentence ... Check all other possible gaps" | Verified: `api_doc._CTX_GLOSS` has `dt: ""` and `frame: ""` (so `K` shows an empty note through `ctx_field_gloss`), `mouse` is a parenthesized one-liner joining six facts with `;`, `t` is the one word "seconds"; `api_symbol_doc("Ctx")` is "the per-frame context `update` receives: t, dt, frame, mouse"; `_VALUE_SHAPE_GLOSS` reads "float\|int -> scalar" style arrows; the stub's `_UPDATE_DOC` (078 D14) joins the ctx fields with `;` in one paragraph. One table feeds `K`, completion detail, the stub and the copilot prompt (`tests/test_script_api_doc.py` pins the join), so the fix is the table. The style is D3 below, researched: PEP 257 plus the Google docstring layout. |
| 6 | DEFECT | "I type a number and a dot: "1." and popup appears with python's words: "and, if, in, is"" | Reproduced: `python_completions("x = 1.\n", 0, 6)` gives `and, if, in, is, not, or`. `completion.PYTHON_SITE` (`(?:\.\w*\|\w+)$`) treats any dot as a member site and asks jedi, which after a float literal's dot offers keywords. A member site is a dot after a NAME or a closing bracket, never after a number token; `x.` where `x` is a float still completes (`as_integer_ratio, ...`, measured). |
| 7 | UX | "When a script returns a uniform which is not used in the shader, we have an error ... u_tmp: pass 'main' has no active uniform 'u_tmp' (orphan key) ... I usually first write something in the script and then use it in the shader later" | `scripting/engine.py::_binding_reject` returns the orphan-key message for a key no active uniform matches; it lands in the script's soft errors and the strip shows it red on the script tab and on the pass's tab. With 078, the shader side already treats such a key as knowledge (it offers `uniform float u_tmp;`), so the state is a normal step of authoring. Decision D5: not an error, no row. The sampler/block rejection and the "pass does not compile" case stay errors. |
| 8 | DEFECT | "this errors pane at the bottom of the editor hovers vim's status bar, align this stuff properly." | `tabs/code.py::draw` sizes the interaction `invisible_button` one cell row SHORTER than the editor image (so a click on the status band does not place the caret behind it), and the strip child is drawn at the cursor the button leaves — one row above the image's bottom, over the library's status band. Fix: place the cursor at the image's bottom edge (`editor_pos.y + size_px[1]`) before the strip. |
| 9 | UX | "Still looks messy ... none and pass has the same color and style, although "pass" is not even an option ... Do we even need this "auto (main)" ... Just keep none, pass0, pass1...., file... ; keep none gray-ish, keep passes green-ish and file... just white" | `widgets/uniform.py` builds three groups for `grouped_combo` (078 W-G): the rules `auto (…)` / `none`, a `pass` caption over the passes, `file...`; the caption is drawn in the same dim color as `none`. Decision D6: the list is `none` / the passes / `file...`, no auto row, no captions, no gaps; the VALUE model keeps `AutoSource` as the default (069 D9's name rule still wires a fresh sampler), the closed combo shows what the value RESOLVES to (the pass in its color, or `none`), an explicit pick writes `PassSource` / `NoSource`. Supersedes 072's row shape (its "auto (paint)" row). `grouped_combo` stays for a list that has groups. |
| 10 | DEFECT | "autoformatting on the initial script state does something. (adds empty line somewhere). The script should be formattes in its initial state." | Reproduced: `format_python(script_stub_for({"main": []}))` differs by one line — ruff wants two blank lines between the import block and `class Behavior`; `script_stub_for` emits one. Fix in the stub, gated: a test asserts the stub is a fixed point of `format_python`. |
| 11 | UX | "put presets button as a little button on the same line as the "Canvas" title ... "Canvas [presets]" ... make this button in a form of a chip ... And the "Reset" button -- align it to the right border of the Document panel ... (make it red?)" | `tabs/document.py::draw` draws `Document name` and `Canvas` captions on one row, then the name input, the two canvas fields, the `presets` combo (`SIZE.CANVAS_PRESETS_W`, `no_arrow_button`) and, after `end_disabled`, the Reset ghost button, all on the second row. Decision D7 after the pick: the presets control becomes a chip on the caption row beside `Canvas`; Reset sits right-aligned on the caption row in the danger tier; the second row is the two inputs only. |
| 12 | UX | "we have this button style which looks like a plain text (e.g "add pass", "open", "Reset" and so on), I don't like this style ... Maybe we should remove this button style at all and use the "Copilot" style button (button with a border) ... Can you craft html with various candidates ... I will pick some" | `ui_primitives.ghost_button` is transparent with the secondary text color and no border; `toggle_button` (the Copilot bar button) is transparent with a 1 px `BORDER` frame; `chip_button` and `pill_button` exist for the tags. The tier system is `imgui-ui §1`. Deliverable made: `button_styles.html` — the Documents panel and the Document tab's top row emulated with the theme's own hex values, six low-emphasis candidates (the current ghost; outline; outline on a dim fill; filled subtle; chip; underlined text) and three Reset variants. The pick is Q2; the generalization is W-J. |
| 13 | DEFECT | "I still see checkerboard... Maybe it has something to do with this dimming that we add to the whole application when copilot is thinking" | The maintainer's guess is right. `ui.py` wraps `_draw_app_panel` in `imgui.begin_disabled(app.copilot_turn_active)` to freeze the controls during a turn, and `_draw_app_panel` is where `_draw_document_image` draws the viewer. imgui's disabled scope multiplies the style alpha by `DisabledAlpha` for every item inside, `image_with_bg` included, so the render is composited at reduced alpha and the quiet checker under it shows through. Headless probes never saw it because they never rendered a turn's frame. Fix: the viewer draws at full alpha — end the disabled scope before it, or push `StyleVar.alpha` 1.0 around the image and backdrop; the controls stay frozen. |
| 14 | UX | (the button question, merged with 11 and 12) | See 11 and 12. |

## Locked decisions (from the stream — constraints, not options)

- **D1. The notes never blink.** The note is sized from measured text, not auto-resized; the
  same fix serves `K`, `F8` and the candidate note.
- **D2. Candidates sort by kind, then by name.** The kind order is a table in `theme.py` next
  to the color and slot tables and walked by the enum test. The order, top to bottom: the
  buffer's own names, engine uniforms, script uniforms, pass and wirable samplers, library
  functions, language keywords and types; alphabetical inside each tier.
- **D3. Docstrings and glosses follow one style: PEP 257 with the Google layout.** A summary
  line in the imperative or as a noun phrase ending in a period; a blank line; a description of
  full sentences; `Args:` / `Returns:` sections where a callable has them; one fact per line,
  never `;`-joined lists. Every `ctx` field, every injected API name and every value shape has
  a complete gloss; `K` never opens empty on an API symbol. The stub's docstrings describe the
  method and point at `K` for the fields.
- **D4. `1.` offers nothing.** A Python member site is a dot after a name or a closing bracket.
- **D5. A script key with no declared uniform is not an error.** No row, no red; the shader
  side already offers the declaration. Sampler/block keys and a pass that does not compile stay
  errors.
- **D6. The source list is `none` / the passes / `file...`.** No auto row, no caption, no gaps;
  `none` dim, passes in the pass-sampler color, `file...` plain; the closed combo shows the
  resolved pass. Supersedes 072's row shape.
- **D7. The Document tab's first row is `Document name` and `Canvas [presets chip]` with
  Reset right-aligned in the danger tier; the second row is the inputs.** In D12's tiers.
- **D8. The viewer draws at full alpha during a copilot turn**; the controls stay frozen.
- **D9. The error strip sits below the editor image**, never over the status band.
- **D10. The stub is a fixed point of the formatter**, gated.
- **D11. The output variable is orange**: a kind with its own slot and a palette token; the
  library lets a host class win for `gl_FragColor` (editor-session ask).
- **D12. The button system is at most four tiers, applied everywhere.** The low-emphasis tier
  is option B of `button_styles.html`: transparent fill, a 1 px `BORDER` frame, secondary text
  (the shape the Copilot bar button has today), hover and active one palette step darker. The
  tier set is fixed at plan-lock of W-J and stays within four — the candidates: primary
  (filled accent, the one call-to-action of a section), standard (B), danger (B with the error
  color, for a destructive verb), toggle (B, filled accent when on). Every button in the app is
  one of them; a site that needs a fifth is a design question, not a new primitive. Chips and
  pills that are not buttons (tags, the presets chip) are their own primitive and stay outside
  the count.
- **D13. A sampler's picture in the `K` note is drawn at the note's content width**, aspect
  preserved.
- **Standing:** fixes at the class; no compat code; `make gates` green before "done"; UI work
  through `ui_primitives.py` / `theme.py`.

## Questions — all answered; the feature is plan-locked (2026-09-05)

**Q1 — answered: the relevance tiers, alphabetical inside each.** Top to bottom: the buffer's
own names (declared uniforms and buffer symbols, plain), engine uniforms (blue), script
uniforms (green), pass and wirable samplers (aqua), library functions (green), language
keywords and types (red). The document's own vocabulary comes before the language's. The
alternative — strictly by color — was rejected because it puts `if` and `vec3` above the
document's own names. Recorded as D2.

**Q2 — answered: option B, then a sweep.** Maintainer: "Let's do something like
"B" option. But let's refine our buttons logic across the whole application, we need to make
the stuff consistent. We need to keep the number of button type tight (3-4max) and apply the
styling consistently across the whole app." Recorded as D12; the task is W-J.

**Q3 — answered: the note's width.** The sampler's picture in the `K` note is drawn at the
note's content width (aspect preserved), not at `SIZE.THUMB_SM`, so the picture reads.
Recorded as D13.

## Out of scope (each with a trigger)

- **A smarter completion ranking** (frequency, recency). Maintainer: "Maybe later we'll
  introduce some more clever sorting". Trigger: he asks after living with kind-then-name.
- **A texture in the library's completion popup rows.** The popup is the library's; the note
  beside it is where the picture goes. Trigger: he asks for it in the popup itself.
- **Theme variants beyond the accent swap.** "Generalizable across themes" is met by tokens.
  Trigger: a second theme lands.

## Workstreams (tasks)

### W-A — Notes: sized, valued, pictured (#1 #3, D1 D13)

`anchored_note` measures its content (`calc_text_size` with the wrap width) and sets the window
size; gains `value: str` (ellipsized) and `texture: moderngl.Texture | None` drawn at the
chosen size. `tabs/code.py` passes the uniform's value (`format_auto_value`) and, for a
pass-sourced or file-bound sampler, its texture, for both `K` and the candidate note. The
picture is drawn at the note's content width with the aspect preserved (D13). Tests: the
measured size is stable across two frames with different content (no auto-resize lag); the
value line is ellipsized for a long array.

### W-B — Completion: sort, and the `1.` site (#2 #6, D2 D4)

`theme.kind_rank`; `offer` sorts its result by `(rank, name)`; `completion.python_site(before)`
replaces the regex: the token before a dot must be a name or a closing bracket. Tests: the
order over a mixed set; `1.`, `x.`, `f().`, `a[0].` sites.

### W-C — The output variable (#4, D11; editor lib)

Editor-session ask: a host class wins for a builtin-classed identifier (or `gl_*` variables leave
the builtin list); re-vendor. Host: `intel/glsl.py::output_declarations`, `SymbolKind.OUTPUT_VARIABLE`,
slot 9, `COLOR.SYN_OUTPUT` from the orange palette entry, `classes()` includes it, `gl_FragColor`
and `gl_FragData` always. Tests: the scan; the class in the text once the library allows it.

### W-D — Docstrings and glosses (#5, D3)

`api_doc.py`: `_CTX_GLOSS` complete (t, dt, frame, mouse with its fields one per line and the
export freeze as its own sentence), `_VALUE_SHAPE_GLOSS` as sentences, `api_symbol_doc` for
every injected name in the same shape; `MouseState` and `EngineContext` docstrings in the
Google layout; the stub's `_UPDATE_DOC` / `_INIT_DOC` in the same layout with `Returns:`. The
copilot prompt block re-rendered from the same table (`tests/test_script_api_doc.py` keeps the
join). Tests: no gloss is empty; no gloss line carries a `;` list; every API name's doc has a
summary line.

### W-E — The orphan key is not an error (#7, D5)

`_binding_reject` returns None for a key with no active uniform; the key is skipped silently
(as engine-owned keys already are). Tests: the soft errors carry no orphan row; a sampler key
still does.

### W-F — The strip under the status band (#8, D9)

One `set_cursor_screen_pos` before the strip. Test: the strip's top equals the image's bottom in
the frame rig.

### W-G — The source list (#9, D6)

`widgets/uniform.py`: rows `none` (dim) / passes (`kind_color(PASS_SAMPLER)`) / `file...`
(plain), one flat group; the closed label resolves `AutoSource` through `wired_pass`. 072's spec
gets a superseded note. Tests: the row list; the closed label for auto/none/pass/file values.

### W-H — The stub is formatted (#10, D10)

`script_stub_for` emits the second blank line; `tests/test_formatting.py` asserts
`format_python(stub).text == stub` for an empty and a populated uniform set.

### W-I — The viewer at full alpha (#13, D8)

`ui.py`: the disabled scope covers the controls, not the viewer (end it before
`_draw_document_image`, or push alpha 1.0 around image and backdrop). Test: a frame with
`copilot_turn_active` draws the image with alpha 1 (read the draw command's color).

### W-J — The button system, swept (#11 #12, D7 D12)

1. **Inventory.** Every button site in `shaderbox/` (`ghost_button`, `primary_button`,
   `button`, `toggle_button`, `danger_button`, `pill_button`, `chip_button`, raw
   `imgui.button` / `small_button` / `selectable`-as-button), each with its role: the section's
   one call-to-action, an ordinary verb, a destructive verb, an on/off state, a tag. A table in
   the wave file, one row per site.
2. **The tiers.** `ui_primitives.py` keeps at most four button primitives, B as the standard
   look; `primary_button`, `danger_button` and `toggle_button` restyled to the same frame so the
   family reads as one; `button` (imgui's filled grey) and any primitive the inventory leaves
   without a role are deleted. Hover / active states from the theme, one step darker.
3. **The sweep.** Every site takes its tier from the table; raw imgui buttons go through a
   primitive; labels re-read against `imgui-ui §2`'s word budget while there.
4. **The Document tab's row** (D7): `Canvas` caption + the presets chip on the caption line,
   Reset right-aligned there in the danger tier, the inputs on the second row.
5. `imgui-ui §1`'s tier table rewritten to the four that exist; a test walks `shaderbox/` for
   a raw `imgui.button(` outside `ui_primitives.py` and fails on one.
6. Display check with the maintainer: the Documents panel, the Document tab, the pass strip,
   the settings and pass modals, the share tab, the lib picker, the copilot bar.

### W-K — Sanitize

`/sanitize`: the 072 supersession, conventions (the note primitive's sizing rule, the disabled
scope excludes the viewer), roadmap.

## Order

1. **W-I, W-F, W-H, W-E** — one-line causes, host-only, visible on the next launch.
2. **W-G, W-B** — small host-only.
3. **W-A, W-D** — the notes and the docs.
4. **W-C** — after the editor session answers the ask.
5. **W-J** — the inventory first, then the tiers, then the sweep.
6. **W-K.**

## Files touched

`shaderbox/ui_primitives.py`, `shaderbox/tabs/code.py`, `shaderbox/tabs/document.py`,
`shaderbox/widgets/uniform.py`, `shaderbox/ui.py`, `shaderbox/completion.py`,
`shaderbox/theme.py`, `shaderbox/intel/glsl.py`, `shaderbox/intel/index.py`,
`shaderbox/intel/symbols.py`, `shaderbox/scripting/api_doc.py`, `shaderbox/scripting/context.py`,
`shaderbox/scripting/engine.py`, `shaderbox/editor/ffi.py` + `shaderbox/resources/editor/*`
(W-C), tests beside each.

## Manual verification (the maintainer, in the app)

1. `K` on one symbol, then on another: the second note appears without a frame of the first;
   `F8` likewise. `K` over a float uniform shows its value; over a long array a cut line; over a
   pass-sourced sampler its picture.
2. Type `u_`: the buffer's names first, then blue, green, aqua, library, red; alphabetical
   within each.
3. `gl_FragColor` and a user `out` variable read orange.
4. `K` over `ctx.dt`, `ctx.frame`, `ctx.mouse`, `Ctx`, `Vec3`, `Array`: each a summary line and
   full sentences, no `;` lists; the stub's docstrings read the same way.
5. Type `1.` in the script: nothing opens; `x.` on a float still lists members.
6. Return `{"u_tmp": 1.0}` with no declaration: no error row; add a sampler key: still an error.
7. The error strip sits under the status band.
8. The source list is `none` / passes / `file...`; a fresh sampler's closed combo shows the pass
   it wires to.
9. `Ctrl+Shift+I` on a fresh script changes nothing.
10. Send a copilot message over an opaque render: the render stays opaque under the dim.
11. Option B on `add pass`, `open`, `Reset`, `Close`, `Cancel` and every other verb, one look
    across the panels, modals, the share tab, the lib picker and the copilot bar; the Document
    tab's first row is `Document name`, `Canvas [presets]`, Reset at the right border.
