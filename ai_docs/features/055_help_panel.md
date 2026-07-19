# 055 — In-app Help panel (the shader contract, natively rendered)

## Goal

A stranger opening ShaderBox has no answer to "what is `#version 460 core` for, where does `u_time`
come from, what else does the engine give me, how do I call `SB_fbm` without an `#include`?" Today
that knowledge exists ONLY as inline comments in the UV-Mango starter — deletable, un-discoverable,
and silent about everything the starter doesn't happen to use.

Ship a **Help modal** (`F1` / a `Help` menu-bar item) that explains ShaderBox's thin GLSL contract
IN THE APP, natively rendered — never an "open a markdown file" hand-off. Left column = section
list; right pane = prose + GLSL snippets; each snippet has **Insert at caret** so the panel is a
TOOL, not a reading surface.

The engine-uniform section is GENERATED from `ENGINE_DRIVEN_UNIFORMS` and the shortcuts section from
`COMMAND_SPECS`, so the help is structurally incapable of drifting from the code it documents.

## Out of scope

- **Project/nodes/export/copilot orientation.** Only the shader contract + shortcuts ship now.
  Trigger: the maintainer (or a user report) says a stranger is lost on something outside the shader
  file — then add a section; the section list is data, so it costs one entry.
- **Full markdown** (headings, lists, links, images) in the renderer. Sections carry their own title;
  body prose uses the EXISTING `markdown_text` vocabulary (`**bold**`, `` `code` ``, ``` fences).
  Trigger: a section genuinely needs a nested list or a heading INSIDE its body.
- **Searching help content.** The section list is short (5-6 entries). Trigger: >10 sections.
- **Folding the floating cheatsheet overlay into Help.** REJECTED by research, not deferred: the
  overlay (`widgets/cheatsheet.py`) draws on the FOREGROUND draw list and live-filters to the chords
  valid RIGHT NOW (scope-gated) — a different artifact from a static browsable list. Help renders its
  own Shortcuts section from the same `COMMAND_SPECS` registry (unfiltered, all categories); the
  overlay keeps its own `show_cheatsheet` toggle + Settings checkbox. One source, two presentations.
- **A `docs/` site or repo-level user docs.** Trigger: a SECOND documentation page appears (project
  model, publishing) — that's when a structure earns its place. One page needs no structure.
- **Migration / back-compat** of any kind (per `conventions.md`). N/A.

## Design decisions (lock-in)

1. **Content is DATA, not a markdown blob** — `shaderbox/help_content.py`: no imgui, no `App`, so it
   is unit-testable without a GL CONTEXT (it does import `core` for `ENGINE_DRIVEN_UNIFORMS`, which
   transitively pulls moderngl/glfw at import time — that costs collection time, not a context;
   `tests/test_cross_project_tools.py` already imports the same constant this way). Do NOT "fix"
   this by re-listing the three uniform names: `conventions.md` pins `ENGINE_DRIVEN_UNIFORMS` as
   their ONE home. The claim is "no imgui, no App", never "no transitive deps". A frozen `HelpSection` dataclass:
   `key: str`, `title: str`, `body: str` (markdown-lite prose), `snippet: str` (GLSL, may be empty).
   `help_sections() -> list[HelpSection]` builds the list, so the two generated sections (D2, D3)
   are computed at call time rather than frozen at import. A section with an empty `snippet` simply
   draws no code block and no Insert button.

2. **The engine-uniform section is GENERATED from `ENGINE_DRIVEN_UNIFORMS`.** A hand-typed list rots
   the moment a builtin is added (the parallel-lists drift smell — `conventions.md`). The section
   renders one declaration line per engine uniform, from a single `{name: (glsl_type, doc)}` table in
   `help_content.py` whose KEYS are asserted to cover `ENGINE_DRIVEN_UNIFORMS - TABLE_UNIFORMS`
   (the glyph tables are engine machinery, never user-facing — excluded by name, not by hoping).
   A NEW engine uniform with no table entry FAILS a test (D6), not silently ships an incomplete doc.

3. **The shortcuts section is GENERATED from `COMMAND_SPECS`**, grouped by `CATEGORY_ORDER`, as
   two-column `label` / `chord_to_str(spec.default_chord)` rows (whitespace-separated columns like
   the cheatsheet widget — NOT hand-rolled dot leaders, which exist nowhere in `ui_primitives`). Uses the DEFAULT chord, not
   `effective_bindings` — `help_content` is a leaf with no `App`; the live-rebind view is the
   overlay's job (which already reads `effective_bindings`). Sections carrying rebound chords is the
   revisit trigger.

4. **The modal mirrors the lib picker** (`popups/lib_picker/` — the proven "browse reference content"
   shape): `popups/help.py`, a free `draw_help(app)` whose body is `_draw_body(app) -> bool`
   returning `keep_open` (the `/imgui-ui` §7.3 contract — never an inverted `should_close`),
   `modal_window(...)`, left `begin_child` section list (`selectable` rows — nav-reachable per
   `/imgui-ui` §8; no `no_nav_inputs` needed, that rule governs the main window's regions and a modal
   owns the frame), right `begin_child` content pane, bottom action row. Esc closes via an unguarded
   `is_key_pressed(escape)` → `keep_open = False` (Help hosts no inline input, so it needs none of
   the picker's input-owns-Esc gating; `escape_has_job` already fires for any open popup).
   New `PopupState.HELP` member (the mutex stays structural). Selection lives in
   `App.help_section: str`, **initialized to the first section key in `App.__init__` AND reset in
   `open_help()`** — the ctor init is load-bearing, not belt-and-braces: smoke/any harness that sets
   `popup_state` directly bypasses the opener, and an unknown key must not `IndexError` the content
   pane (which also falls back to the first section). Not persisted.

5. **Snippets are ACTIONABLE — `Insert at caret`**, reusing the lib picker's exact primitive
   (`session.editor.replace_text_in_current_cursor(text)` + `app.editor_focus_requested = True`) as a
   new **`App` method** `insert_text_at_caret(text) -> bool` (it reads
   `get_current_session_if_exists()` and writes `editor_focus_requested` — genuinely `App` state, so
   a method is the honest shape; a free function taking `app: App` parked in `app.py` matches no
   existing pattern, those live in `widgets`/`popups`/`tabs`). `lib_picker/filtering.py::insert_name`
   is refactored to call it (one insert funnel, per the funnel rule — not two copies).
   Enablement + the disabled-hover tooltip copy match the picker verbatim
   (`editor_was_ever_focused and current_editor_path is not None`). Insert CLOSES the modal (mirrors
   the picker: the editor isn't drawn behind a modal, so the focus latch needs the close).

6. **`F1` opens Help; the cheatsheet toggle moves to `Alt+/`.** F1 was freed this session (the
   Ctrl+/ collision with the TextEditor's hardwired `toggle_comments`) and F1-is-help is the universal
   convention. `TOGGLE_CHEATSHEET` takes `Alt+/`: Alt-chords route `route_always` (`commands.py::
   route_flag`) so it survives an active text input, and `/` keeps the old mnemonic. **`K.slash` must
   be ADDED to `commands.py::_BINDABLE_KEYS` in the same wave** — it is absent today, so
   `capture_chord` could never re-record `Alt+/` and the command would be silently un-rebindable,
   breaking the "all of these are rebindable" promise in `scripts/README.md` (bare `F1` is already
   fine: function keys are in `_BINDABLE_KEYS` + exempt from `chord_needs_modifier`). The stale
   `# F1, not Ctrl+/: …` comment on the spec row is REWRITTEN, not carried (it narrates history —
   `conventions.md ## Code rules`). A `Help` menu-bar item (direct-click, beside `Examples`) is the
   discoverable twin of the chord.

7. **Rendering reuses `ui_primitives`** — `markdown_text(body, app.font_14_bold)` for prose (that is
   the actual bold-font field; there is no `font_bold`); the snippet
   renders as a code block through the SAME styling the chat's fenced blocks use (feed the snippet to
   `markdown_text` wrapped in ``` fences, so there is ONE code-block look in the app and no new
   primitive). **A snippet is ALWAYS fed fenced** — that is an invariant, not an implementation
   detail: the fence's own backticks are what keep `markdown_text` off its no-markers fast path,
   which would otherwise render the snippet as unstyled prose. Verified against the parser: fence
   lines are dropped, interior lines keep their leading whitespace (GLSL indentation survives), and
   a blank line renders as a space. (`text_colored` is printf-shaped in the C++ API but this
   binding takes `fmt: str` and passes it through — a `%` in a snippet renders literally; verified
   headlessly, no escaping needed.)

8. **The generated content is pinned by tests** (`tests/test_help_content.py`, GL-free): (a) the
   engine-uniform table COVERS `ENGINE_DRIVEN_UNIFORMS - TABLE_UNIFORMS` exactly — a new builtin
   with no doc entry fails here, the wire that keeps D2 honest; (b) every `HelpSection` has a
   non-empty `key`/`title`/`body`, keys are unique, and `len(help_sections()) >= 5` (an empty list
   would `IndexError` at `open_help`, and a vacuous "all sections are valid" passes over `[]`);
   (c) the shortcuts section names every `CommandCategory` that has at least one spec.

9. **Chord uniqueness becomes a REGISTRY invariant, not a per-feature test.** A "does the shortcuts
   section mention F1-Help" assertion is theater: it stays green when `TOGGLE_CHEATSHEET` ALSO still
   holds F1, which is exactly the double-dispatch bug this swap exists to remove (`_dispatch_registry`
   loops every spec with no first-wins break). Instead `tests/test_command_routing.py` gains one test:
   no two `COMMAND_SPECS` entries share a `default_chord` within overlapping scopes (reuse
   `settings.py::_scopes_overlap`, the same rule the interactive rebinder already enforces — the
   static table has NO such guard today). This protects every future chord edit, not just this one.

10. **The consumer is exercised headlessly, not just the producer.** `help_content` tests can all pass
    while the modal is never dispatched. So `scripts/smoke.py` calls `app.open_help()` (the real
    opener — NOT a bare `popup_state = PopupState.HELP` assignment, which would bypass the very
    state-reset D4 promises), runs the draw stretch, and asserts `app.help_section == help_sections()
    [0].key` — the wire that proves reset-on-open is connected. A `CommandId.HELP` with no callback
    already fails smoke by `KeyError` at first dispatch (`hotkeys.py`), so that wire is free.

11. **`PopupState.HELP` renders no node previews — a decision, not an oversight.** `ui.py::
    update_and_draw` renders nodes only when no popup is open, with ONE escape hatch
    (`elif popup_state == EXAMPLES` → render the example nodes for the animated grid). HELP shows no
    previews, so it correctly falls in the no-render branch and the gate is left untouched. Revisit
    if a Help section ever wants a live preview (then it needs its own `elif`, or it renders black).

12. **`insert_text_at_caret` returns `bool` (inserted / not), and both call sites close on it.** The
    picker today closes the modal unconditionally after `insert_name`, relying on an upstream
    `can_insert` gate — duplicating that gate in Help is how the two drift into "no-op and close".
    The shared primitive owns the outcome; the caller closes only when it returns True. **Help's
    Insert additionally requires the current editor tab to be a node SHADER** (`current_editor_path`
    ending `.frag.glsl`): the picker inserts a bare function name, harmless anywhere, but a
    `uniform float u_time;` block dropped into a `script.py` or a lib file is nonsense the existing
    gate happily allows.

13. **The taught contract is what the ENGINE enforces, verified against the compile path — not a
    stricter folk version.** Wrong help is worse than none, so these four facts are pinned here and
    the section copy must not exceed them:
    - `#version 460 core` is **required** — nothing injects it; `resolver.py` splices the lib
      preamble only AFTER an existing `#version`/`#extension` header.
    - `in vec2 vs_uv` is the real vertex-stage output (`resources/shaders/default.vert.glsl`:
      `vs_uv = a_pos * 0.5 + 0.5`), carrying **[0,1]** UVs.
    - The fragment output is bound **by LOCATION, not by name** — there is no
      `glBindFragDataLocation` and no `layout(location=…)` anywhere in the compile path, so a single
      `out vec4` takes location 0 whatever it is called. Teach "declare one `out vec4` (the examples
      call it `fs_color`)", NEVER "it must be named `fs_color`".
    - `u_resolution` is **`vec2`** (while `u_time`/`u_aspect` are `float`), and the shipped examples
      leave it commented out — the generated table's types must reflect that.

## Files touched

- `shaderbox/help_content.py` (new) — `HelpSection`, the engine-uniform table, `help_sections()`,
  the two generators. Leaf: no imgui, no `App`.
- `shaderbox/popups/help.py` (new) — `draw_help(app)`: modal + section list + content pane + actions.
- `shaderbox/app.py` — `PopupState.HELP`; `open_help()`; `help_section` field (init in `__init__`,
  reset in `open_help`); the `insert_text_at_caret(text) -> bool` method; the `CommandId.HELP`
  entry in `_build_command_callbacks`.
- `shaderbox/commands.py` — `CommandId.HELP` ("Help", `F1`, `C.TOOLS`); `TOGGLE_CHEATSHEET` chord
  `F1` → `Alt+/` + its stale comment rewritten; `K.slash` added to `_BINDABLE_KEYS`.
- `shaderbox/ui.py` — `draw_help(app)` in the popup dispatch; `Help` menu-bar item.
- `shaderbox/popups/lib_picker/filtering.py` — `insert_name` delegates to `insert_text_at_caret`;
  the picker's call site closes on its `bool` return (D12).
- `scripts/README.md` — the shortcuts TABLE **and the prose line above it** ("press F1 … for the
  cheatsheet" becomes false; F1 is Help, `Alt+/` is the cheatsheet).
- `tests/test_help_content.py` (new) — D8.
- `tests/test_command_routing.py` — the chord-uniqueness invariant (D9).
- `scripts/smoke.py` — `app.open_help()` + the `help_section` reset assert (D10).
- `ai_docs/roadmap.md` — 055 row + banner.

## Manual verification

(Consumer-driven, one falsifier each, per `dev_flow.md` step 7.)

1. **F1 opens Help; Escape/Close closes it.** `make run` → F1 → the modal opens on the first section.
   Falsifier: nothing opens, or the cheatsheet overlay toggles instead (a stale chord).
2. **Sections switch + content renders.** Click each section: prose wraps, inline `code` chips
   render, the GLSL snippet shows as a code block. Falsifier: clipped/overlapping text (a wrap bug),
   or a snippet rendering as prose.
3. **Insert at caret works and is correctly gated.** With a shader tab focused: pick the engine-
   uniform section → Insert → the declarations land at the caret and the modal closes, editor
   re-focused. With NO editor touched since launch: the button is disabled and hovering shows the
   "click into the code editor first" tooltip. Falsifier: insert into the wrong file/position, or
   enabled with no editor.
4. **Generated content is genuinely live.** The engine-uniform section lists exactly `u_time`/`u_aspect`/
   `u_resolution` (no glyph tables); the Shortcuts section lists every category with F1 shown as
   Help and `Alt+/` as the cheatsheet. Falsifier: a missing/extra uniform, a stale chord.
5. **Cheatsheet still works on its new chord.** `Alt+/` toggles the overlay; the Settings checkbox
   still governs it. Falsifier: dead chord, or the overlay and Help both reacting to one key.
6. **`make check` + `make smoke` + `uv run pytest` green.** Smoke exercises the HELP draw path.
   Falsifier: import error, popup-mutex assert, the D6 coverage test.

## Review history

**Pre-impl round 1 (2026-07-17, two adversarial opus reviewers: correctness&design PARTIAL,
verification&blast-radius PARTIAL — all real findings folded in above, no re-spawn needed):**
- Would-not-run, fixed: `app.font_bold` doesn't exist (it's `font_14_bold`) — an `AttributeError` on
  first draw that pyright would also have blocked.
- Wrong-content, fixed (the most valuable finding): the fragment output is location-bound, NOT
  name-bound — the panel would have taught a stricter contract than the engine enforces. D13 now
  pins all four taught facts against the compile path.
- Silently-broken, fixed: `K.slash` is absent from `_BINDABLE_KEYS`, so moving the cheatsheet to
  `Alt+/` would have made it un-rebindable while still dispatching.
- Test-theater, replaced: the "shortcuts section contains an F1-Help row" assertion stays GREEN when
  `TOGGLE_CHEATSHEET` also still holds F1 — i.e. it cannot see the double-dispatch bug the swap
  exists to fix. Replaced by a registry-wide chord-uniqueness invariant (D9) that guards every
  future chord edit.
- Consumer-vs-producer, fixed: all three original tests asserted on `help_sections()` output, which
  passes even if the modal is never dispatched (D10 now wires an asserting smoke stretch through the
  real `open_help()`).
- Latent crash, fixed: smoke sets `popup_state` directly, bypassing `open_help()`, so `help_section`
  must be initialized in `__init__` too (D4).
- Also folded: the leaf/GL-free claim corrected (D1), `insert_text_at_caret` moved to an `App` method
  and given a `bool` return (D5/D12), Help's insert gated to `.frag.glsl` tabs (D12), Esc + the
  `keep_open` contract named (D4), the no-preview render-gate decision recorded (D11), dot-leaders
  dropped (D3), `scripts/README.md`'s prose line added to Files-touched.
- Rejected as false positive: "a `%` in a snippet is a printf hazard in `text_colored`" — verified
  headlessly, this binding takes `fmt: str` and passes it through; `%` renders literally. (The
  reviewer correctly flagged it as inferred-not-tested.)
- Non-finding confirmed: bare `F1` IS capturable/rebindable (function keys are in `_BINDABLE_KEYS`
  and exempt from `chord_needs_modifier`), so Help does not become an un-rebindable command.

**Post-impl round 1 (2026-07-17, two adversarial opus reviewers: correctness&UI PARTIAL, spec-
fidelity&content-truth PASS — converged after triage, no re-spawn):**
- Fixed: the **shortcuts snippet was insertable into a shader** — the `.frag.glsl` gate guards
  WHERE a snippet lands, not WHETHER it is GLSL, so inserting the chord table would have been a
  guaranteed compile error. `HelpSection.insertable` now marks a display-only snippet and the
  button is omitted entirely (which also delivers D1's "no Insert button for an empty snippet",
  previously drawn-but-disabled with a misleading tooltip).
- Fixed: `_insert_target_ok` now reads the typed `EditorTab.kind == "shader"` instead of re-deriving
  it from a `.frag.glsl` filename test (correct today, but a user-named `*.frag.glsl` lib file would
  have fooled it — and the lib IS user-editable, as this very panel teaches).
- Fixed (content): "a colour picker for `vec3`" was optimistic — `ui_models.py` auto-selects the
  colour widget only when the name ends in `color`; copy + snippet corrected (`u_tint_color`).
- Fixed: the roadmap row + banner (the one Files-touched gap, flagged by both reviewers).
- Verified, no action: **both shipped snippets COMPILE** — the reviewer linked them headlessly on a
  real EGL context through `ShaderLibIndex.build` + `resolve_usage` + `default.vert.glsl`, zero
  errors; every factual claim in all five sections checked TRUE against engine code; the D2 coverage
  assert and the D9 chord invariant were falsified-tested (an injected duplicate chord was caught).
- Accepted as-is: help.py's own Esc branch is dead code (the app-level `_handle_escape` runs first
  and already closed the popup) — harmless, idempotent, and mirrors the picker; `tree.py`'s
  context-menu insert ignores the new bool (imgui closes a context popup itself; unchanged from
  pre-refactor behavior); `help_sections()` rebuilds 3×/frame while open (microseconds, and the
  rebuild is what keeps the content live).

## Open questions for the user

None — the two the maintainer deferred ("tune later after manual review") are the section CONTENT
wording and whether orientation sections get added; both are data-only edits post-review.
