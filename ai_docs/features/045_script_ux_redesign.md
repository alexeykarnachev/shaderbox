# 045 — script UX redesign (DRAFT)

> **STATUS: SPEC DRAFT — NOT plan-locked.** Captured verbatim from the maintainer's design voice message
> (foo chat, 2026-06-13). The current 042 script UI/UX was judged poor; this is the redesign wave.
> **FIRST step when work resumes: a LARGE review swarm** that collects all the concrete requirements +
> code details from the codebase (the exact widgets, files, current behavior this changes), THEN the
> usual dev flow (plan-lock → pre-impl review → implement → post-impl review → sanitize). This draft is
> written to be implementable from a HEADLESS box (no display during implementation) — so it is
> deliberately exhaustive about exact widgets, layout, files, and the current code each change touches.
> Nothing from the source message is dropped; the maintainer's "надо подумать" items are preserved as
> OPEN QUESTIONS, not silently resolved.

Builds on 042 (which shipped the headless engine surface + the management verbs as MACHINERY behind a
weak placeholder UI). 045 is a UI/UX rebuild over that done machinery + two small backend changes (the
stub docstring shape; removing the curated `exec` globals). All the engine verbs already exist:
`create_script` / `detach_script` / `reset_script` / `get_brain_status` / `get_script_file_for` /
`is_uniform_scriptable` / `stub_for` / `brain_stub_for` on `ProjectSession` (see 042 spec + dev_flow
`scripting/` map). The redesign REPLACES the 042 affordances (chips/context-menu/strip) and the OS-editor
launch — it does not add a parallel system.

## Goal

A discoverable, intuitive, in-app scripting UX: attach/edit/disable a script from a clearly-visible
per-row control (no hidden right-click), edit ALL scripts INSIDE ShaderBox's own code editor with a
tab bar for multiple open scripts, and a node-level control mirroring the per-uniform one for the
node-brain. The Python script behaves like a normal Python file (no curated sandbox globals).

---

## Part A — Remove the 042 affordances that are wrong

A1. **Remove the per-row right-click context menu entirely.** The "Right-click a uniform for script
actions" menu (`widgets/uniform.py::_script_actions_menu` + the "Right-click a uniform for script
actions" hint line in `tabs/node.py::draw`) is unintuitive — the maintainer did not find the option on
the first try. DELETE the context menu, its hint line, and the per-row `confirm_delete_popup` wired to it.
The verbs it carried (Make scriptable / Open / Restart / Detach script) move to the new per-row control
(Part B) and the editor (Part C).

A2. **Remove the yellow `py` pill (the `script_chip`).** The current driven-row indicator
(`ui_primitives.py::script_chip`, the `pill_button`-based "py" pill in `widgets/uniform.py`'s driven
branch) goes away — "не будет такого пила". The new per-row script control (Part B) replaces it. (Decide
during impl whether `script_chip` the primitive is deleted or repurposed — see Part B.)

A3. **Remove "New node-brain script" from the three-dots (`...`) node-actions popup.** In
`tabs/node.py` the `node_actions_popup` (opened by the `...##node_actions` ghost button) currently has
"Save as template" + "New node-brain script". REMOVE the "New node-brain script" item. Leave ONLY "Save
as template" in the popup for now. (Maintainer note: the three-dots mechanism itself is a temporary
solution — "Save as template" will later move elsewhere and the three-dots removed — but that is NOT this
wave; this wave only removes the brain-script item.) Brain-script CREATION moves to a prominent place
(Part D).

A4. **Scripts are opened in OUR code editor, NOT the OS.** Remove the `open_in_file_manager` /
`open_file_in_default_app` launch path for scripts (`App.open_script_file` / `App.create_script_for`
currently call `open_file_in_default_app`). We do NOT open the scripts folder and do NOT hand the user off
to their OS text editor — everything is done inside ShaderBox's own editor (Part C). (The
`open_file_in_default_app` util helper added in 042 may become unused — check + remove if so.)

A5. **Remove the curated math globals from the script `exec` namespace.** In
`scripting/behavior.py::_build_globals`, the curated math vocab (`sin`/`cos`/`sqrt`/`clamp`/`lerp`/`mix`/
`pi`/`tau`/`math`/…) and the `_no_import` shim (which blocks `import` with a friendly message) are REMOVED.
The maintainer is explicit: a script is a normal Python file — the user can `import math` (or anything)
themselves and call `math.sin(...)` etc. No safety/sandbox concerns (single local user writing their own
code); the "pre-loaded math" convenience is an over-complication to delete. **CONSEQUENCE the swarm must
catch:** the current `stub_for` / `brain_stub_for` output AND any shipped example/docstring AND the smoke
seed (`scripts/smoke.py` `u_wave.py` uses bare `sin(ctx.t)`) AND the dogfood scenarios use bare
`sin/clamp/lerp` with NO import — removing the globals BREAKS them all. The migration must rewrite every
stub/example to `import math` + `math.sin(...)` (or whatever the maintainer prefers) so a freshly-created
script still compiles + runs. The exec globals likely keep ONLY what the class machinery needs
(`__build_class__`, `__name__`, the base `ScriptBehavior`, `Ctx`, the output types `Vec2/3/4`/`Array`/
`Text`/`MouseState`, and the builtin types annotations resolve against) — the swarm must enumerate the
MINIMUM globals that keep `class Behavior(ScriptBehavior)` + eager method annotations working once the
math vocab is gone, since `exec` with a custom globals dict does NOT auto-populate `__builtins__` the way
a normal module does. (Check: does removing `_no_import` mean `__import__` must be restored to real
builtins so `import math` works? Almost certainly yes — the user needs real `import`.)

---

## Part B — The per-uniform script control (the new visible affordance)

B1. **Row layout (unchanged spine + a new trailing control).** The per-uniform row in
`widgets/uniform.py::draw_ui_uniform` keeps its current spine and ORDER:
`[ type pill ] [ uniform name ] [ the control widget ]`, then ADD a new **script button/control AFTER
the control widget** (trailing, end of the row). The type pill (drag / color / …) stays exactly as it is.
The script control is the visible, discoverable entry point that replaces the right-click menu + the `py`
pill.

B2. **The maintainer's mental model for the control (a starting image, not a lock):** a grey
"tools"-style icon, looking un-pressed/inactive by default. Click it → the script opens (in our editor,
Part C) and you can edit it. Click the icon again → the script becomes DISABLED (inactive). So the
control is conceptually a toggle: attach/enable ↔ disable, with "click opens the editor."

B3. **OPEN QUESTION (maintainer: "надо подумать", explicitly unresolved — do NOT pick silently):** the
icon-toggle is insufficient as a state machine. Two cases it doesn't cover:
  - How do I **view a script while it is INACTIVE** (disabled but I want to read/edit it)?
  - How do I **hide/disable a script while it is ACTIVE** without losing the ability to reopen it?
  A single icon-toggle conflates "open in editor" with "enable/disable." The maintainer suspects an icon
  may not be the right widget here. **This is the central design problem of the wave — the swarm + the
  plan-lock must produce the full state machine** (states: no-script / script-present-active /
  script-present-inactive; transitions: create, open-in-editor, enable, disable, detach; and which gesture
  drives each). Candidate shapes to weigh at plan time: a split control (an open affordance + a separate
  enable/disable toggle), a pill with a state + a context affordance, a small two-part button. Decide with
  concrete imgui mockups; this is exactly where 042's single-gesture design failed.

B4. **"Disabled script" semantics (NEW — a script can exist but be inactive).** 042 has no notion of an
inactive-but-present script (a file either drives a uniform or doesn't). 045 introduces a DISABLED state:
the `.py` exists on disk + is openable/editable, but the engine does NOT tick it (the uniform returns to
manual control). The swarm must design the persistence: how "disabled" is recorded (a sibling marker? a
rename like `u_x.py.disabled`? a per-node manifest? a field — but binding is by-filename and stateless per
041, so a marker file or a disabled/ subdir is more in keeping). It must survive reload + restart, and the
engine's `reload`/`tick` must skip a disabled binding. This is a real engine change, not just UI.

B5. **When a script is ACTIVE, disable the uniform's value widget BUT allow changing its TYPE.** This is
the 042 read-only behavior, refined: while a script drives a uniform, the value widget (slider / color
palette / text box / etc.) is DISABLED — the user cannot set the value by hand (the script owns it). BUT
the **type pill stays enabled** — the user can still switch the input type (e.g. drag ↔ color), because
the type is only a VIEW of the uniform; the script's values are then displayed/interpreted through the
chosen view. (042 currently shows a read-only `format_auto_value` readout and skips the whole control
branch — 045 instead draws the real widget but `begin_disabled`'d, so the live value shows through the
widget itself, and the type pill remains clickable.)

B6. **Uniform Text: move the char-count out of the script column.** The text-uniform branch
(`widgets/uniform.py`, the `input_type == "text"` path) currently draws a `caption_text(f"{len(text)}/{cap}")`
char-counter (e.g. "46/64") in the trailing column — exactly where the new script control goes. MOVE the
count OUT of that column: put it in the uniform NAME (in parentheses, e.g. the name cell shows
`u_label (46/64)`), or similar. The maintainer: "не очень важно [where exactly], но оттуда надо это
убрать." (Also check the `array` branch's `{len}/{cap}` and `[...]  ({cap})` captions on lines near it —
same column-collision risk; the swarm should confirm whether they need the same move.)

---

## Part C — In-app code editor for scripts (tabbed, multi-file)

C1. **Edit scripts in ShaderBox's own code editor.** Reuse the existing goossens `TextEditor` pane
(`tabs/code.py`, `app.editor_sessions: dict[Path, EditorSession]`, path-keyed). A `.py` path is opened
INSIDE the app, not in the OS. (The binding supports `Language.python()` — verified present in
imgui_color_text_edit — so the editor can syntax-highlight Python; `get_session` currently hardcodes
`set_language(Language.glsl())`, which must become suffix-aware.)

C2. **Multiple open scripts → a TAB BAR at the top of the editor pane.** This is a new capability: today
the code pane shows ONE file at a time (the node shader OR one lib file, via the `_explicit_editor_path`
override + the `< back to node` chrome in `code.py::draw_chrome`). 045 needs MANY open files with a tab bar
across the top of the editor area. The user switches between: the node's shader, a uniform's script, the
node-brain script (and presumably open lib files too — confirm scope at plan time).

C3. **Tab labels = the file path relative to the project root dir.** Since everything is relative to the
project directory, a tab's label can be the relative path of the file from the project root (e.g.
`nodes/<id>/scripts/u_color2.py`), or something of that shape. Maintainer: "или что-нибудь вроде того...
тут тоже надо подумать" — so the exact label format is an OPEN QUESTION (full relative path may be too long
for a tab; consider `<node-name>/u_color2.py` or just the filename with a tooltip showing the full path).

C4. **The editor open-set is now a list, not a single override.** This supersedes the
`_explicit_editor_path: Path | None` single-slot model + the `< back to node` chrome. The swarm must design:
the open-tabs list (ordered), the active tab, open/close/switch operations, what closing the last tab does,
whether the node shader is a permanent/pinned first tab, and how this interacts with the existing
`editor_sessions` cache + the FPE-behind-modal guard (`tabs/code.py` skips drawing the editor while any
popup is open — `conventions.md ## Known quirks`). NOTE: this is the long-deferred "multi-file editor —
tab bar / file tree / split" from `todo.md` (decision 8 of `015_shader_include_library.md`) — 045 finally
triggers it. Reconcile with that deferral.

C5. **Opening a script** (from the per-uniform control B2 or the node control D) switches the editor to
that `.py`'s tab (creating the tab + the `EditorSession` if not already open), with Python highlighting.
Save (Ctrl+S, the existing editor save path) writes the file; the engine's `reload_scripts` mtime poll
picks it up and re-instantiates (existing 041 hot-reload — no change needed there).

---

## Part D — The node-brain control (mirror of the per-uniform one)

D1. **A node-level script control, same shape as the per-uniform one.** The node-brain (whole-node script)
gets a control built the SAME way as the per-uniform control (Part B) — whatever final shape that takes
(pill / tools-icon / button — same OPEN QUESTION as B3). It mirrors the per-uniform affordance but acts on
the node's `script.py`.

D2. **Placement: the node header row.** Draw the node-brain control in the node's header, where
`node name / resolution / [...]` currently live (`tabs/node.py::draw`, the row with the node-name input +
the resolution combo + the `...` ghost button). The brain control sits up there with them.

D3. **Brain creation moves to a prominent place (out of the three-dots).** Since A3 removes "New node-brain
script" from the `...` popup, creation now lives on this visible node-header control (e.g. the control in
its no-script state IS the "create brain" affordance, mirroring how the per-uniform control creates a
per-uniform script). Exact gesture = the B3 state-machine decision applied to the node.

D4. **The brain's active/disabled/error states reuse the per-uniform state machine + the engine
accessors** (`get_brain_status` already returns the sentinel error + driven count + soft-key errors — 042).
Where the brain's error / driven-count / soft-key list renders in the new design (a strip? attached to the
header control? a tooltip?) is an OPEN QUESTION the swarm must place — 042's separate "brain strip" may or
may not survive the redesign.

---

## Part E — The script stub / docstring rework

E1. **Docstrings must be proper Python docstrings, owned by the right scope.**
  - The **`update` method docstring** carries the per-frame context reference: `ctx.t` (elapsed seconds),
    `ctx.dt` (delta seconds), `ctx.frame`, `ctx.mouse.x`/`ctx.mouse.y` — this CONTEXT info belongs in the
    `update` method's docstring, NOT the class docstring (042's `stub_for` puts it in the class docstring —
    wrong).
  - The **class docstring** carries something higher-level: what this behavior does (e.g. "Drive u_color2
    each frame"), maybe a bit more — but NOT the ctx field reference.
  - The **`__init__` docstring**: state whether `__init__` may do work (it may — "наверное разрешаем"), and
    that whatever runs in `__init__` runs ONCE — at app start, before rendering, on reload, etc. Write that
    in the `__init__` docstring.

E2. **Later: an opt-out for docstring generation.** A future setting (Settings panel) to NOT generate these
docstrings when creating a script. NOT this wave — "потом всё сделаем, потом оптимизируем." Note it as a
follow-up deferral; do not build the toggle now.

E3. **Stub bodies must work WITHOUT the curated globals (ties to A5).** Once A5 removes the math vocab,
`stub_for` / `brain_stub_for` must emit working bodies — either using only literals/`self` (the current
defaults like `Vec3(0,0,0)` already do) or showing the `import math` pattern in a comment/example so the
user sees how to reach math. Ensure a freshly-stubbed script compiles + runs with the new (minimal) globals.

---

## Backend changes summary (the non-UI surface 045 touches)

- `scripting/behavior.py::_build_globals` — strip the curated math vocab + `_no_import`; restore real
  `import` (the minimum globals that keep the class form + eager annotations working). (Part A5.)
- `scripting/engine.py` `stub_for` / `brain_stub_for` — rework docstrings (method/class/init scoping) +
  ensure bodies compile under the new globals. (Parts E1, E3.)
- `scripting/engine.py` reload/tick + `ProjectSession` — the new DISABLED-but-present script state (skip
  ticking a disabled binding; persist the disabled marker; survive reload/restart). (Part B4 — a real
  engine change.)
- The migration of every bare-`sin/clamp/lerp` usage (stubs, `scripts/smoke.py` seed, dogfood scenarios,
  any shipped example) to explicit `import math`. (Part A5 consequence.)

## UI surface 045 touches

- `widgets/uniform.py` — remove the context menu + `py` chip + driven-row read-only branch; add the
  trailing per-row script control (B); disable the value widget but keep the type pill when active (B5);
  move the text char-count out of the script column (B6).
- `tabs/node.py` — remove the "New node-brain script" three-dots item (A3) + the right-click hint (A1);
  add the node-header brain control (D); place brain error/status in the new design (D4).
- `tabs/code.py` + `app.py` — the multi-tab editor (C): suffix-aware `set_language` (Python),
  the open-tabs list replacing `_explicit_editor_path`, the tab bar, open/close/switch. Supersedes the
  `< back to node` chrome.
- `app.py` — remove the OS-editor launch for scripts (A4); the open-script-in-editor path (C5); check/remove
  now-unused `open_file_in_default_app`.
- `ui_primitives.py` — the new per-row/node script control primitive (shape TBD per B3); likely retire/
  repurpose `script_chip`; possibly retire `notice_strip`/`confirm_delete_popup` if the redesign drops them
  (confirm at plan time — they may still be used).

## OPEN QUESTIONS (maintainer's explicit "надо подумать" — resolve at plan-lock, do NOT silently pick)

1. **The script-control state machine (B3) — THE central problem.** States no-script / active / inactive;
   how to view an inactive script, how to disable an active one without losing reopen — a single icon-toggle
   is insufficient. Produce the full state machine + the concrete widget(s) with imgui mockups.
2. **The exact widget for the script control (B2, D1)** — tools-icon vs pill vs button, for both the
   per-uniform row and the node header. The maintainer is unsure an icon fits.
3. **Where "create script" lives prominently (D3)** now that it's out of the three-dots + the right-click.
4. **Tab label format (C3)** — full relative path vs node-name/filename vs filename+tooltip.
5. **Tab-editor scope (C2/C4)** — does the tab bar also hold open lib files + the node shader as tabs, or
   only scripts? Is the node shader a pinned first tab? What does closing the last tab do?
6. **Disabled-script persistence (B4)** — marker file / `disabled/` subdir / rename / manifest; must be
   stateless-by-filename-compatible (041) and survive reload+restart.
7. **Where the brain error/status renders (D4)** — keep 042's brain strip, or fold into the header control?
8. **Text char-count new home (B6)** — in the name parens vs elsewhere (low importance, but must leave the
   script column).

## Follow-ups (NOT this wave)

- Docstring-generation opt-out setting (E2).
- The three-dots (`...`) mechanism removal + "Save as template" relocation (A3 note) — temporary, future.

## NOT in this draft (to be collected by the resume-time review swarm)

This draft captures the maintainer's stated requirements. It deliberately does NOT yet enumerate: the exact
current line-level behavior of every touched widget, the full `editor_sessions`/`current_editor_path`
mechanics, the complete list of bare-math usages to migrate, the FPE-modal interaction with a tab bar, or
the imgui tab-bar API specifics. **The first resume step is a large review swarm to collect all of that from
the code**, then plan-lock with the OPEN QUESTIONS answered, then the usual dev flow.
