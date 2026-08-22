# Audit: UI surfaces and extension seams (current state)

Source: code read of `ui.py`, `app.py`, `tabs/*`, `widgets/*`, `popups/*`, `commands.py`,
`help_content.py`. Only the parts bearing on a pipeline/interactivity feature are kept here;
this is not a full UI catalogue.

## Layout

**No docking** (`grep -rn "dock"` -> three commented-out lines in `theme.py`). Layout is
hand-computed geometry in `ui.py::update_and_draw`.

Exactly three top-level imgui windows: the full-screen `"ShaderBox - UI"`, the floating
`"Copilot"` (drawn after the main window closes so it stays on top), and the transient
rendering overlay. The cheatsheet is not a window — it paints on the foreground draw list.

Inside the main window: menu bar, then a two-column split (editor column + copilot bar on the
left, app panel on the right) with a draggable splitter. The **app panel** holds the
current-node preview image on top and a control panel below (node grid on the left, node
settings tab bar on the right).

The preview image feeds `app.script_mouse` via `item_normalized_mouse` — that is the sole
consumer of cursor-over-preview. The whole app panel is wrapped in
`begin_disabled(app.copilot_turn_active)`.

## Tab bars

Two, and only two (`grep -rn "begin_tab_bar"`).

**Editor tabs** — kinds are `Literal["shader", "script", "lib"]` on
`editor_types.EditorTab(path, kind, node_id)`. Path is the tab identity; all opens funnel
through `App._focus_or_add_tab`, which re-focuses rather than duplicating. Labels are
node-derived (`"{node_name} (shader)"`). Below the tab row sits the shared **error strip**,
capped at 3 rows with click-to-jump; a script tab's engine errors are adapted to `ShaderError`
shape so one strip serves both.

**Node-settings tabs** — the registry is `ui.py::_NODE_TABS`, exactly three entries:
`(Node, NodeTab.NODE, tabs/node.draw)`, `(Render, ...)`, `(Share, ...)`. Selection persists as
`app_state.active_node_tab`.

## The four extension seams

These are the registration points a pipeline feature would use.

**(a) A new node-settings tab.** Add a module under `tabs/` exposing `draw(app) -> None`; add
a `ui_regions.NodeTab` member; append a `(label, NodeTab.X, draw)` triple to `_NODE_TABS`.
`grep -rn "_NODE_TABS"` returns the definition and one loop — nothing else. A direct-jump
chord is NOT automatic: it is a `CommandId.FOCUS_TAB_*` entry plus a `command_callbacks` wire.

**(b) A new modal popup.** Add a `PopupState` member; add `App.open_x()` calling
`_open_popup`; write `popups/x.py` with a `draw_x(app)` that early-returns unless
`popup_state == PopupState.X`; call it from the popup block in `update_and_draw`. The single
`PopupState` field IS the "at most one open" mutex. **Note: `popup_suppresses(scope)` returns
True unconditionally — an open modal kills every command.**

**(c) A new editor tab kind.** Add a value to the `EditorTabKind` Literal; handle it in
`tab_label`, in `code.draw`'s session-resolution branch, and in the errors branch; add an
`App` opener building an `EditorTab`.

**(d) A new Share outlet.** Implement `exporters/base.Exporter` and register it in
`App.__init__`; `tabs/share.py` and the Settings Integrations loop pick it up automatically.

## The shared insert seam

`App.insert_text_at_caret(text) -> bool` is documented as "The one insert seam (lib picker +
help panel)". It returns whether the text landed, so the caller closes its modal only on a
real insert. The Help panel additionally gates on `app.active_tab.kind == "shader"` so a GLSL
block cannot land in a lib file or a `script.py`.

Both content-delivery popups matter as precedent for how a new "cascade example" would reach
a node:

- **Examples** (`Alt+E`) — a 3-column grid over `app.ui_node_examples` (live-rendering nodes;
  `update_and_draw` has an explicit branch rendering every example each frame while the popup
  is open). "Open a copy" calls `App.create_node_from_example` — it creates a **new node** and
  never modifies the current one.
- **Lib picker** (`Ctrl+P`) — inserts only the **function name** at the caret, not the body
  (the resolver splices bodies at compile time). Or "Open file at declaration" opens a
  `kind="lib"` tab scrolled to the line.

## Commands

24 commands in `commands.py::COMMAND_SPECS`, 1:1 with `CommandId`. Categories: File, Node,
Editor, View, Tools. Chords are imgui KeyChord ints; `app_state.key_bindings` overlays
defaults into `app.effective_bindings`, and a chord-uniqueness invariant guards edits.

Free-key note for any new feature: the registry already uses `Ctrl+1/2/3` for the three node
tabs, so a fourth tab has no established chord pattern beyond adding `Ctrl+4`.

## Settings

Two separate persistence stores, and the distinction matters:

- **Non-secret UI state** -> `UIAppState` (pydantic, `extra="forbid"`), saved on quit, loaded
  fail-soft per key. Adding a setting = a field with a default + a widget.
- **Secrets/integrations** -> `IntegrationsStore`, its own file, saved **eagerly** at the
  point of change.

Editor settings are applied on **close only** — calling a `set_*()` on the TextEditor while
the modal is open FPE-crashes it.
