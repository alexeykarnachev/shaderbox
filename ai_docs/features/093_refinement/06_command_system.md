# 093 / finding 17 — The command system

The app's verbs, designed as one system: what a user can invoke, what each is called, which
menu it lives under, in what order, and which chord fires it. `shaderbox/commands.py`'s
`COMMAND_SPECS` IS this table; the menu bar, the palette, the keyboard cheatsheet, the
Settings rebinder and the Help panel's shortcuts section all render it in this order, so a
verb is filed once and shows up everywhere. The maintainer's instruction that produced it:
"you must not assume any pre-existing order. You have all the code, all the commands we
have, all the features, you should design the whole system!" — the first landing of the
menu bar had rendered the cheatsheet's grouping, which was never designed.

## The rules

1. **A category is the OBJECT a verb acts on**, and the categories run in the order a
   desktop bar reads: the app's files and settings (File), the open document (Document), its
   passes (Pass), the code editor (Editor), what is shown (View), and Help.
2. **Within a category, groups; within a group, most-used first.** A group is opened by
   `separator_before` on its first spec. The first spec of a category never carries one.
3. **A label names its object**, so it reads the same under a menu heading, in the flat
   palette and on the cheatsheet: `Reset document`, `Next code tab`, `Play/stop script`. A
   cycling verb says `Next ...`; a toggle says `Toggle ...`; a surface-opening verb is the
   surface's name (`Settings`, `Projects`, `Shader library`, `Command palette`). No trailing
   ellipsis anywhere.
4. **Every command has a menu home**; a verb that exists only as a button or a chord is the
   defect. A button that opens a command's surface takes the command's label
   (`command_label`).
5. **A destructive verb carries `confirm_label`** and renders as a confirm submenu wherever it
   is a menu item (the bar, a tile's menu).
6. **Chords are defaults with muscle memory behind them**: this design keeps every existing
   chord and binds nothing new. `Import passes` and `Open folder` are unbound.

## The map, as it renders

```
File
  New document        Ctrl+Shift+N
  Save                Ctrl+S
  ─
  Projects            Alt+O
  Settings            Alt+S
  ─
  Quit                Ctrl+Q

Document
  Open script         Alt+R
  Open graph          Alt+G
  Open folder
  ─
  Play/stop script    F5
  Reset document      F6
  ─
  Delete document ▸ Move to trash    Alt+D

Pass
  Add pass            Alt+A
  Import passes
  ─
  Open shader         Alt+C
  Pass settings       Alt+P
  ─
  Next pass           Alt+Right
  Previous pass       Alt+Left

Editor
  Format code         Ctrl+Shift+I     (editor scope)
  Next error          F8
  ─
  Next code tab       Ctrl+Tab
  Close code tab      Ctrl+W           (editor scope)
  ─
  Shader library      Alt+L

View
  Document panel      Ctrl+1
  Uniforms panel      Ctrl+2
  Render panel        Ctrl+3
  Share panel         Ctrl+4
  ─
  Next channel view   Alt+V
  ─
  Toggle copilot      Alt+J
  Next copilot layout Ctrl+H           (copilot scope)
  ─
  Command palette     Ctrl+Shift+P

Help
  Help                F1
  Keyboard cheatsheet Alt+/
  ─
  Examples            Alt+E
```

## What is NOT a command, and why

- The graph canvas's `Fit` and `Arrange` and the node menu's `Group...` / `Dissolve` /
  `Leave group`: verbs on a view or a selection that only the canvas holds; they live on the
  canvas and node menus. Revisit if a graph tab gains its own scope.
- The pass tile's `Delete pass` and the document tile's `Open` / `Open folder` / `Delete`:
  object verbs with a target the menu supplies; the bar's `Delete document` and `Open folder`
  act on the CURRENT document, which is what a bar item can mean.
- The Document tab's resolution mode, aspect chips and canvas fields; the Uniforms tab's sort
  and pass selector; the Render / Share tabs' own buttons: state controls and exporter verbs,
  named where they are.

## What changed against the first landing

The five view-focus verbs came back into the bar under View (an `in_menu` flag went with
them); `Tools` went, its verbs re-filed by object (`Settings` to File, `Add pass` /
`Import passes` / `Pass settings` to Pass, `Shader library` to Editor, `Command palette` to
View, `Examples` / `Help` / the cheatsheet to Help); `Open shader` moved from Editor to Pass;
`Next` / `Previous pass` from Document to Pass; seven labels renamed to name their object;
`OPEN_DOCUMENT_DIR` added; the enum and the table reordered to the map; the separators set
per group. `tests/test_menus.py` pins the map's group structure and the table's order.
