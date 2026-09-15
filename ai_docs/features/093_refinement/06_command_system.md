# 093 / finding 17 — The command system

The app's verbs, designed as one system: what a user can invoke, what each is called, which
menu it lives under, in what order, and which chord fires it. `shaderbox/commands.py`'s
`COMMAND_SPECS` IS this table; the menu bar, the palette, the keyboard cheatsheet, the
Settings rebinder and the documentation modal's shortcuts section all render it in this order, so a
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
5. **A destructive verb confirms in the confirm modal, from every surface** (093 W4). Its
   `App` method builds the request — a title naming its target, one line of consequence, the
   button's word — and every surface calls that one method: a menu item, a bar item, a
   button, a chord, the palette. So the table needs no field for it, the item is plain and
   keeps its chord hint, and the palette offers every `in_palette` spec again. Destructive
   means the loss has no undo in the app: a reset's histories and clock, a chat's
   conversation. A document delete moves to the project trash, and carries the confirm
   because nothing in the app recovers it.
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
  ─
  Open folder
  ─
  Play/stop script    F5
  Reset document      F6
  ─
  Delete document     Alt+D

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
  Next copilot layout  Ctrl+H           (copilot scope)
  Clear chat           (copilot scope)
  ─
  Command palette     Ctrl+Shift+P

Help
  Documentation       F1
  Keyboard cheatsheet  Alt+/
  ─
  Examples            Alt+E
```

## What is NOT a command, and why

- The graph canvas's `Frame all` and `Arrange` and the node menu's `Group...` / `Dissolve` /
  `Leave group`: verbs on a view or a selection that only the canvas holds; they live on the
  canvas and node menus. Revisit if a graph tab gains its own scope.
- The pass tile's `Delete pass` and the document tile's `Open folder` / `Delete` (its `Open`
  went in wave 6: the tile's click is the open):
  object verbs with a target the menu supplies; the bar's `Delete document` and `Open folder`
  act on the CURRENT document, which is what a bar item can mean.
- The Document tab's resolution mode, aspect chips and canvas fields; the Uniforms tab's sort
  and pass selector; the Render / Share tabs' own buttons: state controls and exporter verbs,
  named where they are.
- A modal's own verbs: the Projects modal's `New` / `Open other...` / `Delete` and Settings'
  `Reset library...` act inside the modal that holds their target and confirm with its armed
  danger row (a modal over a modal is not the mechanism's shape); the modal itself is the
  command (`Projects`, `Settings`).
- The copilot chat's `Close` and the `Copilot` chip are second paths to `Toggle copilot`; its
  `Clear` button is the `Clear chat` command's own button, and since 093 W4 it opens the same
  confirm the menu item does.

## What changed against the first landing

The five view-focus verbs came back into the bar under View (an `in_menu` flag went with
them); `Tools` went, its verbs re-filed by object (`Settings` to File, `Add pass` /
`Import passes` / `Pass settings` to Pass, `Shader library` to Editor, `Command palette` to
View, `Examples` / `Help` / the cheatsheet to Help); `Open shader` moved from Editor to Pass;
`Next` / `Previous pass` from Document to Pass; seven labels renamed to name their object;
`OPEN_DOCUMENT_DIR` and `CLEAR_COPILOT_CHAT` added; the enum and the table reordered to the
map; the separators set per group. `tests/test_menus.py` renders the table and compares it
to the fenced map above, parsed from this file, so the map and the table cannot drift (a
review found the earlier set-based test passed a moved `Save` and a moved `Shader library`);
it also pins that no category opens with a separator and that the palette offers every
`in_palette` spec.

Wave 4 then reversed the confirm's shape: `confirm_label` is deleted, a destructive item is
plain and carries its chord hint again, and the confirm is the modal the verb's own callback
opens (rule 5).
