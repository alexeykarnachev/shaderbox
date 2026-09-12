# 092 review — the interaction model, demonstrated

Angle: is the round-3 interaction surface buildable on moderngl + glfw + imgui-bundle 1.92.801
with no node-editor library, and what does it cost the existing funnels. Everything in the
checklist below was run, not reasoned about: a standalone headless probe (hidden glfw window,
own imgui context, the imgui-bundle glfw backend, synthetic mouse events via
`io.add_mouse_pos_event` / `add_mouse_button_event` / `add_mouse_wheel_event`) drawing two
draw-list nodes with per-node `invisible_button`s inside one pannable/zoomable `begin_child`.
The probe lives outside the repo, in the session scratchpad, as
`graph_probe.py`; its essential lines are pasted below.

---

## 1. The probe

Two nodes in graph space, one screen transform, one child:

```python
def g2s(state, origin, p):                      # graph -> screen
    return (origin.x + (p[0] + state.pan[0]) * state.zoom,
            origin.y + (p[1] + state.pan[1]) * state.zoom)

imgui.begin_child(CANVAS_ID, imgui.ImVec2(600, 500),
                  child_flags=imgui.ChildFlags_.borders,
                  window_flags=imgui.WindowFlags_.no_scrollbar
                  | imgui.WindowFlags_.no_scroll_with_mouse   # frees the wheel for zoom
                  | imgui.WindowFlags_.no_move)
origin = imgui.get_cursor_screen_pos()
dl = imgui.get_window_draw_list()

# 1. the canvas's own hit target FIRST, declaring itself overlappable
imgui.set_cursor_screen_pos(origin)
imgui.set_next_item_allow_overlap()
imgui.invisible_button("##canvas_bg", imgui.ImVec2(avail.x, avail.y),
                       imgui.ButtonFlags_.mouse_button_left
                       | imgui.ButtonFlags_.mouse_button_right
                       | imgui.ButtonFlags_.mouse_button_middle)
bg_active, bg_hovered = imgui.is_item_active(), imgui.is_item_hovered()
if bg_active and (io.mouse_delta.x or io.mouse_delta.y):
    state.pan[0] += io.mouse_delta.x / state.zoom
    state.pan[1] += io.mouse_delta.y / state.zoom

# 2. wheel zoom, anchored: convert under the cursor before and after, correct the pan
if imgui.is_window_hovered(imgui.HoveredFlags_.child_windows) and io.mouse_wheel:
    before = s2g(state, origin, tuple(io.mouse_pos))
    state.zoom = max(0.2, min(4.0, state.zoom * (1.1 ** io.mouse_wheel)))
    after = s2g(state, origin, tuple(io.mouse_pos))
    state.pan[0] += after[0] - before[0]
    state.pan[1] += after[1] - before[1]

# 3. per node: picture on the draw list, then the hit rect, ALSO overlappable
for node in state.nodes:
    p0, p1 = g2s(...), g2s(...)
    dl.add_rect_filled(p0, p1, ...);  dl.add_rect(p0, p1, ...)
    size = max(6.0, 14.0 * state.zoom)
    imgui.push_font(font, size)
    dl.add_text(imgui.get_font(), size, (p0[0] + 4, p0[1] + 2), col, node.name)
    imgui.pop_font()

    imgui.set_cursor_screen_pos(p0)
    imgui.set_next_item_allow_overlap()          # so the ports + overlay on top are reachable
    imgui.invisible_button(f"##node_{node.name}", imgui.ImVec2(p1[0]-p0[0], p1[1]-p0[1]),
                           imgui.ButtonFlags_.mouse_button_left
                           | imgui.ButtonFlags_.mouse_button_right)
    if imgui.is_item_active() and (io.mouse_delta.x or io.mouse_delta.y):
        node.pos[0] += io.mouse_delta.x / state.zoom     # screen delta / zoom == graph delta
        node.pos[1] += io.mouse_delta.y / state.zoom
    if imgui.begin_popup_context_item(None):      # None == "the previous item", see §2 FAILS
        ...
        imgui.end_popup()

    for i, port in enumerate(node.ports):        # 7px dots, own buttons
        cx, cy, r = ..., ..., PORT_R * state.zoom
        dl.add_circle_filled((cx, cy), r, col)
        imgui.set_cursor_screen_pos((cx - r, cy - r))
        imgui.invisible_button(f"##port_{node.name}_{port}", imgui.ImVec2(2*r, 2*r))
        if wire_mode and imgui.is_item_active():
            state.wire_drag_from = f"{node.name}.{port}"
        if wire_mode and state.wire_drag_from and imgui.is_item_hovered() \
                and imgui.is_mouse_released(imgui.MouseButton_.left):
            state.wire_drop_on = f"{node.name}.{port}"
```

The rubber band and the wire mid-drag:

```python
drag = imgui.get_mouse_drag_delta(imgui.MouseButton_.left, 0.0)
anchor = s2g(state, origin, (io.mouse_pos.x - drag.x, io.mouse_pos.y - drag.y))
dl.add_bezier_cubic(a, (a[0]+60, a[1]), (b[0]-60, b[1]), b, col, 2.0 * state.zoom, 0)
```

---

## 2. Probe results

22 scenarios, all green in the final run. Each line carries the observed value.

| | scenario | observed |
|---|---|---|
| WORKS | drag a node, zoom 1.0 | graph delta `40.0, 20.0` for a 40,20 screen drag; pan stayed `0.0, 0.0` |
| WORKS | drag a node, zoom 2.0 | graph delta `20.0, 10.0` for the same 40,20 screen drag — `io.mouse_delta / zoom` is the whole rule |
| WORKS | pan on empty canvas | pan `50.0, 30.0` for a 50,30 drag off `is_item_active()` on the bg button |
| WORKS | the node button beats the bg button beneath it | `dragging='scene'`, pan `0.0, 0.0` — **only after** `set_next_item_allow_overlap()` moved onto the bg button |
| WORKS | wheel zoom anchored at the cursor | zoom `1.100`, `io.mouse_wheel` read `1.0` inside the child, the graph point under the cursor moved `5.7e-14` px |
| WORKS | `WindowFlags_.no_scroll_with_mouse` frees the wheel | without the flag the child scrolled to `416.0`; with it `io.mouse_wheel` still reads `-3.0` and scroll stayed `0.0` |
| WORKS | right-click context menu on a node's invisible button | `menu_open_for='scene'` |
| WORKS | right-click on empty canvas opens the canvas menu, not a node menu | `menu_open_for=''`, `canvas_menu=True` — requires the `str_id=None` form, see FAILS below |
| WORKS | double-click a node | `is_mouse_double_clicked` on a hovered node button reports `node_double='scene'`, and the same press also sets `node_clicked='scene'` |
| WORKS | hover a 7px port dot, zoom 1.0 | `port_hovered='scene.u_src'`, hit box 7.0 px |
| WORKS | hover the port 2 graph-px off-center, zoom 1.0 | still `scene.u_src` |
| WORKS | hover a 7px port dot, zoom 0.5 | `port_hovered='scene.u_src'` with a **3.5 px** hit box — imgui hits it, a human will not |
| WORKS | text at zoomed sizes | `push_font(font, 14 * zoom)` → `get_font_size` `{0.37: 6.0, 1.0: 14.0, 1.73: 24.0, 2.5: 35.0}` |
| WORKS | a fractional push_font size is crisp | asked → `(get_font_size, get_font_baked().size)`: `{5.18: (5.0, 5.0), 19.18: (19.0, 19.0), 24.22: (24.0, 24.0)}` — 1.92 bakes a **real face per integer size**, so no bitmap stretching; the size is quantized to an integer, not the text blurred |
| WORKS | the child clips the draw list at its own rect | clip rect `(9.0, 9.0) -> (607.0, 507.0)` for a 600-wide child; no `push_clip_rect` needed |
| WORKS | a node placed outside the child answers no hover | `node_hovered=''` at a point over an out-of-child node — hit testing is clipped with the drawing |
| WORKS | an overlay button on top wins the click | `overlay_clicked='scene'`, `dragging=''` (the gear/✕ pattern the strip already uses transfers) |
| WORKS | without an overlay the node body takes that same press | `dragging='scene'` |
| WORKS | drag a wire port → port | `from='scene.u_src'`, `drop='blur.u_tex'`, bezier drawn mid-drag |
| WORKS | the same drag released over empty canvas | `from='scene.u_src'`, `drop=''`, pan `0.0, 0.0` — the port owns the drag, the canvas does not pan under it |
| WORKS | rubber band from the bg button | band `(364.0, 334.0) -> (434.0, 384.0)`, `get_mouse_drag_delta` `(70.0, 50.0)` |
| WORKS | `add_image_rounded` with a real moderngl texture | `dl.add_image_rounded(imgui.ImTextureRef(tex.glo), p0, p1, (0,0), (1,1), col, 6.0)` returned clean — the node picture is a rounded blit of the pass's own target, exactly as `preview_cell` does it today |
| NOT TRIED | `channels_split` for z-ordering wires under nodes | the probe drew wires after nodes on one list; splitting is the standard fix and the binding exposes `channels_split` / `channels_set_current` / `channels_merge` |
| NOT TRIED | the layout / arrange algorithm at real document scale | out of this angle; the mock's rank layout is the reference |
| NOT TRIED | any visual judgement | no window manager on this box (imgui skill §0) — every aesthetic call is the maintainer's |

### The two things that FAILED first, and why they are rules, not footnotes

**1. `set_next_item_allow_overlap()` goes on the item submitted FIRST, not the one on top.**
The naive reading (put it on the node, so the node wins over the canvas) produced a canvas
where nodes were entirely dead: dragging a node panned the view instead. Isolated:

```
allow_overlap on bg = False  {'bg_hov': True,  'bg_act': True,  'node_hov': False, 'node_act': False}
allow_overlap on bg = True   {'bg_hov': False, 'bg_act': False, 'node_hov': True,  'node_act': True}
```

imgui gives an overlapping hit to the **earliest** submitted item unless that item declares it
may be overlapped. So the chain is: the canvas background declares `allow_overlap` → each node
body declares `allow_overlap` → the ports and any overlay button are submitted last and win.
Three levels, each one line, and the whole canvas is inert if any level is missed. The imgui
skill §3 states the rule for the strip's overlay-over-cell case; the canvas makes it a
three-deep chain, which is the part worth writing down.

**2. `begin_popup_context_item` with an explicit `str_id` fires on a right-click anywhere in
the window.** With `f"##menu_{node.name}"` passed, right-clicking empty canvas opened the
*node's* menu. The binding's own docstring says it: *"Use str_id==None to associate the popup
to previous item."* `pass_list.py::_draw_context_menu` passes `f"##pass_menu_{name}"` and gets
away with it because each tile is its own child window, so "anywhere in the window" is that
one tile. On a single canvas child holding every node, the explicit id is a bug. The graph's
node menus must pass `None`; the canvas menu is opened by hand:

```python
if bg_hovered and imgui.is_mouse_released(imgui.MouseButton_.right) \
        and not state.node_hovered and not state.port_hovered:
    imgui.open_popup("##canvas_menu")
```

### Three smaller binding facts the probe turned up

- `io.mouse_clicked_pos` is **not exposed** by this binding (`AttributeError`, the IO object
  has `mouse_clicked` only). The rubber band's anchor comes from
  `imgui.get_mouse_drag_delta(button, 0.0)` subtracted from `io.mouse_pos`, which is exposed.
- `io.fonts.build()` is **gone** in 1.92 (dynamic font loading); calling it is an
  `AttributeError`. Nothing in the repo calls it — noted so a probe author does not.
- A popup is **context-global and outlives the widget state**: a scenario that left a menu open
  tainted the next one until the probe dismissed it with a click far outside. There is no
  global "close every popup" in this binding. Relevant to the graph because the canvas menu
  and a node menu are two popups in one child.

---

## 3. The interaction table

`bg` = the canvas background `invisible_button`; `node` / `port` = the per-node and per-port
`invisible_button`s. "Funnel" names the existing App / ProjectSession entry point — the graph
adds **no new write path**, which is the strongest structural argument for it.

| gesture | what it does | imgui calls | existing funnel | conflict |
|---|---|---|---|---|
| **click a node** | pick the pass as output + open its shader tab | `is_item_clicked(MouseButton_.left)` on `node` | `App.pick_pass(document_id, name, focus_editor=False)` — exactly what `pass_list._draw_pass_tile` calls | none. Same verb, second surface |
| **click a box** (group) | pick the bundle's output pass | same | `App.pick_pass` with the bundle output's name | none |
| **click empty canvas** | clear the selection | `is_item_clicked` on `bg` | new widget-local state | none |
| **double-click a node** | open the pass in the editor and focus it | `is_item_hovered()` + `is_mouse_double_clicked(left)` | `App.pick_pass(..., focus_editor=True)` | **the single click fires too** — the probe saw `node_clicked='scene'` on the same press that produced `node_double='scene'`. Since both route to `pick_pass`, the double just adds focus; no latch needed, but the widget must not make single-click do something the double must undo |
| **double-click a box** | enter the group's tab | same on the box's `node` button | new: the graph's own tab state | none |
| **right-click a node** | Settings / Delete / Leave group / Group… | `begin_popup_context_item(None)` in `context_menu_style()` | `App.open_pass_settings`, `session.delete_pass` + `App.close_editor_for_path`, `session.set_pass_group(id, name, "")` | **`str_id` must be `None`** (§2). Reuse `pass_list._draw_context_menu`'s item set so the two surfaces cannot drift |
| **right-click empty canvas** | Add pass here / Arrange / Fit | `open_popup` + `begin_popup` gated on `bg_hovered and not node_hovered` | `App.open_add_pass`; the new pass takes the cursor's graph position | the hand-rolled gate is required, see §2 |
| **drag a node** | move it; write `position` on the pass entry | `is_item_active()` + `io.mouse_delta / zoom` | new write on `PassEntry` (brainstorm fixed item 5) | none. Write on mouse-release, not per frame — a per-frame `save_ui_document` on every drag frame would write the document file ~60×/s |
| **drag a box** | translate every member's position | same, applied to the member list | same write, N entries | none |
| **drag a wire from an output dot** | write `PassSource(name)` into the target sampler | press on `port`, `is_item_active()` holds it, `is_mouse_released` over another port drops it | `session.set_sampler_source(document_id, pass, uniform, PassSource(name))` — the same write the uniforms panel row makes | none. The probe confirms the source port keeps the drag even when released over empty canvas (pan stayed 0) |
| **drop a wire on empty space** | write `NoSource()` | the release with `wire_drop_on == ""` | `session.set_sampler_source(..., NoSource())` | none |
| **drag from an input port** | see Decision 2 — rewire or disconnect is unsettled | same mechanics, opposite end | same funnel | none mechanically |
| **rubber-band multi-select** | select every node whose rect intersects the band | `is_item_active()` on `bg` + `get_mouse_drag_delta` | new widget-local selection set | **collides head-on with pan**: the probe shows one and the same `bg` drag produces both a pan of `50,30` and a band of `(364,334)->(434,384)`. They must be split by button or modifier — Decision 1 |
| **shift-click a node** | add/remove from the selection | `is_item_clicked` + `io.key_shift` | selection set | none. `io.key_shift` is a plain read, no registry entry |
| **pan** | translate the view | `is_item_active()` on `bg` + `io.mouse_delta` | widget-local | see rubber band |
| **wheel zoom** | scale about the cursor | `io.mouse_wheel` inside the child + the before/after correction | widget-local | needs `WindowFlags_.no_scroll_with_mouse` on the child or the wheel scrolls it instead (measured: 416 px of scroll) |
| **fit** | set zoom+pan from the node bounding box | a button / menu item; pure arithmetic | widget-local | none |
| **arrange** | rewrite every `position` from the rank layout | a button / menu item | the `position` write, N entries, one save | none |
| **Delete** | delete the selected pass(es) | `imgui.shortcut` or `is_key_pressed` | `session.delete_pass` + `App.close_editor_for_path` (the pass-file/tab teardown `pass_list._delete_pass` already does) | **unbound today.** `Delete` is in `commands._BINDABLE_KEYS` but no spec claims it, and `chord_needs_modifier` would refuse a bare `Delete` as a *rebindable* chord (only F-keys are exempt). A canvas-scoped bare Delete therefore cannot go through the registry as written — Decision 4 |
| **Escape** | go up one group tab / clear the selection | — | `hotkeys._handle_escape` | **Escape has an owner and a strict priority ladder**: revert-confirm → any popup → palette → chat focus → editor caret. The graph is nowhere in it, and `App.escape_has_job()` gates the glfw filter that swallows a jobless Esc. A graph Escape needs a new rung *and* a new `escape_has_job` clause — Decision 4 |
| **arrows** | move the selection / nudge a node | — | — | **`Alt+Left` / `Alt+Right` are taken**: `CommandId.PREV_PASS` / `NEXT_PASS`, `C.DOCUMENT`, scope GLOBAL, routing through `App.step_output_pass` → `pick_pass`. On the graph these read naturally as "step the output pass", which is what they already do; leave them alone. Bare arrows have the same `chord_needs_modifier` problem as Delete |
| **Ctrl+Tab** | cycle code tabs | — | `App.cycle_code_tab` | `CommandId.CYCLE_CODE_TAB`, scope GLOBAL, and the comment on it is explicit that Ctrl+Tab is the app's because `nav_enable_keyboard` is off and `no_nav_focus` keeps it that way. If the graph becomes an `EditorTab.kind`, Ctrl+Tab cycles **onto** it — which is either the feature or the bug, Decision 5 |
| **copilot turn** | the whole canvas freezes | `imgui.begin_disabled(app.copilot_turn_active)` around the canvas, as `pass_list.draw` does | — | none, and it is one line. But note `begin_disabled` does **not** stop the draw list from painting, so the frozen canvas still shows live pass pictures, which is right |

Two cross-cutting notes the table cannot hold:

- **Every write goes through `ProjectSession`, which calls `save_ui_document` on each write.**
  `set_sampler_source`, `set_pass_group` and `set_output_pass` each save. A drag that writes a
  position per frame, or a "Group" verb that calls `set_pass_group` once per selected pass,
  is N document writes. Batch at the gesture's end.
- **A cycle-refusing drop needs the pure planner, not a try/write/undo.** `pass_graph.plan_passes`
  takes a `Wiring` and returns `GraphError`s; build the hypothetical wiring dict, plan it,
  refuse before calling `set_sampler_source`. This is 091's pattern and it costs nothing.

---

## 4. The tab question: a third `EditorTab.kind` vs. a row inside the Document tab

### What the code actually requires

`EditorTab` is `(path: Path, kind, document_id)` and **`path` is the identity everywhere**:

- `App._focus_or_add_tab` dedupes on `existing.path == tab.path`.
- `tabs/code.py::_tab_id_suffix` returns `f"##{tab.path}"` — the imgui id, deliberately not the
  index, so drag-reorder survives.
- `App.is_tab_dirty(tab)` looks up `self.editor_sessions.get(tab.path)`.
- `App.close_editor_for_path(path)` is the teardown funnel, keyed by path.
- `code.draw` resolves `session = app.editor_sessions.get(current_path)` and, when absent,
  **creates one** (`open_shader_lib_file` or `get_session_for_path`) — a libeditor instance
  bound to a file on disk.

So a graph tab keyed by a group path (`""` for the root, `"bloom"` for a group) has to answer
four questions the record does not currently have room for:

| what the record needs | for a graph tab |
|---|---|
| **the key** | a group path, which is not a `Path` on disk. Either `path` stops being a filesystem path (and `_focus_or_add_tab`, `_tab_id_suffix`, `close_editor_for_path` all keep working by accident because they only ever compare and format it), or the record grows a second identity field and every keyed call site learns which one to use |
| **`is_tab_dirty`** | reads `editor_sessions[path]`, which will never hold a graph tab → always False. That is the right answer (the graph has no buffer) but it is right by accident, not by design; a future `editor_sessions` keyed differently would silently make a graph tab dirty |
| **close** | `App.close_tab` pops the tab and reanchors. Fine as-is. But `Ctrl+W` (`CLOSE_CODE_TAB`, scope EDITOR) is gated on `app.editor_focused`, which a graph tab never sets — so a graph tab could not be closed by keyboard, only by its ✕ |
| **`tab_label`** | the function branches on `kind` for `"lib"` / `"script"` / else-shader, deriving from `pass_name_of(tab.path)`. A graph tab needs a fourth branch and a label that is not file-derived (`"<document> (graph)"`, or the group path for a group tab) |

And the body: everything in `code.draw` after `_draw_tab_row` — the session fetch, the
read-only lock, the error strip, the markers, the completion plumbing — assumes a buffer. A
graph tab means an early branch right after `_draw_tab_row`, before `session = ...`. That is
one `if`, but it is an `if` in the hottest, most-hand-tuned function in the UI.

### What the pane swap needs

The maintainer's planned zen mode / pane swap is the strongest argument *for* the tab-kind
route and also the one thing neither route settles: whichever it is, the canvas must be a
**pane-agnostic widget** — a free function taking the document, a position store, and a
view state (pan/zoom/selection/current group), drawing into whatever `begin_child` it is
handed. `ui.py`'s left/right split already gives each side a child of a computed size, so the
widget only needs to not assume its own width. Write it that way and the tab question becomes
*where it is mounted*, reversible in an afternoon, rather than a fork in the design.

### Recommendation

**Start as a row inside the Document tab; keep the widget pane-agnostic so the tab-kind move
stays cheap.** Three reasons, in order of weight:

1. The Document tab already sits in a `begin_tab_bar` in `_draw_document_settings`, and
   `ui_primitives.text_tab_row` is the lighter color-only selector the Uniforms tab uses (`tabs/uniforms.py`) — the
   graph's own root/group tabs are exactly that shape, and nesting a `text_tab_row` inside the
   Document tab needs no change to `EditorTab`, `is_tab_dirty`, `tab_label`, `close_tab`, or
   `code.draw`. Zero edits to the editor's tab machinery, which is the part with a
   `select_pending` latch, a display-order reconciliation, and a comment explaining why the id
   is the path.
2. The strip lives there too. A second view of the same thing belongs beside the first, and
   091's "the strip stays as it is" means the two are siblings, not replacements.
3. The tab-kind route is not blocked, just not first. Nothing in staging (1) read-only canvas
   or (2) drag and wiring depends on it. When zen mode arrives and the canvas wants the whole
   editor pane, the widget moves and `EditorTab` grows its fourth kind then — with the canvas
   already proven.

The cost of being wrong in this direction is small (the Document tab is narrower — the app
panel can go down to 360 wide, and the graph wants room). The cost of being wrong in the other
direction is edits to `EditorTab`'s identity contract before anything is drawn.

---

## 5. Decisions the maintainer must make

**1. Pan and rubber-band are the same gesture. Which button does which?**
The probe shows one left-drag on empty canvas producing both. Three shapes:

```
A  left-drag on empty = rubber band;  middle-drag or space+drag = pan
   (Blender/Figma-ish; costs a middle button this repo never uses)
B  left-drag on empty = pan;          shift+left-drag = rubber band
   (one button, one modifier; pan is the commoner verb so it gets the bare drag)
C  left-drag on empty = rubber band;  left-drag anywhere + Alt = pan
   (Alt is already the app's modifier of choice — but Alt+drag is not a chord
    the registry sees, so no conflict either way)
```

**2. Dragging from an INPUT port: rewire or disconnect?**
Fixed item 6 only names the output→input direction.

```
A  grab-and-carry: pressing a filled input port detaches the wire and you now drag
   its loose end; releasing on another input moves it, releasing on empty writes
   NoSource on the ORIGINAL. One gesture does move and disconnect.
B  input ports are drop targets only; a wire is removed by right-click > Disconnect
   on the port, or by dropping a new wire on it. Fewer gestures, more menu.
C  both: drag carries, and the port menu also has Disconnect.
```
Worth noting against A: fixed item 6 says a drop on empty space writes `NoSource`, which
already gives A its disconnect for free.

**3. Merged ports on a box (round 3 B) vs one port per slot (round 3 A).**
This is an interaction question, not only a picture: with merged ports, one drop rewrites
every slot behind it, so a user cannot wire `fx_bright` and `fx_blur` to different sources
without entering the group. With per-slot ports, the bloom box shows three `scene` wires at
the root, which is what the picture honestly is. The mock's E case (Radiance Cascades imported
whole) is the one where merging clearly wins — three members reading `paint` fold to one port.

**4. Delete and Escape on the canvas — how do they reach the registry?**
Both have a problem today.

- `Delete`: `commands.chord_needs_modifier` refuses a bare non-F key, so `CommandId` cannot
  carry a bare `Delete`. Options: (a) bind `Alt+Delete` / `Alt+X` as a normal registry command
  in `C.DOCUMENT`; (b) read `is_key_pressed(Key.delete)` inside the canvas widget gated on
  "the canvas is hovered and a node is selected", which is off-registry and invisible to the
  cheatsheet and the rebinder; (c) no keyboard delete — the context menu only, as the strip
  has it today.
- `Escape`: `hotkeys._handle_escape` is an ordered ladder and `App.escape_has_job()` gates the
  glfw filter that swallows a jobless Esc. "Escape goes up a group tab" (fixed item 3) needs a
  rung in that ladder *and* a clause in `escape_has_job`, or the press is swallowed before
  imgui ever sees it. Where in the ladder: after the popups, before the chat focus, is the
  natural place — but it means Escape in a focused editor still goes to vim, which is right.

**5. If the graph ever becomes an `EditorTab.kind`, does Ctrl+Tab cycle onto it?**
`CYCLE_CODE_TAB` is scope GLOBAL and its handler focuses an unfocused editor first. A graph
tab in that rotation means Ctrl+Tab can land on a surface with no buffer, and `App.cycle_code_tab`
focuses the editor as part of the cycle. Either the graph tab is skipped by the cycle, or
`cycle_code_tab` learns that some tabs are not editors. Not urgent under the §4 recommendation,
but it is the thing that decides whether the tab-kind move is one `if` or three.

**6. When does a drag write to disk?**
Every `ProjectSession` write calls `save_ui_document`. Proposed as a constraint rather than a
question, unless the maintainer disagrees: positions are written **once on mouse release**,
and a multi-pass verb (Group, Arrange, Dissolve) writes **once for the whole set**. That means
either a batched variant of `set_pass_group` / a position write, or a save-suppressing scope.

---

## 6. False trails

Things that look like the answer and are not. Each was tried or checked against the binding.

- **"Put `set_next_item_allow_overlap()` on the node so it wins over the canvas."** Backwards;
  it goes on the canvas. This one silently produces a canvas where nodes are completely dead
  and every drag pans, which reads like a coordinate bug and is not.
- **"Give each node its own `begin_child`, like `preview_cell` does."** Fixed item 1 already
  rules it out for the zoom reason, and the probe confirms the reason is real: the node's text
  is drawn with `push_font(font, 14 * zoom)` and the picture with `add_image_rounded` at
  arbitrary scale, neither of which a child window's fixed-size content can do. The strip's
  per-cell child exists to stop overlay buttons perturbing the parent's content size (imgui
  skill §3); on the canvas nothing flows, every position is absolute inside one child, so the
  jitter the child was defending against cannot arise.
- **`imgui.begin_drag_drop_source` for wires.** It exists in the binding (with
  `set_drag_drop_payload_py_id` / `accept_drag_drop_payload_py_id`), and it is the wrong tool:
  it owns the visual feedback (a tooltip-shaped preview follows the cursor), which is not a
  bezier ending at the cursor, and it wants a source *item*, which fights the
  press-and-hold-the-port model that the probe shows working in nine lines. Use
  `is_item_active()` + `is_mouse_released()`.
- **`push_clip_rect` on the canvas child.** Unnecessary — the probe shows the child's own clip
  rect at `(9,9)->(607,507)` for a 600-wide child, applied to both the drawing and the hit
  testing (a node at graph x=700 answered no hover). `push_clip_rect` is still needed for the
  foreground-list case `pass_list._draw_group_outline` uses, which the graph does not have
  because nothing on the canvas is a child window painting over its parent.
- **"Read `io.mouse_wheel` and let imgui scroll too."** Measured: without
  `no_scroll_with_mouse` the child scrolls 416 px on the same wheel the zoom handler reads, so
  the canvas both zooms and slides. The flag is not optional.
- **`io.mouse_clicked_pos` for the rubber-band anchor.** Not exposed by this binding at all.
- **`is_mouse_hovering_rect` for node hover instead of per-node buttons.** The skill §3 already
  warns it ignores window ordering and popup blocking, and it would also mean hand-rolling the
  active/click/double-click state the probe gets free from `invisible_button`.
- **"A fractional font size will look blurry, so snap zoom to integer steps."** 1.92 bakes a
  real face per size (`get_font_baked().size` tracked the ask at every zoom tested). Zoom can
  be continuous; only the *size* rounds to an integer, which is a 1-pixel quantization of the
  text, not a blur. Snapping zoom is a legitimate choice for other reasons — it is not forced
  by the font system.
- **Reusing `pass_list._draw_context_menu` verbatim.** Tempting and nearly right, but it passes
  an explicit `str_id`, which is safe only because each tile is its own window. Extract the
  *item set* into a shared function, not the `begin_popup_context_item` call.

---

## 7. Verdict

**PASS.** Every gesture in the round-3 interaction surface was driven in a real imgui frame on
this exact stack and behaved: node drag correct under zoom, cursor-anchored wheel zoom to
within 6e-14 px, per-node and per-port hit testing with correct priority against a full-canvas
background button, a context menu anchored on a node, double-click, wire drag from port to
port with a live bezier, a rubber band, clipping of both drawing and hit testing at the child's
edge, crisp text at every zoom, and a rounded blit of a real moderngl texture. No node-editor
library is needed and none of it is exotic — the entire canvas is `invisible_button`,
`is_item_active`, `io.mouse_delta`, `io.mouse_wheel`, and one draw list.

Two qualifications on the PASS, neither of which is a stack limit:

- **A 7 px port dot is 3.5 px at zoom 0.5.** imgui hits it (measured), a human will not. The
  port hit box should have a floor in *screen* pixels rather than scaling all the way down —
  a one-line `max()` on the radius used for the button, while the drawn dot keeps scaling.
- **The keyboard is where the friction is, not the mouse.** Delete and Escape both collide with
  existing structure (`chord_needs_modifier` refuses a bare Delete; Escape's ladder and
  `escape_has_job` gate have no graph rung), and those are the two bindings the canvas most
  wants. Decision 4 is the one that has to be answered before staging (2) rather than during
  it — everything else on the list can be decided while the read-only canvas is being built.
