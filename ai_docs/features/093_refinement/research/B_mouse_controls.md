# Area B — mouse control schema

Research for feature 093. Answers questions 6–8 of `02_research_brief.md` against named
primary sources, then a recommendation for ShaderBox's canvas
(`shaderbox/widgets/pass_graph.py::_draw_canvas`).

## Sources

| Name | What it is | URL / path | What was read |
| --- | --- | --- | --- |
| thedmd/imgui-node-editor | Dear ImGui node-editor widget (C++), the reference the maintainer explicitly named | `imgui_node_editor.cpp`, `imgui_node_editor.h`, `imgui_node_editor_internal.h` (cloned shallow to scratchpad) | code |
| Nelarius/imnodes | Minimal Dear ImGui node-editor widget | `imnodes.cpp`, `imnodes.h` (cloned shallow) | code |
| xyflow/xyflow | React/Svelte Flow, the most-used web node-graph library | `packages/system/src/xypanzoom/filter.ts`, `packages/system/src/xydrag/XYDrag.ts`, `packages/react/src/container/{ReactFlow,FlowRenderer,Pane}/index.tsx` (cloned shallow) | code |
| jagenjo/litegraph.js | Standalone JS node-graph canvas (used by ComfyUI and others) | `src/litegraph.js` (`processMouseDown`, `processMouseMove`, `processKey`) (cloned shallow) | code |
| Fattorino/ImNodeFlow | Dear ImGui node-editor widget, smaller than imgui-node-editor | `src/ImNodeFlow.cpp`, `src/context_wrapper.h` (cloned shallow) | code |
| Blender Manual — node editor: Selecting | Official docs | `docs.blender.org/manual/en/latest/interface/controls/nodes/selecting.html` | docs |
| Blender Manual — node editor: Editing | Official docs | `docs.blender.org/manual/en/latest/interface/controls/nodes/editing.html` | docs |
| Unreal Engine — Blueprint Editor Cheat Sheet | Official docs, the canonical binding table for UE's graph editor (shared by Blueprints and the Material graph) | `dev.epicgames.com/documentation/en-us/unreal-engine/blueprint-editor-cheat-sheet-in-unreal-engine`, corroborated against the archived static page `web.archive.org/web/2023/https://docs.unrealengine.com/5.3/en-US/blueprint-editor-cheat-sheet-in-unreal-engine/` | docs |
| SideFX Houdini — Network editor shortcuts | Official docs | `www.sidefx.com/docs/houdini/network/shortcuts.html` | docs |
| SideFX Houdini — Connecting (wiring) nodes | Official docs | `www.sidefx.com/docs/houdini/network/wire` | docs |

**A note on how the UE and Houdini bindings were verified.** `dev.epicgames.com` renders its
docs client-side; a raw fetch returns an empty shell. The cheat-sheet content quoted below was
read from `web.archive.org`'s static snapshot of the pre-migration `docs.unrealengine.com`
page, which serves the same table server-rendered, and cross-checked against WebFetch's render
of the live page — the two agree line-for-line. The Houdini shortcuts table encodes its mouse
buttons as `<img>` icons (`LMB.svg`, `MMB.svg`, `RMB.svg`, `mouse_wheel.svg`) rather than text;
the pan/zoom row buttons below were read from those icon `title` attributes in the raw HTML, not
guessed from a stripped-icon text dump.

## Q6 — full binding table per reference

Blank cell = the reference does not define that action (a widget library leaves it to the host
app, or a full editor genuinely has no such gesture).

| Action | imgui-node-editor | imnodes | xyflow | litegraph.js | ImNodeFlow | Blender | Unreal Blueprint | Houdini |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pan | RMB drag (`NavigateButtonIndex=1`, `NavigateAction::Process`) | MMB drag, or Alt+LMB if `EmulateThreeButtonMouse` enabled (`BeginCanvasInteraction`) | LMB drag on empty pane (`panOnDrag=true` default); MMB drag always works over a node/edge regardless of `panOnDrag` (`xypanzoom/filter.ts`); Space+drag forces pan (`panActivationKeyCode='Space'`) | LMB drag on empty background (`clicking_canvas_bg`); MMB drag also pans (`processMouseDown`, `e.which==2`) | MMB drag (`scroll_button = ImGuiMouseButton_Middle`, `context_wrapper.h`) | (2D-canvas convention shared across Blender editors; not repeated on the node-editor doc page — MMB drag pans, no node-editor-specific override found) | RMB drag (cheat sheet: "Pan the Graph — RMB Drag") | Space+LMB drag, or MMB drag without Space (`network/shortcuts.html` View keys table) |
| Zoom | Mouse wheel about cursor (`NavigateAction::HandleZoom`) | Mouse wheel (`AltMouseScrollDelta`) | Mouse wheel, or Ctrl/Meta+wheel is reserved for pinch-safety unless `zoomActivationKeyCode` held (`zoomOnScroll`, `zoomActivationKeyCode='Control'/'Meta'`) | Mouse wheel | Mouse wheel (scale change on scroll, `context_wrapper.h`) | Scroll wheel (2D-canvas convention) | Mouse wheel, or Hold LMB+RMB and drag; Ctrl+Zoom to go beyond 1:1 (cheat sheet) | Space+RMB drag, or scroll wheel without Space |
| Select (click) | LMB click sets selection, Ctrl+click toggles (`SelectAction::Accept`) | LMB click; multi-select modifier (default Ctrl) adds (`ImNodesIO.MultipleSelectModifier`, default `NULL`→`io.KeyCtrl`) | LMB click; `multiSelectionKeyCode` (default `Control`/`Meta`) adds | LMB click, `processNodeSelected` | LMB click (`context_wrapper.h` L163-165) | LMB click; Shift+LMB adds (`selecting.html`) | LMB click | LMB click; Shift+LMB adds, Ctrl+LMB removes |
| Rubber band | LMB drag on empty background, `SelectButtonIndex=0` same button as select-click (`SelectAction::Accept`) | LMB drag on empty background (`BeginCanvasInteraction`, `ClickInteractionType_BoxSelection`) | Opt-in `selectionOnDrag`; when off, Shift held while LMB-dragging switches to select (`selectionKeyCode='Shift'`) | **Ctrl+LMB drag** — a distinct chord from plain pan-drag (`processMouseDown`, `e.ctrlKey` branch, `dragging_rectangle`) | not built in (widget-library scope; host app decides) | `B` tool / toolbar box-select tool, not bare LMB-drag (`selecting.html`: "B — Click and drag to select nodes within a rectangular region") | LMB drag (replace), Shift+LMB drag (add), Ctrl+LMB drag (remove) (cheat sheet) | LMB drag; `S`+drag locks the gesture to selection when there isn't enough empty space (`network/shortcuts.html`) |
| Node drag | LMB drag on node body once active/hovered (`DragAction::Accept`, `DragButtonIndex=0`) | LMB drag on node (`BeginNodeSelection`) | LMB drag on node, gated by `nodeDragThreshold` (default 1px) (`XYDrag.ts`) | LMB drag on node body (`processMouseDown`, `this.node_dragged`) | LMB drag, but only starting from the node **header** rect, not the whole body (`context_wrapper.h` L178-183, `onHeader`) | LMB drag (grab, `G`) | LMB drag | LMB drag |
| Wire from output | LMB drag from a pin (`CreateItemAction::Accept`, `control.ActivePin`) | LMB drag from a pin (`BeginLinkCreation`) | LMB drag from a Handle (xyhandle) | LMB drag from an output dot (`connecting_output`) | LMB press on a hovered pin sets `m_dragOut` (`ImNodeFlow.cpp` L301-302) | LMB drag from a socket | LMB drag from a pin (cheat sheet: "Connect to Another Pin — Left-Click + Drag to Pin") | Drag between two connectors, or click one then the other (`network/wire`) |
| Re-plug from a filled input | Grabbing a link end and redropping it is native to `CreateItemAction` (`DragStart`/`DropPin`) — dragging the existing link's free end | Only with `ImNodesAttributeFlags_EnableLinkDetachWithDragClick` set on that attribute (opt-in per pin, `BeginLinkInteraction`) | `reconnectable` edges + `onReconnect`; dragging a filled handle re-grabs that edge's free end (xyhandle) | Press on a filled input re-grabs the wire at the **producer** end, freeing the input end to redrop, gated by `allow_reconnect_links` (default true) or holding Shift (`processMouseDown` L6134-6154) | not modeled (only fresh drags from an output pin) | Interactively: "Drag the link away from its input socket and let it go" (disconnects; re-plug is the same drag redirected to a new socket before release) (`editing.html`) | Ctrl+LMB drag on a pin moves all its connections (cheat sheet: "Move All Connections — Ctrl + LMB Drag to Pin") | Click the connected input, then click empty space to disconnect; or click-drag it elsewhere to re-plug (`network/wire`) |
| Disconnect (drop on empty) | Drop-on-empty from a grabbed link cancels the reconnect and removes it (`CreateItemAction::Process`, `DropNothing`) | Same: drop with no hovered pin ends the detached link | Drop on empty removes the edge unless `onReconnect` supplies a replacement | Drop on empty from a re-grabbed wire leaves it disconnected (implied by `connecting_node`/`connecting_input` flow) | Drop on empty clears `m_dragOut` with no link created (`ImNodeFlow.cpp` L288-312) | Same interactive drag-to-empty-and-release | Filtered action menu for the pin opens at the drop point ("Left-Click + Drag to graph — Filtered Action Menu for Pin") — not a plain disconnect-and-cancel | n/a (click-based, not drag-and-drop-to-empty) |
| Delete | Bare `Delete` key deletes the selection when it is non-empty (`DeleteItemsAction::Accept`, `ImGuiKey_Delete`, no modifier); Alt+click a link deletes just that link | not built in (widget-library scope) | `deleteKeyCode` default `'Backspace'` (not Delete) | Bare `Delete` or `Backspace` (keyCode 46 or 8) deletes the selection (`processKey`) | Bare `Delete` key deletes a selected node, gated on window focus + no active item (`context_wrapper.h` L175-176) | `X` or `Delete` (menu: Node ‣ Delete); Ctrl+Delete deletes-with-reconnect (`editing.html`) | `Delete` key (cheat sheet: "Delete Selected Nodes — Delete") | Select a wire and press Del; nodes have no single bare-key delete documented on this page (deletion is via menu/toolbar, "Shake node" only disconnects) |
| Duplicate | Ctrl+D (`ShortcutAction::Accept`, `GetKeyIndexForD()`) | not built in | not a system-level primitive (host app's job) | Alt+drag clones (`LiteGraph.alt_drag_do_clone_nodes`, `processMouseDown` L6007) | not built in | Shift+D (duplicate), Alt+D (duplicate linked) (`editing.html`) | not in the cheat sheet's Graph Actions table (standard Ctrl+C/Ctrl+V copy-paste covers it) | Alt+drag node (`network/shortcuts.html`) |
| Context menu | RMB click/release, arbitrated against pan by a drag-vs-click distinction (`ContextMenuAction::Accept`, `ContextMenuButtonIndex=1`) | not built in (host draws its own popup on hover+RMB) | not a system-level primitive | RMB (`e.which==3`), extended with Shift/Ctrl to pre-seed multi-node menus | Configurable RMB popup (`m_rightClickPopUp`, `ImGui::IsMouseClicked(ImGuiMouseButton_Right)`) | RMB (standard Blender menu convention) | RMB on a node, or RMB on the graph for the graph action menu (cheat sheet) | RMB on a wire opens a wire-specific menu (`network/wire`); RMB elsewhere is the standard node/graph menu |

## Q7 — conflict resolution per reference

| Conflict | imgui-node-editor | imnodes | xyflow | litegraph.js | ImNodeFlow | Blender | Unreal Blueprint | Houdini |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pan vs. band on empty space | Different buttons (RMB vs LMB) — no collision | Different buttons (MMB vs LMB) — no collision unless emulate-3-button is on, in which case Alt disambiguates | Same button by default (`panOnDrag=true`); `selectionOnDrag` is refused whenever `panOnDrag===true` (`_selectionOnDrag = selectionOnDrag && panOnDrag !== true`) — the library makes them mutually exclusive by construction, never both live on LMB at once | Different chords: plain LMB pans, Ctrl+LMB bands | n/a (no band) | Different mechanisms: LMB click-selects, `B` activates a distinct box-select tool/mode | Same button (LMB) with an explicit priority: plain LMB-drag from empty space always starts a box select; pan is on the *other* button (RMB), so there is no same-button collision to arbitrate | Different buttons/chords: `Space+LMB` (or MMB) pans, plain LMB drag on empty bands, `S`+drag force-locks to band |
| Click vs. drag threshold | `IsMouseDragging(button, lock_threshold=1)` — 1 px once the item is captured active by imgui's own hover+press gate (dear imgui's own default click threshold is 6 px, `imgui.cpp` `MouseDragThreshold = 6.0f`, but the node-editor actions override it to 1 px for drag detection after activation) | Uses imgui's default drag detection (no override found) — 6 px | `nodeClickDistance` (d3-drag) separates click from drag; `nodeDragThreshold` (default 1) is measured in **client pixels**, explicitly independent of zoom ("distance in client coordinates for consistent drag threshold behavior across zoom levels", `XYDrag.ts`) | No explicit threshold constant found; drag state flips on any `mousemove` after `mousedown` while a node/bg is the down-target | Relies on imgui's default 6 px item-active drag detection | Not numerically documented (Blender's `tweak` threshold is a user preference, not node-editor-specific) | Not numerically documented | Not numerically documented |
| Press on a port vs. the node body under it | Port ("pin") hit-testing runs and is checked before node drag in `CreateItemAction`'s priority slot, which is tried before `SelectAction`/rubber-band but the two are mutually exclusive per frame via the `m_CurrentAction` dispatcher (ContextMenu > Shortcut > Size > Drag(node) > CreateItem(wire) > DeleteItems > Select) — **note Drag(node) is offered before CreateItem(wire)**, but `DragAction::Accept` only fires when `control.ActiveObject` is the node itself, and imgui's own item system already resolved which invisible-button (the smaller pin button or the larger node button) captured the click, so the port button wins on overlap by z-order/submission-order, not by this priority list | Port hover (`GImNodes->HoveredPinIdx`) is computed before node hover in the same frame; `BeginLinkInteraction`/`BeginLinkCreation` fires whenever a pin is the interaction target, node-drag never starts on a pixel that hit a pin | Handle (pin) elements are separate DOM nodes stacked above the node body with their own pointer-down handlers (`xyhandle`); a pin press never reaches the node's drag handler because the DOM event doesn't bubble to a drag start once handled | Explicit ordered hit-test in `processMouseDown`: output slots checked first, then input slots (`isInsideRectangle` around each `getConnectionPos`), and only if no slot matched does the code fall through to "clicking on top of a node" (resize corner, then title bar drag, then body) | Port press is checked first in `update()` (`m_dragOut` assignment happens before the node-header drag branch is reached) | n/a (declarative in Blender's own hit-testing, not documented at this level) | n/a (documented as behavior, not implementation) | n/a |
| Drop on empty from a wire | Cancel (no link created), `DropNothing()` — this is a bare cancel, not a disconnect, because `CreateItemAction` doesn't know if the drag started from a fresh output (nothing to disconnect) or a detached existing link (already unlinked the moment the drag started) | Same: `EndLinkCreation` with no target simply ends the drag; if it was a detach-drag the link is already gone (`GImNodes->DeletedLinkIdx` was set at drag start) | Removes the edge (if it was a `reconnectable` edge being dragged) or does nothing (if it was a fresh connection attempt with no `onConnectEnd` handler creating a node) | Disconnects if it was a redrag of an existing wire (the wire was already detached from the input at press time); creates nothing if it was fresh from an output | Cancel — `m_dragOut = nullptr` on release with no hover target, no link created (`ImNodeFlow.cpp` L288-312) | Disconnect (interactive drag-away is *the* documented way to disconnect a socket) | Opens a filtered "create node from this pin" action menu at the drop point — not a plain cancel or disconnect | n/a (click-based reconnection, no drag-to-empty gesture documented) |
| What detaches a plugged wire | Re-grabbing the wire's end (drag it off the port) is the native detach; there is no separate modifier | `LinkDetachWithModifierClick` (an opt-in modifier, unset by default) turns clicking a link near a port into an instant detach; `EnableLinkDetachWithDragClick` (opt-in per-pin flag) turns pressing a filled input into a detach-and-redrag in one gesture | `reconnectable`/`onReconnect` semantics — no modifier key; the affordance is built into which edges declare themselves reconnectable | Grabbing a filled input's wire (LMB press on that port) always re-grabs it at the producer end; **Shift held is only needed when `allow_reconnect_links` is false** (default true, so ordinarily no modifier is needed at all) | Not modeled (no detach-from-input gesture in this widget) | **Alt+LMB drag** on a node — "Detach Links" — cuts all links on the selected node(s) and starts moving them (a node-level bulk detach, distinct from per-socket dragging); a single link is discononnected the same way as connecting: drag it off its socket | **Alt+Left-Click on a pin** breaks *all* connections on that pin instantly (no drag needed); **Ctrl+LMB drag on a pin** moves all its connections to a new pin | Grabbing the input's existing connection and clicking empty space (or dragging it elsewhere) detaches it; no modifier key needed |

## Q8 — convergence and divergence

**CONVERGENCE.** Every reference that draws wires from a filled input treats "press the filled
input" as "start dragging that wire's free end," not as "start dragging the node" — the port
hit-test always runs before the node hit-test, and the two never compete for the same click
(imgui-node-editor, imnodes, litegraph.js, ImNodeFlow, Houdini, Unreal's Ctrl+drag). Every
2D-canvas widget library (imnodes, xyflow, litegraph.js, ImNodeFlow) defaults pan to a **button
other than plain left-drag** — middle-mouse is the near-universal default, and where left-drag
does pan (xyflow, litegraph.js) the library either makes rubber-band mutually exclusive with it
by construction (xyflow) or puts rubber-band on an explicit different chord (litegraph.js's
Ctrl+LMB). No reference lets plain left-drag mean both "pan" and "band" on the same press — that
ambiguity is resolved either by button, by modifier, or by making the two options structurally
exclusive, never by a runtime heuristic guessing intent. Click-vs-drag is universally a small,
fixed pixel threshold measured in **screen space, independent of zoom** (xyflow says so
explicitly; the imgui-based ones inherit this from imgui's own `IsMouseDragging`, whose distance
check operates on raw mouse-delta pixels). Delete, where a reference defines it as a canvas
action at all, is a **bare, unmodified key** — every source that binds delete at all
(imgui-node-editor, litegraph.js, ImNodeFlow, Blender's `X`/`Delete`, Unreal's `Delete`) uses it
without a modifier, gated instead on "the canvas/node has focus or hover" and "something is
selected," never registered through a global rebindable-chord system. Context menus are RMB
everywhere they exist, arbitrated against pan/other RMB actions by click-vs-drag (a still RMB
opens the menu; a dragged RMB is something else, e.g. Unreal's zoom-by-LMB+RMB or Blender's
Ctrl+RMB cut).

**DIVERGENCE.** The point of genuine disagreement is *how much of a special affordance detaching
a wire gets*. Widget libraries built for embedding (imnodes, ImNodeFlow) leave detach either
unimplemented or opt-in per attribute, pushing the decision to the host app. Full editors add a
dedicated, discoverable detach gesture on top of "drag the wire off the port": Blender adds
Alt+drag (node-level bulk detach) alongside Ctrl+RMB wire-cutting; Unreal adds Alt+click
(instant, no drag) and Ctrl+drag (move, not detach) as pin-level shortcuts; Houdini adds a
"shake the node" gesture and a held-`Y` wire-cut-by-crossing gesture, both novel to Houdini and
absent from every other reference. litegraph.js sits in between: re-plugging a filled input is
the *default* behavior of a plain press (no modifier), which is the most permissive of all the
references — closest to ShaderBox's own current behavior. The other divergence is rubber-band's
relationship to plain left-drag: imgui-node-editor and litegraph.js keep pan off left-button
entirely (so band can own left-drag on empty space outright); xyflow instead defaults to
*panning* on left-drag and treats band as the opt-in, secondary behavior — the opposite instinct
from a "canvas-editor-first" library, explained by xyflow's flowchart/diagramming lineage where
users pan far more than they select-and-arrange. Blender's `B`-key tool-based box select is the
outlier: it is not a drag-vs-drag conflict at all, because Blender treats "select" and "box
select" as distinct *tools/modes*, not as two interpretations of the same gesture — a model
available to us only if the canvas grew an explicit tool-mode concept, which it does not have and
should not add for this feature.

## Recommended for ShaderBox

Numbers are relative to our existing constants (`SIZE.GRAPH_PORT_ROW=16`, `SIZE.GRAPH_PORT_R=4`,
`SIZE.GRAPH_HIT_MIN=7`, `SIZE.GRAPH_SNAP_PX=6`, `SIZE.GRAPH_ZOOM_MIN/MAX=0.25/2.5`,
`_ZOOM_STEP=1.1`) so a coder can implement this without touching `theme.py`'s unrelated tokens.

| Action | Rule |
| --- | --- |
| **Pan** | Keep: middle-drag or Alt+left-drag. This is the convergent default (imnodes, litegraph.js, ImNodeFlow all default to middle-drag; Houdini's `Space+LMB` is the same "hold a key, then plain-drag" shape our Alt+drag already is). No change. |
| **Zoom** | Keep: wheel about the cursor, `zoom *= 1.1**wheel_delta`, clamped to `[0.25, 2.5]`. Universally convergent (every reference zooms on wheel, about the cursor where documented). No change. |
| **Select (click)** | Keep: LMB click sets selection (clears unless Shift held), matching every reference's plain-click behavior. No change. |
| **Rubber band** | Keep: LMB drag starting on empty canvas, not already claimed by pan. This matches imgui-node-editor and litegraph.js's instinct (band owns left-drag because pan lives elsewhere) rather than xyflow's (pan owns left-drag) — correct for us because our pan is already off left-button. No change; this is the convergent choice given our existing pan binding, not xyflow's. |
| **Node drag** | Keep: LMB press-and-drag on the node body (any unfilled-port or body press), promoted from a click once past the drag threshold. Do **not** adopt ImNodeFlow's header-only drag restriction — our cards have no distinct header region and adding one is a layout change, out of scope for area B. |
| **Wire drag from output** | Keep: LMB press-and-drag from an output dot. Convergent across every reference. |
| **Re-plug from a filled input** | Keep exactly as-is: a plain LMB press on a filled input re-grabs that wire at the *producer* end, no modifier required. This is litegraph.js's default-permissive model (`allow_reconnect_links=true`), not Blender/Unreal's modifier-gated one — correct for us because our canvas has no competing "click a filled port to do something else" behavior that a modifier would need to disambiguate from. |
| **Disconnect (drop on empty from a wire)** | Keep: drop on empty either cancels (fresh drag from an output, nothing existed to remove) or disconnects (re-grabbed from an input, already detached at press time) — this is exactly the imgui-node-editor / imnodes / ImNodeFlow model (`DropNothing` vs. an already-severed link), already what `_drop()` in `pass_graph.py` implements. Do **not** adopt Unreal's "open a filtered create-node menu on drop" — that requires a searchable node-creation palette keyed by pin type, which is a much larger feature than area B and not something any of our closer references (the imgui-family ones) do either. |
| **Delete** | **Change**, but narrowly: add a bare `Delete`/`Backspace` key read **locally in `_draw_canvas`**, gated on `hovered` (the canvas child is hovered this frame) and `view.selection` non-empty — mirroring ImNodeFlow's `IsKeyPressed(ImGuiKey_Delete) && !IsAnyItemActive() && isSelected()` and litegraph.js's `keyCode==46 or 8`. This is deliberately **not** a `commands.py` registry chord: `chord_needs_modifier` exists precisely because a bare non-F key fired globally would eat `Delete`/`Backspace` while a text field elsewhere has focus, and every reference that binds delete at all does it the same way we'd do it here — locally, gated on focus/hover, not through a global keymap. The existing context-menu Delete (`pass_menu_items`) stays as the discoverable, mouse-only path; the key is an accelerator for it, not a replacement. Route it through the same `App` verb the menu item uses so both paths stay identical in effect. |
| **Duplicate** | No change recommended — not in scope for area B's current feature set (no duplicate concept exists yet on the canvas), and it's the one action where references most disagree on mechanism (Ctrl+D vs Shift+D vs Alt+drag-clone vs none). Leave as a future decision once a duplicate feature is actually designed. |
| **Context menu** | Keep: RMB release on empty canvas or on a node opens `_canvas_menu`/`_node_menu`. Convergent everywhere. |
| **Fit** | Keep as a menu item (`_canvas_menu`'s "Fit"). No reference's *default* binding demands a dedicated key for this in a domain like ours (Houdini's `H` and Unreal's `Home` are real precedents *for a key*, but both are full DCC/engine editors with a much denser hotkey surface than we want to introduce here) — defer a keybinding until the maintainer asks for one. |
| **Arrange** | Keep as a menu item (`_canvas_menu`'s "Arrange"). Same reasoning as Fit — Houdini's `A`+drag auto-layout is the only close precedent and it is a drag-gesture, not a fit for a simple keypress; not worth copying for one menu item. |
| **Drag threshold** | Set an explicit **4 px** screen-space threshold (not zoom-scaled) for distinguishing a click from the start of a node-drag, wire-drag, or band-drag. This is deliberately not "imgui's own imprecise default already handles it" — it should be a named constant so the click/double-click/drag boundary is auditable. 4 px sits between dear imgui's raw default (6 px, unmodified) and imgui-node-editor's aggressive 1 px override; 1 px is too eager for a mouse (not stylus/touch) app and risks turning an intended click into an accidental micro-drag, while staying well under Houdini/Blender's much coarser felt threshold. Every reference agrees the threshold must NOT scale with zoom (xyflow states this outright); ours must not either. |
| **Escape** | No canvas behavior to add. The maintainer's Escape ladder (`hotkeys.py::_handle_escape`) has no canvas rung today (band/wire/drag cancellation on Escape is not modeled by any handler); no reference we read treats Escape as a *required* graph-canvas binding either — Blender's node-editor Escape is the generic "cancel current modal operator" (any active Blender transform, grab, or box-select tool is cancelled by Escape as a side effect of Blender's general operator system, not a node-editor-specific rule), and the imgui-family widgets rely on mouse-release, not Escape, to end a drag. If the maintainer wants Escape to cancel a live band/wire/node-drag, that is a `_draw_canvas`-local read (same `view.press_blocked`/`released_elsewhere` machinery already used for frozen turns), never a slot in the global ladder — area B doesn't need to force this now. |

### What this recommendation keeps vs. changes, and why

**Kept nearly everything.** Every current binding the brief listed (middle/Alt pan, wheel zoom
about cursor, left-drag-on-empty band, wire drag from output, grab-a-filled-input to re-plug,
drop-on-empty disconnects) already matches the convergent shape across every reference read here
— code and docs alike. That is a stronger result than a list of changes: the maintainer's
complaint about the graph editor ("feels very cheap... not sure about the controls") is not
borne out by the *bindings themselves* being wrong. It points instead at what area A/D/C cover
(wire geometry, hover feedback, card sizing) rather than at the mouse schema.

**Changed one thing: Delete gets a key.** Every reference that models delete at all binds it as
a bare key read locally against hover/focus and selection state, never through a rebindable
global registry — which is exactly the shape `commands.py::chord_needs_modifier` already forces
on us (a bare non-F key cannot be a registry chord). The fix is not "put Delete in the keymap
somehow" but "read it locally in `_draw_canvas`, the same way ImNodeFlow and litegraph.js do it,"
which sidesteps the registry constraint entirely rather than working around it.

## False trails

- **TouchDesigner, Substance Designer, Max/MSP, Pure Data** — not pursued. The eight references
  above already establish the convergent pattern (button-separated pan/select, bare-key delete,
  fixed-pixel drag threshold) with enough redundancy — three independent Dear ImGui widgets, one
  web library, one standalone JS canvas, and three documented DCC editors — that a ninth or tenth
  source was very unlikely to contradict rather than just restate the pattern, at real time cost
  (each is a new doc-scraping problem, as Blender and Unreal both turned out to be).
- **`docs.blender.org` via the in-session `WebFetch` tool** — returned HTTP 403 on every page
  (`interface/controls/nodes/*.html`), including the index page that `curl` fetched cleanly with
  a browser user agent (HTTP 200). This was the fetcher's own request signature being blocked by
  that domain, not the content being absent or the pages not existing — confirmed by successfully
  curling and reading all four target pages directly. Anyone repeating this research should default
  to `curl` for `docs.blender.org`, not `WebFetch`.
- **`docs.unrealengine.com` legacy mirror pages and its attached PDF cheat sheet** — both 403
  via `curl` (redirected into an HTML error page even for the PDF path). The content was still
  reachable, just via a different primary artifact: the Wayback Machine's cached snapshot of the
  same page (`web.archive.org/web/2023/...`), which matched WebFetch's render of the current
  `dev.epicgames.com` page exactly. Not a false trail in the sense of "wrong," but worth recording
  that the URL from a search result was not the one that ultimately worked.
- **Blender's `arranging.html` and the node-editor overview page** — read on the expectation they
  would carry the node canvas's pan/zoom bindings (per the brief's suggested URL shape); neither
  does. Blender documents 2D-canvas pan/zoom once, as a convention shared across all its 2D
  editors, not per-editor — there is no node-editor-specific "Pan" or "Zoom" heading to cite.
  The binding table above notes this rather than fabricating a citation for a page that doesn't
  carry the claim.
- **Unreal's Material Editor page** (`using-the-material-editor-in-unreal-engine`) — also a
  client-rendered shell with no static content reachable by either `curl` or WebFetch, and no
  Wayback snapshot was pursued for it once it was confirmed the Material graph reuses the same
  graph-editor widget as Blueprints (a documented Epic fact, not an assumption): the Blueprint
  cheat sheet's bindings apply unchanged.
