# 049 — Reopening a node's shader after its tab is closed (DRAFT — problem only)

> **Status: DRAFT.** This spec states the PROBLEM only. No solution is committed yet — the design
> (button vs toggle vs auto-reopen, where it lives, whether it's a new `ui_primitives` glyph) is
> deliberately left open and gets decided when this feature is picked up. Do not implement from this
> draft; start by locking the design.

Filed from a maintainer observation (2026-06-15) while reasoning about the post-048 tabbed editor.

## The problem

Since the editor became tabbed (045 → 048), a node has two editable artifacts — its fragment shader
(`shader.frag.glsl`) and its optional brain script (`scripts/script.py`) — each shown as its own tab
in the left-hand code panel. There is an explicit affordance to OPEN the script (the `</>` "brain"
glyph in the node header → `App.open_script_for`). **There is no symmetric affordance to (re)open the
shader.** A shader tab only appears via `App.ensure_shader_tab`, which is called from exactly one
place: `_on_current_node_changed` — i.e. only when the *selected node changes*.

This leaves a dead end. Reproduction:

1. A node is selected; its shader tab and script tab are both open.
2. Close the shader tab (the tab's `x`). Only the script tab remains open for that node.
3. There is now **no direct way to get the shader back** for the *currently selected* node.
   Selecting the same node again does nothing — `current_node_id` doesn't change, so
   `_on_current_node_changed` never fires, so `ensure_shader_tab` is never called.

The only recovery is a workaround the maintainer flagged as bad UX: select a *different* node, then
select the original node again, which finally triggers `ensure_shader_tab` and reopens the shader —
but that side trip also focuses (and may have opened) the other node's shader, leaving an extra tab to
clean up.

## Why it's a real gap, not a misuse

The open/reopen paths are asymmetric by construction:

- **Script:** `open_script_for(node_id)` exists, is wired to a header glyph, lazily creates the file,
  and `_focus_or_add_tab`s it. A user can always summon the script.
- **Shader:** the equivalent `ensure_shader_tab(node_id)` *exists as a method* but has **no UI trigger
  of its own** — it rides only on the node-selection-changed event. So once the shader tab is closed,
  nothing the user can press re-opens it for the node they're already on.

`ensure_shader_tab` already does exactly the right thing (focus-or-open the node's shader tab,
idempotent via `_focus_or_add_tab` keyed on path) — the gap is purely that no user-reachable control
calls it for the current node.

## Scope marker (for whoever picks this up)

- This is a UI-affordance gap; the underlying tab machinery (`editor_tabs`, `_focus_or_add_tab`,
  `ensure_shader_tab`, `close_tab`) is sound and needs no rework to fix the symptom.
- The fix touches UI flow + likely a new control in the node header (`tabs/node.py` /
  `ui_primitives.py`) — that makes it feature-shaped (spec + the normal flow), not a drive-by edit.
- Read `/imgui-ui` before any UI work; route any new control through `ui_primitives.py` + `theme.py`
  per the hard rules (no hand-rolled glyph at the call site).

## Relevant code (entry points, not a design)

- `App.open_script_for` / `App.ensure_shader_tab` / `App.close_tab` / `App._focus_or_add_tab`
  (`shaderbox/app.py`).
- `App._on_current_node_changed` — the sole caller of `ensure_shader_tab`.
- The node-header script control + `script_glyph` (`shaderbox/tabs/node.py`,
  `shaderbox/ui_primitives.py`).
- The tab bar render + click/close handling (`shaderbox/tabs/code.py`).
