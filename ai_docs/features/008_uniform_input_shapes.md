# Feature 008 — Generalized uniform input shapes

## Goal

Make every uniform row carry a **leading input-shape pill** that lets the user change *how* a
uniform is edited (its "input shape") without changing the underlying GLSL type. The pill is
present on **every** row — interactive when the uniform has >1 valid shape, **disabled** (greyed,
same width) when it has exactly one. This fixes the ragged alignment (some rows had a pill, some
didn't, so control columns and trailing names started at different x) and generalizes the
ad-hoc drag↔color / array↔text toggles into one uniform mechanism.

After this lands: every uniform row begins with a fixed-width pill showing the current shape;
clicking it (when >1 shape is valid) cycles to the next valid shape; the rest of the row renders
that shape's editor. Single-shape uniforms (texture / buffer / engine-driven) show a disabled
pill for alignment + "this is fixed" signal.

## Out of scope

- **Changing the GLSL uniform type.** The pill only changes the *input shape* (UI), never the
  declared GLSL type or the value's wire format.
- **Dropdown / popup-menu selection.** Shipping with **click-to-cycle** for all shape counts
  (2 and 3+). See Decision 3 — the interaction is deliberately consolidated into one function so
  switching to a dropdown later is a single-function change, NOT a refactor.
- **New input shapes beyond the current set** (`texture` / `buffer` / `array` / `color` / `text` /
  `drag` / `auto`). This feature generalizes the *selection* of existing shapes, not new ones.

## Design decisions (locked)

### 1. `valid_input_types(ui_uniform)` is the single source of truth for the shape set
A pure function in `ui_models.py` returning an ordered `tuple[UIUniformInputType, ...]` for a
given uniform, derived from `gl_type` / `dimension` / `array_length` / `is_ubo` / `name`. No imgui.
`reset_input_type()` picks the **default** as the first element of this set (or a name-based
refinement, see the table). The pill cycles within this exact set. Nothing else computes "what
shapes are allowed" — every consumer calls this one function.

Shape sets (cycle order; first = default unless a name-rule refines it):

| Uniform | valid set (cycle order) | default |
|---|---|---|
| engine (`u_time` / `u_aspect` / `u_resolution`) | `(auto,)` | auto |
| UBO | `(buffer,)` | buffer |
| sampler2D | `(texture,)` | texture |
| scalar dim 1 / 2 | `(drag,)` | drag |
| scalar dim 3 / 4, name ends `color` | `(drag, color)` | color |
| scalar dim 3 / 4, other | `(drag, color)` | drag |
| uint array (len>1) | `(array, text)` | text if name ends `text` else array |
| float / int array (len>1) | `(array,)` | array |

`auto` (read-only `name: value` readout) is **only** the locked shape for engine-driven uniforms
(`u_time` etc., which the render loop overwrites every frame — editing is meaningless). It is NOT a
user-selectable cycle option: a static readout is never a shape a user wants to switch a real input
*into*, and it renders badly for arrays (full dump + name overflow). Single-shape numeric types
(dim-1/2 drag, float array) just show a disabled pill for alignment.

### 2. The pill is on every row; disabled when the set has one element
Drawn at fixed `SIZE.CHIP_W` so every row's control column starts at the same x. One element →
`imgui.begin_disabled()` around the pill (greyed, non-interactive) showing the locked shape's
label. The disabled pill still occupies `CHIP_W` → alignment holds for texture / buffer / engine
rows too.

### 3. Selection interaction is consolidated into ONE function (the swappable seam)
`draw_input_type_selector(ui_uniform)` in `widgets/uniform.py` is the **only** place the picker
UI + interaction lives. It calls `valid_input_types(...)`, draws the pill, and mutates
`ui_uniform.input_type`. The row body (`draw_ui_uniform`) and every value-rendering branch read
`ui_uniform.input_type` and never know how it was chosen. **Switching cycle → dropdown is a
rewrite of this function's body alone** — the valid-set, the disabled rule, the alignment slot,
and the draw branches are untouched. Do not scatter selection logic into the branches.

### 4. Default classification unchanged → existing nodes look identical until clicked
`reset_input_type()` still yields today's defaults (drag / color / array / text). Persisted
`input_type` in `ui_uniforms` is honored as before. The only visible change on load is the new
leading pill column.

## Acceptance

- Every uniform row starts with a `CHIP_W`-wide pill; control columns + trailing names align
  across all rows regardless of shape.
- texture / buffer / engine rows show a **disabled** pill.
- Clicking an interactive pill cycles through `valid_input_types()` (wrapping) and re-renders in the new
  shape with no crash for any shape transition (regression: the tuple-vs-list text crash, 706f1c5).
- `make smoke` passes against `projects/dev/` (incl. the all-input-types test node 5edc2e8e).
- Selection logic exists in exactly one function (Decision 3).

## Review-round fixes (4-agent swarm, converged)

- **HIGH — stale/out-of-set persisted `input_type` crashed the panel.** A node.json could persist an
  `input_type` no longer in `valid_input_types()` (hand-edit or a shape-set change); the value-type
  asserts in `draw_ui_uniform` then threw mid-row → unbalanced imgui stack. Fix: `UIUniform.
  snap_input_type()` re-snaps an out-of-set value to a valid one, called each frame in the
  `tabs/node.py` reconcile loop before the auto/active split. (Broader `UINodeState`
  invalid-value-drops-the-node gap parked in `todo.md`.)
- **LOW — resolution dedup `tuple==list`.** `canvas.texture.size` (list) was compared to `(w,h)`
  tuples and never matched; a canvas-sized texture showed a duplicate combo entry. Fix:
  `current_size = tuple(...)`.
- **LOW — scalar `int`/`uint` drag crash.** A `uniform int` classifies to `drag` (dim 1) but its
  value is a Python `int`; the dim-1 branch asserted `float`. Fix: branch on `int` →
  `imgui.drag_int`, accept `int|float`.
