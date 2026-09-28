# 104 — The instancing on-ramp

Feature 100's instanced passes are usable only by someone already told they exist. Ten
surfaces, all in, ordered by harm.

Research and evidence: `ai_docs/features/101_instancing_research/01_spec.md` (Part 3, D1–D10).
Gates on 102 and 103, so that what this teaches is true.

## Goal

An author who has never been told about instancing can find it, and nothing the app says
about shaders is false for an instanced pass.

## Design decisions

- **D1. All items ship, and the text-only ones ship FIRST without waiting.** A subset left
  standing leaves an author on a partial on-ramp, which is the state this feature exists to
  end. But review measured that **only item 6 is genuinely blocked** (by 102's blend choice)
  and **nothing here is blocked by 103 at all** — no item consumes copilot output. So items
  1, 2, 9 and 10, which are the two false help sentences, the README gap and the example
  ordering, are pure text and constant edits with no engine dependency. They are the
  highest-harm, lowest-cost entries in the feature and there is no technical reason to hold
  them behind another wave. Ship them as soon as the wave opens.
  **Revisit if** a text item turns out to depend on a 102 surface after all.
- **D2. Ordered: wrong text, then wrong picture, then silence.** The research's original
  "ordered by harm" claim was false — it called the prompt the only item that did damage,
  when the help modal states invariants instancing breaks. Text that is WRONG outranks text
  that is MISSING.
- **D2a. The panel badge and the graph badge are DIFFERENT WORK.** An earlier draft of this
  spec said the graph item was "Same" as the panel one. It is not. The uniform panel is imgui
  with an existing precedent for a state mark (a uniform row recolours its name rather than
  drawing a glyph). **The graph canvas is not imgui at all** — it is a custom GPU renderer
  drawing through `NodeSpec`/`BodyRow` structs at a fixed 120-byte stride
  (`graph_canvas/render.py`), so a badge there is a new `BodyRow` or a node-struct field plus
  its FFI, not a draw call. Size and schedule them separately; the graph half may land later
  than the rest without holding the feature.
- **D2b. A badge is a CHIP, not a button.** So the button-tier rule does not apply, but three
  do: colour and size through `theme.py` tokens only, never a hand-rolled `push_style_color`;
  no icon-font glyph; and on the graph canvas the jitter rule bites, since an overlay using
  `set_cursor_screen_pos` perturbs the parent's content size. `ui_primitives.text_chip` is the
  closest passive shape. Read `/imgui-ui` before drawing either.
- **D3. A capability badge is per-PASS; a live count is per-FRAME.** Instancing is a property
  of a pass, read from the FLATTENED source — so a `flat in` spliced in from a `lib:` include
  counts and the open tab may not reveal it, which is itself a reason the badge earns its
  place. Capability is fixed at compile and belongs on the pass surfaces; the live count is
  per-frame state and belongs where per-frame state already shows, reading 102's outcome.
- **D4. `sb_instanced` is NOT documented, and the engine must SAY which names are which.**
  It is engine-internal, exists only in the generated vertex stage, and is in
  `RESERVED_NAMES` — an author who declared it is refused. Documenting it would invite
  exactly that. Only `vs_quad` is user-facing.

  **This forces a code change before the gate can exist.** `instanced.RESERVED_NAMES` holds
  four names of three different kinds: `vs_quad` (document it), `sb_instanced` and `a_corner`
  (never document them), `vs_uv` (user-facing, documented in prose elsewhere). **Nothing in
  the engine marks which is which** — the set exists to reject author declarations, a
  different question. So a gate asserting "every reserved name is documented" would demand
  documenting `sb_instanced`, contradicting this very decision. **Split the set**: a
  user-facing set and an engine-internal set, with `RESERVED_NAMES` derived as their union so
  the two cannot drift. That partition is what the gate enumerates from, and building it is
  part of this feature.

## The ten

| # | Surface | What it needs |
|---|---|---|
| 1 | help modal, first section | It calls `vs_uv` and the full-screen quad "fixed". Both are false for an instanced pass. **Wrong text, in the section a new user reads first.** |
| 2 | `vs_quad` documented nowhere user-facing | The one name an author cannot guess, and guessing wrong is silent — reaching for `vs_uv` draws a canvas-wide vignette with no error. |
| 3 | help content's engine vocabulary | `vs_quad` beside `vs_uv`. Not `sb_instanced` (D4). **Needs a NEW section, not a row**: the engine-uniform section renders `uniform {type} {name};` lines off a uniform table, and `vs_quad` is an `in` varying — it does not fit that shape, and `vs_uv` is not in that table either (it lives in item 1's prose). |
| 4 | shader autocomplete | No notion of a `flat in` field; `vs_quad` offered nowhere. |
| 5 | the script stub | Generated from introspected scriptable uniforms, so an instanced-only document gets "(no scriptable uniforms)" and `return {}`. It can know: `entity_fields` is already on the pass — but `script_stub_for`'s SIGNATURE must change, it takes uniforms only. Shares its code site with 103 D4b (the copilot-facing half); whichever wave lands first does it. |
| 6 | blend mode undocumented | Now a per-pass choice (102 D5); document the choice, not the old constant, including that alpha mode is order-dependent (102 D5 reverses only one of feature 100 decision 7's three clauses). **The control itself lands in 102** alongside the surfaces it builds; this item is the help/conventions/README half. The one item genuinely blocked by 102. |
| 7 | uniform panel | No mark for an instanced pass. Per D3. |
| 8 | graph canvas | A mark for an instanced pass — **but see D2a: this is a packed-struct/FFI change in a custom GPU renderer, not the imgui edit item 7 is.** Beware: `graph_canvas/render.py` is full of the word "instance" for UI GPU instancing, which is unrelated; a grep-driven reader would wrongly conclude the graph already knows about the feature. |
| 9 | README | Never mentions it; the scripting bullet names "a physics step, an integrator" and stops at uniform-driving. This is the itch.io pitch. |
| 10 | examples browser ordering | Entity Flock sorts LAST — and its description is the single best in-app explanation of the feature, gated behind the last gallery item. |

| 11 | the `Passes` help section | It enumerates every per-pass property — `smooth`, `repeat`, Runs, size, format — and is silent on instancing and on blend. A SECOND help section that is wrong by omission, distinct from item 1's wrong-by-assertion one. |
| 12 | the pass-settings modal | Where `format`, `scale` and `iterations` are chosen and explained by tooltip. It is the natural home for the blend control 102 D5 requires "in the pass panel", and it fell between items 6 and 7 in an earlier draft. |

Script autocomplete for `@instances` is folded into 4 rather than listed separately:
completion inside a dict literal is the same surface and the fiddliest part of it.

## Gates

- **Every user-facing engine name is documented, enumerated from D4's user-facing set** — the
  partition D4 requires the engine to grow, since today nothing in the code distinguishes a
  name to document from one to hide. `sb_instanced` and `a_corner` asserted ABSENT from the
  docs, so D4 cannot erode. **The absent half is a silence assertion**, so it needs a
  contact proof in the same test: assert the doc text was actually read and non-empty, or
  "correctly absent" and "searched the wrong string" return the same green.
- **No help text asserts an invariant an instanced pass breaks.** The break: delete the
  "three things are fixed" sentence's replacement and restore the current wording — note that
  wording is in the tree TODAY and the help tests pass, so the break is "leave it as it is",
  not "put it back". **This gate is the wrong instrument if written as prose analysis**: a
  checker cannot decide "asserts an invariant". Narrow it to a pinned list of phrases known to
  be false for an instanced pass, and say in the gate's own text that it covers those phrases
  and nothing else, so the next reader does not take the gap for an oversight.
- **An instanced pass is marked; a fullscreen pass is not** — a real contrast pair differing
  only in the property under test. The "is not" half is a silence assertion and needs the
  second observable: prove the fullscreen pass's row was DRAWN in the same frame (its name
  present, its thumbnail rendered), or a fixture that never reached it reports "no mark"
  identically to a correct one.

## Files touched

- `shaderbox/help_content.py` — item 1's first section, item 3's new section, item 11's
  `Passes` section.
- `shaderbox/instanced.py` — D4's user-facing / engine-internal partition.
- `shaderbox/intel/index.py`, `shaderbox/intel/glsl.py`, `shaderbox/completion.py` — item 4.
- `shaderbox/scripting/engine.py` — `script_stub_for`'s signature (item 5).
- `shaderbox/tabs/uniforms.py` — item 7.
- `shaderbox/widgets/pass_graph.py`, `shaderbox/graph_canvas/` — item 8 (D2a).
- `shaderbox/popups/pass_settings.py` — item 12.
- `shaderbox/constants.py` — `EXAMPLE_ORDER` (item 10).
- `README.md` — item 9.
- `ai_docs/conventions.md` — item 6's blend half, if 102 has not already added it (102 D5a).
- Tests: `test_help_content.py`, plus new panel/graph badge fixtures.

## Out of scope

- **The blend CONTROL itself** — 102 D5 builds it alongside the surfaces it touches; item 6
  documents it. **Trigger:** none; this is a split, not a deferral.
- **The copilot's prompt and its generated API doc** — 103's, and no item here consumes them.
  **Trigger:** none.

An earlier draft said "Nothing. This is the whole on-ramp", which was false: review found two
surfaces missing (now items 11 and 12).

## Open questions for the user

None blocking.
