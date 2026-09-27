# 101 — The instanced-entity on-ramp

STUB. Not researched, not designed. Filed so the gap has a home; every section
below is a placeholder except the inventory, which is measured.

## Goal

Make instanced passes (100) discoverable from inside the app. The mechanism works
and is gated; the four names an author needs -- `@instances`, `flat in`,
`pos`/`radius`, `vs_quad` -- exist only in one shipped example and in
`conventions.md`. Nothing in the editor, the panels or the copilot knows the feature
is there, so it is usable only by someone already told about it.

## Out of scope

Everything 100 deferred stays deferred, and none of it blocks this: script hot reload
restarting a simulation (`ai_docs/features/100_instanced_entities/02_hot_reload.md`),
export re-simulating from t=0 with a different dt sequence, off-thread simulation with
snapshot interpolation, and a liveness mechanism (stable slots, generation counters).
**Trigger** for each: the maintainer asks, or a second instanced example makes one of
them the thing in the way.

## The inventory

Eight surfaces, measured against the code. Ordered by harm, not by effort.

1. **The copilot prompt is WRONG, not merely silent.** `copilot/prompt.py:142` tells the
   model that heavy stateful compute -- naming "a boids flock" -- should step CPU state
   and push it as an ARRAY uniform. That is the technique 100 replaced, and it caps at
   about a thousand vec4 before the link fails. Asked for a particle system today, the
   copilot builds the superseded thing. This is the only item that does damage rather
   than withholding help.
2. **Shader autocomplete.** `intel/index.py` builds `SCRIPT_UNIFORM` candidates from a
   script's returned literals; it has no notion of a `flat in` field, and `vs_quad` is
   offered nowhere. The shader is where an author starts, so this is where the absence
   is felt first.
3. **`glsl_docs.py`** documents `vs_uv` in `VARIABLES` for hover. `vs_quad` is absent,
   and it is the one name whose meaning cannot be guessed -- quad-local, -1..1, versus
   `vs_uv`'s 0..1 across the canvas.
4. **The script stub.** A new `script.py` starts from a template teaching the
   plain-uniform path. The cheapest place to show the other one.
5. **Help content.** `help_content.py` carries an engine-uniform section with a gate
   asserting every user-facing builtin is documented. `vs_quad` and `sb_instanced` are
   engine-provided and appear in neither.
6. **The uniform panel** shows nothing for an instanced pass: no entity count, no field
   list, no sign the pass is instanced. Reading the shader is the only way to know.
7. **The graph canvas** does not mark an instanced pass either; its tile is any other
   tile.
8. **Script autocomplete.** `intel/python.py` completes Python generally and knows
   nothing of `@instances` or of a pass block holding one. Ranked last: completion
   inside a dict literal is the fiddliest of these and buys the least.

## Design decisions

None. Nothing is locked.

## Files touched

Unknown until designed. The inventory names the modules each item lands in.

## Open questions for the user

- Which of the eight are in, and in what order? Item 1 is the only one that is
  currently harmful; 2, 3 and 4 look like the cheapest real help.
- Does an instanced pass want a VISIBLE mark in the panel and the graph (6, 7), or is
  the shader's own declaration enough?
- Should the copilot be able to WRITE an instanced pass, or only stop recommending the
  old technique? The second is a prompt edit; the first needs the tools to carry a
  population and is a much larger piece.
