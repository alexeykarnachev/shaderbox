# 101 — The instanced-entity on-ramp

STUB. Not researched, not designed. Filed so the gap has a home; every section
below is a placeholder except the inventory, which is measured.

## Goal

Make instanced passes (100) discoverable from inside the app. The mechanism works
and is gated; the four names an author needs -- `@instances`, `flat in`,
`pos`/`radius`, `vs_quad` -- are documented only in `conventions.md`'s two feature-100
entries and demonstrated only in one shipped example. Nothing in the editor, the panels or the copilot knows the feature
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
3. **`vs_quad` is documented for the author nowhere in the app.** `glsl_docs.VARIABLES`
   is the wrong home -- it holds `gl_*` builtins only, and `vs_uv` is not in it either.
   The two places that do carry the shader contract are `help_content.py`'s shader
   section and `copilot/prompt_context.py`; `vs_quad` belongs beside `vs_uv` in both. It
   is the one name whose meaning cannot be guessed, and guessing wrong is silent: reach
   for `vs_uv` to shape an entity and the pass draws a canvas-wide vignette of hard
   squares with no error.
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

## Also in this wave: Python syntax highlighting

Reported by the maintainer, unrelated to instancing but in the same editor surfaces, so
it rides along rather than waiting for a wave of its own.

`self`, `__init__` and method names draw as plain identifiers in a `script.py`. Measured
against the lexer that colours them, `~/src/editor/src/lex_python.odin`: `PYTHON_KEYWORDS`
is complete and correct -- `class`, `def`, `None`, `True` are all there and do colour --
and the gap is everything that is not a keyword. Python has no `self` keyword, no rule for
a dunder, and no notion that the name after `def` is a definition rather than a use, so
the lexer has nothing to match on and the words fall through to plain text.

That puts the fix in the editor library, NOT in this repo: `lex_python.odin` classifies,
and shaderbox only maps a class to a colour in `syntax_colors.py`. Either the library
learns the cases below and shaderbox picks up a rebuilt `.so`, or the library exposes
them and shaderbox classifies them host-side the way it already classifies GLSL
identifiers. Which of those is the first decision this item needs.

The cases, in the order they are noticeable:

- `self` and `cls` -- the first parameter, everywhere it appears.
- Dunders: `__init__`, `__name__`, and the rest of the form `__x__`.
- A definition name: the identifier after `def` and after `class`, which reads
  differently from a call.
- A decorator, `@property` and the like, including the `@`.
- A type annotation after `:` and after `->`.

The reference is the maintainer's own nvim -- gruvbox hard, no italic comments -- whose
treesitter groups resolve to (MEASURED, `nvim_get_hl`):

    @variable.builtin  #fe8019   self, cls
    @constructor       #fe8019   __init__
    @function.method   #b8bb26   method names
    @function          #b8bb26   function names
    @keyword           #fb4934
    @type              #fabd2f
    @string            #b8bb26
    @comment           #928374
    @number            #d3869b

Those are the SOURCE, not the target: shaderbox has its own palette in `theme.py` and the
slot map in `syntax_colors.py`, and matching gruvbox's hexes exactly would fight it. What
carries over is the DISTINCTION -- that a builtin variable, a constructor, a method name
and a plain local are four different things -- and shaderbox already has kinds for most of
them (`PY_MEMBER`, `PY_API`, `PY_LOCAL`), currently all mapped to `SYN_IDENT`.

**Open:** whether the editor library learns these or exposes them; whether shaderbox's
palette gains slots for the new distinctions or reuses existing ones; and whether the
GLSL side has the same gap (its lexer emits keywords, numbers and builtins, and a
host-classified identifier fills the rest -- so the answer is probably no, but it is
worth one look while the code is open).

## Design decisions

None. Nothing is locked.

## Files touched

Unknown until designed. The inventory names the modules each item lands in; the
highlighting item lands in `~/src/editor` first, which is a different repo with its own
ABI and rebuild step -- see `conventions.md` on the vendored `graph_canvas` binary for
the shape that takes.

## Open questions for the user

- Which of the eight are in, and in what order? Item 1 is the only one that is
  currently harmful; 2, 3 and 4 look like the cheapest real help.
- Does an instanced pass want a VISIBLE mark in the panel and the graph (6, 7), or is
  the shader's own declaration enough?
- Should the copilot be able to WRITE an instanced pass, or only stop recommending the
  old technique? The second is a prompt edit; the first needs the tools to carry a
  population and is a much larger piece.
