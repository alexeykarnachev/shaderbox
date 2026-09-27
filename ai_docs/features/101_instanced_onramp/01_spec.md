# 101 — The instanced-entity on-ramp

STUB. Not researched, not designed. Filed so the gap has a home; every section
below is a placeholder except the inventory, which is measured.

## Goal

Make instanced passes (100) discoverable from inside the app. The mechanism works
and is gated; the four names an author needs -- `@instances`, `flat in`,
`pos`/`radius`, `vs_quad` -- are documented only in `conventions.md`'s three feature-100
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

Ten surfaces, measured against the code. Ordered by harm, not by effort; 9 and 10
were found by a later read and are appended rather than renumbered into place.

1. **The copilot prompt is WRONG, not merely silent.** `shaderbox/copilot/prompt.py:142` tells the
   model that heavy stateful compute -- naming "a boids flock" -- should step CPU state
   and push it as an ARRAY uniform. That is the technique 100 replaced, and it caps at
   about a thousand vec4 before the link fails. Asked for a particle system today, the
   copilot builds the superseded thing. This is the only item that does damage rather
   than withholding help.
2. **Shader autocomplete.** `shaderbox/intel/index.py` builds `SCRIPT_UNIFORM` candidates from a
   script's returned literals; it has no notion of a `flat in` field, and `vs_quad` is
   offered nowhere. The shader is where an author starts, so this is where the absence
   is felt first.
3. **`vs_quad` is documented for the author nowhere in the app.** `shaderbox/glsl_docs.py::VARIABLES`
   is the wrong home -- it holds `gl_*` builtins only, and `vs_uv` is not in it either.
   The two places that do carry the shader contract are `shaderbox/help_content.py`'s shader
   section and `shaderbox/copilot/prompt_context.py:21`; `vs_quad` belongs beside `vs_uv` in both. It
   is the one name whose meaning cannot be guessed, and guessing wrong is silent: reach
   for `vs_uv` to shape an entity and the pass draws a canvas-wide vignette of hard
   squares with no error.
4. **The script stub.** A new `script.py` comes from `scripting.engine.script_stub_for`,
   which GENERATES the stub from the passes' introspected scriptable uniforms -- it is not a
   static template, so showing the other path means the generator learning whether any pass
   is instanced (it can: `entity_fields` already sits on `RenderPass`).
5. **Help content.** `shaderbox/help_content.py` carries an engine-uniform section whose gate
   (`tests/test_help_content.py`) walks `ENGINE_DRIVEN_UNIFORMS`. `vs_quad` belongs in
   that section's vocabulary and is absent. **`sb_instanced` does NOT**: it is written
   straight onto the program at `shaderbox/core.py:629`, is outside
   `ENGINE_DRIVEN_UNIFORMS` (so the gate could never have caught it), exists only inside
   the GENERATED vertex stage, and is in `instanced.RESERVED_NAMES` -- an author who
   declared it would be refused. Documenting it as a builtin would invite exactly that.
   The earlier draft asked for both; only `vs_quad` is user-facing.
6. **The uniform panel** shows nothing for an instanced pass: no entity count, no field
   list, no sign the pass is instanced. Reading the shader is the only way to know.
7. **The graph canvas** does not mark an instanced pass either; its tile is any other
   tile.
8. **Script autocomplete.** `shaderbox/intel/python.py` completes Python generally and knows
   nothing of `@instances` or of a pass block holding one. Ranked last: completion
   inside a dict literal is the fiddliest of these and buys the least.
9. **Additive blending is undocumented, and it is the loudest surprise.** An instanced
   pass draws with `blend_func = ONE, ONE`, hardcoded at `shaderbox/core.py:640`. It is a
   decided thing with a measured rationale (spec 100 D7: order-independent, no sort), but
   it never reached `conventions.md`, help, or the copilot prompt -- it lives only in a
   comment inside the example's shader. An author writing an instanced pass by hand gets
   summed overlaps and no way to learn why from inside the app. Not in the original
   inventory; it outranks several items that are.
10. **The example sorts LAST in the browser.** `constants.EXAMPLE_ORDER` puts Entity Flock
   at the end, the least-seen slot for the newest and least-guessable feature. A one-line
   reorder, zero risk, and it is the only item here that costs nothing to try.

CHECKED AND REFUTED, so a later wave does not re-raise it: a hand-made instanced pass does
NOT get a white frame. Spec 100 D6 wants RGBA16F, and `pass_graph.DEFAULT_DTYPE` is already
`"f2"`, so the default is correct without the author doing anything.

## Also in this wave: Python syntax highlighting

Reported by the maintainer, unrelated to instancing but in the same editor surfaces, so
it rides along rather than waiting for a wave of its own.

`self`, `__init__` and method names draw as plain identifiers in a `script.py`. Measured
against the lexer that colours them, `~/src/editor/src/lex_python.odin`: `PYTHON_KEYWORDS`
is complete and correct -- `class`, `def`, `None`, `True` are all there and do colour --
and the gap is everything that is not a keyword. Python has no `self` keyword, no rule for
a dunder, and no notion that the name after `def` is a definition rather than a use, so
the lexer has nothing to match on and the words fall through to plain text.

Where the fix lands is decided by ONE measured constraint, and it splits the cases in two.
The host's only channel is `ed_set_word_class(h, word, slot)`: an EXACT-word, whole-buffer,
**position-blind** table (`src/word_class.odin`). It can say "`self` is a builtin variable
everywhere"; it cannot say "the `update` after `def` is a definition and the `update` in a
call is not", because it has no positions to say it with. `highlight_set_ranges` -- which
the earlier draft of this spec named as the host channel -- is Odin-internal
(`src/highlight.odin:38`, called by `ffi/ffi.odin:542`) and **does not cross the C ABI**;
neither does `language_override`, which takes an Odin procedure pointer. So the host cannot
push spans today.

    reachable by word table    `self` / `cls`, dunders, a fixed numpy/API vocabulary
    needs positions            the name after `def` / `class`, decorators, annotations

For the second group the work crosses into the editor repo, which is BY DESIGN: this wave
writes the requirements and an editor session implements them against `~/src/editor`, the
same split the vendored `graph_canvas` binary already uses. So all three routes are open and
the choice is on merit, not on which repo it touches:

  - **Grow the ABI a span-push.** The library already has the primitive
    (`highlight_set_ranges`) and the per-session highlighter; this is an export, a
    buffer-version argument so a stale set is discarded, and a rule for how host spans
    compose with the lexer's. Most capable, and the only route that serves annotations.
  - **Teach the Odin lexer the cases.** Cheapest to call, and bounded by `Token_Class`'s
    seven members: `self`, `__init__` and a method name would all come out `Builtin`, the
    same colour as `len`. It draws the distinction the maintainer asked for only if
    `Token_Class` grows, which is a change to a deliberately language-neutral enum.
  - **Ship only the word-table cases.** No editor change at all; `self`, `cls` and dunders
    colour, and definitions/decorators/annotations stay plain.

Whichever wins, the word-table cases can ship FIRST and independently, because they need no
editor change -- so the on-ramp does not wait on a cross-repo round trip.

What makes the span route cheap to consider: **jedi is already a dependency, already
in-process, already serialized on `intel/worker.py`'s one thread**, and it answers every
case with positions. MEASURED, `Script.get_names(all_scopes=True, references=True)` on the
shipped flock script: `self` reports `type=param` at each definition and `type=statement` at
each use, `update` reports `type=function` with `is_definition()` true at 13:8, `property`
appears as the decorator's own name, and annotations resolve as statements. Cost is **7.5 ms
warm** for 124 lines (32 ms at 4x that, 46 ms cold on first `Script`) -- a per-rebuild cost
on the worker, the same cadence `_feed_classes` already runs at, not a per-frame one.

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

### What the check of GLSL found, and why this is one subsystem problem

GLSL does NOT have the same gap, and the reason is the finding. Highlighting is built in
two halves: the library's lexer spans what a word list can decide, and the host names
identifiers the lexer left plain through `ed_set_word_class`, a WORD -> SLOT table applied
as a post-pass (`src/word_class.odin`). MEASURED, `shaderbox/syntax_colors.py::_KIND_SLOT`
joined to `GlslIndex.classes`:

    host-fed (in `classes()`)   ENGINE_UNIFORM(7) PASS_SAMPLER(8) OUTPUT_VARIABLE(9)
                               LIB_FUNCTION(6) SCRIPT_UNIFORM(6)
    popup-only (never fed)     GLSL_KEYWORD GLSL_TYPE GLSL_BUILTIN GLSL_VARIABLE
                               WIRABLE_SAMPLER PY_KEYWORD PY_BUILTIN PY_API
    plain (slot 0)             PASS_UNIFORM BUFFER_SYMBOL PY_MEMBER PY_LOCAL GLSL_MEMBER

A kind's SLOT and its HALF are independent: `LIB_FUNCTION` and `SCRIPT_UNIFORM` are
host-fed and draw in slot 6, the lexer's builtin green, which is a colour choice and not
drift. What decides the half is membership in `classes()`, and every one of its five kinds
is GLSL -- it is the only `classes()` in the package, with one call site
(`shaderbox/tabs/code.py:410`, via `_feed_classes`).

The reason a script gets nothing is narrower and more concrete than "the path does not
exist": `shaderbox/tabs/code.py:1036` reads `if tab.kind != "script": _glsl_index_for(...)`,
so a script tab never builds an index and never calls `_feed_classes` at all. The table is
per editor session (a field on the ffi session struct), so a script's table is simply empty
-- no shader tab's classes leak into it. `self` is not a missing keyword, and not a missing
subsystem either; it is a language the one feed call is guarded away from.

MEASURED on the shipped flock script (124 lines): 302 of 316 identifier occurrences draw
plain, 95%, and `self` is 41 of them -- the most frequent word in the file. Extending the
lexer's `PYTHON_BUILTINS` (22 names) buys almost nothing by comparison: the only builtins a
real script calls are `min` and `abs`, both already there, because the rest of its
vocabulary is numpy.

That also bounds the library's share of the work. `Token_Class` (`src/language.odin`) has
seven members -- None, Keyword, String, Comment, Number, Operator, Builtin -- and is
deliberately language-neutral, its own comment saying "adding a language is adding a lexer
and a case". A constructor, a method definition and a builtin variable are three
distinctions that enum cannot carry, so teaching the Odin lexer about `self` would colour
it Builtin, the same as `len` and `SB_*`. **The distinctions belong host-side, where the
GLSL ones already are.**

### The next session reviews the whole subsystem, not these symptoms

The maintainer's read is that this is either a half-built feature or a bug, and the
measurement above says half-built: one of two halves exists for one of two languages.
Patching `self` into a word list would deepen that rather than fix it. So the next session
starts by establishing the subsystem's shape and only then decides what to add.

What it must answer, before writing anything:

- **Is the two-halves split right?** The library lexes what a word list decides; the host
  classifies what needs a symbol table. State it as a rule, then check every kind against
  it -- `LIB_FUNCTION` and `SCRIPT_UNIFORM` are host knowledge sitting in the lexer's
  slots, which either has a reason or is a second instance of the same drift.
- **Why is there no `PythonIndex.classes`?** `shaderbox/intel/python.py` already computes
  `PY_MEMBER`, `PY_API`, `PY_LOCAL` for completion. Establish whether the feed is missing
  or deliberately withheld, and whether one `classes()` can serve both languages.
- **Are the seven `Token_Class` members enough** once the host half covers both? If yes,
  the library needs no change at all and this stays in shaderbox.
- **What does the slot map cost?** There are a fixed number of syntax slots. Adding
  distinctions means either new slots or reuse; find the ceiling before designing.
- **Which kinds are plain because they SHOULD be?** Slot 0 is a legitimate answer for a
  local variable. Decide per kind rather than colouring everything that currently is not.

Then, and only then, the five Python cases: `self`/`cls`, dunders, the name after `def`
and `class`, decorators, annotations.

**Do not take the symptom list as the work.** `self` being plain is one visible instance;
the review is of the subsystem, and its deliverable is a statement of how highlighting is
supposed to work that the code then matches.

## Design decisions

None. Nothing is locked.

## Files touched

Unknown until designed. The inventory names the modules each item lands in. The
highlighting item MAY reach `~/src/editor`, depending on which of the three routes above
wins: the word-table cases need no editor change, while spans or a richer `Token_Class` do.
Cross-repo is planned here and implemented by an editor session -- see `conventions.md` on
the vendored `graph_canvas` binary for the ABI-and-rebuild shape that takes.

## Open questions for the user

- Which of the ten are in, and in what order? Item 1 is the only one that is currently
  harmful; 10 is free; 2, 3, 4 and 9 look like the cheapest real help.
- Does an instanced pass want a VISIBLE mark in the panel and the graph (6, 7), or is
  the shader's own declaration enough?
- Should the copilot be able to WRITE an instanced pass, or only stop recommending the
  old technique? The second is a prompt edit; the first needs the tools to carry a
  population and is a much larger piece.
- For highlighting, which of the three routes -- word-table only, a span-push in the ABI,
  or a richer `Token_Class` -- and does the word-table half ship first on its own?
