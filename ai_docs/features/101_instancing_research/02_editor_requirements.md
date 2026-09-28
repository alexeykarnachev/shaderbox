# Editor-library requirements: host-pushed semantic highlighting

For the session working in the editor library repo. **This states WHAT must hold and WHY,
with the evidence behind each claim. Every technical decision is yours** -- the API shape,
the coordinate unit, the storage, the merge order, the naming. Where this document names a
shape it is reporting what the existing code already does, not prescribing.

The consuming application is ShaderBox, but **the mechanism must be general**: the library
is reused across projects, so nothing here may be specific to ShaderBox, to Python, or to
any one parser. A host with a tree-sitter grammar, an LSP, or its own compiler must be able
to use the same channel.

Citations are `file:line` in the editor repo unless prefixed `shaderbox/`. Everything marked
MEASURED was produced by running code.

## The problem, in one paragraph

A host can currently tell the library "colour the word `self` everywhere"
(`ed_set_word_class`, `ffi/ffi.odin:216`). It cannot tell it "colour characters 8..14 on
line 13", because the only channel is a word -> class map applied position-blind
(`src/word_class.odin:138`). So a host that has computed real semantic information --
which `update` is a definition and which is a call, which name is a decorator, which is a
type annotation -- has no way to express it. MEASURED in ShaderBox: 302 of 316 identifier
occurrences in a real 123-line script draw plain (95.6%).

## What the library already says about this

This is not a new feature. It is a promise the library makes and the ABI does not keep.

- `src/language.odin:46-47` names the intended hosts: *"it is how an application plugs in a
  parser -- a tree-sitter grammar, an LSP's semantic tokens, its own compiler's output"*.
- `src/language.odin:28-29`: *"`None` is a first-class choice rather than an absence: a
  plain-text buffer, or a host that pushes its own spans through highlight_set_ranges and
  wants nothing overwriting them."*
- `ffi/README.md:946-949` repeats it **in the ABI's own contract document, addressed to a C
  host**: *"Language `None` is a choice, not an absence. It turns highlighting off and
  leaves spans a host pushed itself alone."*

No ABI call can keep that promise. `highlight_set_ranges` (`src/highlight.odin:38`) is a
public package procedure -- called from `ui/main.odin:175,405` -- but it is not exported,
and `language_override` (`src/language.odin:55`) cannot cross either, because it takes an
Odin procedure pointer. The contrast that proves the barrier is the pointer and not policy:
`language_override_path` DOES cross (`ffi/ffi.odin:984`), taking only a string and an int.

`ffi/README.md:919-922` also names the gap plainly: *"The classification is LEXICAL, so it
is by spelling, not position: in `float mix; ... mix(1.0, 2.0, mix)` every `mix` reads as
the builtin."*

## R1 — A host can push position-based spans across the C ABI

The core requirement. A host computes spans however it likes and hands them over; the
library draws them.

Constraints the existing code imposes, each with its evidence:

- **Staleness must be decidable, and the host cannot compute the version itself.**
  `highlight_set_ranges` stores a version (`src/highlight.odin:52`) and `highlight_class_at`
  returns 0 when it differs from the buffer's (`:59-62`), so a stale set is discarded rather
  than misapplied. But the version is the BUFFER's, and a C host has no access to it:
  `ed_revision` (`ffi/ffi.odin:398`) is NOT `buffer.version` -- `ffi/ffi.odin:47-52` adds a
  base to stay monotonic across `ed_set_text`. **Whatever the API looks like, a host must be
  able to say which text it computed against, and must be able to learn that its answer was
  too late.** This matters because the first consumer is asynchronous (R5).
- **Codepoint boundaries.** Already solved at `src/highlight.odin:44-49` -- a boundary
  landing mid-codepoint snaps outward, the end snapping UP -- but only when the buffer is
  passed; the `b: ^Buffer = nil` default silently disables it. Pinned at the Odin level by
  `test_span_boundaries_snap_outward` (`src/highlight_test.odin:55-72`). A host's spans can
  be arbitrary, so this must not be bypassed.
- **Ordering.** The highlighter does not require sorted spans and says why
  (`src/highlight.odin:63-69`, pinned by `test_unsorted_spans_still_resolve`), and
  `word_classes_apply` defends itself by sorting a copy (`src/word_class.odin:157`,
  `:192-195`). But overlap resolution is first-match-wins in scan order, so for an unsorted
  set INSERTION order decides which of two overlapping spans wins. Either define that or
  remove the ambiguity.
- **Ownership.** `Span` is POD (`{start, end: Pos, class: u32}`) and `highlight_set_ranges`
  copies on append, so a host's buffer may die on return. This is SIMPLER than the word
  table, which must clone its string keys (`src/word_class.odin:44-46`) and pays for it with
  the `key_of` scan (`:115-123`). Keep it that way -- no host pointer should outlive the call.
- **The coordinate unit is your call.** `Pos` is a BYTE offset (`src/types.odin:22`), which
  is what `Span` already holds. Every other ABI call that names a location takes line/col and
  converts via `ffi_pos_at_column` (`ffi/ffi.odin:421,1014,1760`). There is no precedent for
  byte offsets crossing the boundary. Byte offsets are cheaper and match `Span`; line/col
  matches every existing call and costs two conversions per span. **Decide it.**
- **Bulk-transfer precedent, for whatever shape you choose.** Two patterns exist.
  `ed_primitives` (`ffi/ffi.odin:645`) is pointer+cap returning a count, chosen because
  per-item ctypes crossing measured ~0.55 us and ~1.1 ms per frame (`:635-643`), and it
  CONVERTS into a flat `FFI_Primitive` (`:452`) rather than exposing the internal struct.
  `ed_complete_begin`/`push`/`cancel` (`:168,192,318`) is push-based, chosen because *"the
  FFI surface inverts control: the host PUSHES candidates in, rather than supplying a
  callback the core calls back into. That is also the easier shape across a language
  boundary"* (`:160-165`). Both argue against a callback and against a retained host array.

## R2 — Host spans must survive a retokenization

**This is the requirement most likely to be missed, because the obvious implementation
silently fails it.** Exporting `highlight_set_ranges` as-is does NOT satisfy R1.

`ffi/ffi.odin:542` calls it unconditionally inside the retokenize block, and
`highlight_set_ranges` CLEARS first (`src/highlight.odin:39`). So host-pushed spans placed
in `s.hl` are discarded on the next buffer-version move -- i.e. on the next keystroke. Host
spans need storage of their own and a merge into the span set inside that block.

Related: `wc_revision` (`ffi/ffi.odin:30`) exists because *"the buffer version alone is not
enough to decide whether to re-lex: a host re-feeding the table while the text stood still
must reach the screen on the next frame, not on the next keystroke"* (`:24-28`). A host
pushing spans between keystrokes has exactly that problem.

## R3 — Host spans must draw under `Language.None`

The configuration the README promises them in is the one where the current gate turns them
off.

`ffi/ffi.odin:531` gates the retokenize block on
`s.lang != .None || word_classes_len(...) > 0`. With language None and no word table --
precisely "a host doing its own colouring" -- the block never runs. `hl_active` decides the
same thing at four sites in THREE different forms: `:224` and `:957` are
`lang != .None || word_classes_len > 0`; `:235` is `lang != .None` alone, the post-clear form
inside `ed_clear_word_classes` where the table is empty by construction; and the gate at
`:531` is a fourth spelling. `layout_emit` receives nil when the flag is false (`:548`), and
`ed_class_at` short-circuits on it (`:1006`).

**This exact bug has already shipped once, for the word table.** `ffi/ffi.odin:522-527`
records it verbatim: *"gating the whole block on a language made word classes silently inert
in the one configuration the document promises them in. Measured: class 7 on a word read
back as 0 under None and 7 under GLSL."*

So: **break it before believing it.** Set `Language.None`, push spans, confirm nothing
draws, then fix, and say in the commit which break was tried. A gate that has not been seen
to fail is not known to work -- and this one has a recorded precedent of passing while
broken.

We name the sites only because a change to what activates highlighting has to reach all of
them, and they do not agree today. What to do about that is yours.

## R4 — More syntax classes than exist today

MEASURED: `Theme.syntax` is `[10]Color` (`src/theme.odin:53`), index 0 meaning "no class",
so nine usable classes. **SIX are the built-in lexers' own** — 1 keyword, 2 string, 3 comment,
4 number, 5 operator, 6 builtin, as `src/theme.odin:48` and `ffi/README.md:952-954` both
state. That leaves **three host-assignable (7, 8, 9)**, and ShaderBox has all three: engine
uniforms, pass samplers, the fragment output. **Zero are free.**

(An earlier draft of this document said "four lexer-owned, five assigned". That reached the
right total of nine by double-counting 1 and 6 as host assignments — ShaderBox shares those
two with the lexer rather than owning them. The conclusion was right and the arithmetic was
not, which matters because this is the number you would size the work from.)

The first consumer needs four more distinctions in one language (a builtin variable like
`self`, a constructor, a method definition, a type annotation) while keeping every existing
one. A general mechanism cannot cap a host's vocabulary at the number GLSL happened to need.

`Token_Class` (`src/language.odin:13-21`) has seven members and is deliberately
language-neutral -- *"adding a language is adding a lexer and a case"*. **It should not need
to change**: it is the LEXER's vocabulary, and everything here is host-fed. `src/theme.odin:175-182`
already maps classes to the array by name, which is a hint about how cheap widening is, but
the count is your call.

**One option we did not want to foreclose, since it may be cheaper than widening:** the
ceiling could be per-LANGUAGE rather than global. GLSL does not use every class Python would
want, and a per-language palette would give a second language its distinctions out of space
the first is not using, with no change to the array's size. We have no view on which is right
-- it is named only so the choice is made deliberately rather than by default.

## R4a — Say what a host must not assume about class numbering

Related to R4 and cheap to state: whatever the ceiling becomes, a host needs to know whether
class numbers are stable across versions of the library, and whether a class it does not
recognise is safe to push. The concrete artifact is `shaderbox/syntax_colors.py::_KIND_SLOT`,
a literal map from the host's own symbol kinds to class numbers, written into source. A
renumbering upstream silently recolours every one of them, which is why this wants to be a
contract rather than an implementation detail.

## R5 — The first consumer is asynchronous, and that is a design input

ShaderBox will produce spans with jedi on a worker thread (MEASURED: median 7.2 ms warm on a
123-line script, 30.4 ms at 4x, ~37 ms cold, plus a one-time ~42 ms import). It cannot answer
on the frame that asked.

The consequence for the API: **a host will routinely push spans computed against text that
has since changed.** R1's staleness requirement is what makes that safe. The host side owns
the policy for what to do about it -- re-request, or accept a brief mismatch -- but the
library must make the situation detectable rather than silently misapplying a stale set.

Nothing here asks the library to do the async work. It asks it not to assume synchrony.

**A size and rate budget, so this is designed against numbers rather than against a shape.**
MEASURED on the first consumer: ~300 names on a 123-line script, so **hundreds of spans per
push**, not thousands. Push rate is at most once per edit burst (ShaderBox will debounce
rather than push per keystroke). The worst case we care about is a 4000-line buffer. We ask
for the budget explicitly because `src/word_class.odin:149-156` records this exact class of
defect being got wrong once — 4000 lines cost 159 ms against the lexer's own 37 before the
cursor fix — and a span API with no stated size expectation invites the same shape.

## R5a — Say how a span set is replaced, cleared, and what survives `ed_set_text`

Three questions the word table answers and a span API must too, because a host cannot guess
them: does a push REPLACE the previous set or add to it; is there a clear; and what happens to
a pushed set when the buffer is replaced wholesale. The word table has all three
(`ed_clear_word_classes` at `ffi/ffi.odin:230`, replace-on-set at `src/word_class.odin:92`,
class 0 removes at `:70-73`), which is why it is easy to write against.

## R5b — Say which wins when a span and a word-table entry cover the same identifier

**This one we cannot answer for you, and we cannot proceed without it.** R6 keeps both
mechanisms, and the first consumer deliberately feeds BOTH for one language: `self` and the
dunders on the word table, definitions and the rest as spans. So the two channels will cover
the same buffer and, eventually, the same word. Today `word_classes_apply` fills only gaps the
lexer left (`src/word_class.odin:21-25, 184-186`); whether host spans are a third tier above
both, or join that same gap-filling pass, is a CONTRACT question rather than an implementation
one. Whatever you choose, it wants to be stated rather than emergent.

## R6 — The word table stays, and keeps its job

Do not replace `ed_set_word_class` with spans. They answer different questions, and the
distinction is sharper than "two mechanisms":

**A name-keyed fact is edit-invariant; a position-keyed fact is not.**
`word_classes_apply` re-runs against the new buffer positions on every retokenization
(`ffi/ffi.odin:540`), so "these 40 names are engine uniforms" survives every edit for free.
`highlight_class_at:59` discards a whole span set on any version change, so a span-fed host
must recompute and re-push on every keystroke. Using spans for a fixed vocabulary would mean
recomputing a constant list every frame the buffer moves.

`src/word_class.odin:19-20` also notes the composition property: applying the table outside
the lexers means *"a host's OWN lexer gets the same treatment for free"*. That property
should extend to host spans too -- a host using both should get both.

ShaderBox will use exactly this split: `self`/`cls`/dunders on the word table (instant, no
parser, and correct because those names mean the same thing everywhere), spans only for the
genuinely positional distinctions. GLSL stays entirely on the word table, because there is no
GLSL parser and the regex index it does have is name-keyed by type.

## A note on R3's configuration

R3 asks for host spans to draw under `Language.None` because that is the configuration the
README promises them in, and because the library records that exact gate shipping broken once.
Worth knowing while you weigh it: **ShaderBox itself will not be in that configuration** — it
sets `Language.Python` on script tabs, so `hl_active` is already true for it. R3 protects the
generality the documentation claims, not this consumer. We still want it, and we would rather
say which it is than let it read as the gate guarding our case.

## Deliverable

Whatever API you judge correct, plus:

- the `Language.None` break-and-restore demonstrated for R3, named in the commit;
- a test that the ABI wrapper passes the buffer for codepoint snapping (the Odin-level test
  cannot catch a wrapper that forgets the argument);
- whatever `ffi/README.md` needs so its existing promise becomes true rather than aspirational.

Tell ShaderBox the resulting call signatures and the class-count ceiling; that side will be
written against whatever you land on.
