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
precisely "a host doing its own colouring" -- the block never runs. `hl_active` is computed
identically at `:224`, `:235` and `:957`, `layout_emit` receives nil when it is false
(`:548`), and `ed_class_at` short-circuits on it (`:1006`).

**This exact bug has already shipped once, for the word table.** `ffi/ffi.odin:522-527`
records it verbatim: *"gating the whole block on a language made word classes silently inert
in the one configuration the document promises them in. Measured: class 7 on a word read
back as 0 under None and 7 under GLSL."*

So: **break it before believing it.** Set `Language.None`, push spans, confirm nothing
draws, then fix, and say in the commit which break was tried. A gate that has not been seen
to fail is not known to work -- and this one has a recorded precedent of passing while
broken.

Note also that `hl_active` being copy-pasted at three sites is the duplicate to REMOVE
rather than to guard: one procedure, three callers.

## R4 — More syntax classes than exist today

MEASURED: `Theme.syntax` is `[10]Color` (`src/theme.odin:53`), index 0 meaning "no class",
so nine usable slots. Four are taken by the lexer's own classes (string, comment, number,
operator). In ShaderBox the remaining five are all assigned. **Zero are free.**

The first consumer needs four more distinctions in one language (a builtin variable like
`self`, a constructor, a method definition, a type annotation) while keeping every existing
one. A general mechanism cannot cap a host's vocabulary at the number GLSL happened to need.

`Token_Class` (`src/language.odin:13-21`) has seven members and is deliberately
language-neutral -- *"adding a language is adding a lexer and a case"*. **It should not need
to change**: it is the LEXER's vocabulary, and everything here is host-fed. Extending the
theme's class space is the part that matters. `src/theme.odin:175-182` already maps slots to
the array by name, which is a hint about how cheap this is, but the count is your call.

## R5 — The first consumer is asynchronous, and that is a design input

ShaderBox will produce spans with jedi on a worker thread (MEASURED: median 7.2 ms warm on a
123-line script, 30.4 ms at 4x, ~37 ms cold, plus a one-time ~42 ms import). It cannot answer
on the frame that asked.

The consequence for the API: **a host will routinely push spans computed against text that
has since changed.** R1's staleness requirement is what makes that safe. The host side owns
the policy for what to do about it -- re-request, or accept a brief mismatch -- but the
library must make the situation detectable rather than silently misapplying a stale set.

Nothing here asks the library to do the async work. It asks it not to assume synchrony.

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

## Deliverable

Whatever API you judge correct, plus:

- the `Language.None` break-and-restore demonstrated for R3, named in the commit;
- a test that the ABI wrapper passes the buffer for codepoint snapping (the Odin-level test
  cannot catch a wrapper that forgets the argument);
- whatever `ffi/README.md` needs so its existing promise becomes true rather than aspirational.

Tell ShaderBox the resulting call signatures and the class-count ceiling; that side will be
written against whatever you land on.
