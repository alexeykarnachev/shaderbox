# 105 — Semantic highlighting

`self`, `__init__` and method names draw plain in a script. MEASURED: 302 of 316 identifier
occurrences in a real 123-line script draw plain (95.6%), and `self` is 41 of them.

**There are TWO causes and they want different fixes**, which an earlier draft of this spec
conflated. For `self`, `cls` and the dunders — the largest single share of that 95.6% — the
cause is simply that **no Python word-class feed exists** (D6); they need no positions and no
spans. For definitions, decorators and annotations the cause is that the host's only channel
is **position-blind**, which is what the span mechanism exists for. Fixing the first is
ShaderBox-only and unblocked; the second needs the editor library.

Research and evidence: `ai_docs/features/101_instancing_research/01_spec.md` (Part 4).
**Independent of 102–104 — runs in parallel from the start.**

## Goal

A general mechanism by which a host pushes semantic highlighting it computed itself (D-B),
with ShaderBox as its first client. Not a Python special case, not a shaderbox workaround.

## Two repos

- **The editor library** grows the mechanism. Requirements are WRITTEN:
  `ai_docs/features/101_instancing_research/02_editor_requirements.md` (R1–R6, with the
  evidence behind each). **Every technical decision there is the editor session's** — API
  shape, coordinate unit, storage, merge order, class count. Hand that document over when
  this feature starts (D-C).
- **ShaderBox** becomes the first client once the API lands.

## Design decisions

- **D1. Two mechanisms, two jobs — the word table stays.** A name-keyed fact is
  EDIT-INVARIANT; a position-keyed one is not. The word table re-applies against new
  positions on every retokenization, so a fixed vocabulary survives every edit for free,
  while a span set is discarded on any version change. Using spans for engine uniforms would
  mean recomputing a constant list every frame the buffer moves.
- **D2. `self`/`cls`/dunders go on the WORD TABLE; only positional distinctions get spans.**
  Not a compromise — those names mean the same thing everywhere, so position-blindness is
  correct for them, and it halves the exposure to D3's latency.
- **D3. The producer is asynchronous, and the redraw gate must learn about it.** jedi is
  already a dependency, already in-process, already serialized on one worker thread: 7.2 ms
  warm on the flock script, 30.4 ms at 4x, ~37 ms cold. It cannot answer on the frame that
  asked.

  **An earlier draft called this "colour lags one to three frames, acceptable". Review showed
  that is not what happens, because the editor panel is a CACHED TEXTURE** redrawn only when
  `render_state` moves (`editor/render.py:99-122`, gate at `tabs/code.py:1077`). That tuple
  carries undo index, cursor, mode, selection, scroll, prim count — **nothing that moves when
  a span set is pushed while the text stands still.** Two regimes follow:
  - **during a burst**: each keystroke moves the undo index so the panel redraws, but the
    answer in hand was computed against older text and the library discards a version-mismatched
    span set wholesale — so colour drops to PLAIN for the whole burst, then snaps back;
  - **at rest**: the final answer lands with the buffer idle, `render_state` does not move, and
    **the colour never appears at all** until some unrelated state change. A feature that
    intermittently does not happen, invisible to any test that types and then asserts.

  **So `render_state` gains a span-feed revision dimension**, mirroring the library's own
  `wc_revision` — which exists for exactly this shape ("a host re-feeding while the text stands
  still must reach the screen on the next frame, not on the next keystroke"). Gated by
  `tests/test_editor_ffi.py::test_render_state_reacts_to_every_editor_dimension`, which walks
  the domain with real mutations and must gain the span case.

  **Second mitigation: debounce on idle** rather than submitting per keystroke. The burst
  regime then produces no stale answers to drop, and it removes the head-of-line problem in
  D3a at the same time.
  **Revisit if** measured colour-settle time after a burst exceeds 200 ms on the flock script.
- **D3a. The spans request must not block completion.** `PythonWorker._next` pops in insertion
  order, one request at a time, on one thread. A 7.2 ms SPANS job sits in front of a COMPLETE
  job for the same burst, so completion latency — a shipped feature — regresses by up to the
  spans cost on every keystroke. Debounce (D3) or a second queue; decide and measure.
- **D3b. DECIDED: drop-and-re-request, and the library decides it rather than a host
  predicate.** `ed_set_spans` (editor `05f90ac`) takes the `ed_revision` the text was read at
  and returns 1 applied, 0 that revision is no longer the buffer's — nothing changed, the
  previous set stands — and -1 for a class outside 1..9. There is no offset salvage, which is
  the right call: a set computed before a newline was typed would put every span below that
  line one row out, landing a definition's colour on the WRONG word rather than on no word.

  The staleness test is THE PUSH ITSELF. A host-side comparison would be a second derivation
  of the buffer's own version in a different number space, and only the library's answer says
  what got drawn. Gated by `test_a_stale_span_set_is_rejected_rather_than_misapplied`, which
  asserts both halves: the refusal returns false AND the previously applied set is still
  colouring its characters afterwards.

  **A correction to the prediction above, measured:** an applied set is ANCHORED in the
  buffer and follows edits until the next push, so the "colour drops to PLAIN for the whole
  burst, then snaps back" regime does not happen. Between an edit and the debounced answer
  the previous set keeps colouring the characters it named. That changes the JUSTIFICATION
  for the debounce — latency and head-of-line, not staleness. Pinned by
  `test_an_applied_set_follows_an_edit`. `ed_set_text` DROPS the set, and `ed_revision` rises
  across it, so the feed re-requests on the next idle frame with no special case.
- **D4. A span result must not be dropped because the caret moved.** The worker's existing
  match predicate requires caret agreement; spans are a property of the TEXT. A span request
  needs a revision-only predicate.
- **D5. GLSL stays entirely on the word table.** The structural reason, which is stronger than
  the headcount an earlier draft gave: the GLSL index is **name-keyed by type** —
  `GlslIndex.symbols` is `Mapping[str, Symbol]` and `classes()` returns
  `dict[str, SymbolKind]`. There is no position anywhere in the data structure, and no GLSL
  parser exists in the project to supply one. Spans would mean widening the extractors to emit
  offsets: more code, same job, more ways to be wrong, and inheriting the staleness problem for
  gains nobody has asked for. **Revisit if** a GLSL parser lands — that is the fact that would
  change, and it is a fact rather than a preference.
- **D6. A PYTHON-SIDE class feeder is written. The guard is not simply removed.** An earlier
  draft of this spec said `if tab.kind != "script"` (`tabs/code.py:1036`) "is the one line
  that makes the whole Python half unreachable", and that was wrong in a way that would have
  produced a wrong one-line change. That guard keeps the **GLSL** indexer off Python tabs, and
  `_glsl_index_for` is what re-feeds word classes as a side effect of rebuilding a GLSL index
  — its fingerprint and its `build()` are entirely GLSL (pass name, sampler values,
  `wired_pass`, engine uniform types). Removing the guard would run the GLSL index on a Python
  script and produce a meaningless index.

  The real work, none of which existed in the earlier draft:
  - a Python producer of word classes — `classes()` is defined on `GlslIndex` and filters five
    GLSL-only kinds, so `_feed_classes` has no Python counterpart to call;
  - its own index/fingerprint path for script tabs, not the GLSL one un-guarded;
  - a result type carrying POSITIONS: `PythonResult.symbols` is `tuple[Symbol, ...]` and
    `Symbol` has no position field, so a span answer is a new result type rather than a new
    request kind.
  **Revisit if** the two index paths converge enough to share a fingerprint.

## The distinctions

| case | mechanism | producer |
|---|---|---|
| `self`, `cls` | word table | a fixed name list — no parser |
| dunders (`__init__`, `__name__`) | word table | a fixed rule — no parser |
| the name after `def` / `class` | spans | jedi `get_names` (`type=function`/`class`, `is_definition()` true) |
| decorators | spans | **NOT jedi — needs `parso` or `ast`** |
| annotations after `:` and `->` | spans | **NOT jedi — needs `parso` or `ast`** |

**D7. RESOLVED — all five cases ship, and the second producer costs LESS than the first.**
parso answers definitions, decorators and annotations from one parse at **6.63 ms** on the
123-line flock script, against **7.46 ms** for jedi's `get_names` alone — so the positional
half is cheaper than the producer the original estimate was sized against, and jedi is not on
the span path at all. On an unparseable buffer, which is the normal state mid-edit, parso
recovers rather than raising: an unclosed paren, a dangling `def`, a half-typed annotation, an
unterminated string and a truncated file all measured **6.26–6.74 ms** and kept the whole
file's structure, so the deferral trigger ("if the second producer's cost on an unparseable
buffer cannot be bounded") is not met. A 4x buffer costs 24.9 ms.

parso also fixes a defect jedi would have shipped: `get_names` reports an IMPORTED name as a
definition at its import site, so `from dataclasses import dataclass` would have coloured
`dataclass` as though the script defined it. The tree says `funcdef`/`classdef` structurally.
Gated by `test_an_import_is_not_a_definition`.

Annotations and decorators share `PY_DECORATOR`, because the contract froze four kinds and a
fifth was not mine to add.

The original statement of the problem, which the measurements above resolve: An earlier draft of this spec, and the research it cites, claimed `get_names` answers
all five cases. Review ran it and it does not: a decorator `@property` and a bare reference to
`property` are **byte-identical in every field `get_names` exposes** — name, type,
`is_definition()`, description. The same for `int` as an annotation versus `int` anywhere
else. "Annotations resolve as statements" is literally true and operationally useless, because
every reference is a statement.

So `parso` (already installed, a jedi dependency) or `ast` has to supply those two, with its
own cost, its own threading and its own behaviour on unparseable text — which is the normal
state of a buffer mid-edit. **D3's measurements, D3a's head-of-line analysis and the editor
requirements' size budget were all sized against `get_names` alone.** Either re-cost against
the second producer or ship the three cases jedi can answer and say which two are deferred.
**Trigger for deferring:** if the second producer's cost on an unparseable buffer cannot be
bounded, decorators and annotations wait. **NOT MET — see the measurements above.**

**One thing verified rather than assumed: the column unit.** The library wants CODEPOINT
columns and parso could have reported codepoints, UTF-16 code units or bytes. MEASURED on a
line with an astral character, where all three differ: parso says 11 where UTF-16 says 14 and
bytes say 20, and the library's `ed_class_at` agrees with 11 — a span pushed at the byte
column reads back as class 0. No conversion is needed. Gated by
`test_the_column_unit_is_codepoints_not_bytes`, whose fixture holds a 3-byte character BEFORE
the name so the two columns differ by six; at the ASCII defaults they are equal and the gate
would pass with a byte-based producer.

**Zero free syntax classes GLOBALLY, and that turned out not to be the constraint.** The
palette is PER-EDITOR, and ShaderBox makes one editor per source path with the language fixed
at that same site, so slots 7/8/9 mean an engine uniform and a pass sampler in a `.frag.glsl`
buffer and a definition and a decorator in a `.py` one. No buffer holds both vocabularies, so
no library change and no theme widening were needed — `editor_palette` takes a language and
`get_session` passes it. The editor session confirmed the ceiling stays at 9, that the numbers
index `Theme.syntax` directly and are stable across versions, and that a class beyond 9 is
REFUSED rather than clamped. The count below is still the right arithmetic for the question it
answers:

Nine exist. **SIX are lexer-owned (1-6: keyword, string,
comment, number, operator, builtin)** and three are host-assignable (7, 8, 9) — all three
already taken here by engine uniforms, pass samplers and the fragment output. An earlier draft
said "four lexer-owned, five assigned", which reached the right total of nine by
double-counting 1 and 6 as host assignments; the conclusion held and the arithmetic did not.
That matters because the editor requirements are what the editor session sizes its work from.

**An option that must be put to the editor session rather than assumed away: is the class
ceiling per-language or global?** Five ShaderBox kinds already map to class 0 (plain), and
GLSL does not use every slot Python would want. A per-language palette would give Python the
classes it needs out of space GLSL is not using, with no widening of the theme array — cheaper
than the widening the requirements currently steer toward.

## Gates

Library-side gates (the `Language.None` break, surviving a retokenization) belong to the
editor session and live in R2/R3 of the requirements — **not restated here**, so there is one
copy of an implementation instruction and it sits in the repo that owns the code. Note also
that ShaderBox runs script tabs under `Language.Python`, never `None`, so that gate protects
generality rather than this consumer.

ShaderBox-side:

- **A script tab feeds word classes at all** (D6) — break it by removing the Python feeder and
  watching `self` go plain. NOT "restore the guard": the guard is correct and stays.
- **The colour reaches the screen while the buffer stands still** (D3) — push a span set with
  no edit and assert the panel redrew. Break it by reverting the `render_state` dimension and
  watching the colour never appear. This is the gate for the failure mode that is otherwise
  invisible to every test that types and then asserts.
- **The four new kinds are actually EMITTED.** The enum gate
  (`tests/test_intel_sources.py::test_every_kind_has_a_color`) walks `SymbolKind` and checks a
  colour, a rank and a slot per kind — and review confirmed its blind spot: **a kind wired with
  all three that nothing ever produces passes clean.** Four kinds go through that gate, so one
  case per kind asserting it appears on real source. Break by removing the producer while
  leaving colour and slot in place, and watch the enum gate stay green while this one fails.
- **A stale span set is rejected rather than misapplied** (D3b), asserting BOTH that the push
  is refused and that the previously applied set is still colouring its characters afterwards
  — a return-code assertion alone would miss the second half.
- **A spans job never delays a completion job** (D3a). Ungated when first written, and found
  by mutating the queue back to insertion order and watching nothing fail.
- **The column unit is codepoints** (D7), on a fixture whose byte and codepoint columns differ.

## Files touched

The editor side landed at `05f90ac` and is vendored here (`ed_set_spans`, `ed_clear_spans`;
108 -> 110 exports, re-derived with `nm -D`).

- `shaderbox/intel/python.py` — the class/span producer, plus whatever second producer D7
  needs.
- `shaderbox/intel/worker.py` — a spans request kind, a position-carrying result type, a
  revision-only match predicate (D3a's queue question).
- `shaderbox/intel/symbols.py` — four new `SymbolKind` members.
- `shaderbox/syntax_colors.py` — `_KIND_COLOR`, `_KIND_SLOT`, `editor_palette` for each.
- `shaderbox/theme.py` — four new `COLOR.SYN_*` tokens.
- `shaderbox/editor/ffi.py` — `Slot`, and the new span call once the library lands it.
- `shaderbox/tabs/code.py` — the Python feed path beside `_feed_classes` (D6).
- `shaderbox/editor/render.py` — `render_state`'s span dimension (D3).
- Tests: `test_intel_sources.py` (its hardcoded `0 <= slot <= 9` bound moves),
  `test_editor_ffi.py` (`test_render_state_reacts_to_every_editor_dimension`).

## Out of scope

- **Teaching the Odin lexer Python by name.** Excluded by D-B, and bounded anyway: the lexer's
  class vocabulary cannot tell a builtin variable from a constructor from a method name.
  **Trigger:** none — this is a decision, not a deferral.
- **Highlighting for any language beyond GLSL and Python.** **Trigger:** a third language ships
  in the app.
- **Decorators and annotations, IF D7's second producer cannot be bounded** on an unparseable
  buffer. **Trigger:** the measurement in D7.

## Open questions for the user

None. Both decisions the builder owed a measurement are taken: **D3b** is drop-and-re-request,
which `ed_set_spans` enforces rather than a host predicate, and **D7** ships all five cases
because parso answers the three positional ones at 6.63 ms and stays bounded at 6.26-6.74 ms on
every unparseable buffer tried.
