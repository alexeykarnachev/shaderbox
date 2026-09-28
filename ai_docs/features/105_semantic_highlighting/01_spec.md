# 105 — Semantic highlighting

`self`, `__init__` and method names draw plain in a script. The cause is not a missing word
in a list: the host's only channel is position-blind, and the one call that feeds it is
guarded off script tabs. MEASURED: 302 of 316 identifier occurrences in a real 123-line
script draw plain (95.6%), and `self` is 41 of them.

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
- **D3. The producer is asynchronous, and the colour may lag.** jedi is already a dependency,
  already in-process, already serialized on one worker thread: 7.2 ms warm on the flock
  script, 30.4 ms at 4x, ~37 ms cold. It cannot answer on the frame that asked, so colour
  lags text by one to three frames while typing. Unlike completion latency this is VISIBLE —
  briefly wrong colour rather than an absent popup. **Revisit if** measurement in the running
  app shows the lag reads as broken rather than as settling.
- **D4. A span result must not be dropped because the caret moved.** The worker's existing
  match predicate requires caret agreement; spans are a property of the TEXT. A span request
  needs a revision-only predicate.
- **D5. GLSL stays entirely on the word table.** There is no GLSL parser — the index is nine
  regexes, and nothing in the project provides one. Spans would mean widening those regexes
  to emit offsets: more code, same job, more ways to be wrong, and inheriting the staleness
  problem for gains nobody has asked for.
- **D6. The script tab's index guard is removed.** `if tab.kind != "script"` is the one line
  that makes the whole Python half unreachable.

## The distinctions

| case | mechanism |
|---|---|
| `self`, `cls` | word table |
| dunders (`__init__`, `__name__`) | word table |
| the name after `def` / `class` | spans |
| decorators | spans |
| annotations after `:` and `->` | spans |

Four of these need syntax classes that do not exist: nine slots total, four lexer-owned, five
already assigned here, **zero free**. That is R4 in the editor requirements.

## Gates

- **The one with a recorded precedent, and it must be broken first:** host spans draw under
  `Language.None`. The library records this exact bug shipping once for the word table. Set
  `Language.None`, push spans, confirm nothing draws, then fix. Named in the commit.
- Host spans survive a retokenization — break it by clearing them on a version move and watch
  a keystroke wipe the colour.
- A script tab feeds classes at all (D6): break it by restoring the guard.
- A stale span set is rejected rather than misapplied (D4's revision rule).

## Out of scope

- Teaching the Odin lexer Python by name. Excluded by D-B, and bounded anyway: the lexer's
  class vocabulary cannot tell a builtin variable from a constructor from a method name.
- Highlighting for any language beyond GLSL and Python.

## Open questions

None on this side. The editor-side technical choices belong to that session by D-C.
