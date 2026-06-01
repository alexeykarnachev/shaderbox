# 020 · 13 — GLSL whitespace-invariant matcher (the lexer)

The next step in the maintainer's 2-step sequence (roadmap Active-context). Slice-12 closed the
*spiral symptom* — a model whose `old_str` indentation didn't match the source (6-space vs 4-space)
retried the same failing byte-exact `edit_shader` until `max_iterations`. Slice-12's fixes were
behavioral guards: a whitespace-normalized *hint* echoing the exact bytes to copy, a retry cap, a
chat-surfaced cutoff. They make the failure cheap and visible — they do NOT make the edit succeed.

This slice fixes the failure at its **source: the match itself**. `edit_shader` matches `old_str`
against the source by `str.count` / `str.replace` — byte-exact. A model that reproduces a region
correctly token-for-token but indents it differently STILL gets a 0-match and must re-copy. The fix:
match by **token-stream equality** — lex both source and `old_str` into a stream of GLSL tokens and
compare the streams, ignoring only the inter-token whitespace. A token-equal region IS the edit
target; the replacement still splices into the EXACT original byte span (whitespace preserved), so
nothing about the file's formatting changes — only the *locate* becomes whitespace-invariant.

py2glsl ships no GLSL lexer (checked); `shader_lib/parser.py` is regex machinery (function-signature
/ identifier / comment-strip), not a tokenizer — neither is reusable here. This is a new ~50-line
leaf.

---

## Goal

A minimal GLSL tokenizer + a token-stream matcher, wired as `edit_shader`'s matching layer:

1. **`glsl_lex(src: str) -> list[Token]`** — a ~50-line GLSL tokenizer. Each `Token` carries its
   **kind**, its **raw text exactly as it appears** (`raw`), and its **byte span** `(start, end)` into
   the source. Whitespace and comments are NOT emitted as tokens (they're the thing we want to ignore),
   but the spans of the emitted tokens still index the original source verbatim.

2. **`token_match(src: str, old_str: str) -> list[tuple[int, int]]`** — lex both, find every
   contiguous run of source tokens whose `(kind, raw)` sequence equals `old_str`'s token sequence,
   and return each match as a `(byte_start, byte_end)` span into `src` (start of the first matched
   token, end of the last). The span is the EXACT original bytes — replacement preserves the source's
   own surrounding whitespace. **Matching algorithm (locked):**
   - **Anchored, full-length, all matches.** A match at source-token index `i` requires the source
     token run `src_tokens[i : i+len(needle)]` to equal the FULL `old_str` token list by `(kind, raw)`
     — a prefix of a longer equal run does NOT extend; a subsequence does NOT count. Return EVERY such
     `i`'s span — NEVER early-return on the first (I4: ambiguity must be visible).
   - **Non-overlapping, left-to-right (matches `str.replace`).** After accepting a match at `i`, resume
     scanning at `i + len(needle)`, not `i + 1` — so two source regions are never double-covered. This
     reproduces `str.replace`'s non-overlapping greedy behavior, and keeps the `replace_all` splice
     well-defined (the returned spans never overlap).
   - **Empty needle → no match.** If `old_str` lexes to ZERO tokens (empty / whitespace-only / a lone
     comment), return `[]` (0 matches). This routes through the existing `matches==0` branch — never a
     zero-length span (which would splice `new_str` at byte 0 and corrupt the file). `glsl_lex("")`
     returns `[]`.

3. **Wire it into `edit_shader`'s apply path** as the matcher, *replacing* byte-exact `str.count` /
   `str.replace`. Token-equality widens what *locates* (every byte-exact match is also a token match,
   so nothing that locates today stops locating) — with ONE behavioral caveat: comment bytes that sit
   *inside* a matched span are now part of the replaced span (the lexer skips comments, so the matched
   span runs first-token→last-token and any comment between them falls inside it). So a byte-exact
   `old_str` carrying an embedded/trailing comment still MATCHES, but the splice overwrites that
   comment with `new_str`'s bytes — acceptable (the model controls `new_str`; see Out-of-scope
   comment-aware matching). The match-COUNT semantics (`0` = not found, `>1` = ambiguous unless
   `replace_all`) are UNCHANGED — they now count token-run matches instead of byte-substring matches.

### The 4 locked invariants (from the prior design session)

These were settled with adversarial-safety review last session; they are LOCKED, not open:

- **I1 — Lex-first.** Compare token streams, never raw strings. Whitespace between tokens is the only
  thing ignored. (A newline vs a space vs 4-vs-6 spaces between two tokens is invisible to the match.)
- **I2 — Maximal munch.** Each token is the longest legal lexeme at that position (`<=` is ONE token,
  not `<` then `=`; `1.0e-3` is one number; `SB_perlin_noise_3` is one identifier). Without this,
  `a<=b` and `a< =b` would tokenize identically AND `a<=b` vs `a < = b` would mis-split — maximal
  munch makes the token boundaries deterministic and whitespace-position-independent.
- **I3 — Raw token text.** A token compares by `(kind, raw)` where `raw` is the verbatim source slice
  — NOT a normalized/canonical form. So `1.0` ≠ `1.00`, `0xFF` ≠ `255`, `vec3` ≠ `vec3 ` (the trailing
  space is never part of `raw`). We ignore whitespace *between* tokens, never *inside* a token, and we
  never canonicalize a token's value. This keeps the match honest: only formatting differs, never
  meaning.
- **I4 — Uniqueness guard.** `token_match` returns ALL matches; the apply path keeps the existing
  count-based ambiguity rule (`>1` match fails unless `replace_all`). Token-equality can make an
  `old_str` that was byte-unique become token-ambiguous (two regions identical up to whitespace). That
  correctly surfaces as the existing "not unique — add context / replace_all" error, NOT a silent
  wrong-region edit. The widened matcher must never *reduce* the safety of the ambiguity check.

---

## Out of scope (each with a trigger)

- **A full GLSL grammar / parser / AST.** This is a *lexer* — a flat token stream, no nesting, no
  grammar. The matcher needs only lexeme-level equality. **Trigger:** a future semantic tool
  (`rename_symbol`, `outline`, `add_uniform` — Slice-2+ deferred) needs structure; build the parser
  then, on top of this lexer.
- **Preprocessor awareness (`#define` expansion, `#if` evaluation).** `#version`, `#define`, `#line`,
  etc. lex as ordinary tokens (a `#` punctuator + identifier + …) — the matcher treats a preprocessor
  line like any other token run. No macro expansion, no conditional stripping. **Trigger:** a tool
  needs to match against post-preprocessor source (none does — the user edits the literal file text).
- **Comment-aware matching.** Comments are skipped by the lexer (like whitespace), so an `old_str`
  that differs from the source ONLY in a comment will token-match and the replacement splices the
  source's bytes (the source's comment, since the span is the source's). This is acceptable —
  matching is about locating, the `new_str` carries whatever comment the model wants. **Trigger:** a
  trace shows a model losing a comment it meant to preserve because the comment wasn't part of the
  match.
- **The slice-12 whitespace HINT path stays.** `_ws_normalize` / `_whitespace_near_match` (the
  echo-exact-bytes hint on a 0-match) is NOT removed — see Design decision 4. **Trigger:** if the
  lexer makes whitespace-only misses impossible (so the hint never fires), a later cleanup can retire
  it; not in this slice (the hint still covers genuinely-absent `old_str`).
- **Auto-fuzzy / Levenshtein / near-miss APPLY.** Slice-12 already rejected silent fuzzy-apply; this
  slice does not reintroduce it. Token-equality is exact-at-the-token-level, not fuzzy. **Trigger:** a
  second distinct near-miss mode (not whitespace) shows up in a trace.

---

## Design decisions
*(numbered, lock-in only)*

1. **New leaf module `shaderbox/copilot/glsl_lex.py`.** Pure text → tokens; no GL, no imgui, no App,
   no copilot-package imports. It is a leaf the App-side apply path imports (like `shader_errors.py`
   is a leaf `core.py` imports). It lives under `copilot/` because the copilot edit path is its only
   consumer today; if a non-copilot consumer ever appears it graduates to a top-level leaf, but
   speculative placement is premature. The public surface is `Token`, `TokenKind`, `glsl_lex`,
   `token_match`.

2. **`Token` is a frozen dataclass `(kind: TokenKind, raw: str, start: int, end: int)`.** `TokenKind`
   is a `StrEnum` — `IDENT`, `NUMBER`, `PUNCT`, `PREPROC_HASH` (the `#`), with room to add. The matcher
   compares `(kind, raw)`; `start`/`end` are byte offsets into the lexed source (the span the matcher
   returns). Frozen + leaf types only (primitives + the enum) — cycle-free, hashable, testable.

3. **Tokenizer is a maximal-munch scanner over character classes (I2), comments + whitespace skipped
   POSITIONALLY.** The scan loop advances an index `i` over `src`; at each step: skip a whitespace run;
   skip a `//…\n` line comment or a `/*…*/` block comment (same GLSL comment rules
   `shader_lib/parser.py::strip_comments` encodes — no strings, no nested block comments — but
   re-implemented as an *index advance*, NOT a `re.sub` pre-strip: pre-stripping would make every
   token's `start`/`end` index the stripped string, breaking the §1 "spans index the original source
   verbatim" guarantee). Then emit the longest token at `i` (the order below IS the maximal-munch
   priority):

   - **NUMBER** — tried before `.` punctuation and before identifiers. The exact sub-grammar (pin —
     this is the one class that can FALSE-match if loose):
     - **hex int:** `0[xX][0-9a-fA-F]+` then optional `[uU]` suffix. Hex is its OWN branch (the `f`/`e`
       in `0xFe`/`0xFF` are hex DIGITS, never a float-suffix or exponent — so the hex branch consumes
       greedily and never enters the float/exp logic).
     - **float:** matches any of `digits . digits?`, `. digits`, or `digits` — followed by an OPTIONAL
       exponent `[eE][+-]?digits` and an OPTIONAL float suffix `f|F|lf|LF`. So `1.0`, `.5`, `1.`, `1e3`,
       `1.0e-3`, `2.0f`, `1.0lf` are each ONE NUMBER. The `[+-]` is consumed ONLY immediately after
       `e`/`E` — never as a leading sign and never elsewhere; so `a-1` is IDENT `a`, PUNCT `-`, NUMBER
       `1` (the `-` is subtraction, NOT a signed number), and `1.0e-3` keeps its exponent sign inside
       the one NUMBER. A bare `.` not followed by a digit is NOT a number (falls through to PUNCT `.`).
     - **decimal int:** `digits` then optional `[uU]` (reached only when no `.`/exp/suffix follows —
       it's the float grammar's degenerate "digits, no fraction" case, so one regex covers both).
     A practical single regex (anchored at `i`), tried in this order, captures it; the test table
     (Manual verification) pins every case incl. the reject cases (`a-1`, `.` alone, `0xFe`).
   - **IDENT** — `[A-Za-z_]\w*`.
   - **PUNCT** — a multi-char operator from a maximal-munch table tried LONGEST-FIRST
     (`<<=`,`>>=` → `<=`,`>=`,`==`,`!=`,`&&`,`||`,`++`,`--`,`+=`,`-=`,`*=`,`/=`,`<<`,`>>` → single-char
     `;{}()[].,+-*/%<>=!&|^~?:`). An unrecognized byte is emitted as a single-char `PUNCT` (never crash
     — robustness over strictness; a stray char becomes its own token and still has to match).
   - **PREPROC_HASH** — a `#` emits as `PREPROC_HASH`; the rest of the directive lexes as ordinary
     tokens (no preprocessor awareness — Out-of-scope).

   GLSL has **no string literals**, so there is no string-scanning state (simplifies the lexer
   materially).

4. **The slice-12 hint path is RETAINED as a no-op safety belt, NOT because it still does useful
   work.** With token-matching the whitespace-only miss the hint was BUILT for now *succeeds*
   (matches=1) — and on reflection (review B2) the hint can essentially never fire usefully again:
   `_whitespace_near_match` finds a region matching `old_str` ignoring whitespace, but token-equality
   already absorbs ALL inter-token whitespace, so any region the hint could find, the matcher matches
   directly (matches≥1, so we never reach the `matches==0` hint branch). The remaining 0-match cases
   (a typo'd identifier, an intra-token difference like `1.0` vs `1.00`) are ones `_ws_normalize` also
   can't bridge → it returns `""`. So the hint is retained UNCHANGED and unconditionally (token-match
   first; on `matches==0`, compute the hint as before — it just returns `""` in practice), keeping the
   diff minimal and the fallback honest rather than deleting working-but-now-dead code mid-slice.
   Retirement is the Out-of-scope trigger, not this slice.

5. **Replacement splices by non-overlapping byte span, not `str.replace`.** `token_match` returns the
   matched `(start, end)` spans in source order, non-overlapping (§2 algorithm). Single match:
   `src[:start] + new_str + src[end:]`. `replace_all`: rebuild by accumulating across the ordered spans
   — `out = []; cursor = 0; for (s, e) in spans: out.append(src[cursor:s]); out.append(new_str);
   cursor = e; out.append(src[cursor:]); "".join(out)` (offset-stable because the spans don't overlap
   and we walk left-to-right with a moving cursor). This replaces `src.replace(old_str, new_str)`. The
   match count is `len(spans)`; the `0` / `>1` branches are unchanged. `new_str` is inserted
   **verbatim** — the model controls the new region's formatting exactly (we never reformat it).

6. **No change to `EditResult`, the tool layer, the agent loop, or the prompt.** The widening is
   entirely inside `_copilot_apply_shader_edit`'s `_on_main`. `edit_shader`'s description still says
   "match EXACTLY including whitespace" — that remains the SAFE instruction to the model (copying
   exact bytes always works); the lexer just makes a whitespace-divergent copy ALSO work instead of
   failing. We do NOT advertise whitespace-invariance in the prompt (that would invite the model to be
   sloppy on purpose, and token-ambiguity failures would rise). The matcher is a safety net, not a
   documented contract. **Open question O1 flags this** — confirm we keep the prompt unchanged.

---

## Files touched
- **NEW `shaderbox/copilot/glsl_lex.py`:** `TokenKind` (StrEnum), `Token` (frozen dataclass),
  `glsl_lex(src) -> list[Token]`, `token_match(src, old_str) -> list[tuple[int, int]]`. The ~50-line
  leaf. Comment skipping re-implements the GLSL comment RULES `shader_lib/parser.py::strip_comments`
  encodes (no strings, no nested block comments) as a POSITIONAL index-advance (NOT a `re.sub`
  pre-strip — would break byte spans; NOT a cross-package import — `shader_lib` is not a copilot dep).
  A one-line pointer comment notes the deliberate rule-duplication so both move together if GLSL
  comment rules ever change.
- **`shaderbox/app.py`** (`_copilot_apply_shader_edit::_on_main`): replace `src.count(old_str)` +
  `src.replace(old_str, new_str)` with `token_match` + non-overlapping byte-span splicing (Design 5).
  The `matches==0` hint branch is unchanged; the ambiguity branch is unchanged (now counting
  token-run matches). Import `token_match` from the new leaf.
- **NEW `tests/test_glsl_lex.py`:** unit tests for the lexer + matcher (no GL, no App) — see Manual
  verification. This is the primary verification surface (the matcher is pure and fully unit-testable).
- **`tests/test_copilot_loop.py`:** the `_fake_caps.apply_edit` models the App with
  `str.count`/`str.replace`. Re-point it at the REAL `token_match` (import from `glsl_lex`) + the same
  span-splice, so the fake stays faithful to production by construction, not a hand-copied second
  matcher. Existing assertions unchanged (byte-exact inputs still match). Add ONE new test: a
  whitespace-divergent `old_str` against the seeded source applies (matches=1) — the headline behavior.
- **`shaderbox/copilot/capabilities.py`** (`EditResult` comment only): the inline comment on `matches`
  ("occurrences of old_str found") becomes "token-run matches" — the FIELD is unchanged (Design 6), the
  comment is corrected to state the now (conventions `## Code rules`). No type/shape change.
- **Docs:** `roadmap.md` (020 row note + Active-context: step 1 of the 2-step sequence done, step 2
  = Slice 2 is now next); this spec; the `12_edit_robustness.md` hint note gets a one-line "now the
  fallback under 13" pointer if it claims to be the primary path.

## Manual verification
- `make check` + `make smoke` green.
- **`tests/test_glsl_lex.py` (the real proof — pure unit tests):**
  - *Lexer basics:* `glsl_lex("vec3 p = u_pos;")` → IDENT `vec3`, IDENT `p`, PUNCT `=`, IDENT `u_pos`,
    PUNCT `;` with correct spans; round-trip each token's `src[start:end] == raw`.
  - *Maximal munch (I2):* `a<=b` → IDENT `a`, PUNCT `<=`, IDENT `b` (NOT `<` then `=`); `>>=` → one
    PUNCT (NOT `>>` then `=`).
  - *Comments + whitespace skipped:* a source with `//` and `/* */` comments and varied indentation
    lexes to the same `(kind, raw)` stream as the comment/whitespace-stripped version, AND every
    token's span still indexes the ORIGINAL source (`src[start:end] == raw`, proving the positional
    skip — not a pre-strip).
  - *Number class — accept table:* `1`, `1u`, `1.0`, `.5`, `1.`, `1e3`, `1.0e-3`, `2.0f`, `1.0lf`,
    `0xFF`, `0xFFu`, `0xFe` each lex as a SINGLE NUMBER with correct `raw`.
  - *Number class — reject table (FALSE-match guards, A5):* `a-1` → IDENT `a`, PUNCT `-`, NUMBER `1`
    (the `-` is NOT absorbed as a signed number); `1.0e-3` keeps its exponent sign in the one NUMBER
    (contrast: it does NOT split at `-`); a lone `.` (e.g. in `a.b`) → PUNCT `.`, never a NUMBER;
    `.5` ≠ `. 5` (the first is one NUMBER, the second is PUNCT `.` + NUMBER `5` — so they do NOT
    token-match each other).
  - *Raw-text honesty (I3):* `1.0` and `1.00` produce DIFFERENT tokens (raw differs) → `token_match`
    of one against the other → 0 spans (a numerically-different lexeme must NOT match).
  - *`token_match` whitespace-invariance:* source `"  vec3  p\t=\tu_pos ;"` matches `old_str`
    `"vec3 p = u_pos;"` → exactly one span, and the span's bytes are the SOURCE's (with the source's
    odd spacing), so a `src[:start] + new + src[end:]` splice preserves the rest of the file.
  - *Uniqueness / wrong-region guard (I4):* an `old_str` that token-matches two whitespace-variant
    regions → 2 spans (the apply path's `>1` ambiguity rule then fires — proving a token-ambiguous
    `old_str` surfaces as "not unique", NEVER a silent single wrong-region edit).
  - *Non-overlapping advance (A2):* `old_str` of token run `[a, a]` against source `a a a a` → 2
    spans, NOT 3 (after a match the scan resumes past the consumed run, matching `str.replace`).
  - *`replace_all` multi-span splice (A2/A1):* given ≥2 non-overlapping spans, the accumulate-splice
    produces the correct text and every untouched region is byte-preserved.
  - *Empty / degenerate needle (A3):* `token_match(src, "")`, `token_match(src, "   ")`, and
    `token_match(src, "// just a comment")` each → `[]` (0 matches), NEVER a zero-length span at byte 0.
  - *True 0-match:* a token typo (`u_poss`) or a dropped token → 0 spans (the hint path then runs).
- **Headless app-path check:** drive `_copilot_apply_shader_edit` (or via `run_turn` with a fake
  client) where `old_str` differs from the seeded source ONLY in indentation — assert the edit now
  APPLIES (matches=1, compiles) instead of returning a 0-match hint. (Extend `test_copilot_loop.py`.)
- **Live (maintainer, UN-headless):** re-run the original "showcase all uniforms" prompt that caused
  the slice-12 spiral; confirm a whitespace-divergent edit now applies on the first try (the
  transcript shows `edit_shader -> ok` not the hint). Hand to maintainer with a one-line note.

## Open questions for the user
- **O1 — Keep the `edit_shader` prompt unchanged (Design 6)?** I recommend YES: keep telling the model
  to copy exact bytes (always-correct instruction), let the lexer silently rescue whitespace
  divergence, do NOT advertise invariance (advertising invites sloppiness → more token-ambiguity
  failures). The alternative is a one-line "minor whitespace differences are tolerated" note in the
  description — louder but riskier. Default: unchanged.
- **O2 — Lexer home: `copilot/glsl_lex.py` vs a top-level `glsl_lex.py` leaf?** I default to under
  `copilot/` (Design 1) since the edit path is the only consumer today; a future non-copilot consumer
  graduates it. Flag if you'd rather place it top-level now (it's a pure leaf either way).

## Review history
- **Plan-locked (maintainer).** O1 = keep the `edit_shader` prompt unchanged (lexer is a silent safety
  net, not an advertised contract). O2 = lexer lives at `shaderbox/copilot/glsl_lex.py`.
- **Pre-impl review (1 agent, adversarial), all findings folded in:** A3 empty-needle → `[]` and A5
  number sub-grammar were correctness traps now pinned in §2 + Design 3; A2 non-overlapping
  left-to-right advance locked in §2/Design 5; B1 narrowed the "loses no current match" claim (embedded
  comment bytes inside a span get replaced); D1/D2 added the number reject-table, the wrong-region/I4
  guard, the non-overlap and `replace_all`-splice tests, and the empty-needle test; B2 corrected
  Design 4 (hint retained as a no-op safety belt, not "still fires"); B3 the test fake now imports the
  real `token_match`; C2 comment-skip is positional (not a `re.sub` pre-strip) with a duplication
  pointer; C4 the `EditResult.matches` comment is corrected. No redesign — all were spec-pinning gaps.
- **Post-impl review (1 agent, adversarial) → FIX-THEN-SHIP, the one real finding fixed:** the NUMBER
  branch gated on `str.isdigit()`, which is `True` for unicode chars `\d` rejects (`²`, `①`) → the
  float regex returned `None` → an `assert m is not None` crashed the lexer, contradicting the
  "never crash" contract. Fixed: gate the number probe on ASCII `[0-9]` and fall through to the
  single-char PUNCT path on a non-match (no assert). Added `test_lexer_never_crashes_on_odd_input`,
  `test_division_is_not_a_comment`, `test_unterminated_block_comment_terminates`, and strengthened the
  non-overlap test to assert disjoint spans. Two NITs (annotate the `_skip_trivia` local, span-value
  assert) folded in. Matcher safety design (I1-I4, no single wrong-region match) confirmed sound.
