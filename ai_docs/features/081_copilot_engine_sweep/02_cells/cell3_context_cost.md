# CELL 3 — F3 context cost. Moderator: opus. KEY CLAIM VERIFIED by main session.

## THE FINDING INVERTS: the token-SAVING mechanism is the token cost

VERIFIED INDEPENDENTLY (main session, own slicing of the 478 context records):
  requests where the tools array JUST GREW : n=35  mean cached 0.029  (86% under 5%)
  requests where it was unchanged          : n=374 mean cached 0.627  (7% under 5%)
(The cell measured 0.107 vs 0.648 with a slightly different grouping; both show the same
collapse. My slicing is the harsher one.)

MECHANISM: the tools array precedes every message, so adding ONE tool changes bytes at the very
front of the request and invalidates the ENTIRE cached prefix. registry.py:57-61 sorts by name
so the array is byte-stable FOR A GIVEN SET — correct as far as it goes, but a LARGER set is
different bytes at position 0.
COST: 490,039 excess uncached tokens on those requests vs the peer rate = $0.168 = 4.7% of the
entire $3.54 run — spent by the feature whose PURPOSE is to save tokens.

RE-PAID EVERY TURN: agent.py:483 `loaded_tools: set[str] = set()` is a LOCAL of run_turn —
turn-scoped by construction (verified). 26 of 37 load_tools calls re-requested a tool already
loaded earlier in the same attempt. agent.py:908 already DETECTS this and only tells the model.

## The floor's composition is the opposite of what F3 implied
billed input 10,336,145 | cached 52.4% | cost $3.5366
  tools= array   2,551,176 est tok = 24.7%   <- THE BIGGER HALF
  static prompt  1,836,549 est tok = 17.8%
  floor total    4,387,725         = 42.5%   (confirms F3's 42.7%)
  project_context 1,463,939        = 14.2%
=> A fix aimed at VISUAL CRAFT targets the smaller half of the smaller half.

CORRECTION TO MY F3: the corpus carries 23 tool names, not 34 (18 eager + 5 lazy). 35
ToolDefinition( sites exist in source but the rest never reached a request. "20 of 34 never
called" is really 6 ALWAYS-PRESENT-AND-NEVER-CALLED (publish_telegram, publish_youtube,
read_script, render_image, render_video, switch_document) + 2 lazy-and-never-called. Only the
first six are billed.

## Only 20.6% of floor tokens were ever billed FRESH
A constant prefix is nearly free. Any proposal that shrinks a constant prefix claims ~1/5 of its
gross number. THE ARITHMETIC:
| change | real (fresh-only) |
|---|---|
| cut the whole cold publish/render surface | $0.044 |
| cut all 6 never-called eager schemas | $0.050 |
| cut VISUAL CRAFT entirely | $0.037 |
| **fix the load_tools cache damage** | **$0.168** |
Every prose deletion recovers ~1% of the run. The cache defect is 4.7%.

## VISUAL CRAFT is NOT dead weight — it is the section paying for itself
prompt.py:192 ships an ACES tonemap literal, reproduced in SIX independent finisher composites
(proj-avrzhgnu, proj-0pjh2fk_, proj-5t19vg0c, proj-m3pqx_c0, proj-p2cewi_z, proj-qqnhdwmz), one
of them character-for-character INCLUDING WHITESPACE. The library ships no tonemap helper (F5),
so this prompt text is the ONLY source. Cross-checks F5 exactly.
BUT uptake is PER-DIRECTIVE, not per-section: the same lines 161-162 prescribe AA via fwidth
(0 of 17 projects) and dither (1). FORM & LIGHT and 3D & LOCAL FRAMES: zero hits, because no
attempt's task was 3D — a corpus COVERAGE GAP, not proof the guidance is dead.
Obeyed: EDITING one-write-per-file-per-step (0/86 violations); SCRIPTING accumulator watershed
(5/5 scripts implement real state, none faked with sin(t)); USING TOOLS load-then-call (37/37).
Violated: FEEDBACK's "do not re-apply the same edit" (luna re-issued byte-identical edit_shader
4x, twice with old_str == new_str); ADDRESSING's "call documents by NAME, never by id" (gemini
printed raw ids). HOW TO WORK's language rule is vacuous here — no Cyrillic ever appeared.

## Proposals (moderator's ranking)
P1 (DO FIRST) — make add_pass and set_pass EAGER (passes.py:113,129 eager=False -> True).
  CLASS: the two hot lazy tools stop forking the tools array mid-turn.
  COST: +1005 cached tok/req (real $0.034). RECOVERS $0.168. NET +$0.134/run = 3.8%.
  Also eliminates 37 tool calls.
  THE BREAK: assert no request's tools= array grows mid-turn for a multi-pass build; then flip
  add_pass back to eager=False, run a multi-pass scenario, watch the check fail on the turn
  load_tools fires; restore and name the break in the commit.
P2 — make publish_telegram/publish_youtube/render_image/render_video lazy; drop RENDER & PUBLISH
  and TELEGRAM+YOUTUBE prose to a one-line pointer at load_tools. Justify as SURFACE HYGIENE
  (~$0.044), not cost. Fixes a real asymmetry: telegram.py is uniformly eager=False (VERIFIED,
  lines 145-220) while those two publish tools are eager — an inconsistency, not a decision.
P3 — LEAVE VISUAL CRAFT ALWAYS-ON; prune per-directive. There is NO prose-lazy machinery
  (load_tools gates SCHEMAS only); building one would fork the cache into two lineages — the
  exact defect P1 exists to remove. ACES earns its six reproductions; fwidth AA at 0/17 is a
  WORDING problem, not a placement problem.

## A live instance of F7 found here, with a mechanism
attempts 4-8 share sha 3da796ab yet show TWO tools-block sizes, flipping mid-run at a turn
boundary (att4 t7, att7 t6, att8 t6). `git log -S` pins the text to c8960e1. The station
recorded a sha that was STALE WHILE THE TREE CHANGED UNDER IT. Feeds cell 6.

## False trails
- "The static block changed size, so an edit blew the cache" — no; two sizes = two code versions.
- "20 of 34 tools are billed dead weight" — only 6 are billed.
- "Cutting the static prompt saves 17.8%" — it saves 20.6% OF 17.8% ~= 3.7% gross, ~$0.037.
