---
name: copilot-llm-agent-design
description: "Read at the START of any work on the ShaderBox copilot or any LLM-agent design — proactively, before writing tool/prompt/edit code, not only when something is broken. Use when: adding or changing a copilot tool; editing the system prompt, the prompt-block tiers, or the replay history; touching the edit/write tools or any content-addressed matcher; handling a secret/credential through the agent; measuring or reasoning about agent token cost/latency; reacting to a trace where the model misbehaved (mislocated an edit, described a blank render, looped, ignored a fact); OR deciding whether a new guard/prompt-rule earns its place. It states the ONE actor-model the whole copilot is designed around and hangs every hard-won corollary off it — so a fix targets the class, not the instance. The detailed locked decisions live in conventions.md; this is the connective why."
user_invocable: true
---

# Designing for the LLM actor — the one model and its corollaries

Everything the ShaderBox copilot does well is a corollary of **one model of the actor**. State it once,
then read every design choice as falling out of it. The per-decision locked rules live in
`ai_docs/conventions.md ## Design decisions` (the copilot bullets) — this skill is the connective
tissue: *why* those rules are the same rule, so a new fix targets the CLASS, not the instance.

> Source: the 2026-06-13 nightly audit ("Every LLM-facing lesson is a corollary of one actor-model"),
> filed via `ai_docs/features/046_knowledge_base_refactor.md`. The editing-tool arc
> (020·14→036→038→039) is the recurring case study; the dogfood skill (`/dogfood`) is where the
> convergence discipline is exercised.

---

## The actor model

The model has exactly **two reliable behaviors**, and the whole design leans on them:

1. **It copies text verbatim.** Give it the exact bytes to reproduce and it will; ask it to *describe*
   a location, a count, a coordinate, an intent — anything it must synthesize rather than copy — and it
   will be imprecise in ways no guard can fully catch.
2. **It is blind to anything outside its token stream.** If a fact isn't in the prompt/tool-result it
   sees THIS turn, it does not exist for the model. A flag in our code, a panel on screen, a file on
   disk — invisible unless serialized into the stream.

Two failure modes follow directly: **don't make the actor synthesize what it could copy** (corollary 1),
and **don't rely on it knowing what you never put in the stream** (corollary 2). Every section below is
one of these two, specialized.

---

## 1. Content over coordinates (corollary 1 — copies verbatim)

**Address source by CONTENT, never by location.** A line number / anchor quote is something the model
must SYNTHESIZE and reproduce precisely; it inherits the model's imprecision, and the engine then either
trusts it (silent mislocation) or second-guesses it (a guard pile + false rejects). Content addressing
is safe BY CONSTRUCTION: you can only replace what you quoted verbatim; every failure is LOUD (no-match
/ multi-match), not silent. The line/anchor scheme was built (020·14), re-anchored (036), grew five
guards (038), and was deleted whole by 039 for `edit_shader` (old_str/new_str) + `write_shader`
(whole-file). Two more rules from that arc:

- **Match on the token stream, but DON'T ADVERTISE invariance.** A whitespace-invariant token match
  (`copilot/glsl_lex.py`) lets a model's slightly-reformatted `old_str` succeed — but if you TELL the
  model "whitespace doesn't matter," it gets sloppier in ways the matcher can't absorb. Be lenient
  silently; keep the contract strict in the prompt.
- **Detect degenerate inputs and return a TRUTHFUL error.** A comment-only `old_str` that lexes to zero
  tokens, an empty `new_str` whole-function delete — enumerate these and reject with a message the model
  can act on, never a silent match-anything.

Locked detail: `conventions.md` "Copilot source edits are content-addressed ONLY". Revisit only per
039's Out-of-scope trigger, and then via a verified number+text checksum, never bare anchors.

---

## 2. Feed back the text it ACTUALLY produced (corollary 2 — blind outside the stream)

When the engine transforms the model's output before using it (sanitizing glyphs, normalizing escapes),
feed the SANITIZED text back into the history — the model must see what actually happened, not what it
typed. A stale verbatim copy of state is **worse than no copy**: the model trusts it. So the replay
`history` is natural-language-only (user message + one engine-derived turn-summary), and the full source
is re-fetched LIVE each turn, never persisted (`conventions.md` "The copilot replay `history` is
NATURAL-LANGUAGE ONLY"). The working-set scratchpad rebuilds the current source every iteration for the
same reason — the live value, never a remembered one.

---

## 3. Prompt assembly for prefix-caching — least-volatile first

The provider caches a PREFIX of the prompt; a cache hit is ~4× cheaper (not free, ~5 min TTL). So order
blocks **least-volatile → most-volatile** so the stable head stays cacheable across turns:
`[static policy < rare project facts < dialogue < pending per-turn state]`.

- **Never put per-turn volatile state in the system prompt** — it busts the cache for the whole prefix.
- **Logical rank ≠ physical position**: a block's importance doesn't decide where it goes; its
  VOLATILITY does.
- **A build-time-frozen block is WRITE-ONLY** — if it must reflect live context, it has to be a
  loop-time closure rebuilt each iteration, not a string captured once at construction.
- **Tools serialize in sorted order** so the tool block is byte-stable across turns (an unsorted dict
  reorders and busts the cache).

The block-prompt machinery (`copilot/prompt.py` `PromptBlock`/`Volatility`/`build_prompt`) composes
these; a new tier is a named block at its volatility rank, never a string-concat. Locked detail:
`conventions.md` (the NL-history bullet, invariant 3).

---

## 4. Fact / hint / guard taxonomy + tool-count discipline

Three ways to influence the actor, in descending reliability:

- **Facts as DATA succeed; facts as CONSCIENCE fail.** A render-result fact line ("NOTHING is visible")
  rides the channel the model already reads and steers it. A standing prompt rule asking it to "always
  check before describing" is a conscience plea a cheap model ignores. Put the fact on the EXISTING
  channel (the tool result) rather than adding a new opt-in inspect tool the model won't call.
- **Tool count must not grow casually.** A lazy model won't call an optional inspection tool, and every
  tool's description is re-billed on every iteration (it dilutes attention from the load-bearing rules).
  Prefer enriching an existing tool's result over a new tool. (Cost mechanics: §6.)
- **The better-model babysitter test — the gate for every reactive guard.** When a trace shows the
  agent misbehaving, ask: *would a strictly BETTER model still trip here?* If NO, it's model
  carelessness the vendor amortizes for free — building a guard is permanent prompt tax against a
  transient flaw. Only a CLASS derivable from OUR pipeline's design (a missing affordance, a false tool
  message, a real coherence hole) earns a guard. Two hard corollaries: never change a destructive op's
  behavior on a heuristic GUESS the model can't see (a silent clamp masks the error — a loud reject lets
  it self-correct); a provider/transport quirk gets a tool-side invisible normalizer at the parse
  boundary, never a standing prompt rule.
- **The trace-patch STOPPING RULE.** Stop the review-and-patch loop once a fresh session has no terminal
  failure, no NEW failure class, and only residuals that fail the better-model test. A new INSTANCE of a
  known class is NOT a trigger for another round; a new CLASS in a different session is. (The 020·29
  overfit audit cut 3.5 of 6 proposed guards under this rule.)

Locked detail: `conventions.md` "A copilot tooling/prompt GUARD earns its place only if a strictly
BETTER model would still need it." The stopping rule + convergence discipline are exercised in `/dogfood`.

---

## 5. A secret travels on its OWN typed channel, end to end (corollary 2, inverted)

The model must NEVER see a secret, and the secret must never leak into anything the model's stream
touches. So a credential gets a DEDICATED typed path the whole way: a `GateResponse.secret` distinct
from the normal confirm answer, a separate `execute(secret=)` parameter (not folded into the args dict
the model authored), a REDACTED echo into history (prefix only — the live store holds the full value),
and the input buffer is never persisted. Don't reuse the generic confirm gate or smuggle the secret
through a normal arg. Locked detail: `conventions.md` (the `GateKind.CREDENTIAL` path, feature 020·19).

---

## 6. Agent cost measurement — three facts people conflate

- **Billed input is the SUM across iterations** (the cost driver — every iteration re-sends the tools
  block + accumulated history). **Context-fullness is iteration-0 input** (the "when do I compact?"
  driver). They are different numbers; a gauge that reads the summed rollup pegs full on any multi-step
  turn (025 caught this at review).
- **Cache-hit dominates re-run cost.** A repeated prefix reports ~99% cached (~4× cheaper) within a
  turn and within the ~5 min TTL — so a cold first sample and a warm re-run give OPPOSITE cost verdicts.
  Never gate a cost decision on a single cold sample; measure with the real probe (`scripts/token_probe.py`),
  cold AND warm.
- **A visualization must answer exactly ONE named question.** If you can't name the question (028's "when
  do I compact?"), you're showing data, not signal — don't build it.

---

## How to use this when a trace looks wrong

1. Name which actor-behavior the failure violates: did we make it SYNTHESIZE something (→ §1/§5, redesign
   so it copies), or rely on a fact NOT in its stream (→ §2/§3/§4, put the fact on the channel)?
2. Apply the better-model test (§4) before writing any guard. If a better model wouldn't trip, log the
   datum and stop — don't patch.
3. If you're adding a SECOND guard to second-guess the model's intent, the CONTRACT is unsound — go to
   `conventions.md` "Structural impossibility over guard-piles" and redesign, don't add guard #3.
