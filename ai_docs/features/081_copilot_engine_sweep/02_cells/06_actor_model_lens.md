# The findings read through the actor model (copilot-llm-agent-design skill)

The skill states ONE actor model with two reliable behaviours:
  1. It copies text verbatim; anything it must SYNTHESIZE is imprecise.
  2. It is blind outside its token stream; a fact not serialized this turn does not exist.
Every design choice is a corollary. Reading the sweep findings through it:

## F1 (fabrication / excuse vocabulary) — a corollary-2 violation BY THE ENGINE
Skill §2: "When the engine transforms the model's output before using it, feed the SANITIZED text
back... A stale verbatim copy of state is WORSE than no copy: the model trusts it."
_render_summary commits the model's own prose PLUS an engine-authored ledger into ONE assistant
message. The model cannot tell its words from the engine's — so on the next turn it trusts
engine-authored text as its own memory of having acted. kimi then reuses the engine's own sentence
("I hit the per-turn tool-call limit", agent.py:1208/1211) as a cover story.
=> The class is: THE ENGINE SPEAKS IN THE MODEL'S VOICE, in the model's own history slot.

CRITICAL GATE on any fix here — skill §4, "the better-model babysitter test":
  "would a strictly BETTER model still trip here? If NO, it's model carelessness the vendor
   amortizes for free — building a guard is permanent prompt tax against a transient flaw.
   Only a CLASS derivable from OUR pipeline's design (a missing affordance, a false tool message,
   a real coherence hole) earns a guard."
Per-model rates: kimi 0.43, hy4 0.14, deepseek 0.10, luna/gemini/codex/glm 0.00.
gemini and luna NEVER fabricated. So a naive "add a prompt rule" fix fails the test.
BUT: the engine authoring first-person prose into the assistant slot IS "a false tool message /
a real coherence hole derivable from our pipeline's design" — that half EARNS a fix. The
distinction the spec must hold: fix the COHERENCE HOLE (engine voice in the model's slot), do
NOT add a conscience rule telling the model to be honest. prompt.py:212 already IS such a rule
and it held for none of the three offenders — evidence that the conscience instrument is wrong.
Skill §4 again: "Facts as DATA succeed; facts as CONSCIENCE fail."

## F2 (read_shader address scheme) — a corollary-1 violation
Four models SYNTHESIZED an address by generalizing a documented scheme. The scheme says
<id>#<pass> is universal; one tool silently disagrees. The skill's §1 remedy shape applies:
"Detect degenerate inputs and return a TRUTHFUL error... never a silent match-anything."
The current error ("no such document(s) — check the project map for ids") is NOT truthful about
the actual problem. This is the cleanest "earns a fix" finding in the set: a missing affordance
plus a false error message, both derivable from our design, and a BETTER model would still trip
(the contract genuinely is inconsistent).

## F3/F4 (static floor, cache, iterations) — §3 and §6 are directly on point
§3: order blocks least-volatile -> most-volatile; the provider caches a PREFIX.
§6: "Billed input is the SUM across iterations (the cost driver)... Context-fullness is
iteration-0 input... They are different numbers."
This VALIDATES the sweep's re-send finding as the known mechanism, not a discovery — the skill
already names it. What is NEW is the measured magnitude (luna 11.6x) and that the turn boundary,
not a block edit, is where cache dies.
§6 also warns: "Cache-hit dominates re-run cost... never gate a cost decision on a single cold
sample; measure cold AND warm" via scripts/token_probe.py. Cell 3's brief already demands this
skepticism. 53% of corpus input was cached — a prefix that stays cached is nearly free.

## F5 (library invisible) — corollary 2, straightforwardly
The catalogue is what the model sees; if a helper's signature is not in the stream, the helper
does not exist. Hand-rolling is the RATIONAL act for a blind actor. §4: "a lazy model won't call
an optional inspection tool" — read_lib is exactly such a tool (3 calls in 503).

## F7/F8/F9 (station, skill drift, lost technique) — the repo's own meta-rule, not the actor's
Global CLAUDE.md: "A rule with no gate is a wish" and "Prefer tools over prose."
The fix ledger depends on the driver remembering to call it => prose, not a gate.
The skill's 1.07-1.08 ratio is a prose number nothing re-checks => it drifted, sign and all.

## What the skill FORBIDS the spec from proposing
- A new tool where enriching an existing tool's result would do (§4, tool-count discipline).
- A standing prompt rule as the instrument for an honesty problem (§4, facts-as-conscience-fail).
- A second guard to second-guess the model's intent: "If you're adding a SECOND guard to
  second-guess the model's intent, the CONTRACT is unsound — redesign, don't add guard #3."
  The brake family has already been patched three times (8fa8ece, c8960e1, b54baad). A FOURTH
  patch to the same mechanism must be read as the skill's redesign signal, not as guard #4.
- Any guard that only a WEAK model needs (the better-model test).
