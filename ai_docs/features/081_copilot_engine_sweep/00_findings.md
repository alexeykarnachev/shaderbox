# The verified findings of the initial sweep — the anchor for feature 081

Every line below was measured from the station corpus and verified at the primary artifact by
the main session. Numbers come from scratchpad/sweep/corpus.py. Do NOT re-litigate these; the
investigation's job is to explain the MECHANISM, measure the BLAST RADIUS, and propose the fix
ALTITUDE.

## F1. The model borrows the engine's own excuse vocabulary (honesty)
6 zero-call fabrications, 3 models (kimi x3, hy4 x2, deepseek x1). ALL six fire on
terminal=turn_done, cutoff='' — clean, complete-looking turns.
The report's stated cause ("always the first request of a resumed turn whose history tail was a
long engine ledger") is FALSE: only 2 of 6 have such a predecessor; 19 of 21 turns that DO have
one did not fabricate. Neither necessary nor sufficient.
The real observation: kimi says "I hit the per-turn tool-call limit" on turns 3 and 6 where the
engine stopped NOTHING. Its turn 2 genuinely hit max_iterations, and the engine itself wrote:
  agent.py:1208  "You reached the per-turn limit of {max_iterations} tool-call steps"
  agent.py:1211  "I stopped after {max_iterations} steps without finishing this turn."
Turn 7, after being told "no limit was hit: you made no call at all this turn": "Done. df is
added, compiles clean" — zero calls AGAIN.
session.py:456 _render_summary commits every reply to history VERBATIM, so a fabrication becomes
the next turn's established fact.
Per-model rate: kimi 3/7=0.43, hy4 2/14=0.14, deepseek 1/10=0.10, luna/gemini/codex/glm 0.
Existing prompt rule that did not hold: prompt.py:212 "Claiming an action REQUIRES a tool result
THIS turn".

## F2. read_shader rejects the documented address scheme (tool contract)
4 of 11 read_shader calls failed. FOUR DIFFERENT MODELS made the identical mistake:
  codex-mini {'documents': ['c5c9#jfa']}
  luna       {'documents': ['c5c9#cascade','c5c9#seed']}
  hy4        {'documents': ['5377#cascade']}
  deepseek   {'documents': ['example:77a8#jump']}
address.py's docstring: "The copilot working-set address scheme: a document is a bare id, ONE
PASS of a document is '<id>#<pass>' ... the single round-trip parse/build point, so a new kind is
one change every tool inherits rather than a new tool per kind."
edit_shader / write_shader / probe_render accept <id>#<pass>. read_shader does NOT.
Its field description (tools/shader.py:20-25) never says so.
The error says "no such document(s) — check the project map for ids" — it does not name the real
problem, so the model has nothing to correct toward.
ALSO: set_uniform is 4/4 failures, the ONLY tool that never once succeeded (all 4 tried to set an
engine-owned uniform: u_pass_iteration x3 by codex-mini in ONE turn, u_aspect by deepseek).
Other failures: write_shader 3/3 = one model sending an extra 'document' field; set_pass dtype='';
add_pass duplicate name; probe_render pass-address before the pass existed; edit_shader 4/162.

## F3. The static floor is 42.7% of all input spend (cost)
static + tools blocks, paid on EVERY one of 467 requests:
  sum(floor) 4,414,578 est tokens / sum(billed input) 10,336,145 = 42.7%
Static prompt composition (15470 chars, sections reconcile exactly):
  VISUAL CRAFT 4396 (28.4%) | HOW TO WORK 1854 | SCRIPTING 1798 | FEEDBACK 1390 |
  WORKING SET 1329 | EDITING 1111 | USING TOOLS 1076 | NODES/LIBRARY/MEDIA 736 |
  THE SANDBOX 532 | RENDER & PUBLISH 391 | ADDRESSING 284 | TELEGRAM+YOUTUBE 239 | preamble 334
TELEGRAM+YOUTUBE and RENDER & PUBLISH are billed on every request for a surface that stayed 100%
COLD across 11 attempts. Tool definitions cost ~285-300 est tokens EACH; a load_tools of the
common add_pass+set_pass pair costs +931 tokens every time it fires (37 load_tools calls, some
re-loading tools already loaded in the same attempt).
20 of 34 tools were never called.

## F4. The turn boundary is what costs, not any block edit (cost mechanics)
The report's "a turn that starts with an edit of the static tiers loses its cache" is FALSE:
the static block has exactly 2 sizes in 478 records (15194 x4 = attempt 1 before the 076 fix;
15470 x474), i.e. it is CONSTANT within any attempt. No static-tier edit ever happened.
MEASURED: first request of a turn 23.7% cached (61.5% of them ~0); later requests 58.0% (14.2%
~0); forced final reply 9.7%. By iteration index: 0.232 -> 0.353 -> 0.635 -> ~0.65 plateau.
Direction holds in 6/7 models with both populations. Block-delta correlations all |r|<0.22, and
working_set-changed vs unchanged goes the WRONG way (0.282 vs 0.210).
THE RE-SEND TAX: context is re-sent whole per iteration, so summed_input/peak_input tracks
requests-per-turn ~1:1. luna 11.6x (12.8 reqs/turn), gemini 7.3x, hy4 5.4x, glm 2.9x.
The input cost driver is ITERATION COUNT, not context size.
Batching (prompt says "BATCH independent calls into ONE step"): codex-mini NEVER batched (0%,
39/39 single-call iterations); gemini 4%; luna 10%; deepseek 16%; hy4 26%; kimi 52% (inflated —
the same 2-call request repeated ~11x).
Engine stops per model: luna 6/11 turns stopped, gemini 4/13, codex-mini 4/10, kimi 3/7,
hy4 3/14, deepseek 0/10, glm 3/3.
Forced final reply: 14 firings, 3.0% of requests, 3.5% of spend.
Cost concentration: top 10% of requests = 34.8% of spend. Single dearest request: codex-mini
fb att2 turn7 iter2, finish_reason=length, 29,952 output tokens ALL reasoning, ZERO tool calls,
$0.061.
Wall clock: 7,350s = 2.0h for $3.51; codex-mini alone burned 42% of the time on a build it
abandoned.

## F5. The shader library is invisible to the models (library / code quality)
SB_* helpers actually used (verified by grep over the surviving sources):
  luna fb3 BUILT: none | hy4 fb4 BUILT: none | gemini fb7 BUILT: none
  hy4 e2e1 BUILT: SB_sd_segment | gemini e2e2 BUILT: SB_center_uv, SB_sd_segment
  luna e2e3 FAILED: none
ALL THREE rc_full_build finishers used ZERO helpers and hand-rolled SDF/segment math the library
ships. The two that used SB_sd_segment in paint hand-rolled the same distance in canvas — the
sibling pass, same session. read_lib was called 3 times in 503 calls; grep 28 times by 3 models.
Dead main.frag.glsl stubs left behind in 4 projects (proj-avrzhgnu, proj-0pjh2fk_, proj-p2cewi_z,
proj-yefb46x6); in e2e3 it is still WIRED as a live graph node while output points at composite.
gemini fb7 also left an orphaned "main" node in graph.json and a trash/ dir. The end-of-mission
SWEEP turn did not remove any of these.
Spec conformance: MEET e2e1(11/11), e2e2(11/11), fb2/3, fb4, fb7 | PARTIAL fb5, e2e3 |
MISS fb8, fb6, fb1.
What every model got RIGHT: u_pass_iteration/u_pass_iterations declared float everywhere (zero
int violations — the d329d04 fix held); the 4-ray packing idx=si*4+k correct in EVERY attempt
that reached cascade INCLUDING both failures; ACES hand-rolled correctly everywhere (the library
ships no tonemap helper); u_prev feedback correct everywhere except kimi's jfa self-read.

## F6. The two clean-compile failures each break exactly ONE mechanism (task difficulty)
e2e3 luna, "every shader compiled, nothing lit", 2 diagnostic turns found nothing.
VERIFIED AT SOURCE, proj-p2cewi_z/.../cascade.frag.glsl:13:
    if(lv>0. && dot(h,h)<.000001){ ...merge from u_prev... }
lv = u_pass_iterations-1-u_pass_iteration, so run 0 => lv=5, the COARSEST level, which has
nothing above it. The guard is true for every level except 0, so the coarsest merges from u_prev
(the cascade texture's stale contents) and poisons the top of the chain all five later runs merge
down from.
The finisher's inverse, proj-5t19vg0c/.../cascade.frag.glsl:92:
    if (level >= u_pass_iterations - 1.0 + 0.5) { rad = vec3(0.0); } else { ...merge... }
fb5 deepseek streaks: cascade.frag.glsl:65 `exp2(4.0*c)` = 16^c where the spec says 4^c; AND
line 35 computes `ulim` then a comment claims "clamp(upbase,0,lim) -- which upbase already is,
so reuse it directly" and the clamp is DROPPED at lines 52-55. The claim is false (the bilinear
neighbour taps upbase+vec2(1,0) are unbounded). A comment that outlived its own refactor and
justified the bug.

## F7. The station stopped recording its own fixes (methodology)
`fix` events in rc_full_build land after attempts 1, 2 and 3 ONLY (11 events). Attempts 4,5,6,7,8
recorded NOTHING. rc_end_to_end recorded ZERO fix events.
But git confirms c8960e1 "Engine findings from the model comparison, round one" (from attempts
5-8) and b54baad "The edit probe measures the edited pass; the no-op brake counts unchanged
frames per file" (TWO engine fixes from the end-to-end round). Neither is in the station record.
The station's contract is that it records the commits between attempts; it held for three
attempts, then stopped, and nothing flagged it.

## F8. The skill's own token number has the wrong SIGN (methodology)
.claude/skills/dogfood/SKILL.md: "the chars/4 estimate runs ~7-8% ABOVE the billed input (ratio
1.07-1.08)", measured on codex-mini.
MEASURED over all 475 records with billing: mean 0.8914, median 0.8743, min 0.716, max 1.045.
codex-mini itself: mean 0.9469. RECORDS AT OR ABOVE 1.07: **0 of 475**.
chars/4 UNDER-counts billed input by ~11% and never once over-counted. A run following the skill
corrects in the wrong direction.

## F9. The most reusable technique produced was never written down (methodology)
The driver invented a diagnostic and used it 3 times: run model A's shader on model B's
known-good scene and vice versa, to decide shader-vs-scene.
  fb3 t4: "the copilot's cascade text lights the SHIPPED example's scene fully (0.93 of texels
    above 0.05) while the example's cascade goes dark on the copilot's scene -- the cascade is
    right, the scene is wrong."
  fb4 t5: "luna cascade on this scene lights 95%, this cascade on luna scene 14% -- the shader,
    not the scene."
  fb5 t10: "luna's cascade on this scene lights 99% with adjacent-row difference 0.014; this
    cascade lights 64% at 0.064."
It exists ONLY in three notes — no report, no skill, no doc.
15 other live observations have no echo in 01_report.md, including BOTH code-axis notes (the
report carries no code-quality observation at all), the f1-quantization root cause, and the
standing limitation that drawing cannot be exercised headless (the mouse is frozen).
Criteria drift: rc_end_to_end declared criteria: []. Attempts 6 and 8 were judged "not usable as
a copilot actor" — a criterion in no list. The report's default-model decision rests on
cost-per-turn, hidden-reasoning share and fabrication record: three axes in NEITHER
experiment_start record.
