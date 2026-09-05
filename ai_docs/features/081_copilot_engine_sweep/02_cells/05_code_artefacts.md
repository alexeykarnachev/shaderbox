# The code the models actually produced (11 project dirs, all alive)

## The two clean-compile failures each break exactly ONE mechanism

### e2e3 luna — "every shader compiled, nothing lit", 2 diagnostic turns found nothing
VERIFIED AT SOURCE by main session. proj-p2cewi_z .../cascade.frag.glsl:13
    if(lv>0. && dot(h,h)<.000001){ ...merge from u_prev... }
lv = u_pass_iterations-1-u_pass_iteration, so run 0 gives lv=5, the COARSEST level, which has
nothing above it. The guard is true for every level except 0, so the coarsest merges from
u_prev — the cascade texture's stale contents — and poisons the top of the chain that all five
later runs merge down from.

The finisher's inverse (proj-5t19vg0c hy4 .../cascade.frag.glsl:92):
    if (level >= u_pass_iterations - 1.0 + 0.5) { rad = vec3(0.0); } else { ...merge... }
ONE condition separates a lit scene from a black one. It compiled clean, and the model's own
diagnosis probed the wrong texture (note e2e3 t4: "FLAT black at t=0.5 twice, then 99% ink --
it read jfa or df instead").

### fb5 deepseek — "abandoned at the merge", streaks survived 3 correction rounds
Two defects, cascade.frag.glsl:
1. line 65: `float lo = BASE * (exp2(4.0*c) - 1.0)/3.0;`  => 16^c, spec says 4^c.
   Every other cascade in the corpus uses exp2(2.0*c) or pow(4.0,c).
2. line 35: `vec2 ulim = ...` computed, then a comment claims "clamp(upbase,0,lim) -- which
   upbase already is, so reuse it directly" and the clamp is DROPPED at lines 52-55. The claim
   is false: the bilinear neighbour taps (upbase + vec2(1,0) etc.) are not bounded by anything.
   The shipped reference clamps AT THE SAMPLE SITE. A comment that outlived its own refactor and
   justified the bug.
The run summary says "the reference addressing pasted verbatim" — the source shows the clamp
was dropped in the paste. The summary described the symptom, not the diff.

## Spec conformance verdicts
| attempt | model | verdict | items |
|---|---|---|---|
| e2e1 | hy4 | MEET | 11/11 |
| e2e2 | gemini | MEET | 11/11 |
| fb2/3 | luna | MEET | 10 pass, 1 partial |
| fb4 | hy4 | MEET | 10 pass, 1 fail (paint dtype) |
| fb7 | gemini | MEET | 9 pass, 2 fail (paint+canvas dtype) |
| fb5 | deepseek | PARTIAL | 7 pass, 2 fail, 2 absent |
| e2e3 | luna | PARTIAL | 9 pass, 1 fatal fail (merge guard) |
| fb8 | kimi | MISS | 5 pass, 2 fail, 4 absent |
| fb6 | glm | MISS | 3 pass, 7 absent |
| fb1 | codex-mini | MISS | 0 pass — two void main() in ONE file, doesn't compile |

## The library is invisible to the models — VERIFIED
SB_* helpers actually used, per project:
  proj-m3pqx_c0 (luna fb3, BUILT):   none
  proj-avrzhgnu (hy4 fb4, BUILT):    none
  proj-0pjh2fk_ (gemini fb7, BUILT): none
  proj-5t19vg0c (hy4 e2e1, BUILT):   SB_sd_segment
  proj-qqnhdwmz (gemini e2e2, BUILT):SB_center_uv, SB_sd_segment
  proj-p2cewi_z (luna e2e3, FAILED): none
ALL THREE rc_full_build finishers used ZERO helpers and hand-rolled SDF/segment math the
library ships. Even the two that used SB_sd_segment in paint hand-rolled the same distance in
canvas — the sibling pass, same session. read_lib was called 3 times in the whole corpus;
grep 28 times by 3 models.

## Dead stubs left behind — VERIFIED
main.frag.glsl still present in proj-avrzhgnu, proj-0pjh2fk_, proj-p2cewi_z, proj-yefb46x6.
In e2e3 it is still WIRED as a live graph node while output points at composite. gemini fb7
also left an orphaned "main" node in graph.json plus a trash/ dir from a discarded document.
The end-of-mission sweep turn did not remove these.

## What every model got right (no violations anywhere)
- u_pass_iteration / u_pass_iterations declared FLOAT in every attempt that declares them.
  Zero int violations corpus-wide — the d329d04 compile-error fix worked.
- The 4-ray packing idx = si*4+k, angle 2pi(idx+0.5)/rays: identical and correct in EVERY
  attempt that reached cascade, INCLUDING both failures. The packing was never the problem.
- ACES tonemap: hand-rolled Narkowicz everywhere, correct — and expected, since the library
  ships no tonemap helper.
- u_prev feedback: correct everywhere except kimi's jfa, which self-reads via u_jfa.
