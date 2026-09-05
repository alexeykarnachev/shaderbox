# 081 — the copilot engine sweep over the 077 corpus

The 077 model comparison left a station corpus nobody had mined: 68 turns, 467 requests, 503 tool
calls, 478 context records, 11 surviving project dirs, $3.51. This feature is what that corpus
says about the ENGINE. The findings it is built on are in `00_findings.md`; each was measured from
the log and then verified at the primary artifact, and six investigation cells then took one
cluster each to establish mechanism, blast radius and fix altitude.

Every number below traces to the corpus loader recorded in `00_findings.md`. Where a cell
overturned a finding the sweep started from, the correction is stated rather than quietly dropped —
three of the sharpest results here are corrections.

## Goal

Close the engine defects the corpus demonstrates, at the altitude the evidence supports, and leave
a gate behind each one. Explicitly NOT to make the copilot write better shaders: every fix below
either removes a contract the models could not satisfy, or makes a fact visible that the engine
already knows.

## What the corpus overturned

Four claims that were documented as established, disproven by measurement. They are listed first
because two of them would otherwise steer this feature wrong.

1. **"A turn that starts with an edit of the static tiers loses its cache."** The static block has
   two sizes in 478 records (15194 in attempt 1's four requests, pre-076; 15470 in the other 474) —
   constant within any attempt. No static-tier edit ever happened. What is true: first request of a
   turn 23.7% cached, later requests 58.0%, forced final reply 9.7%. The TURN BOUNDARY is the cost.
2. **"Five zero-call fabrications, always after a long engine ledger."** Six, not five; and only 2
   of 6 sit behind a long ledger or forced end, while 19 of 21 turns that DO have such a
   predecessor did not fabricate. Neither necessary nor sufficient.
3. **The dogfood skill's "chars/4 runs 7-8% ABOVE billed (1.07-1.08)".** Measured over all 475
   records with billing: mean 0.8914, codex-mini itself 0.9469, records at or above 1.07: **0 of
   475**. The sign is wrong — chars/4 UNDER-counts by ~11%.
4. **"The station stopped recording fixes after attempt 3."** An overcount by this sweep. Attempts
   4-8 opened within 3.7 seconds of each other on one identical sha (a parallel fan-out), so
   `commits_between` correctly returned empty. The real hole is three commits (`c8960e1`,
   `60e089b`, `b54baad`) that land BETWEEN rounds, after every attempt of their round was opened,
   and are therefore unreachable by any future sweep.

## Design decisions

**D1 — `read_shader` accepts `<id>#<pass>`, and an example returns its non-output passes.**
`read_shaders` (`backend.py:722-724`) reads `document.render_pass` — the OUTPUT pass only. The
shipped Radiance Cascades example has six passes totalling 287 lines with `composite` as output, so
`read_shader` on the canonical reference returns 28 lines of presentation code. The 110-line
`cascade.frag.glsl` — the algorithm every model had to reinvent, and the one BOTH clean-compile
failures got wrong — was unreachable by every tool in the copilot. Four independent models
(codex-mini, luna, hy4, deepseek) passed a pass address to `read_shader` and were told "no such
document(s)"; across the corpus 249 pass-addressed calls succeed and only these 4 fail. The reject
enumerates the real pass names, in the backend where `document.passes` is in scope
(`backend.py:1670-1674` is the shape to copy). A project pass-read joins the working set as its
DOCUMENT, mirroring `read_working_set` (`backend.py:774-777`, D11); only the example path returns
per-pass listings.

**D2 — the pass-address contract gets one registry-enumerated gate.** `c8960e1` fixed
`probe_render` and wrote into `tests/test_copilot_pass_tools.py:115-116` that "every other tool
takes the pass address". That was false when written — `git show c8960e1 --stat` touches
`tools/inspect.py`, never `tools/shader.py` — and it now reads as established. The replacement
enumerates the domain from `registry.definitions()`: every tool with an address-carrying field is
in exactly one of PASS_AWARE or DOCUMENT_ONLY, so a new tool is a test failure until classified.
This also forces the decision on the latent sibling, `render_image`/`render_video`.

**D3 — the split does NOT move into `_copilot_resolve_document_id`.** It has ~13 callers, several
of which must reject a pass address loudly: deleting or renaming `c5c9#cascade` is a category
error, and silently downgrading it to the document is what `conventions.md:686-689` forbids
("NEVER change a destructive edit's behavior on a heuristic GUESS the model can't see"). The shared
root is the source-read seam, matching `conventions.md:670-678` — the convention 065 did not apply.

**D4 — a zero-call reply that claims work is answered by the engine, not by a prompt rule.**
On zero-call turns the models forged engine telemetry byte-for-byte. The provenance is proven: the
tool emits `ok — compiled clean` with U+2014, the models wrote `ok -- compiled clean` with ASCII,
and `sanitize.py:24` maps that character only on the commit path into history — em-dash form in
replies 0, ASCII form 3. The models copied HISTORY'S RENDERING, so `_render_summary`
(`session.py:456-469`), which joins model prose and engine ledger into one unmarked assistant
message, is the learning channel. No guard can see this shape: `agent.py:752`
(`silent_after_tool = not text_buf and total_tool_calls > 0`) is the only line that inspects reply
text and is the exact inverse; every other brake lives inside the tool-call loop, so a zero-call
reply falls through `agent.py:744` to the ordinary success terminal at `agent.py:857`. The fix
branches on an integer already in scope at that line, injects one `[engine]` ground-truth message,
and re-streams ONCE. It does not classify the prose.

**D5 — no keyword scan of reply text, and no new honesty prompt rule.** A scan for
"done"/"added"/"compiles clean" is a checker that silently narrows its own domain, fires on honest
turns, and has no enum to mutation-test against. And `prompt.py:212` already IS the honesty rule;
it held for none of the three offenders, and in-context correction failed within one turn (kimi t7
answered a flat "you made no call at all this turn" with another fabrication). Per the design
skill: facts as data succeed, facts as conscience fail. The rule stays, but stops being counted as
a guard.

**D6 — `add_pass` and `set_pass` become eager.** `load_tools` — the mechanism built to save tokens —
is what destroys the cache. Requests where the tools array just grew: 35, mean cached **2.9%** (86%
under 5%); requests where it was unchanged: 374, mean **62.7%**. The tools array precedes every
message, so one added tool changes byte zero and invalidates the whole prefix. Cost: 490k excess
uncached tokens = $0.168 = 4.7% of the run, spent by the token-saving feature. It is re-paid every
turn because `loaded_tools` is a `run_turn` local (`agent.py:482`), and 26 of 37 `load_tools` calls
re-requested a tool already loaded that attempt. Making the two hot pass tools eager costs ~1005
cached tokens per request (real ~$0.034) and recovers $0.168 — net +3.8%, and removes 37 tool calls.

**D7 — the batching instruction is re-justified by COST, not by steps.** `prompt.py:214` sells
batching as conserving steps ("steps are the scarce budget"); a model with 11 of 16 steps left has
no reason to comply. Iteration count is the input cost driver (r=0.934 against r=0.434 for peak
context size), and 87.5% of all corpus input is paid past a turn's first iteration. The
counterfactual at kimi's requests-per-call rate is $0.9959 of $3.3865 = **29.4% of spend**, seven
times the cache lever. This is a prompt change verified by re-measuring requests-per-call, NOT a
gate — and no test may assert the prompt contains the word "BATCH".

**D8 — no fourth brake, and no cost/token brake.** luna, the model with the worst re-send ratio
(12.45 requests/turn), wastes the LEAST: only 4.0% of its beyond-first iterations made zero tool
calls, against deepseek 22.5%, codex-mini 19.4%, hy4 13.6%, gemini 10.7%. Its iterations are 96%
productive — it is un-batched, not looping, so a brake tuned to cut it would cut productive steps.
The three existing patches to this family teach the shape of the risk: `8fa8ece` built one global
counter, `c8960e1` widened what it counted into the same bucket, `b54baad` had to split the scope
because the global count false-fired and killed a legitimate seven-pass build on its second pass —
each hole created by the previous patch's scope choice. Three brakes have never fired at all
(`max_edit_retries`, `max_compile_failures`, `clean_edit_hard`). A cost brake additionally
manufactures a new stop reason for the model to borrow, which is the D4 failure.

**D9 — the context gauge reports the turn's real billed input, in the tooltip only.**
`TurnStats.context_tokens` is fed `first_input_tokens` (`agent.py:502`, "iter-0 context size for
the usage bar") and `context_gauge_readout` (`state.py:106-122`) prints it beside `cost_usd`, which
IS the turn total: two numbers on different bases in one tooltip. On luna's worst turn the gauge
read ~24k against 495,773 billed. `summed_input` appears **0 times** in `shaderbox/`, and
`context_gauge_readout` has **0 tests**. The fill fraction stays per-request fullness — summed
input is not comparable to a per-request budget, and merging them would make the bar meaningless.

**D10 — `PassView` carries reachability.** `widgets/pass_list.py:154-163` already computes the live
set via `evaluation_order` and dims unreachable tiles for the human; `PassView`
(`capabilities.py:212-220`) carries `name, address, listing, uniforms, errors, is_output` and no
reachability field, and `GraphError` appears **0 times** in `shaderbox/copilot/`. The user sees a
dimmed tile; the model sees nothing — which is why four dead `main.frag.glsl` stubs survived every
end-of-mission sweep turn, one still wired as a live graph node. This is one bool from a value the
codebase already computes, not new logic.

**D11 — `_validation_message` names the offending field.** `tools/registry.py:44-46` reads `msg`
off the pydantic error dict and discards the `loc` sitting on the same dict. deepseek failed
`write_shader` three times guessing the problem was content (it changed a comment, then changed
nothing) before dropping the extra `document` field. Two lines, covering all 35 tools.

**D12 — the engine-uniform list is generated, not retyped.** `_SET_UNIFORM_DESC`
(`tools/shader.py:160-169`) and `_CONVENTIONS` (`prompt_context.py:22-23`) name three of the five
engine uniforms; `u_pass_iteration` and `u_pass_iterations` appear on NO static surface and are
exactly what codex-mini spent 600 seconds of wall clock failing on (three identical rejected calls,
turn ended at `cutoff='time_budget'`). Generated from `ENGINE_UNIFORM_TYPES`
(`shaderbox/engine_uniforms.py:14-20`), stating the positive branch: to change what a pass counter does, edit
the shader or change `runs` via `set_pass`. `set_uniform`'s rejection message is exemplary and is
NOT touched.

**D13 — the station asserts its fix ledger against git.** `ledger_gap()` reports, per attempt, the
commits between the previous sha and this one carrying no `fix` record, plus commits after the last
attempt's sha; `build.py` renders a gap as a warning row (it already has a `.warn` span at
`build.py:224`) instead of an absent section, since today zero real commits and three unswept
commits render identically. The same row surfaces `dirty`, which `station.py:189` records on every
attempt and `Attempt` (`log.py:214-227`) has no field for — a write-only integrity signal that
fired on all 8 `rc_full_build` attempts, meaning those runs did not execute the code their sha
names.

**D14 — the estimator constant is measured.** `_CHARS_PER_TOKEN = 4` (`prompt.py:16`) is duplicated
as a bare `// 4` (`context_breakdown.py:128`) and has zero tests. Measured implied divisor across
475 records: 3.57 (per model 3.39-3.85). Set 3.6, import it at the second site so it stops being
two constants, and assert a band. `_trim_history` budgets with this estimator, so today's value is
unsafe in exactly the direction that matters if any bound is ever tightened.

**D15 — `lit_fraction` joins `judge.py`; the station still never judges.** The driver invented a
cross-scene diagnostic and used it three times to decide shader-vs-scene ("luna cascade on this
scene lights 95%, this cascade on luna scene 14% — the shader, not the scene"), and it exists in no
report, skill or doc. `judge.py` has seven primitives and none computes a lit fraction. A primitive
the driver calls BY HAND is the sanctioned shape; `conventions.md:909-916` is a settled decision
whose judging half says "revisit: never", so nothing here is wired into a verdict. The
cross-project source-copy half stays a manual procedure until a second run needs it.

## Out of scope

- **The turn-boundary cache miss** (ceiling 4.3% of input, less in dollars). The alternative —
  persisting the real tool exchange to keep the prefix stable — costs a permanently larger
  `dialogue` block re-paid forever and reintroduces the stale-source hazard `session.py:437-440`
  deliberately closed. **Trigger:** a corpus where first-of-turn requests exceed a third of spend.
- **Trimming.** 0 of 478 records trimmed; budget 150k against a ~37k worst case. **Trigger:** the
  first run that records a trim, or any tightening of `max_input_tokens`.
- **Cutting static prompt prose.** Only 20.6% of floor tokens are ever billed fresh, so every prose
  deletion recovers ~1% of a run. VISUAL CRAFT is the section paying for itself: `prompt.py:192`
  ships the ACES literal reproduced in six finisher composites, one character-for-character, and
  the library ships no tonemap helper. **Trigger:** a measured run where the floor's fresh share
  rises above half.
- **Renaming `target` to `document`.** `edit_shader`/`write_shader` are the only two tools using
  `target` against 16 using `document` — a real inconsistency, wider than this feature should
  decide. **Trigger:** the next feature that adds or reshapes an address-taking tool.
- **The seven never-called eager tools** (~1,548 tok/request, ~723k est tokens = 7% of billed
  input). D6 settles the two hot pass tools; the publish/render family is a separate trade-off, and
  `switch_document` is prompt-critical (`prompt.py:199-200`) regardless of its cold record.
  **Trigger:** a run that exercises the publish surface, or the next prompt-cost measurement.
- **Editing the dogfood skill.** The maintainer excluded it this round. D14's code half stands
  alone; the skill's two FALSE claims and two STALE ones are recorded in `00_findings.md` and
  `02_cells/` for the wave that reopens it.

## Files touched

| file | why |
|---|---|
| `shaderbox/copilot/backend.py` | D1 (`read_shaders` splits the address, reads `document.passes`, enumerating reject) |
| `shaderbox/copilot/tools/shader.py` | D1 (descriptions name `<id>#<pass>` and `example:`), D12 |
| `shaderbox/copilot/tools/registry.py` | D11 (`_validation_message` uses `loc`) |
| `shaderbox/copilot/prompt_context.py` | D12 (`_CONVENTIONS` generated) |
| `shaderbox/copilot/tools/passes.py` | D6 (`eager=True` on `add_pass`/`set_pass`) |
| `shaderbox/copilot/prompt.py` | D7 (batching justified by cost), D14 (`_CHARS_PER_TOKEN`) |
| `shaderbox/copilot/context_breakdown.py` | D14 (import the constant, drop the bare `// 4`) |
| `shaderbox/copilot/agent.py` | D4 (zero-call terminal), D9 (accumulate summed input) |
| `shaderbox/copilot/session.py` | D4 (`_render_summary` marks engine-authored text) |
| `shaderbox/copilot/state.py` | D9 (tooltip reports the turn's real input) |
| `shaderbox/copilot/capabilities.py` | D10 (`PassView` reachability) |
| `dogfood/report/station.py`, `log.py`, `build.py` | D13 (`ledger_gap`, `dirty` field, warning row) |
| `scripts/dogfood/judge.py` | D15 (`lit_fraction`) |
| `tests/test_copilot_pass_tools.py` | D2 (the false comment replaced by the enumerated gate) |
| `tests/…` | the breaks below |

## The break behind every gate

The repo's rule is that an unbroken gate is a wish, and that a gate is done when the thing it
guards has been broken, the gate has named it, and the break restored. Each row states the break
the commit must name.

| decision | the break |
|---|---|
| D1 | Delete the `split_pass_address` call from `read_shaders`; the pass-read test fails with today's "no such document(s)". Second: point the example read back at `render_pass`; the example test loses five of six passes. |
| D2 | Revert `probe_render`'s split (`backend.py:1666`); the gate names `probe_render`. Then add a throwaway tool with a `document` field and no classification; the gate fails naming it. |
| D4 | A fake client yielding text + `LLMDone("stop")` with no tool calls (the `_PlainClient` shape, `tests/test_copilot_loop.py:581`): the turn must not end as a bare `AgentTurnDone`. Delete the branch and watch it fail. Inverse: one real call plus the same prose still ends in ONE stream, or the gate has widened its domain to all turns. |
| D6 | Assert no request's `tools=` array grows mid-turn on a multi-pass build; flip `add_pass` back to `eager=False` and watch it fail on the turn `load_tools` fires. |
| D9 | A fake 3-iteration turn with known sizes: the readout reports their SUM; change one iteration's size and the number moves. Today's code reports iteration 0 and fails the second assertion. |
| D10 | Build a document with an unreachable pass; the working set names it. Cut the wire feeding the flag and the assertion fails. |
| D11 | The message for an extra `document` on `_WriteShaderArgs` contains "document"; revert to `msg`-only and it fails. |
| D12 | Every `ENGINE_UNIFORM_TYPES` key appears in the rendered description; add a sixth name to the table and it goes red before the description is regenerated. |
| D13 | Drop attempt 4's six `fix` records from the real 077 log; the check names exactly those shas. **And the one that matters:** replay the att4-att8 parallel burst and assert it reports ZERO gap — a check that flags correctly-empty windows is worse than none. |
| D14 | Set the constant back to 4; the band test fails naming the ratio. |
| D15 | A synthetic array with a known count above threshold returns the exact fraction; flip one texel and it moves by exactly 1/N. |
| D7 | **No gate.** A prompt change verified by re-measuring requests-per-call in a later run. A test asserting the prompt contains "BATCH" is the reader agreeing with itself and must not be written. |

## Open questions for the user

1. **Order.** The cells' independent first choices are D1 (the unreadable reference), D4 (the
   fabrication terminal), D6 (the cache collapse) and D13 (the ledger gap). D1 and D6 are the
   cheapest with the clearest evidence; D4 is the one that reaches the shipped default. Ship all
   four in one wave, or D1+D6 first and D4+D13 after a re-run confirms the measurement?
2. **D4's greeting case.** `prompt.py:213` blesses a zero-call reply for a greeting or a
   map-answerable question. Zero of 68 turns were that case, so the re-stream would fire on
   nothing here — but a conversational session is a different corpus. Accept one wasted cheap
   request on greetings, or gate the re-stream on the reply containing no question mark?
3. **A verification re-run.** D6 and D7 are measured claims about cost that only a fresh run can
   confirm, and the corpus cost $3.51 for two rounds. Worth one `rc_end_to_end` attempt on hy4
   after the wave lands (~$0.30), or leave verification to the next dogfood round?
