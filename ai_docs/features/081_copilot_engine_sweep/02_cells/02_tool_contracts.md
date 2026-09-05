# Tool-contract defects — verified by the main session against the log

14 failures in 503 calls. Two are ENGINE defects (multi-model, same shape), the rest are thin.

## 1. read_shader rejects the documented address scheme — 4/11 fail, FOUR models
| model | args | result |
|---|---|---|
| codex-mini  | {'documents': ['c5c9#jfa']}            | "no such document(s) — check the project map for ids" |
| gpt-5.6-luna| {'documents': ['c5c9#cascade','c5c9#seed']} | same |
| hy4-preview | {'documents': ['5377#cascade']}        | same |
| deepseek    | {'documents': ['example:77a8#jump']}   | same |

address.py's own docstring: "The copilot working-set address scheme: a document is a bare id,
ONE PASS of a document is '<id>#<pass>' ... This module is the single round-trip parse/build
point, so a new kind is one change every tool inherits rather than a new tool per kind."
edit_shader / write_shader / probe_render all take `<id>#<pass>`. read_shader does NOT, and its
field description (tools/shader.py:20-25) never says so — it says "document ids ... and/or lib:
addresses".

FOUR INDEPENDENT MODELS made the identical generalisation. That is a contract defect, not model
error. The error message compounds it: "check the project map for ids" does not name the real
problem (a pass suffix on a tool that takes document ids), so the model has nothing to correct
toward.

## 2. set_uniform: 4/4 fail — 100%, the only tool that never once succeeded
| model | args | result |
|---|---|---|
| codex-mini x3 | u_pass_iteration = 0.0 | "'u_pass_iteration' is engine-owned ... cannot be set" |
| deepseek      | u_aspect = 1.3333      | "'u_aspect' is engine-owned ... cannot be set" |
Both models tried to set an ENGINE-OWNED uniform. The tool is reachable and advertised, but in
this task every legitimate use was engine-owned, so it was a pure trap. codex-mini repeated the
identical rejected call 3x in ONE turn (t9) — the no-op brake counts repeats, but this predates
/ did not stop it.

## 3. write_shader: 3/3 fails are ONE model sending an extra field
deepseek sends {'document':..., 'target':..., 'new_text':...}; schema (_WriteShaderArgs) has only
target + new_text. -> "invalid arguments - Extra inputs are not permitted". Pydantic strict.
Single-model, so: model error, but the message never names the offending field.

## 4. Thin ones (n=1 each, not systemic)
- set_pass dtype='' (codex-mini): schema accepts str, domain validator rejects. Schema/validator gap.
- add_pass 'canvas' already exists (luna): stale-plan retry.
- probe_render 'e11f#jfa' before jfa existed (gemini): ordering, not shape.
- edit_shader old_str drift 4/162 (2%): ordinary transcription drift, no mechanical trigger.

## Cold tools: 20 of 34 never called
Genuinely unpressured by this task: the whole telegram/youtube/publish/media/import family.
REAL scenario gap: render_image / render_video never called once — every visual check went
through probe_render, so the copilot's own render tools stayed cold across 11 attempts.
Borderline: read_script never called, yet write_script/edit_script were — scripts were authored
blind, never read back.

## Batching: the prompt says "BATCH independent calls into ONE step". Two models ignore it.
| model | reqs | mean calls/req | % multi-call |
|---|---|---|---|
| kimi | 40 | 1.45 | 52% (inflated by churn — same 2-call request x11) |
| hy4 | 81 | 1.23 | 26% |
| luna | 141 | 1.12 | 10% |
| deepseek | 50 | 0.96 | 16% |
| gemini | 99 | 0.94 | 4% |
| codex-mini | 50 | 0.78 | 0% — NEVER batched, 39/39 non-empty iterations were single-call |
Since input is re-sent per request, refusing to batch is a direct linear cost multiplier.

## Edit verification
Explicit probe after an edit, same turn: gemini 54%, deepseek 21%, luna 17%, hy4 11%,
codex-mini 0%, kimi 0%. CAVEAT (agent's own): edit results already carry compile errors and a
render-facts line, so "never verified" overstates it — the free feedback rides the edit result.

## kimi's churn, exact shape
Turn 2: set_pass('paint') / set_pass('seed') alternating, 24 calls, ALL ok:True, issued as ~11
separate 2-call requests. Turn 4: same with jfa/seed. Every call succeeds — set_pass is
idempotent — so this is pure request churn with no state advance, until the brake ends the turn.
