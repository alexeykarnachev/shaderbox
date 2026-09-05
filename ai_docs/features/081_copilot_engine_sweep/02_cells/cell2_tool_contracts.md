# CELL 2 — F2 tool contracts. Moderator: opus. KEY CLAIMS VERIFIED by main session.

## THE SHARPEST FINDING OF THE SWEEP: the reference shader was UNREADABLE
VERIFIED AT SOURCE (main session):
  backend.py:722-724  read_shaders uses `document.render_pass` — THE OUTPUT PASS ONLY.
  The shipped Radiance Cascades example has SIX passes, 287 lines:
    cascade 110 | jfa 57 | paint 46 | composite 28 | df 24 | seed 22
  Its output is `composite`. So read_shader on the canonical reference returns 28 of 287 lines —
  the presentation pass, whose own comment says "this pass is mostly presentation".
  THE 110-LINE cascade.frag.glsl — the exact algorithm every model had to reinvent, and the one
  BOTH F6 failures got wrong (luna's merge guard, deepseek's shell radii) — WAS UNREACHABLE BY
  EVERY TOOL IN THE COPILOT.
  split_pass_address is called at exactly THREE sites (VERIFIED): backend.py:774 (working set),
  :1666 (probe_render), :2359 (edit resolver). NOT on the read path.
=> This causally links F2 and F5: the models did not ignore the reference. They could not read it.
   deepseek's 8 consecutive greps after the failed `example:77a8#jump` read are the visible cost.

## The mechanism: a SCOPE bug with a documentation surface
The pass-address kind shipped as an EDIT-tool feature and was then advertised as AN ADDRESS.
git log 811a757 ("065: the copilot addresses a pass"): "the address scheme carries this and EVERY
EDIT TOOL inherits it through the single resolver."
Four surfaces show the model `c5c9#cascade`, all glossed with the verb EDIT:
  capabilities.py:216  `address: str  # the "<document>#<pass>" handle an EDIT TOOL takes`
  prompt.py:351        `=== PASS {name} (EDIT AS: {address})`
  prompt_context.py:68 `(* = output; EDIT ONE AS {id}#<pass>)`
  prompt.py:83         `EDIT a pass by its own <id>#<name> address`
  passes.py:106        `Then write_shader / edit_shader it at the <id>#<name> address.`
A model does not read "edit as: X" as a verb-scoped capability; it reads X IS THE NAME OF THIS
THING. The 065 commit says so itself one paragraph down: "A model is blind to anything outside
its token stream, so an address kind it is never SHOWN is unreachable." It was shown. It became
reachable. It was reached for on the READ path, where nobody implemented it.
CORPUS PROOF THE FORM IS THE IDIOM: 249 pass-addressed calls, 245 succeed, 4 fail — ALL on
read_shader. Seven models used the form; six used it successfully.
ALSO: all 7 successful read_shader calls in the corpus are the identical `example:77a8`. Every
read of a PROJECT document — the tool's stated purpose — failed.

## THE GATE THAT FAILED — a point fix shipped under a class-wide claim (VERIFIED)
tests/test_copilot_pass_tools.py:115-116, shipped BY c8960e1:
  "# gemini-3.8-flash on the station probed `<id>#jfa` and was told no such document; every
   other tool takes the pass address."
THAT SENTENCE WAS FALSE WHEN WRITTEN. `git show c8960e1 --stat` touches tools/inspect.py;
tools/shader.py is ABSENT (verified). c8960e1 is an ancestor of 134bbf0, the end-to-end round's
sha — so read_shader was still broken AFTER the fix that named its class.
This is the repo's own "a checker that quietly narrows its own domain" family, in its purest
form: the claim was written into a test COMMENT, where it now reads as established.

## The convention that already existed and was not applied
conventions.md:670-678: "A new addressable copilot SOURCE kind gets a `<kind>:` prefix + rides the
EXISTING read/grep, never a parallel tool ... a branch in `_copilot_resolve_source` + the
read/grep builders (the SAME ShaderView/GrepHit, one implementation)."
The pass kind is an addressable source kind. It got a branch in the EDIT resolver
(_copilot_resolve_target, backend.py:2359) and NONE in _copilot_resolve_source (backend.py:647).
The rule existed; the wave that added the kind did not apply it; nothing gated that.
SECOND INSTANCE, already filed and never fixed —
069_tutorial_walk_findings/reviews/wave_d_pre.md:172-174: "read_shaders compiles
document.render_pass only — the OUTPUT pass. Bloom's bright, blur, trail and scene are never
touched by it." Same function, same assumption, found twice by two independent routes.
=> The class is REAL (two members), not asserted.

## Blast radius: precisely TWO members, not a sweep
| consumer | pass address meaningful? | verdict |
|---|---|---|
| read_shaders (backend.py:680) | YES — a source read | **DEFECT, hit 4x by 4 models** |
| _copilot_render_target -> render_image/render_video | YES — probe_render's own sibling | **LATENT** (both tools 100% cold) |
| script tools (_resolve_document_or_current) | no — one script per document (069 D3) | correct |
| set_uniform | no — resolves against render_pass by design | correct |
| delete/switch/rename/duplicate/canvas/media/pass ops | no | correct |
DO NOT push the split into _copilot_resolve_document_id (~13 callers): deleting or renaming
"c5c9#cascade" is a category error, and silently downgrading it is exactly what
conventions.md:686-689 forbids ("NEVER change a destructive edit's behavior on a heuristic GUESS
the model can't see"). The shared root is the SOURCE-READ SEAM, not the id resolver.

## The other verified defects
_validation_message (registry.py:44-46) COMPUTES the offending field name and DISCARDS it:
  first = exc.errors()[0]   # {'type':'extra_forbidden', 'loc':('document',), 'msg':...}
  return f"error: invalid arguments - {first.get('msg','invalid')}"   # loc thrown away
deepseek then failed write_shader 3x guessing the problem was CONTENT (it changed a comment, then
changed nothing) before dropping the extra field. Two lines to fix, covers all 35 tools.

set_uniform's message is one of the BEST in the codebase — names the uniform, states the
constraint, gives the alternative. DO NOT REWRITE IT. The defect is upstream and downstream:
  UPSTREAM: _SET_UNIFORM_DESC (shader.py:160-169) and _CONVENTIONS (prompt_context.py:22-23) name
    THREE of the five engine uniforms (u_time, u_aspect, u_resolution). u_pass_iteration and
    u_pass_iterations appear in NO static prose surface — and are exactly what codex-mini tried.
    The full set is one table: engine_uniforms.py:14-20.
  DOWNSTREAM: codex-mini repeated the identical rejected call 3x and the turn died at
    cutoff='time_budget' — 600 SECONDS on three rejected calls. max_edit_retries=3 is gated on
    is_edit=True, set only on the two shader edit tools, so set_uniform is outside it; the
    tool-agnostic repeat brake nudges at 3 and stops at 6, so three repeats reached only the nudge.

## Cold eager tools, exact
35 registry tools, 15 called, 20 never. Of the never-called, SEVEN ARE EAGER and billed on every
request: render_video 389, publish_youtube 331, render_image 294, publish_telegram 184,
read_script 180, switch_document 170 => ~1,548 tok x 467 requests = ~723,000 est tokens, 7% of
billed input. load_tools re-requests are FREE at the spec level (agent.py filters to `newly`), so
the waste is the EAGER tier, not the lazy one. Hand to cell 3's P2. NOTE: switch_document is
prompt-critical (prompt.py:199-200 makes it mandatory before publish) and should likely stay
eager regardless of its cold record.

## Proposals
P1 (DO FIRST) — read_shader accepts <id>#<pass>, and an example returns its non-output passes.
  Split the handle as _copilot_resolve_target does; read document.passes[name] not render_pass;
  add the ENUMERATING reject copied in shape from backend.py:1670-1674 ("no pass 'X' -- the
  passes are [...]") in the BACKEND where document.passes is in scope. Document `<id>#<pass>` AND
  `example:` on the tool (the latter is undocumented today despite being the only handle it ever
  read successfully).
  CLASS: the read half of the address contract + the example-reference hole behind F5.
  FAILS IF: six views for one example blow the working-set cap. Mitigate by mirroring the existing
  rule — read_working_set already collapses a pass address to its document (backend.py:774-777,
  "one member is one DOCUMENT (D11)"); only the EXAMPLE path needs per-pass listings.
  THE BREAK: test read_shaders(["<short>#<pass>"]) returns that pass; test ["<short>#ghost"]
  returns the message naming the real passes. THEN DELETE the split_pass_address call and confirm
  the first test fails with today's "no such document(s)"; restore. SECOND BREAK: point the
  example read back at render_pass and confirm the example test loses five of six passes.
P2 — ONE registry-enumerated gate for the pass-address contract, replacing the FALSE comment at
  test_copilot_pass_tools.py:115-116. For every ToolDefinition with an address-carrying field,
  assert membership in exactly one of PASS_AWARE / DOCUMENT_ONLY, derived from
  registry.definitions() so a NEW tool is a TEST FAILURE until classified.
  CLASS: the whole domain-narrowing family for addressing; also forces the render-pair decision.
  THE BREAK: revert probe_render's split (backend.py:1666) and confirm the test names
  probe_render; restore. Then add a throwaway tool with a `document` field and no classification
  and confirm the test fails naming it.
P3 — _validation_message names the offending field (use the `loc` already on the dict). Two lines,
  all 35 tools. THE BREAK: assert the message for an extra `document` on _WriteShaderArgs contains
  "document"; then revert to msg-only and confirm it fails.
  FLAG, do not fold in: edit_shader/write_shader are the ONLY two tools using `target` while 16
  others use `document` — a wider rename than F2 should decide alone.
P4 — generate the engine-uniform list in _SET_UNIFORM_DESC and _CONVENTIONS from
  ENGINE_UNIFORM_TYPES rather than retyping three of five; state the positive branch (to change
  what a pass counter does, edit the shader or change `runs` via set_pass).
  PASSES the better-model test: the names are absent from EVERY static surface — a missing
  affordance, not carelessness. THE BREAK: assert every ENGINE_UNIFORM_TYPES key appears in the
  rendered description; add a sixth name to the table and watch it go red.

## False trails
- "read_shader's schema is wrong" — its field description never claimed #pass; fixing only the
  description would leave the example's non-output passes unreadable, the half that actually cost.
- "Push split_pass_address into the shared id resolver" — ~13 callers, several MUST reject loudly.
- "The 4 failures cost 4 calls" — three cost one call each and recovered instantly (the project
  pass was already in the working set). The cost is concentrated in the fourth (8 greps) and,
  invisibly, in the 5 of 6 example passes no model could ever read.
- "set_uniform is a tool-contract defect" — its message is exemplary; the defect is around it.
- "The no-op brake would catch the repeat now" — it nudges at 3, stops at 6; three repeats reach
  only the nudge. The 600s time_budget ended that turn, not the brake.
