# CELL 5 — F5/F6 library + code. Moderator: opus. KEY CLAIMS VERIFIED by main session.

## THE FINDING SOFTENS, AND A SHARPER ONE REPLACES IT
The library is NOT hidden: the lib catalogue is in the prompt at ~9.0% of input spend
(prompt_context.py:41 lib_catalog "name/signature/doc"). Models see it and still hand-roll after
the first pass. So F5 is a SALIENCE-AND-COST finding, not a discoverability one —
and, honestly stated, IT PREVENTED NO OBSERVED FAILURE. Both F6 failures (luna's merge guard,
deepseek's shell radii) broke math the library does not ship. A negative result, stated plainly.

## THE REAL FINDING: a model/human information asymmetry — VERIFIED AT SOURCE
widgets/pass_list.py:154-163 computes reachability for the HUMAN:
    output = document.graph.output
    wiring = document.effective_wiring()
    live = (set(evaluation_order(wiring, output)) or {output}
            if output in document.passes else set(document.passes))
and passes `name not in live` as `stale` into each tile (:174), dimming it with a corner tick.
The comment at :150 states the intent: "Passes the current output does not need never render."

capabilities.py:212-220 PassView carries: name, address, listing, uniforms, errors, is_output.
NO reachability field. errors is list[CompileErrorInfo], never a GraphError.
grep GraphError|graph_errors over shaderbox/copilot/ => **0 hits**. The copilot pipeline drops
graph diagnostics of EVERY kind.

=> THE USER SEES A DIMMED TILE; THE MODEL SEES NOTHING. That is why 4 dead main.frag.glsl stubs
survived, one still WIRED as a live graph node, and why no sweep turn removed them: the model was
never told they were unreachable.

## The dead stubs cost bookkeeping, NOT render time — traced twice, independently
document.py:571 -> plan_for_output -> _order_for (pass_graph.py:409-422) does a BFS from the
target and keeps only reachable passes. The export path (render_media -> _render_video/
_render_image) has no separate unfiltered draw. So the stubs cost a stub .glsl, a graph.json
entry, and working-set tokens — not GPU time. (Corrects any implication that they slowed runs.)

## The delete-tool asymmetry explains the corpus exactly — VERIFIED
tools/passes.py:143  delete_pass      eager=False   (needs load_tools first)
tools/shader.py:579  delete_document  eager=True    (always in the turn-start set)
gemini called delete_document (eager) and it succeeded; cleaning an orphaned PASS needed the lazy
tool nobody loaded. Same tax cell 3 measured, surfacing as a BEHAVIOURAL gap.

## The library ships no tonemap — VERIFIED (grep over shader_lib finds only a text/layout.glsl
false positive). Six finisher composites hand-rolled ACES, copied from prompt.py:192. Cross-checks
cell 3 exactly: models copy what IS in the stream (the prompt's ACES literal) and hand-roll what
is NOT (a helper that does not exist). Pure corollary-2 behaviour.

## Proposals (moderator's ranking)
P1 (DO FIRST) — give PassView a reachability bool from the value pass_list.py ALREADY computes,
  and render one word in _render_working_set_member (prompt.py:324-360).
  CLASS: closes the model/human asymmetry; the model can then see a pass is unreachable and sweep
  it. NOT new logic — one field plus one word, from an existing computation.
  THE BREAK: build a document with an unreachable pass, assert the working set names it; then cut
  the wire feeding the flag and watch the assertion fail.
P2 — add SB_tonemap_aces to the library (6/6 finishers hand-rolled it; the prompt literal is the
  only source today). Evidence about what BELONGS in the library.
P3 — stop create_document leaving a starter stub that becomes dead the moment real passes exist.
P4 — DO NOT add prompt text telling models to use the library. Fails the better-model test and
  adds permanent prompt tax; the catalogue is already there and already paid for.

## Nuance flagged, NOT asserted
pass_graph.py:136 has a _reject_unnamed_pass method with no matching GraphError kind — a
method-name/error-surface mismatch. Outside F5/F6; left for whoever owns graph diagnostics
rather than folded in unverified.

## VERDICT
The library is fully visible at 9.0% of input spend and still unused after the first pass, so F5
is salience-and-cost and prevented no observed failure; the dead stubs cost bookkeeping rather
than render time; and the first fix is handing the copilot the unreachable-pass fact the UI has
been drawing for the user all along.
