# 072 — A sampler has ONE source

Status: **done** (code landed, gates green, two post-implementation reviews at PASS after fixes; the maintainer's hands-on pass per `## Manual verification` is outstanding).

## Goal

A `sampler2D` uniform today has two stores with a precedence rule between them: an edge in
`graph.json` (`inputs`, set in the gear) and a texture in `document.json` (set on the uniform
panel), resolved per frame by the binder as "edge shadows media". The maintainer's words, after
breaking the wiring by renaming a uniform: *"the wiring is just the source of the texture. And we
already have the user path to set up a texture source, on the uniform's panel. The same logic is
split on two different (and probably even conflicting in some corner cases) paths."*

The corner cases are real: loading a file on a wired sampler needs a detour through the gear;
wiring a pass over a loaded texture keeps the texture on disk, invisible until the wire goes;
a renamed sampler leaves an edge the binder ignores but the planner still orders by.

After this feature a sampler's value IS its source, one slot, one row, one precedence: whatever
is written. The gear keeps what is not a per-sampler fact (name, runs, target).

## Out of scope

- Editing a source from the strip (a chip is read-only). Trigger: the maintainer asks for it.
- A second naming shape beside `u_<pass>` / `u_prev`. Trigger: 069 D9's own revisit clause.
- The copilot gaining a `set_sampler_source` tool. Today it wires by naming (prompt) and binds
  media through the user's picker; a pass source is one `edit_shader` away by renaming the
  sampler. Trigger: a dogfood run where the model needs a wire it cannot express by name.

## Design decisions

- **D1. One source per sampler, held in the pass's `uniform_values`.** A sampler's value is one
  of: `PassSource(name)` (reads that pass's live canvas; `name` equal to the consumer is
  feedback), `NoSource()` (reads black, by decision), `AutoSource()` (undecided: the name rule
  decides at bind time), or a texture the user chose (`MediaWithTexture` / `moderngl.Texture`,
  exactly as today). The three GL-free source kinds live in `pass_graph.py`. `PassEntry.inputs`
  is deleted; `graph.json` keeps `output`, per-pass `target` and `iterations`, and its version
  bumps to 2.
- **D2. `AutoSource` replaces the default photo as the "unbound" marker.** `seed_uniform_values`
  seeds a sampler with `AutoSource()`; `is_default_image` and the `DEFAULT_IMAGE_FILE_PATH`
  marker go from every reader (save skip, copilot rows, the panel, the canvas presets). Nothing
  in the app renders a `Pass` outside a `Document`, and the document binder already seeds black
  for every sampler the user did not bind, so the photo was never drawn; `Pass.render` now binds
  a 1x1 black of its own (`Pass._black_texture`, released with the pass) for any value that
  is not a texture, and the document hands it textures only for
  the samplers the wiring fills. The photo resource stays for the starter example's own media.
- **D3. The name rule (069 D9) is unchanged and applies to `AutoSource` only.** One pure function
  in `pass_graph.py`, `wired_pass(source, uniform, consumer, passes) -> str | None`, answers
  "which pass does this sampler read": a `PassSource` naming an existing pass, or the name rule
  for `AutoSource`; `None` for `NoSource`, a texture, or a pass that does not exist.
  `Document.effective_wiring()` maps EVERY pass to `{uniform: pass}` through that one function.
  For a compiled pass the domain is its declared samplers, each read from `uniform_values`
  (absent = `AutoSource`), so a row for a sampler the program no longer declares is not a read.
  For a pass that has not compiled, the domain is the explicit `PassSource` rows it loaded: an
  explicit wire is never lost to lazy compilation (066 D1), and only the NAME rule waits for the
  program. Every whole-graph reader (binder, planner, strip, copilot) goes through
  `effective_wiring`; the panel row asks per uniform through `Document.sampler_source`, which
  reads the same function. `assert_plan_invariants` audits the plan against the wiring it was
  built from, as it audits against the graph today.
- **D4. The planner takes wiring, not the graph.** `plan_passes(wiring)`, `plan_for_output`,
  `evaluation_order` and `assert_plan_invariants` take `Mapping[str, Mapping[str, str]]` (every
  pass a key). `PassGraph` no longer carries edges, so it cannot be the planner's input; the
  binder reads `target` and `iterations` from it as before.
- **D5. Persistence: the source is a uniform row in `document.json`.** `{"pass": "paint"}`,
  `{"none": true}`, the existing file dicts for media and raw textures; an `AutoSource` writes
  no row (the absent key is the undecided state, as an absent edge was). A row that is not a
  file deletes any stale `media/<pass>/<uniform>.*` and `textures/<pass>/<uniform>.*` at save,
  as the default-skip did. The loader maps the two new dict shapes to their sources; an
  unknown dict shape costs that uniform, as today.
- **D6. Verbs.** `wire_pass_input` / `unwire_pass_input` are replaced by
  `ProjectSession.set_sampler_source(document_id, pass, uniform, source)` taking a
  `PassSource | NoSource | AutoSource`; a `PassSource` must name an existing pass. Media keeps
  its writers (the panel's picker, the copilot's `bind_media`) since it is the same slot;
  `unbind_media` writes `AutoSource()`. The delete and rename fix-ups move from the pure graph
  helpers onto the document, which owns the values: `Document.forget_pass_sources(name)` turns
  every `PassSource(name)` into `AutoSource()`, `Document.rename_pass_sources(old, new)` rewrites
  every `PassSource(old)`; `_graph_without` / `_graph_renamed` keep only the entry and the
  output. Both verbs stay transactional with the file, the output and the editor tab.
- **D7. The panel row is the one place a source is chosen.** *(The row SHAPE is superseded by
  079 D6: the list is `none` / the passes / `file...`, with no `auto (…)` row — `AutoSource`
  stays the value a fresh sampler holds, and the closed control shows the pass it resolves to.
  Everything else below stands.)* A sampler row shows a source combo,
  `auto (paint)` / `none` / one item per pass / `file...`, then what it reads: the pass's live
  thumbnail and name, the media thumbnail with its resolution (video filters beside it, as
  today), or the black swatch. `file...` opens the picker the Load button opened; the button
  goes, and so does the `MediaWithTexture` assert at the top of the texture branch, which every
  sampler row reaches. The gear's `Reads` section is deleted; the gear keeps name, runs and the target.
- **D8. The copilot's working set says the source on the sampler's own row.** `u_x sampler2D <-
  paint`, `<- (WxH, image)`, `<- (none; reads BLACK)`, `<- (nothing; reads BLACK)`. The separate
  `inputs:` list on `PassView` goes; a multi-pass document's rows carry it. The prompt's wiring
  sentence stays true (a `u_<pass>` name still wires) and its `inputs:` mention is rewritten.
- **D9. `PassGraph.layout` is deleted.** 070 rejected the spatial view it was reserved for.
- **D10. The tutorial builder reads the sources from `document.json` rows plus the name rule**;
  its test drives the engine's `wired_pass` over the example and compares, as it does today.
- **D11. Examples, the tracked projects and the bloom fixture are hand-edited, no migration.**
  Every edge in the Radiance Cascades example and the bloom fixture is what the name rule
  yields, so their `graph.json` files lose `inputs` and `layout` and gain no rows. The other
  examples and the two project dirs (`projects/dev`, and the older tracked `projects/documents`)
  lose `layout`. The bloom document under `projects/documents/1901ab60-...` wires `u_src`,
  `u_lit`, `u_glow` and `u_trail`, which no name derives, so its `document.json` gains the six
  explicit `{"pass": ...}` rows; its `u_prev` is name-derived and gains none.

## Files touched

`shaderbox/pass_graph.py` (sources, `wired_pass`, planner over wiring, `PassEntry` without
`inputs`, no `layout`), `shaderbox/core.py` (seed, bind, the pass's black), `shaderbox/media.py`
(`is_default_image` deleted), `shaderbox/document.py` (`effective_wiring`, `sampler_source`,
binder, loader, the two source fix-ups), `shaderbox/ui_models.py` (save rows + sweep),
`shaderbox/project_session.py` (verbs, delete/rename), `shaderbox/widgets/uniform.py` (the
source row), `shaderbox/popups/pass_settings.py` (Reads deleted), `shaderbox/widgets/pass_list.py`
(chips from the wiring), `shaderbox/tabs/document.py` (canvas presets),
`shaderbox/copilot/backend.py` + `capabilities.py` (`PassView.inputs` deleted) + `prompt.py` (rows), `ai_docs/features/068_radiance_cascades/build_tutorial.py` + the
rebuilt `tutorial.html`, every `graph.json` under `shaderbox/resources/document_examples`,
`projects/dev`, `projects/documents` and `tests/fixtures`, the bloom project's `document.json`;
tests: `test_default_wiring.py`, `test_pass_verbs.py`, `test_uniform_panel.py`,
`test_copilot_passes.py`, `test_pass_graph.py`, `test_graph_persistence.py`,
`test_tutorial_build.py`, `test_media_bind.py`, `test_uniform_seed_save.py`,
`test_document_graph.py`, `test_pass_hot_reload.py`, `test_gl_lifetime_guards.py`,
`test_script_engine_gl.py`, `test_export_script_wiring.py`, `test_examples_resolve.py`,
`test_copilot_script_tools.py`, `test_ui_prose_budget.py`; docs: `conventions.md` (the 065 D3
and 069 D9 bullets, the deliberate-unused list), `dev_flow.md ### Module map`, the roadmap.

## Manual verification

1. Open Radiance Cascades: the strip's chips and the picture are unchanged; the gear shows no
   Reads section.
2. On `seed`, the `u_paint` row reads `auto (paint)` with paint's thumbnail. Pick `file...`,
   load an image: the row shows the image, the chip goes, the picture changes. Pick `auto
   (paint)`: the image is gone from the row and from `media/seed/` on disk after the save.
3. Pick `none` on `composite`'s `u_paint`: black swatch, the chip goes, the wall disappears from
   the render. Reload the document: still none.
4. Rename `u_paint` to `u_paint0` in `seed`: the row reads `auto (none)` and the black swatch,
   the chip goes; rename it back: the wire returns with no click.
5. Rename pass `paint` to `scene` through the gear: every `auto (paint)` row now reads
   `auto (none)` (the name rule no longer matches) and a row that had `paint` chosen
   explicitly reads `scene`. Delete a pass a row names explicitly: the row returns to auto.
6. Wire safety, by reader: `Document.render` binds through `effective_wiring` (the binder loop);
   `pass_list._reads` reads the same wiring; `tests/test_default_wiring.py` renders a copy of
   the example and asserts pixels through the binder, and goes red if `wired_pass` returns None
   for an `AutoSource`.

## Open questions for the user

None blocking. One default taken, flip it by saying so: `{"none": true}` is the on-disk
spelling of an explicit black; `{"pass": ""}` was the alternative and reads as a typo.

## Review history

Pre-implementation review (one reviewer, anchored to the maintainer's verbatim request and the
code): PARTIAL, four blockers, all folded in above. (1) D3 as first written scoped the wiring to
compiled passes and would have dropped an explicit row until the pass compiled, one frame of
wrong order per load; D3 now reads explicit rows regardless of compile. (2) D6 named the
delete/rename behavior but no seam; the fix-ups now live on `Document`. (3) The tracked bloom
project under `projects/documents` has edges no name derives; D11 now gives it rows. (4) The
panel's `MediaWithTexture` assert was unnamed; D7 now names it. Two non-blocking findings:
`Pass` needs its own black texture (D2 names it), and the planner-audit concern was a false
trail, since the graph was already both the planner's input and the audit's reference. Found
fine and not re-litigated: the stale `inputs` key on a version-2 file is dropped by the existing
per-key salvage; `try_to_release` ignores a source dataclass; the script engine never touches a
sampler slot; checkpoint/revert and duplicate ride `UIDocument.save` unchanged.

Post-implementation review (two reviewers, both anchored to the maintainer's verbatim request
and the code, both probing the real engine on a headless context): spec-fidelity PASS, every
decision D1-D11 realized with the on-disk shapes, the bloom project's render and the maintainer's
own `u_paint0` break demonstrated; code review PARTIAL with six findings, all fixed in this wave:
the copilot's `unbind_media` result and description still taught the deleted default image and
a row spelling the backend no longer emits; `_wired_for` carried a compile trigger no test could
pin, because both of its callers already compile the pass first, so it is now a pure lookup
with the precondition stated and `test_the_member_row_resolves_after_the_compile_that_finds_the_samplers`
pins the caller's compile; four comments still placed the wiring in the gear or
named the shipped default; a raw `moderngl.Texture` in a sampler slot drew the black swatch
instead of its thumbnail; the planner tests carried an identity helper. Also closed while in the
file: a pre-existing `persist=True` kwarg that `TargetConfig` silently ignored. False trails the
reviewers recorded so nobody re-walks them: a broken output pass's healthy ancestor stays undrawn on
the pre-072 code path too; `GRAPH_JSON_VERSION` has no reader and needs none.
