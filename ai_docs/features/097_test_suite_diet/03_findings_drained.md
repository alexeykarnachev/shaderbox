# 097 — found while cutting, then drained

Five agents read the suite and the repository during 097 and reported what was a DEFECT
or dead weight rather than test clutter. None was touched at the time: cutting tests and
changing behaviour are different jobs. They were drained in a later wave, each verified at
the source first — which reclassified one finding entirely and corrected three that named
the wrong file.

## Fixed

**A document handle the schema called optional and the resolver refused.**
`rename_document` / `set_canvas_size` / `duplicate_document` declared `document: str =
Field(default="")` and then resolved it through `_copilot_resolve_document_id`, which rejects
an empty handle on purpose. The declared contract was "empty = the current document" in three
places — `capabilities.py`'s own comment above the three methods, and the sibling tools in
`passes.py` / `script.py` / `media.py` — so the tools were the outlier, not the contract. They
now resolve through `_resolve_document_or_current`, which every other document-addressed tool
already used. The four private `_NODE_DESC` copies (three wordings, one of them omitting the
empty case) collapse into `tools/base.py::DOCUMENT_ARG_DESC`.

The test above it was green because its stub accepted `""` where production refused it. The
stub now binds the REAL resolvers off `CopilotBackend`, so it cannot be kinder than production
again. Break tried: point `rename_document` back at the strict resolver —
`test_backend_rename_and_canvas_run` goes red at the rename assert.

**One alignment rule at two sites in two shapes.** `render_preset.py::_align` had a floor and
`document.py::_render_video`'s inline copy did not, so a sub-block dimension could align to 0.
The function is now `align_for_codec` and both callers use it.

**`DTYPES` and `TARGET_DTYPES` as two independent literals**, neither derived from
`TargetDtype`. `TARGET_DTYPES` is now `get_args(TargetDtype)` and `DTYPES` aliases it. Break
tried: add `"f8"` to `TargetDtype` — `test_every_target_format_has_a_human_label` goes red,
where before the literals would simply have drifted apart in silence.

**A `.gitignore` pattern covering part of its domain.** `projects/*/media/` needs exactly one
intermediate segment, so `projects/dev/media/` was ignored and `projects/media/` was not — two
regenerable Telegram sticker caches were committed, which `dev_flow.md` forbids. The pattern is
`projects/**/media/`; both depths verify with `git check-ignore`, and the two files are
untracked (kept on disk).

**Dead files and symbols.** The two unused Anonymous Pro italics (~195 KB shipped in every
itch.io zip), the orphaned `resources/shaders/editor.frag.glsl` (the real editor shader lives
in the compiled `libeditor.so`), `thumbnail.png` and `probe.ini` at the root,
`ui.py::ViewerGeometry.image_height` (constructed, never read), and four unused parameters:
`pass_graph.py::_draw_canvas(wiring, groups)` and `_snap(picture)`,
`scripting/engine.py::reload(document)` (vestigial at ~30 call sites, two of which already
passed `None` against its `ScriptTarget` annotation) and `tabs/code.py::_python_request(app)`.

**Docs asserting what the code does not do.** `CLAUDE.md` said `build.sh` strips `projects/dev/`
from the bundle; it copies an allowlist that never contains `projects/`. `dev_flow.md` said
`scripts/smoke.py` needs that sandbox as a fixture; smoke seeds its own throwaway project and
must never touch it. `conventions.md` documented a project-fork mechanism (`_copy_into_fork`,
`.creating`) that no longer exists in any form — the bullet now states the rule that still
binds a future copy verb and says plainly that `create_project` / `trash_project` are the
project verbs today. Also `_CTX_GLOSS`/`_CTX_HELP` → `_CONTEXT_GLOSS`/`_CONTEXT_HELP`,
`ENGINE_DRIVEN_UNIFORMS` relocated to `engine_uniforms.py`, and the xdist roster (four named,
nine real) replaced by `grep -rl xdist_group tests/` — a census in a doc drifts by
construction. `BUILDING.md` promised Ctrl+N and "Open dir"; the chord is Ctrl+Shift+N and the
button reads "Open folder".

**Stale pointers in this repo's own source.** `theme.py` and `ui_primitives.py` cited the
deleted `tests/test_ui_prose_budget.py` as the authority for a live constraint; the word budget
now cites `conventions.md`, which is what actually holds it. `.github/workflows/ci.yml` cited a
`todo.md` entry removed in `1426060`.

**A docstring narrating development history.** `document.py::canvas_size_for` explained that the
rule "used to be spelled out at four sites in three syntactic shapes" and named "the bug that
started this feature" — which `CLAUDE.md` forbids. It now says what the function is.

## Reclassified — not a defect

**A command's scope, answered differently by the menus and the hotkeys.** Reported as two places
deciding one thing, against the contract in `CommandScope`'s own comment. It is a deliberate 093
decision, recorded twice (`093_refinement/05_menus_spec.md` M2 and `04_menus_inventory.md`),
which rejects BOTH the focus test and `spec_eligible` for menus by name: the click that opens a
menu has already cleared `editor_focused`, so the strict gate would grey out every scoped item
exactly when it is clicked. A scope names a SURFACE; each surface reads it with the test its own
input model allows. What was actually wrong is that the enum's comment stated only the chord
half as if it were the whole contract — it now states both and says why they differ.

## Left alone, with the reason

**`telegram.py` and `youtube.py` carry byte-identical `status()` and `update()`.** Hoisting them
to `exporters/base.py` would require the ABC — today a pure interface — to assume a `_worker`
and a `_render_state` it does not declare, constraining every future exporter to that internal
shape. The two classes' fields are not even the same types: each has its own private
`_RenderState` and a differently-parameterized `ExporterWorker` (Telegram's event union carries
`_LinkEvent` / `_StickerListEvent`, YouTube's `_ConnectEvent`). There is no shared supertype to
hoist onto without inventing one, and eight lines of duplication is the cheaper side of that
trade.

## Rules whose gate was deleted in 097

Both were cut deliberately, and both need a human to hold them — stated here so the next reader
knows they are conventions rather than enforced:

- The UI word budget (`conventions.md`, and the `/imgui-ui` skill's table).
- American spelling across every surface a reader sees.
- `roadmap.md`'s own "one row, one sentence" rule, which `test_roadmap_shape` held. Measured
  after the cut: the median row is ~470 characters and the longest are 043 (1434), 096 (1262)
  and 089 (1182) — so the drift predates 097 and the gate was not holding the rule anyway. 097's
  own row was the worst offender and was shortened; the older ones were left alone, because
  rewriting a closed feature's row is a judgement call for the maintainer, not a cleanup.
