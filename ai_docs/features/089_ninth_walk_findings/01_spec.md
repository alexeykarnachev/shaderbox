# 089 — Ninth walk findings

The maintainer's ninth batch, five findings (`00_findings.md`, verbatim). Four land here; the
fifth is the editor library's, built in the editor repo and re-vendored, with the lexer half of
W-B riding the same re-vendor. He asked for the batch to run autonomously: research, split,
review, implement through separate agents, review again, then a cold start. So the plan-lock
below records his message as the lock and marks every choice that is mine, each reversible by a
one-line change once he has seen it.

Source: `../TODO`, 2026-09-09.

Size: **high-blast-radius** by count rather than by any one piece -- five workstreams over two
repos, one new on-disk artifact, one generated table re-sourced, one new symbol kind. Two
pre-implementation reviewers, three post, and the sanitization sweep.

---

## Goal

- **W-A -- the shader formatter breaks after the open bracket and keeps `.rgb` on the close.**
  Finding 1: `function(` / `    a, b, c, d` / `).xyz`.
- **W-B -- completion offers every builtin a `#version 460` fragment shader can name.** Finding
  2: `gl_FragCoord` and the rest of the fragment-stage variables, and the gl4 functions the
  es3.0 table never had.
- **W-C -- a feedback pass continues from where it was when the app restarts.** Finding 3: the
  newest frame of every feedback pass is saved with the document and seeds its history on load.
- **W-D -- the FPS panel colors its numbers by budget share and sorts the tree by cost; the
  gpu-over-frame question is answered where a reader will find it.** Finding 4.
- **W-E -- vim's jumplist, `Ctrl+O` back and `Ctrl+I` forward, within one buffer.** Finding 5.
  **EDITOR REPO**, then re-vendored.

---

## Out of scope

Each with the trigger that reopens it.

- **A cross-file jumplist.** The maintainer offered the single-file scope himself. Trigger: he
  asks for `Ctrl+O` to cross a tab after using the single-buffer one.
- **Insert-mode `Ctrl+O`** (vim's one-normal-command). Trigger: he reaches for it and reports
  the miss.
- **An export that continues from the persisted history.** Exports start cold by design
  (`Document.export_isolation`, `test_an_export_does_not_inherit_the_live_history`); the
  persisted texture seeds the LIVE document only. Trigger: he asks for a render that starts
  from the canvas as it is on screen.
- **Persisting a non-feedback pass's canvas.** A pass nothing reads back is recomputed on the
  first frame; only a pass with a history has state worth a file. Trigger: a pass whose input is
  a texture bound at runtime by something outside the document (none exists).
- **Builtin constants (`gl_MaxDrawBuffers` and kin) and vertex-stage variables.** Not typed in
  a fragment shader in practice; the editor never opens a vertex shader. Trigger: he types one
  and the popup has nothing.
- **A regeneration-drift gate for `glsl_docs.py`.** The generator needs a network clone of the
  refpages, which the suite cannot assume. The table is pinned by content tests instead (V3) and
  by the lexer-equality test (V13). Trigger: a re-run of the generator produces a diff nobody
  expected.
- **Other formatter taste** (`AlignOperands`, `ColumnLimit`, brace style). D1 changes exactly
  the shape he named. Trigger: the next formatting complaint.
- **Per-row CPU and GPU side by side in the panel.** One number per row stays (088 D5).
  Trigger: he asks what a GPU row's CPU issue cost was.

---

## Design decisions

Numbered, locked. The ones marked *(mine)* were decided without him and are listed again under
Plan-lock with the one-line change that reverses each.

### D1 -- `GLSL_STYLE` takes clang-format 23's bracket-break options; the member access is the host's one post-pass. (W-A)

The wheel is clang-format 23.1.0 (`clang_format.get_executable`), where `AlignAfterOpenBracket`
is a bool and the block-indent shape is spelled `BreakAfterOpenBracketFunction` +
`BreakBeforeCloseBracketFunction`. Measured on his line, and reproduced byte for byte by the
first reviewer:

```
{BasedOnStyle: LLVM, IndentWidth: 4, TabWidth: 4, UseTab: Never,
 AlignAfterOpenBracket: false, BreakAfterOpenBracketFunction: true,
 BreakBeforeCloseBracketFunction: true, BinPackArguments: true, PenaltyBreakAssignment: 1000}
```

gives

```
    vec3 light = collect_light(
        vs_uv, u_n_rays, u_max_n_steps, band_offset, band_size
    )
                     .rgb;
```

The `PenaltyBreakAssignment` is what stops LLVM's first move, the break after `=` (default
penalty 2). The `.rgb` on its own line is not a penalty outcome: `PenaltyBreakBeforeMemberAccess`
from 2 to 100000 leaves it there, and so do `PenaltyBreakBeforeFirstCallParameter`,
`BreakBeforeBinaryOperators` and `PenaltyReturnTypeOnItsOwnLine`. The only options that attach
it abandon the block-indent shape (`ColumnLimit: 200`, `PenaltyExcessCharacter: 0`). So the
member access after a block-closing bracket is a forced break, and `format_glsl` runs one
post-pass on the formatted text, `_attach_member_access`: **a line that ends in `)` followed by a
line that is whitespace, a dot and an identifier joins into `).ident...` on the first line.**
Applied line by line in order, so a chain (`).bar(x)` then `.rgb;`) folds one link per step and
reaches `).bar(x).rgb;`; and it covers the second shape the reviewers found, a call that fits
while the statement does not (`... = texture(u_sampler, vs_uv)` / `.rgb;` -- the call must be short
enough to fit; a longer one splits into a bare `)` like his), which has no bare `)` line.
In GLSL a line beginning with `.` is only ever a member access, so the rule has no false join:
clang-format writes `);`, `) {` and `),` in every other case and none ends in a bare `)`. It is
idempotent under `format_glsl` (clang-format re-splits the joined line, the post-pass re-joins
it: a fixed point after one round, V2). A line that fits stays one line, as before: `float x =
short(a).x;` is untouched by every option above, and the existing
`test_glsl_formats_with_the_nvim_fallback_style` passes unchanged under the new string.

`GLSL_STYLE` stays one string; the post-pass is a private free function beside it. The convention
bullet on formatting (`conventions.md`, "Formatting is a shipped dependency") gains one clause
naming the post-pass and why (a forced break, not a penalty).

### D2 -- the generator reads the gl4 refpages, names entries from their prototypes, and harvests the fragment-stage variable pages. (W-B)

`scripts/gen_glsl_docs.py` reads es3.0. This repo's shaders are `#version 460`, so the usage line
becomes `git sparse-checkout set gl4`. The switch is NOT a superset by itself: the first reviewer
ran the repo's own `parse_refpages` over both trees and found three es3.0 names that gl4 loses
(`packUnorm2x16`, `packSnorm2x16`, `unpackSnorm2x16`) and four gl4 names that never arrive
(`noise1..4`), for one reason -- a gl4 page whose `refname` is a family name (`packUnorm`,
`noise`) yields no prototype matching `\b<refname>\s*\(`, and the per-name overload filter then
drops the page whole, silently. So the generator now derives the entry names FROM the
prototypes: each prototype's own function name is an entry, grouped, with the page's purpose;
the refname is not the key. And the report that today counts only XML failures also names any
page that yields no entry at all, so the table can no longer shrink in silence.

Excluded by a named set in the generator, with the stage as its reason: the four geometry-stage
verbs (`EmitVertex`, `EndPrimitive`, `EmitStreamVertex`, `EndStreamPrimitive`). Image, atomic
and barrier functions stay: the 4.60 spec's §7.1.5 discusses their behavior inside helper
invocations, which is a fragment-stage fact, so they are legal there. The counts are what the
generator prints on the day; none is written here.

Why `gl_FragCoord` was never there: its page is a `fieldsynopsis` (`in vec4 gl_FragCoord;`),
not a `funcprototype`, and `parse_refpages` drops any page that yields no signature. The
generator now reads those pages too, into a second table `VARIABLES: dict[str,
tuple[tuple[str, ...], str]]` = name -> (declarations, purpose), the `BUILTINS` shape, because
three pages carry two declarations (`gl_PrimitiveID`, `gl_Layer`, `gl_ViewportIndex`: an `in`
and an `out` form, in differing orders) and a single string would pick one arbitrarily. The
variable set is filtered to the fragment stage by a named list in the generator,
`_FRAGMENT_VARIABLES`, which is the 4.60 spec §7.1.5 declaration block verbatim: `gl_FragCoord`,
`gl_FrontFacing`, `gl_ClipDistance`, `gl_CullDistance`, `gl_PointCoord`, `gl_PrimitiveID`,
`gl_SampleID`, `gl_SamplePosition`, `gl_SampleMaskIn`, `gl_Layer`, `gl_ViewportIndex`,
`gl_HelperInvocation` (in) and `gl_FragDepth`, `gl_SampleMask` (out). The refpages do not
encode the stage, which is why this list is hand-held with its citation; the declarations and
purpose still come from the page, never typed.

`glsl_docs.py` is regenerated from a real clone by the implementer (the generator's own gate:
non-zero exit on any skipped or empty page), its module docstring names gl4, and the two prose
gates (`tests/test_prose_spelling.py`, `tests/test_text_artifacts_are_clean.py`) are the first
to run on the new text, since the gl4 pages are a larger body of quoted prose than es3.0's.

### D3 -- a builtin variable is `SymbolKind.GLSL_VARIABLE`, drawn in the builtin slot. (W-B)

`OUTPUT_VARIABLE` means "the one name a shader writes" and `GLSL_BUILTIN` carries call
signatures, so neither fits `in vec4 gl_FragCoord`. A new kind, `GLSL_VARIABLE`: `_KIND_RANK`
beside `GLSL_BUILTIN`, `theme.kind_color` and `kind_slot` the builtin's (slot 6, `SYN_BUILTIN`).
Those three are the only per-kind tables (the reviewer's census over `shaderbox/` and `tests/`),
and `test_every_kind_has_a_color` walks the enum over all three. `_language_symbols` emits one
`Symbol` per `VARIABLES` entry with `signature` = the declarations joined the way builtin
overloads are and `doc` = the purpose, so `K` over `gl_FragCoord` shows `in vec4 gl_FragCoord`
and the popup's detail note shows the purpose. The `glsl` provider offers it with no provider
change (`_glsl_words` returns `index.words` unfiltered; nothing in `intel/` or `completion.py`
special-cases a `gl_` prefix beyond `_BUILTIN_OUTPUTS`). `_BUILTIN_OUTPUTS` (`gl_FragColor` /
`gl_FragData`, both removed from 460 core) stays as it is: they are the legacy-output case.

The text feed is unchanged: `_feed_classes` pushes the host's classes and a builtin's color is
the lexer's (the intel bullet in `conventions.md`). Measured through the vendored binary: of
the 88 names the table holds today, 42 do not lex as a builtin (`acosh` ... `unpackUnorm2x16`,
the list is in the editor session's brief), and of the fourteen variables only `gl_FragCoord`
and `gl_FragDepth` do. D10 closes the whole gap upstream, and V13 is the equality check.

### D4 -- the newest frame of a feedback pass is saved as raw bytes under `feedback/<pass>.bin`, described in `document.json`. (W-C)

What is saved: for every pass whose name is in `Document._feedback` at save time -- the passes
that have a history, which is exactly the set with state worth keeping -- its NEWEST frame, read
with `texture.read()` and written as `feedback/<pass>.bin` inside the document dir. The newest
frame is a document-level question, `Document.newest_frame(name) -> Canvas | None`: the pass's
live canvas once it has drawn (`drawn_frame >= 0`; after frame N the live canvas holds N and the
history N-1), and the history itself before it has -- which is the seeded canvas of a document
loaded and saved without a render, the exact path `duplicate_document` takes (save the source,
load it, save the copy). Without that second branch a duplicate would persist black. The first
branch is also right where the live canvas is BLANK but is the only one whose format matches:
after `set_target` or `set_canvas_size` the pass keeps `drawn_frame >= 0`, its live canvas is
reallocated blank and the history holds the old format or size, which the load-side rule would
reject -- so a blank seed at the right shape is what gets written, and the pass starts black
exactly as a live target change or resize costs it today (measured by the first reviewer).

Its description goes in `document.json` under a new top-level key `feedback`: `{<pass>:
{"file_path", "size", "components", "dtype"}}`, the four fields the raw-texture uniform entry
writes (`ui_models.py::_uniform_entry`, the `TEXTURES_DIR_NAME` branch) -- the same SHAPE, not
the same reader (D5). Raw bytes, never `texture_to_rgba8`: a pass in a graph targets `f2` by
default (063 measured `f1` saturating on the first accumulate; a graph-less single pass is `f1`,
and the bytes work at any dtype) and an accumulator's values sit outside `[0, 1]`, so a PNG
round trip would clamp and quantise exactly the state being kept.

Its own directory, not `textures/`: the save sweep ("Drop media/texture files no surviving
uniform refers to") walks `media/` and `textures/` by what the uniform block references, and a
feedback file is referenced by no uniform. `feedback/` gets its own write-and-sweep by one rule:
the files on disk after a save are exactly the passes in `_feedback`. It runs OUTSIDE the
`if live:` guard the two asset sweeps sit under: a document whose source is broken still holds
its seeded history (the pass never drew, so `newest_frame` is the seed) and must carry it, where
the uniform rows are carried forward from disk. So a Reset (`ProjectSession.reset_document`, the
one funnel, which empties `_feedback`) followed by any save drops the file, and a document that
never fed back writes nothing. The document dir stays the self-contained unit (`conventions.md`,
on-disk lifetimes): duplicate, fork, trash and the copilot's checkpoint carry the file with it.

`FEEDBACK_DIR_NAME` lives in `paths.py` -- the basenames are split today (`MEDIA_DIR_NAME` /
`TEXTURES_DIR_NAME` in `constants.py`; `PASSES_DIR_NAME`, `DOCUMENT_JSON_BASENAME`,
`GRAPH_JSON_BASENAME` in `paths.py`), and `paths.py` is the one `tests/test_document_dir_layout.py`
enumerates against, so the new name joins that closed class and its parametrize list in the
same commit.

### D5 -- the loader seeds the history at the size the graph gives the pass; a never-drawn pass keeps it through the first frame; a file that does not match is ignored. (W-C)

`Document.load_from_dir` reads the `feedback` block after the passes and graph are built.
`document.json`'s top level is a raw dict (`_load_document_metadata`; no pydantic model, so no
`extra='forbid'` reaches it and the new key is additive), and the block is read with the same
per-key `isinstance` guards `_uniforms_by_pass` uses: a malformed entry costs that pass its
history, never the document.

For each entry whose pass exists, the loader allocates the history canvas the way
`_feedback_canvas` does -- a `Canvas` (clamp wrap, the pass's filter and dtype; NOT
`gl.texture(...)`, whose wrap defaults to repeat) with `_feedback_generation[name]` =
`target_generation` -- and writes the bytes with `texture.write`. Two facts the reviewers
measured make the SIZE a computed value, never the live canvas's: at load every pass's canvas is
at the document's full `canvas_size`, and a non-output pass takes its `scale` only inside
`render` (`entry.target.target_size(self.canvas_size)`); so the expected size is that same
expression for a non-output pass and `canvas_size` for the output, and a history allocated from
the live canvas would be rejected for every scaled pass (the bloom chain's, the RC's). The match
rule is strict and fail-soft: the entry's `size` must equal the expected size, its `dtype` the
one the pass's canvas was built with (the target's, or `Canvas`'s `f1` for a graph-less pass), and `components` 4 (every `Canvas` is RGBA, so this is a corruption sniff rather than
a target property); else the file is ignored and the pass starts black, with one
`logger.warning` naming the pass. A missing or unreadable file is the same branch.

**The first frame.** `begin_frame` swaps a pass's history only when the pass drew last frame,
which its comment states -- but `Pass.drawn_frame` and `Document._frame` both start at `-1`,
so on a freshly loaded document the guard `drawn_frame != previous_frame` is false and the swap
runs: the seed lands in the LIVE slot, is overwritten by the first draw, and the reloaded
document reads black (both reviewers' probes). The gate becomes what the comment says: a pass
that has never drawn (`drawn_frame < 0`) has no new history and is skipped. Behavior-neutral
for every existing document (a never-drawn pass's history is black either way) and the
property W-C stands on; V5 asserts the seed's identity survives the first `begin_frame`.
`_swap_feedback`, `render` and the binder are untouched.

What a seed does not survive, stated so nobody expects it to: `set_canvas_size` (the Document
tab's fields, the copilot's tool) reallocates through `Canvas.set_size`, and a history that no
longer matches its live canvas is reallocated black in `_feedback_canvas` -- exactly what a live
resize costs today. `reset_feedback` and `Document.release` release a seeded canvas like any
other (V8 pins both).

### D6 -- when it is written: at the save funnel, which is quit, switch, every graph verb and the copilot's checkpoint. (W-C)

`UIDocument.save` is where every persisted fact of a document is written and the only place
(`conventions.md`, the SAVE-funnel bullet), so the feedback file is written there and nowhere
else. The cost is one `texture.read()` per feedback pass per save -- 7.37 MB and about 4 ms for a
960x960 `f2` target, measured -- paid at quit (`ui.run`: `app.save()` runs before
`app.shutdown()`, so the context is alive), on a document switch, on each of the six graph
verbs, and on each copilot mutation checkpoint (`TurnCheckpoint.snapshot_document` is a full
save through `save_into`, and `RevertExecutor` restores by `copytree` and
`load_document_from_dir`, so a revert of a copilot turn now restores the canvas along with the
source -- the checkpoint's contract, serialize the live object and restore by
reload-and-replace, extended to one more live surface; V7 pins that a snapshot carries the
file). The snapshot dirs are removed with the checkpoint, so the churn is bounded by the
checkpoint store's own cap. No throttle, no separate timer. Nothing is written for a document
with no history, so the common case costs a directory check.

One consequence, measured by the second reviewer rather than assumed: `sync_documents_from_disk`
diffs `document.json`'s mtime and re-reads a changed dir through `_load_one_document_from_disk`,
so a save queues a reload on the next frame -- today already. With D4/D5 that reload replaces the
`UIDocument` wholesale and seeds from the file just written, so the live history after a save IS
the saved one: consistent, and a read after every write. The implementer confirms the reload
does not fight the live `_feedback` and records what it saw under Review history.

### D7 -- a number's color is its share of the frame budget; three bands, one function. (W-D) *(mine)*

Every measured number in the panel -- the `frame` and `gpu` headlines, every tree row, `other` --
draws in `theme.load_color(ratio)` where `ratio = ms / budget_ms` and the budget is
`1000 / target_fps`, the number already on the panel (`global_target_fps` is bounded `ge=30`,
so never zero): below 0.5 `STATE_OK`, from 0.5 to below 1.0 `STATE_WARN`, at 1.0 and above
`STATE_ERROR`. The three tokens exist and are semantic (`conventions.md`, "Color roles are
SWAPPABLE accent vs FIXED semantic"); the collision assert only guards `SELECT`, so no new
token and nothing collides. The three non-measurements (`budget`, `fps`, `target`) stay
`FG_MUTED`, so color means "measured against the budget" and nothing else. A pass at 40% of the
budget reads green, which is the information: a green tree with a red headline says the cost is
in `other`.

The thresholds are two literals in `theme.py` beside the function, named, and V9 pins the bands
at their edges.

### D8 -- the panel draws from a plan; the plan orders children by cost and carries the color; the instrument's order is untouched. (W-D) *(mine)*

`_profile_rows` iterates `span.children` in recording order and `_profile_number` paints every
number `FG_MUTED`. Both go behind one pure function in `ui_primitives.py`,
`profile_rows_plan(profile, fps, target_fps) -> list[ProfileRow]`, a frozen dataclass row
`(depth, name, count, number, color)` in draw order: `frame`, `gpu`, `budget`, `fps`, `target`,
the tree, `other` -- with the measured rows colored by D7 and the three static rows `FG_MUTED`;
with `profile` None only the three static rows. The tree's children come from
`profiling.by_cost(children)`, a free function returning a NEW list sorted by the headline
number descending (GPU ms where the span has one, CPU ms otherwise -- the rule `_span_number`
prints), stable, so equal costs keep recording order and `span.children` itself is never
reordered (V10 pins both). `fps_overlay` then draws the rows and does layout only: the plan
carries the RAW span name and touches no imgui, and clipping to the room before the number stays
in the draw loop, which is what keeps the plan callable headless.

The plan is computed on the SMOOTHED tree the panel already receives (088 D5a): the smoother
keys its paths from `self._last.root` in recording order, so a sort in the plan cannot re-key an
average, and the `FrameProfile` the profiler produced is not reordered (088's "the instrument
stays raw"). `other` stays last, outside the sort, since it is a remainder rather than a span.

The seam exists for the gate: the existing overlay wire test observes only the `profile` handed
to `fps_overlay`, so a sort or a color inside `_profile_rows` would be unfalsifiable there. With
the plan, V10/V9 test the order and the bands on the plan directly and V11 spies the plan call
from the live overlay -- cut the call and V11 is red; reorder inside the plan and V10 is red.

`by_cost` sits in `profiling.py` beside `ProfileSmoother` (GL-free, imgui-free; `profiling.py`
imports `moderngl` alone and must not import `theme`), the plan and the row type in
`ui_primitives.py`, which already owns the panel and reads `theme`.

### D9 -- the gpu-over-frame answer goes to the Help panel, one entry; the panel's strings stay under budget. (W-D)

His question ("Can gpu time be larger than the frame time?") has a yes with a mechanism, and the
mechanism belongs where a reader chose to read (the prose-budget rule): a `Passes`-adjacent Help
entry, "The FPS panel", stating the three facts in plain words -- `frame` is the CPU wall of one
frame, swap included and the target-fps wait excluded; `gpu` is what the GPU spent on that
frame's draws, which the CPU never waits for, so on a GPU-bound shader it is the larger number
and the true bound; and the GPU numbers are read two frames late. No panel string changes;
`help_content.py` has zero sites in `tests/test_ui_prose_budget.py`'s scored domain (probed:
the domain is `ui_primitives` signatures plus four `imgui.*` calls), so the entry is
unconstrained by that gate.

### D10 -- W-E and the lexer sync are the editor repo's; the host re-vendors once, lands its born-red tests IN the re-vendor commit, and changes no chord routing. (W-E, W-B)

The jumplist is the library's (it owns cursor, mode and undo), specified to the editor session
as fixed premises: one buffer, nvim as the oracle, a push at the jump motions (`G`, `gg`, `/`,
`?`, `n`, `N`, `*`, `#`, `%`, `(`, `)`, `{`, `}`, `[[`, `]]`, `:N`) and never at the character
motions, `Ctrl+O` back and both `Ctrl+I` and normal-mode `Tab` forward, a walk back from the
newest entry recording the current position first, nvim's line dedupe, a cap of 100, and no
ABI change (an additive one reported by `nm -D` if unavoidable). Measured before asking: the
vendored sha `5601d13` IS upstream `master`, so the delta is exactly what lands for this batch;
the library dispatches character motions through `resolve_motion` and searches through
`search_jump`, so the push has at least two sites, which the editor session knows; `H`/`M`/`L`
and the mark family do not exist there and are moot. The lexer brief is the union of D3's
measured gap: the fourteen variables, the gl4-only functions, and the 42 documented names that
do not lex as builtins today.

Host-side routing needs nothing: `_drain_editor_input` hands every chord to `ed_key` first, a
consumed chord lands in `editor_consumed_chords`, and `spec_eligible` refuses `OPEN_PROJECTS` for
that frame -- the Ctrl+R precedent, confirmed by a live probe (chord 4659 consumed; Ctrl+O 4656
not consumed at `5601d13`, so `OPEN_PROJECTS` fires today). Once the library consumes `Ctrl+O`
in NORMAL mode, the Projects modal no longer opens from a focused editor, and still opens from
an unfocused one, from insert mode (the library does not consume it there), and from the menu
(`ui.py`, `Projects...`). That is the intended trade and the one he asked for.
`_RESERVED_CHORDS` is untouched: it is the fallback for chords the keymap lacks, and `o` stays
absent from it.

**Landing order, which 087's re-vendor (`0e3dbaf`) is the template for.** W-A, the host half of
W-B, W-C and W-D land as normal commits with `make gates` green. The three born-red tests --
V12 (the jumplist through the binding), V13 (every documented name in the builtin slot; red at
`5601d13` for 42 + 13 + the new functions) and V14b (a real `Ctrl+O` through the drain lands in
`editor_consumed_chords`) -- are written against the old binary, their red output recorded, and
COMMITTED TOGETHER WITH the seven re-vendored files, whose commit message names each test's red
line the way 087's does. They never sit on `dev` red. The re-vendor follows the seven-file
procedure (`conventions.md ## Known quirks`) at the sha the editor session reports, ABI delta
re-derived here from `nm -D` and `abi_probe.py`.

---

## Files touched

**W-A (host):**
- `shaderbox/formatting.py` -- `GLSL_STYLE`, `_attach_member_access`, `format_glsl` runs it.
- `tests/test_formatting.py` -- his line, the fitting-call shape, the chain, the fitting line,
  idempotence.
- `ai_docs/conventions.md` -- the formatting bullet's one clause.

**W-B (host half):**
- `scripts/gen_glsl_docs.py` -- gl4, entries named from prototypes, `_GEOMETRY_ONLY`,
  `_FRAGMENT_VARIABLES`, the `fieldsynopsis` reader, `VARIABLES` emission, the empty-page
  report.
- `shaderbox/glsl_docs.py` -- regenerated.
- `shaderbox/intel/symbols.py` -- `GLSL_VARIABLE` + rank.
- `shaderbox/intel/index.py` -- `_language_symbols` emits variables.
- `shaderbox/theme.py` -- color and slot for the kind.
- `tests/test_completion.py`, `tests/test_intel_index.py` -- V3.

**W-C (host):**
- `shaderbox/paths.py` -- `FEEDBACK_DIR_NAME`.
- `shaderbox/document.py` -- `newest_frame`, the seeding in `load_from_dir` with the match rule,
  the `begin_frame` gate.
- `shaderbox/ui_models.py` -- `UIDocument.save` writes the block and files, sweeps `feedback/`,
  outside `if live:`.
- `tests/test_feedback_persistence.py` (new) -- V5..V8.
- `tests/test_document_dir_layout.py` -- the parametrize list.
- `tests/test_gl_lifetime_guards.py` -- the seeded canvas is released.
- `tests/test_checkpoint.py` (or where snapshots are tested) -- a snapshot carries `feedback/`.
- `ai_docs/dev_flow.md` -- the document-dir data format gains `feedback/<pass>.bin`.

**W-D (host):**
- `shaderbox/theme.py` -- `load_color` + the two thresholds.
- `shaderbox/profiling.py` -- `by_cost`.
- `shaderbox/ui_primitives.py` -- `ProfileRow`, `profile_rows_plan`; `fps_overlay` /
  `_profile_rows` / `_profile_number` draw from it.
- `shaderbox/help_content.py` -- the FPS panel entry.
- `tests/test_profiling.py` -- `by_cost`, the plan, the wire spy; `tests/test_theme.py` (new)
  -- the bands.

**W-E + lexer (editor repo, then the re-vendor commit here):**
- `shaderbox/resources/editor/` -- the seven files + `VERSION`.
- `tests/test_editor_ffi.py` -- V12, V13, V14b (born red, in the re-vendor commit).
- `ai_docs/conventions.md` -- the re-vendor bullet gains this instance in one clause.

---

## Verification

Each step fails for exactly one reason; the falsifier is named; the shape to copy is named.

- **V1 his line (W-A):** `format_glsl` on the ledger's statement returns exactly
  `    vec3 light = collect_light(` / `        vs_uv, u_n_rays, u_max_n_steps, band_offset,
  band_size` / `    ).rgb;`. Falsifier: the old `GLSL_STYLE` (breaks after `=`). Shape:
  `test_glsl_formats_with_the_nvim_fallback_style`.
- **V2 the post-pass (W-A):** three inputs through `_attach_member_access` alone -- the bare
  `)` + `.rgb;` joins; the fitting call `... = texture(u_s, uv)` + `.rgb;` joins; the chain
  `).bar(x)` + `.rgb;` joins -- and `)` followed by `+ 1.0;` is untouched. Idempotence:
  `format_glsl` twice equals once on all of them (shape:
  `test_the_script_stub_is_a_fixed_point_of_the_formatter`). Falsifier: the post-pass removed
  (V1 and the three joins go red together, one cause).
- **V3 the vocabulary (W-B):** `set(VARIABLES) == the fourteen`; `"fma"`, `"noise1"`,
  `"packUnorm2x16"` in `BUILTINS`; `"EmitVertex" not in BUILTINS`;
  `index.lookup("gl_FragCoord").kind is GLSL_VARIABLE` with signature `in vec4 gl_FragCoord`;
  `lookup("gl_Layer").signature` carries both forms; the `glsl` provider offers `gl_Fr` ->
  `gl_FragCoord` and `gl_FrontFacing`. Falsifier: the old table (no `gl_` name reaches
  `_language_symbols`; measured today as `[]`). Shape: `test_intel_index.py`'s kind map and
  `test_the_generated_table_covers_the_builtins_a_shader_uses`.
- **V4 every kind colors (W-B):** `test_every_kind_has_a_color` walks the enum over color, slot
  and rank; a kind added without one KeyErrors it (existing gate, exercised by the addition).
- **V5 the round trip (W-C):** a two-pass document whose feedback pass is at `scale: 0.5` and
  `f2` (the bloom-chain shape; parametrize `f1`/`f2`/`f4` the way
  `test_raw_texture_round_trip.py` does) renders N frames, saves, is loaded into a fresh
  `Document` on the same context; after its first `begin_frame` the history object is the seed
  (identity); after one render its output equals the original's frame N+1 byte for byte
  (`texture.read()` on both), and differs from the cold-start frame. Falsifiers, one per
  assertion: the `begin_frame` gate reverted (the identity assertion fails -- the seed is in the
  live slot); the size computed from the live canvas instead of the graph (the scaled pass
  starts black, the equality fails).
- **V6 starts black stays (W-C):** `test_feedback_starts_black` unchanged and green: a document
  with no `feedback` block seeds nothing.
- **V7 the write and the sweep (W-C):** save after N frames writes `feedback/<pass>.bin` and the
  block; load-then-save with no render writes the SEED's bytes (the duplicate path); a save
  with a broken source still writes it (outside `if live:`); `reset_document` then save deletes
  file and block; a copilot snapshot dir carries `feedback/`. Falsifiers: `newest_frame`'s
  second branch removed (the duplicate case writes black); the sweep removed (the reset case
  keeps the file).
- **V8 the mismatch and the lifetime (W-C):** a `feedback` entry whose `size` disagrees with the
  graph's expected size is ignored -- the pass starts black and the document loads; in
  `tests/test_gl_lifetime_guards.py`, a loaded seed's texture and fbo are released by
  `Document.release` and by `reset_feedback`. Falsifier: the match rule removed (the tampered
  entry seeds where it should not -- the canvas is allocated at the GRAPH's size, so the file
  still fits and `texture.write` does not raise; only a truncated FILE reaches that raise, which
  its own case covers); the release path skipped.
- **V9 the bands (W-D):** `load_color(0.49) is STATE_OK`, `load_color(0.5) is STATE_WARN`,
  `load_color(0.99) is STATE_WARN`, `load_color(1.0) is STATE_ERROR`; and on the plan, a row at
  1.2 x budget carries `STATE_ERROR`. Falsifier: either threshold moved.
- **V10 the order (W-D):** `by_cost` on children at 1, 3, 2 ms yields 3, 2, 1; two GPU spans and
  one CPU span order by their headline numbers; equal costs keep recording order; the input
  list is unchanged and the result is a new list; the plan's tree rows come out in that order
  with `other` last. Falsifier: the sort removed (recording order comes back).
- **V11 the wire (W-D):** in the style of `_capture_the_overlays_profile`, spy
  `ui_primitives.profile_rows_plan` (that module's attribute, since `fps_overlay` resolves the
  name in its own globals at call time; `ui.fps_overlay` stays real so the plan call actually
  runs) while the live overlay draws and assert it was called with
  the smoothed profile and the live `target_fps`. Falsifier: the plan call cut from
  `fps_overlay` (the spy never fires). Two mutations tried and named in the commit: the call
  cut; a `reverse=False` in `by_cost`.
- **V12 the jumplist (W-E, re-vendor commit):** `G` then `Ctrl+O` -> line 0; `Ctrl+I` -> last
  line. Red at `5601d13` (cursor stays on the last line -- measured), green at the re-vendored
  sha. Shape: `test_search_highlights_survive_a_scrolled_view`.
- **V13 the slot (W-B/W-E, re-vendor commit):** every name in `BUILTINS` and `VARIABLES` lays
  out in the builtin slot (`SYNTAX_6`). Red at `5601d13` for 42 documented names plus the new
  ones (measured). Shape: `test_boolean_literals_draw_in_the_keyword_slot`.
- **V14 the routing (W-E):** (a) green today, in W-D's or any host commit: a synthetic
  `editor_consumed_chords = {OPEN_PROJECTS's chord}` makes `spec_eligible(OPEN_PROJECTS)`
  False (the `test_consumed_chord_suppresses_the_registry_spec` shape, chord read from
  `SPEC_BY_ID`); (b) re-vendor commit: a real `Ctrl+O` through `_drain_editor_input` in NORMAL
  mode lands in `editor_consumed_chords` and in INSERT mode does not. Falsifier for (b): the
  `editor_consumed_chords.add` line removed.
- **V15 the gates:** `make gates` green by exit code, unpiped, at every commit; the re-vendor's
  `test_the_binding_mirrors_the_upstream_signature_table` and the enum-coverage test stay
  green; `nm -D` delta equals what the editor session reported.
- **Maintainer's eyes:** the formatted shape on his own file; the green/yellow/red bands and
  whether 0.5 is the right knee; the sorted tree; `Ctrl+O` after a search; the Projects modal
  from the menu now that the chord is vim's in a focused editor; a feedback document surviving a
  restart.

---

## Plan-lock

Locked by the maintainer's message of 2026-09-09 as written: the formatter shape (D1, his
example), the missing builtins (D2, D3), the saved-and-restored canvas (D4..D6), colors and a
sort in the panel (D7, D8), the single-buffer jumplist (D10). He asked for the batch to proceed
without him, so the following are mine and each reverts by one line once he has looked:

- **D7's knee at 0.5 and the red at 1.0** -- two literals in `theme.py`.
- **D8 descending by the headline number** -- the `reverse` flag, or the key, in `by_cost`.
- **D10's normal-mode `Tab` as a second jump-forward** -- one binding row upstream.
- **D9 answering in the Help panel rather than on the panel itself** -- if he wants a hint
  beside `gpu`, that is a two-word caption within budget.
- **D1's post-pass reaching every `)`-ending line, not only a bare `)`** -- the regex's anchor.

---

## Review history

**Round 1 (pre-implementation, two reviewers on opus, read-only with their own probes on a
standalone 460 context, the clang-format wheel and the vendored binary).** Both PARTIAL. Twenty
findings, all accepted; none rejected:

- D5's first-frame claim was false: both probed the swap and found the seed in the live slot
  after `begin_frame(0)` (`-1 == -1`). Fixed by the `begin_frame` gate (never-drawn passes do
  not swap) and an identity assertion in V5.
- D5's match rule rejected every scaled pass (the live canvas is full-size at load; `scale`
  applies in `render`). Fixed by computing the size from the graph entry; V5's fixture is now
  scaled.
- D2's "strict subset" was false by three names, and `noise1..4` never arrived, one cause (a
  family refname). Fixed by naming entries from prototypes and reporting empty pages; the counts
  are dropped.
- `VARIABLES` needed the overloads shape (three pages carry two declarations).
- `FEEDBACK_DIR_NAME`: the basenames are split across two modules; `paths.py` chosen for the
  layout gate, whose parametrize list is now a touched file.
- `model_salvage` does not reach `document.json`'s top level; the guards are
  `_uniforms_by_pass`'s.
- D4's reader was wrong (`gl.texture` defaults to repeat wrap); the reader is a `Canvas` +
  `texture.write`. `components` is always 4.
- V11 had no seam: the wire test sees only the profile handed in. Fixed by `profile_rows_plan`.
- V13 is red for 42 names already in the table, not just the new ones; the editor brief was
  extended and D3 records the measurement.
- V14 was two claims, one of them born red; split into (a) and (b).
- The born-red tests land IN the re-vendor commit (087's `0e3dbaf` is the template), never on
  `dev` red.
- D1's post-pass missed the fitting-call shape and left a chain half-joined; the anchor is now
  any `)`-ending line, applied in order.
- D4's "saved the live canvas" persisted black on the duplicate path (load, save, no render);
  `newest_frame` has a second branch.
- The feedback write and sweep sit outside `if live:`; a seeded canvas gets a lifetime guard; a
  snapshot carrying the file gets a test; `by_cost` returning a new list is pinned.

False trails both recorded, so round 2 does not re-open them: no clang-format option attaches
the member without abandoning the block shape, and the post-pass cannot false-join; the
completion reach needs no provider change and only three per-kind tables exist; the
prose-budget gate scores nothing in `help_content.py` and skips `_`-prefixed helpers, so a color
parameter on `_profile_number` is outside it; `STATE_*` collide with nothing (the assert guards
`SELECT` alone); `test_persistence_completeness.py` needs no rostering (no new JSON-reading
module); `test_generated_artifacts.py` is parametrized over the glyph tables only; the smoother
keys paths in recording order, so a sort in the plan cannot re-key it; the checkpoint and
duplicate paths carry an unknown subdirectory by construction; the budget denominator cannot
be zero.

**Round 3 (implementation, four agents on opus, one worktree each, merged linearly onto `dev`;
`make gates` exit 0 on every branch and on the merged tree).** What the implementers found that
the spec had wrong or had not said, each recorded with its evidence:

- **W-A:** D1's illustrating string for the fitting-call shape split into a bare `)` like his
  line; a shorter call exhibits it, and the text now says so. Third break tried beyond the
  two named: the anchor narrowed to a bare `)` turns the fitting-call and chain tests red,
  pinning the plan-lock choice.
- **W-B:** the generator prints 157 builtins, 14 variables, 33 keywords, 28 types; no es3.0
  name lost, `noise.xml` and `packUnorm.xml` were the empty pages. Three gl4 pages
  legitimately yield nothing (`gl_PointSize`, `gl_Position` declare inside a `gl_PerVertex`
  listing; `removedTypes` is API) and the generator carries them as a named set with what each
  documents, so "report an empty page" and "exit 0" stop contradicting. `_ENTITIES` grew three
  names from the gl4 corpus. The old `mix`-has-three-overloads assertion went red on gl4's
  nine and was rewritten to name two forms rather than a count. Family pages hold names the
  file-name diff missed (`dFdxCoarse`, `packUnorm4x8`, `imulExtended` ...), so the editor's
  lexer brief was re-sent as the table's own 171 names.
- **W-C:** V8's falsifier pointed at the wrong lever (corrected above); a truncated-file case
  was added because that branch releases a canvas. The `feedback` block key and the directory
  are one name, and the layout gate rejected two literal spellings at once, so both read
  `FEEDBACK_DIR_NAME`. Measured through a real `App`: a save moves `document.json`'s mtime,
  the sync releases the live `Document` (its `_feedback` with it) and loads fresh, and the
  first frame after continues from the seed -- consistent, a read after every write. The reset
  case runs through `Document.reset`, since the session half is the script engine's.
- **W-D:** `headline_ms` was lifted so the sort and the printed number agree by construction;
  V11 rides inside `test_the_wire_and_the_abort_path` because one test per process may drive
  `update_and_draw`. For the maintainer's eyes: a parent without a GPU span prints its CPU wall
  and can read cheaper than its GPU-bound child (4 ms green over 9 ms yellow) -- unchanged
  numbers, louder now that they are colored; the headline rule is the lever if it reads wrong.
- **Tooling:** `make gates` writes `$TMPDIR/shaderbox-gates.log`, one path for every worktree, so
  concurrent runs overwrite each other's log; each agent set its own `TMPDIR`. Under `xvfb-run`
  two GPU timer-query tests read a saturated `0xFFFFFFFF` ns on llvmpipe; they pass on the real
  display.
