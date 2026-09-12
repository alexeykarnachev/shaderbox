# 091 post-implementation review — code correctness

Subject: commit `4dc1423` ("091: import another document's passes as a group"). Anchors:
`01_spec.md` (D1–D11, Verification, Implementation notes) and the code. Read-only; every
probe ran against a worktree pinned at `4dc1423`, so nothing here depends on the
implementer's in-flight edits. Where a finding is still live in the working tree as of this
writing, the report says so.

## Verdict

**FINDINGS** — 2 that make the feature wrong, 1 false claim in a gate, 3 minor.

Finding 1 (the cycle) is the one that costs a user their document. Finding 2 (the torn
import) is rarer to trigger but leaves the host dirty in a way no verb undoes. Everything
else is cosmetic or a spec correction.

---

### F1 (high) — the dialog's own defaults can build a cycle, and the import accepts it

`plan_import` (`shaderbox/pass_import.py:88-93`) validates that each handover pair *reads a
fed pass*. It never checks whether the host pass **taking** the handover is itself fed, or
an ancestor of one. So feeding two entry points from two host passes where one reads the
other produces `bundle → host → bundle`.

This is not a hand-crafted input: it is what the dialog proposes. `App.set_import_substitution`
(`shaderbox/app.py`) turns every reader of a newly fed pass **on by default** (D6), so the
user picks two combo entries and clicks Import.

Probe — host `main → mid` (mid is the output), source with entry points `a`, `b` and output
`mix`; `a ← main`, `b ← mid`, handovers left at their defaults:

```
DIALOG DEFAULT subs: {'a': 'main', 'b': 'mid'} handovers: [('mid', 'u_main')]
result: ImportResult(error='', notes=())
  fx_mix: {'u_a': 'main', 'u_b': 'mid'}
  main: {}
  mid: {'u_main': 'fx_mix'}
ERRORS: [('fx_mix', 'passes form a cycle: fx_mix -> mid -> fx_mix.'),
         ('mid',    'pass is not ordered: an input is on a cycle.')]
output: fx_mix   evaluation_order: []
rendered output pixel: [0, 0, 120, 255]
graph_errors after render: [('fx_mix', ...), ('mid', ...)]
```

Driving the real modal gives `draft.rejection == ''` for four consecutive frames with this
exact selection, i.e. the Import button is **enabled**.

What the user sees afterwards: `becomes_output` fired (the host output `mid` was fed), so the
document's output is now `fx_mix`; `evaluation_order` is empty, `Document.render` falls back
to `order = [resolved]` (`shaderbox/document.py:788-791`) and draws the bundle's output pass
alone against black inputs — a plausible-looking wrong picture, not a blank one. Nothing
surfaces the error: `Document.graph_errors` is populated but `grep -rn graph_errors
shaderbox/widgets/ shaderbox/popups/ shaderbox/tabs/` returns nothing, so the cycle is
silent. The document saves in that state.

Still live in the working tree: `grep -n "cycle\|plan_passes" shaderbox/pass_import.py`
returns nothing.

**Fix.** `plan_import` already holds both wirings and every rewrite it intends, so the cycle
is decidable before any write. Build the post-import wiring inside `plan_import` — the host's
wiring with the handover rows applied, plus the copied passes' `sources` rows — and run
`plan_passes` on it; reject with the cycle's own message when it reports one. That keeps the
rejection in the pure function, so the dialog's per-frame plan disables the button and shows
the reason, and the verb rejects the programmatic route for free. It also wants a
`test_pass_import.py` case for the shape above (falsifier: drop the check and the two-fed-pass
plan comes back valid).

A narrower fix — reject a handover whose host pass is in `fed`, or is an ancestor of a fed
pass — closes the reachable shape but not the general one; the `plan_passes` call costs
nothing and covers both.

### F2 (medium) — a failure part-way through the copy loop leaves the host with orphan passes that resurrect on the next load

`ProjectSession.import_passes` (`shaderbox/project_session.py`) writes in this order, per
copied pass, inside one loop:

1. `path.parent.mkdir` + `path.write_text(source_pass.source.text)` — the pass file;
2. `Pass(...)` + `render_pass.compile()`;
3. `_copied_uniform_value` per uniform of the source pass;
4. the plan's `PassSource` rows;
5. `host.passes[host_name] = render_pass` and `entries[host_name] = entry`.

Only after the loop does `host.graph = host.graph.with_passes(...)` run, then the handovers,
the `ui_uniforms` merge, and the single `save_ui_document`. So the plan *is* fully decided
before the first write (D5 holds), but the **execution is not atomic**, and step 3 is a
user-reachable raise: `_copied_uniform_value` re-opens a bound `Image`/`Video` from
`details.file_details.path`, which raises `FileNotFoundError` (PIL) or `ValueError` (`Video`)
when the file has been moved or deleted since the source bound it.

Probe — the bloom fixture with an `Image` bound on `trail` whose file is then unlinked:

```
RAISED: FileNotFoundError [Errno 2] No such file or directory: '.../x.png'
orphan passes in document.passes: ['bloom_blur', 'bloom_bright', 'bloom_composite']
each holds a GL program: {'bloom_blur': True, 'bloom_bright': True, 'bloom_composite': True}
```

`document.passes` has three passes the graph does not. No GL leak — `Document.release`
iterates `self.passes`, so they are freed on close. The damage is persistence:
`UIDocument.save` writes a pass FILE for every member of `document.passes`
(`shaderbox/ui_models.py:429-448`), and the loader enumerates FILES. So the next save of this
document — any verb, the copilot, quit — commits the orphans, and the next load brings them
back as real passes with default entries and **no group**:

```
add_pass err:                      (any later verb saves)
AFTER NEXT SAVE graph.json passes: ['main', 'zzz']
AFTER NEXT SAVE files: ['bloom_blur...', 'bloom_bright...', 'bloom_composite...', 'main...', 'zzz...']
after save+reload, passes: ['bloom_blur', 'bloom_bright', 'bloom_composite', 'main']
groups: {'bloom_blur': '', 'bloom_bright': '', 'bloom_composite': '', 'main': ''}
```

The user is left with a half-bundle carrying no group label, to delete pass by pass. The
exception also propagates out of `import_passes` uncaught — through
`App.import_passes_from_draft` into the frame loop — rather than coming back as
`ImportResult.error`.

**Fix.** Two halves, both small.

Move the fragile work out of the write loop: copy every pass's uniform values into a
`dict[str, dict[str, Any]]` *before* the first `write_text`, so a missing media file is a
rejection with nothing written. `_copied_uniform_value` needs no GL-ordering guarantee
relative to the file writes, so this is a pure reordering.

Then make what remains recoverable: wrap the loop so a raise unlinks the files it wrote and
drops (and releases) the passes it put into `host.passes`, and return the message as
`ImportResult(error=...)`. A verification item belongs with it — inject a failure on the
third pass and assert `document.passes`, `graph.passes` and `passes/` are all as they were
(falsifier: the current code, which leaves three of each).

### F3 (medium) — the commit does not pass `make check`, though the spec's Implementation notes say the gate was green

`ai_docs/features/091_presets/01_spec.md` Implementation notes: *"Landed 2026-09-12; `make
gates` green (check, test, smoke), exit code read unpiped."*

At `4dc1423`:

```
$ uv run ruff format --check .
Would reformat: shaderbox/pass_import.py
Would reformat: tests/test_import_dialog.py
2 files would be reformatted, 313 files already formatted

$ uv run ruff check .
shaderbox/popups/import_passes.py:18  help: Organize imports
Found 1 error.  [*] 1 fixable with the --fix option.
```

`make gates` stops at `check` with exit 2 (`== gates: FAILED at check (exit 2); test and
smoke not run ==`). `test` and `smoke` *are* green when run directly: 2262 passed / 4 skipped,
and `scripts/smoke.py` exits 0 with `smoke: OK (200 frames, 7 documents)`. So the claim is
wrong about one of its three gates, and it is the gate that would have caught it.

Worth noting how this slipped: `make gates` runs `check` twice on purpose and only believes
the second run, because pre-commit exits non-zero whenever a hook *modified* a file. Reading
the first run's "hooks REWROTE files" line as that benign case — without re-running over the
settled tree and without staging what the hooks wrote — produces exactly this commit.

**Fix.** Run `ruff check --fix . && ruff format .`, commit the result, and correct the
Implementation-notes sentence. (The three files are already reformatted in the working tree
as I write this, presumably by a later `make check`.)

### F4 (low) — `CopilotCapabilities.set_pass` declares `group` with no default; the backend gives it one

`shaderbox/copilot/capabilities.py` (Protocol): `..., new_name: str, group: str | None, /`.
`shaderbox/copilot/backend.py` (impl): `..., new_name: str, group: str | None = None`.

So a caller typed against the Protocol cannot omit `group`, contradicting D9's "None = keep":

```
$ uv run pyright _chk_proto.py
error: Expected 1 more positional argument (reportCallIssue)
```

Nothing catches this today because the five existing call sites in
`tests/test_copilot_pass_tools.py` reach `app.copilot_backend` through the `app` fixture,
typed `Any`. Behaviour through the tool is correct — I verified all of D9's cases (below) —
this is only the Protocol drifting from its implementation.

**Fix.** `group: str | None = None` in the Protocol, or drop the default from the backend and
pass `None` explicitly at the one call site in `tools/passes.py` (which already always passes
it).

### F5 (low) — `preview_cell(bordered=False, border_color=<color>)` silently drops the border

`shaderbox/ui_primitives.py:1262-1275`: `border_color` pushes `Col_.border`, but the child
gets `ChildFlags_.always_use_window_padding`, which draws no border — so the pushed colour
has nothing to tint. The style push/pop stays balanced (`n_styles` counts regardless), and
the one caller is safe: `widgets/pass_list.py` passes `bordered=tint is None or border is not
None`, so a tile with an accent or error border is always `bordered=True`. The trap is for
the next caller.

**Fix.** One line in the docstring saying `border_color` needs `bordered=True`, or — better —
assert the combination is not asked for, since it is meaningless rather than merely unusual.

### F6 (low) — `_draw_group` shows an empty group picker on a document with no groups

`shaderbox/popups/pass_settings.py:_draw_group` draws the `##group_pick_*` combo
unconditionally. On a document where nothing carries a group the dropdown holds only "none".
The import dialog gets this right for its own disabled case
(`imgui.begin_disabled(is_output)` around the entry-point combo). **Fix:** wrap it in
`begin_disabled(not existing)`, matching the sibling.

---

## Coverage

Read **end to end** (whole file, not the hunks):

- `shaderbox/pass_import.py` (99 lines) — the whole module.
- `shaderbox/pass_graph.py` (505) — whole file, with attention to `entry_points`,
  `group_slug`, `group_runs`, `with_passes`/`with_group`, `plan_passes`, `wired_pass`,
  `PassEntry.group`'s pattern.
- `shaderbox/popups/import_passes.py` (254) — the whole module.
- `shaderbox/widgets/pass_list.py` (277) — the whole module.
- `shaderbox/ui_models.py` — `ImportDraft`, `UIUniform`, `UIDocument.save`,
  `load_document_from_dir`, `load_documents_from_dir`, `_uniform_entry`, `_existing_rows`.
- `shaderbox/project_session.py` — every line the commit touched plus its neighbours:
  `ImportResult`, `compile_pending_passes`, `offered_entry_points`, `_copied_uniform_value`,
  `set_pass_group`, `import_passes`, `_pass_name_error`, `_graph_without`, `_graph_renamed`,
  the capability wiring.
- `shaderbox/app.py` — the whole 091 block (`PopupState.IMPORT_PASSES`, `import_draft`,
  `open/close_import_passes`, `import_sources`, `import_source`, `select_import_source`,
  `host_readers_of`, `set_import_substitution`, `import_passes_from_draft`,
  `open_pass_settings`, `commit_pass_group`, `close_pass_settings`,
  `create_pass_from_draft`, the command binding) plus `any_popup_open`.
- `shaderbox/ui.py` — `planned_set_mode`, `_tick_frame_state`'s gate and planned-set block,
  `_update_and_draw`'s three render branches, the popup chain.
- `shaderbox/ui_primitives.py` — `preview_cell` in full, `_chip_row`, `row_label`/`label_row`.
- `shaderbox/popups/pass_settings.py` — the commit's additions (`_draw_group`,
  `_existing_groups`, both call sites).
- `shaderbox/theme.py` — the `GROUP_TINTS` block, `group_tint`, both asserts.
- `shaderbox/copilot/tools/passes.py` (157) — whole file;
  `shaderbox/copilot/capabilities.py` + `backend.py` — the `set_pass` / `_configure_pass` /
  `_pass_table` changes and their signatures.
- `shaderbox/core.py` — `Pass.__init__`, `invalidate`, `release`, `release_program`,
  `UniformValue`, `Canvas.release`.
- `shaderbox/media.py` — `MediaWithTexture`, `Image` (all four `src` branches), `Video.__init__`,
  `media_class_for`, `FileDetails`.
- `shaderbox/document.py` — `effective_wiring`, `_reads_of`, `document_dir_of`,
  `graph_errors`, `release`, the render path's order/cycle fallback.
- `shaderbox/util.py` — `try_to_release`.
- `shaderbox/hotkeys.py`, `commands.py`, `paths.py`, `scripts/smoke.py` — the commit's
  additions (each is 1–5 lines).
- `ai_docs/features/091_presets/01_spec.md` (656) — whole file.
- `tests/test_import_dialog.py` (new, whole file), `tests/conftest.py`, the 091 block of
  `tests/test_pass_verbs.py`, and the `Makefile`'s `check` / `test` / `smoke` / `gates`
  targets.

**Skipped, and why:** the non-091 bodies of `shaderbox/ui.py` (1083), `shaderbox/app.py`
(2265), `shaderbox/ui_primitives.py` (1828) and `shaderbox/theme.py` (683) — I read the
commit's hunks plus every symbol the hunks call or are called by, and stopped at the files'
unrelated halves (the exporters, the editor tabs, the copilot panel, the FPS panel). The
other 091 test files (`test_pass_import.py`, `test_pass_graph.py`,
`test_pass_strip_layout.py`, `test_theme.py`, `test_graph_persistence.py`,
`test_render_decoupling_loop.py`, `test_copilot_pass_tools.py`,
`test_ui_prose_budget.py`) I ran rather than read line by line — they are the spec's own
gates, and the spec-fidelity reviewer owns whether they match their items.

**Ran:** the full suite in the pinned worktree (`pytest tests/ -n 8 --dist loadgroup` →
2262 passed, 4 skipped, exit 0); `scripts/smoke.py` → exit 0; `make gates` → exit 2 at
`check`; `ruff format --check` / `ruff check` / `pyright` individually; and 17 `uv run
python` probes driving a real headless `App` the way `tests/conftest.py`'s `app` fixture
does.

---

## False trails

Probed, and fine — recorded so they are not re-checked.

1. **`_copied_uniform_value` over the whole value domain.** Every member of `core.UniformValue`
   plus the three `SamplerSource`s, checking object identity and then releasing the copy and
   reading the source:
   ```
   int/float/tuple/list    -> same object (immutable, or never mutated in place)
   PassSource/NoSource/AutoSource -> same object (frozen, holds no GL)
   Image(path)             -> new Image, file path preserved; src tex readable after copy.release()
   Image(no path)          -> new Image via texture_to_pil (a pixel COPY); src tex alive + readable
   Texture                 -> new glo; src readable after copy.release()
   Buffer                  -> new glo; src readable after copy.release()
   ```
   The `Image(value.texture)` fallback **is** independent: `Image.__init__` routes a
   `moderngl.Texture` through `texture_to_pil`, so the object owns a new PIL image and
   allocates its own texture lazily. The shared `list` is harmless — `widgets/uniform.py`
   always replaces `uniform_values[name]`, never mutates the sequence in place
   (`grep -rn "uniform_values\[.*\]\["` is empty).
2. **No double-release, and no release of anything the source still holds.** Instrumenting
   `try_to_release` across a real import shows 7 calls on `None` and 1 on a `PassSource`,
   releasing nothing. The copy-loop overwrite can only hit a value `_copied_uniform_value`
   just made; the handover overwrite can only hit a `PassSource`/`AutoSource`, because a
   sampler is offered as a handover reader only when `wired_pass` resolves it to a pass, and
   `wired_pass(Image, ...)` is `None`. D6's premise that the overwritten value "may be a
   bound `Image`/`Video`/`Texture`" is **unreachable** — the `try_to_release` there is dead
   defensive code. Verified no value object is shared between source and host after an import
   from a shipped example.
3. **`compile_pending_passes` disturbs nothing the Examples popup or the planned set reads.**
   On the six-pass Radiance Cascades example held in `ui_document_examples`: before, every
   pass `(program=False, first_render_done=False, drawn_frame=-1)`; after, only `program`
   flips to `True`. `Document.first_render_done` stays `False`, so the one-example-per-frame
   pending-first election and the card staleness are untouched. D3's measured claim holds.
4. **`get_item_rect_min/max` after `preview_cell` returns the tile rect.** `(8, 26)`–`(176,
   224)`, i.e. 168×198 = tile + footer line + chip line, on imgui 1.92.8 / imgui_bundle
   1.92.801. The strip's segment accumulation is sound too — traced over five layouts
   (all-one-group wrapping, an interrupting outside pass, two adjacent groups, a run wrapping
   mid-group, an offset wrap): every segment stays inside one row, and a wrapped run yields
   one segment per row.
5. **The group outline's clip and paint order are right.** `parent.get_clip_rect_min/max`
   inside a scrolled strip equals the strip window's visible rect (y 26..146 for a window at
   26 with height 120) while the tile rects run from -34 to 526, so the foreground outline and
   label *are* clipped to the strip. The fill is appended to the parent's draw list after the
   tiles (vertex 226 of 258), but that does not put it over them: a child window gets its own
   `ImDrawList` which imgui merges *after* the parent's at render, so child geometry always
   paints on top. `push_clip_rect`/`pop_clip_rect` and `push_font`/`pop_font` are paired on
   every path.
6. **`fg.add_text` honours `push_font` on this build.** Measured from the vertex buffer:
   "bloom" under `get_font(24)` spans 63×15 px, under `font_12` 34×8. The label is drawn in
   the 12px face as intended.
7. **`always_use_window_padding` is byte-identical to `borders` for origin AND avail.**
   `borders` → origin (8, 8), avail (152, 164); `always_use_window_padding` → identical;
   `none` → (0, 0) and (168, 180). D7's claim holds, and verification 18's narrower assertion
   (origin only) is adequate.
8. **The handover checkbox ids are unique.** On this build the `##` tail does *not* collapse
   the id: `imgui.get_id("A##x") == imgui.get_id("B##x")` is `False`, and two checkboxes
   labelled `grade.u_main##handover` / `mask.u_scene##handover` get different `get_item_id()`
   values, while two identical labels collide as expected. The `pair[0].pair[1]` prefix
   carries the uniqueness; `##handover` is cosmetic. (Note for the UI skill: the repo's
   mental model of `##` as "the id is what follows" is wrong on imgui 1.92.8 — the id hashes
   the whole string.) `indent`/`unindent` are unconditional and paired, and
   `begin_disabled(is_output)` / `end_disabled` pair on every branch including the
   `begin_combo`-returns-False path. Four frames of the real modal with two fed entry points
   and three handover checkboxes raise no assert.
9. **The tab read-back with `tab_select_pending` is self-healing.** Forcing
   `draft.examples_tab = False` without the pending flag has the read-back call
   `select_import_source("", examples_tab)`, which restores the tab imgui owns and clears the
   stale source — so a half-switched draft cannot plan against the wrong tab's document.
10. **`set_import_substitution`'s shared-host-pass bookkeeping is correct.** With `a ← main`
    and `b ← main` and both of `main`'s readers handed over, setting `a` back to `keep` leaves
    `handovers == host_readers_of('main')` — the guard `previous not in {fed for name, fed in
    substitutions.items() if name != entry}` does its job. Same when `a` switches to a
    different host pass instead of to `keep`.
11. **No double-render, and nothing renders behind the wrong modal.** Counting
    `Document.render` calls per frame of the real `_update_and_draw` with two project
    documents and render-all on: closed → 2 (one each), SETTINGS → 0, EXAMPLES → 1 (the
    first-render election caps it), IMPORT on the project tab → 2, IMPORT on the examples tab
    → 2 (current + one example). No document appears twice in any state. With
    `popup_state = PASS_SETTINGS` and a still-populated `import_draft`,
    `planned_set_mode` returns `(False, False)` and exactly the current document renders — the
    `or import_project_tab` disjunct cannot leak into another modal's branch, and the `elif
    PASS_SETTINGS` branch stays reachable.
12. **`set_pass(..., group)` through `tools/passes.py` is correct in every case D9 names.**
    Omitted → pydantic parses `None` → group untouched (and kept when already set);
    `group="fx"` → set, and the echoed table line reads
    `- main [output]: runs 2, target f1 x1, linear, group fx`; `group=""` → cleared;
    `group="2bad"` → `error: a group name starts with a letter...` with the old group intact;
    `group` + `new_name` in one call → the renamed pass carries the group (the
    `_on_pass_renamed` hook at `app.py:729` moves `pass_settings_name` forward, which is also
    why `close_pass_settings`'s rename-then-`commit_pass_group` order is safe — I probed that
    separately and the group lands on the new name).
13. **`plan_import`'s rejections over ten edge cases.** A stale source output (`output_pass`
    → `None` → `""`) picks `renames[copied[0]]`, i.e. the alphabetically first copied pass, as
    the bundle output — odd but spec-consistent (D4) and only reachable on a hand-broken
    `graph.json`. A group with surrounding whitespace is rejected by `PASS_NAME_RE` (the
    dialog `.strip()`s first, so only the programmatic route sees it). Collisions name one or
    several passes correctly, `''` as the group collides on bare names, a handover on an
    unknown host pass is rejected, and a single self-reading pass that is also the output is
    rejected with "is the output and stays". An empty `source_wiring` answers "nothing to
    import: every pass is replaced", which is misleading prose for a document with no passes
    — but such a document does not load.
14. **The source is not mutated in memory either.** After importing from a shipped example,
    the example's pass `source.path`s, its `ui_state.ui_uniforms` keys and every pass's
    `uniform_values` are unchanged, on top of verification 6's on-disk byte-identity. The
    `ui_uniforms` merge uses `setdefault(key, row.model_copy())`; `UIUniform` has only scalar
    fields, so the shallow copy is sufficient.
15. **`UIDocument.save` keeps the merged rows without the import compiling the copies.**
    Confirmed by reading `save`: it compiles every program-less pass before the prune
    (`ui_models.py:414-417`), so verification 7's parenthetical is right and the copies'
    in-import `compile()` is for the panel's sake, not the prune's.

---

## A note on process

My first `make gates` run rewrote three files in the working tree: the run's pre-commit hooks
applied ruff's formatter and import sort to `shaderbox/pass_import.py`,
`shaderbox/popups/import_passes.py` and `tests/test_import_dialog.py`. I restored all three
to their `4dc1423` blobs (`git show 4dc1423:<path> > <path>`, tree verified clean afterwards)
and did the rest of the work in a pinned worktree. The rewrite is what surfaced F3; the diff
was formatting only.
