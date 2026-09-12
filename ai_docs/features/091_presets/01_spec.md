# 091 — Presets: import another document's passes as a group

Status: **locked 2026-09-12**, revised after pre-implementation review round 1 (D1–D11 are
fixed premises; the review's edits are folded in below and listed in *Review history*). The
design was settled in chat on 2026-09-11/12 and sketched in `00_mock.html` (sections 1·C3, 2b
and 3 are the chosen shapes; the rest is the brainstorm record). This file is the
implementation contract.

Size: **mid**. 2 pre-implementation reviewers (correctness & design; verification &
blast-radius: `reviews/pre_*.md`), 3 post (code correctness; architecture & conventions; spec
fidelity, since the import verb touches the persistence funnel and the strip).

---

## Goal

A multi-pass effect built once (a bloom chain, a radiance-cascades stack) can be dropped into
another document and fed from that document's own passes, without rebuilding it pass by pass.

- **Import by copy.** The source document's passes, their targets, run counts, uniform values
  and bound assets are copied into the current document under a **group**. The source is not
  referenced afterwards; editing it changes nothing here.
- **One decision per entry point.** An entry point is a pass of the source that reads no other
  pass of the source. The dialog asks, per entry point, whether it is kept (copied as it is)
  or fed by one of the host's passes, in which case every copied sampler that read it now reads
  the host pass, and the host's readers of that pass are handed to the bundle's output (D6).
  Nothing else is asked.
- **The group is a label on the pass**, written by import and editable by hand in the pass
  settings modal. The strip draws a group's consecutive tiles inside one flush outline with the
  group's name on the border (mock 1·C3). Nothing folds.
- **Sources within reach:** the project's own documents and the shipped examples, in one
  window with two tabs (mock 2b).

The worked example throughout is a five-pass bloom chain (`scene`, `bright`, `blur`, `trail`,
`composite`; `bright.u_scene` and `composite.u_scene` read `scene`, `trail.u_prev` reads
itself). It is not a shipped example: it is the test fixture `tests/fixtures/bloom_chain/`
(and a document in the maintainer's own project). The only multi-pass shipped example is
Radiance Cascades, whose one entry point `paint` has no samplers.

---

## Out of scope

- **A cross-project presets folder** (`app_data_dir()/presets/`, the shader-lib posture). The
  import verb takes a loaded source `Document` (its directory is derived from its pass files by
  `document_dir_of`), and the dialog lists sources from a list of `(label, dict[str,
  UIDocument])` tabs, so a third tab backed by that folder is additive. Trigger: the maintainer
  wants a bundle in a second project.
- **Export a group as a document.** The inverse of import (a group's members written as a
  standalone document dir, boundary samplers as explicit black rows). Trigger: the presets
  folder above, which it would feed.
- **Folding a group** into one tile. Rejected in chat: a folded group must be convex in the DAG,
  which turned into a rule set the UI could not carry. Trigger: none. A group whose members are
  not contiguous simply draws as two runs (D7), which is the case folding would have had to
  special-case.
- **A second, opt-in graph view of the passes.** 070 closed the graph view as the strip's
  replacement for a six-pass document at 480px. In this design round the maintainer reopened it
  as an opt-in second view for a document that has outgrown the strip, hand-drawn on the draw
  list (no `imgui_node_editor`, for a future port). That is its own feature, numbered when it is
  specced; 091 depends on nothing in it, and 070's decision gets its revisit pointer when that
  feature lands. Trigger: 091 has landed.
- **A copilot `import_passes` tool** and a `preset:` read prefix. The pass table and `set_pass`
  do learn `group` (D9), so what the model reads matches `graph.json`. Trigger: a dogfood
  scenario that asks for a bundle by name.
- **A group rename verb.** Renaming a group is retyping it on its members. Trigger: a group of
  more than five passes that the maintainer wanted renamed.
- **Nested groups.** A source that itself carries groups is imported under the new group with
  its inner labels dropped. Trigger: a source with two bundles worth keeping apart.
- **Importing the source's script.** A document has one script (048) and its pass blocks name
  the source's passes. The dialog says the script is not imported. Trigger: a bundle whose
  effect lives in its script.
- **Importing feedback seeds.** Copied feedback passes start black, which is what a freshly
  built `Pass` already does (`_feedback` is filled only at load and at render). Trigger: none.

---

## Design decisions

**D1 — `PassEntry.group: str = ""` is the group.** One field on the existing entry, so
`graph.json` gains one key per grouped pass and nothing else. A group exists while at least one
pass carries its name and is gone when the last member leaves; there is no group table, no
member list, no group object (a second name-keyed structure would have to agree with the
first — `conventions.md`, the lockstep-dicts bullet). The name obeys `_PASS_NAME_RE` (it is
drawn on a border and prefixes filenames). The field rides `load_graph`'s per-entry salvage
unchanged, since the salvage enumerates the model's fields; every existing mutation
(`_graph_renamed`, `_graph_without`, `with_target`, `set_pass_iterations`, `load_from_dir`'s
fill) carries the entry object or goes through `model_copy`, so the group survives them with no
edit. Two sites compare against `PassEntry()`'s field defaults (`app.py` around
`create_pass_from_draft`, target and iterations) and must not be read as "is this entry
default" once `group` exists.

**D2 — the group name is the prefix.** Importing under group `bloom` names the copied passes
`bloom_<source name>`. An empty group name imports the passes under their own names with no
group. The dialog prefills the name from the source document's display name through
`group_slug(name: str) -> str` in `pass_graph.py`: the first word, lowercased, characters
outside `[A-Za-z0-9_]` replaced by `_`, `g_` prepended when the result does not start with a
letter or underscore, and `preset` for an empty name (`Bloom Chain` → `bloom`, `Radiance
Cascades` → `radiance`, `2D SDF` → `g_2d`, `""` → `preset`). A document whose name was never
set displays its directory name, so its slug is a long UUID-shaped word; valid, and the user
retypes it. The first word and not the whole name because a tile is 168px and the 14px face
fits about 21 characters: `bloom_composite` fits, `radiance_cascades_composite` clips.

**D3 — entry points are the roots of the source's effective wiring.** `entry_points(wiring)` in
`pass_graph.py`: the passes whose wiring has no source other than themselves (a self-read is
feedback, not an input). Pure, GL-free. The bloom fixture has exactly one (`scene`), Radiance
Cascades one (`paint`); a single-pass document's one pass is both entry point and output. The
wiring is `Document.effective_wiring()` of the source **after every source pass has been
compiled** (`Pass.compile()` on each with no program and no errors; `_bring_chain_online` alone
stops at the output's chain). This is not belt-and-braces: Radiance Cascades carries no
explicit sampler rows at all, so before compiling its wiring is empty and every pass reads as a
root. Compiling touches only `compile_unit` / `program` / `vbo` / `vao`; the source's
`first_render_done`, `drawn_frame` and feedback are untouched (measured in review round 1). A
source pass whose compile fails contributes its explicit rows only, and the import result
names it (D5).

**D4 — a substituted entry point is not copied; its readers are re-pointed.** For each entry
point the dialog holds a choice: `keep` or a host pass name. The import plan is a pure function
in a new leaf module `pass_import.py`, importing `pass_graph` only:

```python
@dataclass(frozen=True)
class ImportPlan:
    renames: dict[str, str]          # source name -> host name, for every COPIED pass
    sources: dict[str, dict[str, str]]  # host name -> sampler -> host pass it reads
    output: str                      # the bundle's output under its host name
    handovers: dict[str, dict[str, str]]  # HOST pass -> sampler -> `output` (D6)
    becomes_output: bool             # the document output moves to `output` (D6)

def plan_import(
    source_wiring: Wiring, source_output: str, group: str,
    substitutions: Mapping[str, str], handovers: Collection[tuple[str, str]],
    host_wiring: Wiring, host_output: str,
) -> ImportPlan | str
```

`host_wiring` is the host's `effective_wiring()` after the host's passes have been compiled
(D6), and its key set is the host's pass names; `handovers` is the set of `(host pass,
sampler)` pairs the dialog's checkboxes left on (D6).

Returning `str` is the rejection (the message the dialog shows): a copied name already among
the host's passes, a group name failing `_PASS_NAME_RE`, a substitution naming a pass that is
not an entry point or a host pass that does not exist, a handover naming a pair that does not
read a fed pass in `host_wiring`, or a plan that copies nothing (a single-pass source whose one
pass is substituted).

`sources` carries only the WIRED subset: every sampler of a copied pass that the source wiring
filled becomes an explicit `PassSource` there — to the renamed pass, to the host pass when the
source was substituted, and a self-read to the renamed self (written, not skipped as "already
right": `u_prev` resolves by name today and would break the day a group is renamed). Name-rule
wiring does not survive the prefix (`u_blur` no longer names `bloom_blur`; measured:
`wired_pass(AutoSource(), "u_bright", "bloom_blur", …)` is `None`), so materializing is not
optional. Samplers the wiring did NOT fill are not the plan's business: their values ride the
uniform-value copy of D5 unchanged — a bound texture is copied, a `NoSource` stays a decision,
an `AutoSource` stays undecided. An undecided `u_paint` on a copied pass may therefore catch a
host pass called `paint` by name after import; that is the name rule (069 D9) applied to the
host's namespace and is intended, since the alternative (materializing every undecided sampler
to black) would take away the rule the maintainer authors against.

**D5 — `ProjectSession.import_passes` executes the plan and saves, like the six pass verbs.**

```python
@dataclass(frozen=True)
class ImportResult:
    error: str = ""            # the rejection; "" on success
    notes: tuple[str, ...] = ()  # what was imported degraded: a source pass that did not compile

def import_passes(
    self, document_id: str, source: Document,
    group: str, substitutions: Mapping[str, str], handovers: Collection[tuple[str, str]],
) -> ImportResult
```

In order: compile the source's and the host's program-less passes (D3, D6); plan (D4; reject
before touching anything); for each copied pass write its source text to `passes/<host
name>.frag.glsl`, build a `Pass` from that path with the source entry's target and **compile it**
(exactly as `add_pass` does; a never-compiled copy has no live uniforms, so `UIDocument.save`'s
`ui_uniforms` prune would drop every merged row in the same save that wrote it); copy the entry
with `group` set; copy uniform values over the whole of `core.UniformValue` plus the three
`SamplerSource` members, with no default branch: scalars and tuples by value; `PassSource` /
`NoSource` / `AutoSource` by reference (frozen dataclasses, the reference is the value); a
`moderngl.Buffer` via `gl.buffer(buf.read())`; an `Image` re-opened from its file under the
source's `media/<source pass>/` (never `Image(value.texture)`, which comes back with an empty
`file_details` and the media panel loses the path and size); a `Video` re-opened from
`details.file_details.path`; a raw `moderngl.Texture` via `gl.texture(size, components,
data=tex.read(), dtype=...)`. The source directory is `document_dir_of(source)`, the one place
that knows the `passes/` depth. Then overwrite the wired samplers with the plan's `PassSource`
rows; write the handover rows on the host passes, releasing each overwritten value first
(D6); set the output when `becomes_output`; save the HOST's `UIDocument` once through
`save_ui_document` (never the source's: a shipped example lives in the read-only resources dir
and must be byte-identical afterwards). Copying value objects rather than sharing them is what
keeps `Pass.release` (which releases every value it holds) from freeing a texture the source
still uses.

`ui_uniforms` rows are merged from the source's `ui_state` for hashes the host does not have.
`get_uniform_hash` is keyed by name and shape, not by pass, so a copied `u_amount` keeps the
source's row under the same hash with no re-keying, and where the host already has a row of
that name and shape the HOST's row wins: the source's input type and range are lost in that one
case, which is the right precedence and the limit of the merge.

**D6 — the bundle's output takes over the role of the pass that feeds it.** Feeding an entry
point with host pass `scene` is an INSERTION: every host sampler that read `scene` now reads the
bundle's output, and if `scene` was the document's output, the bundle's output becomes the
output. Both shapes fall out of one rule:

```
scene → grade → final        grade.u_scene read scene, now reads bloom_composite; final stays output
scene (output)               scene had no readers; bloom_composite becomes the output
```

The takeover is explicit and per reader, because one reader can legitimately want the raw
pass (a `mask` cutting a shape from the unbloomed scene). Under an entry-point row whose combo
names a host pass, the dialog lists that pass's host readers as `(pass, sampler)` checkboxes,
all on by default: `then read bloom_composite instead of scene: ☑ grade.u_scene ☐
mask.u_scene`. With no readers and the fed pass being the output the line reads
`bloom_composite becomes the output`. Nothing is shown for `keep`. Two fed entry points each
get their own line; both hand over to the same bundle output. The plan carries the result as
`handovers` and `becomes_output` (D4).

The readers come from `host_wiring`, and an uncompiled host pass answers with its explicit
rows only — the same hole D3 closes for the source, measured the same way (the uncompiled
bloom fixture's wiring is `{'blur': {}, …}`, every pass a root). So the host's program-less
passes are compiled when the dialog opens and again inside `import_passes` before planning; a
host pass whose compile fails cannot take a handover (its rows would be carried forward from
disk by `UIDocument.save`'s `program is None` branch and the handover silently dropped), so a
handover naming such a pass is rejected with a message naming it.

A handover row overwrites a host sampler whose old value may be a bound `Image` / `Video` /
`Texture`, so `import_passes` releases it first (`try_to_release`, the same call
`set_sampler_source` makes) and writes `uniform_values` directly, saving once at the end
rather than calling `set_sampler_source` per row, which saves per call.

**D7 — the strip draws a group as one flush outline (mock 1·C3).** In `pass_list.draw`, tiles
keep their order (`strip_order`) and their `SPACE.MD` gap; consecutive tiles of one group on
one row form a run (`group_runs(order, groups)` in `pass_graph.py`, pure, by adjacency and
never by name, so a group split by an outside pass is two runs). The mock's 4px in-group gap is
dropped: a per-run gap would break `tiles_per_row`'s single-gap arithmetic, and the outline
reads as one thing at 8px once the members' own borders are gone. `tiles_per_row` is then
genuinely untouched (one caller).

Per run the parent draw list gets a rounded rect **inset by 1px** from the run's outer tile
edges in the group's tint (`tiles_per_row` charges the last tile no trailing gap, so a full
row's slack is as little as 4px at a 700px panel and an outside rect clips), and the group
name on the top border at the run's left, on a `BG_SURFACE` fill so it reads over the line and
inside the first tile's top rather than above the strip. The rect is emitted BEFORE the run's
tiles from the same positions `same_line` will place them at (the strip knows every tile's
rect from `tiles_per_row` and the cursor), so the tiles paint over it; the label's fill is
emitted after the run. A run that wraps is two runs with the label on each.

Member tiles draw with the group tint as `bg_color` at low alpha and no border of their own:
`preview_cell` gains `bordered: bool = True`, which drops `ChildFlags_.borders` from the child's
flags — the tile's border is imgui's own child border, and `border_color` only tints it, so a
colour cannot turn it off. Dropping the flag also removes imgui's border inset, so an unbordered
cell pads its content by the border size to keep the picture and footer on the same pixel as
its bordered neighbours (the maintainer's locked answer: the cards keep their size). The strip
passes `bordered=border is not None`, so the accent output border and the red error border
still win on a grouped tile.

**D8 — group tints are theme tokens, picked by a stable hash.** `COLOR.GROUP_TINTS`, five hues
from the palette that no accent preset and no state color uses: `purple_b`, `green_n`,
`aqua_n`, `orange_n`, `blue_n` (`blue_b` is the blue accent's primary and `COLOR.TAG`, so it is
out). `group_tint(name) -> color` indexes by `zlib.crc32(name.encode()) % len(...)`, never by
`hash()`, which is salted per process. An import-time assert beside the `SELECT` invariant
pins that no tint equals an accent primary or a state hue as the belt; the GATE is a pure test
over the tuple (verification 7), because an import-time assert cannot be tripped from a test
without rewriting `theme.py` on disk.

**D9 — the group is editable by hand in the pass settings modal and by the copilot.** The
modal gains a `group` row under `name` (mock 3): an input with a combo of the document's
existing groups; committing writes `session.set_pass_group(document_id, name, group)` (a
seventh verb, validated by `_PASS_NAME_RE` or empty, saved like the others). In create mode
the row edits `draft.entry.group` and `create_pass_from_draft` applies it through
`set_pass_group` after `add_pass`, the way the draft's target and runs are applied. The tile
context menu gains `Leave group` on a grouped tile. The copilot's `set_pass` gains `group: str
| None` (None = keep, `""` = leave) APPENDED to the positional-only signature in
`capabilities.py`, `backend.py` and `tools/passes.py` (the five positional test call sites
lengthen; an inserted parameter would break them), and `_pass_table` appends `, group <name>`
to a grouped row's format string.

**D10 — the import dialog is one modal in the `PopupState` mutex (mock 2b).**
`PopupState.IMPORT_PASSES`, drawn by a new `popups/import_passes.py` with the module's own
early-return guard, imported and called in `ui.py`'s popup chain (which
`tests/test_project_management.py::test_every_popup_state_has_a_draw_call` already gates by
AST), opened from an `import…` button beside `add pass` and from a palette command
`IMPORT_PASSES` (a `COMMAND_SPECS` row AND an `app.command_callbacks` handler, both gated by
`test_command_registry_coverage.py`). `open_import_passes` refuses while
`app.copilot_turn_active`, through `_copilot_busy_blocked`, since the palette route does not
pass through the strip's `begin_disabled`. Escape reaches `close_import_passes` through its
own branch in `hotkeys.py`, the way `PASS_SETTINGS` does; the bare `popup_state = CLOSED`
fallthrough would leave the draft populated.

Its transient state is an `ImportDraft` dataclass on `App` (`ui_models.py`, beside
`PassDraft`): the active tab, the selected source id, the group buffer, `substitutions:
dict[str, str]` keyed by entry point, `handovers: set[tuple[str, str]]` (D6), and
`rejection: str`, the plan's message as of the last drawn frame (empty when the plan is
valid), stored so a test can read what the button read. The body: a tab row `This project |
Examples`; a card grid of that tab's documents through `draw_document_preview_button` (the
Examples modal's own grid), the current document excluded from the project tab; the selected
source's description; the `group` field; one row per entry point, `<name> ← [combo]` with
`theirs (copy it)` first and the host's passes after, the entry point's readers listed beside
it in the 12px face, and under a row fed by a host pass the D6 checkbox line; a note naming
the bundle's output and whether it becomes the document's; a note that the source's script is
not imported; `Import N passes` (disabled with the rejection in red while `plan_import`
rejects) and `Cancel`. Selecting a source compiles its program-less passes and the host's
(D3, D6), resets the group buffer to D2's slug, the substitutions to `keep` and the handovers
to empty; picking a host pass in a combo resets that entry point's handovers to all of that
pass's readers. The plan is recomputed each frame the dialog draws (never cached on
selection), which is how the button's enabled state and the message stay honest as the user
types.

**D11 — while the dialog is open, the planned render set is the open tab's documents plus the
current document.** The predicate `_tick_frame_state` computes as `examples_open` becomes "the
Examples popup, or the import dialog with its Examples tab active", and `planned` /
`planned_documents` / `current_planned` follow it unchanged (090 D10). The render chain gets NO
new branch: `ui.py`'s `elif EXAMPLES:` block takes that same predicate, so the
one-example-per-frame first-render election, the `renders_this_frame` gate and the profiler
span keep one home — `ui.py` already carries two copies of that budget rule and the funnel
bullet names the second sibling as the trigger, so a third copy is out. With the project tab
active the dialog renders the ordinary set, which is what the `if not any_popup_open():` branch
does once it takes the predicate's negation.

---

## Data model changes

`graph.json`, per pass entry:

```json
"bloom_blur": { "target": {...}, "iterations": 1, "group": "bloom" }
```

`GRAPH_JSON_VERSION` stays 2: an absent key reads as `""` and the field has a default, so
every existing file loads unchanged (verified: today's loader logs the key as unknown and
loads the rest, so the field is additive in both directions). No other file changes shape.
`projects/dev/` needs no hand edit.

---

## Files touched

- `shaderbox/pass_graph.py` — `PassEntry.group`, `PassGraph.with_group(name, group)`,
  `entry_points(wiring)`, `group_slug(name)`, `group_runs(order, groups)`.
- `shaderbox/pass_import.py` — new leaf: `ImportPlan`, `plan_import`. Imports `pass_graph`
  only.
- `shaderbox/project_session.py` — `ImportResult`, `import_passes`, `set_pass_group`.
- `shaderbox/ui_models.py` — `ImportDraft`.
- `shaderbox/app.py` — `PopupState.IMPORT_PASSES`, `import_draft`, `open_import_passes`
  (busy-guarded), `close_import_passes`, `import_passes_from_draft`, the `IMPORT_PASSES`
  command binding; `create_pass_from_draft` applies the draft's group.
- `shaderbox/commands.py` — `CommandId.IMPORT_PASSES` + its `COMMAND_SPECS` row.
- `shaderbox/hotkeys.py` — the `IMPORT_PASSES` Escape branch.
- `shaderbox/popups/import_passes.py` — new modal (D10).
- `shaderbox/popups/pass_settings.py` — the `group` row, edit and create modes (D9).
- `shaderbox/widgets/pass_list.py` — the `import…` button, the group outline (D7), `Leave
  group` in the context menu, `bordered=border is not None` on member tiles.
- `shaderbox/ui_primitives.py` — `preview_cell(bordered=...)` with the inset compensation.
- `shaderbox/ui.py` — the modal's draw call in the popup chain; the planned-set predicate
  (D11).
- `shaderbox/theme.py` — `GROUP_TINTS`, `group_tint`, the invariant (D8).
- `shaderbox/copilot/backend.py`, `capabilities.py`, `tools/passes.py` — `group` appended to
  `set_pass`, the table suffix (D9).
- `scripts/smoke.py` — stamps a group on two non-adjacent passes of the multi-pass document
  (verification 11).
- `tests/` — see Verification.
- `ai_docs/conventions.md ## Design decisions` — three bullets: the group is a label on the
  entry (D1, with the fold rejection and why); import is by copy with entry-point substitution
  and insertion (D3/D4/D6); group tints are stable-hash theme tokens (D8).
- `ai_docs/roadmap.md` — the 091 row and the banner.
- `ai_docs/dev_flow.md ### Module map` — `pass_import.py`, `popups/import_passes.py`, the
  strip's group outline.

---

## Verification

Each item names the falsifier that makes it red. Pure items carry no fixture; the `app`
fixture items build a real headless App; item 4 uses `test_document_graph.py`'s standalone
`gl_ctx`.

1. **`entry_points`** (`tests/test_pass_graph.py`, pure): over hand-built wirings, the
   bloom-shaped and RC-shaped wirings answer one root each (`scene`, `paint`); a wiring whose
   ONLY edge is a self-read (`{"acc": {"u_prev": "acc"}}`) answers `["acc"]`; a two-input
   compositor answers both its leaves; a pass with no wiring entry is a root. Falsifier:
   counting a self-read as an input makes the `acc` case answer `[]`. The self-read case must
   be its own wiring: in both real shapes every self-reader also reads a sibling, so the bug is
   invisible there.
2. **`group_slug`** (`tests/test_pass_graph.py`, pure): the four inputs of D2. Falsifier: a
   document named `2D SDF` yields `2d`, which fails `_PASS_NAME_RE` and makes every import from
   it reject over a name the user never typed.
3. **`plan_import`** (`tests/test_pass_import.py`, new, pure): (a) every wired sampler of a
   copied pass is explicit under host names, including the self-read written as
   `PassSource("bloom_trail")`; (b) a substituted entry point is absent from `renames` and its
   readers point at the host pass; (c) a copied name colliding with a host pass rejects and
   names it; (d) substituting the only pass rejects with "nothing to import"; (e) an empty
   group copies under bare names; (f) a handover pair rewires that host sampler to the bundle
   output and an unchecked one is untouched; (g) `becomes_output` is true only when a fed pass
   is the host output; (h) a source pass that contributes no edges still plans. Falsifier for
   (a): drop the materialization and every inter-pass edge of the bundle is gone, only feedback
   survives.
4. **A rendered import reads its bundle** (`tests/test_document_graph.py`, `gl_ctx` + the
   `_document` helper): a host whose `main` renders a known constant, the bloom fixture
   imported with `scene` fed by `main`, render, and the output canvas's red is the value the
   bundle produces from that constant. Falsifier: drop the materialization (3a) and the output
   reads the unbloomed or black value. ("Above the starter's black" would not do: the starter
   is UV Mango, a gradient.)
5. **`import_passes` end to end** (`tests/test_pass_verbs.py`, the `app` fixture): the
   five-pass bloom fixture copied into `tmp_path` and loaded with `load_document_from_dir`
   (as `test_lazy_compile.py` and `test_default_wiring.py` do; it is not in
   `app.ui_documents`), imported into the starter under group `bloom` with `scene` fed by the
   starter's `main`. Reload from disk with `_reload`; assert the four copied files beside the
   host's own under `passes/`, the four entries with `group == "bloom"`, the rows
   `bloom_bright.u_scene == {"pass": "main"}`, `bloom_composite.u_blur == {"pass":
   "bloom_blur"}` and `bloom_trail.u_prev == {"pass": "bloom_trail"}`, the output
   `bloom_composite` (the starter's only pass was the output), and that the SOURCE document's
   passes still hold their own textures after `release()` of the host. Falsifier: share the
   `Image` object instead of copying it and the source's sampler reads a released texture.
   A second scenario gives the host a second pass `grade` reading `main`, never rendered, and
   makes `grade` the output; imports with `main` fed and `grade.u_main` handed over: after
   reload `grade.u_main == {"pass": "bloom_composite"}` and the output is still `grade`, since
   the fed pass was not the output. Falsifiers: (i) compute `host_wiring`
   without compiling the host and the reader is never offered; (ii) leave `grade` program-less
   at save time and `UIDocument.save` carries its disk rows forward, dropping the handover.
6. **The source stays untouched** (`tests/test_pass_verbs.py`): import from a shipped example
   and assert its directory is byte-identical afterwards (content of `graph.json`,
   `document.json` and every `passes/*.glsl`). Falsifier: call `save_ui_document` on the
   source `UIDocument` and the shipped example is rewritten in the working tree.
7. **A merged row survives the save** (`tests/test_pass_verbs.py`): give the source a uniform
   with a non-default input type in its `ui_state.ui_uniforms`, import, reload, and assert the
   row is present on the host with that input type. Falsifier: skip `compile()` on the copied
   passes and the prune drops every merged row in the same save that wrote it.
8. **A source pass that does not compile** (`tests/test_pass_verbs.py`): a source dir one of
   whose pass files does not compile still imports its other passes, and `ImportResult.notes`
   names the broken pass. Falsifier: swallow the compile failure and the user gets a bundle
   with a silently black member.
9. **The group survives every existing verb** (`tests/test_graph_persistence.py`): rename
   keeps it, delete drops it, a reload reads it, a `graph.json` without the key loads as `""`.
   Falsifier: `_graph_renamed` rebuilt from a fresh `PassEntry()`.
10. **`group_runs`** (`tests/test_pass_strip_layout.py`, pure): consecutive members form a run,
    an outside pass between members splits it into two, an ungrouped pass is its own run of
    one. Falsifier: grouping by name instead of by adjacency merges the split.
11. **Group tints** (`tests/test_theme.py`, pure): (a) `group_tint` is pinned by VALUE at
    three names' crc32 indices written into the test as literals (`zlib.crc32(b"bloom") % 5`
    is the same every run; `hash("bloom") % 5` differs per process, so the pin is red on
    essentially every run under `hash()`); (b) `set(GROUP_TINTS)` is disjoint from the accent
    primaries and every `STATE_*` hue, and has no duplicate. Falsifier for (b): put `yellow_b`
    in the tuple.
12. **The copilot table and `set_pass(group=)`** (`tests/test_copilot_pass_tools.py`): a set
    with `group="fx"` shows `group fx` in the echoed table; `group=""` clears it; an invalid
    group name is an error, not a silent no-op.
13. **The command is registered** — `tests/test_command_registry_coverage.py` fails on a
    `CommandId` with no `COMMAND_SPECS` row and, separately, on one with no
    `app.command_callbacks` handler.
14. **The modal is wired** — `tests/test_project_management.py::test_every_popup_state_has_a_draw_call`
    goes red on a `PopupState` member whose draw is not called in `ui.py` (7 called ==
    `len(PopupState) - 1` today).
15. **The draft's resets** (`tests/test_pass_draft.py`'s App-driven shape): open, select a
    source, set handovers, select a second source: `handovers` is empty and the group buffer
    is the second slug; re-pick a host pass: `handovers` is that pass's readers; Cancel and
    reopen: the initial state; `app.close_import_passes()` (the Escape funnel) resets it
    too. Falsifier: rely on the `hotkeys.py` fallthrough and the draft survives Escape.
16. **The busy guard** (`tests/test_pass_draft.py`): with `app.copilot_turn_active = True`,
    `app.open_import_passes()` leaves `popup_state` CLOSED and pushes the busy notification.
    Falsifier: delete the guard; the strip button's `begin_disabled` passes the suite either
    way.
17. **The plan is recomputed per frame** (`tests/test_pass_settings_layout.py`'s pumped-frame
    shape): open the dialog with group `2bad`, pump a frame, `draft.rejection` names the
    group; set the buffer to `ok` without changing the source, pump one frame,
    `draft.rejection` is empty. Falsifier: compute the plan on selection only.
18. **The unbordered tile does not move its contents** (`tests/test_pass_settings_layout.py`'s
    measured shape): a `preview_cell` drawn with `bordered=False` places its image at the same
    screen position as one drawn bordered. Falsifier: drop `ChildFlags_.borders` with no
    compensating pad and grouped pictures sit 1px off their neighbours.
19. **Smoke** (`scripts/smoke.py`): beside the frame-42 multi-pass strip, stamp a group onto
    two NON-adjacent members in `strip_order` (`session.set_pass_group`) and keep them for
    the rest of the loop, so the outline, its label, the split-run path and the unbordered
    member tiles all execute on the parent draw list under the real frame loop. Falsifier:
    `add_rect` inside a tile's child window; today no smoke document carries a group, so the
    path runs zero times.

Manual, the maintainer's walk: the outline and label on the real strip at 480 and 1040; the
dialog over the six examples and his own Bloom Chain; importing Radiance Cascades into a
document that paints its own `paint`.

---

## Open questions for the user

Resolved at plan-lock (2026-09-12):

1. **D6** — the maintainer asked for the insertion: the bundle's output takes over the fed
   pass's readers and its output role, shown as a per-reader picker, explicit and concise.
2. **D2** — first word.
3. **D7** — member tiles lose their own border inside the outline; the outline is the same
   weight as a tile border and the cards keep their size.

---

## Review history

**Round 1 (pre-implementation, 2026-09-12): both PARTIAL, folded in.** Reports:
`reviews/pre_correctness_design.md`, `reviews/pre_verification_blast.md`.

Accepted and applied: the Bloom Chain anchor corrected to the test fixture (both reviewers;
the first said it did not exist in the repo, which was wrong — it is
`tests/fixtures/bloom_chain/` and a document in the maintainer's project, so the prose keeps it
as the worked example and the tests reach it by copying the fixture); D7's three strip claims
(gap unchanged and the mock's 4px dropped, the outline inset by 1px, `bordered` as the child
flag with the inset compensated, `bordered=border is not None`, draw order); D5's copy roster
over the whole value domain, `Image` re-opened from its file, `source_dir` dropped for
`document_dir_of`, the hash-merge precedence, compile-before-save so the prune keeps merged
rows; D6's release path, the host compile (the same hole D3 closed for the source), the
rejection of a handover on a host pass that does not compile; D8's `blue_b` collision (five
tints); D9's positional append, the table suffix and the create-mode row; D10's busy guard,
the Escape branch in `hotkeys.py`, `rejection` on the draft; D11 rewritten as one predicate
over the existing branches; verification items 1, 3, 4, 7 (now 11), 10 (now 15) and 11 (now
19) rewritten, and items 2, 6, 7, 8, 14, 16, 17, 18 added.

Rejected: the first reviewer's E6 asked to delete the graph-view bullet as contradicting 070.
070 closed the graph view as the strip's replacement; the maintainer reopened it in this
session as an opt-in second view and chose hand-drawn rendering, which the reviewer did not
have in its anchors. The bullet is rewritten to say exactly that, and 070 gets its pointer
when that feature lands rather than now. The same reviewer's suggestion to materialize every
undecided sampler of a copied pass to black was declined: the name rule is what the maintainer
authors against, and D4 now says why.

False trails recorded by the reviewers, not to be re-checked: D3's compile loop disturbs no
source state (measured); `UIDocument.save`'s prune and asset sweep keep merged rows and copied
assets once the copies are compiled; `load_graph` rides the new field; every `PassEntry`
mutation site carries the entry; `PopupState` is enumerated in one test only;
`test_button_tiers.py` scans directory-wide; `test_persistence_completeness.py` is not tripped
by a new field; the shipped examples never receive a write from D5's sequence.
