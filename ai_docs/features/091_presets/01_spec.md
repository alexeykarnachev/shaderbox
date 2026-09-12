# 091 — Presets: import another document's passes as a group

Status: **locked 2026-09-12** (D1–D11 are fixed premises; the three open questions are resolved below). The design was settled in chat on 2026-09-11/12 and
sketched in `00_mock.html` (sections 1·C3, 2b and 3 are the chosen shapes; the rest is the
brainstorm record). This file restates those decisions as the implementation contract.

Size: **mid**. 2 pre-implementation reviewers (correctness & design; verification &
blast-radius), 3 post (code correctness; architecture & conventions; spec fidelity, since the
import verb touches the persistence funnel and the strip).

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
  the host pass. Nothing else is asked.
- **The group is a label on the pass**, written by import and editable by hand in the pass
  settings modal. The strip draws a group's consecutive tiles inside one flush outline with the
  group's name on the border (mock 1·C3). Nothing folds.
- **Sources within reach:** the project's own documents and the shipped examples, in one
  window with two tabs (mock 2b).

---

## Out of scope

- **A cross-project presets folder** (`app_data_dir()/presets/`, the shader-lib posture). The
  import verb takes a loaded source `Document` plus its directory, and the dialog lists
  sources from a list of `(label, dict[str, UIDocument])` tabs, so a third tab backed by that
  folder is additive. Trigger: the maintainer wants a bundle in a second project.
- **Export a group as a document.** The inverse of import (a group's members written as a
  standalone document dir, boundary samplers as explicit black rows). Trigger: the presets
  folder above, which it would feed.
- **Folding a group** into one tile. Rejected in chat: a folded group must be convex in the DAG,
  which turned into a rule set the UI could not carry. Trigger: none; 092's graph view draws a
  group as a region and needs no contraction.
- **The graph view** (092). Trigger: 091 has landed.
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
- **Importing feedback seeds.** Copied feedback passes start black. Trigger: none; the export
  path resets histories anyway (065 D10).

---

## Design decisions

**D1 — `PassEntry.group: str = ""` is the group.** One field on the existing entry, so
`graph.json` gains one key per grouped pass and nothing else. A group exists while at least one
pass carries its name and is gone when the last member leaves; there is no group table, no
member list, no group object (a second name-keyed structure would have to agree with the
first — `conventions.md`, the lockstep-dicts bullet). The name obeys `_PASS_NAME_RE` (it is
drawn on a border and prefixes filenames). The field rides `load_graph`'s per-entry salvage
unchanged, since the salvage enumerates the model's fields.

**D2 — the group name is the prefix.** Importing under group `bloom` names the copied passes
`bloom_<source name>`. An empty group name imports the passes under their own names with no
group. The dialog prefills the name from the source document's display name: its first word,
lowercased, characters outside `[A-Za-z0-9_]` replaced by `_`, and `g_` prepended when the
result does not start with a letter or underscore (`Bloom Chain` → `bloom`, `Radiance Cascades`
→ `radiance`, `2D SDF` → `g_2d`). The first word and not the whole name because a tile
is 168px and the 14px face fits about 21 characters: `bloom_composite` fits,
`radiance_cascades_composite` clips. A pure function `group_slug(name: str) -> str` in
`pass_graph.py`, pinned by a test over those three inputs.

**D3 — entry points are the roots of the source's effective wiring.** `entry_points(wiring)` in
`pass_graph.py`: the passes whose wiring has no source other than themselves (a self-read is
feedback, not an input). Pure, GL-free, tested on the shipped examples' shapes: Bloom Chain and
Radiance Cascades have exactly one (`scene`, `paint`); a single-pass document's one pass is
both entry point and output. The wiring is `Document.effective_wiring()` of the source **after
every source pass has been compiled** (`Pass.compile()` on each with no program and no
errors; `_bring_chain_online` alone stops at the output's chain). A source pass whose compile
fails contributes its explicit rows only, and the import notification names it.

**D4 — a substituted entry point is not copied; its readers are re-pointed.** For each entry
point the dialog holds a choice: `keep` or a host pass name. The import plan is a pure function
in a new leaf module `pass_import.py`:

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

`host_wiring` is the host's `effective_wiring()` and `host_passes` is its key set; `handovers`
is the set of `(host pass, sampler)` pairs the dialog's checkboxes left on (D6).

Returning `str` is the rejection (the message the dialog shows): a copied name already in
`host_passes`, a group name failing `_PASS_NAME_RE`, a substitution naming a pass that is not
an entry point or a host pass that does not exist, a handover naming a pair that does not read a fed pass, or a plan that copies nothing (a
single-pass source whose one pass is substituted). Every sampler of a copied pass that the source wiring
filled becomes an explicit `PassSource` in `sources`: to the renamed pass, or to the host pass
when the source was substituted; a self-read points at the renamed self. Name-rule wiring
does not survive the prefix (`u_blur` no longer names `bloom_blur`), so materializing is not
optional. Samplers the wiring did not fill keep their value (a bound texture is copied, a
`NoSource` stays a decision, an `AutoSource` stays undecided and may catch a host pass by name,
which is the rule working as designed).

**D5 — `ProjectSession.import_passes` executes the plan and saves, like the six pass verbs.**

```python
def import_passes(
    self, document_id: str, source: Document, source_dir: Path,
    group: str, substitutions: Mapping[str, str],
) -> str   # "" on success, else the rejection
```

In order: plan (D4; reject before touching anything); for each copied pass write its source
text to `passes/<host name>.frag.glsl` and build a `Pass` from that path with the source
entry's target (as `add_pass` does); copy the entry with `group` set; copy uniform values —
scalars and tuples by value, a `moderngl.Buffer` via `gl.buffer(buf.read())`, an `Image` via
`Image(value.texture)`, a `Video` re-opened from its file, a raw `moderngl.Texture` via
`gl.texture(size, components, data=tex.read(), dtype=...)`; then overwrite the wired samplers
with the plan's `PassSource` rows; set the document output to the plan's `output` (D6); save
through `save_ui_document`, whose funnel writes the pass files' rebinding, the graph, the
rows and the assets under `media/<host name>/`. Copying value objects rather than sharing them
is what keeps `Pass.release` (which releases every value it holds) from freeing a texture the
source still uses. `ui_uniforms` rows are merged from the source's `ui_state` for hashes the
host does not have, so a sampler's `texture` input type and a drag's range come along.

**D6 — the bundle's output takes over the role of the pass that feeds it.** Feeding an entry
point with host pass `scene` is an INSERTION: every host sampler that read `scene` now reads the
bundle's output, and if `scene` was the document's output, the bundle's output becomes the
output. Both shapes fall out of one rule:

```
scene → grade → final        grade.u_src read scene, now reads bloom_composite; final stays output
scene (output)               scene had no readers; bloom_composite becomes the output
```

The takeover is explicit and per reader, because one reader can legitimately want the raw
pass (a `mask` cutting a shape from the unbloomed scene). Under an entry-point row whose combo
names a host pass, the dialog lists that pass's host readers as `(pass, sampler)` checkboxes,
all on by default: `then read bloom_composite instead of scene: ☑ grade.u_src ☐ mask.u_src`.
With no readers and the fed pass being the output the line reads `bloom_composite becomes the
output`. Nothing is shown for `keep`. Two fed entry points each get their own line; both hand
over to the same bundle output. The plan carries the result as `handovers` and
`becomes_output` (D4), and `import_passes` writes the handover rows as explicit `PassSource`s
on the HOST passes and moves the output when the flag is set. The default readers are
computed from `host_wiring` at the moment a host pass is picked, so a checkbox set survives
until the source or the combo changes.

**D7 — the strip draws a group as one flush outline (mock 1·C3).** In `pass_list.draw`, tiles
keep their order (`strip_order`) and their gaps; consecutive tiles of one group on one row form
a run. Per run the parent draw list gets a rounded rect 2px outside the run's tiles in the
group's tint, and the group name on the top border at the run's left, on a `BG_SURFACE` fill
so it reads over the line. A run that wraps is two runs with the label on each. Member tiles
draw with the group tint as `bg_color` at low alpha and no border of their own (`preview_cell`
gains a `bordered: bool = True` parameter; the accent output border and the red error border
still win). `tiles_per_row` is untouched, since gaps do not change. Contiguity is not a rule:
when a rewire puts an outside pass between two members, the group simply draws as two runs.

**D8 — group tints are theme tokens, picked by a stable hash.** `COLOR.GROUP_TINTS`, six hues
from the palette that no accent preset and no state color uses (`blue_b`, `purple_b`,
`green_n`, `aqua_n`, `orange_n`, `blue_n`); `group_tint(name) -> color` indexes by
`zlib.crc32(name.encode()) % len(...)`, never by `hash()`, which is salted per process. An
import-time assert beside the `SELECT` invariant pins that no tint equals an accent primary or a
state hue, and `tests/test_theme.py` breaks it by putting `yellow_b` in the tuple.

**D9 — the group is editable by hand in the pass settings modal and by the copilot.** The
modal gains a `group` row under `name` (mock 3): an input with a combo of the document's
existing groups; committing writes `session.set_pass_group(document_id, name, group)` (a
seventh verb, validated by `_PASS_NAME_RE` or empty, saved like the others). `set_pass` gains
`group: str | None` (None = keep, `""` = leave the group) and `_pass_table` prints
`group <name>` on grouped rows. The tile context menu gains `Leave group` on a grouped tile.

**D10 — the import dialog is one modal in the `PopupState` mutex (mock 2b).**
`PopupState.IMPORT_PASSES`, drawn by a new `popups/import_passes.py`, opened from an `import…`
button beside `add pass` and from a palette command `IMPORT_PASSES`. Its transient state is an
`ImportDraft` dataclass on `App` (`ui_models.py`, beside `PassDraft`): the active tab, the
selected source id, the group buffer, `substitutions: dict[str, str]` keyed by entry point, and `handovers: set[tuple[str, str]]`
(D6). The body: a tab row `This project | Examples`; a card grid of that tab's documents
through `draw_document_preview_button` (the Examples modal's own grid); the selected source's
description; the `group` field; one row per entry point, `<name> ← [combo]` with `theirs (copy
it)` first and the host's passes after, the entry point's readers listed beside it in the
12px face, and under a row fed by a host pass the D6 checkbox line; a note naming the
bundle's output and whether it becomes the document's; `Import N passes` (disabled with the rejection
in red while `plan_import` rejects) and `Cancel`. Changing the source resets the group buffer to
D2's slug, the substitutions to `keep` and the handovers to empty; picking a host pass in a
combo resets that entry point's handovers to all of its readers. The current document is excluded from the project
tab. The plan is recomputed each frame the dialog draws, which is how the button's enabled
state and the message stay honest; the source's passes are compiled on selection (D3), one
frame's cost.

**D11 — while the dialog is open, the planned render set is the open tab's documents plus the
current document.** `ui.py`'s planned-set branch treats `IMPORT_PASSES` as `EXAMPLES` does
(090 D10) when the Examples tab is active, and as the ordinary set otherwise, so a card's
thumbnail is live and no document nothing shows takes an interval. The first-render sweep
(066 D2) admits one example per frame exactly as the Examples modal does.

---

## Data model changes

`graph.json`, per pass entry:

```json
"bloom_blur": { "target": {...}, "iterations": 1, "group": "bloom" }
```

`GRAPH_JSON_VERSION` stays 2: an absent key reads as `""` and the field has a default, so
every existing file loads unchanged. No other file changes shape. `projects/dev/` needs no
hand edit.

---

## Files touched

- `shaderbox/pass_graph.py` — `PassEntry.group`, `PassGraph.with_group(name, group)`,
  `entry_points(wiring)`, `group_slug(name)`, `group_runs(order, groups) -> list[list[str]]`
  (the strip's runs, pure).
- `shaderbox/pass_import.py` — new leaf: `ImportPlan`, `plan_import`. Imports `pass_graph`
  only.
- `shaderbox/project_session.py` — `import_passes`, `set_pass_group`; `_graph_renamed` and
  `_graph_without` already carry the entry, so a rename or delete keeps or drops the group
  with it.
- `shaderbox/ui_models.py` — `ImportDraft`.
- `shaderbox/app.py` — `PopupState.IMPORT_PASSES`, `import_draft`, `open_import_passes`,
  `close_import_passes`, `import_passes_from_draft`, the `IMPORT_PASSES` command binding.
- `shaderbox/commands.py` — `CommandId.IMPORT_PASSES` + palette entry.
- `shaderbox/popups/import_passes.py` — new modal (D10).
- `shaderbox/popups/pass_settings.py` — the `group` row (D9).
- `shaderbox/widgets/pass_list.py` — the `import…` button, the group outline (D7), `Leave
  group` in the context menu.
- `shaderbox/ui_primitives.py` — `preview_cell(bordered=...)`.
- `shaderbox/ui.py` — the modal's draw call in the popup chain; the planned set (D11).
- `shaderbox/theme.py` — `GROUP_TINTS`, `group_tint`, the invariant (D8).
- `shaderbox/copilot/backend.py`, `capabilities.py`, `tools/passes.py` — `group` in the table
  and in `set_pass` (D9).
- `tests/` — see Verification.
- `ai_docs/conventions.md ## Design decisions` — three bullets: the group is a label on the
  entry (D1, with the fold rejection and why); import is by copy with entry-point
  substitution (D3/D4); group tints are stable-hash theme tokens (D8).
- `ai_docs/roadmap.md` — the 091 row and the banner.
- `ai_docs/dev_flow.md ### Module map` — `pass_import.py`, `popups/import_passes.py`, the
  strip's group outline.

---

## Verification

Each item names the falsifier that makes it red.

1. **`entry_points`** (`tests/test_pass_graph.py`): Bloom-shaped and RC-shaped wirings answer
   one root each; a self-reading root is still a root; a two-input compositor answers two.
   Falsifier: counting a self-read as an input makes the feedback root disappear.
2. **`plan_import`** (`tests/test_pass_import.py`, new, GL-free): (a) every wired sampler of a
   copied pass is explicit under host names, including `u_prev`; (b) a substituted entry point
   is absent from `renames` and its readers point at the host pass; (c) a copied name colliding
   with a host pass rejects and names it; (d) substituting the only pass rejects with "nothing
   to import"; (e) an empty group copies under bare names; (f) a handover pair rewires that host sampler
   to the bundle output and an unchecked one is untouched; (g) `becomes_output` is true only
   when a fed pass is the host output. Falsifier for (a): drop the
   materialization and `bloom_composite.u_blur` reads black in a rendered host.
3. **`import_passes` end to end** (`tests/test_pass_verbs.py`): the Bloom Chain example
   imported into the starter document with `scene` substituted by `main`; reload from disk;
   assert the four files under `passes/`, the four entries with `group == "bloom"`, the rows
   `bloom_bright.u_src == {"pass": "main"}` and `bloom_composite.u_glow == {"pass":
   "bloom_blur"}`, the output `bloom_composite`, and that the source document's passes still
   hold their own textures after `release()` of the host (D5's copy). Falsifier: share the
   `Image` object instead of copying it and the source's sampler reads a released texture.
4. **A rendered import is not black** (`tests/test_document_graph.py`): after item 3, render
   the host and assert the output canvas's mean is above the starter's black. Falsifier: the
   materialization bug of item 2(a).
5. **The group survives every existing verb**: rename keeps it, delete drops it, a reload
   reads it, a `graph.json` without the key loads as `""`
   (`tests/test_graph_persistence.py`). Falsifier: `_graph_renamed` rebuilt from a fresh
   `PassEntry()`.
6. **`group_runs`** (`tests/test_pass_strip_layout.py`): consecutive members form a run,
   an outside pass between members splits it into two, an ungrouped pass is its own run of
   one. Falsifier: grouping by name instead of by adjacency merges the split.
7. **The theme invariant** (`tests/test_theme.py`): `group_tint` is stable across two
   processes (spawn a subprocess and compare), and putting `yellow_b` in `GROUP_TINTS` makes
   import fail. Falsifier: `hash()`.
8. **The copilot table and `set_pass(group=)`** (`tests/test_copilot_pass_tools.py`): a set
   with `group="fx"` shows `group fx` in the echoed table; `group=""` clears it; an invalid
   group name is an error, not a silent no-op.
9. **The command is registered** — `tests/test_command_registry_coverage.py` already fails on
   an unrouted `CommandId`.
10. **The modal's state resets** (`tests/test_pass_draft.py`'s shape): opening, selecting a
    source, cancelling, reopening shows the prefilled slug and `keep` substitutions again.
11. **Smoke**: the strip with a grouped document draws without an assert (the frame draws on
    the parent's draw list, outside the tiles' child windows).

Manual, the maintainer's walk: the outline and label on the real strip at 480 and 1040; the
dialog over the six examples; importing Radiance Cascades into a document that paints its own
`paint`.

---

## Open questions for the user

Resolved at plan-lock (2026-09-12):

1. **D6** — the maintainer asked for the insertion: the bundle's output takes over the fed
   pass's readers and its output role, shown as a per-reader picker, explicit and concise.
2. **D2** — first word.
3. **D7** — member tiles lose their own border inside the outline; the outline is the same
   weight as a tile border and the cards keep their size.
