# 084 — Project management

A project has no verbs. `File → Open project...` is the only one, and everything else a user needs —
make a new project, fork the current one, move between two of them, see which one is open — has no
surface at all. This feature gives the project the verb set the document already has.

Source: the maintainer, verbatim — *"it seems like we still don't have any reliable and convenient
way to create a project in shaderbox? how can I create a new project? Or how can I save-as the
current project?"*

---

## The audit

What exists today, established by reading the code rather than remembered:

- **One verb.** `CommandId.OPEN_PROJECT` (Ctrl+O) → `App.open_project` → `pfd.select_folder` →
  `App._init(picked)`. It is the whole of project management.
- **Creation works by accident.** `ProjectPaths.for_root` does `mkdir(parents=True, exist_ok=True)`
  on the root and all five subdirs, so picking a fresh empty folder in that dialog DOES build a
  valid project. Nothing says so, and the starter seed is deliberately skipped for `open_project`
  (it would "pollute a folder the user picked expecting it empty"), so the reward for finding the
  trick is an app with no document and a blank editor.
- **`_init` is already a complete, re-entrant switch.** `release()` tears down copilot, exporters,
  every editor session, the editor panel and every document's GL resources; `session.load()` rebuilds
  paths, lib index, documents, `app_state` and scripts. **The expensive half of a switcher already
  exists and runs on every Ctrl+O.**
- **The switch loses work.** `release()` persists only the copilot conversation. `App.save` — the
  funnel that flushes the editor buffer, writes the current document and writes `app_state.json` —
  is reached from exactly two places: the quit path in `ui.py`, and vim's `:w` in `hotkeys.py`.
  `open_project` calls neither. Switching today silently discards the unsaved buffer and the
  app_state.
- **A project has no name.** `UIAppState` has no name field and the window title is the constant
  `"ShaderBox"`, so a project's identity is its directory name and it is displayed nowhere.
- **Nothing copies a directory.** `shutil.copytree` appears nowhere in `shaderbox/`. A fork is new
  code.
- **`<app_data>/projects/` already exists** (`App.default_projects_root_dir`), holding `default/`.
- **The pointer is a bare path** at `<app_data>/project_dir`, and it can point anywhere. It was
  found pointing at `/tmp/sb-b-t2d07nr8` — a real project, with the maintainer's live two-pass work
  in it, living in a directory the OS clears on reboot. Nothing in the app said so. Moved by hand to
  `<app_data>/projects/radiance_cascade` before this feature was drafted.
- **A dead pointer silently recreates an empty project at the dead path.** `is_first_launch` tests
  whether the pointer FILE exists, not whether the directory it names does; so a pointer at a
  deleted directory reads as "not first launch", `ProjectPaths.for_root` recreates the whole
  skeleton with `exist_ok=True`, `load_documents_from_dir` returns `{}`, and because `first_run` is
  False the starter seed does NOT fire. The app comes up empty, with no error and no gallery, and
  rewrites the pointer to the path it just recreated. This is exactly what a `/tmp` project turns
  into after a reboot.
- **A safe directory copy already has a proven shape in-repo.** `copilot/revert.py::_swap_in_snapshot`
  copies to a staging sibling first and only then removes the destination, so a torn copy can never
  destroy the target.

---

## Goal

Four verbs and one indicator, built as one coherent surface rather than four menu items:

- **New project** — name it, it is created under the projects root, seeded with a starter document,
  and opened. Lands somewhere usable, never blank.
- **Duplicate project** (the "save-as") — fork the current project to a new name, switch to the
  fork. The original is left exactly as it was.
- **Switch project** — move between projects without navigating a folder picker.
- **Open a project from anywhere** — the folder picker, demoted from a top-level verb to one
  control inside the modal, for a project outside the projects root.
- **The open project is visible** — its name, always, without opening anything.

And the correctness fix the above makes mandatory:

- **A project switch saves the outgoing project first.** Every path that leaves a project routes
  through one funnel.

---

## Out of scope

- **Rename a project.** Falls out of D3 (the name IS the directory name), so a rename is a directory
  move with the pointer and any open handles to fix up. Trigger: the maintainer renames a project by
  hand and finds the app confused by it.
- **A project template / "new from this project" gallery.** Duplicate covers the real need.
  Trigger: the maintainer asks for more than one starting point.
- **Per-project window title.** D8 puts the name in the app's own chrome instead; a glfw title
  change is a second place to keep in sync for no gain. Trigger: the maintainer wants to tell two
  ShaderBox windows apart in a taskbar.
- **Migrating any old project layout.** No migration code, per `conventions.md`. The `/tmp` project
  that prompted this was moved by hand, once, outside the app — which is the sanctioned fix.
- **A copilot tool for project verbs.** The copilot works inside one project; switching the project
  under a running turn is a hazard, not a feature. Trigger: a dogfood run where the model genuinely
  needs a second project.

---

## Design decisions

Numbered, locked. Open questions are separate, below.

### D1 — ONE surface. The modal replaces `Open project...` outright; Ctrl+O opens the modal.

Every project verb lives in one modal (`Projects`, Ctrl+O). `CommandId.OPEN_PROJECT` and the
`Open project...` menu item are **removed**, not kept alongside it — the modal is a superset of what
they did, and two doors to the same room is the ad-hoc shape this feature exists to avoid.

The folder picker does not disappear; it stops being a top-level verb and becomes one control
*inside* the modal (`Open other...`), for the case the switcher structurally cannot cover: a project
outside the projects root. So the modal carries every verb — select a row, New, Duplicate, Delete,
Open other — and the File menu carries one item, `Projects...`.

**`Open other...` is the only way to reach a project the list cannot see**, and the list sees
`<app_data>/projects/*/` plus the open one (D4). It runs `pfd.select_folder` and hands the result to
`switch_project`, exactly as today's `open_project` does. Its known limit, accepted rather than
solved: switching AWAY from an outside project drops it off the list again, so returning means
navigating the picker once more. A recents file would fix that and is rejected in D4 for the reasons
there; the picker remembers its own last directory, and the normal path — New and Duplicate — puts
projects in the root where this never arises. Revisit if the maintainer works outside the root often
enough to re-navigate the picker repeatedly.

**The chord stays Ctrl+O**, transferred from the retired command. It is the muscle-memory key for
"get me to a project", the modal is now what that means, and Alt+O would strand the reflex on a
removed command. `tests/test_editor_ffi.py::test_ctrl_o_reaches_the_app_while_focused` pins Ctrl+O
as belonging to no editor keymap (the host must not swallow it) — it names `OPEN_PROJECT` only in a
comment, so the chord transfers without touching that test's assertion.

This is the whole of the removal: one `CommandId` member, one `COMMAND_SPECS` entry, one callback
row, one menu item, and `App.open_project` loses its `pfd` body to the modal's control. Nothing else
in the app references either symbol.

### D2 — The modal is a LIST of rows: name, document count, path.

A grid was drafted first (`preview_cell` tiles, composed like `examples.py`) and rejected on the
rendered sketch: the examples grid earns its thumbnails because each example LOOKS like something,
while a project's picture is either blank or a borrowed render of one document inside it, which
misrepresents a project holding four. It spends a 150x150 square per project to show two words.

So: one row per project — name, document count, and **path**. The path is the column that earns the
list. D4 unions the projects root with the open project wherever it lives, and in a grid an outside
project is indistinguishable from a rooted one; in a row the difference is legible at a glance. That
is not hypothetical — it is the exact confusion that left a real project in `/tmp` for a day.

The open project's row carries an `open` marker in the accent color. `preview_cell` is NOT the
primitive here; a row is a `selectable` plus columns, which is also what makes it a keyboard-nav
stop (an `invisible_button` is not).

**The row shape is not invented** — `popups/lib_picker/tree.py::_draw_function_leaf` already draws
exactly this: `imgui.selectable(f"{name}##leaf_{name}", is_selected)`, the selection held as a field
on a state object, and `same_line` columns after it. Copy that: the `##id` keys on the project's
PATH (stable, unique even for two same-named projects in different roots), never on the display
name.

**The modal's layout, top to bottom.** The list, then ONE verb row:

```
Projects                                                        [modal]
+----------------------------------------------------------------------+
| radiance_cascade   1 doc   open    ~/.local/share/shaderbox/projects/ |  <- selected
| default            1 doc           ~/.local/share/shaderbox/projects/ |
| sticker_studio     4 docs          ~/.local/share/shaderbox/projects/ |
| rc_experiment      2 docs          ~/src/scratch/                     |
|                                                                      |
| [New] [Duplicate] [Open other...] [Delete]              [Close]      |
+----------------------------------------------------------------------+
```

**One rule decides where a verb goes: a row click SELECTS, and the verb row acts on the selection.**

The first draft had Duplicate drawn per-row while New and Open-other sat in a header — two homes for
verbs with nothing distinguishing them, which the maintainer read as arbitrary because it was. Every
verb is now in the one row.

This costs the design an explicit selection, which D7 previously avoided ("a tile click switches
immediately, so there is never a selected-but-not-open target"). That avoidance is what forced
Duplicate out of the row in the first place, so the trade is worth naming: **a row click selects; it
does not switch.** Switching is a double-click, or Enter on the selection. The selection is what
Duplicate and Delete need to mean anything, and it is what lets Delete exist at all.

Tiers: `New` is `primary_button` (this modal exists because creating was impossible), `Duplicate`
and `Open other...` are `standard_button`, `Delete` is `danger_button`, `Close` is
`standard_button` alone on the right. Five labelled buttons in a row is at the limit of what § 1's
tier rules tolerate, which is why `Delete` disables rather than adding a sixth confirm control.

`New` and `Duplicate` each swap the verb row for their inline name input (D6), so only one input is
ever live — the lib picker's mutual-exclusion rule, one reset method both openers call first. That
same reset covers ALL of the modal's transient state, and `open_projects()` calls it: the row
selection, the armed-delete target, and both inline inputs. `App.open_settings` is the in-repo shape.

**The selection lives on `App` as transient state**, never in `UIAppState`. A persisted selected-path
would be a second dead-pointer class, and D12's three-layer rule needs it reachable from a free
`draw(app)` function anyway. Same for the armed-delete target and the pending switch.

**`list_projects()` returns a small frozen dataclass per project** — name, path, document count,
is_open — computed when the modal OPENS, not per frame. Per-frame it would glob every candidate's
`documents/*/` on every draw. The document count comes from counting those dirs, never from parsing
any JSON (which would also drag `project_session.py` into the persistence roster's `json.load` scan
for no reason).

### D3 — A project's name IS its directory name.

No name field is added to `UIAppState`. Reasons, in order: a document's identity works this way
already; a name stored inside the project can disagree with the folder that holds it, and then the
switcher shows one thing while the path says another; and a stored name buys only rename-without-
move, which is out of scope. The New-project dialog therefore validates its input as a directory
name (D6).

Revisit if the maintainer wants two projects with the same display name in different roots.

### D4 — The switcher lists the projects root, plus the open project wherever it is.

The list is `<app_data>/projects/*/` filtered to directories that look like a project (a
`documents/` dir — the same "is it loadable" posture `sync_documents_from_disk` takes per document),
UNION the currently-open project even when it lives elsewhere. Sorted by name; the open one drawn
selected.

This is deliberately NOT a recents file. A recents list is a second on-disk store to keep fail-soft,
prune, and de-stale (a recent that no longer exists), and it earns none of that here: the directory
listing IS the list, it cannot go stale, and a project the user made through this feature is always
in it. The union term is what keeps today's `/tmp` project — and any hand-picked folder — reachable
rather than vanishing from its own switcher.

Revisit if the maintainer keeps projects in several roots and wants them all listed.

### D5 — Every project switch goes through ONE funnel that saves first.

`App.switch_project(path)`: `save()` → `_init(path)`. The switcher, New, Duplicate and Open-other
all call it; none calls `_init` directly.

**"Structurally impossible" is a claim about call sites, so a test asserts them.** `_init` has
exactly two callers today (`__init__` and `open_project`) and must have exactly two after
(`__init__` and `switch_project`). An AST walk over `app.py` pins that — the shape
`tests/test_button_tiers.py::_raw_button_calls` already uses for the same job. Without it, D5 is a
convention a fourth caller silently breaks; with it, the fourth caller turns a test red and names
itself. This closes the live data-loss bug (the audit's
fourth bullet) and — more importantly — makes it structurally impossible for the three new verbs to
reintroduce it, since none of them can reach `_init` without passing the save.

`save()` is already the right funnel: it flushes the current editor buffer, writes the current
document, mirrors the layout prefs and writes `app_state.json` + integrations. It carries its own
copilot busy-gate for the document half, and the `_copilot_busy_blocked` guard `open_project` has
today moves onto `switch_project` — one gate covering every verb instead of one verb.

**The switch is DEFERRED to the top of the next frame, never run inside the modal's draw.** This is
the finding that most changes the shape, and it is specific to moving the verb into a popup. Today
`open_project` is called from `_draw_menu_bar`, the FIRST thing drawn — nothing has submitted a
texture yet, so `release()` freeing every GL object is safe. A popup body draws near the END of the
frame (`ui.py` draws the popups after the editor panel, the canvas backdrop and the document image
have each pushed an `add_image` carrying a raw `glo` into the draw list), and `imgui.render()` +
`imgui_renderer.render()` run after that. Releasing textures from inside the popup would leave the
draw list holding freed GL names — the released-texture binding error the smoke exists to catch,
on a path the smoke never drives.

So the modal never switches. It sets `app.pending_project_switch: Path | None`, closes itself, and
returns; `_tick_frame_state` consumes it BEFORE any drawing, which is where `_init`'s teardown is
safe. `render_defer.py` is the existing precedent for "do the expensive thing outside the draw".
The same deferral covers `Open other...`: `pfd_block` spins the main thread until the dialog
returns, and doing that mid-popup freezes a half-drawn frame.

**The busy gate sits at the TOP of `switch_project`, before `save()`, and this is load-bearing rather
than stylistic.** `App.save` gates only its DOCUMENT half on `_copilot_busy_blocked` and writes
`app_state.json` + integrations regardless — deliberately, because the quit path calls `save()` even
mid-turn and user-owned settings must persist. Calling `save()` first from an ungated
`switch_project` would therefore half-save during a copilot turn: app_state written, the current
document silently skipped, and then `_init` tears the project down anyway. Refusing the whole switch
up front is the only shape that cannot half-save.

`release()` (called by `_init`) additionally saves the OUTGOING copilot conversation before the
worker is torn down, and it reads `self.paths` — which `session.load` only rebinds afterwards — so
the existing order (release, then load) is what makes that write land in the right project. Nothing
in this feature may reorder those two.

**`save()` alone does NOT close the whole data-loss bug, and the claim is narrowed rather than
dropped.** `flush_current_editor` flushes the ACTIVE tab only, while `release()` closes every
session in the path-keyed `editor_sessions`. A user with three shader tabs open and edits in two
loses the inactive one — today, and after this feature too unless the funnel flushes all of them.
It does: `switch_project` flushes EVERY dirty session, not just the current, which is one loop and
removes the asterisk from the goal. It also calls `save_imgui_ini()`, the one line that otherwise
distinguishes this funnel from the quit tail (`save` → `save_imgui_ini` → `release`).

So the funnel is, in order: refuse if a copilot turn is in flight → flush every dirty editor
session → `save()` → `save_imgui_ini()` → `_init(path)`.

**A failed `_init` is not caught, and that is a decision rather than an oversight.** `_init` can
raise on a corrupt `app_state.json` or a permission error, AFTER `release()` has torn everything
down — leaving an App with no documents, no editor panel, no exporters. Retrying the previous
project would mean a second `_init` from an already-half-released state, which is a worse place to
debug from than a clean crash with a log line naming the project. Today's `open_project` has the
same exposure; this feature multiplies the paths but not the failure mode. Revisit if the maintainer
actually hits it.

### D6 — New project: a name, validated, under the projects root.

The New flow is an inline input inside the modal (the lib picker's inline-input pattern —
`is_item_deactivated_after_edit` commits, Esc cancels, an `x` cancel button on the right), not a
second modal and not a save-file dialog. The typed text is a directory name under
`default_projects_root_dir`.

Three more rules from the same pattern, named because omitting any one is a visible bug:

- **Focus is a ONE-SHOT** `needs_focus` flag consumed by `set_keyboard_focus_here(0)` on the input's
  first draw. Grabbing focus every frame resets the caret blink and re-asserts the nav cursor.
- **The outer Enter and Esc are suppressed while the input has focus**, gated on `is_item_focused()`
  read immediately after the `input_text`. This is not academic here: D2 binds Enter on the selection
  to SWITCH PROJECT, so without the gate, typing a name and pressing Enter would both create the
  project and switch to the unrelated selected row.
- **The `x` click is itself what deactivates the input**, so the deactivate is captured into a local
  and applied only if the cancel branch did not run — otherwise cancelling also commits.

Validation, all three refusing with the reason in the same slot rather than a toast: empty, a name
that already exists in the root, and a name that is not a safe single path segment (a separator, a
`.`/`..`, a character the OS rejects). One shared validator, used by New and Duplicate both, so the
two cannot drift.

On commit: create the dir via `ProjectPaths.for_root`, switch to it (D5), and **seed the starter
document**. The seed is the difference between "a new project" and "an empty folder": `first_run`
already does exactly this via `seed_starter_document`, and the reason that call is gated to first
run — don't pollute a folder the user picked — does not apply to a folder the app just created at a
name the user typed for this purpose.

### D7 — Duplicate: save, then copy the directory on disk, then switch.

`duplicate_project(source, new_name)`: `save()` the live state first, then copy the source project
dir to the new name (via D11's staging-then-rename), then switch to the copy. The source is the
SELECTED project (D2's rule); saving first matters only when the selection is the open one, and
costs nothing when it is not.

Save-before-copy is the discipline the copilot's `duplicate_document` already states in one line —
*"persist live state so the load is a full copy"* — and without it the fork silently loses whatever
was only in memory.

**The copy is a disk copy, never a live-object copy.** `dev_flow.md` (`### make test`, the
"what is NOT the fix here" note) records the measurement: documents hold live moderngl handles, so a
shallow share lets one App's `release()` free another's textures, and a deepcopy raises
`cannot pickle 'mgl.Context'`. That note is about caching example documents across App instances
rather than about forking, so it is evidence for the constraint, not a ruling on this feature — but
the constraint is the same one, and a fork is therefore `copytree` + `_init`, reloading everything
from files.

**What travels: everything.** The whole project dir, `copilot/` conversation and `renders/`
included. Maintainer's call, and it is the honest default — the user asked for a copy of the folder.

**The policy is ONE named function, so changing it later is a one-place edit.** What to copy is
decided by a single module-level predicate in `project_session.py`:

```python
def _copy_into_fork(entry: Path) -> bool:
    # Whether one entry of a project dir travels into a fork. Everything, for now.
    return True


def _fork_ignore(src: str, names: list[str]) -> set[str]:
    # copytree's hook is (dir, names) -> names-to-skip, NOT a per-path predicate.
    return {n for n in names if not _copy_into_fork(Path(src) / n)}
```

`copytree`'s `ignore` callable is `(src_dir, names) -> ignored_names`, called once per directory —
so the predicate needs the adapter above rather than being passed straight in. It is written here
because a signature mismatch is exactly what an implementer papers over with a guess.

Today the predicate says yes to everything, INCLUDING two things worth naming rather than
discovering: `exporter_scratch/` (created by `_rewire_exporters`, not by `ProjectPaths.for_root`, so
a live project dir has six subdirs and not five) and `copilot/checkpoints/` (per-turn rollback
snapshots, which the fork inherits pointing at a turn history that now exists in two projects).
Both are defensible under "the user asked for a copy of the folder", and both are one line if they
stop being. Excluding the conversation later is the same one line.

The alternative shape — inlining the decision as a `copytree(..., ignore=shutil.ignore_patterns(...))`
argument at the call site — is what makes such a policy hard to move later: it reads as an
implementation detail of the copy rather than a decision anyone is allowed to revisit, so it grows a
second copy elsewhere the moment a second caller appears.

### D7b — Delete arms, refuses the open project, and moves to trash.

Delete is in scope (maintainer's call, reversing this spec's first draft). Three rules make it safe
enough to sit beside a switch verb:

**It arms.** One click reddens the selected row and turns the verb row into
`Delete to trash?  [Yes] [No]` — the row it acts on is the one drawn red, so repeating the name in
the caption would spend two of the four budgeted words saying what the highlight already says.

It borrows `cell_delete_confirm`'s RULE, not the function: Yes is the PRIMARY tier and No the
standard one, because the red already carries the danger and a filled-red confirm would not read.
The function itself is unusable here — it positions absolutely over a grid cell (`origin`/`avail`,
`set_cursor_screen_pos` throughout) and this confirm is a flat row, so it is
`primary_button("Yes")` + `standard_button("No")` inline. Reaching for a raw `imgui.button` instead
fails `test_button_tiers.py`.

One armed target at a time, cleared by selecting another row and by closing the modal — a stale arm
surviving a selection change is how Yes deletes the wrong project, which is the worst bug this
feature can produce.

**The OPEN project cannot be deleted.** `delete_project` REFUSES independently of the button state —
the model guard is the gate, the disabled button and its `switch away first` caption are the
cosmetic half. That order matters: a refusal living only in draw code is a guard no headless test
can reach, and the verification row would then be testing a mechanism that does not exist. Deleting the live project would mean tearing down every GL resource, choosing a
replacement and switching, all inside a confirm; that is a second feature wearing a confirm dialog's
clothes. Switch away first.

**It MOVES, never `rmtree`.** Destination `<app_data>/trash/<name>`, and `<name>_<ms>` ONLY on
collision — the exact shape `_delete_document_unguarded` uses, which tries the bare name first and
appends `int(time.time() * 1000)` only when that path exists. An earlier draft said the suffix was
unconditional, which is a different behavior wearing the word "mirroring": it would make every
trashed project read `radiance_cascade_1757178123456` instead of the readable common case. The
verification row asserts the FIRST delete lands on the bare name, so the two schemes are told apart.

`<app_data>/trash/` needs a creator; `paths.py` gains `project_trash_dir()` beside the existing
`shader_lib_trash_dir()` / `log_dir()` / `copilot_trace_dir()`, same `mkdir(parents=True,
exist_ok=True)`-on-read idiom.

Nothing else needs cleaning: the deleted project is never the open one, so no editor session, no
conversation and not the pointer can be referring to it. And nothing ever sweeps
`<app_data>/trash/` — a trashed project stays until the user removes it by hand. That is the whole
recovery story, stated rather than left implied: recovery is a `mv`, and the app never issues a
recursive delete of a directory the user named.

### D8 — The open project's name is shown in the menu bar, right-aligned.

The menu bar is drawn every frame and always visible, it already mixes dropdowns with flat items by
deliberate choice, and it is the one piece of chrome that is never covered by a modal. The name
draws right-aligned there, dim, as text — not a button (clicking it would need a meaning, and
`Projects...` already has one two items away).

### D9 — One command REPLACES one command; one new popup state.

`CommandId.OPEN_PROJECT` becomes `CommandId.OPEN_PROJECTS` (`Projects`, Ctrl+O, category FILE) —
a rename plus a new handler, not an addition beside the old one. Plus `PopupState.PROJECTS`.

Same four edit sites as any command (the `CommandId` member, the `COMMAND_SPECS` entry, the
`_build_command_callbacks` row, the menu item), each an edit rather than an insert. The cheatsheet,
the rebinder, the palette and the help content all iterate `COMMAND_SPECS` and need no per-command
edit — so the cheatsheet's FILE section reads `Projects  Ctrl+O` with no further work.

A persisted rebinding of the retired id is dropped by the existing fail-soft load: `key_bindings` is
a `dict[str, int]` holding only non-default chords, and an unknown key costs that one binding. No
migration, per `conventions.md`.

### D10 — A pointer at a vanished directory recovers to a SEEDED default, and opens the modal.

The startup resolve tests the pointer FILE's existence; it must test the DIRECTORY's too. The exact
condition, in `App.__init__`:

```python
is_first_launch = project_dir is None and not self.project_dir_file_path.exists()
```

becomes true when `project_dir is None` AND (the pointer file is absent OR the path it names is not
a directory). That is ONE condition, but it feeds THREE consumers that must all agree, and a fix
applied to some of them is this repo's most expensive bug family: `is_first_launch` decides
`first_run`, `first_run` decides the starter seed, and the resolve branch decides which directory
loads.

**`first_run` must be TRUE on this path**, so `seed_starter_document` fires and the user lands in a
usable project rather than a blank one — which is the entire point of the decision. But `_init`'s
`if first_run: self.open_examples()` would then open the EXAMPLES gallery, and the single-field
popup mutex means examples and Projects cannot both be open. Projects is the right question here
("which project?", not "which example?"), so the auto-open becomes
`if first_run and self.popup_state is PopupState.CLOSED`, and the recovery path sets
`PopupState.PROJECTS` after `_init` returns. A genuine first launch still gets the gallery, because
nothing set a popup before it.

**The pointer is rewritten to the RECOVERED project, never to the dead path.** The earlier draft
said "not rewritten", which overstates it: `persist_pointer` stays True, and `_init` writes
`str(self.project_dir)` AFTER `session.load` has rebound `self.project_dir` to the default. So the
write lands on a live project by construction. What must not happen is the current behavior —
recreating the dead path's skeleton and pointing at that.

**`<app_data>/projects/default` is itself stale on this machine** (`nodes/`, `media/`, `trash/`,
`app_state.json`, and NO `documents/` — a pre-rename layout). So D4's "looks like a project" filter
excludes it from the switcher, and the recovery would land somewhere the list cannot show. It is
hand-fixed in this wave, which is the sanctioned no-migration fix: delete the stale dir and let the
first launch re-seed it, then `git add` nothing (it lives in app data, not the repo).

### D11 — The fork copies to a staging sibling, then renames.

`copilot/revert.py::_swap_in_snapshot` already established the shape: copy to `<name>.creating`
beside the target, and only on a complete copy rename it into place. A half-copied project dir that
looks like a real one is worse than no copy — `list_projects` (D4) would list it and the switcher
would offer it.

A leftover `.creating` from a crashed copy is swept by the next duplicate, and `list_projects`
filters it out by name.

### D12 — The core verbs live in `ProjectSession`; `App` owns the switch.

`conventions.md` puts the project lifecycle in `project_session.py` (headless, no glfw/imgui
context) and has `App` forward to it. So: the pure-disk half — enumerate the projects root, validate
a name, create a project dir, copy a project dir — is `ProjectSession` code with no `App` in it, and
therefore testable headlessly. The half that tears down and rebuilds GL state (`switch_project`)
stays on `App`, because `_init`/`release` are App's.

The draw code is a new `shaderbox/popups/projects.py` — a `draw(app)` free function per the
three-layer rule, using `modal_window` and the button tiers.

---

## Blast radius of the retirement

`OPEN_PROJECT` / `open_project` / the `Open project...` menu item were grepped across
`shaderbox/`, `tests/`, `scripts/` and `dogfood/`. The retirement touches almost nothing:

- **`tests/test_editor_ffi.py::test_ctrl_o_reaches_the_app_while_focused`** names `OPEN_PROJECT` in a
  COMMENT only; its assertion is that Ctrl+O belongs to no editor keymap and the host must not
  swallow it. The chord stays Ctrl+O, so the test passes unchanged. The comment is updated in the
  same wave so it does not name a retired symbol.
- **Nothing else references either symbol** outside `app.py`, `commands.py` and `ui.py` — the four
  edit sites D9 names.
- **Nothing enumerates `PopupState` exhaustively.** `scripts/smoke.py` and the pass tests name
  individual members; adding `PROJECTS` breaks none of them. The smoke drives popups by assignment,
  so the new modal is only exercised there if the smoke is extended — which is why the Verification
  table carries its own row for it rather than assuming the smoke covers it.
- **The command surfaces are DERIVED, not listed.** `test_command_registry_coverage.py` asserts
  `set(SPEC_BY_ID) == set(CommandId)` and `set(app.command_callbacks) == set(CommandId)`, and the
  cheatsheet, palette, rebinder and help all iterate `COMMAND_SPECS`. So a RENAME needs no edit in
  any of them — but an id added without a spec or a callback fails there, which is the existing gate
  D9's verification row leans on rather than duplicating.
- **`help_content.py`** builds its shortcuts from the registry, so `Projects` appears in the Help
  panel automatically; `test_every_bound_spec_reaches_the_help_shortcuts` asserts exactly that.
- **No new persisted store**, so `tests/test_persistence_completeness.py`'s roster is untouched. This
  is a consequence of D3 and D4 rather than luck: the project list is a DIRECTORY LISTING and the
  name is the directory name, so there is no JSON to keep fail-soft. The pointer file stays a bare
  path, exactly as today. A recents file would have added a rostered store and its whole corruption
  battery — a cost D4 declines.

---

## The strings, pre-scored

`tests/test_ui_prose_budget.py` walks the package AST and fails an over-budget string, so the copy
is settled here rather than discovered at gate time. A button label is 3 words, a caption 4.

| String | Kind | Words | Where |
|---|---|---|---|
| `New` | button | 1 | verb row |
| `Duplicate` | button | 1 | verb row |
| `Open other...` | button | 2 | verb row |
| `Delete` | button | 1 | verb row |
| `Close` | button | 1 | action row |
| `Yes` / `No` | button | 1 | armed-delete confirm |
| `Delete to trash?` | caption | 3 | armed-delete confirm |
| `switch away first` | caption | 3 | under a disabled Delete |
| `name already used` | caption | 3 | inline-input refusal |
| `Enter creates` | caption | 2 | under an open inline input |
| `Projects` | menu item + modal title | 1 | File menu, modal |

Three earlier drafts were over budget and are recorded so they are not re-proposed:
`switch away to delete this one` (6), `Enter creates - Esc cancels` (5), and
`click a row to switch` (5). The Esc half is dropped rather than shortened: Esc-cancels is the
convention every other inline input in the app already follows, and the `x` button is the visible
affordance for it.

The menu-bar project name (D8) is NOT authored copy — it is the project's own directory name, a
derived value, and § 2's rule is that a derived value goes in the control rather than the label.

---

## Files touched

| File | Change |
|---|---|
| `shaderbox/project_session.py` | `list_projects`, `validate_project_name`, `create_project`, `copy_project_to` (staging-then-rename, D11), `trash_project` (D7b), `_copy_into_fork` + `_fork_ignore` — pure disk, no GL |
| `shaderbox/paths.py` | `project_trash_dir()`, beside the existing trash/log/trace helpers |
| `scripts/smoke.py` | the popup sweep derived from `PopupState`, closing the pre-existing gap for `EMOJI_PICKER` and `SHADER_LIB_PICKER` in the same wave |
| `scripts/README.md` | the Ctrl+O row: `Open a project` -> `Projects` |
| `shaderbox/app.py` | `switch_project` (the D5 funnel), `open_projects`, `new_project`, `duplicate_project`, `delete_project`, `pick_project_dir` (the demoted picker); `open_project` retired; the startup pointer resolve (D10); `PopupState.PROJECTS`; the callback row |
| `shaderbox/popups/projects.py` | NEW — the modal: the row list, the New/Duplicate inline inputs, the verb row, the armed-delete confirm |
| `shaderbox/commands.py` | `OPEN_PROJECT` → `OPEN_PROJECTS`, label `Projects`, chord kept at Ctrl+O |
| `shaderbox/ui.py` | `Projects...` menu item; the right-aligned project name (D8); `draw_projects(app)` in the popup block; the deferred-switch consume in `_tick_frame_state` |
| `tests/test_project_management.py` | NEW — see Verification |
| `ai_docs/roadmap.md` | the 084 row + the Active-context banner |
| `ai_docs/conventions.md` | D5 (the save funnel) and D7 (disk-copy-only fork) as design decisions; the `open_project` sentence in the active-project-pointer bullet renamed |

---

## Verification

Each check fails for exactly one reason, and each names the falsifier — the input that SHOULD break
it — per the dev-flow rule. All headless; the app fixture builds a real `App` against a tmp project.

| Invariant | Test | Falsifier — the break that must turn it red |
|---|---|---|
| A switch saves every dirty tab, not just the active one | open TWO shader tabs, edit both, leave the second active, `switch_project(other)`, switch back, assert BOTH edits are on disk | flush only the current session |
| `_init` has exactly two callers | AST-walk `app.py`, assert `_init(` appears only inside `__init__` and `switch_project` | add a fourth call site anywhere |
| Each verb saves at its own consumer | for New, Duplicate and Open-other separately: edit a buffer, run the verb, assert the edit reached disk | give one verb its own `_init` path |
| A new project is usable, not blank | `new_project("x")`, assert the project dir exists with the full layout AND `ui_documents` is non-empty | drop the `seed_starter_document` call |
| A duplicate carries live state | edit a uniform, assert the SOURCE's `document.json` still holds the OLD value (proving it is memory-only), duplicate, assert the fork holds the NEW one | remove the `save()` before the copy |
| A duplicate leaves the original alone | census the source's `documents/` after the duplicate returns, edit the fork, assert the census is unchanged | copy to the final name with no staging, so a half-copy loads |
| The switcher lists the root plus the open project | with the open project OUTSIDE the root, assert it appears in the list exactly once | drop the union term |
| A name is validated | assert empty, duplicate, `..`, and a separator each refuse and create no directory | accept the string and let `mkdir` decide |
| The command surface is complete | assert `OPEN_PROJECTS` is in `COMMAND_SPECS` and in `command_callbacks`, and that `OPEN_PROJECT` is gone from both | rename the id but leave the old callback row |
| A switch mid-turn is refused whole, not half-saved | set `copilot_turn_active`, `switch_project(other)`, assert the project did NOT change AND the outgoing document's on-disk state is untouched | put the gate after `save()` instead of before it |
| A dead pointer recovers to a seeded project | under a tmp `SHADERBOX_DATA_DIR`, write a pointer at a deleted dir, build `App(project_dir=None)`, assert it loads the default | restore the file-exists-only test |
| The recovery seeds rather than landing blank | same setup, assert `ui_documents` is non-empty | pass `first_run=False` on the recovery path |
| The recovery does not recreate the dead path | same setup, assert the dead directory still does not exist | let `for_root` run on the pointer's path |
| The pointer is repointed at the live project | same setup, assert the pointer file names the default, not the dead path | write the pointer before resolving |
| The recovery opens Projects, not Examples | same setup, assert `popup_state is PopupState.PROJECTS` | leave `_init`'s unconditional `open_examples()` |
| A torn fork leaves no listable project | make `copytree` raise mid-copy, assert no new entry in `list_projects` and no bare partial dir | copy straight to the final name |
| A deleted project is recoverable | delete, assert the dir is gone from the root AND present under `<app_data>/trash/` with its files intact | swap the move for an `rmtree` |
| The open project cannot be deleted | call `delete_project` on the open one, assert it refuses and the dir is untouched | drop the guard and let the confirm decide |
| A row click selects and does NOT switch | set the selection to another project, assert `project_dir` is unchanged; then run the switch verb and assert it changed | make the click call `switch_project` |
| An armed delete clears on a selection change | arm Delete on one project, select another, assert nothing is armed | leave the armed target set across selections |
| Both New and Duplicate reach the ONE validator | drive Duplicate with `..`, a separator and a used name; assert each refuses and creates nothing | give Duplicate its own name check |
| A leftover `.creating` is never listed and is swept | create `<root>/foo.creating/documents/`, assert `list_projects` omits it, duplicate to `foo`, assert the stale dir is gone | drop the name filter, or the pre-copy sweep |
| Every `PopupState` member actually draws | for each member: set it, render a frame, assert no exception and the state survived | add `PROJECTS` without its `draw_projects(app)` call in `ui.py` |
| Trash uses the bare name, suffixing only on collision | delete, assert `trash/<name>` exactly; recreate and delete again, assert a second suffixed entry and the first still bare | suffix unconditionally |

`tests/test_command_registry_coverage.py` already walks the registry, so a `CommandId` with no spec
or no callback fails there too — the last row above is the explicit statement of it.

The one check that is NOT falsifiable headlessly is the menu-bar name placement (D8): layout
geometry reads differently headless, per the UI skill. That is a maintainer `make run` glance, and
the spec says so rather than pretending a headless assert covers it.

---

## Open questions for the user

All four are **resolved**; kept here as the record of what was asked and answered.

1. **The projects root** — confirmed: `<app_data>/projects/`, which now holds `default/` and
   `radiance_cascade/`. (D4.)

2. **The `/tmp` project** — done, before drafting: copied to
   `<app_data>/projects/radiance_cascade`, verified byte-identical, verified loadable by a real
   headless `App` (both passes, right output pass), pointer repointed, original removed.

3. **What a fork copies** — everything, for now, with the policy isolated in ONE predicate so
   changing it later is a one-place edit. (D7.)

4. **The chord** — the modal REPLACES `Open project` and inherits Ctrl+O; the folder picker becomes
   `Open other...` inside it. (D1, D9.)


---

## Review history

Two pre-implementation reviewers (opus), one on correctness/design and one on verification/blast
radius. Both returned NEEDS CHANGES. What they found, and what was done:

**Accepted, and they changed the design:**

- **The switch cannot run inside the modal's draw.** Popups draw near the end of the frame, after
  the editor panel, canvas backdrop and document image have each pushed a raw texture handle into
  the draw list; `release()` would free those before `imgui.render()` reads them. Today's
  `open_project` is safe only because the menu bar draws first. D5 now defers the switch to
  `_tick_frame_state`. This is the finding that most changed the shape, and it exists only because
  the verb moved into a popup.
- **D10 contradicted itself.** "Exactly as a first launch does" opens the EXAMPLES gallery, and the
  popup mutex means Examples and Projects cannot both win; passing `first_run=False` to dodge that
  also skips the starter seed, producing the blank app D10 exists to prevent. Now stated as three
  consumers of one condition, with the examples auto-open gated on `popup_state is CLOSED`.
- **`<app_data>/projects/default` is a stale pre-rename layout** with no `documents/`, so the
  recovery would land in a project D4's own filter excludes. Hand-fixed in this wave.
- **`save()` does not close the whole data-loss bug** — `flush_current_editor` flushes the ACTIVE
  tab while `release()` closes every session, so inactive dirty tabs are lost. The funnel now
  flushes all of them, and calls `save_imgui_ini()`.
- **`copytree`'s `ignore` hook is `(dir, names) -> names`,** not the per-path predicate D7 showed.
  The adapter is now written out.
- **The trash shape was not what it claimed to mirror** — `_delete_document_unguarded` uses the bare
  name and suffixes only on collision; D7b had specified an unconditional suffix.
- **`_init` had no gate on its call sites,** making D5's "structurally impossible" aspirational. An
  AST test now pins it at two callers.
- Plus: the `cell_delete_confirm` function is unusable in a flat row (its rule transfers, not the
  call); the one-shot focus flag and outer-Enter suppression were missing from D6 and would have
  made Enter both create a project and switch; `paths.py` needed a `project_trash_dir()`.

**Rejected as already fixed** (both reviewers read the spec mid-edit): the two over-budget strings
`switch away to delete this one` and `Enter creates - Esc cancels` had already been corrected and
pre-scored, and the copilot busy-gate ordering was already stated. Re-checked against the file
rather than argued.

**Rejected on the merits:** the "duplicate leaves the original alone" falsifier as originally
written ("copy by reference") names a break that cannot be written, since a live-object copy raises
rather than succeeding quietly — the row was rewritten around a reachable break instead of kept with
an unreachable one.
