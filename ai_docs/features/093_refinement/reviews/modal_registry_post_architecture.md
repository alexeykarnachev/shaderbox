# 093 W4 post-implementation review — architecture and conventions

Commit under review: `d5bf84c` against `ai_docs/features/093_refinement/07_modal_registry_spec.md`.
Baseline for diffs: `17d235c`. Read end to end: every file in `git show --stat d5bf84c`
except `tests/test_confirm.py`, `tests/test_project_management.py`,
`tests/test_import_dialog.py`, `tests/test_pass_draft.py`, `tests/test_pass_verbs.py`
and `tests/test_document_reset.py`, which were read at the sections the brief names
(the cleanup/Esc/opener tables and the repointed structural checks) rather than whole —
test *behavior* is another reviewer's role; this pass judged their DOMAINS.

**Verdict: PARTIAL.** Four findings, none should-not-land. The import graph, the
layering, the code rules, the deletions and the doc sweep are all clean and
demonstrated. Three of the four findings are gates whose domain is narrower than
the rule they are named for, which is the family `conventions.md` itself flags as
the most expensive; the fourth is a stale doc the spec's own file list required.

## What holds (demonstrated)

**The import graph is exactly what R2 and the conventions bullet claim.**

```
$ grep -n "from shaderbox.popups" shaderbox/app.py          -> no match
$ grep -n "^from" shaderbox/popups/__init__.py              -> collections.abc, dataclasses, shaderbox.app
$ grep -rn "popups.registry" shaderbox/popups/              -> no match
$ grep -n "from shaderbox.app" shaderbox/ui_primitives.py shaderbox/theme.py  -> no match
$ grep -n "^from\|^import" shaderbox/commands.py            -> dataclasses, enum, imgui_bundle  (still a leaf)
$ grep -n "^from\|^import" shaderbox/ui_models.py           -> no popups, no new module for ConfirmRequest
```

`ui_models.py` gains `ConfirmRequest` on the back of one added stdlib import
(`from collections.abc import Callable`) and nothing else — the payload sits where
`app.py` already imports from, which is the whole point of pre-implementation F1.

**`Modal` in `popups/__init__.py` rather than `registry.py` is the right call, and
the spec should be updated to say so.** The spec's R2 put `Modal` in `registry.py`
while also requiring that the registry import every popup and that every popup build
a `Modal`. Those three cannot hold together: `registry` -> `confirm` -> `registry`.
The package root importing `App` alone breaks it, the registry stays the leaf R2
meant, and the two banned escapes (`TYPE_CHECKING`, a function-body import) are
avoided. The commit message states the reasoning and `conventions.md` already
carries the landed shape ("the shared `Modal` type lives in `popups/__init__.py`,
which imports `App` alone"). The only thing not updated is R2's own code block in
`07_modal_registry_spec.md`, which still shows the dataclass under
`popups/registry.py holds the registry` — a reader who opens the spec after the
conventions bullet sees two homes. Fix is one sentence in R2.

**Code rules: clean.** No `if TYPE_CHECKING` anywhere in `shaderbox/` (the one grep
hit is prose in `copilot/backend.py`'s docstring). No `@staticmethod`; the three
`@classmethod` in `ui_models.py` are pre-existing alternate constructors (present at
`17d235c`, count 3, unchanged). No function-body imports in `shaderbox/` outside the
two sanctioned lazy-SDK seams (`exporters/youtube.py`, `copilot/llm/openrouter.py`);
the one body import the diff adds is `tests/test_profiling.py::_pass_settings_frames`,
which already had that shape at baseline and only had `PopupState` renamed to
`ModalId`. No suppression added (`ruff check shaderbox/` passes; `pyright` 0 errors).
No `TODO`/`FIXME` in any changed file. No `Any` on a real param in the diff. Full
annotations throughout the new code.

**Comments state present facts, never history.** Every comment the commit adds to
`shaderbox/` was read: the Esc precedence list, the appearing-frame edge-trigger
note, the `close_current_popup`-is-scope-legal note, the mutex's structural claim.
Each names a constraint that holds now. The `popups/__init__.py` docstring's "the
two would otherwise cycle" is a present-tense invariant, not a bug story.

**No per-modal draw wrapper survived.** `grep -rn "modal_window" shaderbox/` shows
exactly one caller: `popups/registry.py::draw_modal`. Eight `draw_*` wrappers are
gone and `ui.py` imports `{"draw_modal"}` alone, which the chrome gate pins.

**The deleted symbols are gone from every live surface.**

```
$ for s in PopupState popup_state close_popup copilot_revert_target \
           confirm_menu_item confirm_label _draw_revert_modal; do
    grep -rn "$s" shaderbox/ scripts/ .claude/; done   -> none, all seven
```

`ai_docs/` retains them only in `023_app_refinement_wave.md` and `05_menus_spec.md`'s
reversal notes, which are historical records describing what those waves did — correct
per the doc rules. `tests/test_modal_chrome.py` names them in `_RETIRED`, which is the
deletions gate itself.

**Docs.** `conventions.md`'s registry bullet, the confirm bullet and the command-table
bullet all describe the landed code; `dev_flow.md`'s module map gains `popups/registry.py`,
`popups/__init__.py` and `popups/confirm.py` and drops `confirm_menu_item` from the
`ui_primitives.py` entry; the skill's §7.1/§7.2/§7.3/§7.4 are rewritten to the registry
and the plain-item confirm; `05_menus_spec.md`'s M5/M8/M9/M12 carry reversal notes that
point at `07`; `06_command_system.md` rule 5 is the modal rule and its fenced map lost
the `▸` submenus; `01_spec.md` gains the wave-4 entry; `00_findings.md` row 18 records
the maintainer's question. No TODO or deferred marker in any of them. Every cross-pointer
I followed resolves.

**Where things live — all correct.** `ConfirmRequest` in `ui_models.py` beside
`ImportDraft`/`PassDraft`. The confirm verbs on `App`. `_delete_pass` moved out of
`widgets/pass_list.py` into `App.delete_pass` (the widget layer should not own editor
teardown). `inline_input_owns_esc` on `ShaderLibFileManager` in `shader_lib/file_ops.py`,
read as `app.shader_lib_files.inline_input_owns_esc` with no delegating facade on `App` —
exactly what the `InlineInput` bullet requires. `close_modal(app, forced: bool = False)`
defaults to the Esc semantics and is forced only by `draw_modal`, so a new caller gets
the conservative behavior by default; that is the right default direction.

`switch_project`'s `_init` clears `self.confirm` and writes `self.modal = None` directly
rather than calling the funnel — correct, and matching R2's carve-out: it runs in
`_tick_frame_state`, where `close_current_popup` would assert (084 D5).

**`App.clear_confirm` earns its existence** despite being a one-line setter with one
production caller: `tests/test_confirm.py::_CLEANUP_METHOD` spies every row's cleanup
by method name, and a lambda body would give the CONFIRM row no seam. It is the seventh
entry in a uniform table, not a gratuitous method.

## Findings

### 1. The mutex walk covers `popups/` and `widgets/` but not `tabs/`, which is the same layer (gate, domain narrower than its rule)

`conventions.md`'s three-layer bullet names three peer leaf surfaces: "`widgets`/`popups`/`tabs`
= pure draw functions taking `app: App`, each a LEAF surface". The registry bullet's
rule is stated without qualification — "rejects any `app.modal` write under `popups/` or
`widgets/`" — but R2's actual invariant is "No module under `popups/` or `widgets/`
assigns `app.modal`", and the gate implements that literal list:

```
$ git show d5bf84c:tests/test_modal_chrome.py | grep -n '_package_sources'
246:    _package_sources(("popups", "widgets")),
247:    ids=[name for name, _ in _package_sources(("popups", "widgets"))],
```

`tabs/` is outside the domain. Demonstrated by mutation — with `app.modal = None`
inserted into `shaderbox/tabs/document.py::_draw_document_reset` (one line above the
`reset_document_confirmed` call that the wave itself just rewired):

```
$ uv run pytest tests/test_modal_chrome.py -q -p no:randomly
................................................................ [100%]
64 passed in 0.84s
```

A leaf surface writing the mutex is exactly the defect the gate exists for — it closes
around the per-modal cleanup the funnel owns — and `tabs/document.py` is a file this
wave edited, so it is not a hypothetical location. `tabs/` also hosts `code.py`,
`uniforms.py` and `render.py`, all taking `app: App`.

Fix: `_package_sources(("popups", "widgets", "tabs"))`. Cheap, and it makes the gate's
domain equal the conventions rule's domain rather than a subset of it. (The mutation
above was introduced into the live tree by a concurrent reviewer, not by me; I observed
it, ran the gate against it, and left the file alone.)

### 2. The chrome gate's `_BODIES` table can point a row at another modal's body and stay green (gate, silently self-narrowing)

`_BODIES` maps each `ModalId` to its leaf body functions, and
`test_every_registry_row_has_leaf_bodies_listed` checks only that every id is PRESENT
and its tuple non-empty. Nothing ties a row to the module it belongs to, so a row that
names the wrong function passes every clause below it — and that modal's real chrome
then goes unchecked entirely. Demonstrated, mutated in place and restored:

```
$ sed -i 's/ModalId.EMOJI_PICKER: (emoji_picker._draw_body,),/ModalId.EMOJI_PICKER: (help._draw_body,),/' tests/test_modal_chrome.py
$ uv run pytest tests/test_modal_chrome.py -q -p no:randomly
................................................................ [100%]
64 passed in 0.84s
$ cp <backup> tests/test_modal_chrome.py && git diff --quiet tests/test_modal_chrome.py && echo RESTORE VERIFIED
RESTORE VERIFIED
```

The spec's break #9 (pointing the pass-settings row at its dispatcher) IS caught, but
only incidentally — the dispatcher binds no `keep_open`. A row pointed at any other
real body binds and returns `keep_open` and ends in a Close row, so all three clauses
pass on the wrong function.

The gate's own docstring names the family it belongs to ("walking the dispatcher passes
while checking nothing, which is the checker-narrows-its-own-domain family"), and the
skill's §7.1 repeats it — so this is the stated concern, one level up, unguarded.

Fix is one assertion in `test_every_registry_row_has_leaf_bodies_listed`: each listed
body's `__module__` must be the module that defines that id's `Modal`. That is
derivable — `BY_ID[modal_id]` gives the row, and `row.body.__module__` (or
`row.size.__module__` for the lambda rows) gives the owning module — so it needs no
second hand-written table.

### 3. `popups/lib_picker/__init__.py` keeps a dead `modal_window` import, and ruff is configured not to see it

```
$ git show d5bf84c:shaderbox/popups/lib_picker/__init__.py | grep -n modal_window
24:from shaderbox.ui_primitives import modal_window, primary_button, standard_button
```

One occurrence in the file — the import. The `draw_lib_picker` wrapper that called it
was deleted by this commit; the name is now unreachable. It is invisible to the gate
because `pyproject.toml` silences the rule for this filename shape:

```
[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]  # unused imports
```

That ignore exists for the legitimate re-export idiom, and `lib_picker/__init__.py` is a
package root that happens to also be a real module — so the exemption meant for re-exports
lands on a file that has genuine dead code. `pyright` also reports 0 errors on it.

The other candidates are clean: `widgets/copilot_chat.py` still uses `sanitize_display`,
`Message`, `caption_text`, `primary_button` and `standard_button` after the revert modal's
deletion, so nothing was orphaned there.

Fix: drop `modal_window` from that import line. Worth considering separately whether the
`F401` ignore should be narrowed to files that actually re-export, since this is the shape
that hid it.

### 4. The roadmap banner is stale and still describes the reversed submenu — the spec's own file list required it

`07_modal_registry_spec.md ## Files touched` ends with "`00_findings.md` row 18 (his
question as the finding), **the roadmap banner**". The banner was not touched:

```
$ git show --stat d5bf84c | grep -i roadmap   -> no match
$ git log --oneline -1 -- ai_docs/roadmap.md  -> 9b44d7c  (the previous wave)
```

Its content now contradicts the landed code on the exact point this wave reversed:

```
ai_docs/roadmap.md:33
group box, the canvas and a document tile (`Delete` confirms through a submenu), the name
```

and the banner frames wave 4 as future work ("Each finding is a new ledger row (18+) in
`00_findings.md`, fixed as wave 4 by ..."), when wave 4 is what this commit is.

The banner is the cold-start chain's step 2 and `CLAUDE.md` calls it "the authoritative
'what's next?'" — so a fresh session reads a next-step that is already done and a UI
shape that no longer exists. `01_spec.md` was correctly updated to "waves 1-4 landed",
which makes the two documents disagree.

Fix: rewrite the Active-context block in full (its own comment says "Rewrite this block
IN FULL each time it changes. Do NOT append.") to name wave 4 as landed and the visual
review of the confirm modal as the next input.

## Judged, not a finding

**The prose budget: `_CONFIRM_LINE_BUDGET = 12` and the clause-joiner exemption are
sound, and the exemption IS call-site-scoped.** Measured, all seven request sites,
21 scored strings:

```
title   2w/5   'Revert ""?'              line  12w/12  'Shaders edited since that message are restored to their state before it.'
title   3w/5   'Delete pass ?'           line  10w/12  'Its wiring and position are lost; the shader file stays.'
title   4w/5   'Move  to the trash?'     line   7w/12  'Nothing in the app brings it back.'
title   2w/5   'Reset ?'                 line   8w/12  'Feedback histories, the clock and the script restart.'
title   3w/5   'Clear the conversation?' line   7w/12  'The transcript and its checkpoints are dropped.'
title   2w/5   'Delete ?'                line   4w/12  'It moves to .trash.'   (x2, the lib tree)
verb: 1w/3 at all seven
```

Against the rulebook §2 and the gate's design this holds on four counts. (a) The
exemption is scoped by `site.call == "ConfirmRequest" and site.parameter == "line"` —
a predicate on the ROW, so it covers the confirm's consequence line at every authoring
site and nothing else; the title and the verb stay under the ordinary heading and button
budgets and stay clause-checked. (b) It is a budget, not a waiver: 12 words rejects a
paragraph, and only one of the six distinct lines uses more than 10. (c) The reason is
written at the rows and is a claim about the READER, not about this string — a
destructive confirm is read by someone who stopped, and "what is lost; what survives"
is two clauses by design; `'Its wiring and position are lost; the shader file stays.'`
would lose its second half under the one-clause rule, and that half is the reassurance.
(d) It replaces a strictly worse exemption: the old `_OVER_BUDGET` row for
`copilot_chat.py::_draw_revert_body` at **20 words**, which this deletes.

One caveat worth stating rather than filing: the revert line is 12 words against a
budget of 12, so the number was fitted to the longest existing string. That is the
normal way a budget gets picked here, and the next line over 12 fails loudly rather
than silently — but it means the budget has zero headroom and the next destructive verb
is likely to arrive wanting a thirteenth word. The right response then is to shorten the
line, not to raise the number; `conventions.md` already says a budget is revisited "by
changing it in one place — the number in the test — not by exempting sites one at a time",
which this respects.

The `_UNMEASURABLE` entry for `confirm.py::_draw_body` is honest, not a suppression: the
three strings there are forwarded names (`request.title`, `request.line`, `request.verb`)
that the walk genuinely cannot read, and each is scored at its own authoring site — so
nothing is lost. The commit message's "ruled out" list says a `_UNMEASURABLE` entry for
the CALL SITES was rejected for exactly the right reason.

**Duplication: nothing worth extracting, with one borderline.** The four `*_confirmed`
verbs on `App` are not repetition a helper would remove — each differs in all four
request fields (title, line, verb, callable), so a builder helper would only relocate the
argument list and add a layer between the verb and the copy it authors. Keeping the
literal `ConfirmRequest(...)` at the verb is also what lets the prose gate score each
string at its own site, which is load-bearing for the finding above; a helper would turn
all seven into `_UNMEASURABLE`. Leave them.

The borderline is `popups/lib_picker/tree.py`, where `_confirm_file_delete` and
`_confirm_dir_delete` are byte-identical but for `delete_file` vs `delete_dir`:

```python
def _confirm_file_delete(app: App, path: Path) -> None:
    app.request_confirm(ConfirmRequest(title=f"Delete {path.name}?", line="It moves to .trash.",
                                       verb="Delete", on_confirm=lambda: app.shader_lib_files.delete_file(path)))
def _confirm_dir_delete(app: App, path: Path) -> None:
    app.request_confirm(ConfirmRequest(title=f"Delete {path.name}?", line="It moves to .trash.",
                                       verb="Delete", on_confirm=lambda: app.shader_lib_files.delete_dir(path)))
```

Two call sites, one differing token. A shared `_confirm_trash_delete(app, path, remove)`
taking the callable would collapse them — but it would also make both strings
unreadable to the prose walk, trading a five-line duplication for two `_UNMEASURABLE`
rows. At N=2 the duplication is cheaper than the exemption. Leave it; revisit if a third
`.trash` verb lands.

**No hand-rolled action row that `confirm.py` should share.** Every modal body in the
repo writes its own `imgui.dummy((0, SPACE.MD))` + button row — there is no
`action_row` primitive in `ui_primitives.py`, and the chrome gate pins the SHAPE by AST
rather than by a shared helper. `confirm.py` follows that established pattern exactly, so
it introduces no new duplication. Extracting a helper now would be a repo-wide change
outside this wave and would defeat the AST gate, which reads the literal calls.

**`wrapped_caption` instead of the spec's `caption_text` is a correct deviation.** R4
specified `caption_text`, which does not wrap; at 380px a 12-word line needs to. The
commit reports the rendered PNG was read and the line wraps inside 380px. Not a finding,
but the spec's R4 text is now one word off the code.

**`open_copilot_revert` breaks the `*_confirmed` naming family.** Five verbs read
`<verb>_confirmed`; the sixth reads `open_copilot_revert`. The spec's R5 table specified
that name, and nothing derives the family programmatically (`grep -rn "_confirmed" tests/`
finds no name-based enumeration), so it is cosmetic. Worth noting only because
`*_confirmed` itself reads as "already confirmed" rather than "asks first" — if the family
is ever renamed, `request_*` would say what these do.

## False trails

Each of these looked like a finding and is not. Recorded so the next reviewer does not
re-spend the time.

- **`App.delete_pass` vs `ProjectSession.delete_pass` vs `CopilotBackend.delete_pass` —
  the copilot's delete skips the editor teardown.** True, and pre-existing, unchanged by
  this wave. At `17d235c` the teardown lived in `widgets/pass_list.py::_delete_pass`,
  equally unreachable from `copilot/backend.py`, which binds `pass_delete=self.delete_pass`
  (the SESSION method) at `project_session.py:409` — identical at baseline. Moving the
  teardown from the widget to `App` neither creates nor widens the gap. Not this wave's.

- **`tabs/document.py` writes `app.modal = None`.** That line is a concurrent reviewer's
  live mutation testing break #23-ish, not committed code. `git show d5bf84c:shaderbox/tabs/document.py
  | grep app.modal` returns nothing. It is cited in finding 1 only as the demonstration
  vehicle for a real domain gap, which I verified independently against the gate's source.

- **`widgets/copilot_chat.py` orphaned imports after `_draw_revert_modal`'s deletion.**
  Checked all five candidates; every one still has other uses in the file. Clean.

- **The `_BODIES` table is hand-maintained, which R6 was supposed to eliminate.** The
  spec explicitly kept it ("its leaf bodies ... listed in the test beside the registry
  entry") because a `Modal.body` that dispatches cannot be walked. The table is the
  smallest honest version of that. The real gap is that it is unvalidated against the
  owning module (finding 2), not that it exists.

- **`close_modal`'s `forced` defaults to `False` — a new caller could close too weakly.**
  Deliberate and correct: `False` is the Esc semantics, which is the conservative one
  (it declines rather than discarding a half-typed rename). `draw_modal` is the only
  caller that forces, and the conventions bullet documents the split.

- **`ModalId` became a `StrEnum` where `PopupState` was an `Enum`.** Not a behavior change
  the wave needs to justify — the values were already strings, `scripts/smoke.py` asserts
  `isinstance(app.modal, ModalId)`, and the test ids read as `modal_id.value`. Fine.

- **`request is None` in `confirm._draw_body` returns `False` and force-closes.** That is
  the right handling of an impossible-by-construction state (the modal only opens via
  `request_confirm`, which sets `confirm` first), and it satisfies the chrome gate's
  "binds and returns `keep_open`" clause through the normal path. No dead branch concern.

## Gate

Not run whole: the working tree carried another reviewer's live mutations during this
pass, so a `make gates` exit code would have been a claim about their tree, not the
commit. The tracked tree was verified byte-identical to `d5bf84c`
(`git diff --quiet HEAD -- shaderbox/ tests/`) at the moment the wave's own suites ran:

```
$ uv run pytest tests/test_confirm.py tests/test_modal_chrome.py \
                tests/test_ui_prose_budget.py tests/test_menus.py -q -p no:randomly
815 passed, 4 skipped in 6.51s
$ uv run ruff check shaderbox/     -> All checks passed!
$ uv run pyright shaderbox/popups/lib_picker/__init__.py -> 0 errors, 0 warnings
```

The commit's own claim (`make gates`: exit 0, smoke passed) is consistent with this and
with the 24 named breaks, three of which the commit reports as having found real test gaps.
