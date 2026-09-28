# 108 nightly sweep — progress log

A log of what happened, appended after each wave. Not a plan; the plan is `01_spec.md`.
On resume, read this file and `git log` first — they are the truth about where the work
stopped.

## Phase 1 — presence scan DONE

Twelve scans, one question each, presence + one example only. Baseline `make gates`
green before any of it: exit 0, smoke RAN (not skipped).

### PRESENT
- **GL lifetime**: `core.py::_upload_instances` sets `self.vao = None` on a buffer
  outgrowing its reserve, with no `.release()`. `invalidate` and `compile` both release
  first. VERIFIED BY MAIN SESSION against the three sites.
- **Swallowed failure**: `draw_copyable_text` returns `False` on a missing clipboard
  backend; `popups/lib_picker/preview.py` discards it while two other callers handle it.
  `app.py::apply_syntax_theme` logs an exception on a user-triggered theme switch and
  does not notify.
- **Abstraction leak, narrow**: `tabs/code.py` reaches `app.session.script_engine.*` at
  three sites for `cached_source` and `returned_value_types`, past a `ProjectSession`
  that forwards every sibling call. The same file uses the forwarders correctly at seven
  other sites, so the inconsistency within one file is the evidence.
- **Repeated rule**: "a document keeps at least one pass" as `> 1` in `pass_list.py` and
  `== 1` in `project_session.py`. VERIFIED: those are the only two deciding sites.
- **Skill instructs a violation**: `sanitize/SKILL.md` says to add a `todo.md` entry, and
  again in its report template. VERIFIED both sides verbatim against `CLAUDE.md:14`,
  `dev_flow.md:934` and `todo.md`'s header. The skill predates the 2026-07-27 freeze.
- **Stale roadmap claim**: row 020 lists `delete_lib_file` and `bind_media` as deferred;
  both are live gated tools. VERIFIED by grep. `undo_edit` has zero hits and is
  genuinely absent.
- **Unreached branches**: `GateKind.FILE` is absent from `copilot_chat.py`'s dispatch
  because file gates route through a separate `ask_file` slot — structurally unreachable
  there, not a missing case. `theme.py::apply_theme`'s `accent` has one caller that never
  passes it.
- **Defect class from 120 commit bodies**: a string-keyed lookup resolving a missing or
  malformed key to a plausible-but-wrong value. Five instances, all in the theme/capture
  resolver, five rounds to close. Whether it lives elsewhere is being probed.

### ABSENT — recorded so a later sweep does not re-run them
- **Per-item state left behind on teardown**: ABSENT. All fifteen keyed stores on `App`
  and `ProjectSession` were tabulated against the document-delete, project-switch,
  document-switch and pass-delete chains. Every one is CLEARED or structurally cannot
  leak. Two teardown mechanisms (explicit pop, and wholesale reinit on `_init`) both
  actually run — the second is easy to miss on a `.pop`-only grep.
- **Duplicated decisions**: essentially ABSENT. Canvas bounds funnel through
  `clamp_canvas_size`; `MAX_ITERATIONS` is imported everywhere; `RESERVED_NAMES` is
  built as a union with a comment saying it exists so the halves cannot drift; path
  basenames have a dedicated gate (`test_basename_is_never_respelled`); pydantic
  defaults are never restated. The delete-pass pair above is the one hit.
- **Oversized files**: LEAVE WHOLE, all three, on evidence rather than taste —
  `conventions.md` records a maintainer decision for `app.py`, a live gate enforces
  `ui_primitives.py`'s single flat closure, and `backend.py` is one class over one
  shared brake-state machine.
- **Test-suite padding**: ABSENT. Zero lines in TOOLING or PROSE across the top fifteen
  files. Every `.md`-reading test reads a SHIPPED resource, not documentation. The
  weight is earned.

### FALSE TRAILS — do not re-litigate
- `document.py::resample_canvas` allocates before releasing on purpose; the old canvas
  must stay readable during the blit, and its docstring says so.
- `core.py::compile`'s explicit buffer release is redundant-but-deliberate, measured at
  the same live count either way; its comment names `invalidate`'s twin as the one that
  genuinely leaks.
- `except ... : continue` in enumeration loops is skip-one-bad-item, not error hiding.
- `uniform_coerce.py` and `scripting/` naming `moderngl` types is not a layering breach:
  GL-free means needing no live context, and the Module map's own text names the type.
- `copilot/backend.py`'s `len(document.passes) < 2` picks a prompt view, not a delete
  eligibility — merging it with the delete guard would be wrong.
- `editor_script_types` and `editor_completion_offered` look uncleaned but are
  whole-dict reassigned on any key mismatch, so no stale entry is reachable.
- `graph_canvas/render.py`'s `.frag.glsl` literals are the widget's own built-in shaders,
  unrelated to a document's pass files.
- The `ship` skill committing on `master` is the sanctioned promotion path, carved out by
  `CLAUDE.md` itself — not a violation of "commit on dev".

### PROCESS NOTE
One scan agent ran `uv add --group dev pytest-cov` despite a read-only brief, caught it
via `git status` and reverted it. Tree confirmed clean. `pytest-cov` is NOT a dependency
of this project and coverage was not run.

## W-4 — the skill that instructed a violation. DONE (eb32d9de)

`sanitize/SKILL.md` told a session to file a new `todo.md` entry in THREE places, not
the two the scan found: the walk step, the convention-audit step, and the sweep-report
template's "Y added" column. Found two by grepping the skill against the rules, and the
third only by re-sweeping the whole skill directory afterwards — the usual shape, where a
removal is cleaned up in the file class the author expected and left in the one they did
not. The remaining `todo.md` mention under `.claude/` is dogfood's, stating the rule
correctly.

## W-5 — the stale roadmap claim. DONE (5e0fb8e4)

Row 020 listed `delete_lib_file` and `bind_media` as parked scope decisions. Both are
live, registered, gated tools, and row 052 has said so all along — the two rows
contradicted each other and the later one was wrong. `undo_edit` (zero hits under
`copilot/`) and semantic editing (`edit_shader` still matches `old_str`/`new_str`
substrings) stay. Corrected rather than deleted: the row is frozen history.

## W-1 — the dropped VertexArray. DONE (3a8605f4)

`core.py::_upload_instances` dropped a live VAO on a population outgrowing its reserve.
Two lines to fix, and the gate is the whole point: GL hands released names back out, so
the rebuilt VAO landing on the SAME name is what proves the old one was freed rather
than forgotten. Break (delete the release, keep the assignment) → the VAO takes name 3
where the fixed code recycles name 2. Restored and verified before the suite ran.

The fixture asserts it REACHED the path before asserting anything: a VAO existed,
buffers were allocated, and at least one buffer's size actually grew. An existing
sibling test already grew a population past capacity and never looked at the VAO — the
fixture made contact and measured the wrong thing, which is why this stayed invisible.

## W-3 — the malformed key. DONE (7bf51547)

The probe's verdict: **the class has exactly one live instance outside the theme
resolver**, and it is the same shape as theme round four. `PASS_NAME_RE` asked with
`re.match`, where `$` matches before a trailing newline.

`"glow\n"` was a legal pass name, so the collision check behind the guard could not see
it either — a second pass beside `"glow"`, written to `passes/glow\n.frag.glsl`. Rename
was worse: refused `"b"`, accepted `"b\n"`, then moved a real file onto that path.
Reached from the copilot's `add_pass`/`set_pass`, neither stripping, while the sibling
`rename_document` and duplicate both do. The UI cannot reach it (single-line
`input_text`, stripped).

Gated at BOTH sites and BOTH verbs — one of N identical paths being pinned is how a
sibling drifts back. The add and rename bad-name lists never held a trailing-newline
case, which is where a reader looks and why it survived.

### The probe's clean answers, recorded so no later round re-runs them
`wired_pass`, `_auto_source`, `namespace_error`, the copilot address parser, the
document-id resolver, `Theme.capture`, `resolve_palette_refs`, `group_tint`,
`entity_fields` and the script engine's `KeyFailReason` domain all answered every
malformed input correctly. Several are clean by a documented guard rather than by luck:
`KeyFailReason` enumerates its domain with `get_args` and derives the silent set by
SUBTRACTION, so a new member cannot default to silence — the direct opposite of the class.

### Unreachable by a guard upstream, NOT findings
- `validate_project_name` accepts `"a\nb"`; its only caller is fed by a single-line
  `input_text` and is not copilot-reachable.
- `instanced.validate_fields` misses `"vs_quad "`; `intel/glsl.py` strips and
  `isidentifier()`-checks before it, so no field name ever carries whitespace.
- `theme_file._parse_hex` accepts a trailing newline; both callers read values from a
  `splitlines()` loop that cannot produce one.
- `PassEntry.group`'s pydantic pattern is anchored correctly and rejects both newline
  cases — the defect was confined to the two `.match()` call sites.

## W-2 — the silent failure. DONE (d4995e35)

The scan found two discarded sentinels; the wave found **five clipboard call sites in
five spellings** — one notified on success only, two discarded the result, one used
`contextlib.suppress`, and a fifth had grown its own `copy_to_clipboard` helper inside
the lib picker. All five advertise "Copy" in a tooltip. None could report a failure.

**The fix is the return TYPE, not a handler per site.** `draw_copyable_text` answered
`bool`, collapsing "the copy failed" and "there was nothing to do" into one `False` — so
even the caller that checked the result could not distinguish them. Three states now:
`None` unclicked, `""` landed, the reason otherwise. One helper owns the `pyperclip`
call and its message names the fix rather than only reporting a failure.

`draw_link` deliberately ignores the reason: it opens a browser on the same click, so
that action is visibly not a no-op.

**The AST half of the gate earned its place on its first run**, by failing and naming
`popups/lib_picker/filtering.py` — the fifth site, which my inventory had missed. I had
written "four call sites" in the commit draft. Breaks tried: reintroduce an inline
`pyperclip.copy` in a widget, and make a failed copy answer `""` like a successful one.
Each fails its own half.

## W-6 — the repeated rule. DONE (ed524fc5)

`Document.can_delete_a_pass`, beside `render_pass` whose docstring already says why the
last pass cannot go. Both consumers ask it.

**This consolidation does buy coverage, and I checked rather than claiming it** — 107
closed with a review finding that its equivalent claim was false. Breaking the predicate
to `> 0` fails FOUR tests across THREE surfaces: the copilot's `delete_pass` tool, the
menu, and the session verb. Two of those tests already existed and now route through the
predicate; before, each surface needed its own break and the menu item's enabled state
had no behavioural coverage at all.

## Status: all six waves landed, gates green at each

| wave | subject | commit |
|---|---|---|
| W-4 | the skill instructing a banned todo.md entry | eb32d9de |
| W-5 | roadmap row 020's two shipped-but-listed items | 5e0fb8e4 |
| W-1 | the dropped VertexArray | 3a8605f4 |
| W-3 | the trailing-newline pass name | 7bf51547 |
| W-2 | the unreported clipboard failure | d4995e35 |
| W-6 | the twice-spelled delete rule | ed524fc5 |

### Two process notes for the next sweep
- **`make gates` went red on three of six waves, every time for the same reason**: ruff
  reformatted a file and exited non-zero for having done so, with pyright at 0 errors.
  The second run was green each time. The spec's constraint section predicted this
  exactly; it cost nothing because it was written down beforehand.
- **My inventory of a class was short in W-2 and the gate caught it, not me.** The
  lesson is not "count more carefully" — it is that an AST gate over the whole tree finds
  what a grep-and-read of the sites you already know about cannot, and it is worth
  writing even when the class looks small enough to enumerate by hand.

## W-7 — a second instance of W-2's class, found by its inventory (01986f4a)

The W-2 inventory (an agent tracing every failure sentinel to a user action) ranked
**`:w` announcing a save that did not happen** above anything else it found. Verified
directly: `App.save` answered `None` whether it wrote, was refused mid-copilot-turn, or
raised — so `hotkeys.py` pushed "Saved" on top of the lock warning or the error. The
handler's own comment says `App.save` was chosen as the one funnel precisely so `:w`
would not lie about a save, and then the return value could not carry the answer.

**The same defect as the clipboard**, one layer up: a function whose type cannot express
failure, and a caller that therefore cannot report one. That is now two independent
instances of the class the spec named, in code neither of them shares.

`save()` reports; `Ctrl+S` takes a named `save_command` wrapper because the command
registry holds `Callable[[], None]` and widening it would be the wrong direction.

### A fixture of mine that was kinder than the suite
The `:w` tests measured pushed notifications by diffing the stack's length. The stack is
a `deque(maxlen=5)`, so once full a push does not change the length. **The file passed
alone and failed in the full suite** — the fixture was answering a question about its own
starting state. It clears the stack first now. Recorded because it is the exact shape
107 spent its night on, produced fresh by me while fixing that shape elsewhere.

## Doc citations — three fixed, one false alarm (2698b403)

Stale symbol citations on ground 107 did not cover (it checked `dev_flow.md`'s module
map; these are roadmap rows and a skill): row 106's three colour-table symbols, row 016's
`LibIndex.build`, row 025's `sync_nodes_from_disk`, and the imgui skill's
`widgets/pass_graph.py::_draw_canvas`. Row 025's `App.reload_nodes_from_disk` citation
STAYS — that row records the deletion and says so.

**A FALSE ALARM worth recording**: the scan reported the banner's "a light theme is out
of scope" as contradicted, because `gruvbox_light.theme` ships and is selectable. It is
not contradicted — that picker is `##syntax_theme` and drives `apply_syntax_theme`, which
repaints the EDITORS. The banner is about the app chrome palette, which is still unbuilt.
Two different things with one word in common.

## NOT done, deliberately, and why

- **`project_session.py::_delete_document_unguarded`'s unguarded `shutil.move`.** Ranked
  top of the W-2 inventory and VERIFIED: the document is popped, released and deselected
  before the move, so an `OSError` there leaves it gone from memory and present on disk,
  reappearing on the next load. Not fixed unattended because the fix is a design choice —
  whether a failed move re-loads the document or reports and leaves it — and the function
  has four callers including the copilot's revert path, whose `trash_name` return drives a
  Recover affordance. **A maintainer decision, not a mechanical one.**
- **`copilot/session.py::_run_one_turn`'s `build_context` outside the try.** Reported as
  leaving the worker dead with `in_flight` stuck True. Plausible and serious; not verified
  by me, and not fixed, because reproducing it means driving a real turn.
- **`widgets/uniform.py`'s `contextlib.suppress` on an array parse.** Examined and left
  alone deliberately: it fires per keystroke, so mid-typing `"1, 2,"` is legitimately
  unparseable. Surfacing here is the cry-wolf case the spec warns about.
- **`:wq` closes the tab whether or not the save landed.** True before this sweep, left
  alone; the tab-close path has its own unsaved-changes guard.
