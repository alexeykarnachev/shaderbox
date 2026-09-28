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
