# 097 — found while cutting, NOT fixed

Five agents read the suite and the repository during 097. These are the things they
found that are DEFECTS or dead weight rather than test clutter, so none was touched:
cutting tests and changing behaviour are different jobs, and the maintainer decides
which happens when. Each was verified at the source, not taken from a report.

## Two places that decide one thing differently (the 096 shape, again)

**1. A command's scope is answered twice, and the two answers disagree.**
`CommandScope` states its own contract in a comment (`shaderbox/commands.py`): an EDITOR
command lives by `app.editor_focused`, a COPILOT one by `app.copilot_focused`.
`shaderbox/hotkeys.py` obeys it. `shaderbox/menus.py` substitutes `app.active_tab is not
None` / `app.is_copilot_open` -- strictly weaker, so a menu offers a command the keyboard
refuses. Nothing gates the two against each other.

**2. Three copilot tools default an argument the resolver rejects on purpose.**
`shaderbox/copilot/tools/document_ops.py` declares `document: str = Field(default="")` on
rename / set_canvas_size / duplicate. `CopilotBackend._copilot_resolve_document_id` rejects
an empty handle deliberately, with a comment saying why (a required target must refuse
rather than fall back to the current document). So the default is guaranteed to fail.
Worse, the field's description lives in four modules with two different meanings:
`passes.py` and `script.py` promise "empty = the current document", `document_ops.py` does
not.

This is why the test above it is green: `tests/test_document_ops.py` stubs the resolver
with one that ACCEPTS an empty string. The fake is kinder than production, so the test is
green about a path that cannot happen.

## Dead code and files

- `shaderbox/resources/fonts/Anonymous_Pro/AnonymousPro-Italic.ttf` and `-BoldItalic.ttf`
  (~195 KB): the only font load is a two-branch literal, Bold or Regular, and nothing
  globs the directory. Both ship in every itch.io zip.
- `shaderbox/resources/shaders/editor.frag.glsl`: orphaned by the move to libeditor. The
  real editor shader is inlined as a Python string in `editor/render.py`, and they are not
  even variants of each other (330 vs 460, different uniforms).
- `thumbnail.png`, `probe.ini` at the repo root: zero references. `probe.ini` is a
  HelloImGui file, and this project has no HelloImGui.
- `shaderbox/ui.py::ViewerGeometry.image_height`: constructed, never read.
- Unused parameters: `widgets/pass_graph.py::_draw_canvas(wiring, groups)` and `_snap(picture)`,
  `scripting/engine.py::reload(document)`, `tabs/code.py::_python_request(app)`.
- `exporters/telegram.py` and `exporters/youtube.py` carry byte-identical `status()` and
  `update()`; both are abstract in `exporters/base.py` and read only fields both have.
- `pass_graph.py` declares `DTYPES` and `TARGET_DTYPES` as two independent literals six lines
  apart, neither derived from `TargetDtype`. They agree today.
- `render_preset.py::_align` has a floor (`max(alignment, ...)`); the inline copy in
  `document.py::_render_video` does not, so it can yield 0. Latent: both registered
  exporters take the branch that fills `resolution_details`.

## A .gitignore pattern that covers part of its domain

`.gitignore` has `projects/*/media/`, which needs an intermediate segment. `projects/dev/media/`
(69 files) is ignored correctly; `projects/media/` is NOT, and two regenerable Telegram sticker
caches are committed. `dev_flow.md` states plainly that this cache must never be committed.
Verify with `git check-ignore --no-index projects/media/<file>` (exit 1 = not ignored).

## Docs asserting things the code does not do

Each verified by running the check, not by reading:

- `CLAUDE.md` says `build.sh` strips the `projects/dev/` sandbox from the shipped bundle.
  `build.sh` copies `shaderbox/` plus an allowlist and never copies `projects/` at all.
- `dev_flow.md` says `scripts/smoke.py` needs that sandbox as a fixture. Smoke avoids it
  explicitly and seeds its own tmp project from the shipped examples.
- `conventions.md` documents a project-fork mechanism via `_copy_into_fork`; no such symbol
  exists. `.creating` from the same paragraph IS live, which makes the passage half-true --
  worse than a wholly dead section, since checking one name appears to confirm the rest.
- `conventions.md` names `_CTX_GLOSS` / `_CTX_HELP` (actually `_CONTEXT_GLOSS` /
  `_CONTEXT_HELP`), places `ENGINE_DRIVEN_UNIFORMS` in `core.py` (it is in
  `engine_uniforms.py`), and puts jedi in `worker.py` (it is in `intel/python.py`).
- `conventions.md` lists four modules carrying an xdist group; there are nine.
- `BUILDING.md` promises Ctrl+N for a new document; the chord is Ctrl+Shift+N, and the
  button reads "Open folder", not "Open dir".

## Rules whose gate was deleted in 097

Both were cut deliberately, and both now need a human to hold them -- stated here so the
next reader knows they are conventions rather than enforced:

- The UI word budget (`conventions.md`, and the `/imgui-ui` skill's table).
- American spelling across every surface a reader sees.
- `roadmap.md`'s own "one row, one sentence" rule, which `test_roadmap_shape` held. It is
  currently violated by several rows including 097's.

## In this repo's own source, pointing at deleted tests

`shaderbox/theme.py` and `shaderbox/ui_primitives.py` still cite `tests/test_ui_prose_budget.py`
in comments; `.github/workflows/ci.yml` cites a `todo.md` entry removed in `1426060`.

## A docstring that narrates development history

`shaderbox/document.py::canvas_size_for` explains that the rule "used to be spelled out at four
sites in three syntactic shapes" and names "the bug that started this feature". `CLAUDE.md`
forbids exactly this; it belongs in the commit message or `conventions.md`.
