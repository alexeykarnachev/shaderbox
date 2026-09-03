# 071 W-E — Spelling sweep and gate (D3)

Parent: `01_spec.md § W-E`. Mechanical; no reviewer. The maintainer's words: "i prefer this:
colour -> color".

## What landed

One case-preserving sweep (`colour` → `color`, `Colour` → `Color`, `COLOUR` → `COLOR`) over 29
files: the package (comments, docstrings, the copilot prompt and its tool descriptions, the Help
panel content, the shipped example shaders and one example description, the host editor
binding), the tests (two test NAMES among them: `test_float_export_keeps_the_true_colors`,
`test_a_marker_text_color_reaches_the_glyph_at_column_0`, and the `conventions.md` line that
cites the second by name moved with it), `README.md`, `CLAUDE.md`, the living docs
(`conventions.md`, `dev_flow.md`), both skills, the tutorial body (then `tutorial.html`
regenerated), one `projects/dev` document description and one tracked document under
`projects/documents/` (data, hand-swept, no migration code). Identifiers were already `color`;
no code symbol changed except the two test names.

Left alone, on purpose: the vendored editor set under `shaderbox/resources/editor/` (upstream's
files, re-copied whole on every re-vendor, so a local edit would only drift); the feature
records under `ai_docs/features/` other than the tutorial body (they quote what was said,
including the maintainer's "colour -> color" itself); the dogfood run transcripts (what a model
wrote).

## The rule and its gate

`conventions.md ## Code rules` gains one line: American spelling wherever a reader sees words.
`tests/test_prose_spelling.py` walks every tracked text file minus the three exclusions above
and fails on any hit, listing file, line and text; a second test pins that the roster still
covers the package, the living docs and the tutorial body, and still excludes the vendored set,
so the gate cannot silently shrink to nothing. Shipped in the same commit as the sweep.
