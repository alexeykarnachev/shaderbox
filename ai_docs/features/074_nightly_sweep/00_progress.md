# 074 — progress log

A log of what happened, not a plan. **Append after every wave, before starting the next** — a
file written only at the end does not exist when it is needed. Record what was ruled out and
why, or a resumed session re-litigates a decision already made properly.

On resume: read this file and `git log --oneline -15` FIRST. They are the truth about where the
work stopped; `01_spec.md` is only the plan.

Format per wave:

```
## W-N <name> — DONE | SKIPPED | ABANDONED   <sha>
done-condition: <the checkable statement, written BEFORE the wave started>
did: <what changed>
verification: make gates <green|red>, exit code read unpiped
ruled out: <what was considered and rejected, and why>
surprise: <anything the spec did not predict>
```

---

## Baseline — measured before any wave

Snapshot at `9af08f4`, on branch `dev`. **Re-measure at the start of the night** rather than
trusting these; they exist so a session hitting a red at 3am can tell a wave's damage from a
pre-existing failure.

- `make gates` → exit 0, `GREEN -- check passed, test passed, smoke passed`. Took ~40s.
- `uv run pytest --collect-only -q` → 1700 tests collected.
- Smoke PASSED here because this measurement ran with a display attached. **Overnight it will
  skip instead** (exit 87, gate still green) — that is expected, not a regression.

## Phase 1 — spec prepared

Presence scan by six parallel agents (dead code, duplication, misfiling + layering, file sizes,
comments, safety net). Constraints verified directly by the main session rather than relayed:
the empty `__init__.py`, the gates target's status handling, the path-naming gates, the absence
of a rendering oracle, the visual blind spot.

Ruled out during the scan, with reasons, so no later wave re-opens them:

- **A comment-density sweep.** Attempt-narration comments are ABSENT (a full read of the
  package's 4+-line comment blocks found none), and restating comments are rare. Only the
  verbatim duplicates are worth touching.
- **A misfiling / relocation wave.** Scanned and found absent; many modules carry an explicit
  "leaf module, imports only X" docstring that a reviewer can check mechanically.
- **Splitting `app.py` or `copilot/backend.py` on size.** The repo's conventions record that
  prior extractions already happened and forbid a further split without a fresh pain signal.
- **Backward compatibility of any kind.** The maintainer confirmed explicitly that breaking old
  projects and old saves is fine. No compat shim, no migration, no old-format reader.
- **Anything needing the running app.** No monitor overnight, so the GUI smoke skips on its own
  (exit 87, gate stays green) — no action needed. Coverage stays strong: ~30 test modules drive
  real GL headless, including the GL-lifetime guards; only the real-window frame loop goes
  unchecked. Note per wave which stages ran.
- **Any behaviour work.** No oracle decides rendering correctness, and the visual half cannot be
  checked on this machine.
- **W-2, the case-list wave — STRUCK before the night began.** Adversarial review refuted its
  premise: `_SCRIPT_EDIT_TOOLS` / `_WRITE_TOOLS` partition the `is_edit` universe on two axes the
  registry does not carry, and deriving either regresses the copilot's clean-edit brake silently.
  Do not reinstate it; see the spec's W-2 section for the only defensible alternative.
- **Dependency removal.** Never in scope — every declared dependency is used by a shipped
  feature. Four are reached through the lazy-SDK function-body imports, so a top-level import
  scan calls them unreferenced; that is a scan artifact, not a finding.

---

## Night baseline — re-measured at start of the night

At `91104c7`, branch `dev`, tree clean.

- `make gates` → exit 0, `GREEN -- check passed, test passed, smoke passed`.
  A display WAS attached at this moment, so the smoke actually ran. Later
  waves may report it skipped; that is the expected overnight shape.

## W-0 inventory — done-condition (written before the wave)

`00_inventory.md` exists beside this file, listing every dead-symbol candidate
found, each classified SAFE / CAREFUL / RISKY / NOT-DEAD with the evidence for
its tier, covering all eight symbol kinds the spec names (module functions,
methods, classes, pydantic/dataclass fields, enum members individually, module
constants, private helpers, type aliases individually, whole modules) across
`shaderbox/` and every subpackage, `scripts/`, `tests/`. Tool output recorded.
No source file changed by this wave.

## W-3 comment duplicates — done-condition (written before the wave)

The exact-text duplicate detector run over `shaderbox/` reports zero multi-line
comment blocks appearing more than once, and `make gates` is green. Nothing that
explains a footgun at its own call site is deleted.

### W-3 measurement (done before touching anything)

Re-measured with a duplicate detector over every `.py` under `shaderbox/`
(excluding the vendored `resources/editor/`), at both block and single-line
granularity, plus `scripts/` and `tests/` for the block detector.

- **Multi-line duplicates: exactly one.** `shader_lib/seed.py:139` and `:187` —
  the `root / rel` escape note above two stale-removal loops. This is the one the
  spec seeded, and it is the one copy-paste defect: same file, two adjacent
  functions, the comment carried along with the loop.
- **The spec's "one restating comment" was a FALSE POSITIVE.** It is
  `ui.py:336`'s `# Process hotkeys` above `process_hotkeys(app)` — but that is a
  *section banner label*, part of the `# ----` + label pattern that delimits
  ~14 phases of `update_and_draw` (`:259 Render previews`, `:292 Render
  documents`, `:340 Prepare new frame`, …). Deleting it alone would break the
  pattern; deleting all of them would destroy the file's only navigation aid.
  **No restating comment is removed.**
- **The 4-site imgui note is KEPT, deliberately.** `# Read on the line after the
  input: the item-scoped queries see the LAST submitted item.` appears at
  `popups/lib_picker/tree.py:223,303`, `popups/pass_settings.py:91`,
  `widgets/pass_list.py:204`. Four genuinely different call sites, each with the
  same footgun one line below. This is the spec's keep case — a comment naming a
  failure it prevents, at the site where it prevents it. Hoisting it to one place
  would leave three sites where the reader must go looking.
- **The 5-site EGL fixture note in `tests/` is KEPT.** It quotes a measured
  segfault (module-order-dependent EGL display poisoning); `tests/` is outside
  W-3's scope and the comment is the convention working.
- Section rulers (`# ----`, `# ====`) and the three shared banner labels in
  `exporters/telegram.py` / `exporters/youtube.py` are structure, not prose.

`scanned: multi-line comment blocks and single-line comments >40 chars across
shaderbox/ and every subpackage, plus block-level across scripts/ and tests/;
not scanned: docstrings, non-.py files.`

### W-3 — DONE

did: hoisted the shared half of the two stale-shipped-file loops in
`shader_lib/seed.py` into `_stale_shipped_file`, which carries the escape note
ONCE in its docstring. Each caller keeps its own distinct follow-up (the sync
logs and drops the manifest entry; the reset counts) — the hoist takes only the
question both were asking, so it does not couple two things that must move apart.
verification: `make gates` exit 0 read unpiped, `GREEN -- check passed, test
passed, smoke passed`. **All three stages ran** (a display was still attached).
Mutation-tested the hoisted guard: replacing `if rel_path.is_absolute() or ".."
in rel_path.parts` with `if False` makes
`test_corrupt_manifest_key_cannot_delete_outside_root` fail; restored, 16 passed.
ruled out: deleting one of the two comment copies (both loops need the note —
the duplication was in the CODE, and removing the comment alone would have left
the real defect); the 4-site imgui note (kept, see measurement above); every
section-banner label (structure); the `tests/` EGL note (out of scope, and it
quotes a measurement).
surprise: the spec's second W-3 item — "one restating comment" — did not exist.
It is a section-banner label in `ui.py`, and the file has ~14 of them.

## W-2 replacement (the only sanctioned change in that area) — DONE

done-condition: a test asserts `_SCRIPT_EDIT_TOOLS` and `_WRITE_TOOLS` are
subsets of the registry's `is_edit` set, and it FAILS when a tool is renamed in
one place — verified by mutation, not by reading. The two frozensets themselves
are unchanged.

did: `tests/test_copilot_brakes.py::test_the_brake_tool_sets_name_tools_the_
registry_still_has`. Also covers `_RENDER_AUTHORING_TOOLS`, against the whole
registry rather than the `is_edit` subset — `set_uniform` authors a render
without editing a file, so that set is legitimately broader.
verification: `make gates` exit 0 unpiped, `GREEN -- check passed, test passed,
smoke passed`. Mutation-tested both assertions: renaming `write_shader` inside
`_WRITE_TOOLS` fails at the `_WRITE_TOOLS` line; renaming `set_uniform` inside
`_RENDER_AUTHORING_TOOLS` fails at the authoring line. `agent.py` restored to a
zero-line diff after each.
ruled out: deriving either set from the registry — the spec's struck W-2, and
re-reading the code confirms its refutation (`_edit_target_key` keys the streak
on the artifact axis, `agent.py:1012`'s `tc.name in _WRITE_TOOLS` on the verb
axis, and `registry.is_edit_tool` is already the gate one line above it).

## W-4 ui.py draw code — RECOMMENDATION: LEAVE (no move)

done-condition: a written recommendation answering the spec's three questions,
and — if it says leave — the reason recorded so a later session does not
re-open it.

**Recommendation: leave all seven `_draw_*` functions in `ui.py`.** The doc was
the inaccurate half, and the doc is what changed.

The basis, checked directly rather than argued from the spec:

- **The repo has no layout-module shape to move them INTO.** Every module in
  `tabs/`, `widgets/`, `popups/` is a leaf surface — a small public entry point
  plus private helpers serving it (`tabs/render.py` = `draw` + one private;
  `popups/help.py` = `draw_help` + three; `widgets/pass_list.py` = `draw` +
  five). Not one of them positions siblings, owns a child-region tree, or
  returns geometry for a caller to place other things by. Creating the first
  such module to hold these would invent a shape the repo does not use.
- **`tabs/document.py` is the wrong home specifically.** It draws ONE tab inside
  the document-settings tab bar, and it is one of three peers in `_NODE_TABS`
  (`ui.py:763`). The app panel is the surface CONTAINING that tab bar, three
  levels up (`_draw_app_panel` → `control_panel` child → `_draw_document_settings`
  → tab bar → `document_tab.draw`). Moving the container into a file holding one
  of its grandchildren inverts the containment.
- **The cluster is not separable.** `_draw_app_panel` → `_draw_document_image` →
  `_draw_canvas_backdrop` is one surface split three ways for reading length.
  `_draw_document_image`'s own docstring says why it returns geometry rather
  than drawing self-contained. It also latches per-frame mouse state consumed by
  `app.session.tick(..., mouse=app.script_mouse)` at `ui.py:243` — same file,
  and the ordering hazard between them is only visible in one read.
- **No pain signal.** All seven are private with exactly one caller each, in the
  same file: blast radius is one grep. 817 lines is unremarkable beside
  `app.py`'s 1728 and `ui_primitives.py`'s 1320.

did: fixed `conventions.md`'s "Three-layer UI architecture" bullet instead. It
called `ui.py` a "thin orchestrator owning the frame loop", which described a
file a third smaller than the real one and sent a reader looking for the canvas
viewer in the wrong layer. It now says `ui.py` owns the frame loop AND the
top-level window layout, and says what makes `widgets`/`popups`/`tabs` different
(a leaf draws inside the box it is handed and never positions its siblings) —
so the distinction is checkable rather than a word. `dev_flow.md`'s module map
was ALREADY accurate (it names `_draw_splitter`, `_draw_app_panel`,
`_draw_document_settings` and the left/right split); the two docs now agree.
verification: `make gates` exit 0 unpiped, GREEN, all three stages ran. The
leaf-surface claim was checked against every module in the three directories
before being written down, and softened once: an earlier draft said "one public
entry point", which is false for `popups/lib_picker/filtering.py` and
`search.py` (pure-logic helper modules) and for `widgets/details.py`.
ruled out: moving `_draw_menu_bar` + `_hint` alone — it is the one genuinely
self-contained function of the seven, but moving 38 lines to leave 235 behind
buys nothing and makes the remainder look MORE anomalous. Also ruled out
rewriting the feature specs that cite `ui.py::_draw_app_panel` by name: those
are historical records, and a move would have left the archive permanently
pointing at a file no longer holding the function — a worse doc drift than the
one being fixed.

## W-0 inventory — DONE  1b159f1

done-condition: as written above. Met — `00_inventory.md` exists with per-symbol
evidence, the kept list, and a coverage claim per scan.
did: six parallel sonnet agents, one per symbol kind (module functions+classes,
methods, enum members + type aliases, constants + fields, whole modules +
scripts/tests, private helpers), plus ruff and vulture first. The main session
closed the one gap an agent declared in its own coverage claim (private CLASSES,
which its `def`/module-var pattern could not match — all 51 checked, all live).
verification: measurement only, no source file changed by this wave.
surprise: the repo is far cleaner than the spec's seeds implied. Zero dead
functions, classes, modules, private helpers, enum members or type aliases. The
one category the tools cannot see — write-only dataclass FIELDS — is where every
real find was, and the spec did not predict that category at all.
ruled out: the 14 "dead" enum members in `editor/ffi.py` (the deliberate ABI
mirror `conventions.md` warns a sweep will re-find); the ~8 dogfood harness
methods; every pydantic validator and `model_config`.

## W-1 deletion — DONE  1b159f1

done-condition: every SAFE- and CAREFUL-tier symbol from the inventory is gone,
`make gates` is green with a test count no lower than the 1700 baseline, and
every kept symbol has its reason recorded.
did: removed 1 constant, 1 method, 8 fields (see the inventory table), plus the
`kind` parameter that the `PublishResult.kind` removal orphaned.
verification: `make gates` exit 0 read unpiped, `GREEN -- check passed, test
passed, smoke passed`. All three stages ran — a display is attached tonight, so
the smoke did NOT skip. 1701 tests collected (1700 baseline + the W-2 subset
test).
ruled out: `ScriptError.uniform_name`, `PromptBlock.name`,
`CompletionProvider.name` — kept, with reasons, in the inventory. Also
`shaderbox/resources/textures/default.jpeg`: unreferenced now, but deleting a
resource file is out of this sweep's scope, so it is reported, not removed.
surprise: the gate went red once, on three tests, and every one was CONSTRUCTING
a removed field while asserting on something else. Fixed at the call sites in one
attempt with no assertion weakened. One of them (`fake_publish` in
`test_youtube_exporter.py`) stubs `_copilot_publish` and so mirrors its
signature — which is how the orphaned `kind` parameter announced itself.

## W-5 sanitize — DONE

done-condition: the roadmap banner and a 074 row describe the landed sweep, no
doc names a symbol this sweep removed, `todo.md` is unchanged unless the sweep
actually resolved an entry, and `make gates` is green.
did: rewrote the Active-context banner (200 words, at the file's own cap) and
added the 074 row. Checked `ai_docs/` and the skills for references to every
removed symbol — there were NONE outside this feature's own files, so no doc
needed repair.
verification: `make gates` exit 0 unpiped, GREEN, all three stages.
ruled out: touching `todo.md`. Its single open entry is the live-only UI check
list, whose trigger (a `make run` with a display) this sweep does not fire. One
of its items — a deflected publish rendering a neutral "handed off" line — rides
the same handoff card path W-1 edited; `test_handoff_card_reads_as_handed_off_
not_failed` covers it and passes, but the LOOK of the line is still a display
check and stays outstanding.

## Adversarial review of the landed sweep — PASS, two doc fixes

An opus reviewer was pointed at the five wave commits and told to anchor every
finding to something this session did not write: the code at `91104c7`, the
suite's behaviour when run, the pre-sweep docs, and features 052/069/072.

**Verdict: zero code defects.** Each of the ten removed symbols was re-checked
per-symbol at the baseline tree for any reader at all — including `getattr`
(30 dynamic-access sites swept, none touching them), `asdict`/`replace`/
`model_dump`/`**` splat, and every JSON on disk. None had one. The three edited
tests were diffed function-body-wide: every `assert` line is byte-identical and
only constructor argument lists moved. The `seed.py` hoist was verified by
differential execution over 27 case shapes (missing file, hash mismatch, `..`
and absolute escapes, `sub/../` normalisation) — zero divergences. Independently,
this session ran its own 40-case differential and also got zero.

Two DOC defects found, both now fixed:

1. **`conventions.md:577` documented `AgentToolCard.display` as a live "THIRD
   channel"** — a terse summary the chat shows instead of the heavy `msg`. That
   claim was ALREADY FALSE before the sweep: `58018f8` (June) replaced
   `shown = ev.display or ev.result` with `_tool_card_line`, which reads only
   `ok` and `payload`. Verified at the source with `git show 58018f8`. W-1's
   removal turned a false claim into a dangling symbol reference. The bullet now
   records that there is no third channel, that 020·23's mechanism lost its
   reader in `58018f8`, and that `read_shader` still produces a
   `payload["display"]` key nothing consumes. **This also answers the behaviour
   question W-0 escalated:** the terse line did not regress tonight — it stopped
   rendering three months ago, so the user sees exactly what they saw before.
2. **The leaf-surface rule W-4 wrote had a real counterexample.** The copilot
   chat is its OWN top-level window (with the cheatsheet), drawn after the
   full-screen window closes, and `_apply_layout` anchors it to `app.editor_rect`
   every frame — so it does read the layout's geometry. The bullet now names both
   windows and that one exception, rather than stating an absolute the code
   breaks. (An earlier pass had already loosened "draws inside the box handed to
   it", which reads as banning the internal column layout `tabs/render.py` does.)

ruled out by the reviewer and not to be re-checked: the trace log (no card
serialization), `_publish_result`'s body, `read_lib`'s handler, the archive
conversation JSONs (their `"kind"` is `ResultWidget.kind`, a live field on an
untouched type), and `7b2352d` (test-only, cannot change behaviour).

## Follow-up — `default.jpeg` removed (maintainer asked)

The inventory reported the asset as unreferenced but scoped its deletion out; the
maintainer then asked for it. Removed, and `shaderbox/resources/textures/` went
with it as its only file.

What stands in its place, since the question came up: feature 072's `AutoSource`
marker plus a 1x1 black texture. An unbound sampler holds `AutoSource()` ("nobody
decided about this; its NAME fills it at bind time, or it reads black") instead of
a `MediaWithTexture` pointing at the photo, and `Pass._black_texture` (`core.py`)
binds one lazily-made 1x1 opaque black, released with the pass, for any value that
is not a texture. The photo was never actually DRAWN even before 072 — the document
binder already seeded black for every unbound sampler — so it only ever existed as
an in-memory sentinel that `is_default_image` compared paths against. 072 replaced
that path comparison with a real marker type, which is what left the file orphaned.

checked before removing: no reader by any route (`.py`, `.toml`, `.sh`, Makefile,
`build.sh`); packaging picks resources up by the wildcard
`include = ["shaderbox/resources/**/*"]` and names no file; and no LIVING doc
describes a default-image mechanism in the present tense — the only mentions are
in features 052 and 069's specs, which are historical records and stay as written.
verification: `make gates` exit 0 unpiped, GREEN, all three stages. The smoke
opened a real window and ran the frame loop with the asset gone, which is the
check that matters for a missing resource.
