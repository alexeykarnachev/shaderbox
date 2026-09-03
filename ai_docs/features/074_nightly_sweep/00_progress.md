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
