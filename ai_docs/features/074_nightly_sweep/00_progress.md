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
