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
- **Any behaviour work.** No oracle decides rendering correctness, and the visual half cannot be
  checked on this machine.
