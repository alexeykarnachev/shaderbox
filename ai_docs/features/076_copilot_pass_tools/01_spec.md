# 076 — copilot pass tools

The sub-feature the first station experiment (`dogfood/runs/rc_full_build`, attempt 1) was
blocked on, built under 075's stuck rule ("something that BLOCKS the run → file a sub-feature,
spec it, implement it, fix it, then re-run"). Not plan-locked by the maintainer: he was walking the
RC tutorial and had said to do whatever the run needed; this spec records what was decided so he
can overrule it.

## Goal

The copilot can author a multi-pass document without a human at the pass list: add a pass,
configure its runs per frame / target / output, rename or delete it. 065 stage 8 shipped the
`<id>#<pass>` address and the pass-aware working set "with no new tools", and its check 16 ("the
copilot authors a two-pass document") was never run; asked for two passes, the copilot put two
`main()`s in one file and ran out its time budget (attempt 1, turn 1).

## Out of scope

- **Wiring a sampler by tool.** The name rule (`u_<pass>` reads that pass, `u_prev` its own
  previous run/frame) covers every RC edge, and the working set's sampler row says what each
  reads. *Trigger: a build that needs a sampler named differently from its source.*
- **Revert coverage of a pass add/delete beyond what `_capture_document` already snapshots.**
  *Trigger: a revert after `delete_pass` that leaves the document without the pass.*

## Design decisions

1. **Three lazy tools over the session's existing verbs, not a new mutation path.** `add_pass`,
   `set_pass`, `delete_pass` (`copilot/tools/passes.py`) call `ProjectSession.add_pass` /
   `set_pass_target` / `set_pass_iterations` / `set_output_pass` / `rename_pass` / `delete_pass`
   through the backend on the main thread, so a copilot-made pass is indistinguishable from a
   panel-made one on disk. Lazy, like 052's document-file ops — the long tail — with one
   pre-action line in the static prompt (NODES) naming `add_pass` and the one-pass-one-shader
   rule, since a rule that governs what the model does INSTEAD of acting must be in the stream
   before it acts.
2. **`add_pass` takes the configuration in the same call.** Runs, dtype, scale, filter, wrap and
   the output mark are optional on add, so a jump flood is one call, not three; `set_pass` changes
   only the fields given. A bad dtype, an out-of-range run count and a bad name are loud errors —
   the session verbs reject rather than clamp, and the tool relays the message.
3. **Every result echoes the document's pass table.** The model reads back the state it just
   changed (name, runs, target, the output mark, the `<id>#<name>` edit address) — corollary 2 of
   the actor model: a fact not in the stream does not exist.
4. **`delete_pass` is gated** (`GatePolicy.ALWAYS`), like every destructive tool; the session
   refuses to delete the last pass.
5. **A bare `document` means the current document**, as the edit tools read a bare target.

## Files touched

`copilot/capabilities.py` (`PassOpResult` + three protocol methods), `copilot/backend.py`
(six verb seams in the constructor, `add_pass` / `set_pass` / `delete_pass`, `_pass_table`),
`copilot/tools/passes.py` (new), `copilot/tools/registry.py`, `copilot/prompt.py` (one NODES
line), `project_session.py` (wires the verbs), `pass_graph.py` (`TargetDtype` / `TARGET_DTYPES`),
`scripts/dogfood/analyze.py` (`CANONICAL_TOOLS`), `tests/_caps.py`,
`tests/test_copilot_pass_tools.py`.

## Manual verification

- `tests/test_copilot_pass_tools.py`: a pass added by the tool exists on disk and in `graph.json`
  with the runs and target given; `set_pass` changes only what is given and a rename carries the
  file; rejections are errors; the last pass cannot be deleted; the registry exposes all three
  lazily with delete gated, and an `execute` reaches the capability.
- Live: `rc_full_build` attempt 2 on the station — the copilot creates `paint` and `seed` as two
  passes on the first ask.

## Open questions for the user

- Eager instead of lazy for `add_pass`? Every eager description is billed per iteration; the
  lazy catalogue names it and the static prompt line points at it. Decided lazy; revisit if a run
  shows the model failing to load it.
