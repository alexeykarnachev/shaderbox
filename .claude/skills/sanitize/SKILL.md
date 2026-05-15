---
name: sanitize
description: "Closing-out sweep after any noticeable work — walk todo.md, fix stale doc refs, update stale docs, append a worklog.md entry, run the cold-context check. Triggers: 'sanitize', 'cleanup', 'wrap up', 'close out', 'before you finish', 'update worklog', 'tidy the docs', 'санитайз', 'почисти', 'обнови worklog', 'перед тем как закончить'."
user_invocable: true
---

<command-name>sanitize</command-name>

The closing-out pass after any noticeable work (a fix, a feature, a refactor, an investigation) — so
the repo's state stays clean and a fresh agent with `/clear`-ed context doesn't get lost. Invoke by
hand (`/sanitize`) or do it yourself before saying "done". Not a regiment — a habit that keeps the
repo resumable. For a trivial mechanical change, steps 5-6 (worklog entry + cold-context glance) are
enough; for a feature, the full pass.

Full picture of the dev flow: `ai_docs/dev_flow.md`. The cold-context rule: `CLAUDE.md ## Two meta-rules`.

## 1. `ai_docs/todo.md` — walk it end-to-end

Each entry: check `Trigger` against what was just touched. **Fired** → resolve it inline in this same
commit, OR sharpen the trigger with a concrete condition if it's not the moment. **Resolved** (bug
fixed / deferral closed) → **delete the entry** in the same commit as the fix. If the work spawned a
new deferred fix or a "here be dragons" — **add an entry** with a concrete trigger (use the format +
the good-trigger test in the `todo.md` header).

## 2. Stale refs in touched docs

For the docs the work touched (or that describe the changed code): confirm markdown links and
`@file` refs resolve, nothing points at renamed/deleted files or skills. Quick sweep:
`grep -rn` for the names of anything you renamed/deleted, across `CLAUDE.md` + `ai_docs/` + `.claude/`.

## 3. Stale docs — update in the same wave

Behavior changed → update the doc that describes it, **in the same commit** (a stale rule referencing
a symbol that no longer exists is worse than silence):

- The module map in `dev_flow.md ## Recipes` — if a module moved/split or a new one landed.
- `CLAUDE.md` — if a skill was added/removed, or the structure changed materially.
- `conventions.md`:
  - `## Design decisions` — for a new cross-cutting choice ("we decided X, revisit if Y").
  - `## Known quirks` — if you hit a library/SDK footgun with a workaround.
- Numbers too (a spec says "5 uniform types"; you added one → "6").

(The resumption backlog lives in the worklog `open thread:` line — that's step 5.)

## 4. `ai_docs/conventions.md` — walk it end-to-end

Each bullet still match landed code? Each `## Design decisions` bullet still constrain *future* code
rather than narrate a one-off implementation choice? → delete the narrators; move SDK footguns to
`## Known quirks`; the file grows only when a new rule would have prevented a real mistake.

## 5. `ai_docs/worklog.md` — add an entry

New entry **at the top**, the header's format. Terse — a few lines. If the work landed something on a
branch (not the current one) or opened/closed an MR — that's in `refs`. `open thread:` = what a fresh
agent should pick up — this is also where the ordered resumption backlog lives; if this work cleared
a backlog item or reshuffled the order, update it here.

## 6. Cold-context check (+ re-read `CLAUDE.md`)

Before suggesting the user `/clear` (or just before "done" on a sizeable chunk): glance over
`CLAUDE.md` (the chain anchor — structure, skills list, hard rules), then imagine a fresh agent
asking "what's next?" — walk `CLAUDE.md` → `worklog.md`/`todo.md` → `dev_flow.md` and confirm it
lands on the same next-action with the same blockers in a few reads. If not, the missing context IS
the bug — write it to a file (worklog entry / `open thread:` line / todo trigger), not the chat.
Also: is `worklog.md`'s top entry recent vs `git log -1`? If not, reconstruct from `git log` and add
the missing entry.

## 7. Tooling-improvement glance (optional, droppable if `/sanitize` feels heavy)

Look back at the work just done: did something generic surface that should be baked into a skill/doc
so next time is smoother? User corrected you twice on the same thing → maybe a missing rule/example.
Don't invent improvements; "nothing this session" is a fine answer.

## Summary

End with this, verbatim:
```
SANITIZE:
- todo.md: <N walked; X resolved / Y added / "no drift">
- stale refs/docs: <"clean" / fixed: ...>
- conventions.md: <bullets confirmed; noise audit findings or "walked, no drift">
- worklog: <entry added / "no significant work">
- cold-context: <ok / patched: ...>
- tooling: <nothing this session / suggested: ... / applied: ...>
```

Then one final line — `/clear` readiness:
- Ready: **Ready to `/clear`** — fresh agent resumes from: <quote open-thread verbatim>.
- Not ready: **Not ready to `/clear`** — <one-phrase blocker>.

## What NOT to do

- Don't rewrite docs wider than the work requires.
- Don't delete `todo.md` entries whose fix isn't in this same commit.
- Don't commit autonomously — that's the user's call (unless they've given a standing instruction).
- Don't give long explanations — terse.

> Want this to run automatically (a Stop-hook)? Set it up via the `update-config` skill; by default
> it's a manual/voluntary step.
