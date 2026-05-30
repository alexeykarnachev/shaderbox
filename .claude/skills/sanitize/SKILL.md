---
name: sanitize
description: "Closing-out sweep after any noticeable work — walk todo.md + conventions.md, audit the code against them, fix stale doc refs, update the roadmap banner + rows, run the cold-context check, end with a sweep-report table. Triggers: 'sanitize', 'cleanup', 'wrap up', 'close out', 'before you finish', 'update roadmap', 'tidy the docs', 'санитайз', 'почисти', 'обнови roadmap', 'перед тем как закончить'."
user_invocable: true
---

<command-name>sanitize</command-name>

The closing-out pass after any noticeable work (a fix, a feature, a refactor, an investigation) — so
the repo's state stays clean and a fresh agent with `/clear`-ed context doesn't get lost. Invoke by
hand (`/sanitize`) or do it yourself before saying "done". For a trivial mechanical change, steps 5-6
(roadmap banner + cold-context glance) are enough; for a feature, **execute every numbered step and
end with the sweep-report table** — a missing row means the step wasn't done.

Full picture of the dev flow: `ai_docs/dev_flow.md`. The documentation-discipline rules these steps
enforce: `dev_flow.md ## Documentation discipline`. The cold-context rule: `CLAUDE.md ## Two meta-rules`.

## 1. `ai_docs/todo.md` — walk it end-to-end

Each entry: check `Trigger` against what was just touched. **Fired** → resolve it inline in this same
commit, OR sharpen the trigger with a concrete condition if it's not the moment. **Resolved** (bug
fixed / deferral closed) → **delete the entry** in the same commit as the fix (no `[RESOLVED]` header
— git history is authoritative). New deferred fix / "here be dragons" → **add an entry** with a
concrete trigger (format + good-trigger test in the `todo.md` header).

## 2. `ai_docs/conventions.md` — walk it end-to-end

Each section: stale line refs, outdated "after feature N" notes, superseded rules? Each bullet still
match landed code? **Noise audit (mandatory):** every `## Design decisions` bullet must constrain
*future unwritten code*, not narrate a one-off implementation choice — narrators get deleted or moved
to a code comment; SDK/library footguns go to `## Known quirks`, not `## Design decisions`; pure
code/editing rules belong in `## Code rules`. The file grows only when a new rule would have
prevented a real mistake.

## 3. Convention audit — reasoning, not grep

The deliberate "stop and look around" moment. For each invariant in `conventions.md` (type
annotations, imports-at-top, the comment rule, the suppression allowlist, the three-layer split,
theme-token usage, popup-mutex), walk the relevant files and check whether the code actually
satisfies it. Scope: every file the work touched + every file hosting an invariant the work affects +
a random spot-check of files the work did NOT touch. **Comment-noise check is part of this:** flag
multi-line comments that narrate development history (the bug-we-hit story, the why-we-changed-it
saga) rather than state a now-fact — compress to ≤1 line + canonical-home pointer, or delete.
Findings → fix inline OR file as a `todo.md` deferral with a trigger.

## 4. Stale refs in touched docs

For docs the work touched (or that describe the changed code): markdown links + `@file` refs resolve;
nothing points at renamed/deleted files or skills. Sweep: `grep -rn` for the names of anything
renamed/deleted across `CLAUDE.md` + `ai_docs/` + `.claude/`. The module map in
`dev_flow.md ## Recipes` matches the actual module layout.

## 4.5. Drift-prone artifact sweep (this work + a quick scan of the docs it touched)

Did the work introduce — or leave nearby — a **fragile artifact that rots on the next edit and forces
a pointless re-sync**? Hunt the whole class in `todo.md` / specs / `conventions.md` / `roadmap.md`
banner / code comments:
- **Raw line numbers** — `file.py:NN`, "line 47", a line range. Replace with the **symbol**
  (`App.save`, `mod.py::fn` — greppable, survives edits). (Banned by `conventions.md ## Code rules`.)
- **File-length / live counts** — "`foo.py` (800 L)", "crosses ~900 L", "5 uniform types", "N tabs".
  These are wrong the moment anyone touches the thing. Drop the number, or express a size/count
  *trigger* as a qualitative condition ("when it grows a separable cluster", "when editing feels
  unwieldy"). A current count that must update every edit is the smell.
- **"as of" / per-turn state** that wasn't refreshed (the banner date vs `git log -1`).

**EXEMPT — frozen historical evidence does NOT drift, leave it:** a completed refactor's before/after
(`ui.py 1508 → 294`), "refactored X → Y", "feature N replaced Z", a dated decision record. These
describe a past event that can't change, so the number is permanent, not live. The test: *will this
number/ref be wrong after an unrelated edit next week?* Yes → fix it. No (it's history) → keep it.

## 5. `ai_docs/roadmap.md` — update the banner + rows

**Rewrite** the Active-context banner in full if "what's next" changed (do NOT append; ≤200 words;
date stamp = today). Add or flip the feature's row (status from the vocabulary + `Spec:` pointer).
Keep the row to ONE line — the story stays in the feature spec / commit message. If this work cleared
a queued feature or reshuffled priorities, that shows in the banner, not in a row narrative.

## 6. Cold-context check (+ re-read `CLAUDE.md` + this file)

Re-read `CLAUDE.md` (chain anchor — structure, skills list, hard rules, the documentation-discipline
list) and skim `dev_flow.md` for a step that drifted because the just-landed work exposed a gap. Then
imagine a fresh agent asking "what's next?" — walk `CLAUDE.md` → `roadmap.md`/`todo.md` →
`dev_flow.md` and confirm it lands on the same next-action with the same blockers in a few reads. If
not, the missing context IS the bug — write it to a file (banner / todo trigger), not the chat. Is
the roadmap banner's date recent vs `git log -1`? If not, reconstruct from `git log` and rewrite it.

## 7. Tooling-improvement glance (optional, droppable if `/sanitize` feels heavy)

Did something generic surface that should be baked into a skill/doc so next time is smoother? User
corrected you twice on the same thing → maybe a missing rule/example. Don't invent improvements;
"nothing this session" is a fine answer.

## Sweep report (mandatory for a feature — paste at the end, verbatim)

```
SANITIZE:
| Step | Outcome |
|---|---|
| 1. todo.md walk        | <N walked; X resolved / Y added / "no drift"> |
| 2. conventions.md walk | <bullets confirmed; noise-audit findings or "walked, no drift"> |
| 3. convention audit    | <invariants checked + findings, incl. comment-noise> |
| 4. stale refs/docs     | <"clean" / fixed: ...> |
| 4.5 drift-prone arts   | <line-nums/counts found+fixed, or "none"; frozen-history left as-is> |
| 5. roadmap             | <banner rewritten? row added/flipped? / "no significant work"> |
| 6. cold-context        | <ok / patched: ...> |
| 7. tooling             | <nothing this session / suggested: ... / applied: ...> |
```

A sweep without the full table is incomplete. Then one final line — `/clear` readiness:
- Ready: **Ready to `/clear`** — fresh agent resumes from: <quote the banner's next-action verbatim>.
- Not ready: **Not ready to `/clear`** — <one-phrase blocker>.

## What NOT to do

- Don't rewrite docs wider than the work requires.
- Don't delete `todo.md` entries whose fix isn't in this same commit.
- Don't append to the roadmap banner — rewrite it in full.
- Don't commit autonomously — that's the user's call (unless they've given a standing instruction).
- Don't give long explanations — terse.

> Want this to run automatically (a Stop-hook)? Set it up via the `update-config` skill; by default
> it's a manual/voluntary step.
