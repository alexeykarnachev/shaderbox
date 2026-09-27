# Hot reload and simulation state — the corner cases

The maintainer's steer: "we can restart the full shader when the script is modified. Or we
can support both modes, I don't know, we need to think about this and about the corner
cases." This enumerates them so the decision is made against the list rather than the
headline.

## What happens today

A script recompile makes a FRESH behavior instance — state resets on edit, by design, and
`ScriptEngine.reload` keys on `script.py`'s mtime. A SHADER edit does not touch the script
instance. So the two halves of one document already reload asymmetrically, and the
asymmetry is invisible until the script owns something expensive.

With 50k entities that asymmetry becomes the defining fact of the workflow: editing a
tuning constant restarts the world, editing the fragment shader does not.

## The cases a decision has to answer

1. **A whitespace or comment edit.** The mtime moves, the behaviour is identical, the world
   dies. This is the case that makes "always restart" feel broken in practice.

2. **An edit that changes the entity RECORD's shape** — a field added, a dtype changed,
   capacity raised. Carrying state across this is not merely undesirable, it is unsound:
   the old array does not fit the new record. A restart is the only correct answer here, so
   any "preserve" mode needs to detect this case and fall back.

3. **An edit that changes only a rule** — a force constant, a vision angle. Carrying state
   is exactly what the user wants: tune while watching, the way a shader uniform behaves.

4. **An edit that is syntactically broken.** The instance freezes at the compile error. Does
   the world keep its last state (so a fixed typo resumes), or is it already gone? Today the
   old instance is kept until a good compile replaces it, so state survives a broken edit
   and dies on the FIX — the worst ordering, and the one most likely to be reported as a bug.

5. **`__init__` is where seeding lives.** Preserving state across a reload means `__init__`
   does NOT run, so a script that changes its spawn logic sees no effect until an explicit
   reset. The user then has two mental models of "restart" and no signal for which applies.

6. **Capacity changes under a live buffer.** The engine owns the buffer at capacity; a
   reload that raises capacity must reallocate, which is a GL lifetime event mid-frame.

7. **Export.** Export ticks a fresh instance by design, so export is always case "restart".
   A preserve mode must not leak live state into an export or the exported video stops being
   reproducible.

8. **Project switch and document close.** Both must release the buffer; a preserved-state
   mode adds a lifetime whose end is less obvious than a fresh instance's.

9. **The copilot writes scripts too.** A copilot edit is a script edit. If a write restarts
   the world, an agent tuning a constant has the same destructive effect as the user, and
   its dry-run probe must not.

## The shapes worth considering

- **A. Always restart.** One mental model, `__init__` always runs, no soundness questions.
  Cost: cases 1 and 3, which are the common edits.
- **B. Always preserve, reset on an explicit verb.** Tuning is live. Cost: case 2 is unsound
  and needs detection anyway, case 5 confuses seeding, and the reset verb becomes load-bearing.
- **C. Preserve when the record's shape is unchanged, restart otherwise.** Answers case 2 by
  construction. Cost: the rule is invisible — the same gesture does different things and the
  user cannot tell which without a signal in the UI.
- **D. The script declares it.** A class attribute or a decorator saying whether state
  survives a reload. Explicit, no hidden rule, and it makes case 5 the author's call. Cost:
  one more thing to teach, and a default still has to be chosen.

## What a decision needs, whichever shape wins

- A visible signal of which happened — a restart is a world-destroying event and should not
  be silent.
- Case 2 detected rather than trusted, since carrying a mismatched array is memory corruption
  wearing a plausible picture.
- Case 4's ordering fixed regardless: state dying on the FIX rather than on the BREAK is
  wrong under every shape.
- Export pinned to restart, per the reproducibility rule.
