# 086 revision — READ_ONLY removes the tools and TELLS the copilot why

The maintainer, on the shipped DENY behavior:

> why the fuck the copilot should even see these fucking tools if we set "deny"? The copilot MUST BE
> FULLY AWARE that the user hid the editing tools and that copilot doesn't have access to them...
> stop THESE STUPID WORK-AROUNDS, Copilot must be an intelligent machine, not the fucking dummy bot

He is right, and this supersedes 086 D3's DENY half. **The mode is also RENAMED**, on his follow-up:
*"this should be not 'Deny' we are not denying anything, there is just no edits at all"* — the label
is `Read-only` and the member is `READ_ONLY`. "Deny" named a refusal that, after this revision, never
happens: there is nothing to refuse because there is nothing to call. A name that describes the
mechanism we just deleted would send every future reader looking for it.

What shipped is a guard the model runs into: under DENY the twelve `locks_source` tools stay in
`tools=`, fully described, and each call is
refused after the fact with `error: user declined`. The model cannot plan around a wall it only
discovers by hitting it, cannot explain to the user why an edit did not happen, and pays output
tokens rediscovering the same wall every turn.

## Goal

Under READ_ONLY, the source-writing tools are **absent from the request**, and the copilot **knows they
were withheld and by whom**. Those are one change, not two: hiding the tools alone produces a model
that has quietly lost an ability and says something confused when asked to edit; the prompt fact is
what turns absence into understanding.

| Mode | `tools=` | Prompt | A source call |
|---|---|---|---|
| ALLOW | all | nothing | runs |
| ASK | all | nothing | opens the gate |
| READ_ONLY | **source tools removed** | **one line stating the user set this project read-only** | cannot be made |

## Design decisions

### D1 — the filter lives in `assemble_specs`, the ONE place the tool list is built.

`ToolRegistry.assemble_specs(loaded)` is the single constructor of `tools=` (two call sites in
`run_turn`, both for the same list), and the registry already holds `source_lock` — the field the
gate reads on the worker. So the filter is one predicate in the place that already answers "what
tools does this turn have", not a new seam:

    chosen = [d for d in ... if (d.eager or d.name in loaded)
              and not (self.source_lock is SourceLock.READ_ONLY and d.locks_source)]

**The roster is `locks_source` PLUS the three destructive tools.** The lock's own set (083 D6)
excludes `delete_document` / `delete_pass` / `delete_lib_file` because they already confirm every
time — the right answer for a GATE and the wrong one for a MODE. A project the user called read-only
must not let the copilot delete a shader, which is the most write-like act there is. Render and
publish stay: they produce output and change no source. The whole set is asserted against what each
tool DOES (`mutating`) rather than against a list, so a new source-changing tool joins it or fails.

**`load_tools` must not ADVERTISE them either — the filter alone is not enough.** The lazy-load
path calls `assemble_specs` again, so a withheld tool cannot be loaded; but `load_tools`' own
description carries a CATALOGUE baked at `build_registry` time, before any mode exists. Left alone
it invites the call the filter then refuses: the model loads, is told no, calls anyway, and is
refused again — three wasted turns, structurally invited. The catalogue is rebuilt per mode in
`assemble_specs`, so the offer and the filter cannot disagree.

### D2 — the prompt fact rides `project_context`, at RARE volatility.

The mode is per PROJECT and persisted, so it changes when a project is opened, never mid-turn. That
is exactly `Volatility.RARE`, where the project map and library catalogue already sit — the block
shifts on a project switch and is otherwise part of the cacheable prefix.

One sentence, appended to `_context_block` only under READ_ONLY:

> SOURCE IS READ-ONLY IN THIS PROJECT: the user has turned editing off, so EVERY tool that would
> change or delete this project is unavailable to you — writing shaders and scripts, uniforms,
> passes, documents, canvas size, media, and the delete tools. That is why you cannot see them.
> Reading, grepping, rendering and publishing still work. If asked to change anything, say plainly
> that editing is turned off for this project and that the Allow/Ask/Read-only chip above the chat
> is what changes it. Do not claim a change happened, and do not look for another way to make one.

**The prose must name the same set the filter withholds.** An earlier draft said "shader, script and
pass WRITING tools" while the filter also removed uniforms, documents, canvas size and media — so a
model asked to rename a document would read the notice, conclude renaming was not "writing", and go
hunting. A notice narrower than the roster is worse than none: it invites exactly the call it
cannot serve.

**This is a FACT on the channel the model already reads, not a standing rule** — the distinction the
copilot design skill draws, and the reason this works where a conscience plea would not. It is only
present in the mode it describes, so ALLOW and ASK pay nothing and their prefix is byte-identical to
today's.

### D3 — the cache argument that blocked this was wrong, and saying so is the point.

I defended the shipped design by claiming a mode-dependent tool list would bust prefix caching. That
is false: what the cache needs is a byte-stable block **within a request and across consecutive
turns at the same setting**, which each mode's list is. A READ_ONLY session simply has a smaller
stable list than an ALLOW one.

**The stronger claim — that the mode "cannot change mid-session" — was wrong**, and a review caught
it: the chip was clickable while a turn ran. That is now disabled mid-turn (matching Clear), so the
mode is fixed for the duration of any request that reads it. The correct statement is the narrow
one: **stable within a request**, which is what the cache actually requires, rather than a claim
about sessions that the UI did not enforce.

The real cost of the shipped design is the one I did not count: every turn under READ_ONLY spends output
tokens on calls that cannot succeed, plus a tool result per call, forever.

### D4 — ASK keeps the gate, and keeps every tool. The gate ANSWER stays `DENY`.

`LockAnswer.DENY` is untouched by the rename: answering a gate IS an act of refusal, which is
exactly what "deny" names correctly. Only the MODE changes, because a mode that removes the tools
refuses nothing. Two vocabularies, deliberately not unified.

ASK is the only mode where a per-call question makes sense, and it is unchanged: all tools present,
each source call gated, one `LockAnswer.DENY` latching the turn (085 D1). The gate machinery is not
touched by this revision — what changes is that under READ_ONLY it is now unreachable, because the calls
that would trigger it cannot be made.

**The refuse branch says READ-ONLY, not "user declined".** Reusing 085's decline message here would
tell the model the user refused something they never decided, and it would relay that to the user.
Two facts, two messages: `_DECLINE_MSG` for an answer the user gave, `_READ_ONLY_MSG` for a tool
that is not available. The decline template stays shared between its OWN two branches (gate answer,
turn latch), which is what keeps that vocabulary from drifting.

**A withheld call counts toward the retry cap, and the cap is CHECKED on this path.** The refuse
branch `continue`s before the loop's own cap test, so a model ignoring the notice would otherwise
burn every iteration of the turn on a tool that does not exist — the dummy-bot loop this feature
exists to end. Incrementing the counter alone is a half-fix: the check has to move onto the branch
too, which is what the test caught.

**The refuse branch in the loop STAYS**, and this is deliberate rather than leftover. It is the
belt-and-braces for a source call arriving under READ_ONLY by a path the filter did not cover — a
persisted tool call replayed from history, a future lazy path, a bug. Structural impossibility is
the design; the guard is what makes a hole in it loud instead of silent. Its test keeps driving the
registry predicate directly, so it cannot rot into a branch nothing reaches.

## Files touched

| File | Change |
|---|---|
| `shaderbox/copilot/tools/registry.py` | `assemble_specs` filters `locks_source` tools under READ_ONLY; the `load_tools` handler refuses them with the read-only fact |
| `shaderbox/copilot/prompt.py` | `_context_block` appends the read-only sentence under READ_ONLY; `build_blocks`/`CopilotContext` carry the mode |
| `shaderbox/copilot/prompt_context.py` | `CopilotContext` gains the field, built from the session's lock |
| `tests/test_copilot_source_lock.py` | the absent-tools and present-fact invariants |
| `tests/test_brake_falsifiers.py` | the withheld set equals the `locks_source` roster |

## Verification

1. **Under READ_ONLY the source tools are absent from `tools=`**, and the withheld set equals the
   `locks_source` roster exactly (computed from the registry, not listed). *Falsifier: drop the
   filter → the specs contain `edit_shader`.*
2. **Under ALLOW and ASK the tool list is unchanged** — byte-identical to before this revision, so a
   mode nobody set costs nothing. *Falsifier: filter unconditionally → the ALLOW list shrinks.*
3. **The read-only fact appears in the prompt under READ_ONLY and NOWHERE else.** *Falsifier: append it
   unconditionally → it shows up in an ALLOW context and the assert fires.*
4. **`load_tools` cannot reintroduce a withheld tool**, and says why. *Falsifier: let the lazy path
   bypass the filter → the tool appears in the next iteration's specs.*
5. **The refuse branch still fires if a source call arrives under READ_ONLY anyway** (D4's guard),
   driven through the registry predicate so it stays reachable. *Falsifier: remove the branch → a
   call constructed directly reaches `execute`.*
6. **A model told it cannot edit does not also get a gate**: under READ_ONLY no `AgentGateOpened` is
   emitted for any tool the filter removed.
