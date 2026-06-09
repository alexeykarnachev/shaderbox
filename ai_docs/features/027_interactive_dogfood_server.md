# 027 — Interactive dogfood server (DRAFT — raw idea capture)

> **STATUS: ROUGH DRAFT, NOT PLAN-LOCKED.** This is base-idea capture so a fresh `/clear`-ed session
> knows where to continue. The real spec (design decisions, files-touched, review) gets written +
> reviewed by a dedicated agent pass per `dev_flow.md` feature flow. Treat everything below as a
> starting proposal, not a locked decision.

## Why (the problem this fixes)

The dogfood scenarios in `ai_docs/scenarios/` are **free-form goals + a `Human check:`**, NOT fixed
dialogues. Dogfooding tests whether the driver (Claude) READS each copilot reply and ADAPTS the next
message. A pre-scripted `h.send(...)` sequence replays a recording and defeats the point (hit
2026-06-09: a rigid `dogfood_scenarios.py` was written then deleted; the `/dogfood` skill §1 had told
it to write a throwaway driver).

But true interactivity needs the harness STATE (the copilot worker thread, the EGL context, the
loaded project, the conversation history) to PERSIST across the gaps where Claude stops to read a
reply and think. Two failed shapes:
- **One `python -c` per turn** — loses all state every process exit (worker/EGL/project gone). Dead end.
- **A background process Claude juggles** — Claude must spawn it, track its PID, poll it, not lose it
  across turns. Fragile + clutters the session with long-lived background tasks.

The goal interface: **Claude sends one message and BLOCKS until the copilot's reply comes back** — no
background process to manage, no per-turn respawn. A foreground blocking round-trip.

## Core idea (leading shape — TBD by the real spec)

A long-running **dogfood SERVER process** that owns ONE `DogfoodHarness` for its whole life and
exposes a **blocking request/reply** interface over the filesystem (no sockets, no ports):

1. Claude STARTS the server ONCE (a foreground `Bash` call that the harness keeps alive — OR a
   `run_in_background` Bash task; the interface is designed so Claude does NOT have to poll it).
2. Per turn, Claude writes a REQUEST (the user message + any control: render / approve-gate / decline
   / reload / which-node) to a known path, then issues ONE blocking read that returns only when the
   server has finished the turn and written the REPLY (printed events + final assistant text + the
   trace tail + the render path if asked).
3. The server loops: read request -> drive one copilot turn (`send` + `drive_until_idle`) -> write
   reply -> block on the next request.

The blocking is the crux: Claude's "send + wait" must be a SINGLE tool call that returns the reply,
so from Claude's side it feels synchronous (send a line, get the answer), with zero process juggling.

### Candidate transport mechanisms (the real spec picks one — open question)

- **A: named pipe (FIFO) pair.** `mkfifo` req + resp. Claude `echo msg > req_fifo; cat resp_fifo` in
  ONE bash call — `cat` on the resp FIFO blocks until the server writes. Cleanest "blocking round-trip
  in one call", no polling. Risk: FIFO semantics / partial reads / a hung server blocks `cat` forever
  (needs a timeout).
- **B: request file + reply file + a blocking wait.** Claude writes `req.json`, then runs a tiny
  blocking waiter (`while [ ! -f resp.json ]; do sleep 0.1; done; cat resp.json; rm resp.json`) in one
  bash call. Simpler to reason about than FIFOs; the waiter is the blocking primitive. Server watches
  for `req.json`, deletes it, processes, writes `resp.json`.
- **C: a thin CLI the server exposes** — `uv run python scripts/dogfood_client.py send "msg"` blocks
  and prints the reply; the client talks to the server over A or B internally. Nicest ergonomics for
  Claude (one obvious command per turn), hides the transport. Probably the eventual shape.

Lean: **C wrapping B** (a `dogfood_client.py send/render/approve/decline/reload/stop` CLI over
request-file + blocking-wait). Decide in the real spec.

### What the server must expose (maps to current harness methods)

- `send(user_text)` -> drive one turn -> reply = the new chat messages + final assistant text +
  per-turn cost/tokens (already on `state.last_turn`). (`DogfoodHarness.send` + `drive_until_idle`.)
- **Gate handling**: `drive_until_idle` STOPS on an open gate and returns it. So a reply must be able
  to say "BLOCKED ON GATE: <text>" and the next request answers `approve` / `decline`
  (`h.approve()` / `h.decline()` already exist). The server must NOT auto-approve (the human/Claude
  decides per the scenario — gate-decline is itself a scenario).
- `render(node?, size)` -> returns the PNG path; Claude then `Read`s it. (`DogfoodHarness.render`.)
- `nodes()` -> the id->name map for picking targets / verifying targeting.
- `reload()` — NEW (also a separate `todo.md` deferral): re-run the App conversation-load sequence so
  persistence/restart scenarios become drivable. Folds that deferral in.
- `trace_path` / `session_cost_usd` in every reply footer.
- `stop()` — clean shutdown (`release()` the worker + EGL).

## Out of scope (each a trigger for later)

- Auto-judging / assertions — NO. The judge stays the human reading replies + PNGs. The server only
  transports turns; it never decides pass/fail.
- A GUI / TUI — NO. Filesystem transport only.
- Multi-session / concurrent harnesses — NO. One server = one harness = one project at a time.
- Replacing `scripts/dogfood.py` — NO. The server WRAPS the existing `DogfoodHarness`; the harness
  stays the engine.

## Risks / open questions for the real spec

1. **Blocking-read timeout.** If a turn hangs (a stuck worker, a never-terminal gate), Claude's
   blocking read must time out with a clear message, not hang the session forever. What bound? (turns
   are usually < 30s; a render up to ~60s; publish up to 300s.)
2. **Server liveness across Claude turns.** Does Claude start it with `run_in_background: true` (a
   harness task that survives) or a foreground call that returns a handle? The transport must make
   Claude's per-turn interaction a SINGLE blocking call regardless.
3. **Gate round-trips inside one logical "turn".** A single user message can open multiple gates
   (e.g. a publish that needs token then pack). The protocol must allow N gate exchanges before the
   turn's final reply — reply types: `done` | `gate(text)` | `error`.
4. **Crash recovery.** If the server dies mid-session, Claude must detect it (stale/no reply) and be
   able to restart cleanly (fresh project — acceptable; the scenarios are short).
5. **Stdout noise.** The harness prints a lot (loguru DEBUG, GLFW warnings). The reply must be a CLEAN
   structured payload (JSON?), separate from the server's own log stream, so Claude parses signal not
   noise.

## Files touched (anticipated — confirm in real spec)

- NEW `scripts/dogfood_server.py` — the long-running server loop owning the harness + the transport.
- NEW `scripts/dogfood_client.py` (if shape C) — the blocking per-turn CLI.
- `scripts/dogfood.py` — likely add `reload()` to `DogfoodHarness` (folds the reload deferral); maybe
  expose a structured (not just printed) turn-result so the server returns data, not scraped stdout.
- `.claude/skills/dogfood/SKILL.md` §1 — rewrite to mandate the server + forbid pre-scripted drivers
  (the §1 warning added 2026-06-09 is the interim placeholder).
- `ai_docs/todo.md` — delete the "dogfood must be driven INTERACTIVELY" deferral + the "reload /
  gate-decline" deferral once this lands (in the same commit).

## Manual verification (anticipated)

- Start the server, drive scenario 02 (targeting) turn-by-turn ADAPTIVELY — send a message, read the
  reply, choose the next message based on what the copilot actually did. Confirm state persists across
  the blocking gaps (same project, same conversation, worker alive).
- A gate scenario: trigger a render gate, see the reply BLOCK on it, answer `decline`, confirm the
  copilot reacts (the gate-decline path).
- A reload scenario: a few turns -> `reload` -> confirm the conversation restored (folds the
  persistence deferral).

## Next-session entry point

After `/clear`: read this draft, then run the `dev_flow.md` feature flow from step 2 (plan-draft) —
pick the transport (A/B/C above), write the real numbered Design decisions, spawn pre-impl review,
plan-lock with the maintainer, implement, post-impl review. The base ideas are here so you don't
re-derive them blind.
