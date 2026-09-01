# Night-session brief — autonomous work on the Pi

A self-contained job for an unattended session running on the Raspberry Pi. The maintainer
reviews the result in the morning; nobody is awake to answer a question, so everything needed
to finish is written down here.

**Delete this file in the commit that finishes the job.** It describes one night's work, not a
standing backlog — leaving it behind turns it into the parked-work doc `dev_flow.md` bans.

Read `CLAUDE.md` first (the cold-start chain), then this file. Ignore the ranking of "what's
next" in `roadmap.md`'s banner — the banner says the maintainer picks, and for this one night
they have picked, here.

---

## Authority

Full commit-and-push on `dev`, no confirmation needed. Standard repo rules still hold: commit on
`dev` and never `master`, ASCII-only subject, worklog body, and never leave `projects/dev/`
unstaged. Push when green; a local tree ahead of `origin` is a failure state.

**Do not ship, tag, release, or touch itch.** `/ship` is maintainer-triggered only.

---

## The job

Give ShaderBox a composed gate target — one command, one exit code, that cannot report green
while red — and make the display-less case legible.

This exists because the repo has now twice reported a passing gate that was failing, and the
prose rule meant to prevent it did not. `dev_flow.md ### make check` already says "Judge it by
the EXIT CODE, unpiped"; `conventions.md` repeats it; feature 067 announced a green gate that
was red exactly this way, and the agent surveying for this brief did it again three hours after
reading the rule. A rule with no gate is a wish. Ship the check, not another paragraph.

### What to build

**1. `make gates`** — runs the gates in order, stops at the first real failure, exits non-zero
if any failed. Order cheapest-first so a broken tree fails fast: `check`, then `test`, then
`smoke`.

**2. Make a skipped `smoke` legible.** `scripts/smoke.py` exits **0** when it skips for want of
a display (`scripts/smoke.py`, the "SKIPPED — no GPU window available" branch). That is correct
on its own — a headless box has not failed anything — but it means the skip lives only in log
text, which is the one channel an automated caller discards. So on the Pi a naive composed gate
counts a skipped smoke as a passed one.

`gates` must therefore report three outcomes, not two: passed, skipped, failed. A run where
smoke skipped is **not** the same result as a run where it passed, and the summary line must say
so. How you carry the skip out of `smoke.py` is your call — a distinct exit code that `gates`
interprets is the obvious shape, but keep `scripts/smoke.py` exiting 0 on skip for every
existing direct caller, or say in the commit body why you changed that contract.

**3. Warn when stdout is not a terminal.** When `gates` output is piped or redirected, print a
final line saying `$?` reflects the pipe and naming `tee` / `PIPESTATUS`. Warn, never refuse —
redirecting to a file is legitimate and CI does it. This is the shape that actually holds:
it fires at the moment the information is lost, rather than asking a future agent to remember.

**4. Wire it into the docs.** `CLAUDE.md`'s hard rule currently names `make check` and
`make test` separately; `dev_flow.md ## Recipes` documents each target. Add `gates` as the
single command, keep the individual targets documented (they stay useful alone), and do not
delete the existing exit-code guidance — it explains *why* the target exists.

### The verification that makes it trustworthy

Do not certify this by running it once and seeing green. Mutation-test it in both directions,
and put the results in the commit body:

- Break something `check` catches → `gates` exits non-zero **after running one target, not
  three**. That proves the ordering and the early stop, not merely the exit code.
- Break something only `test` catches → `gates` exits non-zero, and `check` ran first.
- Run it piped → the warning fires, and the exit code is still correct when captured properly.
- Restore everything → `gates` exits 0.

A gate that passes whether or not the bug exists is theater; the falsifier is the deliverable
here. Re-apply reverted edits by hand — never `git checkout --` a file to undo your own
uncommitted change (it eats the fix; `dev_flow.md` documents this).

**Confirm each mutation actually mutated.** A mutation that does not run is indistinguishable
from a gate that does not gate: both show green, and you cannot tell which you have. The sibling
editor repo hit this while building the same target — a `raise` appended after a file's final
`raise SystemExit(main())` was unreachable, reported exit 0, and nearly got read as a broken
Makefile. So put the break somewhere that certainly executes (inside the function under test,
not after its last statement), and confirm the *unmutated* command passed immediately before, so
a green result is known to mean something.

**Do not pipe anything inside the target.** Piping a gate to `tee` inside `gates` reintroduces
exactly the bug it exists to prevent — the same repo did this and a crashing sub-gate reported
green. `set -o pipefail` will not save you: make's shell is `dash` here too. Redirect to a file
and check the status before anything reads it.

---

## Environment: what the Pi can and cannot do

Measured on the Pi (aarch64, Debian 12 Bookworm, glibc 2.36) at `c10a3a3`, the night before.
Baseline the work against these numbers; do not treat a pre-existing failure as something you
broke.

| Gate | Result on the Pi |
|---|---|
| `make check` | **green**, exit 0, 0 pyright errors, ~5 min |
| `make test` | **721 passed, 159 skipped, 46 errors, 42 failed** — see below |
| `make smoke` | **skips** — no DISPLAY. Logs "SKIPPED", exits 0 |
| `make run` | impossible — no display, no GPU |

**The 42 failures and 46 errors are environmental and pre-existing. Do not try to fix them.**

- **42 failures, all `tests/test_editor_ffi.py`** — the vendored `shaderbox/resources/editor/libeditor.so`
  is `ELF 64-bit LSB shared object, x86-64`. It cannot load on ARM, so `ctypes.CDLL` raises at
  import of every test in that file. Rebuilding it for ARM is not this job: `conventions.md`
  (the vendored-editor quirk) says the binary rebuilds from a committed editor-repo sha, and
  the vendored sha `c5fabc8` is a deliberate resting point.
- **46 errors across 8 files** — `Exception: (standalone) XOpenDisplay: cannot open display`.
  Those tests call `moderngl.create_standalone_context()` with no `backend=`, which defaults to
  X11. EGL works fine on this box (verified: `backend="egl"` returns `4.6 (Core Profile) Mesa
  24.2.8`). There is **no env-var override** — `glcontext` selects the backend by a `backend=`
  kwarg, so `GLCONTEXT_BACKEND=egl` does nothing (measured; it changes no result). Making these
  run headless would mean editing the fixtures in 8 test files. That is a defensible separate
  change and **out of scope tonight** — mentioned so you do not rediscover it as a mystery.

The affected files, for reference: `test_document_save_preserves_values`, `test_float_canvas_export`,
`test_gl_lifetime_guards`, `test_help_snippets_compile`, `test_lazy_compile`,
`test_raw_texture_round_trip`, `test_uniform_row_pruning`, `test_video_frame_stepping`.

**So the Pi's honest pass condition for tonight is:** `make check` green, and `make test`
showing **721 passed** with exactly those 42 failures and 46 errors and no others. A new failure
outside that set is yours and must be fixed before pushing.

### Which gates already encode their own verdict (enumerated, not assumed)

Mutation-tested on the dev box before this brief was written, so `gates` can be built on facts
rather than on the hope that each member reports honestly. Every mutation was applied by hand
and reverted by hand, with the unmutated run confirmed green in between:

| Gate | Mutation | Exit |
|---|---|---|
| `check` | a return-type error in `shaderbox/util.py` | **2** |
| `test` | a failing assertion appended to `tests/test_glsl_lex.py` | **2** |
| `smoke` | `raise` inserted as the first statement **inside** `main()` | **2** |
| `smoke` | *unmutated, display-less* — the skip path | **0** |

So all three encode a real failure, and `smoke`'s exit 0 is specific to the skip. That is what
makes item 2 a genuine gap rather than a theoretical one: the only case where exit 0 does not
mean "this gate passed" is the skip, and it is the case the Pi hits every single run.

**A worked example of the unreachable-mutation trap, from building this table.** The first
attempt at the `smoke` mutation renamed `main` and appended a new one at end of file. It exited
2 — but from `NameError: name 'main' is not defined`, because `sys.exit(main())` at module
bottom bound the old name. The gate looked correctly mutated and was not: right exit code,
wrong reason. Inserting the `raise` inside `main()` produced `RuntimeError: mutation probe:
reached inside main`, which is proof the line ran. **Read the failure message, not just the exit
code** — that is the difference between the two.

Because `smoke` cannot run here, item 2 above is the one part you cannot fully verify on the
Pi — you can verify the skip path (that is what the Pi produces), but not the pass path. Say so
plainly in the commit body rather than implying you saw it pass.

### Running the gates over ssh

The Pi's non-interactive shell does not have `uv` on PATH. Prefix with
`export PATH=$HOME/.local/bin:$PATH`. And per the whole point of this job: capture the exit code
**unpiped**, e.g. `make test > /tmp/t.log 2>&1; echo "EXIT=$?"`, then read the log.

---

## Scope discipline

This is a small, mechanical change to build tooling. It should be a modest diff.

If it starts growing — rewriting test fixtures, restructuring the Makefile, touching app code —
stop and leave it undone rather than landing a sprawl nobody asked for. The `dev_flow.md` rule
holds: scope grew means halt. There is nobody to ask at 3am, and the correct autonomous move is
to finish the part that is clearly in scope, push it, and write what you did not do and why in
the commit body.

Do not invent additional work because the machine is idle. If the job is done and gates are
green, stop.

---

## Definition of done

1. `make gates` exists, runs check → test → smoke in order, stops at first failure, one exit code.
2. Skipped is reported as distinct from passed.
3. A non-tty stdout produces the pipe warning.
4. Mutation-tested both directions, results stated in the commit body.
5. `make check` green; `make test` at the documented baseline, no new failures.
6. Docs updated (`CLAUDE.md` hard rule + `dev_flow.md ## Recipes`).
7. **This file deleted** in the final commit.
8. Committed and pushed to `origin/dev`.
