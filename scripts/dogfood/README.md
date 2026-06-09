# Dogfood rig (features 026/027)

The headless copilot-ENGINE dogfood harness + scenarios + run-analyzer. Drives the REAL copilot on a
standalone EGL context (no App/glfw) against a real LLM, interactively, one blocking `uv run` per turn.

- **`harness.py`** — `DogfoodHarness`: builds a real `ProjectSession` on EGL, `send`/`drive_until_idle`/
  `render`/`approve`/`decline`/`nodes`, `dump` (persist + structured JSON turn-result), `create(project_dir=)`
  resume, `clear_context` (memory wipe, same project), `reload`.
- **`analyze.py`** — `uv run python scripts/dogfood/analyze.py <data_dir>` → auto tool-coverage + per-turn
  iteration/token/cost + recoveries + token-growth, as a markdown block and JSON. `--template` +
  `--report-out` fills the `{{AUTO:...}}` slots of the report template.
- **`REPORT_TEMPLATE.md`** — the report skeleton; `{{AUTO:...}}` filled by `analyze.py`, `{{HUMAN:...}}`
  written by hand. The FILLED copy lands in `ai_docs/features/NNN_dogfood_report_<run>.md` (durable,
  roadmap-linked — NOT here).
- **`scenarios/`** — goal-driven free-text missions (read whole, drive live, never replayed).
- **`runs/`** — gitignored run artifacts (per-run project dirs, data dir, JSON dumps, traces, PNGs).
  Purge between runs: `rm -rf scripts/dogfood/runs/{data-*,proj-*,*.json}` (regenerable; also clears the
  live OpenRouter key the data dirs hold).

**Operating manual = the `/dogfood` skill** (`.claude/skills/dogfood/SKILL.md`) — the run commands, the
`~/.bashrc` key footgun, the gate rule, the tool-coverage discipline. Don't duplicate them here.
