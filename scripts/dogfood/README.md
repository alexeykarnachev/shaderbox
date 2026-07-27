# Dogfood rig (features 026/027)

The headless copilot-ENGINE dogfood harness + scenarios + run-analyzer. Drives the REAL copilot on a
standalone EGL context (no App/glfw) against a real LLM, interactively, one blocking `uv run` per turn.

- **`harness.py`** — `DogfoodHarness`: builds a real `ProjectSession` on EGL, `send`/`drive_until_idle`/
  `render`/`approve`/`decline`/`nodes`, `dump` (persist + structured JSON turn-result), `create(project_dir=)`
  resume, `clear_context` (memory wipe, same project), `reload`. Judge-side measurement:
  `render_strip(times, node_id)` — one horizontal contact sheet, each sample a fresh-behavior REPLAY
  (correct for a stateful script, unlike a live tick) — and `script_values(times, node_id)`, the
  `ScriptEngine.dry_run` passthrough giving the driven uniforms' values per sample.
- **`analyze.py`** — `uv run python scripts/dogfood/analyze.py <data_dir>` → auto tool-coverage + per-turn
  iteration/token/cost + recoveries + token-growth + the engine-look verdict rows (verdict / look # /
  vision_ok / node, per-turn FINAL verdict, strict-`ASK:` parse rate, engine-look vision spend,
  limit-forced turns), as a markdown block and JSON. `--template` + `--report-out` fills the
  `{{AUTO:...}}` slots of the report template; `--scenario <name>` names the run's scenario;
  `--dialogue` prints the user-visible dialogue and exits (`--project` overrides the dumps-resolved
  project dir).
- **`judge.py`** — GL-free pixel primitives for the motion/fidelity axes (`load_rgb`, `grid_cell`,
  `region_diff`, `bright_centroid`, `color_mask_centroid`, `column_runs`, `farthest_bright_angle`).
  Numbers out, never a verdict. `analyze.py` and `judge.py` import with NO GL and no run-dir side
  effects — the package `__init__` resolves `DogfoodHarness` lazily (PEP 562).
- **`REPORT_TEMPLATE.md`** — the report skeleton, structured as the five axes (fidelity · motion ·
  logic · honesty · process) + the dialogue; `{{AUTO:...}}` filled by `analyze.py`, `{{HUMAN:...}}`
  written by hand. ONE report per scenario; the FILLED copy lands in
  `ai_docs/features/NNN_dogfood_report_<run>.md` (durable, roadmap-linked — NOT here).
- **`scenarios/`** — goal-driven free-text missions (read whole, drive live, never replayed).
- **`runs/`** — gitignored run artifacts (per-run project dirs, data dir, JSON dumps, traces, PNGs).
  Purge between runs: `rm -rf scripts/dogfood/runs/{data-*,proj-*,*.json}` (regenerable; also clears the
  live OpenRouter key the data dirs hold).

**Operating manual = the `/dogfood` skill** (`.claude/skills/dogfood/SKILL.md`) — the run commands, the
`~/.bashrc` key footgun, the gate rule, the tool-coverage discipline. Don't duplicate them here.
