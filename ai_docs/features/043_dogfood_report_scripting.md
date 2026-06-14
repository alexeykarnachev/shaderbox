# Dogfood report — 043 copilot scripting (figure-8 circle + stateful flame)

- **Run:** scripting + flame · 2026-06-14
- **Scenario(s):** ad-hoc — (1) a pulsing, figure-8-drifting glowing circle driven by a script; a
  color-cycle follow-up. (2) a flickering flame with STATEFUL ember particles + a script-driven sway.
- **Model:** openai/gpt-5.1-codex-mini (the in-tree default)
- **Turns:** 5 (2 circle + 3 flame) · **Total cost:** ~$0.10
- NOTE: this run was driven to validate feature 043 (copilot scripting) end-to-end on the REAL engine;
  the trace `data_dir`s were purged (key hygiene) before the analyzer ran, so the AUTO slots below are
  filled from what was read live during the run, not a re-run of `analyze.py`. Numbers are approximate.

## 1. Verdict (HUMAN)

- **Mechanism works (pipeline end-to-end):** YES. The full scripting chain works autonomously on a cheap
  model: `read_script` (returns the stub with the `math.sin(ctx.t)` example) → `edit_shader` (declare the
  uniforms) → `write_script` (a real `update` body) → the agent reads back the compile verdict + the
  driven set + the value-diff motion verdict (`-> u_center, u_radius CHANGE across t (ANIMATING)`). No
  human eye was needed in the agent's loop to know its animation moved.
- **Overall conclusion:** 043 is solid. The agent reached for scripting on the right asks ("make it
  pulse/drift/animate"), routed `set_uniform` away from script-driven uniforms on its own (the prompt
  routing worked), wrote correct stateful `self.*` integration (24 ember particles, a hand-written
  `hsv_to_rgb`), and the motion verdict was honest in every case observed. The run ALSO surfaced two real
  bugs OUTSIDE 043 (a copilot hang, an mp4-render preset bug) — both fixed this session.

## 2. Per-turn (AUTO — from live trace reads)

| Turn | Ask | Tools fired | Result | cost |
|---|---|---|---|---|
| 1 | scripted pulsing figure-8 glow circle | create_node, read_shader, edit_shader, read_script, write_script | ANIMATING, clean | ~$0.016 |
| 2 | + rainbow color cycle; set u_radius to 0.5 | edit_shader, write_script | ANIMATING (3 uniforms); agent went to script not set_uniform | ~$0.009 |
| 3 (flame 1) | flame with stateful ember particles via arrays | create_node, write_shader, write_script (+ retries) | drives 3 arrays, ANIMATING | ~$0.035 |
| 4 (flame 2) | rising embers + flickering tongues + hot gradient | write_shader ×several, write_script | over-corrected to horizontal banding; **process HUNG post-turn** | ~$0.069 |
| 5 (flame 3) | rewrite shader to a clean vertical flame | write_shader ×several, edit_shader | clean vertical flame, ANIMATING; **also hung post-turn** | ~$0.01 |

## 3. Per-render visual eyeball (HUMAN)

- **circle t=0:** a clean glowing blue circle, centered, on dark — correct.
- **circle strip (t=0→2.0):** the circle visibly MOVES (rises + drifts, figure-8) AND pulses (radius
  grows then shrinks) — exactly the ask. The stateful machinery + the ctx.t math both animate.
- **circle color strip (t=0→10):** red→green→cyan→purple→red while drifting — the value-diff verdict's
  best case (geometry near-static, color changes every frame; a pixel-bbox diff would partly miss it,
  the value-diff caught `u_color CHANGE`).
- **flame turn 3 (t=0):** orange flame-mound, banded — a tame start (embers haven't risen at t=0).
- **flame turn 4 (`_1.png`):** REGRESSION — full-width horizontal scanline noise, not a flame. The agent
  over-corrected. The stateful embers still animated (bright base clusters shift frame to frame).
- **flame turn 5 (`_4.png`) + final strip:** FIXED — a clean vertical flame tongue, hot white-yellow
  core → orange → dark-red, swaying/curving across frames, a flickering ember at the base. Reads as fire.

## 4. Tool coverage (AUTO)

Fired this run: `create_node`, `read_shader`, `edit_shader`, `write_shader`, `read_script`,
`write_script`. The two NEW 043 tools (`read_script` / `write_script`) fired correctly and were the
load-bearing surface. `set_uniform` was deliberately probed (turn 2) and the agent correctly DIDN'T call
it on a script-driven uniform — it went to the script, per the prompt routing.

**Cold tools this run:** `grep`, `read_lib`, `switch_node`, `delete_node`, `render_image`,
`render_video` (the run was scripting-focused, not a coverage sweep).

## 5. Token / cost mechanics (AUTO — approximate)

- **Per-turn context (peak iteration in=):** ~8-23k tok; the flame turns ran ~20k peak.
- **Per-turn cost:** ~$0.009-$0.069; the dearest was flame turn 4 (the one that hung — many write_shader
  iterations + an 18k reasoning burn).
- **Token growth shape:** normal; no runaway. The expensive turns were iteration-count, not context.
- **Recovery counts:** the agent self-corrected a `u_ember_pos` array-shape coercion error (the
  per-key motion fact named it: "value does not match vec2[24] — provide a list of 48 numbers") and a
  stale-`old_str` edit miss — both recovered same-turn.

## 6. Honesty / visual-blindness (HUMAN)

- **Honesty judgment:** GOOD. Every `write_script` reply matched the engine facts — the agent claimed
  ANIMATING only when the value-diff said so, and never described a visual it couldn't see beyond the
  ink/bbox/luma the render line gave it. The per-uniform "drives u_x, u_y, u_z" + the "u_color constant"
  detail steered it correctly. The empty-dict/no-op trap never fired (the agent always returned a real
  dict). The motion verdict's value-diff design proved itself on the color-cycle (the case it beats a
  pixel-bbox diff).

## 7. TODOs (HUMAN)

### (a) Improve the COPILOT / agent
- **FIXED THIS SESSION — the hang.** A stalled LLM stream left the non-daemon worker blocked inside
  `for ev in client.stream(...)` (cancel checked only at iteration boundaries) riding the httpx 600s
  default; `release()`'s 5s join timed out and interpreter `_shutdown` hung joining the survivor
  (py-spy-confirmed: MainThread in `_shutdown`, worker in `queue.get()`). Fixed: a per-delta cancel check
  + a bounded 120s client timeout (commit `167509f`); resolved the `todo.md` "Stop unresponsive" deferral.
- **The orphan-key steer.** When the agent wrote a script before declaring the uniform it got an orphan
  key; the message now says "declare it in the SHADER first" (committed in the 043 post-post wave). Watch
  whether a cheaper model still loops here in a future run.
- **The agent over-corrects on a vague visual ask** (flame turn 4: "flickering tongues" → horizontal
  banding). This is model-quality, not a pipeline gap — a better model wouldn't trip; no guard earned.

### (b) Improve the DOGFOODING framework / harness / skill
- **FIXED — `render_video_mp4`** (H.264 for iPad; WebM doesn't open on iOS). A FREE preset gave ffmpeg a
  stray `-s 0x0` broken pipe — uses FIXED_DIMS now (commit `ae71996`).
- **FIXED — skill update** (commit `80fe0c6`): always `timeout`-wrap a turn process (a hang otherwise
  never exits/dumps); `py-spy dump --pid` is the hang-diagnosis tool; the frame-STRIP is the cheapest
  visual motion check (loop `render_at`, composite, Read one sheet).
- **Open — a `render_strip` harness method.** I hand-rolled the PIL contact-sheet 3× this run; it's a
  ~10-line method over the existing `render_at`. Add it next dogfood that eyeballs motion.
- **Open — report-before-purge discipline.** I purged the trace `data_dir` before writing the report, so
  this report's AUTO slots are hand-approximated. Run `analyze.py --report-out` BEFORE the `rm -rf`.

### (c) Improve the LIBRARY
- None this run. The flame used hand-written GLSL + SB_ helpers (SB_center_uv / SB_sd_circle / SB_glow on
  the circle); no library gap surfaced. The agent wrote its own `hsv_to_rgb` in Python (correct — that's
  a script helper, not a shader-lib concern).
