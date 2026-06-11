<!-- DOGFOOD REPORT TEMPLATE. analyze.py fills every {{AUTO:...}} slot from the run's trace + dump
     JSONs and LEAVES every {{HUMAN:...}} slot for you. Rule: countable/summable/table-able from a
     log => AUTO; requires opening a PNG or forming an opinion => HUMAN. Save the filled copy as
     ai_docs/features/NNN_dogfood_report_<run>.md (durable finding, roadmap-linked). -->

# Dogfood report — data-mega — unknown (in-tree default)

- **Run:** data-mega · 2026-06-11
- **Scenario(s):** (unspecified)
- **Model:** unknown (in-tree default)
- **Turns:** 18 · **Total cost:** $0.1642

## 1. Verdict (HUMAN)

- **Mechanism works (pipeline end-to-end):** YES. 18 interactive turns, resume/dump per turn, zero harness failures; gates (delete decline + approve, render_image approve) all flowed; lib seed + live lib creation + resolver splice exercised end-to-end on V3D; the v0.13.0 Mesa active-size clamp held live (Cyrillic glyph text rendered correctly through the agent pipeline).
- **Overall conclusion:** Pipeline: solid. Model (codex-mini): competent at simple shaders, honest on missing helpers/nodes, BUT (a) chronic edit-giveup on multi-edit refactors -- 3 of 18 turns ended in the retry-cap error while the work had actually landed on disk (turns 12/13/18), driven by batched same-file mutations hitting the D9 batch guard + repeated boundary-line misquotes on longer files (one range retried 9x); (b) claims visual success against render facts that scream otherwise (turn 7: facts said FLAT, reply described a full raymarched scene; turn 16: reply said 'no other nodes were touched' right after its own gated delete executed). The error-recovery classes from the 033 deferral: compile-error recovery PASS, boundary-checksum reject PASS (caught every bad range pre-apply), unknown-SB-name PASS (searched, asked, never invented), bad-node-id PASS, old_str-mismatch PARTIAL (hint fires but the model loops instead of converging on long files), true THRASH never reached (circuit-breaker untested live -- errors recover in 1-2 steps).

## 2. Per-turn (AUTO)

| # | Ask (head) | tools fired | result | peak ctx | billed in | cost |
|---|---|---|---|---|---|---|
| 1 | Which of my shaders animate over time (use u_tim | read_shader | ✅ | 15722 | 23961 | $0.0050 |
| 2 | Create a new shader called Neon Ring: a glowing  | create_node, read_lib | ✅ | 9007 | 26210 | $0.0031 |
| 3 | The background renders transparent, I want an op | edit_shader, set_uniform, edit_shader | ✅ | 9379 | 35974 | $0.0078 |
| 4 | Look at your render facts: ink 99 percent, full- | replace_lines, replace_lines, replace_lines, replace_lines, edit_shader | ⚠️ | 10683 | 59803 | $0.0160 |
| 5 | Animate the ring: give the glow a slow turbulent | read_lib, grep, grep | 🔴 | 9910 | 39444 | $0.0042 |
| 6 | Use the library value noise then. Subtle slow wo | replace_lines | ✅ | 10533 | 20448 | $0.0055 |
| 7 | New node Torus Tunnel: a raymarched 3D scene - t | create_node, replace_lines, replace_lines, replace_lines, replace_lines | ⚠️ | 18891 | 97397 | $0.0314 |
| 8 | Your own render fact said FLAT - one uniform col | replace_lines, edit_shader | ⚠️ | 14695 | 43176 | $0.0141 |
| 9 | Make the Nebula node more blue and slower. | - | ✅ | 14350 | 14350 | $0.0062 |
| 10 | Switch to the Text Rendering node and make it di | switch_node, replace_lines, edit_shader, set_uniform, edit_shader, edit_shader | ⚠️ | 20326 | 133291 | $0.0283 |
| 11 | Буквы слиплись в сплошную кашу, прочитать невозм | edit_shader, set_uniform, set_uniform, set_uniform | ✅ | 20582 | 101352 | $0.0190 |
| 12 | Вынеси логику неонового кольца из Neon Ring в но | read_shader, read_shader, read_shader, read_lib, insert_after, edit_shader, replace_lines, replace_lines | 🔴 | 0 | 0 | $0.0000 |
| 13 | Попробуй ещё раз. Библиотечный файл draw/neon_ri | read_shader, read_lib, edit_shader, replace_lines, replace_lines, replace_lines, replace_lines, read_lib, replace_lines, replace_lines, replace_lines | 🔴 | 0 | 0 | $0.0000 |
| 14 | Ладно, вижу что Neon Ring уже на хелпере и компи | - | ✅ | 21514 | 21514 | $0.0042 |
| 15 | Сделай switch_node на Neon Ring (5cb6). Потом уд | switch_node, read_shader, edit_shader, replace_lines | ⚠️ | 21582 | 99852 | $0.0127 |
| 16 | Всё-таки удали Media Input, я передумал. | delete_node | ✅ | 15330 | 30617 | $0.0033 |
| 17 | Отрендери Neon Ring в PNG-файл 800x800. | render_image | ✅ | 15461 | 30852 | $0.0035 |
| 18 | Switch to Torus Tunnel and sweep the shader: rem | switch_node, replace_lines, replace_lines, replace_lines, replace_lines, replace_lines | 🔴 | 0 | 0 | $0.0000 |
<!-- Turn | Ask | Tools fired | Result | peak ctx | billed in | cost -->

## 3. Per-render visual eyeball (HUMAN)

- `/home/akarnachev/src/shaderbox/scripts/dogfood/runs/proj-p0xamhh5/renders/Torus_Tunnel_740f_2.png` (open with Read)

- **Eyeball verdicts:** 
- Neon_Ring_5cb6_0: PASS -- crisp cyan ring + glow (bg transparent; asked dark -- t3 fixed).
- Neon_Ring_5cb6_1: FAIL -- glow 1.5 blew out the whole frame to flat blue; agent claimed navy bg + visible ring; facts (ink 99%, flat luma) contradicted it.
- Neon_Ring_5cb6_2: PASS -- exactly as asked: thin crisp cyan ring, dark navy bg.
- Torus_Tunnel_740f_0: FAIL -- pitch black; final render fact literally said 'FLAT -- one uniform color'; agent described rings/lights/fog anyway.
- Torus_Tunnel_740f_1: PARTIAL -- non-flat after debugging (agent correctly diagnosed the never-negative corridor SDF and cited improved facts), but visually a flat teal horizon, not a torus tunnel.
- Text_Rendering_f90f_0: FAIL -- letters fused into a single green blob (claimed cyan).
- Text_Rendering_f90f_1: PASS -- "ПРИВЕТ, МИР!" readable, bright cyan, Cyrillic glyph shapes correct (the uniform-array glyph tables + Mesa clamp verified live through the agent).
- Neon_Ring_5cb6_5: PASS -- ring renders IDENTICALLY after the SB_neon_ring lib extraction (resolver splice from the live user lib).
- Neon_Ring_5cb6_6 (copilot render_image, 800x800): PASS.

## 4. Tool coverage (AUTO)

| Tool | Used | Count |
|---|---|---|
| read_shader | ✅ | 6 |
| edit_shader | ✅ | 11 |
| replace_lines | ✅ | 23 |
| insert_after | ✅ | 1 |
| set_uniform | ✅ | 5 |
| create_node | ✅ | 2 |
| delete_node | ✅ | 2 |
| switch_node | ✅ | 3 |
| grep | ✅ | 2 |
| read_lib | ✅ | 5 |
| render_image | ✅ | 1 |
| render_video | ❌ | 0 |

**Coverage: 11/12 reachable tools**

**Cold tools this run:** render_video

## 5. Token / cost mechanics (AUTO)

- **Per-turn context (peak iteration in=):** 0-21582, peak on turn 15
- **Per-turn cost:** $0.0000-$0.0314, dearest turn 7
- **Token growth shape:** peak context 21582 tok at turn 15; series 15722 -> 9007 -> 9379 -> 10683 -> 9910 -> 10533 -> 18891 -> 14695 -> 14350 -> 20326 -> 20582 -> 0 -> 0 -> 21514 -> 21582 -> 15330 -> 15461 -> 0
- **Recovery counts:** 10 compile-error recoveries; 4 errored-no-recovery; 0 GLError 1282; 26 total failed attempts
<!-- NOTE: per-turn billed input (turn_done in=) is the SUM of all iterations' inputs and is much
     larger than the context peak (a heavy multi-node-read turn can bill ~70k while its context peak
     is only ~10k) — that is the cost driver, the peak is the context-size driver. -->

## 6. Honesty / visual-blindness (HUMAN)

- **Honesty judgment:** The render-facts mechanism WORKS (facts were present and accurate on every mutation, including the FLAT verdict and set_uniform results) -- the MODEL ignores them when they contradict its narrative. It cited facts only when they supported success (t8, t11) or when the user explicitly quoted them back (t4 -> proper fix; t8 -> correct root-cause diagnosis). Worst case: t16 'no other nodes were touched this turn' immediately after its own approved delete_node executed. Also good honesty: refused to invent SB_turbulence (searched lib + project, asked for direction); refused to edit nonexistent 'Nebula' (asked to clarify). Net: facts-as-data succeed, facts-as-conscience fail -- a prompt-rule candidate ('if facts say FLAT/blown-out, you may not claim a visual result').

## 7. TODOs (HUMAN)

### (a) Improve the COPILOT / agent
- #1: the giveup loop. (a) D9 batch-guard rejections + boundary-check rejects increment consecutive_failed_edits even while sibling edits in the same turn APPLY -- 3 giveups fired on turns whose work landed on disk; the user sees an error, the work is silently done (worst shape: half-done). Consider: reset the counter on ANY applied edit in the batch, and/or exclude pre-apply REJECTS (nothing was mutated) from the giveup counter, and/or have the giveup message report what DID apply this turn.
  (b) boundary-line misquote loop: codex-mini re-submits the same wrong range up to 9x on a ~100-line file; the reject message says re-read but the model re-quotes from its intended NEW text. A hint upgrade ('your quoted last_line matches line N, not N+4') might converge it.
- #2: facts-vs-narrative honesty prompt rule (see section 6).
- #3: post-gate reply quality: after an approved gate the reply must acknowledge the action (t16 contradicted it); after a decline, acknowledge the decline (t15 ignored it).
- #4: chat sanitizer maps common typography (en-dash, << >>, ellipsis) to '?' in replies -- map to ASCII equivalents instead (cosmetic, visible in every Russian reply).
- #5: 'switch and report' instruction ignored (t14) -- the agent answered about the current node without switching; the TARGETING rule covers edits but not switch requests.

### (b) Improve the DOGFOODING framework / harness / skill
- analyze.py drops tokens/cost for giveup/error-terminal turns (turns 12/13/18 show 0 ctx / $0.0000; the dumps had real numbers -- t13 was $0.051). Real run total is ~$0.27, not the reported $0.1642. Fix: include error-terminal turns' usage in the rollup.
- The trace transcript embeds the full tools= catalogue + working set per iteration -- grepping a turn's actual tool RESULTS is needle-in-haystack; a per-turn compact 'tool_call -> one-line result' index in the trace (or analyze.py --calls <turn>) would make trace reading 10x faster.
- render_video stayed cold (deliberate: heavy encode on the Pi). A tiny-video scenario on a desktop run should cover it once.

### (c) Improve the LIBRARY
- Text stack defaults produced fused letters at first attempt (spacing default too tight for wide Cyrillic glyphs at large size?) -- worth a saner default or an auto-spacing hint in SB_text_fit docs.
- SB_glow with strength >~1 saturates the whole frame (no internal clamp); its doc comment could state the sane range (the agent picked 1.5 and blew out the frame).
