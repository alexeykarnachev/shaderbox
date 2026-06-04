# 020 · 23 — The untangle: array uniforms, full-turn history, honest visuals, lean chat

One coherent wave fixing a tangle of copilot-quality issues that a brainstorm/review swarm
(`wcn3g2pqn`) traced to a SINGLE false premise plus three independent surface issues, all reproduced
live in session `copilot_dev_2026-06-04_12-05-57`. Driven by the maintainer + the swarm's
probe-verified synthesis.

## The diagnosis (one chain + three surfaces)

THE CHAIN (one false premise): the 020·20 guard rejects ALL `array_length>1` uniforms because it
believed a mis-shaped array write "silently pops off-thread". A moderngl probe DISPROVES this — a bad
array write RAISES synchronously; the only silent surface is `Node.render`'s own debug-logged
try/except. That guard is the head domino: it blocked the agent's CORRECT `set_uniform('u_text',
codepoints)`, forcing it to edit the GLSL into `uniform uint u_text[64] = uint[](...)` (illegal — a
uniform can't be default-initialized) → C1060 → broken program → "no active uniform" → `replace_lines`
thrash to the iteration cap. THEN cross-turn history blindness erased the trace (`_commit_turn`
persists only a flat user/assistant pair, discarding the tool trajectory), so the agent confabulated
when asked why.

THREE SURFACES (independent of the chain, same session):
- **B. The agent claims visual results it cannot see.** Across 4 "I see nothing / black screen" turns
  it asserted "text now appears centered", "there you go, should be visible" — it compiled clean but
  the geometry was wrong and it has no vision. The prompt says "never claim a visual result"; it ignored it.
- **C. The chat dumps the full ~1500-line shader source on every `read_shader`.** `session.py` appends
  the full read result to the tool_status Message → the user sees the whole source in chat. The USER
  doesn't need it (the editor shows the code); the AGENT does (it edits by line number).
- **D. Chat prose isn't selectable — dragging it moves the window** (deferred in 021; now to fix).

## Goal

Remove the false premise at the root so the agent travels the human's road, make a failed turn
self-diagnosable, stop the agent from claiming what it can't see, and de-clutter the chat — without
the compound-local-fix trap that created this mess.

## Out of scope (each with a trigger)

- **History-window trim** (the bound). Persisting full turns makes `history` grow monotonically;
  `max_input_tokens` is a verified DEAD constant (no trim exists). This wave persists; the trim is a
  SEPARATE follow-up. **Trigger (a real `todo.md` blocker):** a multi-turn `read_shader`-heavy session
  approaches the provider context limit, OR the next touch of `build_messages`.
- **Surfacing `Node.render`'s silent try/except to the agent.** The one true silent-failure surface
  (`core.py`) stays untouched — benign for correct-by-construction coercion. **Trigger:** a value that
  passes coercion but fails the GL write is observed.
- **The agent SEEING the render (vision / a render-to-preview the agent reads).** B only makes the
  agent HONEST (describe + ask), not sighted. **Trigger:** the visual-variant-optimizer feature.

## Design decisions (numbered, lock-in) — taking the swarm's recommended defaults

### D1 — Array-uniform coercion (the real bug, head of cascade) — PROBE-PINNED SHAPES
Extend `app.py::_coerce_uniform_value` to branch on `uniform.array_length`. The moderngl write shapes
are PROBE-VERIFIED on the real ctx (not guessed): a `dim==1` array wants a FLAT list of `array_length`;
a `dim>1` array wants `array_length` NESTED `dimension`-tuples (a flat list RAISES). So:
- `array_length == 1` → existing scalar/vec logic UNCHANGED.
- `array_length > 1`, **uint text array** (`gl_type == GL_UNSIGNED_INT`, `dimension == 1`) → accept a
  `str` (→ `util.str_to_unicode(value, array_length)`, REUSING the UI's exact converter — it
  truncates+null-pads+ord()s) OR a `list[int]` (its OWN truncate+null-pad+int-coerce — `str_to_unicode`
  takes a str, so the list arm can't reuse it; a small `_pad_codepoints(list, n)` helper next to it, or
  3 inline lines). Emit `int` elements (the `{GL_UNSIGNED_INT: int}` map) so it round-trips node.json
  as int. Result is a FLAT list of `array_length` ints (dim==1) — matches what the UI text widget stores.
- `array_length > 1`, **numeric, `dim == 1`** (float[N]) → a FLAT list of exactly `array_length` numbers
  (matches the UI's flat store + the probe). REJECT other lengths with `None`. NO padding.
- `array_length > 1`, **numeric, `dim > 1`** (vecN[M]) → exactly `array_length * dimension` numbers OR
  `array_length` nested `dimension`-tuples → NEST into `array_length` rows of `dimension` (probe: nested
  required). REJECT other lengths/shapes with `None`. NO padding. (There is NO existing UI path that
  sets a vec-array, so this arm is net-new but probe-grounded.)
- A `str` on ANY non-(uint-text-array) uniform → `None` (so `set_uniform('u_speed', 'fast')` errors).
- Widen the return annotation to include `list[int]` + the nested numeric shape.

### D2 — Delete the guard; one feedback channel
In `_copilot_set_uniform` DELETE the `array_length > 1` reject block — array uniforms now fall through
to `_coerce_uniform_value` + the existing `coerced is None` branch. Update that error string to teach
the array shapes (pass a uint text array's text as a string `"Hello\nWorld"`, or a list of N
codepoints; a numeric array as N*dim numbers). Sampler/block/engine-driven rejects + the write tail
(try_to_release + dict-assign) UNCHANGED.

### D3 — `set_uniform` tool arg accepts a string
`tools/shader.py::_SetUniformArgs.value`: widen to `float | int | str | list[float | int]` and rewrite
the Field description to teach the three shapes in ONE place (number = scalar; list = vector / numeric
array; string = uint text array, auto-converted to codepoints — the same control the user has in the UI).
The ONLY tool-surface change; no new tool, no new converter.

### D4 — Persist the FULL turn (cross-turn memory) — TAIL BOUNDARY PINNED
Add a `messages: list[LLMMessage] = field(default_factory=list)` field to the terminal dataclasses
(`AgentTurnDone`/`AgentError`/`AgentCancelled`). THE TAIL BOUNDARY (pre-impl B2 — the head is **2
system + history + user**, NOT 1): in `run_turn`, capture `head_len = len(messages)` IMMEDIATELY after
`build_messages(...)` (before the loop appends anything); the terminal events carry ONLY
`messages[head_len:]` (the assistant/tool tail this turn produced). Do NOT recompute the boundary in
`_commit_turn` from `len(self.history)` (off-by-one on the 2 system messages + racy with `_drop_turn`).
`session.py::_run_one_turn` reads the tail off the buffered `terminal` event; `_commit_turn` appends the
user message ONCE (as today) + the captured tail, with `error_text` as the trailing assistant note on a
fail. INVARIANTS:
- **Tool-group completeness (pre-impl H1):** before committing, for the trailing assistant message,
  verify EACH `tool_call.id` has a matching later `role=tool` `tool_call_id` in the tail; if any is
  missing (a cancel returned mid-batch after appending the assistant-with-tool_calls but before all
  results — `agent.py` cancel sites), DROP that trailing assistant + any partial tool messages after it.
  An orphaned `tool_call_id` 400s the next stream. One helper, run on every terminal tail.
- **Clean no-tool turn (pre-impl Q5):** a greeting/question turn has a tail of just `[assistant(text)]`
  (no tool messages) — must still persist `user` + `assistant` correctly. The slice must NOT assume a
  tool message exists; today's `assistant_text`/`error_text` path covers it — preserve it.
- `_commit_turn` stays the SOLE writer; the `_drop_turn` teardown guard stays; the two bare-except
  AgentError fallbacks (which never see `run_turn`'s `messages`) commit with an EMPTY tail (the field
  default) → no-op-safe, persists the user+assistant(error) pair like today.
- Keep `sanitize_display` on assistant CONTENT only (never `tool_call_id`/`arguments`).
- Readers (pre-impl L1/L2, verify-only — confirmed safe): `_HistoryModel`/`from_runtime`/`to_history`
  ALREADY round-trip `tool_call_id` + `tool_calls`; `build_messages` splices `*history` verbatim; the
  transcript renderer reads `state.messages` (Message roles only), NEVER `self.history` — history !=
  messages, so the renderer never sees a `role=tool`. No persistence change needed.

### D5 — Prompt: value-vs-declaration + the GLSL invariant + honest visuals (EXTEND, don't duplicate)
The existing prompt ALREADY has both rules (pre-impl Q4) — EXTEND them in place (both stay in the
cacheable `_SYSTEM_PROMPT` prefix), don't add competing sections:
- EXTEND the existing "Change a runtime VALUE... edit the SOURCE for LOGIC / add/remove/rename" bullet:
  a uniform's CONTENT — a number, vector, OR the codepoints of a uint[] text array — is a runtime VALUE
  set via `set_uniform` (pass a text array's text as a plain STRING; ShaderBox converts it, the same
  control the user has in the UI). And the GLSL invariant: a uniform can NEVER be default-initialized
  (`uniform uint u_text[64] = uint[](...)` does NOT compile) — to change what an array holds, use
  `set_uniform`, never an edit. (Keep it value-vs-declaration so add/remove/rename/reshape stays legal.)
- EXTEND the existing "YOU CANNOT SEE / never claim a visual result" section with the MISSING escalation
  (the real behavioral delta — the agent re-asserted across 4 "I see nothing" turns): after a change,
  state what you CHANGED + that it compiled, ASK the user to look. If the user says "I see nothing /
  black screen / nothing there", do NOT re-assert success — treat it as a REAL failure, re-read, and
  reason about the math. (Note: D5's efficacy depends on D1/D2 landing — removing the dead-end that
  forced the confabulation is the actual fix; the honesty rule alone is weak.)

### D6 — Chat: read_shader summary, not the full source (the user has the editor)
The AGENT's IN-FLIGHT tool result (the `msg` in `run_turn`'s live `messages` THIS turn) keeps the FULL
line-numbered listing (it edits by line number this turn). But TWO other copies get the SUMMARY:
- **Chat display:** the tool_status Message shows `read <Name> — <N> lines, <U> uniforms, compiled
  clean` (or the short error list if it has errors — errors are short + actionable).
- **Persisted history (the D6↔D4 fork — RESOLVED to the LOWER-RISK path at impl):** the persisted
  `role=tool` content stays the FULL `msg` (correctness-safe — replay just carries more tokens; the
  freshness guard means the agent never EDITS against a prior-turn listing anyway). Summarizing the
  PERSISTED copy was the reviewer's recommendation for size, but it forks the in-flight vs persisted
  string (the wave's highest-risk seam) — so this wave summarizes ONLY the CHAT DISPLAY, and the
  history-growth TRIM (already a filed out-of-scope follow-up) is the proper lever for size. This keeps
  D4's tail capture a clean verbatim slice with no per-tool-message substitution.
- **Mechanism (pre-impl Q3 — the clean seam, NOT a `name=="read_shader"` special-case):** the tool
  returns the summary in its `payload` (read_shader already builds a payload dict; add `"display"`);
  `AgentToolCard` gains a `display: str = ""` field threaded EXACTLY like `widget`
  (`_widget_from_payload`-style); `session._apply_event` uses `ev.display or ev.result` for the
  tool_status line; `_commit_turn` persists the summarized tool content for a tool that supplied a
  `display`. Generalizes the 020·20 result-line cleanly.

### D7 — Chat: selectable prose + title-bar-only window drag
- `WindowFlags_.no_move` on the chat window (`_WINDOW_FLAGS`) — the title bar STILL drags (imgui
  special-cases it under no_move; the window has a title bar — `begin("Copilot")`, no `no_title_bar`),
  the body no longer moves the window. (NOT the global `io.config_windows_move_from_title_bar_only` —
  too broad, rejected.) **CORNER/BOTTOM_STRIP force pos every frame (unaffected — `no_move` just stops
  the futile drag-snap-back jitter). FREE (pre-impl Q1): repositioning NARROWS to title-bar-only — this
  IS the intended "move only from the top bar" behavior, but it's a real change to FREE; the maintainer
  make-run-checks "drag the FREE window by its title bar".**
- Text getting OUT of a message — RESOLVED to (b) at impl. EMPIRICAL: `input_text_multiline` (the only
  imgui surface that natively selects) does NOT word-wrap — it horizontal-scrolls long lines, which
  looks wrong for prose, and auto-height word-wrap math over it is fragile + risks nested scrollbars
  inside the transcript `begin_child`. `text_wrapped` wraps but can't select. So this wave ships the
  robust path: `no_move` kills the body-drag (the actual reported bug — the window stops moving when you
  try to interact), + a per-message COPY affordance on user/assistant messages (copies the message text
  to the clipboard, reusing `pyperclip` via a `ui_primitives` helper). True in-line drag-selection of
  WRAPPED prose is a real imgui limitation worth a separate spike, not a forced bad multiline now — a
  `todo.md` follow-up. The `tool_status` branch (summary + 020·21 button) is untouched.

### D8 — NOT changed (verify only), and traps to avoid
- `max_iterations` stays 12 (never the binding constraint — the dead-end was).
- The f90f5ff9 template ships `uniform uint u_text[MAX_TEXT_LEN]` + a flat-int node.json — reached via
  create_node(template) then set_uniform, never an edit. Verify, don't change.
- AVOID: the flat-tuple coercer spec (vecN[] needs nested rows — moderngl raises on flat); pad/truncate
  for NUMERIC arrays (silent corruption); an edit-retry valve targeting the already-handled stale path;
  claiming `max_input_tokens` absorbs the growth (it's dead); bumping max_iterations as if it's the fix.

## Files touched (anticipated)

- `app.py` — `_coerce_uniform_value` array branch (D1); delete the `_copilot_set_uniform` guard + error
  string (D2); the read_shader summary seam if App-side (D6).
- `util.py` — reuse `str_to_unicode` (verify signature) (D1).
- `copilot/tools/shader.py` — `_SetUniformArgs.value` widen + description (D3); read_shader summary vs
  full-listing split (D6).
- `copilot/agent.py` — `messages` field on the terminal dataclasses + populate at yield sites (D4);
  `set_uniform` no longer hits the array reject path.
- `copilot/session.py` — `_run_one_turn` capture + `_commit_turn` full-turn append + completeness
  invariant (D4); the tool_status display-text seam (D6).
- `copilot/persistence.py` — verify `_HistoryModel` round-trips the persisted tool tail (D4).
- `copilot/prompt.py` — D5 (value-vs-declaration + GLSL invariant + honest visuals).
- `widgets/copilot_chat.py` — `no_move` + selectable prose (D7); the summarized tool_status render (D6).
- `ui_primitives.py` — a `read_only_prose` primitive if D7(a) (D7).
- `ai_docs/todo.md` — the history-window-trim blocker (out-of-scope).
- tests — set_uniform array (str + list, uint pad, numeric reject + nest), full-turn history round-trip
  + no-orphan-tool_call, read_shader summary display.

## Manual verification (maintainer, in-app)

- "Create an SDF text shader, write Hello World" → ONE create + ONE set_uniform("u_text","Hello\nWorld")
  succeeds (no source-edit cascade); the text renders.
- A numeric array uniform: a wrong-length value REJECTS cleanly (no silent pad).
- Ask "why did that fail" after a failed turn → the agent can SEE its own tool trajectory + errors
  (history persisted) and answers truthfully, no confabulation.
- The agent NEVER says "it's centered/visible now" — it says "I changed X, compiled clean, check the
  preview"; on "I see nothing" it re-reads instead of re-asserting.
- A read_shader shows a SUMMARY in chat, not the full source; the editor still has the code.
- Select text in a chat message (drag) → it selects; dragging the title bar moves the window; dragging
  the body does not.
- (Headless: the coercion matrix, the history round-trip + orphan check, the read summary; the chat
  selection/drag fit is a maintainer make-run check.)

## Open questions for the user

Resolved at plan-lock (maintainer + swarm defaults): combined wave; read_shader = summary-line-only;
numeric arrays reject (no pad); set_uniform accepts a string; max_iterations stays 12; the history trim
is a filed follow-up, not this wave. Lock at impl (low-surface mechanical): D7 selection mechanism
(read_only_prose vs no_move+Copy); whether the PERSISTED read_shader tool result is summarized too
(weigh against the replay-needs-line-numbers + the trim deferral).

## Review history

- **Pre-impl review (2 agents — uniform/history core + chat/prompt/blast):** confirmed the probe-verified
  premise and the additive posture. Folded in: (D1, B1/M1) coercion shapes are PROBE-PINNED — `dim==1`
  flat, `dim>1` nested rows (a re-probe on the real ctx confirmed: `vec3[2]` needs nested, `float[4]`
  needs flat, `u_text` flat-64); the uint `list[int]` arm needs its OWN pad (can't reuse `str_to_unicode`).
  (D4, B2/H1/H2) the tail boundary is `head_len` captured in `run_turn` (head = 2 system + history +
  user, not 1), orphan-tool_call drop on the captured tail at every terminal site, the except-fallbacks
  commit an empty tail, the clean no-tool turn still persists user+assistant. (D6, Q3) the persisted
  read_shader result is SUMMARIZED too (safe — the freshness guard forces same-turn re-read), with the
  in-flight-vs-persisted fork flagged as the highest-risk seam; threaded via an `AgentToolCard.display`
  field like `widget`, not a tool-name special-case. (D7, Q1/Q2) FREE-layout repositioning narrows to
  title-bar-only (intended); selectable prose is user/assistant-bodies-ONLY (tool_status keeps
  caption+button — coexistence dissolves), auto-height to avoid nested scroll. (D5, Q4) EXTEND the
  existing value bullet + "YOU CANNOT SEE" section, the real delta being the "on 'I see nothing' don't
  re-assert, re-read" escalation. Persistence already round-trips the tool tail (verify-only).
