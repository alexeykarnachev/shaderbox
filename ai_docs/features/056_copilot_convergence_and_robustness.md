# 056 — Copilot convergence enforcement & robustness wave

Two halves, one feature. **Half 1 (the headline): convergence ENFORCEMENT** — the bounded
iterate-until-the-eye-agrees loop that 054 deferred and whose trigger FIRED in the 054 dogfood (the
model got "lacks detail in stars" from the aimed eye, made ONE edit, still didn't land the stars, and
over-claimed "blue canton with 50 stars" anyway). **Half 2: a robustness wave** closing the verified
defects the 2026-07-27 code recon surfaced — script-edit brakes that don't work, an auto-look that can
inspect the wrong node, invisible vision spend, a dogfood harness whose context semantics diverge from
the app, and a cluster of misleading error paths.

Why one feature: half 2's A1 (auto-look targeting) is a structural prerequisite of half 1 — a
convergence loop steered by a look at the WRONG node converges on garbage — and the probe-result
plumbing (CONV verdict + C1 usage + C2 classification) is one capability-signature change paid once.

Provenance: all Half-2 defects were verified first-hand against the tree at `ecaa469`; the round-1
and round-2 review swarms re-verified every claim (incl. the three then-relayed items E2/E3/E4, now
all confirmed REAL) at cited lines. No unverified claims remain.

## Half 1 — convergence enforcement (slice CONV)

### The class being fixed (the better-model test, applied explicitly)

Not model babysitting: the coherence hole is OURS. The engine holds evidence (an aimed eye read of
the frame vs the ask) and structurally cannot act on it more than once per turn; the turn's durable
record then keeps only the model's claim, not the eye's. A strictly better model still lives inside
that structure — it can over-claim once and the record has no counter-voice. 054 pre-registered the
trigger and it fired. What earns NO guard: heuristic "is it trying" judgments — the loop below is
bounded by a cap and by deterministic tool-facts only.

### Today

`agent.py:748-773`: when a turn changed the render and the model never called `probe_render`
(`ran.looked()` gate), the engine takes ONE aimed look (`look_for` = the user's ask), injects the
observation with provenance, and gives the model one iteration to react. Two holes: (a) the model
replying "done" while the eye said otherwise ends the turn — one edit later the 054 over-claim
lands; (b) the `looked()` gate lets the model OPT OUT of the aimed look entirely by calling
`probe_render` itself with any narrow `look_for` of its own — the looked-then-over-claimed branch
has zero enforcement (and the system prompt actively tells it to look before claiming).

### New shape

1. **The eye gains a bounded verdict, produced as data — and the engine STRIPS it before the model
   sees anything.** `_VISION_SYSTEM`, when a `look_for` hint is present, mandates one final contract
   line: `ASK: met | not-met | unclear` — an evidence-gated observation of whether the frame shows
   what `look_for` describes. MANDATED `unclear` when the ask is not verifiable from a single frame:
   relative asks ("darker", "slower"), non-visual asks (a uniform value, a rename), compound asks'
   non-visual parts. Binary+unclear, not 4-way (`partial` was behaviorally identical to `not-met`).
   Ordering pinned: the existing `look_for:` segment first, the ASK line LAST (two "final"
   instructions must not interleave). **The engine parses the line out of the eye's reply and
   REMOVES it from every model-facing `msg`** — the model's own `probe_render` calls also carry
   `look_for`, and a raw done-ness label on that surface would reinstate the witness-vs-judge hole
   the wording work closes. The verdict travels ONLY in the structured result (C0) and the engine's
   own decisions. Contract token + parse + strip live in ONE new tiny module
   (`copilot/vision_contract.py` — constants + parse/strip helpers) imported by BOTH
   `llm/openrouter.py` (prompt text) and `backend.py` (parse+strip); a neutral home, because
   `backend.py` deliberately imports nothing from `llm/` (the client arrives as an injected
   closure). A missing/garbled line parses as `unclear` (and there is nothing to strip).
2. **Witness-vs-judge wording reconciled, not contradicted.** The standing `_VISION_SYSTEM` line
   "NEVER a verdict that the shader is correct/good/done" and the FEEDBACK block's "the eye is a
   WITNESS, not a judge; YOU decide whether your intent was met" both get reworded to carve out the
   ASK line explicitly: no aesthetic judgment, no done-ness opinion — the ASK line is a factual
   report of whether the *named look_for content* is on the pixels, nothing more; the skeptical
   default applies to presence-claims about visible content, and anything not decidable from the
   frame is `unclear`, never `not-met`.
3. **The engine look is UNCONDITIONAL on the model's own looks.** The `looked()` gate is removed
   (verified: its one reader is this block). The aimed look fires at `finish_reason == "stop"`
   whenever: a render-authoring tool succeeded since the LAST engine look (or there was no engine
   look yet), `looks_used < copilot_convergence_max_looks`, an iteration remains, not cancelled,
   non-empty text. (The gate does NOT read the ask — same as today.) The model's own probes no longer suppress it —
   the 054 evasion closes. (Cost: possibly one extra vision call ~$0.002 on turns where the model
   looked; the per-node vision cache dedupes an identical (frame, look_for) pair.) **Accepted
   residual:** the empty-text forced-reply path (`agent.py:701-743`, a token-budget cutoff) returns
   before the look block and stays unenforced — a rare branch whose reply is engine-forced recap.
   **Trigger to revisit:** a trace where an over-claim ships via the forced-reply path.
4. **Loop control is tool-fact-based, never pixel-based.** Re-look requires a successful
   render-authoring call since the previous ENGINE look. Bookkeeping is explicit, not name-scanned:
   the loop keeps a local marker (the `_RunLog` entry index at the last engine look) and asks a new
   `_RunLog` "mutated since index" reader — the engine's look and the model's probes both record as
   `probe_render`, so a name scan would let a model probe reset the window. A model that replies
   without further mutation ends the turn (its reply stands; the last observation is already in the
   stream and the summary). No frame-hash heuristics — a pixel hash is trivially satisfied by any
   1px change AND spuriously unstable on live-value nodes; the CAP is the bound, tool facts are the
   gate. (A `met` verdict followed by further mutation re-looks too — seeing the newest frame is
   the point; bounded by the cap. So cap>1 turns may spend looks with no disagreement — that is
   working sight, not enforcement, and it's inside the cost statement.)
5. **Every SUCCESSFUL look keeps 053's additive contract; vision-off stays exactly today's no-op.**
   A look with `vision_ok` injects the observation (round-aware provenance — see 6) and
   `continue`s, giving the model one iteration — for EVERY verdict. `met`/`unclear` add nothing
   else. `not-met` adds one framing sentence: "the engine's eye reports the ask NOT met: <read>.
   Fix it, or reply honestly stating what is missing — do NOT claim it is done." (An honest staged
   reply — "stage 1 done, stars next" — is not a done-claim; both exits stay open. The loop never
   forces edits.) **Vision off / failed / no API key ⇒ NO injection and NO extra iteration** — the
   existing pinned behavior (`test_no_injection_when_vision_absent_from_probe`) is preserved
   verbatim; a vision-less user pays nothing new.
6. **Round-and-target-aware provenance.** `_auto_look_fact` currently asserts "you are ending the
   turn WITHOUT looking at the result" and "on the current frame" — the first is false on rounds
   ≥ 2 (and whenever the model did look), the second is false once A1 retargets the probe at a
   non-current node. The provenance text is parameterized by round AND names the probed node
   whenever it is not the current one. The engine must never state a falsehood on the model channel
   (decision 12).
7. **The verdict survives into the turn record.** A new explicit `TurnSummary` field (in
   `agent.py` — that's where `TurnSummary` lives; NOT a ledger line — the ledger soft-caps at 8 and
   is mutating-tools-only) rendered by `_render_summary` before the cap, e.g.
   `eye: ask not-met after 3 looks — <short read>`, emitted only when the final verdict is
   `not-met`. No persistence-schema change: history stores only the rendered NL string. The next
   turn's model and the user inherit the honest state.
8. **One knob, sibling-consistent semantics.** `copilot_convergence_max_looks: int = 3` — total
   engine looks per turn; `0` = off (no engine look, the master switch); `1` = a single aimed look
   (053/054-like, but now unconditional on the model's own looks — the one behavioral delta at
   cap=1); default 3. REPLACES the `copilot_vision_auto_look_on_turn_end` bool (verified readers:
   `config.py:62` + the agent gate — no tests read it, it was never Settings-reachable).
   Settings-tunable via the standard limits plumbing. `config.py`'s "seven user-facing" header
   comment is updated.
9. **Trace + observability, wired and tested.** Every parsed verdict (incl. `unclear`/garbled)
   emits a trace event (e.g. `ask_verdict`) carrying the raw ASK line and the probed node — with a
   DETERMINISTIC TEST per outcome (met / not-met / unclear / garbled), so the inertness detector
   cannot itself ship inert. The dogfood parse-rate is extracted by grepping the trace for the
   event kind (`grep 'ask_verdict' <trace>` — analyze.py is NOT extended; out of scope).
10. **Empty ask** (reachable only from the harness — the App rejects empty sends): no ASK line
    demanded, every verdict parses `unclear`, looks stay mutation-gated up to the cap (each a
    generic baseline read, no `not-met` framing possible) — the uniform rule, no special case.
11. **Interplay with the other brakes and gates:** a giveup/force-end path bypasses the look block
    (unchanged). `_clean_streak_fact`'s premise line ("none of which the user has seen, and you
    can't see the result either") is updated to acknowledge the engine eye. CONV looks are probes,
    never edits, and never feed the streaks. A turn that already performed a gated irreversible
    action (publish) CAN still be re-opened by a `not-met` verdict — accepted: the gate still
    guards any second attempt and the irreversible ledger is uncapped, so no new risk surface.

### Cost statement

Worst case per turn vs today: +2 iterations (cached-prefix re-sends — ~4× cheaper on hits within
the TTL — plus the working set) and +2 vision calls (~$0.002 each), PLUS the now-unconditional
first look on turns where the model looked itself (~$0.002). Bounded by the cap; default 3 chosen
so the 054 fixture (one fix-round short) converges with a round to spare. The micro-dogfood
records: (a) the ASK-line parse rate (all-`unclear` run = INCONCLUSIVE, never PASS), (b) the
not-met precision on an ask-IS-met fixture (see Manual verification), (c) the cost delta cap=3 vs
cap=1 — measured warm-vs-warm (run each arm twice inside the cache TTL, compare the second runs,
and report the trace's `cache=` rates alongside; a cold/warm mix gives opposite verdicts per the
skill's cost rule). The cap=1 baseline isolates the multi-round increment; the unconditional-look
increment is the flat ~$0.002 noted above, not part of that delta.

## Half 2 — robustness wave

**Slice A — auto-look targeting:**

- **A1. The engine look targets the node the turn actually changed.** `agent.py:759-763` hardcodes
  `"node": ""` (current node) while the look fires for edits with an explicit non-current target —
  the eye can inspect node A after the turn edited node B, and the injected fact then steers the
  model to a false "it did NOT land". Fix: `_RunLog` records the raw node-arg string (via the
  existing `_NODE_ARG_KEYS` vocabulary, NOT a new ad-hoc lookup) of the last successful
  render-authoring call, ONLY when it is a plain node address — a `lib:`/`example:` target (not
  probeable: the probe resolver returns None for them) records nothing and the look falls back to
  `""`/current, preserving today's useful look instead of erroring it away. The raw string is
  resolved by the existing probe resolver at execute time; `run_turn` holds no resolver. The
  injected fact names the probed node when non-current (CONV item 6).

**Slice B — script-edit brake parity:**

- **B1. `_edit_target_key` covers the script tools and namespaces the artifact kind.**
  `agent.py:111-120` reads `args["target"]`; the script trio's arg is `node` (`tools/script.py`),
  so every script edit keys to the `"<current>"` sentinel and all nodes' scripts share one streak.
  Fix: read through `_NODE_ARG_KEYS`, and key as a (kind, target) TUPLE — kind = shader|script by
  tool name — so a node's GLSL and its `script.py` (two files) don't share a streak. NOT a
  `script:<id>` string (that form reads as the rejected addressing overload). Also fix the sibling
  reader `agent.py:1067-1073` (the giveup note reads `args.get("target")` and misnames the node for
  script edits).
- **B2. A clean whole-file write resets ITS OWN kind's streak.** `agent.py:997-999` resets only the
  literal `"write_shader"`; a clean `write_script` — the sanctioned whole-file convergence for
  scripts — currently INCREMENTS the streak and can itself trip the hard stop. Fix: both write
  tools reset, each within its (kind, target) key.
- **B3. A broken script edit is not "clean".** `_format_write_result` (`tools/script.py:78-86`)
  returns `payload=None` on a compile error, so `applied_with_errors` (`agent.py:964-968`, keyed on
  `payload["errors"]`) never fires for scripts: the compile-thrash nudge is unreachable AND the
  broken edit counts toward the clean-edit streak — both signals inverted. Fix: the compile-error
  branch returns `payload={"errors": [...]}` where the value is a LIST of per-error entries
  (`session._tool_card_outcome` does `len(payload["errors"])` — a string would render "312 compile
  errors"). The force-restore branch (`backend.py:2034-2056` returns `ok=True` with the restore
  note in the SAME `compile_error` field) must NOT produce an `errors` payload — give the note a
  distinct `ScriptWriteResult` field (mirroring `EditResult.restored_note`) so a successful restore
  never registers as applied-with-errors. Note: the existing exact-payload pins in
  `tests/test_copilot_script_tools.py` cover the CLEAN branch only (unchanged by B3); the
  error-branch payload has NO coverage today — B3's tests add it (incl. the restore case). Also fix
  the stale `tools/base.py:72-75` comment ("edit_shader / write_shader" — stale since the script
  tools set `is_edit`).
- **B4. The one-write-per-file-per-batch guard covers scripts.** `_batch_mutated` is added at
  `backend.py:2234,2289` and CHECKED only in `apply_full_rewrite` (`:2327`); two `write_script`
  calls on one node in one batch both apply — the second composed against text the first replaced.
  Fix: mirror the shader shape exactly — `write_script` checks the guard, `edit_script` does not
  (the check lives at the write level, NOT in the shared `_apply_script_text` tail); keys join the
  same set under a non-address form (a tuple, per B1). Negative test: `write_script` then
  `edit_script` on one node in one batch must BOTH apply.

**Slice C — vision cost & visibility (one plumbing change with CONV):**

- **C0. `probe_render`'s capability signature returns a structured result.** Today
  `caps.probe_render` returns a bare `str` — no seam for a verdict, a vision-availability flag, or
  usage. Fix: it returns a small frozen dataclass (msg + vision_ok + verdict-or-None + usage-or-
  None); the tool handler keeps the model-facing `msg` EXACTLY as today (post-strip — CONV item 1)
  and forwards the rest via `payload` (payload never reaches the model). Touches
  `capabilities.py`, `backend.py`, `tools/inspect.py` (note: its `ok` is derived from
  `msg.startswith("error:")` — preserve), `tests/_caps.py` and the probe fakes in
  `test_vision_auto_look.py` / `test_probe_clock_and_turn_end.py` / `test_vision_probe.py`. This
  single change carries CONV's verdict, C1's usage, C2's classification, and C3's provenance key.
- **C1. Vision spend is accounted.** `describe_image` (`llm/openrouter.py:166-208`) returns only a
  string; its billed call is invisible to `TurnStats`/`session_cost_usd`. Fix: return text + usage.
  Mechanism note (grounded round 2): the STREAMING path opts into usage via
  `stream_options={"include_usage": True}` (`openrouter.py:244`) — a non-streaming call returns
  token counts natively but OpenRouter's `cost` field needs its own accounting opt-in; the
  implementer MUST verify the exact incantation against the installed SDK + OpenRouter docs before
  writing it (likely `extra_body={"usage": {"include": True}}`), and the C1 test asserts a NONZERO
  cost folds from a faked usage-bearing response. The fold happens in `agent.py` at the look call
  site (the only place `usage` accumulates — `TurnStats` reads it); COST only — vision token
  counts ride the per-look trace event, never `TurnStats.reply_tokens` (that gauge means the MAIN
  model's reply). A vision cache hit carries no usage and folds $0. The `describe_image` closure's
  declared type changes with it (`project_session.py:165-167`, `backend.py:499`).
- **C2. Vision failure is truthful.** `describe_image`/`_probe_png` swallow all errors to
  `""`/debug-log; the probe result then shows facts alone — indistinguishable from vision-off,
  while the system prompt promises an eye. Fix: when vision is ENABLED but the look failed, the
  probe `msg` gains one suffix: `(vision look unavailable this step)` — class only, no exception
  text (secret hygiene). The CONV loop treats vision-unavailable per item 5: no injection, no
  extra iteration. The `"visual (" in look_msg` gate at `agent.py:765` is replaced by the
  structured `vision_ok`.
- **C3. The engine's look is visible to the user.** Today the engine-initiated look yields no
  `AgentToolCard` — an invisible billed call. Fix: yield a card with `name="probe_render"` (keeps
  `label_for`, persistence, and the dogfood analyzer intact) and a payload-shape key
  (`payload["engine_look"] = True`) — the sanctioned per-RESULT vehicle — which
  `session._apply_event` renders as a VISIBLE tool-status line ("the engine checked the render
  against your ask"; today the line is gated on `ev.widget is not None` — this adds the second
  branch), not just an anonymous snippet square. Accepted: the snippet square itself stays an
  unattributed `probe_render` square (no synthetic tool names; the visible line carries the
  attribution). No per-card cost display (cards have no cost vocabulary; the spend rides
  `session_cost_usd` via C1). Live-app eyeball required (below).

**Slice D — context/harness hygiene:**

- **D1. Working-set reset moves into the session.** The per-turn clear lives in `app.py:613`
  (`copilot_send`); a `CopilotSession` driven directly (the harness's in-process REPL / multi-send
  mode, any future headless driver) never clears, so the working set accretes across turns —
  unbounded context growth and cost semantics diverging from the App. (The dogfood skill's
  canonical one-process-per-turn mode resets by construction — the divergence bites the multi-send
  mode.) Fix: a NEW `CopilotCapabilities.reset_working_set()` (the membership list lives on
  `ProjectSession._copilot_working_set`, `project_session.py:132`, reached only through injected
  closures — `CopilotSession` has no handle today): backend method + ProjectSession closure +
  `tests/_caps.py` member; `CopilotSession.enqueue_turn` calls it before the queue put; `app.py`
  drops its local rebind. A `logger.warning` if called while `in_flight` (mirrors
  `reset_conversation`). It also clears D3's evicted-this-turn record.
- **D2. `read_script` stops double-paying.** `read_shader` returns confirmation-only (source rides
  the working set — `tools/shader.py:270-318`); `read_script` both adds to the working set
  (`backend.py:1978`) AND returns the full listing — every script read pays twice. Fix: mirror
  `read_shader` (confirmation + error summary) for a node WITH a script; the STUB case keeps the
  inline listing (verified: the working set does NOT render a stub — `script_source_view` returns
  empty for a script-less node, so inline is the stub's only channel). Related polish:
  `create_node`'s "Read it before editing" (`backend.py:1041` already ws-adds the node) → reword to
  say the source is already visible below.
- **D3. The working set gets a size bound (true LRU, loud eviction).** `read_working_set` renders
  every member's full source + script each iteration, uncapped; `scratchpad_reserve_tokens` is only
  a trim-time reserve — overflow silently exceeds `max_input_tokens` (`prompt.py` keeps a 4-turn
  floor regardless). Fix, at the ADD seam (`project_session.py::_copilot_ws_add` — membership does
  NOT live in `backend.py`/`prompt.py`): `copilot_working_set_max_nodes: int = 6` (config-only, NOT
  Settings — an internals knob), move-to-end on re-touch (genuine LRU — oldest-added FIFO would
  evict the node being hammered); the just-touched node is newest by construction, and the current
  node needs no protection (`read_working_set` unions it in regardless of membership — the rendered
  set is cap+1 at most). Eviction is LOUD to the model: **the chosen surface is a signature
  change** — `read_working_set()` returns `(views, evicted_addresses)`; the evicted record lives on
  `ProjectSession` beside the list, accumulates within a turn, and is cleared by
  `reset_working_set` (D1); `render_working_set` (`prompt.py`) renders the block-level line
  "dropped from the working set (size cap): X — re-read to view". Touches the Protocol
  (`capabilities.py:323`), `tests/_caps.py`, the `session.py` consumer, and the
  `tests/test_working_set.py` pins. A silently vanished source is the "stale copy worse than no
  copy" class — hence loud. Documented interaction: an evicted lib-consumer leaves
  `invalidate_lib_consumers`' iteration (`backend.py:2307-2318`) — its compile state may go stale
  until re-read; the eviction line covers awareness (structural fix out of scope, trigger below).

**Slice E — misleading error paths:**

- **E1. A torn stream is not "incompatible model".** `agent.py:669-700`: `done is None` ⇒ `fr=""`
  ⇒ the user is told to pick a different model in Settings. Fix: `done is None` ⇒ "connection
  dropped mid-reply — try again" (no Settings advice); the incompatible diagnosis stays for a real
  unknown finish_reason. Note: unreachable via the shipped client (its `_stream_impl` always yields
  LLMDone; exceptions are caught upstream) — the fix is cheap truthfulness; its test uses a fake
  client omitting `LLMDone`; don't perturb the `fr` read at the look gate (`fr == "stop"`).
- **E2. A cancelled turn keeps screen == history.** `session.py:251-253` drops `streaming_text`
  with no assistant bubble while the cancel terminals (`agent.py:636,865,899`) carry `text_buf`
  into the committed summary — the user loses text the history keeps. Fix: on `AgentCancelled` with
  non-empty summary reply text, append the partial text as the visible assistant message (screen
  keeps what history keeps). Do NOT touch the error-terminal ghost-drop
  (`tests/test_torn_stream.py:91-104` pins it deliberately).
- **E3. `content=None` never reaches the wire.** `_render_summary` returns `None` for an all-empty
  turn (`session.py:425-437`, reachable via Stop at iteration 0); `_commit_turn` appends it and
  `_to_wire_message` emits a literal `content: null` assistant message (`openrouter.py:84`) — some
  providers 400; it also persists silently. Fix: a minimal PLACEHOLDER summary ("(turn ended with
  no reply)") — keeps the documented user/assistant pairing invariant (skipping only the assistant
  message would break it; skipping both would drop the user's real message).
- **E4. Precheck handoffs are visible.** `agent.py:877-883`: a deflected call (e.g.
  `publish_telegram` without credentials) yields no card and no counter increment — the user sees
  nothing. Fix: yield the card with `ok=True` and `payload={"handoff": True}` — the payload-shape
  branch in `_tool_card_outcome` renders a neutral "handed off" outcome, and `ok=True` keeps the
  snippet square (`StepRecord.ok` colors it, `widgets/copilot_chat.py`) from reading as a red
  failure (a deflection is not a failure). `total_tool_calls` is NOT incremented (it feeds the
  incompatible heuristic and giveup notes).

## Out of scope (each with a trigger)

- **Reference-image targets** ("match this picture"). Unchanged from 054. **Trigger:** users want
  to supply a reference to match.
- **Parsing ASK verdicts from the MODEL's own `probe_render` calls** (their `look_for` is the
  model's, not the ask — a verdict against the wrong question; the line is stripped from their
  results either way). The unconditional engine look covers enforcement. **Trigger:** traces show
  the engine look duplicating a model look often enough to matter in cost.
- **Routing the forced no-tools reply through the eye** (CONV item 3's accepted residual).
  **Trigger:** a trace where an over-claim ships via the forced-reply path.
- **Ask-shape classification / gating CONV off non-visual asks structurally.** v1 relies on the
  contract's mandated `unclear` + the ask-IS-met precision metric. **Trigger:** the micro-dogfood's
  not-met precision shows the skeptical eye systematically flagging satisfied/non-visual asks.
- **Smarter single-node truncation / token-based working-set budget.** D3 caps member COUNT only.
  **Trigger:** a trace where one legitimately huge node (not accretion) overflows the reserve.
- **Structural fix for evicted lib-consumers** (invalidation iterating all project nodes instead of
  the working set). **Trigger:** a real trace where a stale evicted consumer ships a wrong compile
  state to the model.
- **A structured cross-turn convergence memory** (the eye's verdict driving the NEXT turn's plan).
  v1 records the verdict in the summary only. **Trigger:** a dogfood where the model re-claims
  success next turn despite the recorded not-met verdict.
- **Extending `scripts/dogfood/analyze.py` for verdict events** (extraction is a trace grep).
  **Trigger:** a second dogfood metric wants the same extraction.
- **`registry.status_for` dead `args` param, `GateKind.FILE`/BULK unreachable branches, numpy-ifying
  `render_facts`.** Cleanups, not defects. **Trigger:** next feature touching
  `tools/registry.py` internals / the facts hot path respectively.
- **Per-tool actionable messages for unexpected tool exceptions** (`tools/registry.py:146-150`,
  class-name-only — deliberate secret hygiene). **Trigger:** a trace where the model retry-loops on
  a swallowed exception class.

## Design decisions (locked)

1. **Verdict = data from the eye, parsed AND STRIPPED by the engine, binary + mandated `unclear`.**
   The shared contract module (`vision_contract.py`) pins prompt↔parse↔strip. The raw label never
   reaches any model-facing string. Degradation: `unclear`/`met` (vision working) behave as 053's
   single inject-and-continue; vision-off/failed/keyless behaves as today's NO-injection no-op
   (existing pin preserved). Further looks fire on further MUTATION within the cap regardless of
   verdict (item 4); the `not-met` FRAMING (and the summary record) ride only an affirmative
   `not-met`.
2. **Loop gating is tool-facts only**, tracked by an explicit loop-local marker + a `_RunLog`
   since-index reader (never a tool-name scan — model probes share the `probe_render` name).
3. **The engine look is unconditional on the model's own looks**; the forced-reply path is the one
   stated, triggered residual.
4. **`copilot_convergence_max_looks` replaces the auto-look bool** (0=off master switch, 1=single
   unconditional aimed look, default 3; Settings-tunable). `copilot_working_set_max_nodes` is
   config-only.
5. **A1 records the RAW arg string** (via `_NODE_ARG_KEYS`), node-addresses only; `lib:`/`example:`
   targets fall back to current-node. No resolver in `run_turn`.
6. **B1/B4 keys are (kind, target) tuples**, never `<kind>:` strings (the rejected addressing
   form).
7. **B3's `errors` payload is a LIST shape-compatible with the shader tools'**; the force-restore
   note moves to its own result field and never produces `errors`.
8. **C0 is the single plumbing change** for verdict/vision_ok/usage/provenance; the model-facing
   `msg` string is byte-identical to today's AFTER the ASK-line strip (which exists only when the
   new contract line was emitted — pre-056 outputs carry nothing to strip).
9. **C1 folds COST only** into `TurnStats.cost_usd`/`session_cost_usd`, in `agent.py` at the look
   site; vision tokens ride the trace event, never `reply_tokens`. The usage-request incantation is
   verified against the installed SDK before implementation.
10. **C3/E4 visibility rides payload-shape keys** (`engine_look` / `handoff`) — the sanctioned
    per-RESULT rendering vehicle; E4 cards are `ok=True`; no synthetic tool names, no per-card
    cost.
11. **D1's reset is a capability method called at ONE choke point** (`enqueue_turn`); it clears the
    D3 eviction record too.
12. **All engine-injected texts are facts-as-data** (skill §4) and must be TRUE in context (round-
    aware, target-aware provenance) — a false engine statement on the model channel is the
    misleading-tool-message class.
13. **D3's eviction surface is the `read_working_set() -> (views, evicted)` signature change** —
    explicit over clever; the Protocol, fakes, session consumer, and prompt renderer move together.
14. **NO backward-compat/migration anywhere** (hard rule). Config/integration fields follow the
    existing additive-default pattern; the deleted bool's readers are simply updated (pre-release,
    no shims).

## Files touched

- `shaderbox/copilot/vision_contract.py` (NEW) — ASK contract constants + parse/strip helpers.
- `shaderbox/copilot/agent.py` — CONV loop (counter, unconditional gate, verdict handling, loop-
  local look marker + `_RunLog` since-index reader, round/target-aware `_auto_look_fact`,
  `TurnSummary` verdict field — `TurnSummary` lives HERE), C1 cost fold at the look site, A1
  (`_RunLog` target recording), B1/B2 keys + the `:1067` sibling reader, E1, E4,
  `_clean_streak_fact` reword.
- `shaderbox/copilot/llm/openrouter.py` — `_VISION_SYSTEM` reword + ASK contract line (from
  `vision_contract`), `describe_image` usage opt-in + (text, usage) return.
- `shaderbox/copilot/backend.py` — C0 structured probe result, verdict parse+strip
  (`vision_contract`), C2 suffix + vision_ok, B4 script-write guard check (the `_batch_mutated`
  annotation widens to tuple keys), B3 restore-note field, D2 result rewording, `describe_image`
  dep type (`:499`), the `read_working_set` implementation (`:797`) + the new evicted-record
  reader closure for D3.
- `shaderbox/copilot/capabilities.py` — C0 signature, D1 `reset_working_set`, D3
  `read_working_set` shape, B3's `ScriptWriteResult` restore-note field (`:239`).
- `shaderbox/copilot/tools/inspect.py` — payload forwarding (C0/C3; `ok` derivation preserved).
- `shaderbox/copilot/tools/script.py` — B3 payload; `tools/base.py` — stale comment.
- `shaderbox/copilot/session.py` — CONV summary rendering in `_render_summary`, D1 call in
  `enqueue_turn`, E2, E3, C3 visible-line branch + E4 `handoff` outcome (`_apply_event`,
  `_tool_card_outcome`), D3 consumer.
- `shaderbox/copilot/config.py` — the two knobs (bool deleted; header comment count updated);
  `exporters/integrations.py` — `CopilotIntegration` field + `apply_user_limits` call;
  `popups/settings.py` — the limits row.
- `shaderbox/project_session.py` — D1 closure, D3 LRU + eviction record at `_copilot_ws_add` +
  the evicted-record closure wiring, `describe_image` closure type (`:165-167`).
- `shaderbox/copilot/prompt.py` — `render_working_set` eviction line (D3).
- `shaderbox/app.py` — drop the local working-set rebind.
- `shaderbox/copilot/trace.py` — the `ask_verdict` + per-look usage events (if a new kind is
  needed).
- Tests — `tests/_caps.py` (probe signature, reset member, `read_working_set` shape),
  `test_vision_auto_look.py` (rewritten around the int knob: multi-round, unconditional-on-looked,
  cap, verdict gating, vision-off no-injection pin KEPT, round/target-aware provenance, A1
  targeting incl. the `lib:` fallback, ASK-strip pin), `test_probe_clock_and_turn_end.py` /
  `test_vision_probe.py` (signature), `test_copilot_script_tools.py` (B3 error-branch + restore-
  case coverage — clean-branch pins untouched), new brake-parity tests (B1 tuple keys, B2 per-kind
  reset, B3 thrash-nudge-for-scripts), B4 batch pair test, D1 reset test (bare `CopilotSession`,
  NO app fixture — the X-crash todo forbids joining an app-fixture module; D3's NEW tests likewise
  live in a bare module even though `tests/test_working_set.py`'s existing pins get updated in
  place), D3 LRU/eviction tests, E1-E4 tests, C1 nonzero-cost fold test, verdict-trace-event tests
  (met/not-met/unclear/garbled), prompt-contract test (`_VISION_SYSTEM` carries the constant, the
  parser accepts the demanded format, the strip removes the line, ordering after the `look_for:`
  segment), `test_copilot_user_limits.py` (knob name).

## Manual verification

Deterministic (headless; each check fails for exactly one reason; consumers named):

- CONV wiring falsifiers: cap=1 ⇒ multi-round test red (reader: the look counter); garbled ASK
  line ⇒ parses `unclear`, injects-and-continues with NO not-met framing (reader: the parser
  fallback + decision 1's additive path); no-mutation-after-look ⇒ no re-look (reader: the
  since-index gate); `not-met` + mutation ⇒ re-look with the not-met framing (reader: the injected
  fact text); `unclear` still INJECTS and re-opens one iteration (053 parity); vision-off ⇒ NO
  injection (the kept 053 pin).
- Prompt-contract falsifiers: constant absent from `_VISION_SYSTEM` ⇒ red; parser rejects the
  demanded format ⇒ red; strip leaves the label in `msg` ⇒ red.
- Trace falsifiers: each verdict outcome (met/not-met/unclear/garbled) emits `ask_verdict` with
  the raw line ⇒ cut the event ⇒ red. Summary falsifier: a not-met final verdict appears in
  `_render_summary`'s output ⇒ cut the field ⇒ red.
- A1: spy asserts the probed node == the last-mutated plain-node target; a `lib:` last-target
  falls back to current (both directions pinned).
- B3 falsifier: revert the payload fix ⇒ script-thrash test red; force-restore case ⇒ NO errors
  payload (revert ⇒ red).
- D1 falsifier: two `enqueue_turn`s on a bare `CopilotSession` ⇒ working set empty at turn 2 (no
  LLM needed).
- D3: 7th member evicts the LRU (not the re-touched one — reorder pinned), the evicted address
  appears in the rendered block line, and `reset_working_set` clears the record.
- C1: a faked usage-bearing vision response folds a NONZERO cost into the turn stats.
- C2 falsifier: vision enabled + a failing look ⇒ the `msg` carries the unavailable suffix; vision
  DISABLED ⇒ no suffix (both directions pinned — the suffix on the wrong branch is the bug).
- D2 falsifier: `read_script` on a scripted node returns NO listing in the model-facing body (the
  working set carries it); on a script-less node the stub listing IS returned inline.
- E1-E4, B1/B2/B4: as listed in Files-touched tests.
- `make check` + the copilot test modules green.

Behavioral (micro-dogfood on this box, WSL Mesa headless, cheap model — the real gate):

- **Fixture 1 (054 one-shot flag, ask hard to meet):** PASS = the turn does not end with an
  over-claim while the eye reports not-met — either the loop drove the fix or the reply honestly
  states the gap. Record the ASK-line parse rate from the trace (`grep ask_verdict`) — an
  all-`unclear` run is INCONCLUSIVE, never PASS.
- **Fixture 2 (ask-IS-met, e.g. "make the background solid dark blue" on a fresh node):** records
  not-met PRECISION — a satisfied simple ask must not draw a `not-met` verdict / phantom extra
  rounds. A systematically-wrong skeptical eye here fires the ask-shape out-of-scope trigger.
- Cost delta cap=3 vs cap=1, warm-vs-warm with `cache=` rates reported (per the cost statement).
- A script micro-mission: trace shows tuple brake keys (no `<current>` pollution), a deliberately
  broken script edit ⇒ the thrash nudge fires.
- Harness working set: two sends in ONE process ⇒ turn-2 request's working set contains only
  turn-2 reads.

Live-app (maintainer, next `make run` on a display box — this box is headless): eyeball the C3
visible engine-look line + the E4 neutral handoff card + the Settings row for the new knob. **Fold
the FULL standing `todo.md` [VERIFY] 053 checklist into the same run so the entry can be deleted:**
the four vision-badge states (supports / no-image / not-recognized / couldn't-verify), no
`status_slot` height jitter, no daemon-thread re-kick storm on rapid model-field edits, and one
real `probe_render(look_for=...)` on an animated node (the 3-frame strip read) against a live key.

## Open questions for the user

None blocking — the maintainer delegated autonomous execution (2026-07-27); robust defaults per
dev_flow. Judgment calls a reviewer may still challenge: the default cap 3; cost-only fold; the
`(views, evicted)` signature choice for D3.

## Review history

**Round 1 (2026-07-27, 3 Opus reviewers: design-vs-conventions, verification/blast-radius,
feasibility-vs-code): PARTIAL / PARTIAL / FAIL-on-Half-1.** All Half-2 defect claims re-verified
REAL (incl. the three then-relayed E2/E3/E4). Convergent criticals, folded in v2: witness-vs-judge
contradiction → reworded contracts + shared constant + binary verdict; the claimed frame-hash "already
exists" was FALSE → tool-facts loop control + honest C0 plumbing; the `looked()` opt-out → unconditional
engine look; `unclear`⇒no-re-open would have regressed 053 → additive degradation; working-set ownership
mislocated (`project_session.py`, not `backend.py`) → D1 capability + D3 LRU at the add seam; C3 card
cost promise unachievable → payload-shape visible line; knob 0/1 collision → one int knob; B3 list shape
+ force-restore exclusion; B1 tuple namespacing; B4 write-level check; C1 usage opt-in + cost-only fold;
files-touched completeness. Rejected: parsing ASK from the model's own probes (deferred with trigger);
pixel-hash even as dedup (dropped from control flow).

**Round 2 (2026-07-27, 2 Opus reviewers: design-resolution, verification/blast-radius): PARTIAL /
PARTIAL — targeted edits, no redesign.** Folded in v3: the ASK line must be STRIPPED from all
model-facing text (it would have leaked into the model's own probe results — the witness hole at
another surface) + the shared module gets the strip helper + a neutral home (`vision_contract.py`,
`backend` must not import `llm/`); vision-off degradation pinned to today's NO-injection no-op (v2's
letter would have added an iteration for every vision-less user and killed an existing test pin);
skeptical-eye false-`not-met` risk → mandated-`unclear` wording + an ask-IS-met dogfood fixture with a
recorded precision + an ask-shape out-of-scope trigger; the forced-reply unenforced exit stated as a
triggered residual; the verdict trace event + summary field got their own deterministic falsifiers
(the round-1 "inert detector" class one level up); D3's eviction surface locked to the
`(views, evicted)` signature change with the record cleared by D1's reset; `TurnSummary` relocated to
`agent.py` (v2's `state.py` row was a round-1 relay error; no persistence implication — history stores
the rendered string); C1's mechanism claim corrected (streaming opts in via `stream_options`, not
`extra_body`; incantation to be SDK-verified) + the fold site named (`agent.py`) + the closure type
(`project_session.py:165`); `_auto_look_fact` must name a non-current probed node; E4 pinned to
`ok=True` + `handoff` payload (StepRecord/chat squares color on `ok`); loop bookkeeping pinned to a
since-index reader (a name scan would let model probes reset the window); cap=1 ≠ 053 exactly (the
unconditional look is the stated delta) + the cost-delta measurement made warm-vs-warm with `cache=`
rates; the 053-badge todo fold expanded to the FULL checklist; B3's test-pin claim corrected (clean
branch untouched; error branch was uncovered). Accepted as residuals (with triggers): forced-reply
exit; met-verdict re-looks spending the cap on busy turns; post-publish re-open (gates still gate).

**Round 3 (2026-07-27, 1 Opus convergence checker): PASS — "lock and implement".** All 13 round-2
blocking items verified resolved in text; every spot-checked code claim held. Two editorial fixes
applied in the same wave: decision 1's stale "only not-met buys extra rounds" clause aligned with
item 4 (mutation gates re-looks; not-met gates only the framing + summary), and the empty-ask path
unified (gate doesn't read the ask; verdicts parse `unclear`; no special one-look case). Files-
touched gained the `read_working_set` impl + evicted-record closure (backend/project_session),
`ScriptWriteResult`'s home (`capabilities.py`), and C2/D2 got named falsifiers. Spec LOCKED
(maintainer-delegated autonomous lock).

**Post-impl (2026-07-27, 3 Opus reviewers + 1 final checker): PASS after one fix wave.** The 15-item
consolidated batch (red `test_content_editing` pin — the one module outside the implementer's run
list; eye-summary truncation; engine-look card honesty gates; eviction short-ids + rendered-set
filter; stale eye-note; model-probe usage fold; handoff visibility; contract parse-last/strip-all;
conditional ASK instruction; cap=0 semantics; blind-look trace; multi-round reply accumulation)
applied by the same implementer; final checker verified all 15 landed. `make check` 0 errors; full
per-module sweep 59 modules green (4 pre-existing headless-GL segfault modules unchanged).

**Micro-dogfood (2026-07-27, WSL + `GALLIUM_DRIVER=d3d12` GPU, `codex-mini`):** ASK parse rate
**9/9** looks across 7 runs (met×2, not-met×7, zero `unclear`/blind); ask-IS-met fixture: 1/1
correct `met`, no false not-met, no phantom rounds. One run showed the full arc not-met → fix →
`met`. The 054 flag fixture ended with NO over-claim — the model honestly reported the remaining
gap and asked to continue (the 054 failure mode is dead); brakes fire on script edits (B1 live);
the C3 attributed line renders. Cost (warm pairs, cache 81-87%): cap=1 $0.037 vs cap=3 $0.136 on
the flag one-shot — but single-sample arms with edit-volume variance dominating (a 1-look cap=3 run
cost $0.110), so read it as "multi-round ≤ ~2-3× on a hard ask, bounded by the cap", not a precise
increment. **Craft findings for the NEXT slice (not 056 defects):** (1) the flag's ANIMATION is
incoherent — the canton deforms independently of the stripes (per-layer domain warps instead of one
cloth field); the craft block lacks a "one object = ONE deformation field" lesson; (2) the eye
cannot judge MOTION COHERENCE — the 3-frame strip answers "does it change" but not "do the regions
move together", so the defect sailed past two not-met verdicts aimed at stars/dullness; (3) the
maintainer-proposed fix for judging: ground-truth-checkable dogfood fixtures (physics/logic/3D base
cases) instead of the all-axes-at-once flag.