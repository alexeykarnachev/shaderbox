# 020 capability-wave — locked decisions (the conversation, distilled)

> The verbatim-intent record of the maintainer↔Claude design conversation that PRECEDES
> `11_capability_wave_spec.md`. The spec must honor every line here. Review agents check the
> spec against THIS file (plus `99_synthesis.md §0`, which still binds). If the spec and this
> log disagree, this log wins; if this log and `99 §0` disagree, the later decision (here) wins
> and the spec must note the override.

## A. Interface & scope
- A1. ONE free-form chat. No canned/discrete actions, no buttons-as-modes. (`99 §0 #3`)
- A2. The user expresses ANY action in natural language (precise or vague); the agent plans + executes under the hood.
- A3. Provider = OpenRouter via the `openai` SDK. Model TBD but a sane default is picked on first run. (`99 §0 #5`)
- A4. Hard sandbox: the agent has NO python/shell/code execution, NO OS introspection, NOT EVEN
  awareness of the OS name. Its entire universe = ShaderBox's own capabilities. No `run_shell`,
  no editing ShaderBox's own python source. Stated in the system prompt (Layer 1).
- A5. The agent has NO computer vision. It CANNOT see rendered pixels. Its only render-correctness
  oracle is `compile_unit.errors`. The system prompt says this explicitly; the agent must never
  claim a visual result it cannot verify — it describes what it changed and asks the user to look.

## B. Context strategy
- B1. The current shader source is ALWAYS in the agent's context, in full. Accepted token cost;
  it is the core value. It is a PROMPT-LEVEL entity (always FRESH, never accumulating stale copies),
  NOT a history entity — history never accumulates old shader versions.
  - B1a. (refined 2026-05-31) PLACEMENT = via the `get_current_shader()` TOOL, NOT an always-present
    system/context block. Reason = prompt-CACHE health, verified against Claude Code (`cli.js` uses
    `cache_control`/`ephemeral` + a "File has been modified since read → re-read" freshness guard;
    files come via the Read tool, not a context block). During active dev the source CHANGES every
    turn; a volatile block in the cache-warm FRONT of the prompt (the old "always in context" idea)
    busts the prefix cache for everything after it every turn — the worst case. Via-tool puts the
    volatile source AFTER the stable prefix (identity/rules/tool-specs/lib-catalogue), so the
    expensive prefix stays cached; you pay full price only for the fresh source bytes the turn it's
    fetched. So via-tool is USUALLY CHEAPER than always-in-context, not more expensive.
  - B1b. FRESHNESS rule (our simplification of CC's "modified since read"): the editor is LOCKED for
    the whole turn (decision E) and there is exactly ONE current node, so the source can change ONLY
    between turns. => at prompt-assembly, STRIP prior `get_current_shader` results out of history
    (replace with a short `[shader source from turn N — re-fetch with get_current_shader]` marker, the
    cc-server `_compress_old_tool_results` move). The agent re-fetches fresh each turn (read-before-edit
    discipline); history never carries a stale or duplicated source. Consequence: the system prompt
    instructs "read the shader with get_current_shader before editing it."
- B2. The library is the token-relief valve: the agent sees lib functions as SIGNATURE + DESCRIPTION,
  NOT bodies (bodies pulled in only when the agent greps/reads explicitly). => good lib descriptions
  + signatures matter. GROUNDED: `ShaderLibFunction` already carries `signature`, `doc` (`///` block),
  `calls`, `body` — so the catalogue is extractable TODAY; only a `///` doc-comment authoring
  convention needs encouraging. "The lib should dance around the agent, not vice versa" — we adjust
  the parser/lib system for the agent's convenience as needed.
- B3. The highest context altitude is the PROJECT. No cross-project hopping (maybe later). One agent
  instance per project: chat history + agent state stored PER-PROJECT. (This OVERRIDES `99 §0 #6`
  which put the transcript under a global `app_data_dir()` — now it is per-project.)

## C. Tools
- C1. Tools of varied granularity: grep (across lib impls / other nodes' code / docs / everything),
  render, compile, editor management, code edit, create_node, lib CRUD, telegram ops, youtube ops, etc.
- C2. NO agent-inside-the-agent tools (no `do_something(prompt)` that hides a sub-agent). Flat,
  mechanical tools the OUTER agent composes — this keeps the in-process compile-feedback loop honest
  (the outer agent sees the error and self-corrects).
- C3. Use NATIVE OpenRouter/OpenAI tool-calls (NOT a hand-rolled structured-output protocol). Decided
  after pushback: the "navigable catalogue for an enormous tool count" need is real but SEPARABLE from
  the wire protocol — solved by a meta-tool (search_tools / list_tools) loading schemas on demand
  (the ToolSearch pattern), on TOP of native tool-calls. Native calls = provider-tuned reliability on
  cheap models + the existing seam (LLMToolSpec.parameters, role:"tool" results, streaming
  ToolCallStarted/Completed) already built for it.
- C4. Tool-count scaling = eager-core + lazy-long-tail. The ~8–10 most-used tools always carry full
  schemas (zero discovery hop); the long tail lives behind search_tools/list_tools. The split is
  DEVELOPER-tunable config (in CopilotConfig or sibling), INVISIBLE to the user.
- C5. The catalogue tree (category → tool summaries) is GENERATED from the registry
  (`ToolDefinition` + the existing `ToolRegistry.describe()`), single source of truth, no parallel doc.

## D. Environment-awareness (three layers, by volatility)
- D1. Layer 1 — static capabilities MAP in the system prompt (~30-50 lines): what ShaderBox is, what
  the agent can do, the sandbox boundary, the no-vision fact, and "for step-by-step how-tos call
  read_doc(topic)". Orientation, not schemas.
- D2. Layer 2 — the tool catalogue via meta-tools (= C3/C4/C5).
- D3. Layer 3 — how-to docs: a dedicated set of hand-written markdown files (one per workflow:
  youtube_upload, telegram_stickers, lib_authoring, rendering, …), exposed via list_docs()/read_doc()
  and grep. The SAME files render in-app for the user later (one source, two readers — no duplication).
  Docstring-derived API-doc auto-extraction is DEFERRED to a separate later feature, NOT this wave.
- D4. A DOCS ANTI-DRIFT step is added to `/sanitize` + `dev_flow.md`: the how-to docs must match the
  agent's REAL capabilities (same discipline as the roadmap-row / freetype-glyph-atlas drift rule).

## E. Editor concurrency — kept EXTREMELY simple (the "deep hole" we refuse to dig)
- E1. When the user interacts with the copilot, the editor is LOCKED for the whole turn.
- E2. At the START of each turn: lock editor → auto-save (silent flush, NO prompt) → snapshot the
  current code into the prompt. The agent operates atomically/non-concurrently.
- E3. The agent NEVER asks the user "save or not?" — that is explicitly rejected as nonsense.
- E4. NO dirty-buffer refuse-guard, NO force-flush question, NO concurrent-edit handling. This
  DISSOLVES `99 §0 #1`'s "falling-edge auto-flush hook + refuse-guard" — we just flush at turn start.
- E5. Optional opt-in/out: auto-focus the editor onto the file the agent is changing, so the user
  watches edits land live. Internal compile-retry attempts ARE visible (code changes in the locked
  editor + new renders) — a feature, not noise.

## F. The gate / interactive-widget mechanism (first-class engine primitive)
- F1. A special agent message type that BLOCKS the agent loop until the user responds — the safety/
  confirm/disambiguation primitive, generalized. (`99 §0 #9`; `state.py` already scaffolds the
  `pending_action` role + `resolved` flag.)
- F2. The agent PROCEDURALLY generates a set of options the user must choose from before the loop
  continues. Pre-defined templates: yes/no, yes/no/cancel, free choice list, etc.
- F3. This is generalizing into a FAMILY of interactive, agent-blocking, in-chat widgets — not just
  confirm. Also includes: a CREDENTIAL/SECRET input widget (for entering an OpenRouter / Telegram /
  YouTube key inline, for user convenience) when a capability needs a key that is absent.
- F4. Confirm policy (what triggers a gate): destructive ops (delete node/lib file) ALWAYS confirm;
  external publish (telegram/youtube — anything leaving the machine) ALWAYS confirm regardless of
  bulk; bulk ops above a threshold confirm; single reversible edits / uniform sets flow free.
- F5. Missing-credential is a GUIDED HANDOFF, not a tool failure: the agent points to the relevant
  Layer-3 how-to doc and/or surfaces the inline credential widget (F3) — never a raw error.

## G. Agent status is a first-class citizen
- G1. The agent streams its current status to the chat (status pill / bubble — exact UI tuned later).
- G2. A concise auto-generated status message per tool call is part of the protocol, not an
  afterthought (cf. cc-server `_format_tool_status` templates).

## H. Render-to-file
- H1. NO async/parallel batch rendering. Renders run SEQUENTIALLY on the GL main thread. "Render 20
  videos" = the user waits. Explicitly NOT worth the concurrency complexity for v1.
- H2. BUT provide a first-class engine primitive: an AGENT PROGRESS BAR the agent can push updates to
  (sibling to the gate widget family — F/G) for long sequential work.
- H3. Artifact destination = an explicit output path; the agent picks a sensible default (project
  `.trash/` or a `renders/`-style dir) and ALWAYS tells the user the path. Never silently /tmp.
- H4. Image-vs-video is the agent's inference from the in-context source (does it use `u_time`?).
  No decision-tool needed — just `render_image` + `render_video` tools; the agent picks.
- H5. (R3, decided 2026-05-31) Render = ONE blocking `bridge.run_on_main` op; the UI FREEZES for the
  encode, EXACTLY like clicking render in the Share tab today (shipped behavior — `share_tab.update`
  main-thread `for i in range(n_frames)` ffmpeg loop). NO chunking (a `core.py` refactor we declined),
  NO worker-thread render (GL is main-thread-only). Mitigations: (a) a global "Rendering '<node>'…
  please wait" modal via the existing `ui_primitives.modal_window`, painted ONE FRAME BEFORE the freeze
  via a two-phase commit (fast bridge op sets `app.copilot_render_status` → frame draws modal → next
  frame runs the encode → clears it); (b) `render_op_timeout_s = 60.0` (new `CopilotConfig` field) via a
  `bridge.run_on_main(fn, timeout=…)` overload — the 5s default would trip on any real encode. H2's
  within-render progress bar is DROPPED (a frozen frame loop can't paint it); the BATCH bar (node N of
  M, driven by the loop between calls) survives.

## I. History / cost / loop limits
- I1. History does NOT accumulate shader versions (= B1). The real history bloat is TOOL RESULTS;
  compress old tool-result rounds to short markers (cc-server `_compress_old_tool_results`) rather
  than hard-stopping. (This SUPERSEDES the earlier "hard-stop on overflow" lean.)
- I2. Bounded self-correction retry loop on compile errors: the agent reasons on pre-existing AND
  self-introduced errors, fixes, retries — with a reasonable retry cap; on exhaustion it stops and
  shows the user the error rather than burning the full iteration budget.
- I3. `max_iterations` / `max_input_tokens` / `max_tokens_per_turn` are constants in `CopilotConfig`
  (already scaffolded), NOT user-tuned, NOT on `UIAppState` (avoids the migration discipline).
- I4. On cutoff (max-iterations / budget): emit a "here's what mutating work I already did" note
  (cc-server `_executed_actions_note`), driven by the mutating-tools set.

## J. Reference-implementation lessons to honor (ovelia + cc-server, re-read this session)
- J1. Prompt assembly ordered LEAST-VOLATILE → MOST-VOLATILE for OpenRouter prefix-cache friendliness
  (ovelia `build_system_prompt` / `build_user_message`). For us: static identity+rules+capabilities-map
  → lib catalogue (rare) → project/node list (rare-ish) → CURRENT SHADER SOURCE + compile errors
  (per-turn, volatile) → history → the new user message.
- J2. The agent loop is CONVERSATION-LIST based (cc-server `AgentLoop.run`): deep-copy messages, append
  the assistant tool-call message + each `role="tool"` result, re-stream. => the stubbed
  `run_turn(client, registry, config, user_text)` signature is TOO THIN; it needs the conversation/
  history list, not just `user_text`. SPEC MUST flag this seam change.
- J3. Tool results are `Error:`-prefixed STRINGS, never raise; generic message only, detail to the log
  — exception text leaks internal data into LLM context (cc-server `ToolRegistry.execute`; our
  `tools/registry.py` already does this).
- J4. NEVER `logger.exception` on an upstream LLM error — OpenRouter error bodies echo the prompt
  (secrets/PII) (ovelia `_log_upstream_error`).
- J5. grok footgun: assistant `content=""` + tool_calls breaks grok → coerce empty content to None on
  the wire (ovelia `_to_wire_message`).
- J6. grok footgun: tool-call arguments can be DOUBLE-ESCAPED JSON → un-escape (cc-server
  `_unescape_double_escaped_strings`).
- J7. OpenRouter streaming specifics: `max_completion_tokens` (not `max_tokens`),
  `stream_options={"include_usage": True}`, `extra_body={"reasoning": {"effort": "minimal"}}`; cost
  from `usage.cost` and it CAN be None per-chunk — accumulate carefully (ovelia `_stream_impl`).
- J8. Tool-call streaming accumulation: per-index `_ToolCallBuilder` (id/name set once, arguments
  concatenated across deltas), emit Started when id+name first known, Completed at stream end.
- J9. Sanitize user-supplied text spliced into the prompt (strip control chars) — prompt-injection
  guard (ovelia `_sanitize_title_for_prompt`, cc-server `sanitize_name`). For us: shader source,
  node names, lib names spliced into the prompt are DATA, never instructions; mark history inert.
- J10. Model fallback chain by role + IPv4 transport pin is OPT-IN for the maintainer's dead-v6 box,
  default dual-stack (`99 §0 #4`, ovelia `_build_client` does the v4 pin; we keep it transparent/opt-in).

## K. What STAYS from the synthesis (not re-litigated)
- worker-first build order; one plain threading.Thread + blocking queue (NOT asyncio); the
  CopilotBridge worker→main GL round-trip; shader edits = write the .glsl file (GL-free) → existing
  hot-reload recompiles → read errors back via ONE bridge round-trip (the edit tool returns the
  post-recompile errors, don't poll); CopilotCapabilities frozen-dataclass leaf seam (app→copilot
  one-way, cycle-free); pydantic `model_json_schema()` for tool schemas; the `copilot/` package
  mirrors `exporters/`; `app.py` gains a handle + 2 drain calls + the capabilities builder, NOT a split.
