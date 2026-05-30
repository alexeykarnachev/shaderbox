# 020 Copilot Agent — research synthesis + refactor-prep plan

> **The merged conclusion of the research phase.** Reconciles the angle reports (`01`–`09`) + the
> grounding (`00`) into one picture. This is NOT the feature spec — it's the input the spec is
> written from.
>
> **Method note:** 7 parallel research agents (`01`–`07`), each anchored to real source (re-read
> 2026-05-29, file:line cited throughout) + the three reference agents (cc-server / marginalia /
> ovelia). Two follow-up investigations (`08` auto-save, `09` LLM-layer) were spawned after the
> maintainer's first round of answers. The reports **converged with no contradictions**; two factual
> claims that could have been hallucinations were verified directly against the repo (see §6).
>
> **⚠️ READ §0 FIRST — the maintainer's decisions override several of the original reports'
> assumptions** (notably: OpenRouter NOT Anthropic; one free-form chat, NO canned "explain"-style
> actions; this is architecture-shape work, not a capability/phasing plan).

---

## 0. Maintainer decisions (override the reports where they conflict)

Answers to the §7 open questions + the reframing they imply. **These win over any conflicting
statement in `01`–`07`.**

1. **Auto-save the editor — generalized, always.** The editor auto-flushes whenever it could go
   inconsistent (focus-loss, shutdown, node/lib switch, any external action). Investigated in
   **report `08`**: the verdict is a **generalized hook** (a True→False falling-edge detector on the
   editor's focus, firing one `flush_session(path)`), NOT scattered manual calls. This is a near-term
   standalone change AND it dissolves most of the copilot's dirty-editor clobber risk (one residual
   race when the agent writes while the editor is focused on that exact file — a small refuse-guard).
2. **Build order** — moot; the P0 no-LLM bridge spike does the de-risking, go worker-first.
3. **NO canned/discrete actions. ONE free-form chat widget — that is the entire interface.** No
   "explain my shader" / "compile that" buttons or modes. The user expresses any action (discrete or
   vague) in natural language; the agent builds a plan and executes under the hood. **This kills the
   "ship read-only Phase 2 as a product" framing from report 07** — there is no read-only product;
   there is one chat that can do whatever its tools allow. (The phasing in §4 still holds as an
   *internal build order* — read-only tools first because they're GL-free and easy — but it is NOT a
   user-facing feature gate or a shipping checkpoint.)
4. **IPv4/IPv6 egress** — test it, but **do NOT overfit to the maintainer's box.** The client must
   default to normal dual-stack (works on users' machines); an IPv4 pin is opt-in for the maintainer's
   dead-v6 dev box only. (Reports `04`/`09` both flag this; `09` is explicit: default dual-stack.)
5. **Provider = OpenRouter, NOT Anthropic.** A cheap model TBD later (codex/grok/google — model
   choice deferred). **This overrides report `04`'s Anthropic-SDK design.** Yank the low-level LLM
   interface from the reference projects (cc-server primarily — also Python). Study + plan the
   cross-cutting concerns: **pricing/cost tracking, inference-cost logging, tooled responses,
   structured output.** Investigated in **report `09`** (the LLM-layer study) — OpenRouter returns
   real charged `usage.cost` so no local pricing table is needed; the seam is OpenAI-chat-shaped
   (`parameters` not `input_schema`, `role:"tool"` results, JSON-string args needing parse).
6. **Transcript persists across restart** (saved under `app_data_dir()`). No migration concern —
   nothing exists to migrate yet; it's new state.
7. **The "freetype glyph-atlas text-rendering shader" doc claim is wrong** — filed as a `todo.md`
   deferral (verified: `fonts.py`'s glyph-atlas is dead code with zero consumers; the shipped text
   template is an SDF segment synth). The wrong claim is in **four** doc surfaces (CLAUDE.md, README,
   itch page.yaml, feature-004 spec) — the todo covers the consistent fix + the itch live-page
   re-check. Not fixed now (out of scope for this research).
8. **Capabilities/tool-catalog = LATER.** First outline **clean encapsulated modules with neat,
   robust seams** (a prompt-assembly module, a tools module, an LLM-client module, etc.) — do NOT mix
   architecture with capability details. **This reframes the whole deliverable:** the next step is a
   *module-skeleton + seams* spec, not a capability/phasing plan. (Report `09` §"module skeleton" is
   the starting point for the LLM layer; report `02` for the tool-registry/capability seam.)
9. **An action-required message type** — a special message the agent can push into the chat that
   **blocks further streaming until the user responds** (the safety/confirm primitive, generalized).
   Detail later; note the seam now (the chat protocol needs a "pending user action" turn that gates
   the agent loop).

**The reframing in one line:** this is no longer "plan a phased copilot feature with a capability
catalog." It is "design the clean encapsulated module skeleton + seams for a single free-form chat
agent over OpenRouter, with auto-save, cost-tracking, structured output, and an action-required
message type baked into the seams." Capabilities get brainstormed *after* the skeleton is sound.

---

## 1. The headline (read this first)

**Build it. The interface is ONE free-form chat widget — no canned/discrete actions, no read-only
"explain my shader" product (maintainer decision §0 #3). The phasing below is an INTERNAL build
order only (read-only tools are GL-free and land first as a convenience) — NOT a user-facing feature
gate or a shipping checkpoint.** This wave scaffolds the module skeleton + seams (see
`10_skeleton_plan.md`); capabilities are a later brainstorm (§0 #8).

The architecture is *not* novel invention — ShaderBox already solved "worker thread that can't touch
GL, talking to a single-threaded GL frame loop" **twice** (Telegram, YouTube exporters). The copilot
reuses that exact contract. The **one genuinely new mechanism** is a *synchronous round-trip from the
worker back INTO the main thread* (a tool call must block until its GL op runs a frame later and
returns a result) — a small, well-contained `MainThreadBridge`. Everything else is assembly of
patterns the repo + the three reference agents already prove.

**Refactor-prep is tiny.** The audit's verdict (report 03, the adversarial pass especially): almost
every "gap" is the feature's own tool layer in disguise. There is **one MUST-do-before** (a document,
already written — the GL partition), **one cheap SHOULD-do-before** (a ~15-line uniform-setter
extraction), and **app.py does NOT need splitting.** The temptation to "tidy the codebase before the
big feature" is exactly the large-diff-no-symptom anti-pattern the maintainer's own rules warn
against — resist it.

**The differentiator is the in-process compile-feedback loop** (report 06): the agent edits → the
real driver recompiles in-process sub-frame-fast → the agent reads the *source-mapped* error at the
exact file+line it edited → self-corrects. No cloud coding agent gets a loop this tight. The honest
limit: the agent's only oracle is `compile_unit.errors` — it **cannot see the rendered pixels**, so
"compiled" ≠ "looks right." v1 must be honest about that and never claim a visual effect it can't
confirm.

---

## 2. The consensus architecture (what every report agrees on)

```
                          ┌─────────────────────────────────────────────┐
   MAIN THREAD            │  glfw frame loop  (ui.py::run → update_and_draw)
   (GL + imgui)           │                                              │
                          │  • bridge.drain()  ← runs queued GL ops      │ ← NEW drain point,
                          │  • copilot.update() ← drains event queue     │   beside share_tab.update
                          │  • draw Copilot tab (4th tab, reads          │   (ui.py:179)
                          │    render-state struct — no GL, no queue)    │
                          └───────▲───────────────────────┬─────────────┘
                                  │ MainThreadOp           │ CopilotEvent
                          (worker blocks on done.wait)     │ (token deltas, status,
                                  │                        ▼  tool cards — drained per frame)
                          ┌───────┴───────────────────────────────────────┐
   WORKER THREAD          │  Copilot worker (plain threading.Thread,        │
   (NO moderngl,          │   sync OpenRouter streaming — NOT asyncio)      │
    NO imgui)             │                                                 │
                          │  agent loop:  LLM.stream() → tool_use blocks →  │
                          │    registry.execute(name, input) → repeat       │
                          │      • GL-free tool → runs inline (file write,   │
                          │        lib CRUD, reads)                          │
                          │      • GL-touching tool → bridge.run_on_main()   │
                          │        (blocks for the result)                   │
                          │  imports ONLY: CopilotCapabilities, ILLMClient,  │
                          │    ToolRegistry  (never App, never moderngl)     │
                          └─────────────────────────────────────────────────┘
```

The decisions, each with its owning report and the cross-checks that confirmed it:

| Decision | Choice | Reports that agree |
|---|---|---|
| **Where the agent loop runs** | one plain `threading.Thread` + blocking `queue.Queue`, sync OpenRouter streaming (the YouTube-exporter shape, NOT telegram's asyncio) | 01 (owns), 02, 09 all assume sync worker |
| **GL marshalling** | a `MainThreadBridge`: worker pushes a `MainThreadOp` (closure + `threading.Event`), blocks on `done.wait(timeout)`; main thread drains once/frame in `update_and_draw`, runs the closure, sets the event | 01 (owns), 03 (the partition), 02+04 (consume) |
| **Shader edits** | write the `.glsl` file (GL-free worker) → the existing `_reload_if_changed` hot-reload recompiles on the main thread. The "free lunch." Read errors back via ONE bridge round-trip (forces compile, returns `compile_unit.errors`) — don't poll | 01, 03, 06 all independently |
| **The capability seam** | a frozen `CopilotCapabilities` dataclass of bound callables, built in `App.__init__` (exactly like `_build_command_callbacks` + `ShaderLibFileManager`'s injected callbacks). The `copilot/` package imports ONLY this leaf type, never `App` → cycle-free by construction | 02 (owns), 01 (`AppHandle`, same thing), 03 (F4/F10 resolution) |
| **Tool schema** | pydantic `args_model.model_json_schema()` (ovelia), NOT hand-written JSON schema (cc-server) — one definition gives schema + validation; pydantic already a dep. OpenAI/OpenRouter key is `parameters` (NOT Anthropic `input_schema`) | 02 (owns), 09 |
| **Handler return** | ovelia's `(ok, message_for_llm, payload)` triple, NOT cc-server's bare `-> str`. Payload drives the visual affordance (jump to the created node) | 02 (owns), 05 (the chip/jump UI depends on it) |
| **Registry vs commands.py** | separate registry, same shape; share the *verb* at the `App`-method level, never the registry infra (command callbacks are zero-arg, can't carry `set_uniform(id,name,val)`) | 02 (owns) |
| **LLM provider** | **OpenRouter** (cheap model TBD — codex/grok/gemini, a string id), via the `openai` SDK pointed at OpenRouter, behind a small sync `LLMClient` Protocol seam; **stream**, don't block. Cost comes back on `usage.cost` — no local pricing table. (Overrides 04's Anthropic design — see §0 #5.) | 09 (owns) |
| **IPv4 pin** | the httpx client defaults to **dual-stack** (works on users' machines); an IPv4 pin (`local_address="0.0.0.0"`) is **opt-in** for the maintainer's dead-v6 dev box only — do NOT overfit. Write the seam pin-ready (one constructor arg / settings flag). | 09 (§6.1, owns), §0 #4 |
| **Prompt caching** | least-volatile→most-volatile prompt assembly (ovelia ordering) for any provider-side caching; OpenRouter passes provider caching through — NOT a v1 seam concern (no Anthropic `cache_control` blocks) | 09 (§3.0), 06 (§5.6) |
| **API key home** | extend `IntegrationsStore` with `CopilotIntegration(openrouter_key, model)` — same cleartext-at-`app_data_dir()` posture + the existing cleartext deferral covers the keyring migration of all three. The `_SAVE_LOCK` already serializes render+worker writes | 03, 09, 05 all independently |
| **Chat UI placement** | a 4th tab `Copilot` in the right-panel tab bar (`Node/Render/Share/Copilot`) — the only option that keeps editor (LEFT) + live preview (TOP) both visible while chatting; reuses the active-region + `Ctrl+4` tab-jump machinery; ~10-line diff | 05 (owns) |
| **Chat MVP UI** | flat `wrapped_caption` rows (no bubbles), streaming = the growing string redrawn each frame, tool-calls as dim status lines, `Ctrl+Enter` sends (Enter=newline, avoids the feature-019 nav collision), `unconnected_gate` for the not-configured state, confirm-before-delete via `cell_delete_confirm`/`danger_button`. NOTE: no canned action buttons — it's one free-form chat (§0 #3) | 05 (owns) |
| **Capabilities** | DEFERRED to a later brainstorm (§0 #8) — this wave scaffolds the module skeleton + seams (`10_skeleton_plan.md`), not the tool catalog. (Reports 06/07's "explain + edit MVP" / "read-only Phase-2 checkpoint" framings are SUPERSEDED: one free-form chat, no read-only product gate.) | §0 #3/#8, 10 (owns) |

---

## 3. Refactor-prep plan (the actionable pre-feature work)

The audit (report 03) is deliberately ruthless. The full priority table is there; the decision:

### MUST do before the feature
- **The GL-free / GL-touching verb partition** — *done* (report 03 §2 is the deliverable). It's not
  code, it's shared knowledge the threading design and the spec both consume; re-deriving it per-tool
  guarantees an inconsistency (one author assumes `set_uniform` is GL-free — it is NOT for
  sampler/buffer arms, and races the render loop even for scalars). **No code; satisfied.**

### SHOULD do before (one small, behavior-preserving prep commit — recommended but a clean defer too)
- **Extract `App.set_uniform_value(node_id, name, value)`** (main-thread; does the `try_to_release`
  dance from `util.py:102`; validates against `valid_input_types()`) and re-point the inline write at
  `widgets/uniform.py:228-230` to it. ~15 lines + one call-site swap. Why prep and not defer: the
  off-thread inline version is easy to get wrong (GL texture leak + a data race against the render
  loop iterating `uniform_values`, `core.py:289-338`). It's the maintainer's own "already-solved-twin
  / shared primitive" shape — the read path is clean, the write path is the gap.
- **Fold in `create_node(template_id)`** in the same commit: extract the body of
  `create_node_from_selected_template` (`app.py:1070`) into `create_node(self, template_id: str) ->
  str` (returns the new node id); the existing method becomes a one-line caller. Strictly cleaner even
  ignoring the copilot.
- Same commit also extracts a `set_uniform_input_type` headless verb (the input-shape is mutated the
  same inline way at `widgets/uniform.py:84-92`, snapping via `snap_input_type()`).
- Run `make check` + `make smoke` (touches UI/lifecycle); the UI must look identical (behavior-preserving).

### Sequencing call (report 07)
Land the prep as **separate green pre-feature commits**, not folded into the feature's Phase 0 —
they're behavior-preserving refactors that can coexist with the old call sites, so keeping the
green-refactor / red-feature split gives clean bisects. (Unlike feature 001's delete-and-replace, this
prep doesn't *remove* anything.)

### Everything else → INTO the feature, not prep
`CopilotCapabilities` interface (can't shape it before the tools exist — speculative as prep),
`CopilotIntegration` (wants to live next to the settings UI), the chat UI primitive (build it WITH
the widget), the unsaved-editor-clobber guard (the shader-write tool owns it), short-id resolution (a
pure helper in the copilot package). And **do NOT split app.py** — the copilot lands in its own
`copilot/` package mirroring `exporters/`; `app.py` gains only a handle + the bridge drain (+20-40
lines, not +400).

---

## 4. Internal build order (NOT a user-facing feature gate)

Per §0 #3, the product is one free-form chat — there is **no read-only "explain my shader" release**
and **no user-facing phase gate**. The order below is purely an **internal build sequence** on
**ascending GL-thread-risk** (read-only tools are GL-free → land first as a convenience), each step
independently verifiable. The user never sees these phases; they're how WE build, not what WE ship.
**This scaffold wave is `prep` + the structural shells of P0/P1 (the skeleton + seams).** Capabilities
(the P2–P4 tool catalog) are the later brainstorm (§0 #8).

| Step | Lands | Verifiable by | Needs |
|---|---|---|---|
| **prep** | `App.set_uniform_value` + `create_node(template_id)` + `set_uniform_input_type` (green refactor commit) | `make check` + `make smoke`; UI unchanged | — |
| **P0** | the no-LLM command-queue spike: a worker enqueues a dummy mutation, the main thread drains+applies it, smoke-proven. **The de-risk for the only un-precedented part (the `MainThreadBridge`).** | `make smoke` extended to drive a fake op through the bridge across frames | — |
| **P1** | the `LLMClient` seam + OpenRouter impl + a **non-tool** chat (just talks). Proves the call doesn't freeze the loop, the key loads, the worker joins cleanly. | manual: type a message, get a streamed reply; smoke: worker spins up + tears down headlessly (no key → gated, no crash) | key in `IntegrationsStore`; the 4th-tab UI shell + gate |
| **P2** | the tool registry + `CopilotCapabilities` seam + read-only tools (GL-free snapshots, no marshalling). | Tier-1 pytest (handlers via in-memory caps, no GL) | the seam, the registry, the context snapshot |
| **P3** | mutation tools (`edit_shader` via the hot-reload free lunch, `set_uniform`, `set_uniform_input_type`) + the **compile-feedback loop** (`MainThreadBridge` round-trip for error readback) | Tier-1 (fake caps) + Tier-2 smoke (real caps, GL ops across frames); manual | P0 bridge, prep verbs, confirm-before-destructive UX |
| **P4** | orchestration: `create_node`/`delete_node`, budget + old-tool-result compression, lib authoring (gated), render/export (reuse exporter marshalling) | Tier-2 smoke; manual end-to-end | all prior |

(Reports 06/07 framed P2 as a "read-only explain-my-shader shipping checkpoint." That framing is
SUPERSEDED by §0 #3 — read-only-first is a build convenience, not a product the user ever sees as a
distinct release.)

---

## 5. Risk register (the ones that need a decision or a mitigation up front)

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Threading/GL-affinity marshalling** — the `MainThreadBridge` is the one un-precedented part; the exporters only marshal *output, one-way, GL-free worker*; the copilot marshals *mutations the user watches* with GL effects | med | high | **The #1 derailment risk.** De-risk with the **P0 spike before any LLM cost** — it's the go/no-go gate for the mutation half. |
| **App-close mid-run deadlock** — worker blocked in `op.done.wait()`, main thread left the frame loop, never drains again, `join()` hangs | med | high | `Copilot.release()` calls `bridge.cancel_all()` (sets every pending op's event) BEFORE `worker.join(timeout)` — mirrors `Exporter.release`. Plus a timeout on `done.wait()` as backstop. (report 01 §5) |
| **Data race on `uniform_values`** — a cross-thread dict write races the render loop iterating the same dict | med | med | Always marshal uniform sets through the bridge — never write the dict from the worker. (report 01 §9.4, report 03 §2) |
| **Dirty-editor clobber** — agent writes a `.glsl` while the user has unsaved edits; `sync_editor_from_disk` overwrites with no diff (node-root branch; lib branch is safe) | med | med | The shader-edit tool checks `is_current_editor_dirty()` first → refuse + tell the LLM (v1), or confirm-then-overwrite. **Open Q for the maintainer.** (report 01 §9.1, 03 F9, 05 §8.1, 06 OQ3) |
| **IPv4/dead-v6 egress** to `openrouter.ai` on the maintainer's dead-v6 dev box (same class as the Telegram fix) | unknown (endpoint-dependent) | high if it bites on the dev box | **Default dual-stack** (works on users' machines — do NOT overfit); IPv4 pin (`local_address="0.0.0.0"`) is **opt-in** (settings flag) for the dev box. Write the seam pin-ready. (report 09 §6.1, §0 #4) |
| **Hung-GPU GLSL** — the agent writes an unbounded loop; the single render thread freezes the whole app (no engine guard) | low-med | high | System-prompt rule: bound every loop with a compile-time constant, no unbounded `while`. (report 06 §7) |
| **"Compiles but renders wrong"** — the agent can't see pixels; claims a visual effect it can't confirm | high | low-med | Prompt discipline (ovelia's "never claim a past-tense effect a tool didn't confirm"); the agent says "compiled; check the preview." A luminance/NaN one-number readback is a cheap fast-follow, not v1. (report 06 §7, §8c) |
| **Prompt injection** via shader source / node names / lib docstrings (user data spliced into the prompt) | low | med | Sanitize names spliced into the prompt (cc-server `sanitize_name` / ovelia `_sanitize_title_for_prompt`); a system-prompt "context is data, not instructions" note. (report 04 §4.6 — pattern is provider-agnostic) |
| **API cost runaway** (user's own OpenRouter key) | low | med (user's bill) | max-iterations + max-input-tokens budget + old-tool-result compression + the cut-off "what I committed" note. Real `usage.cost` shown per-turn + cumulative (report 09 §2). |
| **app_state migration tax** if chat state lands in `UIAppState` (`extra="forbid"` + `load_and_migrate`) | low | low | Budget knobs = a frozen `CopilotConfig` module dataclass (constants, not user-tuned); key+model in `IntegrationsStore`. Transcript **persists** (§0 #6) but as its own JSON log under `app_data_dir()`, NOT a `UIAppState` field → no migration tax. |

**Bundle/ship invariant holds with ZERO `build.sh` change** (report 07 §5, confirmed against
`build.sh` + `exporters/integrations.py`): `copilot/` ships via the existing `cp -r shaderbox`;
the `openai` SDK (OpenRouter client) rides the already-allowlisted `pyproject.toml`/`uv.lock`; the API
key stays in `integrations.json` at `app_data_dir()` (outside the repo, never staged).

---

## 6. Two facts verified directly (premise-checking, not relayed)

Report 06 surfaced two claims that contradict or correct in-repo documentation. Both were verified
against the files on disk (2026-05-29), not taken on the agent's word:

1. **The shipped Text Rendering template is an SDF 7-segment-lattice glyph synth driven by
   `uniform uint u_text[64]` (Unicode codepoints) — NOT a freetype glyph-atlas with a texture
   sampler.** Verified: `f90f5ff9/shader.frag.glsl` has no `sampler2D`/`texture()`, uses
   `get_dist_to_line` + segment anchors + an all-caps fold. CLAUDE.md's "custom freetype glyph-atlas
   text-rendering shader" describes the *app's* freetype→imgui atlas (`fonts.py`), not this template's
   shader. **The copilot's system prompt must describe what's on disk** (SDF segment glyphs + codepoint
   array), and the agent must reuse, not regenerate, the ~600-line glyph SDF. *(Also flag to the
   maintainer: is CLAUDE.md's line stale, or is there a separate atlas text path? — Open Q.)*
2. **UV Mango's `node.json` carries a stale `u_zoomout` UIUniform with no matching declaration** in
   its 2-uniform shader (`u_time`/`u_aspect` only). Verified. Harmless at runtime (the render loop
   iterates `get_active_uniforms()`, the live program, not the saved `ui_uniforms` map). **The
   takeaway for the copilot:** the authoritative uniform list is `get_active_uniforms()` (live
   reflection), NEVER the saved `UINodeState.ui_uniforms` dict — a "list this node's uniforms" tool
   must read the live program.

---

## 7. Open questions — RESOLVED (see §0)

The original 9 open questions were **all answered by the maintainer** — the answers are §0
"Maintainer decisions." Summary of dispositions so a spec author doesn't reopen them:

1. **Dirty-editor policy** → §0 #1: generalized auto-flush hook (report 08) dissolves the common case;
   the residual focused-editor race is a small refuse-if-dirty-and-focused tool guard.
2. **Build order** → §0 #2: worker-first; the P0 no-LLM spike does the de-risking.
3. **Phasing checkpoint** → §0 #3: **NO read-only product / no user-facing phase gate** — one
   free-form chat. (Was "the biggest scoping call"; now decided.)
4. **IPv4** → §0 #4: default dual-stack, pin opt-in, don't overfit; test the endpoint but don't gate on it.
5. **Model/provider** → §0 #5: OpenRouter, cheap model TBD; not a v1 concern beyond the seam.
6. **Transcript persistence** → §0 #6: persists (own JSON log; no `UIAppState` migration).
7. **CLAUDE.md text-shader description** → §0 #7: stale, filed as a `todo.md` deferral (verified).
8. **Lib-catalog injection threshold** → §0 #8: a capability/prompt detail, deferred to the later
   brainstorm (not this scaffold wave).
9. **Confirm-before-delete granularity** → §0 #9: an action-required message type (seam noted in
   `10 §7`); the exact granularity is a later detail.

**Genuinely still-open (all later-brainstorm, none block the scaffold wave):** the tool catalog, the
prompt content, the cheap-model choice + retry policy, the lib-catalog threshold (#8), the
action-required-message exact shape (#9), the chat-UI MVP-vs-gold cut. Per §0 #8 these are
deliberately NOT decided now — the scaffold lands the modules + seams first.

---

## 8. The spec scaffold (what `020_copilot_agent.md` will contain)

When the maintainer signs off on scope, the locked spec follows the house section shape, sourced from
the reports:

> **NOTE:** this scaffold wave's spec covers the MODULE SKELETON + SEAMS only (`10_skeleton_plan.md`).
> A later capability spec covers the tool catalog / prompt / UX. The scaffold below reflects that cut.

- **Goal** — scaffold the `copilot/` package: the module skeleton + the five seams + the `App`/`ui.py`
  wiring + the secret-store field, with stub/empty capability modules. One free-form chat, OpenRouter.
  NO tool catalog, NO prompt content this wave. Source: `10_skeleton_plan.md`, §0.
- **Out of scope** (each with a trigger) — the tool catalog + prompt content + UX detail (the later
  capability brainstorm, §0 #8); the agent *authoring* lib functions (015 export-from-selection
  deferral); pixel-readback "see the output" / vision (report 06 §8c — file the luminance/NaN
  one-number readback as a fast-follow trigger); media/texture uniform loads (the agent can't supply
  pixel data); the pop-out chat window (gold, report 05 §9.1); per-action undo + the per-turn mutation
  journal (report 05 §8.3); multi-provider beyond OpenRouter (the `LLMClient` seam keeps it open).
- **Design decisions** (numbered, lock-in) — the §2 table, each decision a numbered entry with its
  rationale + the report it's drawn from; plus the §0 maintainer decisions.
- **Files touched** — new `copilot/` package (`capabilities.py` leaf, `bridge.py`, `errors.py`,
  `config.py`, `state.py`, `context.py`, `prompt.py`, `tools/{registry}.py`, `llm/{api,openrouter}.py`,
  `agent.py`, `session.py` — per `10 §1`); `app.py` (+handle, +`_build_copilot_capabilities`, +the
  construction-order + `release()` guard + the live-`integrations_store` wiring — see the §5 fixes);
  `ui.py` (4th tab + the two drain calls); `tabs/copilot.py` (the widget shell);
  `ui_primitives.py` (the transcript-body wrap primitive); `commands.py` (`NodeTab.COPILOT` +
  `FOCUS_TAB_COPILOT`/`Ctrl+4`); `popups/settings.py` (the key + model block);
  `exporters/integrations.py` (`CopilotIntegration`); `pyproject.toml` (`openai` — the OpenRouter
  client); `tests/` (`test_copilot_*.py` Tier-1) + `scripts/smoke.py` (`--copilot` Tier-2 poke).
- **Manual verification** — what only a `make run` hand-check covers (the app can't be screenshotted):
  streaming feel, the real LLM round-trip, the preview updating after an edit, the tab-column width,
  the confirm-before-delete affordance. (report 05 §0.)
- **Open questions** — §7 above.
- **Review history** — populated as the feature is reviewed.

**Review plan** (high-blast-radius per dev_flow → scale up): 2 pre-impl + 3+ post-impl reviewers run
as a convergence loop (`/review-agent-loop`), at least one anchored to a non-self-authored artifact
(the running app / the exporter precedent / the user's verbatim ask). Feature-specific review
dimensions: **threading correctness + deadlock** (the bridge, cancellation, `release()` ordering),
**GL thread-affinity** (no moderngl on the worker — the smoke test + a reviewer dedicated to it),
**prompt-injection**, **tool-handler correctness** (the capability seam + validation), **imgui
patterns** (the `/imgui-ui` skill — wrapped text, no jitter, the focus-grab + FPE deferrals),
**spec-fidelity**.

---

## 9. Bottom line for the go/no-go

- **Feasible** — the hard part (threading + GL marshalling) reuses a contract the repo already runs
  twice; the new piece (the synchronous worker→main bridge) is small and de-riskable with a P0 spike
  before any LLM cost is spent.
- **Cheap prep** — one document (done) + one ~15-line behavior-preserving refactor commit. app.py is
  NOT split.
- **Build-orderable** — the internal sequence (prep → P0 bridge spike → seam → tools) lets each step
  verify independently; but the product is one free-form chat (§0 #3), not a phased release.
- **The crown jewel** (the in-process compile-feedback loop) is real and is the thing no cloud agent
  has — but its honest limit (can't see pixels) must be designed in, not papered over.
- **The decisions are made** (§2 table + §0); the open questions are **resolved** (§7). This wave =
  scaffold the skeleton + seams (`10_skeleton_plan.md`); the capability brainstorm comes after.
