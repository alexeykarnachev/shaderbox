# 020 Copilot Agent — Phasing, Risk, Testing, Spec Scaffold (the META angle)

> Research report. Companion to `00_grounding.md` (the factual anchor). Siblings `01`–`06` (threading,
> tool-registry seam, refactor-prep audit, LLM integration, chat UI/UX, GLSL domain) were not yet on
> disk when this was written — cross-refs below point at where each report *will* answer a stub, so the
> synthesis (`99`) can wire them in. This report owns: **phasing, prep-sequencing, the risk register, the
> testing strategy, the bundle/ship check, a spec scaffold, the review plan, and the adversarial case.**

---

## 0. TL;DR (the go/no-go inputs)

- **Phase it.** Five phases (0–4), each independently shippable + verifiable. The phase boundary is
  drawn at **escalating GL-thread risk + escalating blast radius**, not at feature-completeness — so the
  scariest threading work is isolated and proven before any mutation tool exists.
- **Prep is its own pre-feature wave**, not Phase 0 of the feature. The verbs (`set_uniform_value`,
  `create_node(template_id)`) and the secret-store decision are pure refactors with their own value and
  their own (cheap) verification; folding them into the feature couples a green-refactor commit to a
  red-feature commit. Recommend: **2–3 prep commits, `make check` + `make smoke` green at each, THEN
  Phase 1.**
- **#1 derailment risk: the threading/GL-affinity marshalling design.** It's the one part with no
  precedent the agent can copy verbatim (the exporters marshal *render output*; the copilot marshals
  *arbitrary mutations the user is watching*). De-risk with a **Phase-0/1 spike**: a no-LLM "command
  queue" that lets a worker thread enqueue `set_uniform_value` and watch the preview change, proven by
  smoke before a single token is spent on an LLM.
- **"Are we overbuilding?" verdict: phase the ambition, don't cut it.** The honest smaller bite is
  Phase 1+2 (chat that *explains* the shader + read-only introspection, zero mutation). That alone is a
  shippable, valuable, low-risk product. Phases 3–4 (mutation, orchestration) are where the cost/risk
  lives — gate them behind Phase 1+2 shipping and being *used*. So: not premature, but the maintainer
  should pre-commit to stopping after Phase 2 if the threading spike or the UX disappoints.

---

## 1. Phasing

Each phase is a shippable increment. The cut line is **GL-thread risk + blast radius**, ascending — the
read-only chat (Phase 1–2) never marshals a mutation, so it ships with near-zero risk; mutation (Phase 3)
is where the command-queue marshalling earns its keep; orchestration (Phase 4) is the multi-tool agent
loop on top. Every phase keeps `make check` + `make smoke` green.

| Phase | What lands | Verifiable by | Needs prep / cross-ref |
|---|---|---|---|
| **0 — Threading spike (throwaway/foundational)** | A no-LLM **command queue**: worker thread → `queue.Queue` of `Command` → drained at the top of `update_and_draw` on the main thread, applied there. One proof command (`set_uniform_value`). NOT a tool, NOT shipped UI — a spike behind a debug hotkey or a smoke-only path. | `make smoke` extended: spin up the worker, enqueue a `set_uniform_value`, advance frames, assert `node.uniform_values[name]` changed AND no exception (proves marshalling + GL affinity headlessly). Manual: hit the debug key, watch the preview change. | **Prep verb (a)** `set_uniform_value`. Threading design ← report `01`. |
| **1 — LLM seam + non-tool chat (just talks)** | The `ILLMService`-style seam (Anthropic client behind a Protocol), the worker-thread **agent loop** with NO tools, a chat widget (input + streamed assistant text) via `ui_primitives`. It can *talk about* shaders (the system prompt carries a context snapshot) but takes zero actions. Secret loaded from `integrations.json`. | Manual: type "what does a fragment shader do?", see streamed reply; UI stays responsive during the call (the Decision-3-style `glfw.get_time()` delta check from feature 001). Headless: smoke spins the worker with a **mock LLM client** (no key), asserts the loop produces text + the thread joins on `release()`. | Secret-store decision (prep). LLM seam ← `04`. Chat UI ← `05`. Reuses Phase-0 worker/lifecycle. |
| **2 — Tool registry + READ-ONLY tools** | The tool registry (name/description/pydantic-args-schema/handler, ovelia-style `model_json_schema()`), the **capability interface** (`CopilotCapabilities` — marginalia pattern), and read-only tools: `get_current_node`, `list_uniforms`, `get_shader_source`, `get_compile_errors`, `list_lib_functions`. The agent loop now does tool-call→execute→feed-back. **No mutation, no marshalling needed** (all reads are GL-free snapshots taken on the worker from already-materialized state, or a one-frame snapshot request). | Unit: each handler tested via the **swapped capability impl** (in-memory `CopilotCapabilities`, no glfw/GL) — see §4. Mock-LLM agent-loop test: a scripted "call list_uniforms then answer" transcript, assert the loop executes the tool + composes the final text. Manual: "what uniforms does this node have?" → correct list. | Tool seam ← `02`. GLSL-domain tool shapes ← `06`. Capability swap is `02`'s deliverable; the *test plan* is here. |
| **3 — MUTATION tools + the compile-feedback loop** | `set_uniform_value`, `set_input_shape`, `edit_shader_source` (write the `.frag.glsl` to disk — rides the existing hot-reload free lunch), `create_node(template_id)`, `delete_node`. Each mutation routes through the Phase-0 **command queue** (marshalled to main thread). After an `edit_shader_source`, the agent reads back `compile_unit.errors` as its success/failure signal and can self-correct. | Unit: mutation handlers against the in-memory capability (assert the queued command + its args). Mock-LLM: a "edit shader → (inject a compile error) → read errors → re-edit" scripted transcript, assert the self-correct loop. Manual (the load-bearing demo): **"make the background pulse red"** → GLSL edited, preview pulses red. Error path: agent writes broken GLSL → sees the error → fixes it. | **Prep verbs (a)+(b)**. Command queue (Phase 0). Hot-reload edge cases (unsaved editor buffer) ← `02`/`06`. |
| **4 — Orchestration (multi-step + lib/render)** | The full agent loop polish: max-iterations + token budget, old-tool-result compression (cc-server/marginalia `_compress_old_tool_results`), `create_shader_lib_file` / edit-lib tools, drive a render/export. Multi-tool plans ("create a node, write a plasma shader, set the speed uniform"). The "what I did" executed-actions note on cutoff (ovelia). | Mock-LLM: a long scripted multi-tool transcript hitting the iteration cap, assert compression + the cutoff note. Manual: a compound natural-language request end-to-end. | Lib CRUD verbs already exist (`ShaderLibFileManager`, GL-free — cleanest surface). Render marshalling reuses the command queue. Loop polish ← `04`. |

**Why these cuts (the argument):**

1. **Phase 0 is separate because the threading is the only un-precedented part.** The exporters
   (`exporters/telegram.py`) marshal *render output* (`RenderedArtifact`, a pure value) one-way, worker→
   main, and the worker NEVER touches GL. The copilot must marshal *commands the user is watching* main↔
   worker, where the command's *effect* touches GL. That's a new direction of data flow. Proving it with
   no LLM (no cost, no non-determinism, smoke-testable) before building anything on top is the
   single highest-leverage de-risk. (Cross-ref `01`.)
2. **Phase 1 vs 2 split (talk vs read-only-tools) because the LLM seam and the tool registry are
   orthogonal risks.** Phase 1 proves "an LLM call doesn't freeze the frame loop + the secret loads + the
   thread drains cleanly" with zero tool surface. Phase 2 proves the registry/capability seam with zero
   network non-determinism added (the tools are deterministic; only the LLM picking them is not). Bugs
   isolate.
3. **Phase 2 (read) before Phase 3 (mutate) because read tools need NO command-queue marshalling** —
   they're GL-free snapshots. So the entire tool-registry + capability-interface + agent-tool-loop
   machinery lands and ships *before* the marshalling risk re-enters. A read-only "explain my shader,
   introspect my uniforms" assistant is a complete, shippable product at the end of Phase 2.
4. **Phase 4 last because orchestration (budget, compression, multi-tool) is pure agent-loop polish** on
   top of a proven tool surface — it adds UX robustness, not new system-level risk.

**The minimal-shippable line is the end of Phase 2.** Everything before it is low-GL-risk; Phase 3 is
where the maintainer re-commits budget to the scary part.

---

## 2. Prep-sequencing: separate commits, NOT Phase 0 of the feature

`dev_flow.md` says deferrals get absorbed by the work that triggers them — true, but that governs *where
the design fact lives* (the spec, not a stale todo), not *commit granularity*. Recommendation:

**Land the prep as its own pre-feature wave of small commits, each green, BEFORE Phase 1.** Concretely:

- **Prep commit A — `set_uniform_value(node_id, name, value)` headless verb on `App`.** Today the mutation
  is inline in `widgets/uniform.py:230` (`ui_node.node.uniform_values[name] = new_value`, after a
  `try_to_release(current_value)` at the surrounding draw site). Extract the value-swap + release into a
  headless `App.set_uniform_value` (or a `node_ops` leaf), and have the widget *call* it. This is a pure
  refactor with its own test (`make smoke` + a focused introspection script) and its own value (a
  testable verb the widget shares). Closes todo gap **(a)**.
- **Prep commit B — `create_node(template_id)` arg form.** `App.create_node_from_selected_template`
  (`app.py:1070`) reads `app_state.selected_node_template_id`. Add `create_node(template_id: str)` and
  make the grid path call it with the selected id. Closes gap **(b)**.
- **Prep commit C (decision, maybe no code) — secret store.** The Anthropic key joins
  `integrations.json` (already cleartext Telegram `bot_token` + YouTube `client_secret`/`token_json` at
  `app_data_dir()`, outside the repo — confirmed). Add an `AnthropicIntegration` model. This rides the
  existing `[DEFERRAL] integration credentials stored cleartext` posture (a known, accepted trade-off for
  a local single-user tool) — do NOT introduce keyring here; that's a separate hardening pass with its
  own trigger.

**Why separate, not folded:** a refactor commit is *behavior-preserving and independently verifiable*
(smoke proves the widget still works); a feature commit is *new behavior*. Folding them means a bisect that
lands on the feature commit can't tell "did the verb extraction break, or the agent?" The exporter
refactor (001) is the precedent for the *opposite* (it deleted + rebuilt in one diff) — but that was
because the old and new code couldn't coexist (`sharing.py` was being deleted). Here the prep verbs *can*
coexist with the old call sites (the widget just delegates), so the clean split is available; take it.

**One caveat:** if the verb extraction turns out to be 5 lines and trivially correct, a single
`prep: copilot-ready verbs` commit covering A+B is fine — the rule is "green refactor separate from red
feature," not "one commit per verb."

---

## 3. Risk register

Likelihood (L) / Impact (I) on Low/Med/High. Ordered by L×I.

| # | Risk | L | I | Mitigation |
|---|---|---|---|---|
| R1 | **GL-thread-affinity violation** — a tool handler on the worker constructs/touches a `Node`/`Canvas`/`Texture`/`program[...]` (recall `conventions.md ## Known quirks`: GL objects call `moderngl.get_context()` lazily; off-thread = no context = crash or UB). | High | High | The **command-queue marshalling** (Phase 0): worker handlers NEVER call GL; they enqueue a `Command` drained on the main thread in `update_and_draw`. Mirror the exporter thread-affinity discipline (`conventions.md ## Design decisions`: "Exporters: own thread, GL-free artifacts"). Smoke asserts a worker-enqueued mutation lands without exception. Partition the verbs in the spec (GL-free vs GL-touching, per `00_grounding §3`). |
| R2 | **Threading deadlock / frame-loop stall** — the agent loop blocks the main thread (synchronous LLM call inline), or the queue drain blocks, freezing the UI. | Med | High | Worker thread owns ALL network + agent-loop work (the `## No async` convention's worker-thread pattern). Main thread only `get_nowait()`-drains a bounded queue (exporter precedent: `_progress_queue.get_nowait()` in the draw path). Verify with the feature-001 `glfw.get_time()`-delta responsiveness check during a live call. Non-daemon worker + sentinel + bounded join in `App.release()` (exporter `release()` precedent at `telegram.py:827`, incl. `call_soon_threadsafe(task.cancel)` for in-flight). |
| R3 | **Derailment on the marshalling design itself** (see §8 #3) — the team can't cleanly express "a tool mutates GL state asynchronously and the user watches." | Med | High | The Phase-0 spike IS this mitigation: prove the command queue with no LLM before building on it. If the spike is ugly, the maintainer learns it for the price of one throwaway commit, not a half-built feature. |
| R4 | **imgui FPE-behind-modals deferral trips on the chat widget.** `[DEFERRAL] imgui_color_text_edit render() FPE`: the C++ `TextEditor::Render` div-by-zeros on glyph metrics when not focused / behind a modal; guard #2 is "don't call the editor's `set_*()` while a modal is open." **Does the chat trip it?** Only if the chat uses an `imgui_color_text_edit` `TextEditor` instance (e.g. for a code-block render or a multiline input). | Med | Med | **Decide the chat input/transcript widget does NOT use `imgui_color_text_edit`** — use `imgui.input_text_multiline` for input and plain `text_wrapped`/`text_unformatted` for the transcript. If a syntax-highlighted code block is wanted later, that's a separate decision that re-enters the FPE deferral's territory (re-verify the guard). Lock this as a Design decision so impl doesn't reach for the `TextEditor` reflexively. |
| R5 | **Prompt-injection via shader source / node names / lib content** spliced into the prompt or tool output (a node named `"]] ignore prior instructions"`, a shader comment carrying an injection). | Med | Med | Treat all user/disk-sourced text (node names, shader source, lib function names) as DATA: a `sanitize_name`-style collapse of control chars + length cap (cc-server pattern) on anything spliced into the system prompt; tool outputs return the affected entity *named* but sanitized. Generic error strings out of handlers — never echo raw exception text into LLM context (ovelia `_run_op` / cc-server pattern). Low *impact* for a single-local-user tool (no other user's data to exfiltrate, no privileged backend) — but the agent *can* mutate the project, so a "delete all nodes" injection is real. Mutation tools should be the place a confirmation gate could live (see R8). |
| R6 | **API-cost runaway** — the agent loops, re-sends growing context, burns the user's key. | Med | Med | Per-turn **max-iterations + max-input-token budget** (all three reference agents have this) + old-tool-result compression so context doesn't grow linearly. The user owns their key (name-your-price tool; cost is the user's concern per `00_grounding §4`) — but a runaway *feels* like a bug. Surface token usage in the UI; hard-stop at the budget with the ovelia "what I did" note. |
| R7 | **app_state migration** — a new chat-state field on `UIAppState` (`extra="forbid"`, `load_and_migrate` with 4 migration gens) breaks load of existing projects if added carelessly. | Low | Med | If chat history/settings persist, add the field with a default and extend `load_and_migrate` (the 001 precedent: `extra="forbid"` is what makes the migration *verifiable* — a broken shim raises instead of silently dropping). **Prefer NOT persisting chat history per-project at all** (chat is ephemeral session state, like the exporter mailbox) — then zero migration. If the key store needs a per-project pointer, keep it minimal. Decide in the spec. |
| R8 | **A mutation the user didn't intend** — agent edits/deletes on a misread instruction; the user is watching a live visual app, not reviewing a diff. | Med | Med | The feedback IS the preview + compile errors (the agent's "did it work" signal, `00_grounding §4`). Consider a **per-turn dedup cache** (ovelia, defends a double-create) and, for *destructive* tools (`delete_node`, overwrite shader), a confirmation affordance or an undo path. Edit-shader is non-destructive-ish (the file's prior content + the editor's dirty baseline exist) — but `delete_node` is. Lock the destructive-tool policy in the spec (Phase 3). |
| R9 | **Bundle/ship clean-invariant breakage** — adding `anthropic` + a `copilot/` package trips `build.sh`'s allowlist or forbidden-pattern gate. | Low | Med | Confirmed safe — see §5. `copilot/` under `shaderbox/` ships via `cp -r shaderbox`; `anthropic` enters via `pyproject.toml`/`uv.lock` (both already in `ROOT_FILES`). The key lives in `integrations.json` at `app_data_dir()` — outside the repo, never staged. No allowlist change needed *unless* the copilot ships a resource file (then add under `shaderbox/`). |
| R10 | **No-screenshot verification gap for the chat UI** — the glfw window can't be screenshotted from the agent (`no-screenshot-driven-dev` memory; no WM on the dev display). | High | Low | Accepted constraint, low impact. Headless `make smoke` covers lifecycle/threading; the *visual* chat UX is a **maintainer manual check** (the `dev_flow.md` step-7 pattern). State the limitation once; hand the visual confirmation to the maintainer with a one-line "run `make run`, type X, check the transcript renders." |
| R11 | **Editor-buffer clobber on shader edit** — `edit_shader_source` writes the `.frag.glsl` while the user has unsaved edits in the open editor session; the hot-reload re-sync (`_reload_if_changed`) only re-syncs if texts diverge, so it could discard the user's work or fight the agent. | Med | Med | The hot-reload free lunch (`00_grounding §3`) is real but has this edge. Decide the policy in Phase 3: agent edits flush through the same `App.save`/editor path the user does (disk is source of truth, per `conventions.md` "Inline editor state lives on `App`; disk is the source of truth"), OR the agent refuses to edit a node whose editor is dirty and says so. Lock it; don't discover it at impl. |
| R12 | **Cycle-from-types** — the `copilot/` package needs `app: App` for the capability impl, but `app.py` would import `copilot` to own the worker → the no-`TYPE_CHECKING` rule forces a structural split. | Med | Low | Anticipated by `dev_flow.md ## Feature flow` (the "cycle-from-types signal" — feature 002 precedent) and `00_grounding §6`. The **capability interface** (`CopilotCapabilities` Protocol/ABC) IS the split: `copilot/` imports only the interface (a leaf), `App` implements it and owns the worker. Mirror `editor_types.py` / `paths.py` / `share_state.py` (leaf types extracted to break the cycle). Lock the module layout in the spec. |

---

## 4. Testing strategy (per phase)

The constraints: **no GL in CI** (`make smoke` needs a real GL context, so it's NOT in `make check`),
**no screenshots**, **an LLM that's non-deterministic + costs money + needs a key**. The strategy splits
every test surface into three tiers:

- **Tier A — pure unit tests (`pytest`, already a dev dep), no GL, no key, no glfw.** Run in `make check`-
  adjacent flow (or a new `make test`). The capability-seam swap (marginalia `setBookPageProvider`
  pattern, ovelia's swappable service) makes this possible: tool handlers reach the app ONLY through
  `CopilotCapabilities`; tests inject an **in-memory fake** capability and assert the handler's behavior
  (right command enqueued, right args, right LLM-facing string, generic error on a thrown exception).
  The seam itself is `02`'s deliverable; **the test plan is here.**
- **Tier B — mock-LLM agent-loop tests.** A `MockLLMService` implementing the same `ILLMService` Protocol
  the real Anthropic client implements, driven by a **scripted tool-call transcript** (deterministic:
  "turn 1 → call `list_uniforms`; turn 2 → emit text"). Asserts the loop: parses tool calls, executes via
  the registry, feeds results back, terminates on text, respects the iteration cap, compresses old tool
  results (Phase 4). No network, no cost, fully deterministic. This is how the *agent loop* gets coverage
  without a key.
- **Tier C — `make smoke` extension (headless GL, no key).** The worker thread + command queue lifecycle
  gets a headless check: in `scripts/smoke.py`, with a `MockLLMService` (so no key needed), spin up the
  copilot worker, enqueue a scripted mutation, advance frames, assert (a) the command drained + applied
  on the main thread (`node.uniform_values` changed), (b) no exception, (c) the worker **joins cleanly on
  `app.release()`** (no zombie thread — the 001 shutdown-drainage check). Add invariants alongside the
  existing popup-mutex / `current_node_id` asserts.
- **Tier D — MANUAL ONLY (maintainer).** The real LLM round-trip (needs the key, costs money, non-
  deterministic), the streaming UI, the chat-transcript rendering, the "make it pulse red" end-to-end
  demo, UI responsiveness during a live call. Per `dev_flow.md` step 7 + the no-screenshot constraint.

**Per-phase mapping:**

| Phase | Tier A (unit) | Tier B (mock-LLM loop) | Tier C (smoke) | Tier D (manual) |
|---|---|---|---|---|
| 0 | — | — | command-queue marshalling + GL-affinity + clean join | debug-key → preview changes |
| 1 | LLM-message assembly, secret load | loop produces text, no tools | worker spins w/ mock client + joins | real streamed reply; responsiveness |
| 2 | each read-handler vs in-mem capability | scripted "read then answer" | (read tools GL-free; smoke unchanged) | "what uniforms?" correct |
| 3 | each mutation-handler enqueues right command | "edit → compile-error → re-edit" self-correct | mutation marshalled headlessly | "pulse red" end-to-end; error path |
| 4 | compression, budget-cutoff note | long multi-tool transcript hits cap | (loop polish; smoke unchanged) | compound request end-to-end |

**`make smoke` *can* run headlessly without a key** — that's the whole point of the `MockLLMService`: the
worker-thread lifecycle and the GL marshalling are exercised with a deterministic fake, so the scary
*systems* part is CI-able even though the *LLM* part isn't.

---

## 5. Bundle / ship impact (the clean-invariant check)

Read `build.sh` end-to-end (the allowlist + the abort-on-forbidden gate). Verdict: **the clean-bundle
invariant holds with no `build.sh` change required.**

- **The `copilot/` subpackage ships fine.** `stage_common()` does `cp -r shaderbox "$stage/"` — any
  subpackage under `shaderbox/` (like `exporters/`, `shader_lib/`) ships automatically. No allowlist
  entry per-subpackage. (Same as `exporters/` shipped with zero `build.sh` change in feature 001.)
- **`anthropic` enters via `pyproject.toml` + `uv.lock`**, both already in `ROOT_FILES`
  (`pyproject.toml uv.lock .python-version`). The bundle is a *source distribution* — the user runs
  `uv sync` on first launch, which resolves `anthropic` from the shipped lockfile. No new ship plumbing.
  (`uv add anthropic` updates both files in the prep wave; the `claude-api` skill should be invoked for
  the SDK integration.)
- **The forbidden-pattern gate is unaffected.** `FORBIDDEN_NAMES` (`CLAUDE.md`, `Makefile`, …) and
  `FORBIDDEN_PATHS` (`ai_docs`, `.claude`, …) don't match anything a `copilot/` package introduces. The
  `00`–`07` research docs live under `ai_docs/features/020_copilot_agent/` — already inside the
  `FORBIDDEN_PATHS=(ai_docs …)` gate, so they're *asserted* not to ship.
- **The API key does NOT ship.** Confirmed: it joins `integrations.json` at `app_data_dir()`
  (`exporters/integrations.py:59` → `app_data_dir() / "integrations.json"`), which is **outside the repo
  tree** (default `~/.local/share/shaderbox/`) and never staged by `build.sh` (which only copies
  `shaderbox/` + `ROOT_FILES` + `scripts/README.md`). Same posture as the Telegram token / YouTube creds
  today. No secret leak path.
- **One thing to add (only if it materializes):** if the copilot ships a bundled resource (a default
  system-prompt `.md`, a tool cheatsheet), put it under `shaderbox/resources/` — already covered by
  `pyproject.toml`'s `include = ["shaderbox/resources/**/*"]` AND `cp -r shaderbox`. No new allowlist line.
  A new dev-only artifact (a `copilot_eval/` harness) MUST match a `FORBIDDEN_*` pattern or live outside
  `shaderbox/` so the gate keeps it out — note it in `dev_flow.md ## Build / ship` if added.

---

## 6. Spec scaffold for `020_copilot_agent.md`

Section headers + bullet stubs. Each stub points at the report (`01`–`07`) that answers it. Lock the
**locked** decisions; keep **open** ones in the Open-questions block until plan-lock.

```markdown
# 020 — Built-in coding-copilot agent

Status: research complete → DRAFT (plan-lock pending)
Shape: HIGH-blast-radius feature flow (dev_flow.md ## Feature flow). New subpackage + worker thread +
async-ish loop + new dep — the 001 (exporter) shape is the closest precedent.

## Goal
- A chat-widget agent inside ShaderBox that manipulates the app on the user's behalf: explain shaders,
  introspect uniforms, edit GLSL, set uniforms/input-shapes, manage lib files, drive render/export.
- Delivered in 5 phases (07 §1); the minimal shippable line is the end of Phase 2 (read-only assistant).

## Out of scope (each with a trigger)
- [LOCKED OUT] Multi-provider LLM (OpenAI/local). Anthropic only (house default). *Trigger: a user needs
  a provider we don't support, or a local-model push.*  ← 04
- [LOCKED OUT] Agent CREATES new lib functions from scratch (vs editing/composing existing). *Trigger:
  Phase 4 ships + a user asks the agent to author a reusable SB_* helper.*  ← 06
- [LOCKED OUT] Pixel-readback "see the rendered output" (the agent reasoning over the actual frame
  buffer). Feedback is compile-errors + uniform state, NOT vision. *Trigger: a user asks "why does it
  look wrong" where the answer needs the pixels; revisit with a frame-grab→vision tool.*  ← 06
- [LOCKED OUT] Voice / speech input. *Trigger: never, unless explicitly asked.*
- [LOCKED OUT] Per-project persisted chat history with migration. Chat is ephemeral session state.
  *Trigger: a user asks to resume a conversation across launches.*  ← 05, R7
- [LOCKED OUT] Keyring / OS-secret-store for the API key. Rides the existing cleartext-in-
  integrations.json posture. *Trigger: the existing [DEFERRAL] cleartext-secrets hardening pass.*
- [LOCKED OUT] Tool enable/disable settings UI (marginalia has it). *Trigger: the tool set grows past
  what fits a default-on policy.*  ← 02

## Design decisions (locked)
1. Worker thread owns network + agent loop; main thread drains a bounded command queue in
   update_and_draw. Worker NEVER touches GL (mirror Exporter thread affinity).  ← 01, R1/R2
2. Tool handlers reach the app ONLY through a CopilotCapabilities interface (leaf module) — breaks the
   App↔copilot cycle (no TYPE_CHECKING; feature-002 shape).  ← 02, R12
3. Mutations marshalled main↔worker via a Command queue; reads are GL-free snapshots (no marshalling).
   ← 01/02
4. Tool args = pydantic models, schema via model_json_schema(); extra="forbid" (ovelia pattern).  ← 02
5. Handler returns a string (or ok/msg/payload triple) for the LLM; generic error strings, never raw
   exceptions; user/disk text sanitized before splicing into prompt.  ← 02, R5
6. Chat widget uses imgui input_text_multiline + plain text rendering — NOT imgui_color_text_edit (avoids
   the FPE-behind-modals deferral).  ← 05, R4
7. API key in integrations.json (AnthropicIntegration), app_data_dir(), never shipped.  ← R9/§5
8. Agent loop: max-iterations + token budget + old-tool-result compression + "what I did" cutoff note.
   ← 04, R6
9. edit_shader_source writes the .frag.glsl to disk → existing hot-reload applies it on the main thread
   (the free lunch). Dirty-editor policy: <LOCK in Phase 3>.  ← R11
10. Module layout: copilot/ subpackage (loop, registry, tools/, llm seam) + CopilotCapabilities leaf;
    App owns the worker lifecycle (start lazy, join in release()).  ← 01/02

## Refactor-prep (landed as separate pre-feature commits — 07 §2)
- set_uniform_value(node_id, name, value) headless verb (todo gap a).
- create_node(template_id) arg form (todo gap b).
- AnthropicIntegration in integrations.json (secret store decision).

## Files touched (anticipated)
- NEW copilot/{__init__,loop,registry,llm,capabilities? or a leaf capabilities.py,tools/*}.py
- shaderbox/app.py — worker lifecycle, capability impl, command-queue drain hook.
- shaderbox/ui.py — drain commands at top of update_and_draw; chat widget draw call.
- shaderbox/widgets/ or popups/ — the chat widget (ui_primitives + theme).
- shaderbox/exporters/integrations.py — AnthropicIntegration.
- shaderbox/ui_models.py — chat-state field IF persisted (prefer not; R7).
- scripts/smoke.py — worker-lifecycle + command-marshalling invariants (mock LLM).
- pyproject.toml / uv.lock — anthropic dep.
- conventions.md / dev_flow.md module map / roadmap row + banner / todo (resolve the deferral).

## Manual verification (Tier D — maintainer, per phase)  ← 07 §4
- (P1) streamed reply; responsiveness (glfw.get_time() delta). (P2) "what uniforms?" correct.
- (P3) "make the background pulse red" end-to-end; broken-GLSL self-correct path. (P4) compound request.

## Open questions for the user
- Persist chat history per-project, or ephemeral? (recommend ephemeral — R7)
- Destructive-tool confirmation/undo policy (delete_node, shader overwrite)? (R8)
- Dirty-editor-buffer policy on agent shader-edit (refuse vs flush-through)? (R11)
- Phase-3 go/no-go gate: ship Phase 2 first and see usage, or commit to all 5 now? (§8)

## Review history
- (filled at review time — 07 §7)
```

---

## 7. Review plan (high-blast-radius → scale UP)

Per `dev_flow.md ## Feature flow` high-blast-radius tier: **2 pre-impl + 3+ post-impl review agents,
run as a convergence loop** (`/review-agent-loop` skill carries the protocol — adversarial-not-sympathetic
prompting, triage-then-respawn, late-round false-positive detection, the coverage-statement requirement),
**at least one reviewer anchored to a non-self-authored artifact**. Because the feature is phased, review
per-phase (Phase 3 — mutation — gets the heaviest pass; Phase 1–2 can run the lighter mid-tier).

**Pre-impl (2):**
- *Correctness & design* — internal inconsistencies, convention violations (no-`TYPE_CHECKING`, imports-
  at-top, no-`@staticmethod`), the capability-seam cycle-break, anything contradicting a locked decision.
- *Verification & blast-radius* — does the per-phase manual + smoke list catch realistic bugs; any
  invariant nothing verifies; does it correctly fold the `[DEFERRAL] built-in coding-copilot agent` +
  touch the FPE deferral correctly.

**Post-impl (3+, per phase, convergence loop):**
- *Code correctness* — bugs, **races, GL-context lifecycle, resource/thread leaks, error handling** (the
  threading-heavy dimensions matter most here).
- *Architecture & conventions* — module boundaries (copilot/ leaf vs App ownership), the capability seam,
  duplication, **imgui patterns via `/imgui-ui`** (the chat widget MUST flow through `ui_primitives`).
- *Spec-fidelity audit* (high-blast-radius extra) — walk every locked decision against the diff.

**Review dimensions specific to THIS feature (the checklist reviewers anchor to):**
1. **Threading correctness** — worker owns all network/loop; main thread only `get_nowait()`-drains;
   non-daemon + sentinel + bounded join in `release()`; no deadlock path.
2. **GL-thread affinity** — NO GL in any worker/handler method; every GL-touching effect goes through the
   command queue. (Read the handler methods, exporter-style method-walk, per 001 Manual-verification #10.)
3. **Prompt-injection** — all user/disk text sanitized before prompt splice; generic error strings only.
4. **Tool-handler correctness** — args validated (`extra="forbid"`), right command enqueued, idempotent /
   dedup where needed, destructive-tool policy honored.
5. **imgui patterns** — `ui_primitives`/`theme` tokens only; NOT `imgui_color_text_edit` (FPE deferral);
   modal/jitter rules per `/imgui-ui`.
6. **Spec-fidelity** — every locked decision (esp. the cycle-break and the marshalling) actually landed.

**Non-self-authored anchor:** one reviewer anchors to the running app (the maintainer's Tier-D manual
result) OR to `00_grounding.md` + the sibling reports (artifacts a *different* agent authored) — so the
swarm can't ratify its own contradictions (`dev_flow.md`'s docs-audit cautionary tale).

---

## 8. Adversarial section

**(a) Strongest case to NOT phase — one big feature branch.**
*The case:* the phases share so much machinery (worker, queue, registry, capability seam) that building
them incrementally means re-touching the same files five times — `app.py`'s worker lifecycle, `ui.py`'s
drain hook, `smoke.py`'s invariants all get edited per phase. A single coherent diff (the 001 precedent —
exporter refactor was one big diff) avoids the churn and lets the design settle once. Phasing also risks
shipping a Phase-1 "chat that only talks" that users find pointless, then never finishing.
*Resolution:* **reject for delivery, keep for the seam, partially accept the "design once" point.** The
phases aren't independent *implementations* — they're an **ordered reveal of one design**: you design the
worker + queue + capability seam *up front* (in the spec), then land them in risk-ascending slices. The
churn is real but cheap (each phase adds tools/handlers to an established seam, not re-architecting it).
And the 001 one-big-diff was forced by a delete-and-replace (old + new couldn't coexist); here they can.
The decisive argument: **the threading risk (R1/R3) is un-precedented and must be proven before mutation
exists** — a single diff can't isolate "did the marshalling break or the agent loop break." Phasing is the
bisect. *Partial accept:* design the full architecture in the spec (so the seam is settled once), phase the
*landing*.

**(b) Strongest case the feature is premature / "explain my shader" is the right first bite.**
*The case (stated honestly):* this is a solo itch.io name-your-price tool. A full mutation-capable agent
is a large, ongoing maintenance surface (a new dep, a worker thread, an LLM seam, prompt-injection, cost
UX, the threading risk) for a product whose users are hobbyists writing shaders for fun. The reference
agents (cc-server, marginalia, ovelia) are *products* with users paying for the agent; ShaderBox's agent
is a feature inside a tool. A **read-only "explain my shader / what do these uniforms do / why won't this
compile" assistant** delivers most of the *teaching* value (the thing a GLSL hobbyist actually wants) at a
fraction of the risk — no marshalling, no GL affinity, no destructive-mutation policy, no editor-clobber
edge. The mutation half ("make it pulse red") is a flashy demo but the hard part, and a hobbyist can
arguably just paste the agent's suggested GLSL themselves.
*Resolution:* **this is exactly why the phasing exists — and the maintainer should hear it as a real
stopping point, not just a milestone.** Phase 1+2 IS the "explain my shader" assistant, and it's the
minimal shippable line (§1). The recommendation: **ship Phase 2, let it be used, and make Phase 3 a
deliberate re-decision** — not an automatic continuation. If Phase 2 lands and users love the *explaining*
but never miss the *doing*, stop. The honest verdict: not premature *if phased*; it would be overbuilt if
attempted as one mutation-first push. The phasing converts "are we overbuilding?" from a yes/no gamble
into a checkpoint.

**(c) The single most likely way this DERAILS + how to de-risk early.**
*The derailment:* **the threading/GL-affinity marshalling turns out awkward or fragile** — the team builds
the LLM seam and tools, then discovers that mutating-GL-state-from-a-worker-the-user-is-watching doesn't
compose cleanly with the synchronous frame loop (command ordering, mid-frame mutation, the editor-clobber
race R11, a mutation that needs a *result* read back from GL before the next tool call). This is the part
with NO copy-able precedent (the exporters only marshal output one-way, worker→main, GL-free worker). If
it's discovered late, half the tool layer is built on a broken foundation.
*De-risk (the Phase-0 spike):* **before any LLM work, build the no-LLM command queue and prove it.** A
worker thread enqueues `set_uniform_value`; `update_and_draw` drains + applies it on the main thread; the
preview changes; `make smoke` asserts the round-trip headlessly (mutation lands, no exception, thread
joins). If a tool needs a *result* read back (e.g. compile errors after an edit), prove that round-trip
too (enqueue edit → next frame compiles → worker reads `compile_unit.errors` from a snapshot). This costs
one throwaway/foundational commit and surfaces the hardest unknown for the price of the cheapest phase. If
the spike is ugly, the maintainer re-scopes (maybe Phase 2 read-only is the whole product) before
investing in the LLM seam. **This spike is the go/no-go gate for the mutation half of the feature.**

---

## 9. Open questions (for the maintainer / synthesis)

1. **Phase-3 gate:** commit to all 5 phases now, or ship Phase 2 (read-only assistant) and re-decide
   mutation based on usage? (Recommendation §8b: ship Phase 2, re-decide.)
2. **Chat persistence:** ephemeral session state (recommend, zero migration — R7) or per-project history?
3. **Destructive-tool policy:** confirmation gate / undo for `delete_node` + shader overwrite, or trust
   the preview-as-feedback loop? (R8)
4. **Dirty-editor policy on agent shader-edit:** refuse-if-dirty vs flush-through-disk? (R11)
5. **Command-queue result-read:** does any Phase-3 tool need to read a GL result back *within the same
   agent turn* (e.g. compile-errors after an edit), or can the read be a *next-turn* tool call? The former
   needs a request/response marshalling shape, not just fire-and-forget — the spike (§8c) must answer this.
6. **`make test` target:** the unit (Tier A) + mock-LLM (Tier B) tests want a `pytest` runner — add a
   `make test` and wire it into the pre-commit/CI flow, or run `pytest` directly? (`pytest` is already a
   dev dep.)
7. **Where do the sibling reports land their conclusions** that change a locked decision here — e.g. if
   `01` (threading) concludes the command queue needs a different shape than the exporter mailbox, this
   report's R1/R2/§8c mitigations update in the synthesis (`99`).
```
