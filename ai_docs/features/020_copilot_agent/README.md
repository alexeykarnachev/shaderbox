# 020 Copilot Agent — research phase (reading order)

A built-in coding-copilot agent: an in-app LLM chat that manipulates ShaderBox via tools (edit
shaders, set uniforms, manage the lib, drive render/export). This directory is the **research +
refactor-prep audit** the maintainer asked for BEFORE building it — it is NOT the spec yet. The spec
(`020_copilot_agent.md`) gets drafted from `99_synthesis.md` once the maintainer signs off on scope.

## Read in this order

0. **`10_skeleton_plan.md`** — the CURRENT front of work: the module skeleton + seams design (per the
   maintainer's reframe — architecture-shape first, capabilities later). The next deliverable to build
   the locked spec from. Read after the synthesis for context.
1. **`99_synthesis.md`** — the merged conclusion + **§0 maintainer decisions** (which override several
   of the original reports). The consensus architecture, the refactor-prep plan, the risk register, the
   open questions, the spec scaffold.
2. **`00_grounding.md`** — the shared factual anchor: the stack + the single hardest constraint
   (single-threaded GL frame loop), the seams the copilot attaches to, the three reference agents
   studied (cc-server / marginalia / ovelia) + what ShaderBox does differently.

Then the seven angle reports (each: a recommendation, options-with-tradeoffs, a concrete design, an
adversarial self-attack, open questions):

3. `01_threading_architecture.md` — the worker thread + the `MainThreadBridge` GL-marshalling (the
   one genuinely new mechanism).
4. `02_tool_registry_seam.md` — the `ToolRegistry` + the `CopilotCapabilities` seam (reach the app
   without importing imgui / without a cycle) + the v1 tool catalog.
5. `03_refactor_prep_audit.md` — the leaking-seams audit + the GL-free/GL-touching verb partition +
   the "app.py does NOT need splitting" verdict.
6. `04_llm_integration.md` — the Anthropic tool-use loop, the `ILLMClient` seam, prompt + caching
   design, the API key, the SDK-vs-httpx call + the IPv4 concern.
7. `05_chat_ui_ux.md` — the chat panel (4th tab), message rendering in immediate-mode imgui, the
   streaming/tool-call/error UI, the not-configured gate, the safety UX.
8. `06_glsl_domain.md` — what a GLSL copilot must be good at: the in-process compile-feedback loop
   (the crown jewel), the SB_ lib, uniforms, the GLSL authoring rules for the prompt.
9. `07_phasing_risk_spec.md` — phasing, sequencing, the risk register, the testing strategy, the
   bundle/ship check, the spec scaffold, the review plan.

Follow-up investigations (spawned after the maintainer's first round of answers):

10. `08_autosave_investigation.md` — the generalized editor auto-flush hook (a near-term standalone
    change that also dissolves the copilot's dirty-editor clobber).
11. `09_llm_layer_study.md` — the OpenRouter LLM-client seam yanked from cc-server (cost tracking,
    tooled responses, structured output, the module skeleton). **Overrides `04`'s Anthropic design.**
12. `10_skeleton_plan.md` — the module skeleton & seams (see item 0 above).

## One-line conclusion

Build it, phased; treat read-only "explain my shader" (Phase 2) as a real shipping checkpoint;
refactor-prep is one document (done) + one ~15-line uniform-setter extraction; app.py is NOT split;
the differentiator is the in-process compile-feedback loop.
