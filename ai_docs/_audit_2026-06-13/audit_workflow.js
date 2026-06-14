export const meta = {
  name: 'shaderbox-nightly-audit',
  description: 'Mine ShaderBox feature artifacts for durable knowledge + emergent insights, find recurring mistakes, and audit the dev-flow doc harness for consistency',
  phases: [
    { title: 'Read clusters', detail: 'parallel deep-read of feature-spec clusters into structured findings' },
    { title: 'Mine patterns', detail: 'cross-feature synthesis: emergent insights + recurring-mistake classes' },
    { title: 'Harness audit', detail: 'consistency of dev_flow / roadmap / todo / conventions against their own rules' },
    { title: 'Verify', detail: 'adversarial check of every claimed certain-fix before it is auto-applied' },
    { title: 'Synthesize', detail: 'merge into one report; split certain-fixes from morning-review items' },
  ],
}

const REPO = '/home/akarnachev/src/shaderbox'
const DOCS = `${REPO}/ai_docs`
const FEAT = `${DOCS}/features`

// ---------------------------------------------------------------------------
// Phase 1 — deep-read feature clusters in parallel.
// Each cluster reads a coherent slice of the history so the reader can see
// the arc (what was tried, superseded, re-derived) not just isolated specs.
// ---------------------------------------------------------------------------
phase('Read clusters')

const CLUSTERS = [
  {
    key: 'foundations_001_011',
    label: 'foundations 001-011',
    files: ['001_exporter_refactor.md','002_ui_widgets_extraction.md','003_modelbox_removal.md','004_imgui_bundle_migration.md','005_ui_redesign_foundation.md','006_inline_editor.md','007_release_pipeline_hardening.md','008_uniform_input_shapes.md','009_integrations_rework.md','010_outlet_render_rework.md','011_ui_library_consolidation.md'],
    theme: 'early architecture, exporters, the imgui-bundle migration, the UI three-layer split, release pipeline',
  },
  {
    key: 'authoring_012_019',
    label: 'authoring 012-019',
    files: ['012_youtube_export.md','013_authoring_feedback_loop.md','014_compile_unit_refactor.md','015_shader_include_library.md','016_lib_file_management.md','017_structure_reorg.md','018_keyboard_control.md','019_keyboard_navigation.md'],
    theme: 'shader-include library, compile-unit refactor, structure reorg, keyboard command + nav layers',
  },
  {
    key: 'copilot_design_020a',
    label: 'copilot design 020 (00-10)',
    files: ['020_copilot_agent/00_grounding.md','020_copilot_agent/01_threading_architecture.md','020_copilot_agent/02_tool_registry_seam.md','020_copilot_agent/03_refactor_prep_audit.md','020_copilot_agent/04_llm_integration.md','020_copilot_agent/05_chat_ui_ux.md','020_copilot_agent/06_glsl_domain.md','020_copilot_agent/07_phasing_risk_spec.md','020_copilot_agent/08_autosave_investigation.md','020_copilot_agent/09_llm_layer_study.md','020_copilot_agent/10_skeleton_plan.md'],
    theme: 'the copilot design phase: threading architecture, tool-registry seam, LLM integration, phasing/risk',
  },
  {
    key: 'copilot_build_020b',
    label: 'copilot build 020 (11-20)',
    files: ['020_copilot_agent/11_capability_wave_spec.md','020_copilot_agent/12_edit_robustness.md','020_copilot_agent/13_glsl_lexer.md','020_copilot_agent/14_slice2_line_editing.md','020_copilot_agent/15_edit_safety.md','020_copilot_agent/16_cross_project_tools.md','020_copilot_agent/17_gate_ui.md','020_copilot_agent/18_render_publish_tools.md','020_copilot_agent/19_credential_pack_tools.md','020_copilot_agent/20_ui_ux_polish.md'],
    theme: 'the copilot capability build-out: edit robustness, the GLSL lexer, line-editing (later removed), edit-safety, cross-project tools, gate UI, render/publish/credential tools',
  },
  {
    key: 'copilot_evolve_020c',
    label: 'copilot evolve 020 (21-30 + meta)',
    files: ['020_copilot_agent/21_chat_widgets_and_links.md','020_copilot_agent/22_template_library.md','020_copilot_agent/23_uniform_history_chat_untangle.md','020_copilot_agent/24_ui_polish_wave.md','020_copilot_agent/25_context_fill_indicator.md','020_copilot_agent/27_structural_shader_view.md','020_copilot_agent/28_prompt_tier_architecture.md','020_copilot_agent/29_working_set_scratchpad.md','020_copilot_agent/30_turn_rollback.md','020_copilot_agent/99_synthesis.md','020_copilot_agent/_DECISIONS_LOG.md','020_copilot_agent/README.md'],
    theme: 'copilot evolution: chat widgets, template library, prompt-tier architecture, working-set scratchpad, turn rollback, the decisions log + synthesis',
  },
  {
    key: 'copilot_robust_021_039',
    label: 'copilot robustness 021-039',
    files: ['021_logging_refactor.md','022_copilot_chat_persistence.md','023_app_refinement_wave.md','025_project_session_extraction.md','029_tool_first_class_labels.md','031_parallel_structure_sweep.md','033_copilot_robustness_wave.md','036_anchored_replace_lines.md','038_copilot_polish_wave.md','039_content_addressed_editing.md'],
    theme: 'logging, chat persistence, project-session extraction, the editing-tool saga (036 anchored -> removed by 039 content-addressed), robustness waves',
  },
  {
    key: 'dogfood_026_037',
    label: 'dogfood 026-037',
    files: ['026_copilot_dogfood_harness.md','027_interactive_dogfood_server.md','035_dogfood_report_mega.md','037_dogfood_report_036anchor.md','039_dogfood_report_gate.md','039_dogfood_report_mega.md'],
    theme: 'the dogfood harness arc + the dogfood findings reports (what dogfooding actually surfaced and whether it was acted on)',
  },
  {
    key: 'ui_polish_028_034',
    label: 'ui polish 028/032/034',
    files: ['028_ui_polish_wave.md','032_sdf_shader_library.md','034_ui_polish_wave_2.md'],
    theme: 'the walkthrough-driven UI polish waves + the SDF shader library seed',
  },
  {
    key: 'script_engine_040_045',
    label: 'script engine 040-045',
    files: ['040_uniform_script_engine.md','041_stateful_script_engine.md','042_script_ui.md','044_node_brain_script.md','045_script_ux_redesign.md'],
    theme: 'the CPU-script engine arc: 040 superseded-before-ship by 041 stateful class, 042 placeholder UI superseded by 045, 044 node-brain — a contract redesigned twice and a UI rebuilt once',
  },
]

const CLUSTER_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['cluster', 'durable_insights', 'mistakes_or_smells', 'redesign_signals', 'notable_quotes'],
  properties: {
    cluster: { type: 'string' },
    durable_insights: {
      type: 'array',
      description: 'Reusable, slow-changing knowledge worth lifting into a higher-altitude home (conventions.md, a skill, or kept as a pointer). Each: what it is + why it generalizes beyond the one feature.',
      items: {
        type: 'object', additionalProperties: false,
        required: ['insight', 'evidence', 'why_generalizes', 'proposed_home'],
        properties: {
          insight: { type: 'string' },
          evidence: { type: 'string', description: 'spec file + a short quoted phrase or section name proving it' },
          why_generalizes: { type: 'string' },
          proposed_home: { type: 'string', description: 'where it should live: conventions.md ## section, a skill, dev_flow.md, or "already filed — pointer only"' },
        },
      },
    },
    mistakes_or_smells: {
      type: 'array',
      description: 'Errors, dead-ends, rework, or smells visible in THIS cluster. Tag whether it looks like a one-off or an instance of a recurring class.',
      items: {
        type: 'object', additionalProperties: false,
        required: ['description', 'evidence', 'looks_recurring'],
        properties: {
          description: { type: 'string' },
          evidence: { type: 'string', description: 'spec file + quoted phrase/section' },
          looks_recurring: { type: 'boolean' },
        },
      },
    },
    redesign_signals: {
      type: 'array',
      description: 'Things superseded/reverted/redesigned (e.g. 040->041 contract redesign, 036->039 tool removal, line-editing removed). What was tried, why it failed, what replaced it. These are the richest source of emergent insight.',
      items: {
        type: 'object', additionalProperties: false,
        required: ['what_changed', 'why', 'lesson'],
        properties: {
          what_changed: { type: 'string' },
          why: { type: 'string' },
          lesson: { type: 'string', description: 'the transferable lesson, stated generically' },
        },
      },
    },
    notable_quotes: {
      type: 'array',
      description: 'Up to 3 short verbatim lines from the specs that crystallize a principle worth preserving.',
      items: { type: 'string' },
    },
  },
}

const clusterReads = await parallel(CLUSTERS.map((c) => () => {
  const fileList = c.files.map((f) => `${FEAT}/${f}`).join('\n  ')
  return agent(
    `You are auditing the ShaderBox project's feature-spec artifacts. ShaderBox is a solo real-time GLSL shader playground (moderngl + glfw + imgui-bundle, Python 3.12) with an in-app LLM coding copilot and a CPU-script engine.

Read these spec files IN FULL — this is the "${c.label}" cluster, themed around: ${c.theme}.

  ${fileList}

Also skim ${DOCS}/dev_flow.md (sections "Documentation discipline" and "Feature flow") so you understand the project's own standards for what a good artifact looks like.

Your job is to MINE this cluster for transferable knowledge, NOT to summarize it. Specifically:

1. DURABLE INSIGHTS — knowledge that outlives the feature: an architectural principle, a library/SDK footgun + workaround, a process lesson, a design heuristic that proved out. For each, say WHY it generalizes beyond the one feature and WHERE it should live (conventions.md / a skill / dev_flow.md / already-filed). Skip anything that's purely about one feature's mechanics with no reuse value.

2. MISTAKES OR SMELLS — anything that reads like an error, a dead-end, rework, an over-built thing later torn out, a thing fixed twice, a premise that turned out wrong. Tag each with whether it looks like a ONE-OFF or an instance of a RECURRING class (so the next phase can cluster them).

3. REDESIGN SIGNALS — the richest seam: anything superseded, reverted, or redesigned (this history is FULL of it — 040's contract redesigned by 041 before it even shipped, 036's anchored-replace removed by 039, copilot line-editing built then torn out, 005's wide-screen layout reverted). For each: what was tried, why it failed, what replaced it, and the GENERIC transferable lesson. These are where emergent insights hide — a contract redesigned twice tells you something about how the maintainer's mental model evolved.

4. NOTABLE QUOTES — up to 3 short verbatim lines that crystallize a principle.

Be adversarial and concrete: cite the spec file + a quoted phrase or section name for every claim. Do not invent insights to fill the schema — an empty array is fine if the cluster genuinely has nothing in a category. Quality over quantity. Return the structured object.`,
    { label: `read:${c.key}`, phase: 'Read clusters', schema: CLUSTER_SCHEMA },
  ).then((r) => (r ? { ...r, cluster: c.label } : null))
}))

const reads = clusterReads.filter(Boolean)
log(`Read ${reads.length}/${CLUSTERS.length} clusters. Total durable insights: ${reads.reduce((n, r) => n + r.durable_insights.length, 0)}, mistakes/smells: ${reads.reduce((n, r) => n + r.mistakes_or_smells.length, 0)}, redesign signals: ${reads.reduce((n, r) => n + r.redesign_signals.length, 0)}.`)

const readsBlob = JSON.stringify(reads, null, 1)

// ---------------------------------------------------------------------------
// Phase 2 — cross-feature synthesis. Two independent miners over the SAME
// raw findings, each with a distinct lens, so we get diverse pattern framings
// rather than one model's single read.
// ---------------------------------------------------------------------------
phase('Mine patterns')

const RECURRING_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['recurring_mistakes', 'meta_observations'],
  properties: {
    recurring_mistakes: {
      type: 'array',
      description: 'Mistake CLASSES that recur across 2+ features. The whole point: what does this project get wrong repeatedly?',
      items: {
        type: 'object', additionalProperties: false,
        required: ['class_name', 'pattern', 'instances', 'root_cause_hypothesis', 'cheapest_guardrail'],
        properties: {
          class_name: { type: 'string', description: 'a short memorable name for the mistake class' },
          pattern: { type: 'string', description: 'the repeated shape of the error' },
          instances: { type: 'array', items: { type: 'string' }, description: 'feature/spec citations where it appears' },
          root_cause_hypothesis: { type: 'string' },
          cheapest_guardrail: { type: 'string', description: 'the smallest process/doc/code change that would catch it next time' },
        },
      },
    },
    meta_observations: {
      type: 'array',
      description: 'Higher-order observations about HOW this project evolves (e.g. tendency to over-build then tear out, contracts redesigned before shipping, dogfood-driven hardening). Each with supporting instances.',
      items: {
        type: 'object', additionalProperties: false,
        required: ['observation', 'instances', 'is_strength_or_weakness'],
        properties: {
          observation: { type: 'string' },
          instances: { type: 'array', items: { type: 'string' } },
          is_strength_or_weakness: { type: 'string', enum: ['strength', 'weakness', 'neutral'] },
        },
      },
    },
  },
}

const INSIGHT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['emergent_insights', 'knowledge_to_lift', 'cross_feature_threads'],
  properties: {
    emergent_insights: {
      type: 'array',
      description: 'Non-obvious insights that only appear when you look ACROSS features — things no single spec states but the corpus as a whole reveals.',
      items: {
        type: 'object', additionalProperties: false,
        required: ['insight', 'derived_from', 'so_what'],
        properties: {
          insight: { type: 'string' },
          derived_from: { type: 'array', items: { type: 'string' } },
          so_what: { type: 'string', description: 'the actionable consequence for future work' },
        },
      },
    },
    knowledge_to_lift: {
      type: 'array',
      description: 'Concrete durable knowledge that is currently buried in a feature spec but should be lifted to a higher-altitude home (conventions.md design-decision/known-quirk, a skill, dev_flow). Dedup across clusters; pick the strongest framing.',
      items: {
        type: 'object', additionalProperties: false,
        required: ['knowledge', 'currently_in', 'proposed_home', 'lift_or_pointer'],
        properties: {
          knowledge: { type: 'string' },
          currently_in: { type: 'string' },
          proposed_home: { type: 'string' },
          lift_or_pointer: { type: 'string', enum: ['lift-full', 'pointer-only', 'already-filed'], description: 'whether to copy it up, leave it and add a pointer, or it is already at the right altitude' },
        },
      },
    },
    cross_feature_threads: {
      type: 'array',
      description: 'Narrative threads that span many features (e.g. the editing-tool evolution: byte-match -> token-lexer -> line-anchor -> content-addressed; or the headless-extraction thread). Each thread is itself an insight about the projects trajectory.',
      items: {
        type: 'object', additionalProperties: false,
        required: ['thread', 'arc', 'where_it_points'],
        properties: {
          thread: { type: 'string' },
          arc: { type: 'string', description: 'the sequence of features and how the idea evolved' },
          where_it_points: { type: 'string', description: 'what this trajectory suggests is coming or should come next' },
        },
      },
    },
  },
}

const [recurring, emergent] = await parallel([
  () => agent(
    `You are the RECURRING-MISTAKE analyst for a nightly audit of the ShaderBox project. Below is structured per-cluster findings mined from ~45 feature specs by a fleet of readers. Each cluster lists durable_insights, mistakes_or_smells (tagged looks_recurring), and redesign_signals.

Your ONE job: find the mistake CLASSES that recur across 2+ features. The maintainer explicitly asked "what errors do we repeat from time to time?" — answer THAT. Cluster the individual mistakes_or_smells (especially the looks_recurring=true ones) and the redesign_signals into named classes. For each class: the repeated shape, the instances (cite features), a root-cause hypothesis, and the CHEAPEST guardrail (a doc rule, a checklist item, a test, a code primitive) that would catch the next instance.

Also surface meta-observations about HOW this project evolves — and be honest about whether each is a strength or a weakness (over-building then tearing out is a weakness if it wastes effort, a strength if it is cheap exploration that converges).

Be skeptical: a "class" needs 2+ genuine instances, not one instance dressed up. If something appears recurring but the instances are actually distinct, say so by not including it. Cite features for every instance.

RAW FINDINGS:
${readsBlob}`,
    { label: 'mine:recurring', phase: 'Mine patterns', schema: RECURRING_SCHEMA },
  ),
  () => agent(
    `You are the EMERGENT-INSIGHT analyst for a nightly audit of the ShaderBox project. Below is structured per-cluster findings mined from ~45 feature specs. Each cluster lists durable_insights, mistakes_or_smells, and redesign_signals.

Your job: find the insights that only appear when you look ACROSS the whole corpus — things no single spec states but the trajectory as a whole reveals. The maintainer asked for "emergent insights." Three outputs:

1. EMERGENT INSIGHTS — non-obvious cross-feature truths (e.g. a recurring tension between two values, a design principle the maintainer keeps re-discovering, a category of feature that always over-runs). Derive each from 2+ clusters and state the actionable so-what.

2. KNOWLEDGE TO LIFT — durable knowledge currently buried in a feature spec that belongs at a higher altitude (conventions.md design-decision or known-quirk, a skill, dev_flow). The project's own rule is "one canonical home per concept; file knowledge at the altitude matching its volatility." Dedup across clusters, pick the strongest framing, and say whether to lift-full, leave-a-pointer, or it's already-filed.

3. CROSS-FEATURE THREADS — narrative arcs spanning many features (the editing-tool evolution, the headless-extraction push, the script-engine contract redesigns, the dogfood-driven hardening loop). Each thread is itself an insight about where the project is heading.

Be concrete and cite features. Do not pad — a sharp short list beats a long vague one.

RAW FINDINGS:
${readsBlob}`,
    { label: 'mine:emergent', phase: 'Mine patterns', schema: INSIGHT_SCHEMA },
  ),
])

log(`Pattern mining done. Recurring classes: ${recurring?.recurring_mistakes?.length ?? 0}, emergent insights: ${emergent?.emergent_insights?.length ?? 0}, knowledge-to-lift: ${emergent?.knowledge_to_lift?.length ?? 0}.`)

// ---------------------------------------------------------------------------
// Phase 3 — harness consistency audit. Each auditor reads the live harness
// files and checks ONE dimension against the harness's OWN stated rules
// (dev_flow "Documentation discipline" is the checklist — this is the
// non-self-authored anchor). Findings carry a fix_confidence so the next
// phase can split certain from uncertain.
// ---------------------------------------------------------------------------
phase('Harness audit')

const HARNESS_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['dimension', 'findings'],
  properties: {
    dimension: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['title', 'detail', 'location', 'rule_violated', 'severity', 'fix_confidence', 'proposed_fix'],
        properties: {
          title: { type: 'string' },
          detail: { type: 'string', description: 'what is inconsistent/stale/wrong, with the evidence' },
          location: { type: 'string', description: 'file + section name (NOT line number — line numbers drift) + a quoted phrase so the fixer can grep it' },
          rule_violated: { type: 'string', description: 'which dev_flow Documentation-discipline rule or CLAUDE.md rule this breaks, if any' },
          severity: { type: 'string', enum: ['high', 'medium', 'low'] },
          fix_confidence: { type: 'string', enum: ['certain', 'likely', 'judgment-call'], description: 'certain = the fix is mechanical and unambiguous (a stale path, a broken cross-ref, a line-number citation, a wrong status); judgment-call = needs the maintainer to decide' },
          proposed_fix: { type: 'string', description: 'the exact change. For "certain" findings give the precise edit (old phrase -> new phrase) so it can be verified and applied.' },
        },
      },
    },
  },
}

const HARNESS_DIMENSIONS = [
  {
    key: 'cross_refs',
    label: 'cross-references + stale paths',
    prompt: `Audit dimension: CROSS-REFERENCES AND STALE PATHS. Verify that every file path, spec citation (ai_docs/features/NNN_*.md), section-name reference, and skill name mentioned in roadmap.md, todo.md, dev_flow.md, conventions.md, and CLAUDE.md actually RESOLVES. Use the Read tool and ls/grep (via Bash) to check: do referenced files exist? Do referenced section names exist in the cited file? Are there citations by line number (banned — dev_flow says "Cite by section name, not line number")? Does a feature row cite a spec file that does not exist, or vice versa (a spec with no row)? Are the module-map paths in dev_flow.md "### Module map" still present on disk? CHECK ACTUAL DISK STATE, do not assume.`,
  },
  {
    key: 'roadmap_todo_coherence',
    label: 'roadmap <-> todo <-> features coherence',
    prompt: `Audit dimension: ROADMAP / TODO / FEATURES COHERENCE. Check: (a) every feature directory under ai_docs/features/ has a roadmap row, and every roadmap row with a spec citation points at a real file; (b) feature numbering gaps (024, 026 vs 025, 030, 043 missing?) — is each gap explained or a dangling reference?; (c) "partial" roadmap rows — does each have a corresponding todo.md deferral capturing the unfinished half, per dev_flow "Mid-flight escalation"/"Closing out work"? A partial with no tracked remainder is a leak; (d) "superseded" rows — does the superseding feature actually exist and reference back?; (e) todo.md deferrals whose Trigger has plausibly already fired (the thing was built) and should be deleted per "Resolved entries get deleted in the resolving commit". Read roadmap.md, todo.md fully and ls ai_docs/features/.`,
  },
  {
    key: 'doc_discipline',
    label: 'documentation-discipline rule compliance',
    prompt: `Audit dimension: DOCUMENTATION-DISCIPLINE COMPLIANCE. dev_flow.md "## Documentation discipline" states the project's own rules. Audit roadmap.md, todo.md, conventions.md against them: (a) "Active-context banner gets rewritten NOT appended, <=200 words, no banner-history" — is the banner within budget and free of carry-over/archived phrases? Count its words; (b) "Roadmap rows index; feature specs narrate" — are any rows bloated into multi-sentence narratives that belong in the spec? (rows ARE allowed one dense brief, but flag rows that have clearly become a worklog); (c) "todo.md is a grep-by-trigger index" — does every deferral name a concrete observable Trigger? Are there N entries on the same trigger that should be ONE rolling entry?; (d) "One canonical home per concept" — is any rule restated (paraphrased) in 3+ places, or a "false-inheritance hybrid" (inherits from X then silently drops a clause)?; (e) "Resolved entries get deleted" — any [RESOLVED]/strikethrough/kept-for-posterity retentions? Read the files and quote offending passages.`,
  },
  {
    key: 'conventions_quirks',
    label: 'conventions + known-quirks freshness',
    prompt: `Audit dimension: CONVENTIONS / KNOWN-QUIRKS FRESHNESS. Read conventions.md fully. Check: (a) each "Design decision" has its "revisit if Y" clause and the cited spec exists; is any decision now contradicted by a later feature (e.g. a decision says X but feature 0NN changed it)?; (b) each "Known quirk" workaround — is it still needed, or did a later refactor/version-bump make it dead? Look for quirks referencing removed code (e.g. line-editing tools removed by 039, the anchor machinery, wiring.py deleted by 031, ModelBox removed by 003); (c) the "sanctioned type-ignore / suppression allowlist" — cross-check against the actual suppressions in the codebase (grep for "# type: ignore" and "# noqa" under shaderbox/) — is the allowlist in sync with reality (no undocumented suppressions, no allowlisted-but-removed ones)?; (d) CLAUDE.md "Code rules" claims (no TYPE_CHECKING, no staticmethod, imports at top) — spot-check via grep that the codebase actually holds to them. Quote specifics and cite.`,
  },
]

const harnessResults = await parallel(HARNESS_DIMENSIONS.map((d) => () =>
  agent(
    `You are a harness-consistency auditor for the ShaderBox project's documentation harness. The harness files are:
- ${DOCS}/roadmap.md (Active-context banner + feature rows index)
- ${DOCS}/todo.md (grep-by-trigger deferral index)
- ${DOCS}/dev_flow.md (the SOURCE OF TRUTH for how work happens + "## Documentation discipline" rules)
- ${DOCS}/conventions.md (design decisions + known quirks + suppression allowlist)
- ${REPO}/CLAUDE.md (cold-start chain + hard rules + code rules)
- specs under ${FEAT}/

${d.prompt}

This is a real audit against ACTUAL DISK STATE — use Read, Bash (ls/grep/find), to verify, never assume. The dev_flow.md "## Documentation discipline" section is your anchoring checklist — quote the specific rule each finding violates. For every finding set fix_confidence honestly: "certain" ONLY for mechanical, unambiguous fixes (a path that 404s, a line-number citation that should be a section name, a status that is provably wrong, a dead cross-ref) where you can give the exact old->new edit; "likely" where the fix is probably right but worth a glance; "judgment-call" for anything needing the maintainer's taste. Be precise in the location field so a fixer can grep the exact phrase. Do not invent findings to fill the array — a clean dimension reports few or zero findings. Return the structured object for dimension "${d.label}".`,
    { label: `audit:${d.key}`, phase: 'Harness audit', schema: HARNESS_SCHEMA },
  ).then((r) => (r ? { ...r, dimension: d.label } : null))
))

const harness = harnessResults.filter(Boolean)
const allHarnessFindings = harness.flatMap((h) => h.findings.map((f) => ({ ...f, dimension: h.dimension })))
const certainFindings = allHarnessFindings.filter((f) => f.fix_confidence === 'certain')
log(`Harness audit done. ${allHarnessFindings.length} findings across ${harness.length} dimensions; ${certainFindings.length} marked certain-fix.`)

// ---------------------------------------------------------------------------
// Phase 4 — adversarially verify every claimed CERTAIN fix before it is
// auto-applied. A skeptic re-checks disk state and tries to REFUTE that the
// fix is safe + correct. Only survivors are handed to the main loop to apply.
// ---------------------------------------------------------------------------
phase('Verify')

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['is_real', 'is_safe_mechanical', 'verified_fix', 'verdict_reason'],
  properties: {
    is_real: { type: 'boolean', description: 'is the inconsistency actually present on disk right now?' },
    is_safe_mechanical: { type: 'boolean', description: 'is the proposed fix unambiguous, behavior-neutral for the docs meaning, and not a judgment call in disguise?' },
    verified_fix: {
      type: 'object', additionalProperties: false,
      required: ['file', 'find', 'replace', 'note'],
      properties: {
        file: { type: 'string', description: 'absolute path of the file to edit' },
        find: { type: 'string', description: 'EXACT current text to replace (must be unique in the file) — empty string if not a simple find/replace' },
        replace: { type: 'string', description: 'exact replacement text' },
        note: { type: 'string', description: 'if not expressible as one find/replace, describe the precise edit here instead' },
      },
    },
    verdict_reason: { type: 'string' },
  },
}

const certainVerdicts = certainFindings.length
  ? await parallel(certainFindings.map((f) => () =>
      agent(
        `You are an ADVERSARIAL verifier. An audit flagged the following ShaderBox doc-harness inconsistency as a CERTAIN, mechanical, auto-applyable fix. Your DEFAULT is skepticism: assume it is NOT safe to auto-apply until you prove otherwise from disk.

FINDING:
- dimension: ${f.dimension}
- title: ${f.title}
- detail: ${f.detail}
- location: ${f.location}
- rule_violated: ${f.rule_violated}
- proposed_fix: ${f.proposed_fix}

Do this:
1. Open the cited file(s) with Read and Bash (grep for the exact quoted phrase) and confirm the inconsistency EXISTS right now, verbatim. If the phrase is not found exactly, is_real=false.
2. Decide if the fix is truly MECHANICAL and SAFE: a stale path, a broken cross-ref, a line-number-citation->section-name, a provably-wrong status, deletion of a resolved entry whose work provably landed. If applying it requires ANY taste, rewording choice, or could plausibly be wrong (e.g. "delete this deferral" but you cannot confirm the work landed), then is_safe_mechanical=false — kick it to the maintainer.
3. If safe, produce the EXACT find/replace: the find field must be text that occurs EXACTLY ONCE in the file (include enough surrounding context to be unique) and the replace field the corrected text. If the edit is a pure deletion, set replace to the empty string (and make find the full block to remove). If it cannot be expressed as a single find/replace, set is_safe_mechanical=false OR describe it precisely in the note field and leave find/replace empty.

Be strict. A false "safe" verdict means we corrupt a doc unattended overnight. When uncertain, refuse (is_safe_mechanical=false). Return the structured verdict.`,
        { label: `verify:${f.dimension.slice(0, 14)}`, phase: 'Verify', schema: VERDICT_SCHEMA },
      ).then((v) => (v ? { finding: f, verdict: v } : null)))
    )
  : []

const verified = certainVerdicts.filter(Boolean)
const autoApply = verified.filter((v) => v.verdict.is_real && v.verdict.is_safe_mechanical && v.verdict.verified_fix.find)
const kickedBack = verified.filter((v) => !(v.verdict.is_real && v.verdict.is_safe_mechanical && v.verdict.verified_fix.find))
log(`Verification done. ${autoApply.length} fixes survived as auto-applyable; ${kickedBack.length} kicked back to maintainer review.`)

// ---------------------------------------------------------------------------
// Phase 5 — synthesize everything into one report. The synthesizer writes the
// full markdown so the main loop can drop it on disk and apply the verified
// fixes. It does NOT touch files itself.
// ---------------------------------------------------------------------------
phase('Synthesize')

const payload = {
  cluster_reads: reads,
  recurring_mistakes: recurring,
  emergent_insights: emergent,
  harness_findings: allHarnessFindings,
  auto_apply_fixes: autoApply.map((v) => ({ finding: v.finding, fix: v.verdict.verified_fix })),
  kicked_back_to_maintainer: kickedBack.map((v) => ({ finding: v.finding, why: v.verdict.verdict_reason })),
}

const report = await agent(
  `You are the SYNTHESIZER for ShaderBox's nightly audit. You are given the full structured output of a multi-phase audit fleet (cluster reads, recurring-mistake classes, emergent insights, harness findings, and the verified auto-apply vs kicked-back fix lists). Produce ONE cohesive markdown report for the solo maintainer to read in the morning.

The maintainer's actual asks, in their words:
1. Mine all feature artifacts for useful knowledge + emergent insights.
2. Find weak spots / recurring mistakes the project repeats.
3. Audit the dev-flow harness for consistency / everything in its place.
And: the certain fixes get applied automatically (a separate step handles that); everything else is presented for morning review.

Write the report with these sections, in this order:

# ShaderBox nightly audit — 2026-06-13

## TL;DR
5-8 bullets: the sharpest takeaways across all three asks. Lead with the most actionable.

## Emergent insights & cross-feature threads
The non-obvious cross-corpus findings + narrative threads. This is the part the maintainer most wanted — make it genuinely insightful, not a feature recap. Each insight: the insight, what it's derived from, and the so-what.

## Recurring mistakes
The named mistake classes, each with its pattern, instances (cite features), root-cause hypothesis, and cheapest guardrail. Rank by how often / how costly. Be honest and specific — this is a self-critique the maintainer asked for.

## Knowledge worth lifting
A prioritized table of durable knowledge currently buried in feature specs that should move to a higher-altitude home (conventions.md / a skill / dev_flow). Columns: knowledge | currently in | proposed home | lift-full or pointer. Group the strongest 8-15; do not dump everything.

## Harness consistency
### Auto-applied this run
A table of the fixes that were verified and applied automatically (file | what changed). State plainly these are DONE.
### For your review
The remaining harness findings (the kicked-back certain-fixes + all likely/judgment-call findings), grouped by severity, each with location (grep-able phrase) + proposed fix. These are NOT applied.

## Strengths worth keeping
A short, honest list of what this harness/process does well (so the maintainer doesn't "fix" what isn't broken).

Rules: cite features/files for every claim. Quote grep-able phrases for any harness finding. Be concrete; no filler. Use tables where the content is parallel. This is a long report and that is fine — but every line earns its place. Output ONLY the markdown, starting with the H1.

STRUCTURED AUDIT DATA:
${JSON.stringify(payload, null, 1)}`,
  { label: 'synthesize:report', phase: 'Synthesize' },
)

return {
  report_markdown: report,
  auto_apply_fixes: payload.auto_apply_fixes,
  kicked_back: payload.kicked_back_to_maintainer,
  raw: {
    reads_count: reads.length,
    recurring_count: recurring?.recurring_mistakes?.length ?? 0,
    emergent_count: emergent?.emergent_insights?.length ?? 0,
    harness_findings_count: allHarnessFindings.length,
    auto_apply_count: payload.auto_apply_fixes.length,
  },
  full_structured: payload,
}
