# 046 — knowledge-base refactor wave (nightly-audit knowledge-lift)

A docs/process-only feature: take the durable knowledge surfaced by the 2026-06-13 nightly ultracode
audit and file it into its canonical homes, so the next cold agent inherits it through the cold-start
chain instead of through a one-off report it would never read. NO behavior change, NO code change
beyond possibly adding one new skill file. Maintainer-owned: the audit pre-applied the 6 mechanical
doc fixes + the 5 small B-block fixes (see roadmap); THIS spec is the judgment-call remainder the
maintainer reviews and lands deliberately.

**Source of the full findings:** the 2026-06-13 nightly audit — `ai_docs/_audit_2026-06-13/report.md`
(the raw per-cluster output + the synthesizer's report) and `audit_workflow.js` (the re-runnable
workflow). Both were deleted once this wave filed every item (see ## Status); retrieve them from git
history at commit `4b0f379` if you need the verbatim instances + grep-able phrases the rows below cite.

This spec was the WORK LIST and is now the pointer-of-record: each item's durable rule lives at its
canonical home (## Status maps item → home), not here (one-canonical-home).

---

## Scope split

- **Already applied by the audit run** (DONE on `dev`, see the two audit commits): 6 mechanical doc
  fixes (stale path `text_render.py`→`sanitize.py`; 027 resolved remainders; test count 467/475→502;
  `gl_type` allowlist resync; `_to_wire`→`_to_wire_message`; dev_flow popup-model) + 5 B-block fixes
  (roadmap 005 + 028 → `done`; dev_flow `reset_script`→`reset`; banner trim ≤200; the two copilot
  honesty deferrals merged into one rolling todo entry).
- **This spec (046) — the remainder, maintainer-owned:** two HIGH live-doc rewrites, the
  knowledge-lift table (the bulk of the work), the LOW polish list, and the decision on whether to
  build the `copilot/LLM-agent-design` skill.

---

## A. HIGH — live-index docs that mislead a cold reader (report ## Harness consistency > HIGH)

1. **`conventions.md` "one `TextEditor` per node" design decision is now FALSE — its own revisit
   trigger fired.** The bullet says "One `TextEditor` per node (+ a parallel dirty-baseline dict)"
   and ends "Revisit if multi-file-per-node editing lands." Feature 045 landed exactly that:
   `self.editor_tabs: list[EditorTab]` + `self.active_tab_index` over a path-keyed
   `self.editor_sessions: dict[Path, EditorSession]`; `_explicit_editor_path` is gone.
   **Fix:** rewrite the bullet to the 045 reality (ordered `editor_tabs` + `active_tab_index`,
   path-keyed `editor_sessions`, one `TextEditor` per opened FILE — node shader / node-brain
   `script.py` / per-uniform `u_*.py` / lib files as closable tabs; `flush_current_editor()` flushes
   the active tab; mtime watcher re-syncs, disk wins) and pick a NEW revisit trigger. Staleness is
   certain; wording + new trigger are the maintainer's call.

2. **`roadmap.md` feature row 020 has become a ~600-word / ~30-sentence worklog.** It narrates the
   feature slice by slice (the review-round trail "Roadmap rows index; feature specs narrate" assigns
   to `ai_docs/features/020_copilot_agent/`). **Fix:** collapse to ONE dense brief — current status +
   the partial gap (what-it-does; all waves landed + SHIPPED in v0.13.0; still deferred: lazy
   tool-catalogue D5, delete_lib_file, semantic editing, bind_media/undo_edit per todo.md; spec link).
   Judgment call on density.

---

## B. The knowledge-lift table — durable knowledge to move to its canonical home

(report ## Knowledge worth lifting — that table is the authority; this is the same list as the work
plan. Default action per row: write the durable rule at the proposed home; if it's already partly
filed, extend the existing bullet rather than add a new one.)

### B1. The big one — decide first: a `copilot/LLM-agent-design` skill
The single highest-leverage lift (report TL;DR + "Every LLM-facing lesson is a corollary of one
actor-model"). LLM-agent design knowledge is scattered across ~12 specs (020.04/11/13/16/19/28/29,
033, 036/038/039, dogfood 035/037) with no home. **Decision for the maintainer:** create one skill
that states the actor model ONCE — *the model reliably (a) copies text verbatim and (b) is blind to
anything outside its token stream* — and hangs every corollary off it:
- prefix-cache strata (least-volatile→most-volatile; never volatile state in system; frozen block
  write-only; tools serialized sorted; a stale verbatim copy is worse than none) — 020.04/11/16/28/29
- fact/hint/guard taxonomy + facts-on-the-existing-channel over new opt-in tools + tool-count
  discipline + the better-model babysitter test + the trace-patch stopping rule — 020.29, 033/039, df 037
- content-addressed edit contract (address = content; loud no-match/multi-match by construction;
  token-stream match but don't advertise invariance; detect degenerate inputs + truthful error) —
  020.13, 036/038/039, ui 034, df 035/037
- secret on its OWN typed channel end-to-end (dedicated `GateResponse.secret`; separate
  `execute(secret=)`; redacted echo; never persist the buffer) — 020.19
- agent cost measurement (billed=SUM across iterations; fullness=iteration-0; cache-hit dominates
  re-run cost → never gate on a cold sample; a viz must answer ONE named question) — 025/028, report
  "Token-cost was redesigned…"
If the skill is too heavy, the fallback is `conventions.md ## Design decisions` sub-sections + a
pointer from the `claude-api` skill's caching section.

### B2. → `conventions.md ## Design decisions` (lift-full or extend existing bullet)
- **Structural impossibility over guard-piles** (the deepest emergent law, report "The one design
  law…"): *"if you're adding a SECOND wave of guards to second-guess what an actor MEANT, the
  contract is unsound; redesign so the safe outcome is structural, then delete the guards."* Make it
  the first question in any spec review whose decision section is mostly validation logic. Canonical
  case: 036→038→039.
- **Speculative-machinery rule, disambiguated** (report "Two opposite rules…"): the test is *"is
  REMOVING it churn?"* — surface you must teach/maintain (ABC method, model-visible enum, curated
  namespace where every omission is a NameError) gets cut; inert latent code with zero teach/maintain
  cost + a named near-future consumer stays dormant. Re-derived from scratch in 002/010/041.
- **Latch-needs-`reopen()`** (already partly filed by 017's latch child): a blocking worker↔main
  round-trip is ONE primitive (CopilotBridge / GateChannel are mirror directions); a latched-shutdown
  channel MUST expose `reopen()`; teardown = `cancel_all()` before `join(timeout)` + non-daemon +
  abandon-survivors. — 020.11/17, foundations 001
- **Worker/thread-affinity bundle:** GL objects enforced by METHOD affinity not import; reuse the
  mtime watcher to marshal GL (file-write free-lunch); persist a split at the quiescent between-turns
  boundary (no lock); cross-thread mid-op reactions = injected callbacks not return values; classify
  ctor deps by volatility (getters for project-dependent, freeze only shipped values). — 001/020.01/022/023/025
- **Persistence-evolution posture** (extend the `extra='forbid'` bullet): additive → defaulted +
  loose-typed optional + fail-soft load + provenance-only bump; intentional reset → actively drop
  fields so `extra='forbid'` REJECTS; `extra='forbid'` is the verifiability primitive; unknown
  enum/role → plain-text fallback. — 001, 020.21/28, ui 028
- **Parallel-name-keyed-dict drift smell + remedy** (lift facts ONTO the entity + ONE invariant
  test); 031 confirmed 10 instances repo-wide. — 029/031
- **Snapshot/restore correctness:** disk ≠ live state — serialize the LIVE object, restore by
  reload-and-replace across EVERY live surface; a serialize routine that mutates what it serializes
  (rebinding `source.path`) is latent corruption (add a no-rebind/pure mode); capture is best-effort
  (never fails the guarded op). — 020.30 (on-disk cut already filed by 021)
- **Cross-cutting guarantee → enforce at the single FUNNEL, not per-caller** (pointer from CLAUDE.md
  blast-radius rule; the per-caller bracket stage is a known dead-end — 041 did it 3×). — 041, 020.03
- **Stateless-rebuild over stateful-daemon:** check whether the expensive state is already cheaply
  serialized before building a daemon; a MAJORITY of design agents agreeing is NOT evidence (one
  code-grounded contrarian overturned three). — dogfood 027

### B3. → `conventions.md ## Known quirks` (extend the device-specific MESA entry)
- **Mesa v3d footguns:** final GPU code compiled at FIRST DRAW (≤13 strategies) so prefer data TABLES
  over branchy GLSL; MESA/GLSL/`SHADERBOX_DATA_DIR` env must be a MODULE-TOP side-effect before first
  import + fail-loud (a late assignment reads as "engine broken"). — ui 032, dogfood 026/027

### B4. → `dev_flow.md ## Feature flow step 7` (next to the already-filed UX-gap rule)
- **Verification design:** each manual-check step fails for EXACTLY ONE reason (measured asserts +
  named canaries + disk-readback over "feels responsive"); name the FALSIFIER per invariant (reject a
  test that passes under the bug); a UX gap at manual-check is a FAIL; **an unwired/dead/commented
  mechanism counts as ABSENT — name the reader + the wired test.** This last clause is the cheapest
  guardrail for recurring-mistake #1 (below) and is mechanically greppable. — 001/004/006, 041/044

---

## C. Recurring-mistake classes (report ## Recurring mistakes — 10 classes, ranked freq×cost)

These are a self-critique, not work items per se — but each carries a "cheapest guardrail" that may
graduate into a dev_flow checklist item. The report holds the full instance lists + root causes.
Ranked:

1. **Spec'd safety/guarantee ships as a no-op (scaffolded-but-unwired)** — most expensive, several
   BLOCKER-tier (001, 020.12 `max_edit_retries`, 020.15 editor-lock flag, 022 per-turn save, 033).
   Guardrail → B4 ("name the reader + the wired test"). **This is the #1 guardrail to graduate.**
2. **Stale/intended state trusted instead of ground-truth state** (009/010/012/029/030/044).
3. **Guessing from training-data recall instead of verifying the installed dependency** (004/018/019/020.23/026/031/041).
4. **Speculative abstraction built before the 2nd real consumer, then torn out** (002/001/003/009/020.02-03/041/045/020.14).
5. **Guard-pile retrofit onto an unsound contract** (020.14→036→038→039, 034/038, 024, 009/022). → the structural-impossibility rule (B2).
6. **Prototyped/specced against the lucky case** (012/008/013/031/034/041) — ban silent swallow.
7. **Cross-cutting invariant enforced per-call-site, so a caller is missed** (041/028/016/018/029/031). → the funnel rule (B2).
8. **Locked-decision-vs-shipped-reality drift (append-rot)** (013/016/012/020.11/029) — reconcile reversed locked sections in the same wave (sanitize/cold-context checklist).
9. **Trace-driven overfit guard against a transient model/transport quirk** (029/037/033/035) → the better-model test + stopping rule (B1).
10. **Review anchored to the wrong artifact (stale proto / not-yet-written code)** (040/045/044/041) — already covered by `review-agent-loop`; reinforce by pinning the exact artifact + "this code does not exist yet" boundary in reviewer prompts.

---

## D. LOW — roadmap/todo polish (report ## Harness consistency > LOW)

- **043 named NEXT in banner/todo but has no `pending` roadmap row** — the "first pending row is next"
  header mechanism is dead (zero pending rows). Either add a `| 043 | copilot_write_behavior |
  pending | … |` row or soften the header to allow a banner-only next feature. (046 itself is now
  referenced from the banner as OPEN — same question applies.)
- **A cluster of rows exceed "one dense brief"** — 045/027/041/044/025 (vs median 33-65w); 025
  enumerates its per-commit C1…C4 trail the rule assigns to the spec. Trim per-commit/per-slice
  mechanics on a taste call.

---

## E. Strengths worth KEEPING (report ## Strengths worth keeping — don't "fix" these)

- **Build-then-tear-out is overwhelmingly cheap convergent exploration, not waste** — teardowns
  almost always happen PRE-ship (zero migration cost), each recorded as a conscious decision with the
  distinguishing reason. Don't mistake the churn for thrash.
- **Filing fixes at the right altitude is a genuine disciplined strength** — point bugs converted to
  shared primitives at the root, but ONLY when a real missing-half proves the class; the project
  explicitly REFUSES to generalize on speculation (020.03/023 declined `uniform_helpers.py` after a
  grep showed zero callers). The rare correct version of the legitimate large diff.

---

## Status

DONE (2026-06-14). The work list above is filed; each item now lives at its canonical home, and this
spec is the pointer of record (one-canonical-home — the durable rule lives at its home, not here).

**Where each item landed:**
- **A1** (045 tabbed-editor) → `conventions.md ## Design decisions`, the "Inline editor state lives on
  `App`; one `TextEditor` per opened FILE" bullet (rewritten to the `editor_tabs`/`active_tab_index`
  reality + a new revisit trigger).
- **A2** (row 020 worklog) → `roadmap.md` row 020 collapsed to one dense brief.
- **B1** (the skill) → `.claude/skills/copilot-llm-agent-design/SKILL.md` (decision: a dedicated skill,
  not the conventions-fallback). The audit's "pointer from the `claude-api` skill's caching section" is
  a NON-ACTION: `claude-api` is a Claude Code BUILT-IN skill (not a repo/user-owned file), so it can't be
  edited — the new skill is reachable on its own via its description triggers.
- **B2** → `conventions.md ## Design decisions`: four emergent LAWS lead the section
  (structural-impossibility, speculative-machinery, funnel-not-per-caller, stateless-rebuild); the
  latch bullet extended with the one-primitive + teardown recipe; four concrete patterns added
  (worker/thread-affinity, persistence posture, name-keyed-dict drift, snapshot/restore correctness).
- **B3** → `conventions.md ## Known quirks`: the v3d-codegen-at-first-draw reason + the
  `MESA_*`/`SHADERBOX_DATA_DIR` module-top-before-import rule.
- **B4** → `dev_flow.md` step 7: the verification-design rule (one-reason steps + named falsifier +
  "unwired = absent: name the reader + the wired test").
- **C** (10 recurring-mistake classes) — self-critique, not separately filed; the cheapest guardrails
  that graduate are B4 (#1, unwired=absent), B2 structural-impossibility (#5), B2 funnel (#7), and the
  better-model test in the skill (#9). The full instance lists stay in this spec + the report's git
  history.
- **D** → `roadmap.md`: header softened (banner is authoritative "what's next"; a row appears once a
  spec exists); long rows 045/027/041/044/025 trimmed; 028/005 already flipped `partial`→`done` by the
  audit's B-block commit (`83f9439`); the 043 pending-row question resolved by the header soften.

The `_audit_2026-06-13/` report + workflow are deleted in this commit (durable source until filed,
then resolved-entries-get-deleted; git history retains them).
