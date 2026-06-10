# 031 — Parallel-structure sweep (the 029 smell class, repo-wide)

> **STATUS: PENDING — researched + adversarially verified, ready to implement in a fresh session.**
> Provenance: a 20-agent hunt workflow on top of feature 029's exemplar (7 lens-finders →
> dedup/rank synthesis → one adversarial refuter per finding). 10 findings CONFIRMED (each
> re-verified on a fresh read of HEAD `66b2a92`, several by execution), 2 refuted, ~8 dropped
> at triage. The refuted/dropped list is at the bottom — do NOT re-propose those.
> No backward-compat constraints: internal APIs change freely (maintainer's explicit call).

Implementation order = the list order (value-to-blast ranked). Each item is independently
committable; run `make check` + tests per item, `make smoke` after the UI-touching ones (8).

## 1. BUG: two suffix→media-class maps drifted — a .webm video uniform silently kills its node on reload

- `constants.py::SUPPORTED_MEDIA_EXTENSIONS` (`{".png": "Image", ".mp4": "Video"}`) vs
  `constants.py::IMAGE_EXTENSIONS`/`VIDEO_EXTENSIONS` (5+2 suffixes incl. `.webm`). The picker
  (`widgets/uniform.py`) offers ALL extensions and `Video.save` preserves the source suffix, so a
  picked `.webm` lands in `media/`; on reload `core.py::Node.load_from_dir` dispatches via
  `SUPPORTED_MEDIA_EXTENSIONS` + a `globals()` string lookup → KeyError → swallowed by
  `ui_models.py::load_nodes_from_dir`'s skip-log → the node vanishes from the app (dir survives
  on disk). git: both maps born already-inconsistent in one commit.
- **Fix:** delete `SUPPORTED_MEDIA_EXTENSIONS`; one resolver `media_class_for(suffix)` (in
  `media.py`, built from IMAGE/VIDEO_EXTENSIONS, direct class refs — kill the `globals()`
  indirection; `core.py` already imports Image/Video). Both `load_from_dir` and the picker
  dispatch through it; keep an explicit raise for unknown suffixes so corrupt media still fails
  loudly into the skip-log. Invariant test: every `MEDIA_EXTENSIONS` suffix resolves; plus a
  `.webm` save→load round-trip if a GL fixture is cheap.

## 2. `_RESULT_WIDGET_KINDS` frozenset hand-copied in two files, restating `state.py::ResultWidgetKind`

- `copilot/agent.py` and `copilot/persistence.py` each declare
  `frozenset({"open_url", "open_path"})` guarding fail-soft parsers — a third widget kind added
  to the Literal but missing either copy silently drops the widget (parse or reload), no error.
  Also makes `persistence.py`'s `cast(ResultWidgetKind, ...)` structurally sound instead of
  coincidentally sound.
- **Fix:** `RESULT_WIDGET_KINDS = frozenset(get_args(ResultWidgetKind))` next to the Literal in
  `state.py`; both files import it, both locals deleted. Zero behavior change.

## 3. Shipped-template id list hand-maintained in four files (+ a separately-hardcoded starter id)

- Same 3 UUIDs in `app.py::_TEMPLATE_ORDER`, `scripts/smoke.py::_TEMPLATE_IDS`,
  `scripts/dogfood/harness.py::_TEMPLATE_ORDER` (self-confessed `# mirror app.py's list`),
  `tests/conftest.py::_TEMPLATE_IDS`; `app.py::_STARTER_TEMPLATE_ID` is a 5th independent
  literal, and `tests/test_cross_project_tools.py` reaches into app's private symbol. Drift
  cost: a 4th shipped template is silently absent from every smoke/test/dogfood-seeded project.
- **Fix:** hoist `NODE_TEMPLATES_DIR`, `TEMPLATE_ORDER` (with the authored-order rationale
  comment), `STARTER_TEMPLATE_ID = TEMPLATE_ORDER[0]` into `constants.py` (leaf module, no
  cycles); all five sites import. Keep the three seed loops as-is (they differ in gating) — NO
  shared seed helper. harness.py's import joins its existing post-`sys.path` block.

## 4. Dead `node` param on the publish capabilities — the un-deleted half of a recorded decision

- `backend.py::publish_telegram/publish_youtube` take `node: str` and immediately `_ = node`;
  both handler call sites (`tools/publish.py`) pass `""`. `conventions.md ## Design decisions`
  records publish as current-node-only with NO node arg ("the anti-pattern").
- **Fix:** drop the first param from both backend methods, both `capabilities.py` Callable
  fields, both call sites; mechanical stub updates in `tests/_caps.py`,
  `tests/test_turn_summary.py` (+ the check scripts if item 9 hasn't deleted them yet).

## 5. The `template:` address kind never got its first-class home (missing half of `address.py`)

- `address.py` declares itself "the single round-trip parse/build point" but ships only the
  `lib:` trio; `backend.py` hand-rolls `template:` at six sites (three verbatim
  `f"template:{id[:4]}"` builders, two `startswith` predicates, one `removeprefix`).
- **Fix:** mirror the trio: `TEMPLATE_PREFIX`, `is_template_address()`,
  `strip_template_prefix()`, `template_address(full_id)` (truncation lives in the builder per
  backend's own short-handle contract); six mechanical substitutions; update the address.py
  header to cover both kinds.

## 6. Zero-reference dead code: 7 grep-verified orphans

- `ui_models.py::UIMessage` (+ its now-orphaned `COLOR` import; the feature-001 "lock" rationale
  is dead — `ExporterStatus.message` is a plain str at HEAD); `UIAppState.new_node_name` (one
  write in `popups/node_creator.py`, zero reads; `load_and_migrate` drops unknown keys so old
  app_state loads clean); `exporters/registry.py::get_active`/`available_ids`;
  `media.py::Image.from_color`; `copilot/tools/registry.py::describe` (NOT the lever-2 scaffold —
  that deferral names `specs_for`/`eager_specs`); `app.py` `self.app_start_time` (+ the
  then-sole `import time`); `app.py::_COPILOT_SHORT_ID_LEN`/`_COPILOT_FULL_ID_LEN` (backend.py
  owns the live pair).
- **Fix:** delete all seven + orphaned imports. `make smoke` after (node_creator is on the UI
  path).

## 7. The args-model `extra="forbid"` invariant is hand-copied per class, `_EmptyArgs` defined twice, nothing enforces it

- 18 `model_config` lines across the 4 tool modules; `_EmptyArgs` verbatim in both `telegram.py`
  and `youtube.py`. The forbid is load-bearing (without it pydantic silently swallows
  hallucinated arg keys at `registry.execute`) and `test_every_tool_fully_carded` doesn't check
  it.
- **Fix:** `class ToolArgs(BaseModel)` with the forbid + shared `EmptyArgs(ToolArgs)` in
  `tools/base.py`; all args models subclass it, drop the per-class lines, delete both
  `_EmptyArgs`. Extend the invariant test: every definition's
  `model_json_schema().get("additionalProperties") is False` (checks the actual LLM-facing
  contract, also catches a future stray BaseModel subclass).

## 8. `ui_primitives.wrapped_caption` re-implemented twice

- `widgets/copilot_chat.py::_wrapped_colored` (pixel-equivalent, 4 call sites; its "imgui has no
  colored-wrapped text in one call" comment is false in-repo) + an inline re-roll in
  `popups/lib_picker/preview.py`. Violates the all-UI-through-ui_primitives hard rule.
- **Fix:** delete both, call `wrapped_caption(text, color)`. Visual no-op; gives the helper's
  color param its first real consumers. `make smoke`.

## 9. Both `scripts/copilot_*_check.py` are bit-rotted dead — and they're the ONLY home of security-critical checks

- Verified by execution: `copilot_gate_check.py` imports the renamed-away
  `shaderbox.copilot.context`; `copilot_render_check.py`'s `_stub_caps()` misses three
  now-required `CopilotCapabilities` fields (TypeError) and asserts a stale eager-tool count.
  Zero Makefile/CI/dev_flow wiring — that's how they rotted. Their ~75-line caps stubs are
  drifted twins of `tests/_caps.py::minimal_caps`. Checks living NOWHERE else: credential-token
  redaction (`mask_secret` — zero pytest references), declined-gate path +
  no-orphaned-tool_call_id, gate reopen-after-release, RecoverInfo persistence round-trip,
  finish-reason classification.
- **Fix:** port the nowhere-else checks into pytest on `minimal_caps` (redaction test; gate
  decline + reopen; recover-card round-trip into `test_conversation_persistence.py`;
  finish-reason cases into `test_copilot_loop.py`; structural gate/credential invariants into
  `test_tool_registry.py` with counts DERIVED from the registry, never literals). Then delete
  both scripts; update the four `020_copilot_agent/` spec files that cite them, same wave.
  Budget warning: a ported test may fail because the behavior regressed while dark (esp.
  redaction) — that's the point, but it can grow the wave into a production fix.

## 10. `CopilotCapabilities` is a hand-maintained 1:1 mirror of `CopilotBackend`, bound by a name-restating `wiring.py`

- `wiring.py::build_capabilities` is a verbatim N-line bind of identically-named backend methods
  into the frozen dataclass; sole production constructor is `project_session.py`. Adding one
  capability writes the name 3-4 times. In-repo Protocol precedent: `copilot/llm/api.py::LLMClient`.
  The "dataclass gives construction-time totality" defense is empirically dead — the two broken
  check scripts (item 9) prove stub rot anyway; production totality survives statically (pyright
  checks the backend at the one pass site).
- **Fix:** convert `CopilotCapabilities` to a Protocol with **positional-only** (`/`) method
  params (verified with the repo's pyright: named-param protocol methods are NOT satisfied by a
  Callable-field dataclass fake; with `/` both the backend and the fake pass at 0 errors — and
  today's Callable fields are positional-only anyway). Backend conforms structurally;
  `project_session` passes the backend directly; DELETE `wiring.py`. `tests/_caps.py` keeps the
  `minimal_caps(**overrides)` signature (a frozen dataclass of no-op-defaulted Callable fields),
  so the importing test modules don't change. Largest blast of the wave — do it LAST.

## Refuted / dropped at triage (do NOT re-propose)

- **raise-CopilotToolError-everywhere migration** — REFUTED: the conventions bullet bans
  signaling expected failure via a generic *throw*, not via the handler tuple's `ok=False`
  (the tuple IS the primary result channel); most "unprefixed" evidence was false; some
  messages (partial-success token-set) can't carry an `error:` prefix.
- **Delete GatePolicy.BULK + GateRequest.options scaffold** — REFUTED: recorded
  deliberate trigger-gated deferral in `020_copilot_agent/17_gate_ui.md ## Out of scope`
  ("stays built"), re-ratified by 029's gate_prompt fallback comment.
- Dropped (low value-to-blast / parked-area collisions): command_callbacks completeness assert
  (no drift, no payer); exporter `_UPLOAD_KINDS`/`_BUSY_KINDS` busy-field fold (same-file,
  exporter decomposition is todo-parked); exporter worker/queue plumbing extraction (brushes the
  todo-parked decompose-exporters riskiest seam); `_TYPE_SORT_ORDER` fold (co-located with its
  Literal; both fix shapes carry traps); `is_sampler_uniform` predicate dedup (one-liner, low
  payoff); JSON sidecar fail-soft helper (per-store divergences; but `favorites.py` has a
  genuine `-> "object"` annotation wart worth a one-line fix whenever that file is next
  touched); `danger_menu_item`/tree-rename-input dedup (smallest value, manual-smoke-only
  paths).
