# 029 — Tool labels + gate prompts as first-class tool properties

> **STATUS: DONE.** Research via an 11-agent brainstorm/judge workflow (4 maps -> 4 competing
> designs -> 3-judge panel); the converged design implemented in one wave.

## Problem (the honest version)

Two per-tool facts lived in parallel name-keyed dicts instead of on the tool entity:

- `copilot/state.py::_TOOL_VERBS` — the past-tense chat label (hover breakdown + tool card).
- `agent.py::_GATE_PROMPTS` — the confirm-card prompt template for always-gated tools (the
  orphaned sibling: `gate_policy`/`gate_kind`/`secret_field` already lived on `ToolDefinition`).

The SHIPPED defect: `ToolRegistry.status_for` was a stub returning the raw tool name, so the
live status pill showed `edit_shader` instead of a human phrase. (The spec's original "it
drifts" motivation was prospective — git shows `_TOOL_VERBS` was born complete and never
drifted; the real costs were the stub + the add-a-tool tax of ~5 name-keyed side structures.)

## Design (locked + implemented)

- `ToolDefinition` gains **`label_live`** ("Editing shader" — the in-flight status pill) +
  **`label_done`** ("Edited shader" — tool card + snippet hover), both REQUIRED, and
  **`gate_prompt: GatePrompt | None`** (the confirm-card template; `None` => `build_gate`'s
  generic fallback, kept for unknown names + future BULK-gated tools). Labels are display-only:
  `name` stays the identity key on every durable surface (trace, StepRecord, ledger).
- TWO tenses, not one: `status_for` fires BEFORE the gate, so a single past-tense label would
  claim completion of a declinable action; "Read"/"Set" don't derive mechanically from a
  participle. Decided against the original spec's one-field lean — ratified by all 4 designs.
- The dataclass is `kw_only=True` (all literals were already keyword-form; kills the
  required-before-default field-ordering dance for future fields).
- Registry: `status_for` un-stubbed to `label_live` — **signature `(name, args)` kept**: the
  `args` param is the designed arg-aware-phrasing seam (020/11 §2.3), unused until that lands
  (if it never does, remove the param rather than carry it forever); new `label_for(name)`
  (raw-name fallback — persisted StepRecords may carry renamed/removed tools); new
  `definitions()`.
- DELETED: `_TOOL_VERBS` + `tool_label()` (state.py), `_GATE_PROMPTS` (agent.py). Both label
  and prompt strings were lifted byte-identical, so no user-visible card text changed except
  the live pill (raw name -> phrase — the point).
- Same wave: `session.py`'s `_tool_card_outcome` grep name-check converted to the payload
  shape-key (`"hits" in payload`, matching the adjacent `"errors"` idiom). The `delete_node`
  Recover affordance stays name-keyed — n=1, a `recoverable` trait field would be speculation.
- Invariant test `tests/test_tool_registry.py::test_every_tool_fully_carded`: registry names ==
  analyzer `CANONICAL_TOOLS`, both labels non-empty and != name, `gate_prompt` present iff
  `gate_policy is ALWAYS`. One test closes three drift classes (label, gate prompt, analyzer
  coverage list).

## Out of scope (explicit YAGNI — rejected by the design panel)

- `__post_init__` runtime validation — duplicates the invariant test; turns a missing prompt
  from a red CI test into an app-boot crash.
- A `brief`/`summary` field — zero consumers until the lazy catalogue (lever 2) lands; add then
  as a defaulted field (`d.brief or first_sentence(d.description)` resolver).
- `@tool` decorator / builder DSL — the pyright-checked kwargs literal IS the drift guard.
- Arg-aware status templates, `outcome_renderer`/`recoverable` trait fields, registry-generated
  prompt.py text, label-on-StepRecord (forces a persistence migration and freezes labels into
  old files vs free retro-renames at draw time), any lazy-catalogue implementation.
- Generating the dogfood analyzer's `CANONICAL_TOOLS` from the registry — the offline analyzer
  stays decoupled; the invariant test's set-equality pin is the right altitude.

## Future-fit (verified attach points, zero reshaping)

- **Lever-2 lazy catalogue:** `eager`/`category` + `specs_for`/`eager_specs` already on the
  entity/registry; `definitions()` feeds the compact menu. NOTE: discovery tools
  (`list_tools`/`load_tools`) need registry access from inside a handler, but `build_registry`
  constructs the registry AFTER the builders run — the lazy feature needs a two-phase
  registration (noted in the `todo.md` lever-2 entry).
- **Hierarchical two-stage loading:** a dotted-path convention in `category`
  ("telegram.packs") — a flat string is a degenerate path, no field change.
- **Per-tool stats:** `registry.execute` is the single EXECUTION chokepoint (gate-approved +
  CREDENTIAL included), but declined/handoff attempts bypass it — per-attempt stats key on the
  trace event family (the dogfood analyzer already derives counts/outcomes from trace).
