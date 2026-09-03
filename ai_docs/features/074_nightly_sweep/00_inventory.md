# 074 W-0 — the dead-symbol inventory

Measured at `76cbeea` on `dev`. This is the wave's output: what W-1 acted on and,
just as importantly, what it did NOT, so a later pass does not re-derive it.

**Headline: the repo is almost entirely free of dead code.** Across every symbol
kind and every directory, the only finds were ONE dead constant, ONE dead
forwarder method, and a cluster of write-only dataclass FIELDS. Zero dead
functions, classes, modules, private helpers, enum members or type aliases.

## Tooling

- `uv run ruff check --select F401,F811,F841 .` → `All checks passed!`
  (unused imports and locals are already absent).
- `uv run --with vulture vulture shaderbox scripts tests --min-confidence 60`
  → 106 lines, of which the overwhelming majority are the false-positive classes
  the spec predicted: pydantic `model_config`, `app.py`'s dynamically-read imgui
  style attributes, the dogfood harness's hand-driven API, and `editor/ffi.py`.
- Tools cannot see write-only fields, dynamic reach, or per-member enum use, so
  the categories below were enumerated from the language's own constructs.

## Removed (W-1)

| symbol | kind | file | tier | why it was dead |
|---|---|---|---|---|
| `DEFAULT_IMAGE_FILE_PATH` | module constant | `constants.py` | SAFE | 072's D2 took the default-photo marker out of every reader; `is_default_image` is gone entirely. One occurrence repo-wide (its own definition). |
| `TelegramExporter.bot_token_present` | method | `exporters/telegram.py` | SAFE | one occurrence repo-wide. Its siblings are all reached from `copilot/backend.py`; the capability layer uses `is_connected()` instead. Not on the `Exporter` ABC. |
| `ProjectPaths.copilot_dir` | dataclass field | `paths.py` | SAFE | written at construction, never read. The local `copilot_dir` in `for_root` stays — it derives the two sibling path fields. |
| `ScriptError`… — see KEPT below | | | | |
| `ExporterStatus.in_flight` | dataclass field | `exporters/base.py` | SAFE | every consumer reads `last_progress` / `auth_state`. Distinct from `_RenderState.in_flight`, which is heavily used — that one stays. |
| `_LinkEvent.message` | dataclass field | `exporters/telegram.py` | SAFE | never passed at its one construction site, never read in the `isinstance` branch. |
| `LibFunctionBody.signature` | dataclass field | `copilot/capabilities.py` | SAFE | `read_lib`'s consumer reads `name`/`lib_address`/`body`; `body` already contains the signature. (`LibCatalogEntry.signature` is a DIFFERENT field and is live.) |
| `PublishResult.kind` | dataclass field | `copilot/capabilities.py` | CAREFUL | set at 8 sites, read at none. Removing it made `_copilot_publish`'s own `kind` PARAMETER dead — the cascade was followed. |
| `AgentToolCard.result` | dataclass field | `copilot/agent.py` | CAREFUL | redundant. `msg` reaches the LLM history via `_tool_message` and the trace via the event block; the card's copy is read nowhere. |
| `AgentToolCard.display` | dataclass field | `copilot/agent.py` | CAREFUL | `session.py:216` documents the design that superseded it: a step folds into the snippet's square bar, and only widgets keep a visible line. |

## KEPT, deliberately — do not re-report these

| symbol | why it stays |
|---|---|
| `ScriptError.uniform_name` | data-carrying and passed POSITIONALLY at ~20 sites. Nothing reads it back only because the error maps are keyed by tuples that already carry the name. Removing it costs 20 call sites and loses the diagnostic on the error object. |
| `PromptBlock.name` | `conventions.md` specifies a prompt tier as "a **named** block at its volatility rank". The field is that name. No code branches on it; deleting it would contradict a documented design rule and make the next tier anonymous. |
| `CompletionProvider.name` | same shape: the identity column of a hand-written 3-row registry. Without it the table is three unlabelled tuples. |
| everything in `editor/ffi.py` | the ctypes mirror of a vendored C ABI. A scan re-finds 14 "dead" enum members here (`Kind.BACKGROUND`, `ViewFlag.SHOW_TABS`, `Language.PYTHON`, the `ChromeFlag` status members, …) — exactly as `conventions.md` predicts a sweep will. Completeness against the ABI is the point, and deleting a flag member is SILENT (only the signature table is gate-compared). |
| `PassGraph.version` | round-trip-only by design (`conventions.md`). |
| `ConversationStore.version` | written into every saved `conversation.json` (`"version": 10` on disk) — reached, just not branched on. |
| `GraphError.message`, `LookupPopup.word` | read from tests only, which is still reached. |
| the dogfood harness's ~8 flagged methods | an interactive API driven by hand. |
| pydantic `model_config`, `@model_validator` / `@field_validator` methods | pydantic calls them itself. `_reject_unnamed_pass`, `_id_validator`, `_reset_out_of_range_values` are all validators — decorator verified. |

## Reported, NOT acted on (out of scope by the spec)

- **`shaderbox/resources/textures/default.jpeg`** (178 KB) is now referenced by
  nothing in the package — `DEFAULT_IMAGE_FILE_PATH` was its only reader and 072
  removed the mechanism. **Deleting a resource file is out of scope for this
  sweep**, so it stays. For the maintainer: it is a safe delete whenever the
  Radiance-Cascades-era docs describing a default-image mechanism are reconciled.
- **`AgentToolCard.display`'s producer still exists.** `copilot/tools/shader.py:312`
  computes `payload["display"]` (and `tests/test_edit_messages.py:237` asserts on
  it), but nothing consumed the card field. That is a BEHAVIOUR question — did the
  terse chat summary regress, or was it superseded on purpose? — and behaviour
  goes to the maintainer, not into an unattended commit. The payload key and its
  test are untouched.

## Coverage claims

- `scanned: module-level functions and classes (802 definitions across 125 files),
  via ripgrep word-count plus an independent AST reference walk, across shaderbox/
  and every subpackage; not scanned: methods, nested/closure functions.`
- `scanned: 760 class methods (558 unique names), AST-separated from 85 same-indent
  closures, across shaderbox/; not scanned: editor/ffi.py (deliberate ABI mirror),
  resources/editor/, glsl_docs.py, glyph_tables.py.`
- `scanned: every enum member of every enum class individually, and every type alias
  individually, across shaderbox/ and scripts/; not scanned: functions, classes,
  constants (covered by the other passes).`
- `scanned: 298 module-level constants and ~460 pydantic/dataclass fields across ~90
  classes, over shaderbox/ and every subpackage, each field also checked against
  the JSON keys in projects/; not scanned: editor/ffi.py flags, PassGraph.version,
  model_config (all deliberate).`
- `scanned: the whole-module import graph by AST across all 252 tracked .py files in
  shaderbox/, scripts/, tests/, plus module-level symbols in scripts/ and tests/;
  not scanned: methods inside shaderbox/ classes (covered by the method pass),
  scripts/dogfood/runs/ (gitignored sandbox output, not source).`
- `scanned: ~610 private (single-underscore) module functions, methods and module
  variables across every subpackage of shaderbox/; not scanned: editor/ffi.py,
  private CLASSES — closed separately by the main session, all 51 referenced.`

**Note on method:** the module-graph scan's first pass used a literal grep recipe
that under-detects `from pkg import a, b, c`; it demonstrated a real miss
(`popups/lib_picker/preview.py` reading as unimported) and switched to AST
parsing. Recorded because a scan that quietly narrows its own domain is the
failure this inventory exists to prevent.
