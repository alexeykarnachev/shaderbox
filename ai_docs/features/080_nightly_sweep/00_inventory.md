# 080 W-0 — the dead-symbol inventory

Measured at `e7bf863` on `dev`. This wave's output: what the deletion wave acts on and, just
as importantly, what it does NOT, so a later pass does not re-derive it. **No code changed in
this wave.**

**Headline: the repo is almost entirely free of dead code, and the little there is was left by
the previous sweep.** Across every symbol kind Python has and every directory, four symbols are
dead. One of them is the row an earlier sweep recorded as removed at a commit that never
touched it.

## Method

Reference counts come from an AST enumeration of every declaration, cross-counted against a
word index of every `.py` file in `shaderbox/`, `scripts/`, `dogfood/` **and `tests/`**, plus
`Makefile`, `build.sh`, `pyproject.toml`, `.pre-commit-config.yaml` and the CI workflow. A
symbol with a single occurrence is its own definition and nothing else.

Two properties of that index are load-bearing, and each fixes a way this scan can lie:

- **`tests/` counts.** A scan omitting the test tree calls live symbols dead. During this
  sweep's presence scan one agent did exactly that and reported `pass_graph.DTYPES` dead, where
  three test modules import and assert on it.
- **`ai_docs/` does NOT count.** A first run of this scan indexed the docs too, and
  `bot_token_present` — dead in code, named in three markdown files — scored four references
  and was filtered out of the candidate list. A symbol mentioned only in prose is dead code
  with a paper trail, which is precisely what this inventory exists to find.

Names reachable by string from `shaderbox/resources/**` (`.glsl`, `.json`, `.txt`) are tallied
separately and reported beside each candidate rather than folded into the count, so dynamic
reach is visible instead of silently keeping a symbol alive.

Tooling used as a candidate generator, not as the verdict:

- `uv run ruff check --select F401,F811,F841 .` → all checks passed. Unused imports and locals
  are already absent, so everything below is what a linter cannot see.
- `uvx vulture shaderbox scripts dogfood --min-confidence 0` → 117 lines, of which the great
  majority are the predicted false-positive classes: pydantic `model_config`, imgui style
  attributes assigned by name, `editor/ffi.py`, the dogfood harness's hand-driven API.

## Dead — act on these

| symbol | kind | file | tier | evidence |
|---|---|---|---|---|
| `TelegramExporter.bot_token_present` | method | `shaderbox/exporters/telegram.py:262` | SAFE | one code occurrence repo-wide. Its siblings (`bot_username_value`, `list_packs`, `create_pack`, `select_pack`) are all reached from `copilot/backend.py`; the capability layer asks `is_connected()` instead. Not on the `Exporter` ABC. Recorded as REMOVED by the previous sweep at a commit whose diff for this file is two unrelated deletions. |
| `TOOLS_BLOCK` | module constant | `shaderbox/copilot/context_breakdown.py:25` | SAFE | one code occurrence. Its sibling `EXCHANGE_BLOCK` one line above is passed to `_measure` at line 117 and asserted on in `tests/test_context_breakdown.py`. The `tools=` part is measured through `ContextBreakdown`'s own `tools_chars` / `tools_est_tokens` / `tools_text` instead, so the constant has no job left. |
| `COLOR.SYN_PREPROC` | dataclass field | `shaderbox/theme.py:190` | SAFE | a syntax color from before the editor was vendored. `editor_palette()` maps nine syntax slots and this is not among them; the intel color table does not name it; no `getattr` on `COLOR` exists anywhere in the repo, so there is no dynamic reach. |
| `experiment_dir` | module function | `dogfood/report/log.py:348` | SAFE | one code occurrence. Its siblings `load_experiment` / `load_store` in the same file are used from `scripts/dogfood/drive.py`, `dogfood/report/station.py`, `dogfood/report/build.py` and the tests. The one site that needs `store / experiment_id` (`dogfood/report/station.py:229`) inlines the expression. |

## KEPT, deliberately — do not re-report these

| symbol | why it stays |
|---|---|
| `editor/ffi.py`'s `select_line`, `get_scroll_max`, `get_atlas_distance_range`, `PRIM_STRIDE`, and the flag members `Kind.MISSING_GLYPH`, `ViewFlag.SHOW_SPACES` / `SHOW_TABS` / `HIGHLIGHT_SEARCH`, `ChromeFlag.RELATIVE_NUMBERS` / `STATUS_LINE` / `STATUS_SHOWS_MODE` / `STATUS_SHOWS_RULER` | a ctypes MIRROR of a vendored C ABI. Completeness against the ABI is the point, and deleting an unused flag member is SILENT — only the signature table is gate-compared. Two tests hold the binding to the ABI: one compares `_SIG`'s keys to `nm -D` on the vendored `.so`, the other compares every restype and argtype to the vendored probe. |
| `_reject_unnamed_pass` (`pass_graph.py:136`), `_id_validator` (`ui_models.py:353`) | pydantic `@model_validator(mode="after")`. Pydantic calls them; no code names them. Decorator verified at each site. |
| `_restore_copilot_config` (`tests/conftest.py:74`) | `@pytest.fixture(autouse=True)`. Collected by pytest, never called by name, and its comment records the cross-test config bleed it exists to stop. |
| `_NAV_ONLY_FOCUSABLE` (`tests/test_region_system_is_gone.py:76`) | its own comment says it is named on purpose so a later reader does not restore those widgets without re-running the measurement. Deleting it would delete the warning. |
| `popups/settings.py`'s `TELEGRAM_TOKEN`, `YOUTUBE_CLIENT` | matched by string literal from `exporters/telegram.py:344` and `exporters/youtube.py:266`, each with a comment recording the coupling on purpose. |
| `PassGraph.version`, `ConversationStore.version` | round-trip-only by design; the latter is written into every saved `conversation.json` on disk. |
| `ScriptError.uniform_name`, `PromptBlock.name`, `CompletionProvider.name` | kept by the previous sweep with the reasons written down: the first carries a diagnostic and is passed positionally at many sites; the other two are the identity column of a hand-written registry, and the conventions specify a prompt tier as a *named* block, so that field IS the name. |
| the seven unimported modules in `scripts/` | standalone entry points run by hand, by the `Makefile`, or by CI: `smoke.py`, `gen_glyphs.py`, `gen_glsl_docs.py`, `token_probe.py`, `agent_hub/generate.py`, `dogfood/drive.py`, `dogfood/verify_script_engine.py`. Nothing imports an entry point. |
| `AccentName`, `DensityName`, `RoundingName` (`theme.py:40-42`) | not dead — they type `apply_theme`'s parameters. They fall with the speculative-generality wave or not at all; they are not a separate deletion. |
| `pass_graph.DTYPES` | ALIVE. Zero references inside `shaderbox/`, imported and asserted on by three test modules. Listed here because a scan that omits `tests/` re-finds it every time. |

## The previous sweep's "Removed" table, re-verified entry by entry

Its ten rows are nine removals plus one pointer into its own KEPT section. Eight of the nine
landed; one did not, and its commit message says it did.

| row | verdict |
|---|---|
| `DEFAULT_IMAGE_FILE_PATH` | absent |
| `TelegramExporter.bot_token_present` | **PRESENT** — `exporters/telegram.py:262`, uncalled |
| `ProjectPaths.copilot_dir` | absent as a field; the local in `for_root` remains, as that sweep said it would |
| `ExporterStatus.in_flight` | absent (`_RenderState.in_flight` and `CopilotState.in_flight` are the live, different ones) |
| `_LinkEvent.message` | absent |
| `LibFunctionBody.signature` | absent (`LibCatalogEntry.signature` is the different, live field) |
| `PublishResult.kind` | absent |
| `AgentToolCard.result` | absent |
| `AgentToolCard.display` | absent |
| `ScriptError…` | not a removal — a pointer into that sweep's KEPT section |

**The lesson, and it changes how this sweep reports.** A commit message is not evidence that a
symbol is gone. Every deletion's done-condition here is a grep over the landed tree.

## Coverage claims

- `scanned: every module-level function and class, and every class method, across shaderbox/
  and every subpackage, scripts/, dogfood/ and tests/, by AST declaration cross-counted against
  a word index of all four trees plus Makefile, build.sh, pyproject.toml, the pre-commit config
  and the CI workflow; not scanned: nested and local functions, lambdas, attributes assigned
  dynamically at runtime.`
- `scanned: 185 enum members individually and 854 dataclass and pydantic fields individually,
  across shaderbox/, scripts/ and dogfood/; not scanned: fields of classes declared inside
  function bodies.`
- `scanned: 461 module-level assignments across the same three trees, and 17 type aliases each
  reference-counted by file; not scanned: names bound only inside conditionals.`
- `scanned: every private single-underscore module-level symbol across all four trees; not
  scanned: private symbols declared inside classes (covered by the method pass).`
- `scanned: the whole-module import graph by AST over every tracked .py file, so
  "from pkg import a, b, c" is resolved rather than missed; not scanned: scripts/dogfood/runs/
  (recorded run artifacts, data rather than source) and the vendored ABI probe.`
- `scanned: string reachability from shaderbox/resources/**.glsl, .json and .txt, tallied per
  candidate; not scanned: names assembled at runtime from fragments, which no static pass sees.`
