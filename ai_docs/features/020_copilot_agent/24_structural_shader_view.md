# 020·24 — Structural shader view (the read_shader STRUCTURE block)

Make the copilot's `read_shader` result lead with a concise **structural inventory** of the shader —
every uniform it DECLARES (with type, value, engine-driven flag, declaration line), its `#define`s, its
top-level functions (name + line), and its entry decls — so the agent has a cheap, authoritative picture
of what a shader already contains and stops re-deriving it from the raw source (or guessing wrong).

This is **Phase 1 of the prompt-audit refactor** (the audit report, this session). It is the single
load-bearing fix: it kills the seed-bug *class* (imperative-with-no-state-check + no-cheap-state-snapshot)
at its root. The paired prompt-prose fixes (the declare-rule guard, the redundancy trims, the restructure)
are tracked separately — see *Out of scope*.

## Goal

A node read via `read_shader` returns, ABOVE the line-numbered listing, a STRUCTURE block. Example for the
Text Rendering node:

```
=== Text Rendering (id: a646) ===                              (uniforms abbreviated for the example)
STRUCTURE (what THIS source already declares — do NOT re-declare; engine uniforms are read-only):
  uniforms:
    u_time       float     [engine, L10]  = 12.40
    u_aspect     float     [engine, L11]  = 1.778
    u_zoomout    float     [L13]          = 10.0
    u_offset     vec2      [L15]          = (0.0, 0.0)
    u_text       uint[64]  [L21]          = <inactive>
    u_color      vec3      [L22]          = (1.0, 0.0, 0.0)
  defines: MAX_DIST (L3), MIN_SMOOTHNESS (L4), MAX_TEXT_LEN (L5)
  functions (recognized 68): get_dist_to_line (L24), ... (66 more), main (L266)
  entry: in vec2 vs_uv -> out vec4 fs_color

  1  #version 460 core
  ... <the full cat -n listing, unchanged> ...

errors:
none
```

(The uniform list is NOT capped — the real block shows all 10 declared uniforms; this example abbreviates.
The functions count is "recognized 68" not a guaranteed-complete count — see Decision 6 + the multi-line-sig
limitation in Out of scope. `u_text[64]` shows `64` because the parser resolves the `MAX_TEXT_LEN` `#define`
— Decision 9.)

This directly kills the seed bug: the FIRST content of a read tells the agent `u_time` is already declared
at L10 (and is engine-driven, read-only) **before** it can compose an `insert_after("uniform float
u_time;")`. The `<inactive>` row is the load-bearing half — a uniform declared-but-not-yet-referenced (the
exact seed condition) does NOT appear in GL introspection, so the block must be sourced from a SOURCE PARSE
unioned with the introspected set, not from `get_active_uniforms()` alone.

## Out of scope

Each deferral carries a trigger so a later pass picks it up.

- **The prompt-prose fixes (the other ~38 confirmed audit findings).** The declare-rule guard
  (`context.py` convention), the redundancy trims, the section restructure, the tool-description tightening.
  **Trigger:** this feature lands + is verified, THEN the prose pass is its own wave (the audit report's
  Phase 1 prose-fixes + Phase 2 restructure). The ONE prose change folded into THIS feature: the
  `_READ_SHADER_DESC` sentence + the `_INSERT_AFTER_DESC` clause that point the agent AT the STRUCTURE block
  (without it the new data is unreferenced) — see Design decision 7.
- **Engine-uniform frozenset consolidation (audit G4).** The `{u_time, u_aspect, u_resolution}` triple is
  duplicated in `app.py::_ENGINE_DRIVEN_UNIFORMS`, `ui_models.py` (x2). This feature READS the canonical
  `app.py` frozenset (no new copy) but does not consolidate the others. **Trigger:** next edit to the
  engine-uniform set, or a sanitize pass on the duplication.
- **Function body-end / line-RANGE in the structure block.** The block shows each function's START line
  only (cheap, unambiguous). A full `(start–end)` range needs brace-matching (`parser.find_body_end`).
  **Trigger:** first time the agent demonstrably needs the end line to target a `replace_lines` over a whole
  function without a re-read (no trace shows this yet).
- **Macro-indirection / `#define HASH SB_foo` in the function scan.** The function scanner lists top-level
  `type name(...)` definitions; it does not expand macros. Same limitation already filed for the lib
  parser. **Trigger:** the existing `todo.md [DEFERRAL] lib-author macro indirection` trigger.
- **Structure block on `grep` / `create_node` results.** Only `read_shader` gets it (that's where the agent
  orients before an edit, and the freshness guard guarantees a read precedes any edit). **Trigger:** a trace
  shows the agent needing structure from a non-read tool.
- **Multi-line function signatures + `#if 0`-guarded decls.** `FN_SIG_RE` is single-line-anchored, so a
  function whose `(...) {` spans lines is omitted (the `functions (recognized N)` framing + the prompt
  sentence make this safe — see Decision 10). A uniform inside `#if 0` shows a spurious `<inactive>` row
  (preprocessor not evaluated). **Trigger:** a trace shows the agent re-declaring a multi-line-sig helper, or
  a real shader's `#if 0` block misleads it — then add brace-matching / minimal preprocessor handling.

## Design decisions

1. **The block lives on the `read_shader` RESULT, not the project map / a new context block / a new tool.**
   The project map (`NodeTreeEntry`) is GL-free + value-free ON PURPOSE so it rides the OpenRouter prefix
   cache (`capabilities.py:17-25`, `prompt.py:18-24`); uniform VALUES change per frame and would bust it
   every turn. The seed bug fires at EDIT time, and the freshness guard (`EditResult.stale`) already forces
   a `read_shader` of a node before any edit to it — so a fresh read is guaranteed present exactly when the
   bug can fire. The block rides that read for free: zero new tool, zero "when to call it" rule, zero cache
   risk. It sits AFTER the warm prefix (in a tool result), so it has zero prefix-cache impact.
   **History-budget interaction:** the read_shader `body` (now carrying the block) is persisted in the
   per-turn tail (020·23 D4) and re-fed as history, counted by `prompt._estimate_tokens` and bounded by the
   `_trim_history` window (this session) + `_MIN_KEPT_TURNS=4`. Added cost ≈ 240 tok/read (cap=12), carried
   only for retained turns — small and bounded, not a per-turn-forever cost. So "net-negative" holds only
   when the block actually suppresses a re-read; the honest claim is "small bounded cost, repaid by
   suppressed re-reads on edit-heavy turns."

2. **Uniform rows = a SOURCE-parse of `uniform` declarations, UNIONED with the GL-introspected active set.**
   The source parse is authoritative for "is it declared" (catches declared-but-unused — the seed case) and
   for the declaration LINE. The introspected set (`node.get_active_uniforms()`) is authoritative for the
   live VALUE and the true GL type. Join on name:
   - declared + active → full row `name type [engine?,Lnn] = value`.
   - declared + inactive (not referenced yet) → `name type [Lnn] = <inactive>` (type from the source decl;
     no live value exists).
   - active + not found in the source parse (declared in a lib file, or a parse miss) → `name type = value`
     with no line tag (best-effort; never invent a line).

3. **The engine-driven flag reuses `app.py::_ENGINE_DRIVEN_UNIFORMS` (the SAME frozenset `set_uniform`
   rejects against), never a new list.** A uniform whose name is in that set renders `[engine, Lnn]`. This
   is the single classifier — no drift (audit G2 + the reuse rule).

4. **The structural extraction is a GL-FREE source parse in a NEW pure leaf module
   `copilot/shader_struct.py`** (no App/imgui/moderngl import — same leaf discipline as
   `copilot/glsl_lex.py`; runs on either thread, it's pure text). Hard requirements (each pins a pre-impl
   review finding):
   - **Line numbers are computed against the ORIGINAL source, never against a comment-stripped copy.**
     `shader_lib.parser.strip_comments` collapses a `/* */` block to nothing and shifts every later line up
     — so a decl after a block-comment banner would report a WRONG `[Lnn]`, and a confident-wrong line is
     strictly worse than no block (the block's whole value IS the line). So this module does its own
     **length-preserving comment blanking** (replace comment characters with spaces, keep every `\n`), then
     scans, so a match's line index equals its line in the file the agent sees. (Do NOT reuse
     `strip_comments`; it is length-destroying by design for the lib index.)
   - **Multi-declaration uniforms emit one row per name.** `uniform float u_a, u_b;` yields rows for BOTH
     `u_a` and `u_b` — if `u_b` is the declared-but-inactive one, omitting it re-creates the seed bug on the
     exact input class this feature targets. The uniform regex captures the trailing comma list and splits it.
   - **The array suffix is captured.** `uniform uint u_text[MAX_TEXT_LEN];` parses to name `u_text`, base
     type `uint`, array-size token `MAX_TEXT_LEN` (resolved per Decision 9).
   - **Best-effort, omit-don't-mis-report** for the function/entry scan: a construct the scanner can't
     classify (a multi-line function signature, a rare top-level brace) is omitted, never mis-ranged — the
     full listing is always the ground truth. The known omissions are documented to the agent (Decision 10).
   - The function scan reuses `shader_lib.parser.FN_SIG_RE` (the only fitting regex — NOT `USER_FN_DEF_RE`,
     which is SB_-only) against the length-preserved-blanked source.

5. **`ShaderView` gains structured fields; the flat `uniforms: list[str]` is REPLACED by typed rows.** New
   leaf value objects on `capabilities.py` (primitives + lists only, no-banned-import rule):
   `UniformRow(name, type, value, engine, decl_line)` (value is `str` — pre-rendered or the `<inactive>`
   sentinel; decl_line is `int | None`). `ShaderView` gains `uniforms: list[UniformRow]` plus three new
   fields **with safe defaults so unrelated `ShaderView(...)` constructors don't all need updating**:
   `defines: list[tuple[str, int]] = field(default_factory=list)`,
   `functions: list[tuple[str, int]] = field(default_factory=list)`, `entry: str = ""`. `_view_summary`
   reads `len(view.uniforms)` — unchanged. **The two existing test fixtures that build a `ShaderView` with
   `uniforms=<list[str]>` (`test_copilot_loop.py`, `test_edit_safety.py`) MUST move to `UniformRow`/`[]`** —
   they are in Files touched; pyright (the `make check` gate) fails otherwise.

6. **The functions line is TRUNCATED past a cap (default 12), always showing `main`.** `functions (61):
   first(L24), second(L31), ... (N more), main (L266)`. Keeps the block bounded (the Text Rendering shader
   has 61 functions — uncapped that's a long line every read). Cap is a module constant. `#defines` and the
   uniform list are NOT capped (bounded in practice; the uniform list is the load-bearing data).

7. **Two tool-description sentences are folded in (the only prose change in this feature):**
   `_READ_SHADER_DESC` gains "The STRUCTURE block is the authoritative list of what this shader ALREADY
   declares — consult it before adding a uniform/function." `_INSERT_AFTER_DESC` gains a duplicate-decl
   warning pointing at the block. Without these the new data is unreferenced; the rest of the prose audit
   stays out of scope.

8. **Value formatting matches what the agent sees today (no round-trip fix here).** Vectors render
   `(1.0, 0.0, 0.0)` as `_format_uniforms` does now; the text-array decode + set_uniform-shape round-trip
   (audit G7) is a separate prose/format concern, deferred with the rest of the audit Phase. The
   `<inactive>` sentinel is the only new value rendering.

9. **A `#define`-valued array size is RESOLVED via the same-pass defines map.** `uniform uint
   u_text[MAX_TEXT_LEN]` with `#define MAX_TEXT_LEN 64` renders `uint[64]`. The parser builds the `defines`
   map first, then resolves a bare-identifier array-size token against it; an unresolvable token (not a
   literal, not a known define) renders verbatim (`uint[MAX_TEXT_LEN]`), never invented. This is needed
   because the load-bearing `<inactive>` array (the seed-condition `u_text`) has NO GL introspection to read
   the size from — source is the only source. (NOTE: this is define resolution for ARRAY SIZES only; macro
   indirection in the function-call graph stays deferred per Out of scope.)

10. **The parser's known omissions are stated TO THE AGENT, so absence never reads as "doesn't exist".** The
    function header is rendered `functions (recognized N):` and the `_READ_SHADER_DESC` sentence says the
    list is "the functions the scanner recognized — a multi-line-signature helper may be absent; the full
    listing below is authoritative." Without this, the agent could read a recognized-list miss as "this
    function isn't defined" and re-declare it (a softer seed bug). Known omissions: a function whose
    signature spans multiple lines (FN_SIG_RE is single-line-anchored); a uniform inside an unevaluated
    `#if 0` block (it shows as a spurious `<inactive>` row — preprocessor conditionals are not evaluated).
    Both are low-frequency; documented rather than solved (solving needs brace-matching / a preprocessor —
    deferred, Out of scope).

## Files touched

- **`copilot/shader_struct.py`** (NEW, ~70 lines) — pure leaf: `extract_structure(source: str) -> ShaderStructure`
  where `ShaderStructure` holds the parsed `uniforms` (name, type, decl_line), `defines`, `functions`,
  `entry`. GL-free, no banned imports. The regexes + `strip_comments` reuse.
- **`copilot/capabilities.py`** — add `UniformRow`; extend `ShaderView` (typed `uniforms` + `defines` +
  `functions` + `entry`).
- **`app.py`** — in the `read_shader` `_on_main` builder (~857): call `extract_structure(text)`, JOIN with
  `get_active_uniforms()` (value + engine flag via `_ENGINE_DRIVEN_UNIFORMS`), build the typed `ShaderView`.
  A new `_build_uniform_rows(struct, active)` helper.
- **`copilot/tools/shader.py`** — `_format_view` emits the STRUCTURE block via a new `_format_structure`
  helper; `_view_summary` reads typed uniforms; the two tool-description sentences (Decision 7).
- **`tests/test_shader_struct.py`** (NEW) — the parser unit tests: uniform decls incl. arrays + multi-decl
  (`u_a, u_b`) + macro-sized array (`[MAX_TEXT_LEN]` -> `[64]`), defines, functions, entry, the
  length-preserving comment blank (a block-comment banner before a decl -> the decl's line is the ORIGINAL
  line), the declared-but-inactive union, the engine flag, the functions cap.
- **`tests/test_copilot_loop.py`** + **`tests/test_edit_safety.py`** (EXISTING — must update) — both construct
  a `ShaderView` with `uniforms=<list[str]>`; move to `UniformRow(...)` / `[]` so `make check` (pyright) passes.
- **`tests/test_cross_project_tools.py`** (EXISTING — may need a stamp-positive assertion; see verification).

## Manual verification

Headless (the agent can run these):
- **Parser unit tests** (`test_shader_struct.py`) over the Text Rendering source: `u_time`/`u_aspect` flagged
  engine, `u_text` parsed `uint[64]` (define resolved), multi-decl `uniform float u_a, u_b;` -> two rows,
  `main` present, the defines, the entry decls. **The load-bearing line-number test:** a source with a
  multi-line `/* ... */` banner before a uniform asserts the reported `decl_line` is the ORIGINAL line, not
  the comment-stripped one (pins the CRITICAL pre-impl finding).
- **Headless `read_shader` integration test** (extend `test_cross_project_tools.py`): build a `Node`, compile,
  call the capability; assert the rendered block contains `u_time ... [engine, L` AND a `<inactive>` row for a
  declared-but-unused uniform. **ALSO assert the freshness stamp fired** — `app._copilot_read_revision` gained
  the node's digest after the read (the stamp lives in the exact builder this feature rewrites; only fakes
  test it today, so the rewrite could silently drop it).
- `make check` (the real gate — pyright + ruff). `make smoke` is a **regression guard only** here (it never
  calls read_shader / the copilot — it catches import breakage from the new module, nothing feature-specific).

Maintainer (live, one line): run `make run`, open the copilot, ask it to "add a time-based animation to the
text shader" — confirm it does NOT re-declare `u_time` (the seed-bug repro), and that the chat read-summary
still reads cleanly.

## Open questions for the user

- **Q1 — functions cap default.** 12 (shows first-few + main + "(N more)")? Or higher (say 40, so a
  ~30-function shader shows them all)? Trade-off: token cost per read vs. how often the agent must re-read to
  find a function it didn't see. Lean: 12 — the agent can `grep` a function name it needs.
- **Q2 — replace the flat `uniforms` field, or keep both?** The spec REPLACES the flat `list[str]` with
  typed rows (Decision 5). Any external reader of `ShaderView.uniforms` as strings breaks — but the only
  readers are `_view_summary` + `_format_view`, both updated here. Confirm the clean replacement (vs. an
  additive `uniform_rows` field + keeping the old one, which leaves two sources of truth). Lean: replace.

## Review history

**Pre-impl review (2 agents: correctness/design + verification/blast-radius).** Both: LAND_WITH_FIXES; both
independently confirmed the load-bearing premise (GL strips declared-but-unused uniforms, verified against
`core.py::get_active_uniforms` -> moderngl `Program._members` -> `GL_ACTIVE_UNIFORMS`), so the
source-parse∪introspection design is justified. Findings folded into the spec:
- **CRITICAL (line shift):** `strip_comments` is length-destroying, so reusing it would report wrong `[Lnn]`
  after any block comment. -> Decision 4 now requires length-preserving comment blanking against the ORIGINAL
  source + a dedicated test. (Both reviewers, independently.)
- **HIGH (multi-decl):** `uniform float u_a, u_b;` would drop `u_b` -> seed bug on the target input class. ->
  Decision 4 requires one row per name. (Reviewer 1, H1.)
- **HIGH (macro array size):** the `u_text uint[64]` example is unachievable from source (size is a
  `#define`). -> Decision 9 (resolve array size via the defines map). (Both reviewers.)
- **HIGH (test fixtures break the gate):** `test_copilot_loop.py` + `test_edit_safety.py` build
  `ShaderView(uniforms=<list[str]>)`. -> added to Files touched; new fields get defaults (Decision 5).
- **MEDIUM:** multi-line sigs / `#if 0` omissions -> Decision 10 (state "recognized N" to the agent) + an
  Out-of-scope deferral with a trigger. Freshness-stamp only fake-tested -> verification now asserts the real
  stamp fired. `make smoke` is a no-op here -> demoted to regression-guard. History/persistence cost
  (~240 tok/read) -> noted in Decision 1. Stale "61 functions" count -> "recognized 68".

No SHOULD_NOT_LAND finding; no design disagreement requiring a maintainer call.
