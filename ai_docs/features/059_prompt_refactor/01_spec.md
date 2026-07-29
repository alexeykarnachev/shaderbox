# 059 — Prompt refactor: the SCRIPTING watershed, a generated script-API block, a duplication cut

Review anchor: `00_maintainer_feedback.md` (verbatim voice transcript + the follow-up correction +
the moderator's Q1/Q2 decisions). Reviewers check this spec against THAT file, not against this
file's restatement of it.

Maintainer review of the assembled prompt on the agent-hub page (2026-07-29) found the static system
prompt carrying three kinds of dead weight: a **wrong** rule (the SCRIPTING watershed says "a script
is for values that CHANGE" — the correction is "скрипты мы используем, когда у нас есть state"),
**implementation details** the agent has tools for (the `script.py` path; hand-written Python type
prose that has been stale since feature 054 gave `Vec2/3/4` real vector math), and **verbatim
restatement of tool descriptions** the model already receives in `tools=` (measured: the eager tool
block is ~19.8k chars / ~4.9k tok — the same order as the whole 20 507-char system prompt). This
feature rewrites the wrong rule, replaces prose typing with a generated script-API block delivered
beside the GLSL catalogue in the RARE tier, cuts every line the schemas already carry, and moves the
render-facts legend to sit beside the facts — under a hard constraint that the prompt comes out
shorter, in three separately-gated waves, with every cut carrying a named regression risk.

Baseline: `dev` @ **6ed3c4d**.

---

## Related waves & already-answered feedback

The hub-page review covers engine questions as well as prompt ones. Mapping (M-numbers follow the
anchor's paragraph order; grep them there):

| # | Maintainer point | Status |
|---|---|---|
| M1 | "откуда у нас вообще reasoning… мы же его отключали" | **Wave 1 DONE.** `COPILOT_ENGINE.llm_reasoning_effort = "none"` (289c12f). `"minimal"` is silently ignored on tool-bearing requests; `"none"` is honored (88 vs 1567 out tokens, ~2x cheaper). Ledger + echelon gate: `ai_docs/features/057_dogfood_axes_and_scenarios/05_reasoning_none.md`. |
| M2 | "проблема с нашим тулингом… или модель тупая?" + "идентейшн не нужно соблюдать" | **Answered, no work.** Indentation tolerance already exists: GLSL edits match on the token stream (`copilot/glsl_lex.py`), script edits match plain text with an `indent_shift` re-indent and a whitespace near-match hint (`backend.py::_whitespace_near_match`, `_plain_text_spans`). Tooling-vs-model: the exam run shows 4 compile recoveries across 63 iterations — the tooling is not the cost centre; blind edits are. |
| M3 | "рендер-факты действительно помогают?… side-by-side A/B" | **Wave 4, not this feature.** Planned observation-first (instrument what the model does with the facts line) then a side-by-side render comparison (facts / no facts / facts-without-motion). Trigger recorded HERE: run it after 059's waves land and the base echelon is re-baselined at `effort=none`, so the A/B is not confounded by a prompt rewrite. (`todo.md` is frozen drain-only by maintainer decree — this row is the trigger's home.) |
| M4 | "внутренние детали… инкапсулированы во внутренний конфиг" | **Wave 2 DONE** (6ed3c4d): user-tunable vs engine config split; the hub page renders two tables. |
| M5 | "editing правило — нужно их писать или нет" | This feature, **D4 + D5**. |
| M6 | "text content never a const array… слишком специфично… давай подумаем" | This feature, **D4** — treated as a brainstorm, not a deletion. |
| M7 | "описать, как пользоваться рендер-фактами… прямо рядом с рендер-фактами" | This feature, **D7** (Q1 locked = C). |
| M8 | "Values, Notes, Library… агент получает описание тулзов вместе с тулсетом" | This feature, **D5**. |
| M9 | "должен ли агент знать про путь к файлу" + "правила слишком оверфитнутые" | This feature, **D2 + D1**. **Scope note:** this wave de-overfits the SCRIPTING routing table only. The other routing/rules clusters (USING TOOLS, HOW TO WORK, VISUAL CRAFT) are NOT surveyed here. Trigger: the maintainer reviews one of those sections on the hub page, or a dogfood run shows a routing rule being followed against the user's actual intent. |
| M10 | "может быть нам просто стоит сделать файл с интерфейсом… в тексте это плохо мейнтейнится" | This feature, **D3** — the explicit mandate for the generator. |
| M11 | "всё, что можем bake in… должно быть впечатано… нужен хороший баланс" | This feature, **D6** (investigated; answer is "don't bake") + the whole-file size cut (D8), which is the balance half. |

---

## Measured baseline (`dev` @ 6ed3c4d)

`_SYSTEM_PROMPT` = **20 507** chars (~5 126 tok). Per section (body chars, excluding the blank-line
separators; 13 sections + 12 separators = 20 507):

| section | chars |
|---|---:|
| (preamble) | 325 |
| WORKING SET | 652 |
| EDITING | 2 001 |
| FEEDBACK | 2 472 |
| VALUES, NODES, LIBRARY | 1 733 |
| SCRIPTING | 3 196 |
| VISUAL CRAFT | 4 396 |
| RENDER & PUBLISH | 1 072 |
| TELEGRAM + YOUTUBE | 807 |
| USING TOOLS | 1 067 |
| ADDRESSING | 407 |
| THE SANDBOX | 523 |
| HOW TO WORK | 1 844 |

RARE tier: `_CONVENTIONS` 1 394 chars + the per-project map / lib / example catalogues.
Eager `tools=`: 18 tools, 8 706 chars of description + ~19.8k chars of serialized schema.
Lazy tools: 14, every one carrying a non-empty `catalog_summary` (verified programmatically).

---

## Landing order — three waves, each independently gated

The prompt is a single string; a one-wave rewrite is unbisectable, and D1 is the only change that can
regress a base capability outright. So this feature lands as three commits, each with its own gate,
and **wave A does not proceed to wave B until its gate passes**:

| wave | contents | prompt size after | gate |
|---|---|---:|---|
| **A** | D1 (watershed) + D2 (implementation details out) + D9 (test rewrite) + the D1 half of the tool descriptions | 19 096 | de-hinted 05, 04, 08, 10, 13 (see Validation) |
| **B** | D3 (SCRIPT API block) + D4 (TEXT-rule resolution) + D5 (dedup cuts + the schema-error fixes) | 15 511 | 13, plus the four micro-probes |
| **C** | D7 (legend trim + splice) | 14 435 | 03 + 04 honesty axis (03 needs a control — see Validation); the splice unit test |

Wave A carries the tool-description watershed edits (D1a) because leaving them a wave behind would
put the corrected rule and its negation in the same context window — the exact defect being fixed.
Wave B carries the remaining tool-description corrections (D5).

---

## Design decisions (locked)

### D1 — The SCRIPTING watershed is STATE, not "values that change"

**Wrong today.** The section says a script is for VALUES THAT CHANGE and hands the model an
overfitted example table whose third row is wrong outright:

```
- WHICH tool sets a value -- pick by what the user wants:
    "make it pulse / drift / animate / react over time"   -> write_script (value from ctx.t)
    "make it brighter / bigger / slower" (one fixed value) -> set_uniform
    "add a u_glow uniform"        -> edit_shader to declare it, THEN write_script to drive it
    "change what the shader DOES with a value" (logic)    -> edit_shader (source)
  A script is for VALUES THAT CHANGE; set_uniform / an inline default is for a value that sits.
```

Row 1 sends every pulse/orbit/spin — pure functions of `u_time`, per-pixel work by nature — to the
CPU. Row 3 asserts a `u_glow` uniform needs a script; it does not. This contradicts the engine's own
locked decision (`conventions.md`: "Per-instance state (`self.*`) persists across frames — the reason
CPU scripting exists (a stateless `sin(t)` belongs in the shader)").

**After** (1 785 chars, was 3 196) — one principle, one special case, no table:

```
SCRIPTING (node scripts -- CPU state the shader cannot hold)
- THE WATERSHED: a script exists for STATE -- a value that depends on the PREVIOUS frame (an
  integrator, an accumulator, a state-machine phase, a score, a collision response). A value that is
  a PURE FUNCTION OF TIME -- a pulse, an orbit, a spin, a colour cycle, a wave -- is computed IN THE
  SHADER from u_time and needs NO script. Ask "does this frame's value need last frame's value?":
  yes -> a script; no -> GLSL. One build can need both: script the stateful parts, leave the
  time-pure parts in GLSL.
- The one special case: HEAVY stateful compute (a cloth Verlet sim, particles, a boids flock) --
  step the CPU state each frame and push the result as an ARRAY uniform (`Array([..flat..])` ->
  `uniform vecN arr[M];`), never faked per-pixel. Per-pixel work (noise, ramps, lighting, SDF) stays
  GLSL. Check the sim's ranges/stability before rendering (a blow-up shows only as a black frame).
- INSIDE a script, drive motion from ctx.t, never ctx.mouse: mouse is frozen at center on export and
  in the headless probe, so a mouse-driven uniform reads STATIC even when it is correct.
- A script-DRIVEN uniform is NOT set_uniform-able (a set is overwritten next tick and rejected). To
  change a driven value, edit update -- not the shader default (once driven, the default only seeds
  the initial value). A script writes VALUES only: to add a uniform, edit the SHADER first.
- You SEE a node's script in the WORKING SET (its own SCRIPT sub-section, rebuilt every step) -- no
  separate read for the current node. A write/edit returns its probe verdict: the compile result
  (fix it FIRST, like a shader compile), the uniforms it now drives (0 driven = animates NOTHING),
  and a motion verdict ANIMATING/STATIC.
```

The mouse bullet is deliberately **rescoped**: it previously read as general animation advice ("drive
AUTONOMOUS animation from ctx.t"), which under the corrected watershed would push time-pure animation
INTO scripts — the precise over-fit being removed. It now applies only inside a script, and only as
"ctx.t not ctx.mouse". The frozen-mouse fact is also carried by D3's generated `ctx.mouse` gloss, so
the two statements agree by construction.

**D1a — the tool descriptions carry the watershed too.** `tools/script.py`'s eager descriptions ride
`tools=` on every iteration and currently negate the corrected rule:

- `_READ_SCRIPT_DESC` — "the `update(self, ctx)` that **drives uniforms over time**… returns a STUB
  (… + **one ctx.t example to ADAPT**)". Both clauses teach time-driving, and the STUB clause is also
  factually stale: `script_stub_for` emits commented VALUE examples (`# 'u_x': 0.0,  # float`) and an
  empty `return {}` — there is no `ctx.t` example in it.
  **New:** "Read a node's Python script — the `update(self, ctx)` that drives uniforms from CPU
  state. Returns the source line-numbered. A node with NO script yet returns a STUB (its drivable
  uniforms + their value shapes + an empty `update` to fill in) — read it, then write_script a real
  body. Read this before editing a script you did not just write."
- `_WRITE_SCRIPT_DESC` — "to DRIVE those uniforms every frame (**ANIMATION** / state over time…)".
  **New:** "Create or replace a node's Python script: a `class Behavior(ScriptBehavior)` whose
  `update(self, ctx) -> dict` returns {uniform_name: value} to drive those uniforms every frame. For
  STATE the shader cannot hold — a value that depends on the PREVIOUS frame (`self.*` persists): an
  integrator, an accumulator, a phase machine, a score. A pure function of time belongs in the shader
  (u_time), not here. BEST FOR a fresh script or a full rewrite; for a localized change prefer
  edit_script. Send the COMPLETE script — I compile + motion-probe it and return the verdict."
- `_EDIT_SCRIPT_DESC` — no watershed language and no path (only the bare `script.py` artifact name,
  which D2 keeps): unchanged.

**D1b — the fresh-node stub under the corrected rule.** `script_stub_for`'s emitted BODY is already
neutral (commented value examples, `return {}`) — no change needed there, which is the honest answer;
the time-framing lives only in prose. Two prose edits: `_UPDATE_DOC`'s Args block keeps the ctx
listing but gains one line — "A value that is a pure function of `ctx.t` usually belongs in the
shader instead; this class is for state you keep on `self`." — and `_INIT_DOC` ("Set up state (runs
ONCE…)") is already state-framed and stays. `_UPDATE_DOC`'s stale Vec paragraph dies under D3.

**Risk (two-sided).** Over-correction: the model refuses to script and fakes a physics sim with
`u_time` trigonometry (scenario 05's stated FAIL). Under-correction: it still scripts a pure orbit
(scenario 04's stated FAIL). Mis-mixing: in a build needing both, it routes the wrong halves
(scenarios 10, 13). All three gated below.

### D2 — Implementation details leave the model's view

The maintainer's point is context pollution: "зачем ему полютить, загрязнять контекст такими
деталями… ему нужны ручки". Four channels leak paths today; all four are in scope:

1. **The prompt** — "A node can have ONE Python script at nodes/<id>/scripts/script.py". Removed with
   the D1 rewrite.
2. **`_READ_SCRIPT_DESC`** — "(nodes/<id>/scripts/script.py)". Removed (see D1a). This is the channel
   that actually serves the maintainer's point: the description rides every single iteration, the
   prompt line rode once.
3. **The working-set header** — `prompt.py::_render_working_set_member` emits
   `=== <name> SCRIPT (scripts/script.py) ===`, rebuilt EVERY step for every scripted node in the
   set. Becomes `=== <name> SCRIPT ===`.
4. **Compile-error paths** — `backend.py::_to_error_infos` does `path=str(e.path)`, so every shader
   error line in the working set and in every edit result reads
   `/home/…/projects/dev/nodes/<uuid>/shader.frag.glsl:12: …`. This is the largest leak of the four
   (per error, per step) and was not named in the review. The path is load-bearing for exactly one
   model-facing thing: telling a root-source error from a spliced lib-file error. So it becomes a
   LABEL, not nothing — the node's short id for the root source, `lib:<path-relative-to-the-lib-root>`
   for a lib source (the address the agent already uses), `""` unchanged for the synthetic no-node
   error. Script errors already carry the bare label `script.py` and are unchanged.

   **The mapping happens at RENDER time, not on the value object.** `CompileErrorInfo.path` has two
   INTERNAL consumers that read it as a real filesystem path, and rewriting the field would silently
   break both: `backend.py::_cross_file_note` uses `Path(e.path).resolve() == edited` as its
   own-file guard and `Path(p).relative_to(shader_lib_root())` to label the foreign file, and
   `_edit_error_hints` drops the brace-balance hint whenever that note is non-empty. With a short-id
   label the guard can never match, so EVERY same-file compile error would emit the misleading
   "the error is in …, which this shader pulls in — the file you edited may be fine" note AND lose
   the brace hint — a strict regression on the most common error path. So `_to_error_infos` keeps
   the absolute path on the info object (internal contract unchanged), and the label is applied by
   the six model-facing renderers: `prompt.py::_format_compile_errors`,
   `tools/shader.py::_format_errors`, `tools/script.py::_fmt_errors`, the two inline joins in
   `tools/node_ops.py` (duplicate_node, import_node), and the restore-errors join inside
   `backend.py::_force_restore`. One shared helper, called at each — not a
   per-renderer f-string, so a seventh renderer cannot re-leak the path by omission.

   **This is the one D2 item with real behavioural surface.** Falsifiers, both required:
   (i) a lib-originated error still renders a `lib:` address and a node error renders the node's
   short id; (ii) **a same-file compile error emits NO cross-file note and DOES emit the brace hint**
   — cut the render/value split and (ii) goes red.

**Risk:** the agent asks the user where a script lives, or hallucinates a path into an arg. Low — THE
SANDBOX's "You NEVER type a filesystem path" survives verbatim and every script tool takes a node id.
Falsifier: grep the gate runs' traces for `/` or `nodes/` inside a script-tool argument. For item 4
the risk is that the agent can no longer distinguish two lib files in one error batch — the `lib:`
address preserves exactly that.

### D3 — A generated `SCRIPT API` block in the RARE tier

**Mandate.** M10, verbatim: "может быть нам просто стоит сделать файл с интерфейсом… Просто в тексте
это описывать, это плохо поддерживаемо. Вот мы еще метод добавим, нам придется в тексте искать, где
мы это написали."

**Why a generator rather than the minimal fix.** The minimal fix — hand-correct the stale paragraph
and move on — is real and costs ~30 minutes. It is rejected for one measurable reason: *this exact
class of drift already happened and is live in production today*. `scripting/engine.py`'s stub
docstring `_UPDATE_DOC` (e37c372, 2026-06-14) tells the model and the user

> "Vec2/3/4 are RETURN wrappers, not a math type… A Vec2 has no .x/.y (it is a tuple: v[0], v[1]) and
> Vec2(...) \* n repeats the tuple, it does NOT scale."

while `scripting/outputs.py` (14e70d7, 2026-07-03, feature 054) gives `_Vec` exactly `.x/.y/.z/.w`,
`+ - * /`, `.dot`, `.length`, `.normalized`, `Vec3.cross`. The system prompt states the TRUE version.
So a fresh-node turn today puts a correct statement and its exact negation in the same context
window, and it went unnoticed for four weeks. Hand-correcting produces a third hand-maintained copy;
the maintainer's ask is explicitly for the mechanism that makes the fourth one impossible. The build
stays minimal per the two items below. **Honest net cost:** the generated block is 1 214 chars in
RARE against ~700 chars of STATIC prose it replaces — about **+500 chars overall** (the STATIC+RARE
row above nets to -4 654 only because every other D-item is a cut), bought for drift-proofing. Every other D-item is a net cut; this one is not, and it is the only one.

**Where it lives.** New leaf module `shaderbox/scripting/api_doc.py`, beside the types it describes,
exporting `script_api_summary() -> str`. `prompt_context.py` imports **`from
shaderbox.scripting.api_doc import script_api_summary`** — NOT from the package root:
`scripting/__init__.py` re-exports `engine.py`, which imports `moderngl` and `OpenGL.GL`, and the RARE
block is built GL-free off the main thread. `api_doc.py` itself imports only `context.py` +
`outputs.py` (both GL-free); a test pins that importing it pulls in no `moderngl`.

**How it is generated — two mechanisms, deliberately not three.**

1. **ctx fields** — iterate `EngineContext.__dataclass_fields__` and `MouseState.__dataclass_fields__`,
   rendering `name: type` plus a one-line gloss from a module-level `_CTX_GLOSS` dict. A test asserts
   `_CTX_GLOSS.keys()` equals the dataclass field names exactly, so a new `ctx` field goes red until
   documented. **The `mouse` gloss MUST carry the frozen-at-center caveat** — commit 17ab552 inlined
   that fact next to the ctx intro precisely because a live leak had shown the agent trusting mouse
   motion in a probe; moving the ctx intro to RARE without the caveat would undo a documented fix.
2. **The vector/value surface** — a coverage test over the rendered summary. **Mechanism: `vars(cls)`,
   not `inspect.getmembers`.** `getmembers(Vec2)` leaks `count`/`index` inherited from `tuple` (noise
   that would force an allowlist anyway), returns nothing useful for `Array`/`Text` (their surface is
   the constructor), and — decisively — skips dunders, i.e. it would NOT have caught `__mul__`, which
   is the exact drift (`Vec2(...) * n`) that motivated this work. So: collect public names from
   `vars(_Vec) | vars(Vec2) | vars(Vec3) | vars(Vec4)`, plus an explicit dunder allowlist
   `{__add__, __sub__, __mul__, __rmul__, __truediv__, __neg__}` mapped to the operator glyphs the
   summary prints; for `Array`/`Text` assert the class name and its `__init__` parameter names appear
   (via `inspect.signature`). Adding `Vec3.reflect()` or dropping `__truediv__` turns the test red.

**Explicitly NOT done: lifting `_stub_kind` into a shared table.** The earlier draft proposed one
shared ordered `(label, type, example)` table consumed by both `engine.py::_stub_kind` and the
summary. It does not survive contact with the code: `_stub_kind` is a 7-outcome predicate dispatch
over a live `moderngl.Uniform` (`is_text_array(u)`, then `array_length > 1`, then `dimension`, then
`gl_type in (GL_INT, GL_UNSIGNED_INT)`), and two of its outcomes have *parameterized* example
expressions (`Array([0.0] * {n*dim})`). A table cannot express the dispatch, and moving the dispatch
into `api_doc.py` would drag `moderngl` into the GL-free module. So the dispatch and the example
generation stay in `engine.py` untouched; `api_doc.py` shares only the **type-name → prose gloss
map**, and a test asserts every type name `_stub_kind` can return is a key in that map — the drift
pin without the lift.

**Rendered block** (1 214 chars):

```
SCRIPT API (generated from shaderbox/scripting -- the Python side of a node script):
- `class Behavior(ScriptBehavior)`: `__init__(self)` runs once; `update(self, ctx) -> dict` runs
  every frame and returns {uniform_name: value}. State on `self.*` persists across frames; a key you
  omit (or map to None) stays MANUAL.
- ctx: `t` float seconds, `dt` float, `frame` int, `mouse` MouseState(`x`, `y` in 0..1, y-up --
  FROZEN at 0.5,0.5 on export and in the headless probe).
- Legal value shapes: float|int -> scalar; Vec2(x,y) / Vec3(x,y,z) / Vec4(x,y,z,w) -> vec2/3/4;
  Array(seq) -> a numeric array uniform (flat numbers, or ROWS of Vec/list, auto-flattened);
  Text(str) or a plain str -> a uint[] glyph array. A bare FLAT list also coerces (a vec, or an
  exact-length numeric array); a NESTED bare list does not -- that is what Array is for.
- Vec2/Vec3/Vec4 are real vectors: `.x .y .z .w`, `+ -` (same length), `* /` (scalar or
  component-wise), unary `-`, `.dot(o)`, `.length()`, `.normalized()`; Vec3 also `.cross(o)`.
- `from shaderbox.scripting import ScriptBehavior, Ctx, Vec2, Vec3, Vec4, Array, Text` (the engine
  injects these too); a script is plain Python -- `import math` and the stdlib work.
```

**Correction inside the bare-list clause.** The 054-validated trap guard is restored, but not
verbatim: the current prompt says "A bare `[x,y]` also coerces for a vec, **NOT for an array**", and
that second half is false against `uniform_coerce.coerce_array`, which accepts a bare exact-length
flat list. The real trap is a NESTED bare list (rows), which `coerce_array`'s `all(is_number(v))`
rejects and which `Array(...)` exists to flatten. The block states the true form. Shipping a
code-contradicting clause inside a block whose whole claim is "generated from the code" would defeat
the feature.

**Where it plugs in.** `CopilotContext` gains `script_api: str`; `build_context` fills it;
`_context_block` renders it **after the EXAMPLE LIBRARY block and before CONVENTIONS** — appended
rather than inserted, so the GLSL cluster (project map → lib catalogue → example library) stays
contiguous and the Python block does not split it. Same RARE volatility rank, no new tier.

It is **unconditional** (emitted even for projects with no scripts) for simplicity, not for cache
safety: the RARE block already re-renders on any create/delete/rename/compile-flip, so a
"scripts exist" flip would not be the thing that busts it. The earlier draft's cache justification was
wrong and is withdrawn.

**Risk:** shape errors return (a bare tuple where a `Vec3` is wanted, a nested list into `Array`).
Two backstops make this recoverable rather than terminal: `coerce_one` raises a per-KEY `ScriptError`
carrying `uniform_shape_hint`, and `write_script`'s probe verdict surfaces it in the same step.

### D4 — The TEXT const-array rule: deliberate, then decide

The maintainer explicitly framed this as a brainstorm, not a deletion: "она слишком специфично чтобы
находиться вот прям тут в промпте… по такой логике можно вообще всю хуйню мира сюда занести… Не надо
удалять… я просто обращаю на это внимание, чтобы побрейнштормить этот момент." So it gets the D6
treatment: state the criterion, weigh the options, then lock.

**The rule today** (in EDITING): "TEXT content: NEVER a const array in source — declare
`uniform uint u_text[64];` and `set_uniform("u_text", "Hello\nWorld")` (converted to codepoints;
stays user-editable)."

**Provenance correction.** The earlier draft cited `conventions.md ## Known quirks` as the fact's
home. The primary source is the generated file itself,
`shaderbox/resources/shader_lib/text/glyphs.glsl`, in the comment above `SBT_SPANS`: "a
dynamically-indexed const array (function-local OR global) is demoted to per-thread local memory on
NVIDIA (~100x slower text stack)". `conventions.md` restates it; the GLSL comment is the measurement.

**Criterion** (the same one D5 uses): a prompt line earns its place only if it is cross-tool AND not
expressible in any single tool schema.

**Option 1 — keep it in EDITING.** Costs 175 chars in the highest-attention section. Against: it is a
GLSL *domain* fact, not editing *mechanics*; leaving it there is what invites the next domain fact to
land beside it ("всю хуйню мира").

**Option 2 — move it to the RARE `_CONVENTIONS` block.** 210 chars in the cacheable prefix, sitting
with the other GLSL domain rules (version header, SB_ layering, engine uniforms, aspect, uv-y) —
which is exactly the shape of rule it is. Costs one line of RARE growth.

**Option 3 — rely on the schemas.** Partial coverage, verified: `_SetUniformArgs.value` already says
"a STRING for a uint text array … To change displayed TEXT, set the text uniform with a string — do
NOT edit the source", and `edit_hints.py` echoes "for TEXT skip const arrays entirely: uniform uint
u_text[64] + set_uniform" — but only inside the array-initializer-mismatch hint, i.e. AFTER the model
has already written the const array and hit a compile error. So the schema covers the *set* half and
the engine covers the *recovery*; neither covers the *authoring* choice made before any error.

**Locked: option 2.** The authoring half is uncovered, the failure it prevents is a ~100x performance
cliff that no compile error reports (the shader compiles fine and renders slowly), and the RARE
conventions block is the block whose stated job is "GLSL domain rules". The wording tightens to name
the measured reason, so it reads as a fact rather than a taboo:

```
- TEXT content: a caption is `uniform uint u_text[64];` fed by set_uniform -- NEVER a const array in
  source (a dynamically indexed const array is demoted to per-thread local memory on NVIDIA, ~100x
  slower).
```

The *setting* half ("pass a plain string, it converts to codepoints, don't edit the source") is cut
from the prompt — it is verbatim in `_SetUniformArgs.value`.

**Risk:** a const glyph array lands and nothing errors (a silent perf cliff, invisible to the facts
line). Gate: scenario 13 if its console renders captions; else the marker unit test only — this is the
one D-item whose regression the dogfood cannot see, which is itself an argument for keeping the rule
somewhere rather than dropping it.

### D5 — Cut what the tool schemas already say — after auditing that the schemas are RIGHT

**The audit rule (new, and load-bearing).** Overlap resolves in the schema's favour ONLY after
confirming the schema is correct. Deleting a prompt line that was silently *correcting* a wrong
description would ship the wrong text as the sole survivor. Four eager descriptions are wrong TODAY,
and the prompt is the thing correcting them:

| description | wrong claim | truth |
|---|---|---|
| `_RENDER_IMAGE_DESC` | "Returns the file path + the actual size." | The path goes to `payload`; the model-facing `msg` says "you can't see the result and don't have the path". |
| `_RENDER_VIDEO_DESC` | "Returns the path + duration." | Same — path is payload-only. |
| `_PUBLISH_TELEGRAM_DESC` | "Returns the pack URL." | URL is payload-only; `msg` says "you don't have the link". |
| `_PUBLISH_YOUTUBE_DESC` | "Returns the Studio URL." | Same. |

All four are corrected in wave B to say the result is a button shown to the USER and that the model
gets neither path nor URL — which is what makes cutting the prompt's "you never get the file
path/URL" clause safe. Two further fixes in the same wave:

- **`_PUBLISH_YOUTUBE_DESC` — "YouTube must be connected in Settings."** This is *the deflection the
  prompt rule forbids*, shipped inside the tool the rule is about, and it contradicts
  `youtube_precheck` ("YOU connect it: call set_youtube_credentials … do NOT just send them to
  Settings"). Corrected to name `set_youtube_credentials` as the agent's own affordance.
- **`_RENDER_VIDEO_DESC`** gains the live-source clause `_RENDER_IMAGE_DESC` already has ("you render
  the live source — land your edits first"), so that rule can leave the prompt without a gap. (The
  earlier draft both listed this clause as cut AND kept it in the retained block; resolved here in
  favour of the schema.)

**The cut table.** Every row is covered by an EAGER description (always in `tools=`) unless marked
LAZY, in which case the coverage is spelled out:

| prompt line (cut) | covered by |
|---|---|
| `set_uniform(name, value)`: number / vector / uint[] TEXT as a string | `_SET_UNIFORM_DESC` + `_SetUniformArgs.value` (eager; richer — it also states the script-driven reject) |
| `create_node(name)`: empty source = starter; full source compiles; `switch_to=false` | `_CREATE_NODE_DESC` + all four arg descriptions (eager) |
| `delete_node`: user confirms; "user declined" -> stop + explain; trash-recoverable | `_DELETE_NODE_DESC` (eager, near-verbatim) |
| `switch_node(node)` makes a node CURRENT (no-target edits and publish act on it) | `_SWITCH_NODE_DESC` (eager, verbatim) |
| `read_lib(names)` returns full bodies | `_READ_LIB_DESC` (eager) |
| `grep(query)`: token across nodes + lib, origin-labeled file:line | `_GREP_DESC` (eager, verbatim) |
| render `shape` vocabulary (`native` / `short_*` / `wide_*`, never raw pixels) | `_SHAPE_DESC` on all three render/publish arg models (eager) |
| render_video "ALWAYS from t=0" / "briefly pauses" / "renders the LIVE source" | `_RENDER_VIDEO_DESC` + `_RENDER_IMAGE_DESC` (eager) — the live-source clause only after the wave-B fix above. ("user confirms" is NOT cut: it survives compressed as the retained section header "RENDER & PUBLISH (each user-confirmed)".) |
| "a button is shown to the user" (the tool-result FACT half only) | the four corrected descriptions (eager) — **valid only post-fix**. The REPLY-BEHAVIOUR half ("you never get the path/URL — never invent one") is NOT cut; it survives in the block below. |
| `rename_node` / `duplicate_node` / `set_canvas_size` / `import_node` one-liners | LAZY: discovery via their `catalog_summary` rows inside `load_tools`'s description; full description at load time |
| MEDIA/TEXTURES paragraph | LAZY: `bind_media`/`unbind_media` are NOT in `tools=` until loaded, so coverage is (a) their `catalog_summary` rows for discovery, (b) the **retained** cross-tool bullet "declare `uniform sampler2D u_tex;` FIRST, then bind_media" in the prompt, (c) the full description once loaded. The paragraph's other clauses (picker semantics, the `<- (WxH, image)` row) rest on (a)+(c). |
| Telegram/YouTube step-by-step (bot Start, pack list/create/select/delete) | LAZY: the six/one `catalog_summary` rows; the full descriptions at load time; and — at the moment it matters — the `telegram_precheck` / `youtube_precheck` handoff messages, which are `precheck` callables ON `publish_telegram`/`publish_youtube` (not tools of their own) and fire before the gate when creds/pack are missing. The **retained** "never deflect to Settings / never invent integration state" bullet stays in the prompt because no precheck fires on a bare "connect my telegram" ask. |
| library "call by name, auto-resolves, no #include" | TRIPLICATED: the RARE catalogue header AND `_CONVENTIONS` bullet 2 both say it |
| ADDRESSING: "empty = current, NEVER means all"; the `lib:` prefix meaning | `_ReadShaderArgs.nodes`, `_TARGET_DESC` (eager) |

**What survives** — only what no single schema can express:

```
NODES, LIBRARY, MEDIA (what the tool schemas cannot say)
- Cross-tool order: a new texture input is `uniform sampler2D u_tex;` via edit_shader FIRST, then
  bind_media; a new script-driven uniform is declared in the SHADER first, then driven.
- The library auto-resolves by name -- a lib file has NO standalone compile, so confirm a lib edit
  by touching a consumer node and reading its errors. `write_shader` to a new `lib:` address creates
  the file.
```

```
RENDER & PUBLISH (each user-confirmed)
- **PUBLISH acts on the CURRENT node, takes NO node arg, is EXTERNAL + IRREVERSIBLE. Confirm the
  `current` map mark is the node the user named; `switch_node` first if not. Never skip this.**
- You never get the file path/URL -- the app shows the user a "Reveal render" / "Open in ..."
  button; say it is ready, never invent a path.
```

The path/URL bullet is retained deliberately even after the schema fix: the schema states the FACT
(the model gets no path), the prompt states the REPLY BEHAVIOUR (never invent one, point at the
button). 373 chars total.

```
TELEGRAM + YOUTUBE -- YOUR capabilities (lazy tools: `load_tools` first): drive the whole setup
yourself, never deflect the user to Settings. Integration state is NOT in your context -- never
invent it; report it only from a tool result.
```

```
ADDRESSING (`target`/`node`/`nodes`)
- Copy a node id EXACTLY from the map -- an unknown id is an error, never invent one. `example:` is
  READ-ONLY (read/grep to inspect, `create_node(example=...)` to instantiate).
- In replies, call nodes by NAME, never by id.
```

The PROJECT-MAP/CATALOGUE shortcut bullet stays where it already is, in USING TOOLS (unchanged).

**Risks.** (a) Lazy-tool discoverability now rests on `load_tools`'s catalogue — falsifier: the media
micro-probe below. (b) YouTube/Telegram deflection — falsifier: the two credential-cleared probes.
(c) Publish targeting — the CURRENT-node rule is retained verbatim (a locked `conventions.md`
decision), so only schema-carried mechanics were cut.

### D6 — `#version 460 core` STAYS in the prompt (investigated; baking it is not cheap)

M11 asks for anything bakeable to be baked. A seam exists: `Node.compile()` is the single funnel, it
calls `shader_lib/resolver.py::resolve_usage(self.source, index)`, and `parser.split_root_header`
already isolates the `#version`/`#extension`/`#pragma` header — injecting a missing header there is
~10 lines. It is still the wrong change:

1. **The fast path breaks.** `resolve_usage` short-circuits with `flattened = root.text` when the
   shader references no `SB_*`. Prepending a line ahead of it shifts every driver error by one unless
   that path also emits `#line 1 0` — so the "flattened IS the root" invariant the error mapping
   leans on dies for every lib-free shader.
2. **Blast radius far outside the copilot.** `help_content.py` ships the contract to users ("Three
   things are fixed: the `#version` line (required — nothing is injected for you)"); every shipped
   example, the `create_node` starter, `projects/dev/` nodes and `tests/test_examples_resolve.py`
   assume it. A header-less node on disk also stops being a valid standalone `.glsl`, which
   `import_node` round-trips through the user's filesystem.
3. **No failure to fix.** The model reliably COPIES a fixed header; no dogfood report or trace shows a
   missing-`#version` compile failure. A mechanism that removes no observed failure class is the
   mirror of the guard-that-does-not-earn-its-place rule.
4. **The payoff is ~20 chars.** The `_CONVENTIONS` bullet must stay regardless for
   `vs_uv`/`fs_color`/no-`precision`.

Locked: keep the line, unchanged. Revisit trigger: a dogfood run where a compile fails on a missing or
wrong `#version`, or a product decision to let users author header-less shaders (then it is a feature
with its own spec, not a prompt saving).

### D7 — The render-facts legend: trim to the non-self-glossing residue, then splice (Q1 = C)

**Q1 is locked to option C** by the moderator: the legend rides the FIRST facts-carrying tool result
of a turn. But *which* legend is a second question, and it changes the cost by 3x.

**Pre-action vs post-action.** Every clause of the old 1 629-char bullet is classified. A **post-action
gloss** decodes a line the model is looking at; it may travel with that line. A **pre-action rule**
governs what the model does before or instead of an action; it must stay in STATIC, because on the
turn where the user says "make it warmer" there may be no facts line yet. Three clauses are
pre-action and carry commit-recorded provenance — moving them would regress the fix that added them:

- the relative-colour verify (618af96, "copilot: ink mean-rgb + warm/cool in render facts"),
- blank/cold `t=0` with `ANIMATES` means it DEVELOPS, do NOT re-edit (c39caac, "two-frame probe +
  STATIC/ANIMATES motion verdict"),
- `changed NOTHING on screen` means do NOT re-apply the same edit (2ff249f, "no-op detection").

All three stay in STATIC.

**Option D (adopted): the facts line already self-glosses most of it.** Read against
`edit_hints.py::render_facts` and `backend.py::_render_facts_for`, the emitted text already carries:
`(y=0 bottom)` inline; the whole FLAT verdict ("FLAT — one uniform color rgba(...), max pixel
deviation N/1020 (a blank OR a full-screen fill) NOTHING is visible: do not describe a scene — fix
it, or report the flat frame and ask."); both motion verdicts expanded ("STATIC (unchanged from t=0
to t=1.5s)" / "ANIMATES (the frame changes over time)"); and the no-op line with its full cause list
("dead code, the wrong node/target, a value a script overrides, or a change only visible between t=0
and t=1.5s"). Re-explaining those in a legend is paying twice. The genuinely non-obvious residue is
four terms — what `ink %` counts, what `bbox` is measured in, what `ink mean rgb` averages over, and
what the `luma 0-9` grid is — plus the one diagnostic that only makes sense next to the measured
motion word (a `STATIC` you did not intend = missing `u_time` wiring, not a tuning problem). As
landed (`prompt.py::_RENDER_FACTS_LEGEND`):

```
[how to read the line above] ink % = share of pixels differing from the corner-sampled
background (alpha counts, so a shape on transparency is ink); bbox = where that ink sits, in vs_uv
(hugging an edge = off-center; x 0.00-1.00 = touching both edges); ink mean rgb = the alpha-weighted
mean colour of the DRAWN region only -- the ONLY colour signal you get; luma 0-9 = a 3x3 brightness
grid, top row first; motion: STATIC when you meant it to move = the u_time wiring is missing, not a
tuning issue.
```

**500 chars (~125 tok)** — versus 1 240 for the full legend and 1 629 for the current bullet.

**Cost.** Under C with the 500-char legend: ~125 tokens, emitted at most once per turn, re-sent on each
remaining iteration of that turn (the within-turn tool tail is re-sent but discarded at the turn
boundary — the NL-only history invariant is untouched), and **zero on turns that never render** (a
greeting, a pure read, a question). Option A (keep in STATIC, cached) pays ~125 tok in the cached
prefix on every request of every turn. At observed turn shapes the two are within noise of each
other; C is chosen for the attention clustering the review asked for ("прямо рядом с рендер-фактами…
чтобы attention был скластеризован"), not for the token saving.

**Mechanism.** `prompt.py` owns `_RENDER_FACTS_LEGEND` (the text stays where prompt text lives).
`agent.py`'s run loop holds a per-turn `legend_emitted: bool`; after a tool result returns, if its
message contains the facts marker and the flag is unset, the legend is appended to that message and
the flag is set. **`_forced_reply_facts` routes through the same flag** — a forced-end reply whose
probe is the turn's first facts line must carry the legend, and must not repeat it if an earlier
result already did.

**Pinned by test** (a falsifier, not a smoke check): a two-result turn where both results carry facts
— assert the legend is in the first message and absent from the second; and a turn whose only facts
come from the forced-reply path — assert the legend rides that one. Cut the flag and both go red.

**The FEEDBACK section after** (1 389 chars, was 2 472):

```
FEEDBACK (what you can see)
- The compiler: source-mapped errors, or clean.
- Render facts: a clean mutation's result carries one measured line off a real probe frame, with the
  legend for reading it beside it. It MEASURES, it does not judge.
- You never SEE your render -- the facts line is your ONLY signal about it. Never claim how the
  result LOOKS (pretty, clean, striking) or that a visual goal is achieved beyond what the numbers
  show; state what you changed and what the measurements say, and let the USER judge the look.
- A relative COLOUR ask ("warmer", "bluer", "more saturated") is verified against the facts line's
  `ink mean rgb`, not against your intention -- read it back after the edit.
- `motion: ANIMATES` on a blank/cold t=0 means the effect DEVELOPS over time -- NOT a failed edit;
  do NOT re-edit it. `changed NOTHING on screen` means your mutation had ZERO visual effect -- do NOT
  re-apply the same edit; find the cause.
- Uniform values: check the working-set `uniforms:` row before claiming a value changed. For a
  relative ask ("brighter", "slower"): read the current value there, adjust, let the user confirm.
- A user report of black screen / "no change": treat it as real (clean compile != correct) -- but
  if your render facts or the source CONTRADICT the report, say what the facts show and ASK;
  don't silently re-edit against your own evidence.
```

### D8 — Size

The EDITING section after (1 109 chars, was 2 001) — quoted so every changed section's after-text is
verifiable:

```
EDITING
- `edit_shader` vs `write_shader`: edit_shader for ANY localized change; write_shader only for a
  genuine whole-file replacement, and only while the file is small-to-medium (roughly <=150 lines).
  Past that a full rewrite burns your ENTIRE reply-token budget and can TRUNCATE mid-file (a wasted
  step that lands nothing) -- change just the region instead. Max ONE write_shader per file per step
  (a second is rejected). The script pair (`edit_script`/`write_script`) splits the same way.
- Fix the COMPILE first -- never tune values while it is broken. N broken edits in a row -> the
  engine restores the last clean state ("EDIT UNDONE"): re-read the working set, then rewrite the
  whole block in ONE edit. An edit that returns the file to an earlier state gets an oscillation
  NOTE -- stop and reason.
- Edit SOURCE for logic or to reshape a uniform; `set_uniform` to change a live VALUE (never re-edit
  the number in source). A NEW scalar/vec uniform gets an inline default (`uniform float u_glow =
  0.4;`) which seeds the user's control -- no set_uniform needed; arrays cannot init inline.
```

| section | before | after | delta | wave |
|---|---:|---:|---:|---|
| (preamble) | 325 | 325 | 0 | — |
| WORKING SET | 652 | 652 | 0 | — |
| EDITING | 2 001 | 1 109 | -892 | B |
| FEEDBACK | 2 472 | 1 389 | -1 083 | C |
| VALUES/NODES/LIBRARY | 1 733 | 454 | -1 279 | B |
| SCRIPTING | 3 196 | 1 785 | -1 411 | A |
| VISUAL CRAFT | 4 396 | 4 396 | 0 | — |
| RENDER & PUBLISH | 1 072 | 373 | -699 | B |
| TELEGRAM + YOUTUBE | 807 | 237 | -570 | B |
| USING TOOLS | 1 067 | 1 067 | 0 | — |
| ADDRESSING | 407 | 262 | -145 | B |
| THE SANDBOX | 523 | 523 | 0 | — |
| HOW TO WORK | 1 844 | 1 844 | 0 | — |
| **`_SYSTEM_PROMPT`** (sections + 12 separators) | **20 507** | **14 435** | **-6 072 (-29.6%)** | |
| `_CONVENTIONS` (RARE) | 1 394 | 1 605 | +211 (D4) | B |
| `SCRIPT API` (RARE, new) | 0 | 1 214 | +1 214 | B |
| **STATIC + RARE** | **21 901** | **17 254** | **-4 647 (-21.2%)** | |
| legend (PER_TURN, at most once per turn) | 0 | 500 | +500 on a turn that renders | C |

Per wave: A 20 507 -> 19 096; B -> 15 511 (+1 425 RARE); C -> 14 435.
Token estimate at the repo's `_CHARS_PER_TOKEN = 4`: STATIC 5 126 -> 3 607 tok.

**Advisory ceiling, not a gate.** ~15 000 chars post-wave-C is the target. An overshoot is allowed but
must be justified in that wave's commit message — a *silent* overshoot is the failure mode this line
exists to catch, not a hard boundary that would push a genuinely-needed line out.

### D9 — Prompt-marker tests move with the prompt

`tests/test_craft_prompt.py::test_scripting_section_teaches_physics_via_script` asserts
`"PHYSICS" in p and "Verlet" in p and "ARRAY uniform" in p`. Post-refactor "PHYSICS" survives only in
VISUAL CRAFT, so the test would stay green while the section it names was gutted — a test that passes
whether or not the bug exists. Rewritten to assert the watershed on the section that owns it:

- positive: `"depends on the PREVIOUS frame"`, `"PURE FUNCTION OF TIME"`, `"needs NO script"`,
- retained: `"Verlet"`, `"ARRAY uniform"`,
- negative: the deleted table's phrase `"VALUES THAT CHANGE"` must be ABSENT,
- and the same watershed markers asserted on `_WRITE_SCRIPT_DESC` (the D1a half), with `"ANIMATION"`
  absent from it.

New tests: (a) `script_api_summary()` contains every public name from `vars(_Vec)|vars(Vec2)|
vars(Vec3)|vars(Vec4)` plus the dunder allowlist, and every `EngineContext`/`MouseState` field —
falsifier: add a fake public method to `Vec3` inside the test and assert the check goes red;
(b) `_CTX_GLOSS` keys equal the dataclass field names exactly; (c) every type name `_stub_kind` can
return is a key in the gloss map; (d) the wire, not the definition — `_context_block(build_context(…))`
contains `SCRIPT API`; (e) `import shaderbox.scripting.api_doc` pulls in no `moderngl`;
(f) D2 item 4 — a lib-originated compile error renders a `lib:` address, a node error the node's short
id; (g) D7's legend-once-per-turn splice (above).

**Style note:** the existing file puts `from shaderbox.copilot.prompt_context import _CONVENTIONS`
inside three test function bodies, against the repo's imports-at-module-top rule. The rewrite does not
propagate that pattern; the new assertions import at module top, and the three existing ones are
hoisted in the same wave (one line each).

---

## What does NOT change

- **VISUAL CRAFT, whole** (4 396 chars — after the cut, 30% of the prompt). Feature 054/056 lab-mined
  content with its own marker tests; not in this review's scope. See the M9 trigger.
- **WORKING SET, USING TOOLS, THE SANDBOX, HOW TO WORK, the preamble** — no duplication found; every
  line is a cross-cutting behavioural rule or a negative-space fact no schema states.
- **The block/volatility machinery** (`PromptBlock` / `Volatility` / `build_prompt` /
  `build_messages`), the NL-only history model, `render_working_set`'s contract, the trim reserve.
- **The facts pipeline itself** — `render_facts`, `_render_facts_for`, the motion verdict, no-op
  detection, `probe_render`. Only the prose describing it moves. (Whether the facts help at all is
  M3 / wave 4.)
- **`_stub_kind`'s dispatch and example generation**, and the stub's emitted BODY (D1b, D3).
- **`_CONVENTIONS`' existing six bullets** (version/vs_uv/fs_color; SB_ layering; engine-driven
  uniforms; the aspect rule; the uv-y rule; helper granularity) — the aspect and uv-y rules are
  A/B-validated and test-pinned. D4 only ADDS.

## Files touched

- `shaderbox/copilot/prompt.py` — `_SYSTEM_PROMPT` (D1/D2/D4/D5/D7); `_context_block` renders
  `script_api`; `_render_working_set_member`'s SCRIPT header path strip; `_RENDER_FACTS_LEGEND`.
- `shaderbox/copilot/prompt_context.py` — `CopilotContext.script_api`; `build_context`;
  `_CONVENTIONS` gains the TEXT bullet.
- `shaderbox/scripting/api_doc.py` — NEW: `script_api_summary`, `_CTX_GLOSS`, the type-name gloss map.
- `shaderbox/scripting/engine.py` — `_UPDATE_DOC`'s stale Vec paragraph out, the state line in;
  `_stub_kind` unchanged.
- `shaderbox/copilot/tools/script.py` — `_READ_SCRIPT_DESC`, `_WRITE_SCRIPT_DESC` (wave A);
  `_fmt_errors` gains the shared path->label helper (D2 item 4). `_EDIT_SCRIPT_DESC` unchanged.
- `shaderbox/copilot/tools/publish.py` — the four wrong return claims, the YouTube-Settings
  deflection, render_video's live-source clause (wave B).
- `shaderbox/copilot/backend.py` — the shared path->label render helper + its use in
  `_force_restore`'s error join (D2 item 4). `_to_error_infos`, `_cross_file_note` and
  `_edit_error_hints` are UNCHANGED — the info object keeps the absolute path.
- `shaderbox/copilot/tools/node_ops.py` — the two inline error joins (duplicate_node, import_node)
  route through the same helper (D2 item 4).
- `shaderbox/copilot/agent.py` — the per-turn legend flag + splice, `_forced_reply_facts` routing
  (wave C).
- `tests/test_craft_prompt.py` (rewrite + the three body-import hoists), new
  `tests/test_script_api_doc.py`, plus cases in the copilot loop/error-path tests for D2 item 4 and
  D7's splice.
- `ai_docs/roadmap.md` (row + banner). **`ai_docs/todo.md` is NOT touched** (frozen drain-only).

---

## Validation plan

`make check` + `make test` are the floor; the behavioural gate is the dogfood echelon driven per the
`/dogfood` skill.

**Results:** `02_controls.md` (the controls) + `03_wave_a_gates.md` (all three wave gates + the four
micro-probes).

### Control discipline — which scenarios can gate anything

**Only 04, 05 and 08 have `effort=none` baselines** (`057…/05_reasoning_none.md`: 04 one-shot PASS,
period exactly 2.0 s, $0.007; 05 one-shot PASS, parabolic arcs + rest, $0.027; 08 PASS with one
correction, $0.094). Every other scenario's recorded baseline was produced at `effort=minimal`, which
that ledger shows is a **different actor** — the flag was silently ignored on tool-bearing requests
and reasoning consumed ~100% of output tokens. Comparing a post-refactor `effort=none` run against an
`effort=minimal` baseline measures the flag, not the prompt.

So, **before wave A lands**, run controls on the CURRENT prompt at `effort=none`:

1. **de-hinted 05** (new variant — see below): the variant itself is new, so it has no baseline.
2. **10 (pong)**: baseline was `effort=minimal`.
3. **13 (final exam)**: baseline was `effort=minimal`.
4. **03 (static comp)**: baseline was `effort=minimal`, and wave C's honesty gate cites it. It is a
   3-message static-composition run — the cheapest control in the set, so it is added rather than
   demoting the gate row to an observation.

04, 05 and 08 need no new control. **A gate row may only cite a scenario that has an `effort=none`
control.** Any scenario without one is an observation, not a gate — stated explicitly so a later
reader does not promote it.

### The de-hinted 05 variant

Scenario 05's verbatim opening already says "The physics must live in the python script — integrate
gravity and velocity there; do not fake it with GLSL time math." That names the tool, so it cannot
falsify an over-correction toward "never script": the model can comply by instruction without holding
the rule. The gate therefore also drives a **de-hinted variant** — the same message truncated at
"…until it comes to rest on the floor." — and requires a script from the rule alone. Both verdicts go
in the report; the de-hinted one is the real D1 gate.

**Disagreement, stated.** The consolidated review says 10 and 13 should join the *over-correction*
gates because "their openings don't name a script". They do: 10 says "All game logic lives in the
python script; the shader only draws the state it is given", and 13 says "Run the boat's logic in the
node's script". Both are hinted exactly as 05 is, so neither is a pure over-correction gate. They ARE
added — because they are the only **mixed-routing** gates available: 13 pairs an explicitly-scripted
depth state machine with a sonar sweep, an afterglow and a tumbling mine that are pure functions of
time; 10 pairs a scored state machine with nothing time-pure at all. That discrimination is exactly
what the new rule has to make and what the old table could not even express. Their rows below are
worded accordingly rather than as over-correction rows.

### Gate table

| change | wave | gate (control status) | regression signature |
|---|---|---|---|
| D1 over-correction | A | **de-hinted 05** (control run first) | gravity/velocity faked with `u_time` trig; "physics is in `script.py`" fails |
| D1 under-correction | A | **04** (baselined) | a `script.py` appears for a 2 s orbit; 04's "no python script created" fails |
| D1 under-correction, volume | A | **08** (baselined) | any of the 9 time-pure cells routed to a script |
| D1 mixed routing | A | **13** (control run first) | the sweep/afterglow/mine pushed into the script, or the depth machine left in GLSL |
| D1 mixed routing, state-heavy | A | **10** (control run first) | ball/paddle/score state faked in GLSL, or the run stalls trying to hold state shader-side |
| D1a description edits | A | all of the above (they ride the same runs) | — |
| D2 path removal | A | trace grep on the wave-A runs | `/` or `nodes/` inside a script-tool argument |
| D2 error-path label | A | unit test (f) | a lib error that no longer names its file |
| D3 generated API | B | **13** (control) + the shape unit tests | per-key coercion errors in a write_script verdict; a `Vec` used as a bare tuple |
| D4 TEXT rule | B | **13** if its console renders captions; else the marker test only | a `const uint[]` glyph table in the emitted GLSL (invisible to the facts line — see D4's risk) |
| D5 lazy-tool discoverability | B | media micro-probe ("put this image in the shader") | no `load_tools(["bind_media"])`; the agent claims it cannot bind media |
| D5 Telegram deflection | B | credential-cleared Telegram probe | the reply points at Settings instead of calling `set_telegram_token` |
| D5 YouTube deflection | B | credential-cleared YouTube probe | the reply points at Settings instead of calling `set_youtube_credentials` |
| D5 publish targeting | B | publish probe with a non-current node named | publish fires without a preceding `switch_node` |
| D7 legend splice | C | unit test (g) + **04** (baselined) / **03** (control run first) honesty axis | a second facts line repeats the legend; a mis-stated FLAT/ANIMATES reading |

The four micro-probes are single-turn harness drives, not full scenarios; they need no baseline
because their assertion is on the tool call made, not on a visual outcome.

**Order.** Controls (de-hinted 05, 10, 13, 03) -> wave A -> its gate -> wave B -> its gate -> wave C ->
its gate. Only 03's control is needed before wave C, so it can be run any time before then. D1 is the only change that can regress a base capability outright; the later cuts are cheap to
re-land if the watershed has to be re-tuned.

---

## Out of scope (each with a trigger)

- **VISUAL CRAFT compression** and the wider routing-rules overfit survey (USING TOOLS / HOW TO WORK /
  VISUAL CRAFT). **Trigger:** the M9 row above.
- **The render-facts A/B** (do the facts help at all; the motion-verdict knobs). **Trigger:** the M3
  row above — after 059 lands and the echelon is re-baselined at `effort=none`.
- **Baking `#version`** — D6's trigger.
- **Tool-description rewrites beyond the six named** (four wrong return claims, the YouTube
  deflection, render_video's live-source clause, plus the two script descriptions in wave A).
  **Trigger:** a cut line turns out not to be covered because a description is **thinner than, or
  wrong about,** what this spec's table claims — then fix the description (the schema is the canonical
  home), do not restore the prompt line.
- **A generated GLSL type summary** (the `SB_*` catalogue is already generated; a GLSL-side
  equivalent of D3 is not proposed). **Trigger:** a second stale-GLSL-prose incident of the `Vec` kind.
- **Making the SCRIPT API block conditional on the project having scripts.** **Trigger:** a measured
  RARE-prefix cost problem.
- **Re-tiering `_CONVENTIONS` into STATIC.** **Trigger:** a cache measurement showing the RARE block's
  stable half is worth splitting out.

## Open questions

None blocking. Q1 (legend placement) and Q2 (generator style) were answered by the moderator in
`00_maintainer_feedback.md` and are locked here as **D7 = option C** (carrying option D's trimmed
legend as the payload) and **D3 = hybrid** (names/signatures from code via `vars()` + a dunder
allowlist, authored glosses pinned by coverage tests). D4, previously a candidate open question, is
locked to option 2 with the argument stated in full; if the maintainer rejects the criterion, the
alternative is option 3 (drop the rule and accept an invisible perf-cliff regression class) — which is
why the argument is spelled out rather than deferred.
