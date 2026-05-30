# 020 Copilot Agent — the GLSL domain angle

> Research report. Companion to `00_grounding.md` (threading/seams) and the other `NN_*.md`
> idea reports. **This file is the DOMAIN angle: what a GLSL-shader copilot must be good at, and
> the in-process compile-feedback loop it operates inside.** All symbol/template citations re-read
> from source 2026-05-29; not recalled.

---

## 0. The one-paragraph thesis

ShaderBox's copilot is not "a chat box that writes GLSL." Its differentiator is that it sits
**inside the compiler.** `Node.compile()` runs the user's shader through the real moderngl/driver
pipeline in-process, keeps the last-good program on failure, and surfaces **source-mapped**
diagnostics (`compile_unit.errors: list[ShaderError]`, each `(path, line, message)`). A normal
coding agent writes code blind and hopes; this one can **edit → recompile → read the real driver
error back at the right line → self-correct**, on a millisecond loop, with zero network round-trip
for the verification step. Everything else (lib helpers, uniforms, templates) is in service of that
loop. The domain MVP is therefore "explain + edit the current shader, with the compile loop wired
in" — and almost everything that "sounds easy" is hard precisely where it needs to *see the
rendered pixels*, which the codebase does not give the agent.

---

## 1. Core use-case tiers (grounds the tool catalog from the domain side)

Ordered by escalating capability + escalating blast radius. Each tier is a natural tool-permission
boundary; the MVP cut (§6) draws its line between tiers 2 and 3.

### Tier 0 — Read / explain (zero mutation, zero GL)
- "What does this shader do?" → read the current node's `shader.frag.glsl`, explain the algorithm.
- "Why is it black / red / pulsing?" → read source + read `compile_unit.errors` + read current
  `uniform_values`; reason about it. (NOTE: it can read the *error*, not the *pixels* — see §7.)
- "What uniforms can I tweak here, and what do they do?" → enumerate `get_active_uniforms()` + the
  `UIUniform` shapes + current values.
- "What's `SB_fbm`?" → look it up in the `ShaderLibIndex`, return its signature + `///` doc + body.

These need only **read access to disk + in-memory state**. No marshalling, no GL. Pure win, lowest
risk. This tier alone is genuinely useful (a beginner staring at a Shadertoy-style SDF wants an
explanation more than an edit).

### Tier 1 — Edit the current shader (the core loop; GL via the free hot-reload lunch)
- "Make the background pulse." → add `sin(u_time)`-driven brightness.
- "Add fbm noise to the color." → call `SB_fbm(uv)` (lib auto-resolves) or write inline fbm.
- "Make the circle bigger / softer." → tweak the SDF math.
- **"Fix this compile error."** → read `compile_unit.errors`, edit the offending line, recompile,
  confirm clean. THE flagship loop (§2).
- "Port this Shadertoy snippet to ShaderBox conventions." → rewrite `mainImage(out vec4
  fragColor, in vec2 fragCoord)` / `iTime` / `iResolution` / `gl_FragColor` into ShaderBox's
  `void main()` + `in vec2 vs_uv` + `out vec4 fs_color` + `u_time`/`u_resolution`/`u_aspect`
  contract (§5). This is a *huge* real use case — Shadertoy is the lingua franca and its contract
  differs from ShaderBox's in ~6 named ways the agent can mechanically map.

Mutation mechanism: **write the `.glsl` to disk; the existing mtime watcher recompiles on the main
thread next frame** (grounding §3 "the agent's free lunch"). No GL on the worker thread.

### Tier 2 — Manage uniforms (the live-tweakable surface; §4)
- "Add a color I can tweak." → declare `uniform vec3 u_color = vec3(...);`, and after recompile set
  its `input_type = "color"` + a default value.
- "Set the radius to 0.5." → `set_uniform_value(node, "u_radius", 0.5)`.
- "Make this a slider from 0..1." → drag input shape (already the only valid one for scalar) + set
  a value.

### Tier 3 — Compose with / author the lib (§3)
- "Use my noise helper." → just type `SB_perlin_noise_3(...)`; resolver splices it.
- "Show me what helpers I have." → search/list the `ShaderLibIndex`.
- "Extract this function into the lib." → export-from-selection (explicitly **deferred**, §3).

### Tier 4 — Create a node from scratch (orchestration; §6)
- "Make me a node that renders a rotating starfield." → pick template → replace shader → add
  uniforms → set values → it renders.

### Tier 5 — Render / export
- "Render this as a 3-second looping sticker." → drive `render_media` (GL-heavy; marshalled).
  Ties to the sticker seamless-loop deferral (§6).

**Permission gradient:** tiers 0–1 are read+single-file-write (low risk); 2 mutates live state; 3
touches cross-project lib files (highest blast radius — a lib edit fans out to every node that uses
it); 4–5 orchestrate multiple mutations + GL. A sane v1 ships 0–1, gates 2+ behind confirmation.

---

## 2. The compile-feedback loop — the crown jewel

### 2.1 What the engine already gives us (verified, `core.py`)

`Node.compile()` (core.py:225–276) is the whole gift:

1. It calls `resolve_usage(self.source, active_lib_index())` → `(flattened, sources, source_map,
   resolve_errors)`. The flattened text is what the driver actually sees (root + spliced lib
   preamble).
2. Resolver-domain failures (lib cycle, etc.) become **synthetic** `ShaderError`s appended to
   `unit.errors` — same type, same downstream path as driver errors (core.py:240–249).
3. The real compile is `self._gl.program(vertex_shader=self.vs_source,
   fragment_shader=unit.flattened)`. On `Exception` it captures `err = str(e)` and runs
   `parse_shader_errors(err, unit.source_map)` → `list[ShaderError]` (core.py:256–263).
4. **The invariant that makes the loop safe:** on failure it does **not** touch `self.program`. The
   previous valid program survives, so the preview stays bright (feature-013 invariant, core.py
   docstring lines 226–229). The agent can hammer the shader with broken intermediate edits and the
   user keeps seeing the last-good frame, not a black screen.

So the agent's "did my edit work?" oracle is exactly: `len(node.compile_unit.errors) == 0`. And the
"what broke + where" payload is `[(e.path, e.line, e.message) for e in node.compile_unit.errors]`.

### 2.2 The SourceMap remap — why the line numbers are trustworthy

This is the subtle part the agent benefits from *for free.* `parse_shader_errors`
(shader_errors.py:49–61) matches two driver formats:
- NVIDIA: `0(LINE) : error CXXXX: msg` (`_NVIDIA_ERROR_RE`)
- Mesa/Intel/AMD: `ERROR: 0:LINE: msg` (`_MESA_ERROR_RE`)

The **leading number is a file-id**, not always 0. The resolver, when it splices lib functions into
the preamble, emits `#line N M` directives (resolver.py:73–85): `#line <line_in_file+1> <file_id>`
before each lib function body, and `#line <header_end+1> 0` to restore the root's numbering for the
user's body. Both drivers honour `#line`, so the **line they report is already the source line**;
`SourceMap.resolve(file_id, line)` (shader_errors.py:34–38) just maps the file-id back to the right
`Path`. Net effect for the agent:

- An error in the user's own shader body → `ShaderError(path=<node>/shader.frag.glsl, line=<editor
  line>, ...)`. The agent edits that exact line.
- An error inside a spliced lib function → `ShaderError(path=<lib>/noise.glsl, line=<line in the lib
  file>, ...)`. **The agent is told the bug is in the LIB file, not in the user shader** — it must
  not "fix" the user's call site; it must open the lib file. This is a real correctness property:
  without the SourceMap, a driver error at flattened-line 412 would be meaningless to the agent
  because line 412 doesn't exist in any file the user edits.

Edge cases the agent must tolerate (all already handled by the parser, but the agent's prompt should
know about them):
- **Unparseable driver string** → `parse_shader_errors` falls back to a single `ShaderError(root,
  -1, <whole raw string>)` (shader_errors.py:59–60). `line == -1` means "not locatable" — the agent
  gets the raw text but no line; it must reason from the message, not jump to a line.
- **Unknown file-id** → `SourceMap.resolve` falls back to the root path (shader_errors.py:36–37). So
  a path is always present; worst case it's the root.

### 2.3 The tool design for the loop

Two tools, tightly coupled:

```
edit_shader(node_id, new_source: str) -> str
    # write the .glsl to disk (GL-free, worker-thread-safe).
    # returns "written; awaiting recompile" — does NOT itself compile.

get_compile_status(node_id) -> str
    # read node.compile_unit.errors AFTER the main-thread watcher has recompiled.
    # returns either "OK — compiled clean, N active uniforms: ..."
    # or the rendered error list: "ERROR shader.frag.glsl:20  'fs_color' undeclared identifier\n..."
```

The marshalling reality (grounding §2): `edit_shader` writes the file on the worker thread; the
**main-thread frame loop's `_reload_if_changed` recompiles on the next frame**; the agent then polls
`get_compile_status`. So the loop is: write → (yield a frame or two) → read errors → fix → repeat.
The agent does not call `compile()` itself (that's GL, main-thread only). This is the cleanest seam
and it falls out of machinery that already exists.

**Error-string rendering for the LLM.** Build a compact, line-anchored block, one error per line:
```
shader.frag.glsl:20  syntax error, unexpected '}' 
shader.frag.glsl:23  'u_colour' : undeclared identifier
lib/noise.glsl:41    'p' : redefinition
```
Path basename + 1-based line (convert from the 0-based `ShaderError.line` for human/LLM readability)
+ message. Include a couple of source lines of context around each error line (the agent has the
file; cheap to slice) — driver messages are terse, the surrounding code disambiguates.

### 2.4 Why this beats a normal coding agent

A generic coding agent's verification is: run tests / a linter, parse stdout, often over a shell +
network. ShaderBox's compiler is **in-process, sub-frame-fast, and the error is pre-mapped to the
file+line the agent edits.** There is no "set up a test harness" step — the harness is the app the
user is already running, and the verification target is the very artifact being edited. This is a
qualitatively tighter loop than a cloud coding agent gets.

### 2.5 ADVERSARIAL — is the compile loop overrated?

**The strongest case against:** GLSL driver errors are notoriously terse and driver-specific. Read
the actual parser: `parse_shader_errors` does **no message normalization** — it forwards
`match.group(3).strip()` verbatim. So the agent receives raw vendor text like `0(23) : error C1503:
undefined variable "u_colour"` (NVIDIA) or `ERROR: 0:23: 'u_colour' : undeclared identifier`
(Mesa). The *message* halves differ by vendor; cascade errors are common (one missing `;` produces
five downstream errors); and some real errors surface only at **link** time or as the unparseable
`line == -1` fallback (e.g. "too many uniforms", driver-specific limits). An agent could plausibly:
- chase a *downstream* cascade error instead of the root cause and thrash;
- mis-fix because the vendor message is ambiguous and it has no pixels to confirm the fix is
  *semantically* right (a fix that compiles but renders wrong is invisible to it — §7);
- loop forever on an error class it can't resolve (a driver resource limit, a `line == -1` blob).

**Resolution (all cheap, all in the agent loop, not the engine):**
1. **Hard iteration cap** (the consensus pattern from all three reference agents, grounding §4) —
   e.g. 4 compile attempts per user turn. On exhaustion, **surface to the user**: "I couldn't get
   this to compile; here's the last error and my best guess," and leave the last-good program intact
   (the engine already guarantees the preview didn't go black).
2. **Always fix the FIRST error, recompile, re-read** — never batch-fix the whole error list. The
   cascade collapses: fixing the root often clears the downstream five. This is exactly how a human
   does it and it's the natural shape of the read-error → fix-one → recompile loop.
3. **Feed the source-context window** (§2.3) so the LLM isn't reasoning from the bare vendor string.
4. **Treat `line == -1` and link-time errors as "surface to user" immediately** — don't burn
   iterations on a class the agent structurally can't localize.

Net: the loop is NOT overrated *for syntax/type/undeclared errors*, which are the overwhelming
majority of what a shader author hits and which the source-mapped line + the GLSL knowledge of a
strong model handle well. It IS weak for semantic correctness ("looks wrong but compiles") and for
driver-resource errors — and the cap + surface-to-user keeps those from becoming infinite loops. The
empirical truth here is unknown until tried; the cap makes the downside bounded regardless.

---

## 3. The SB_ shader library as the agent's toolbox (§3)

### 3.1 How the lib works (so the agent uses it right)

The mental model (feature 015, `shader_lib/index.py` docstring, resolver.py): **there is no
`#include`.** A node shader calls `SB_perlin_noise_3(x,y,z)` *as if it were a builtin*; at compile
time `resolve_usage` scans the user text for `\bSB_\w+\b` tokens (`parser.SB_IDENT_RE`), intersects
with the `ShaderLibIndex.functions`, transitively closes over the call graph, topo-sorts, and
splices the needed bodies into a preamble after the `#version` header. The agent's correct behavior
is therefore beautifully simple: **to use a lib helper, just emit the call.** No include line, no
import, no ceremony. If the agent writes `#include` it will pass through as text and produce a driver
error (feature 015 non-goal: "the user can't write them").

Hard constraints the agent must internalize (locked in feature 015):
- **Lib = pure functions only.** A lib function MUST NOT reference `u_time` / `u_resolution` / any
  uniform and expect it injected — "Lib files are pure-function helpers... the lib doesn't get to
  inject globals" (015 non-goals). If a lib function needs time, it takes it as a parameter. The
  agent must pass engine values *in*, never assume the lib declares them.
- **`SB_` prefix is the discoverability convention, not an engine rule.** The resolver only
  auto-resolves `SB_`-prefixed identifiers from user text. Non-prefixed lib functions are private
  helpers (callable only from inside the lib). So the agent surfaces/suggests only `SB_*` functions.
- **No macro dispatch.** `#define HASH SB_hash3` then calling `HASH(x)` in *another lib file* is
  undefined (the regex doesn't trace it) — feature 015 decision 5, the **macro-dispatch deferral.**
  The agent should not author lib code that dispatches functions through macros.
- **User can shadow.** If the user defines `SB_foo` in their own shader, the lib version is
  suppressed (`_collect_used_lib_names` subtracts `USER_FN_DEF_RE` matches, resolver.py:103–108).

### 3.2 How the agent DISCOVERS lib functions

The `ShaderLibIndex` is the catalog (`index.py`): `functions: dict[str, ShaderLibFunction]`, each
with `name`, `signature` (e.g. `"float SB_hash(vec2 p)"`), `body`, `file`, `line_in_file`, `calls`,
and `doc` (the `///` block above the function). This is *exactly* a tool-description payload.

Two design options:

**(A) Inject the catalog into the system prompt.** Render every `SB_*` function as `signature —
doc` into the prompt's "available helpers" section. Pros: zero tool round-trip, the model just knows
what's available and writes the call directly. Cons: scales poorly if the user has 200 helpers
(token cost); the catalog is volatile (mtime fan-out rebuilds the index) so it must live in the
*volatile tail* of the prompt for cache-friendliness (the ovelia least-volatile-first ordering,
grounding §4).

**(B) A `search_lib(query) -> list[signature+doc]` tool.** The agent searches by intent
("noise", "sdf", "palette") and gets back matching `SB_*` functions. Pros: scales to any lib size,
keeps the prompt small. Cons: a round-trip before it can write the call.

**Recommendation:** hybrid. Inject a *compact index* (just `name — one-line doc`, no bodies) into
the prompt when the lib is small (say ≤ ~40 functions), and ALWAYS provide `search_lib` +
`get_lib_function(name)` (returns full body + signature) so the agent can pull the exact signature
before emitting a call with the right argument count/types. The signature matters: the agent must
call `SB_hash(vec2 p)` with a `vec2`, and the only reliable source is `ShaderLibFunction.signature`.
This mirrors marginalia's `buildLibraryContext()` snapshot + on-demand fetch (grounding §4).

### 3.3 Should the agent CREATE lib functions?

Feature 015 explicitly **defers** export-from-selection ("select function in editor → push to lib...
Coming in a later feature; out of scope"). So in the near term:
- **v1: no.** The agent composes WITH the lib, doesn't author it.
- The natural agent capability when it lands is `create_lib_function(file, source)` writing a new
  `.glsl` under `lib_root` (GL-free file write; the watcher rebuilds the index). But this has the
  **highest blast radius** of any edit — a lib function is cross-project and fans out to every node.
  Gate it hard (explicit confirm, show the diff). Also the agent must respect the pure-function +
  `SB_` + no-macro-dispatch rules when authoring (§3.1), which the system prompt must carry.

---

## 4. Uniforms — the live-tweakable surface (§4)

### 4.1 The full picture (how a uniform comes to exist + get a UI)

A uniform is born from the **GLSL declaration**, not from a side table. The flow (verified
`tabs/node.py:128–136` + `ui_models.py`):

1. The agent declares `uniform vec3 u_color = vec3(0.9, 0.5, 0.05);` in the shader source.
2. On recompile, `get_active_uniforms()` (core.py:215–223) reflects it from the live program.
3. The node-panel draw loop sees a new uniform hash, builds `UIUniform.from_uniform(uniform)`
   (ui_models.py:72–88), which reads `gl_type`, `dimension`, `array_length` off the reflected
   uniform and calls `reset_input_type()`.
4. `reset_input_type()` (ui_models.py:107–116) picks a sensible default input shape: if `"color"` is
   a valid shape **and the name ends with `color`** → `color`; if `"text"` valid and name ends with
   `text` → `text`; else the first valid shape. **So naming a vec3 `u_color` auto-gives it the color
   picker** — a real convention the agent should exploit.
5. The render loop (core.py:331–338) populates `uniform_values[name]` from `uniform.value` (the
   GLSL initializer) if the agent didn't set one, so a declared default flows through automatically.

### 4.2 `valid_input_types()` — the legal shapes the agent must respect

From `ui_models.py:90–105`, the input-type is **derived from the GL type**, not free choice:

| Uniform shape | valid_input_types |
|---|---|
| UBO block (`is_ubo`) | `("buffer",)` |
| `u_time` / `u_aspect` / `u_resolution` | `("auto",)` — engine, locked |
| `sampler2D` | `("texture",)` |
| array, `uint[]` | `("array", "text")` |
| array, other | `("array",)` |
| scalar/vec, dim 3 or 4 | `("drag", "color")` |
| scalar/vec, dim 1 or 2 | `("drag",)` |
| everything else | `("auto",)` |

The agent must **never set an input_type outside this set** — `snap_input_type()`
(ui_models.py:118–121) will silently reset an illegal choice, so an illegal set is a no-op, but the
agent should pick legally up front. Key derived rules for the prompt:
- A "tweakable slider" = any scalar/vec → `drag`.
- A "color picker" = `vec3`/`vec4` → `color` (and naming it `*_color`/`u_color` makes it the
  default).
- A "text input" = `uint[]` array → `text` (the text-rendering convention, §5.4).
- A texture = `sampler2D` → only `texture` (the user must then *load* an image; the agent can't
  conjure pixel data).

### 4.3 The "add a tweakable parameter" capability, end-to-end

```
add_tweakable(node_id, decl="uniform vec3 u_glow = vec3(1.0,0.3,0.1);",
              input_type="color", value=[1.0,0.3,0.1])
```
Steps the tool performs:
1. Insert the `uniform ...;` line into the shader source (after the existing uniform block is the
   tidy spot) and `edit_shader` (write to disk).
2. Recompile happens on the main thread (the free lunch). `get_active_uniforms()` now includes it;
   the panel auto-builds the `UIUniform` with a default `input_type` per §4.1.
3. If the agent wants a non-default shape, set `UIUniform.input_type` (legal-only, §4.2). **This is a
   GAP — there is no headless setter today; it's mutated inline in `draw_input_type_selector`
   (uniform.py:84–92).** A `set_uniform_input_type(node, name, type)` verb must be added (grounding
   gap (a) mentions the same inline-mutation problem for values).
4. Set the value: `set_uniform_value(node, name, value)` — **also a GAP** (grounding gap (a)): today
   the value is written inline in `draw_ui_uniform` (uniform.py:228–230) as
   `ui_node.node.uniform_values[name] = new_value` after `try_to_release(current_value)`. The tool
   must replicate that: release the old value if releasable, write the new one. The write itself is
   GL-free (it's a dict write); the *bind* happens on the next render (main thread) — so the worker
   can do the dict write, but must respect the value type (`UniformValue` alias, core.py:93–101).

### 4.4 The engine uniforms — DO NOT CLOBBER

`u_time`, `u_aspect`, `u_resolution` are special-cased in BOTH the render loop (core.py:319–329 —
the engine *overwrites* whatever's in `uniform_values` with the live time/aspect/size each frame) and
in `valid_input_types()` (locked to `("auto",)`) and skipped on save (`ui_models.py:306`). The agent
MUST know:
- These are **provided by the engine**. Declaring `uniform float u_time;` opts in to the engine's
  time feed; the agent never sets their values (any value it writes is overwritten next frame).
- `u_resolution` is `vec2`, `u_aspect` is `float` (= width/height), `u_time` is `float` seconds since
  start. (Confirmed core.py:319–329 + the UV Mango template comments.)
- The agent should treat them as a fixed vocabulary, declare them when needed, and never invent
  `iTime`/`iResolution` (that's the Shadertoy port mistake, §5).

### 4.5 The stale-UIUniform gotcha (re-read from the templates — a real domain trap)

The UV Mango starter's `node.json` lists a `u_zoomout` UIUniform entry, but its
`shader.frag.glsl` declares **no** `u_zoomout` (re-read both files 2026-05-29). This is harmless
because: the sync loop (tabs/node.py:128–136) only ADDS hashes that aren't present and the render
loop only iterates `get_active_uniforms()` (live program), so an orphan `ui_uniforms` entry is never
rendered or bound. **But the agent must not be confused by the saved `ui_uniforms` map** — the
authoritative list of a node's uniforms is `get_active_uniforms()` (the live reflection), NOT the
`UINodeState.ui_uniforms` dict (which can carry stale entries from prior shader versions). Any
"list this node's uniforms" tool reads `get_active_uniforms()`, not the saved map.

---

## 5. GLSL authoring rules for ShaderBox — the system-prompt spec (§5)

This is the literal knowledge the system prompt must carry. Verified against the default VS/FS and
all three templates.

### 5.1 The fragment-shader contract (VERIFIED from `default.frag.glsl` + every template)
- **`#version 460 core`** is the first line. Always. (All templates + defaults.)
- The vertex stage is **fixed** (`default.vert.glsl`, never edited by the user): a fullscreen quad,
  `out vec2 vs_uv = a_pos * 0.5 + 0.5`. So in the fragment shader:
  - **`in vec2 vs_uv;`** — the per-pixel coordinate, **`[0,1]` range** (NOT `[-1,1]`). The starter
    comment: "Coordinate of the current pixel to be shaded." To center, the shaders do `vs_uv * 2.0
    - 1.0` (default.frag.glsl:19) or `(vs_uv - u_offset) * vec2(u_aspect, 1.0)` (text template:579).
  - **`out vec4 fs_color;`** — the output. `main()` MUST write `fs_color`. **NOT `gl_FragColor`**
    (that's legacy/Shadertoy GLSL ES; ShaderBox uses a named `out` under core profile). This is the
    single most common port mistake.
- `void main()` does the work; the last statement is effectively `fs_color = vec4(rgb, a);`.

### 5.2 Engine uniforms (the fixed vocabulary)
`uniform float u_time;` (seconds), `uniform float u_aspect;` (w/h), `uniform vec2 u_resolution;`
(pixels). Declare to opt in; engine feeds them; never set them (§4.4).

### 5.3 The SB_ lib rules (from §3.1, condensed for the prompt)
- Call `SB_foo(...)` directly — no `#include`. Lib functions are pure (no uniform injection); pass
  engine values as args. Only `SB_`-prefixed helpers are public. No `#include` lines, ever.

### 5.4 The text-rendering / glyph convention (VERIFIED from the Text Rendering template)
This is a bespoke ShaderBox idiom the agent must recognize (template `f90f5ff9`):
- Text is a **`uniform uint u_text[MAX_TEXT_LEN]`** (`MAX_TEXT_LEN 64`) — an array of **Unicode
  codepoints**, zero-terminated (`0u` = end of string, `10u` = newline; see main loop lines
  586–609). Not a string type (GLSL has none).
- Its UI input shape is `text` (the only place `("array","text")` from `valid_input_types` is used).
  The UI converts a typed string ↔ codepoint array via `str_to_unicode`/`unicode_to_str`
  (uniform.py:148–158). The agent sets text *content* by writing codepoints, but in practice the
  user types in the `text` widget — the agent's job is to wire the uniform + the glyph SDF, not to
  type the message.
- Glyphs are **rendered by SDF**, not a font atlas (despite the feature blurb's "freetype glyph
  atlas" framing — the shipped Text template is a 7-segment-style **distance-field** glyph synth:
  `get_dist_to_latin_*`, `seg*`, `quarter_ellipse_dist`, all-caps fold of a–z onto A–Z at lines
  479–481). NOTE: this contradicts CLAUDE.md's "freetype glyph-atlas text-rendering shader"
  description — the template on disk is pure SDF segment glyphs, no atlas/texture sampler. **Open
  question for the maintainer (§8):** is there a separate atlas-based text path elsewhere, or is the
  CLAUDE.md description aspirational/stale? The agent's prompt should describe what's actually on
  disk: SDF segment glyphs driven by a `uint[]` codepoint array.

### 5.5 Media-input samplers (VERIFIED from the Media Input template `73ea2431`)
- `uniform sampler2D u_image;` / `uniform sampler2D u_video;` — sampled with `texture(u_image,
  vs_uv).rgb`. A `sampler2D` uniform's value is a `MediaWithTexture` (Image/Video) or a raw
  `moderngl.Texture` (core.py:301–317). The engine supplies a `default.jpeg` if none is set
  (core.py:303). **The agent cannot supply pixel data** — it can declare the sampler and wire the
  sampling math, but loading an actual image/video is a user action (a file dialog,
  uniform.py:170–189). So "add a video input" = declare `sampler2D` + write the `texture()` call +
  *tell the user to load a file*.

### 5.6 Prompt assembly order (cache-friendliness, from ovelia, grounding §4)
Least-volatile → most-volatile: (1) the static authoring rules above; (2) the lib catalog (changes
on lib edit); (3) the current node snapshot — its source, its `get_active_uniforms()` + values, its
`compile_unit.errors`. Only the tail invalidates per turn.

---

## 6. Multi-step domain workflows (tool-call sketches)

### "Create a node from scratch for X"
```
1. create_node(template_id="53724dbd-...")   # UV Mango = the blank-ish starter (gap (b): needs the
                                              #   template_id arg; today reads grid selection)
2. edit_shader(new_node_id, <full new GLSL>)  # write the shader
3. get_compile_status(new_node_id)            # confirm it compiled (loop §2 if not)
4. set_uniform_value(new_node_id, "u_speed", 1.5)   # dial in defaults
5. set_uniform_input_type(new_node_id, "u_tint", "color")
```
Pick **UV Mango** (`53724dbd`) as the from-scratch base — it's the minimal `vs_uv`/`u_time`/`u_aspect`
starter. Pick **Media Input** (`73ea2431`) when the user wants image/video. Pick **Text Rendering**
(`f90f5ff9`) when the user wants text (and reuse, don't regenerate, the 600-line glyph SDF library —
the agent should NOT try to author segment glyphs from scratch).

### "Fix all compile errors"
```
loop (cap 4):
  errs = get_compile_status(node)            # read node.compile_unit.errors
  if no errors: break
  fix the FIRST error only (edit_shader)     # §2.5 — never batch-fix; let the cascade collapse
  # recompile happens on the main thread; poll again
on cap-exhaust: surface the last error + best-guess to the user; preview stays last-good (engine
invariant).
```

### "Make this loop seamlessly for a 3s sticker"
This is a *semantic* edit (rewrite time-driven terms so `f(t) == f(t+period)` — e.g. drive animation
by `sin(2*pi*u_time/period)` instead of raw `u_time`) **plus** a render. The render half ties to the
**sticker seamless-loop deferral** (export side, GL-heavy, marshalled — out of the domain MVP). The
edit half is tier-1 doable; the agent can rewrite the math, but it **cannot verify the loop is
seamless** without reading frames (§7). So: do the math rewrite + explain the technique; defer the
"render + confirm it loops" to the export feature / hand the render to the user.

---

## 7. Domain guardrails + GLSL-specific failure modes (§7)

### What the agent must NOT do
- **Delete a node without explicit confirm.** `delete_node` moves the dir to trash
  (app.py:1028–1053) — recoverable, but still destructive + changes the current selection. Gate it.
- **Overwrite a working shader the user is actively editing.** Grounding §3 flags this: an
  `edit_shader` disk write while the user has unsaved in-editor changes can clobber them (the
  watcher's re-sync only fires when texts diverge). The agent should read the current editor session
  text first, and either confirm or operate on the editor buffer, not blind-write the file.
- **Author infinite-loop GLSL.** A `for` loop with a non-constant or runaway bound, or a `while`
  with no exit, can hang or TDR the GPU (driver reset). The text template caps its loop at
  `MAX_TEXT_LEN` for exactly this reason. The agent must bound every loop with a compile-time
  constant and avoid unbounded `while`. There is **no engine guard** against this — a hung GPU on the
  single render thread freezes the whole app (grounding §2: synchronous frame loop). This is a real
  hazard the prompt must call out.
- **Clobber engine uniforms** (`u_time`/`u_aspect`/`u_resolution`) — §4.4.
- **Write `#include` / `gl_FragColor` / `iTime`** — non-conventions that produce driver errors or
  wrong behavior (§5).
- **Author lib functions that reference uniforms or dispatch via macros** — §3.1.

### The GLSL failure mode the agent CANNOT detect: "compiles but renders black/NaN"
This is the **hardest honest limitation** and must be stated plainly. The agent's only feedback is
`compile_unit.errors`. A shader that compiles cleanly but:
- divides by zero / produces `NaN` → black or garbage pixels,
- writes `fs_color = vec4(0)` everywhere (a logic bug),
- samples a texture that isn't loaded (gets the default.jpeg, looks wrong),
- has a sign/range error (`vs_uv` treated as `[-1,1]` when it's `[0,1]`),

...all **compile clean** and the agent has **no way to know the output is wrong** — it cannot read
the framebuffer. Pixel readback exists in the engine (`canvas.texture.read()`, core.py:454) but it's
GL (main-thread) and there's no "is this image black/NaN" oracle wired anywhere. Feature 013
explicitly left "compiles but renders black" as a *future* trigger (013 out-of-scope). So:
- The agent must **not claim** "I made it pulse red" — it can only claim "I wrote code intended to
  pulse red; it compiled; check the preview." (The ovelia "action requires a tool call / don't claim
  past-tense effects you can't confirm" discipline, grounding §4, maps directly here.)
- Detecting black/NaN output is a **"sounds easy, very hard"** feature (§ below). A v1 must be honest
  that the compile-success oracle is necessary but **not sufficient**, and lean on the user's eyes
  for the visual confirmation.

---

## 8. ADVERSARIAL synthesis

### 8a. The compile-loop-is-overrated case → resolved
Covered in §2.5. Summary: the loop is strong for the syntax/type/undeclared majority (source-mapped
line + a capable model), weak for semantic ("looks wrong, compiles") and driver-resource errors;
the iteration cap + fix-first-error-only + surface-to-user keep the weak cases bounded. Verdict:
keep it as the crown jewel, build the cap and the honest "compiled ≠ correct" framing alongside it.

### 8b. The simplest genuinely-useful domain MVP (v1)
**"Explain + edit the current shader, with the compile-feedback loop."** Concretely, three tools:
- `read_shader(node_id) -> source + active uniforms + values + current errors` (tier 0),
- `edit_shader(node_id, new_source)` (tier 1; disk write, free-lunch recompile),
- `get_compile_status(node_id) -> rendered errors or OK` (the loop oracle).

Plus a `search_lib` / `get_lib_function` pair (read-only lib discovery) so edits can use helpers
correctly. **Deferred from v1:** node create/delete orchestration, uniform value/input-type setters
(they need the two new headless verbs — grounding gaps (a)/(b) — and gate behind confirm), lib
authoring (015 deferral), render/export (GL-heavy + marshalling). This MVP needs **no new engine
verbs at all** for the core loop — `edit_shader` is a disk write, `read_shader`/`get_compile_status`
are reads. It rides entirely on the existing hot-reload + compile-error machinery. That is the
smallest cut that delivers the differentiator. Everything past it is additive.

The MVP is also the safest: a shader edit is one file, recoverable (the editor's undo + the
last-good-program invariant + git on `projects/dev`), and visible to the user (the preview updates).
No node deletion, no cross-project lib mutation, no export side effects.

### 8c. The use case that "sounds great" but is very hard given the codebase
**Anything requiring the agent to SEE the output.** "Make it look more like *this* reference image,"
"is it too dark?", "did my seamless-loop edit actually loop?", "why is it rendering black?" — all
need **pixel readback + a vision model + main-thread GL marshalling**, none of which is wired:
- `canvas.texture.read()` is GL → main-thread only → must be marshalled to the worker (grounding §2).
- Even with the bytes, the agent would need a multimodal model to interpret them, and a sensible
  resolution/encoding pipeline (the render canvas can be 1280×1280; sending raw frames to a vision
  API per turn is slow + costly).
- There's no "is this black / has NaN / matches a target" detector anywhere in the repo.

So "the copilot can see what it made and iterate on the look" is the seductive feature that the
codebase does NOT currently support and that would be a large, multi-seam build (readback marshalling
+ vision model + cost/latency budget). v1 must explicitly **not** promise it; the visual loop stays
human-in-the-loop. (A cheaper partial: a tiny readback that computes mean luminance / NaN-count on
the main thread and hands the agent a one-number "the frame is all black" / "the frame has NaN"
signal — that's a *much* smaller feature than vision and would catch the most common
compiles-but-broken cases. Worth filing as a follow-on trigger, not v1.)

---

## 9. Open questions for the maintainer

1. **Text-shader description mismatch.** CLAUDE.md says "custom freetype glyph-atlas text-rendering
   shader," but the shipped Text Rendering template (`f90f5ff9/shader.frag.glsl`) is a pure SDF
   7-segment glyph synth with NO font atlas / texture (re-read 2026-05-29). Is there a separate
   atlas-based text path the agent should know about, or is the CLAUDE.md line stale/aspirational?
   The prompt should describe what's on disk.
2. **Lib catalog injection threshold.** At what lib size do we switch from "inject the full index
   into the prompt" to "search_lib only"? (§3.2 suggested ~40 functions — needs a real number based
   on the maintainer's actual lib size + token budget.)
3. **Editor-buffer vs disk-write conflict.** When the user has unsaved editor changes and the agent
   wants to edit, do we (a) operate on the live editor buffer via the session, (b) refuse + ask the
   user to save first, or (c) confirm-then-overwrite? (Affects whether `edit_shader` touches disk or
   the `EditorSession.editor` text.) §7 flags the hazard; the resolution is a UX call.
4. **Iteration cap value.** 4 compile attempts/turn is a guess (§2.5). Tune against real behavior.
5. **Should the luminance/NaN one-number readback (§8c) be a fast-follow** to give the agent a cheap
   "it's black/NaN" signal short of full vision? It's small and closes the biggest honesty gap in
   §7. Flag as a trigger.
6. **Codepoint-array text editing.** Should the agent be able to set `u_text` content (write
   codepoints) or is text purely a user-typed-in-the-widget thing? (§5.4 — leaning user-only, but a
   "make it say HELLO" request is natural.)
