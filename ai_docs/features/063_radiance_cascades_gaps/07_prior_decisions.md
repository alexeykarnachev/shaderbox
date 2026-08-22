# Audit: what the repo already decided about multipass

Source: `conventions.md`, `todo.md`, `roadmap.md`, `dev_flow.md`, the feature specs, and git
history (verified first-hand, not relayed).

## The headline: ShaderBox already HAD a multi-pass DAG, and it was never rejected on merit

Four commits in April 2025 built it:

```
14b7574 2025-04-05 Implement basic rendering DAG pipeline
231dbe0 2025-04-05 improve dag pipeline
bcffb14 2025-04-06 improve DAG design
187fe81 2025-04-06 improve dag pipeline
```

`shaderbox/renderer.py` held a `ShaderNode(fs_source, width, height, inputs=[], **uniforms)`
owning its own texture + FBO, and a `Renderer` driving it. The shipped `__main__` demo was a
**three-pass bloom chain**: parallax depth warp at 800x608 -> bright-pass at half res 400x304
-> composite reading `u_input0` + `u_input1`. Per-node resolution was a first-class parameter.
`bcffb14` added a texture-unit exhaustion guard (`ctx.max_texture_units`).

Then `238acb1 2025-04-16 "re-writing this shit"` deleted `renderer.py` (124 L), `graph.py`
(325 L), `gl.py`, and `ui.py`, replacing them with a 147-line `main.py`.

**The commit body is empty. There is no recorded rationale anywhere.** The DAG was not
argued down — it was dropped in a from-scratch rewrite that pivoted to the
single-shader-per-node app. Pickaxe searches for `multipass|multi-pass|render target|ping-pong`
over all commit messages return nothing, and `conventions.md ## Design decisions` does not
mention it at all.

**So this ground is open, not settled.** Restoring multipass does not overturn a decision;
it revisits an abandonment.

### The deleted design, for reference

Worth reading before inventing a new shape — it is the maintainer's own prior solution:

```python
def render(self, renderer):
    self._initialize(renderer)
    for input_node in self.inputs:      # recursive PULL evaluation
        input_node.render(renderer)
    ...
    for i, input_node in enumerate(self.inputs):
        if f"u_input{i}" in self._gl_context.program:
            input_node._gl_context.texture.use(i)
            self._gl_context.program[f"u_input{i}"] = i
    self._gl_context.fbo.use()
    self._gl_context.ctx.clear(0.0, 0.0, 0.0, 1.0)
    self._gl_context.vao.render(moderngl.TRIANGLES)
```

Its properties: **pull-based recursive evaluation**; inputs bound as **positional
`u_input0..N`**; per-node `width`/`height`; uniforms could be **callables** evaluated per frame
(`Uniform.get_value(renderer)`); arity mismatches warned rather than failed.

Its defects, worth not repeating: **no memoization** — a diamond DAG re-renders a shared
ancestor once per consumer; **no cycle guard** — a cyclic graph recurses until the stack
blows; still RGBA8; uniform discovery by **regex over the source** rather than GL
introspection (today's `get_active_uniforms` is strictly better).

## "Node" is vestigial DAG vocabulary

`ShaderNode` first appears in `1ddb306` (before the DAG) and becomes a genuine graph vertex in
`14b7574` — `graph.py`'s `Node` carried `output: NodeOutput`, `inputs`, and its own FBO. After
the rewrite the class survived and the graph did not.

Today's docs define it flatly and never explain the name: help_content says "Every node is one
**fragment shader**"; `dev_flow.md` says "A node is just files on disk". The copilot prompt
writes it with scare quotes: `the user authors .frag.glsl "nodes"`.

The repo has form for renaming vestigial internals once they stop matching reality — roadmap
row 049 records the whole `brain` -> `script` sweep.

## A graph was intended twice

1. The DAG above.
2. `ai_docs/design/README.md`, `## Scope reminder -> Out of scope`:
   "Node-graph editor (`imgui_node_editor`) — **long-term direction, not v1**".
   Caveat: that file is stamped ARCHIVED, a point-in-time snapshot of the feature-005 design
   pass, so it records intent rather than a live commitment. But it is the only occurrence of
   the phrase, and it says *long-term direction*, not *rejected*.

## The one on-point live deferral

`052_copilot_workspace_fluency/02_media_literacy.md`, `## Out of scope`:

> **Binding a render output / another node's output as a texture (feedback / ping-pong).**
> A real ShaderBox capability but a separate feature (multi-pass buffers).
> **Trigger:** a user asks for a feedback/trail/reaction-diffusion effect.

Multi-pass buffers are already filed as a **legitimate future feature with a named trigger** —
and "I want radiance cascades" fires that trigger squarely.

Two other deferrals a pipeline feature would touch:

- **042's `u_mouse` cut** — "CUT: it's a SECOND write-path for the same cursor value (drift
  risk — the y-convention + the export-freeze must hold identically on both) and it touches
  the engine-free `Node.render` path. **Trigger:** a concrete stateless cursor-reactive shader
  is wanted with no script." Also cut there: `MouseState.down`/`.inside`/buttons.
- **010's latent `FitPolicy`** (`RENDER_AT_TARGET` / `SCALE_DISTORT`, with `LETTERBOX`/`CROP`
  deliberately unbuilt) — relevant if new targets need a sizing policy.

`todo.md` itself is FROZEN drain-only with exactly one entry (live-only copilot UI checks), and
no rendering change touches it.

## The 8-bit format was never a decision

**There is no recorded decision about 8-bit vs float targets anywhere** — no conventions
bullet, no spec, no commit message. It is inherited silently from moderngl's `dtype="f1"`
default; no `dtype=` is passed at any of the four `.texture(` sites. Pickaxe for a float
texture dtype returns nothing — **a float target was never even tried.**

Filtering and wrap are likewise unparameterized: `Canvas` never sets `filter`, `repeat_x/y`,
or calls `build_mipmaps()`.

What IS decided (`conventions.md`): "**Render output size is ONE named vocabulary
(`RenderShape`), not raw dims per caller**... Revisit if... a real need for free copilot dims
surfaces (then a `CUSTOM` member, NOT a return to raw w/h — that re-admits the foot-gun)."
Any per-pass sizing must respect that vocabulary rather than reintroducing raw dims.

## GL quirks that constrain new rendering work

From `conventions.md ## Known quirks` (each verbatim-sourced):

- **Dynamically-indexed `const` arrays are not constant storage on NVIDIA** — big lookup
  tables must be UNIFORM arrays. Measured: function-local const ~432 ms/frame, global const
  ~10 ms, `uniform vec4[]` ~0.13 ms. **The glyph tables already consume ~600 of ~1024
  constant-register slots**, so adding several large uniform arrays risks
  `C6020: Constant register limit exceeded`. Mesa/V3D also constant-folds a compile-time glyph
  index and TRIMS the array's active size to a prefix of the declaration.
- **`MESA_*` overrides must be module-top before the first `shaderbox` import** — they are read
  by the driver AT context creation; set late they silently no-op. Compiling this repo's
  `#version 460` shaders on a bare llvmpipe 4.5 context **SEGFAULTS Mesa**, which is why
  `make test` sets them and a bare `pytest` is banned.
- **`#line N M` accepts integers only** for the file id; the host keeps its own id->Path table.
- **A pre-freeze repaint needs `gl.finish()`**, and **every render encode shares ONE post-swap
  firing point**. "A NEW render entry point MUST route its encode here, never call it inline."
- **A live moderngl context must exist before constructing `Image`/`Video`/`Font`/`Canvas`/
  `Node`** — they call `moderngl.get_context()` lazily.

Two more from the `/dogfood` skill, directly on multi-target rendering:

- **Large canvas + many renders WITHOUT a per-frame `texture.read()` goes blank on V3D.**
  Rendering to a >=256px canvas hundreds of times and reading only at the end yields a
  near-empty framebuffer (mean alpha ~7). Fix: read or flush each frame.
- **`GLError 1282 glUseProgram(0)` is a REAL pipeline bug, not harness noise** — it fires on
  bridge-marshalled create_node/write_shader under the standalone context.

## The product thesis does NOT protect single-pass

The pitch everywhere is **uniform introspection -> instant controls** plus the tight
save->recompile loop. README: "Write a fragment shader, get an instant control panel for every
uniform in it, and watch the render change as you drag."

**The words "single-file" and "single-pass" appear in no doc** — not README, CLAUDE.md,
conventions, roadmap, dev_flow, or help_content. The only near-hit is README's descriptive
caption "The scene above is one shader file", about that example, not a rule.

Counter-evidence that minimalism is not the value: the product already ships a cross-project
shader library with auto-splicing, per-node Python scripts, and multi-tab editing. **It has
consistently moved away from single-file minimalism.**

So "multipass would betray the product's simplicity" is not supported by any filed statement —
it would be a new value judgement, and the maintainer's to make.

## Process constraints on the eventual feature

By `dev_flow.md`'s triage a multipass change is a **feature**, and **high-blast-radius**
("anything touching conventions"): upper-end review (extra reviewers, a spec-fidelity auditor),
**plus a sanitization sweep even if it would not normally warrant one**. At least one reviewer
must anchor to an artifact not authored by the implementer.

Two hard rules that bite specifically:

- **NO backward-compatibility / migration code, EVER.** If a new target format reshapes
  `node.json`, hand-edit `projects/dev/` and `git add projects/dev` in the same wave.
- **Watch the cycle-from-types signal** — the no-`TYPE_CHECKING` rule forces structural splits;
  anticipate them in the spec.
