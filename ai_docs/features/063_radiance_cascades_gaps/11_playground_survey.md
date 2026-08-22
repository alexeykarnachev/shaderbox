# Survey: how other playgrounds model multipass

Provenance: ISF, glslViewer, KodeLife, SHADERed, and the moderngl facts are from primary
sources. **Shadertoy's own `/howto` is behind Cloudflare (403 to fetch AND to curl with a
browser UA)** — its claims come from mirrors, community docs, and independent
reimplementations that hit the real API, and are flagged as such.

## The comparison

| Tool | Unit of a pass | Names inputs | Per-pass resolution | Feedback | Format / filter / wrap |
|---|---|---|---|---|---|
| Shadertoy | tab in one document | positional `iChannel0..3`, routed outside source | none (all share `iResolution`) | implicit; bind buffer to itself, end-of-frame swap | RGBA32F fixed; filter/wrap per-channel UI |
| ISF | JSON block in `PASSES` | by `TARGET` name | `"WIDTH": "$WIDTH/16.0"` expressions | `PERSISTENT: true` | `FLOAT: true`; filter/wrap unspecified |
| KodeLife | tree node in one XML | user-named typed param | `resolutionMode` + size | point `FRAME_PREV_PASS` at self | `samplerState` per binding |
| glslViewer | `#ifdef` branch in ONE file | `u_buffer0`, `u_doubleBuffer0` | none | `DOUBLE_BUFFER_N` | none exposed |
| vscode shader-toy | a `.glsl` file | `#iChannel2 "file://other.glsl"` / `"self"` | none | `"self"` | `#iChannel0::MinFilter/WrapMode` |
| shadertoy-local | a file BY NAME | JSON manifest `channels` | `"scale": 0.5` | self-reference in manifest | RGBA32F; per channel |
| SHADERed | `<pass>` in `.sprj` | **positional slot, name ignored** | `rsize` xor `fsize` | manual RT alternation | format per RT |
| Bonzomatic | none (single shader) | hardcoded `texPreviousFrame` | global only | backbuffer copy | hardcoded |
| TouchDesigner | graph node | `sTD2DInputs[i]` by dimensionality | menu; `Use Input` inherits | a dedicated Feedback TOP naming a downstream node | full format list |
| VVVV | graph node | pin -> `Filter(tex0col)` | `Output Size` pin | UNVERIFIED | `OutputFormat` + `Sampler` pins |

## Two convergences — independent designs landing on the same answer

**1. Sizing is ratio-of-output by default, absolute as opt-in.** SHADERed's `rsize`/`fsize`,
TouchDesigner's `Use Input`, VVVV's optional `Output Size`, ISF's `$WIDTH/16.0`. Four
independent designs, same default.

**2. Nobody makes the user manage a ping-pong pair.** Shadertoy, glslViewer, ISF, and
shadertoy-local all make self-reference **implicit** — the user names the buffer they read and
double-buffering is the runtime's problem. **If ShaderBox adopts one idea from this survey,
this is the one.**

## The model that fits ShaderBox's grain best

ShaderBox's grain: one directory + one shader; `node.json` is **app-written derived state**;
uniforms are **introspected from the compiled program**, not declared in config. That last
property is decisive — the machinery to read meaning out of shader source already exists.

**glslViewer's inference model is the same mechanism, extended from uniforms to buffers.**
The user writes:

```glsl
uniform sampler2D u_doubleBuffer0;
```

and the tool infers a pass exists, recompiles the same source with `#define DOUBLE_BUFFER_0`,
and ping-pongs it. One file, one directory, **zero config, and `node.json` gains nothing the
user must edit.** Their shipped example is a complete Game-of-Life-class feedback effect with
no manifest at all.

The trade, stated plainly: glslViewer exposes **no per-pass resolution, format, filter, or
wrap whatsoever**. That is the price of zero ceremony — and for a tool whose canvas is already
one fixed-size RGBA8 target, it is a price ShaderBox is mostly already paying. (Though note RC
specifically NEEDS float and per-target filtering, so a pure glslViewer clone would not carry
it.)

**ISF is the most expressive compact model** — named targets, persistence, float-ness, and
resolution expressions in ~4 lines, with one shader serving all passes via `PASSINDEX`
branching. It fits because "one directory, one .glsl" survives. **The friction is where the
JSON lives:** ISF puts it in a comment header inside the shader; ShaderBox would be tempted to
put it in `node.json` — but `node.json` is currently app-written derived state, and
hand-authored pass declarations would turn it into a file the user must edit and the app must
not clobber. **That is a real change in what `node.json` means and must be decided
deliberately, not drifted into.**

**The pragma family fits with the least invention.** offline-shadertoy's is the tersest thing
that could work:

```glsl
uniform sampler2D iChannel0; // buffer-a.glsl, filter: linear, wrap: clamp
```

The wiring rides on the declaration it configures, so **it cannot desync from it** — which is
exactly the failure mode KodeLife documents (it "never modifies your shaders' source code",
so the uniform declaration and the parameter name can silently drift apart).

## What fights the grain

**SHADERed's `.sprj`** requires a project-level file owning an ordered pass list and a slot
table — precisely the "graph between nodes" ShaderBox does not have. Worse, its binding is
**positional** (`register(t0)`), and the same slot is `posTex` in one shipped example and
`clr` in another: rename the RT and nothing breaks, reorder the slots and everything breaks. A
tool that already introspects active uniforms should bind **by name** and skip the slot table.

**The node graphs fight hardest.** TouchDesigner and VVVV put the pass graph in a spatial
patch with **no authored text form**. Adopting that means building a graph editor and
inventing an inter-node document model — a different product, not a feature. VVVV states the
philosophy outright: "there is no support for multiple passes in shader code... prepare the
passes as individual TextureFX and then plug them together in a patch." That is exactly what
ShaderBox cannot do, because it has no patch.

Worth weighting: **both node tools REFUSE to put multipass in the shader.** For a file-based
tool the opposite must be true.

## moderngl ecosystem: nearly empty

Repo searches for moderngl shadertoy/playground return essentially nothing. `einarf/shadertoy`
— by a moderngl maintainer — is an 11 KB stub, dead since 2019, zero framebuffers.

The one strong artifact is `LagPixelLOL/shadertoy-local`: passes declared **by filename**
(`image.glsl`, `buffer_a.glsl`, `common.glsl`), a JSON manifest wiring only the channels, and
a `_Target` class exposing `read_texture`/`write_fbo` as derived properties over an index flip
that **swaps immediately after each pass** — which is what makes self-read mean "last frame"
and other-read mean "this frame".

## Two implementation traps, whichever model wins

- **`texture.repeat_x/y` default to `True` (GL_REPEAT)** in moderngl — the wrong default for a
  feedback target, and RC's edge-clamp requirement makes it actively wrong. Both real projects
  set it False explicitly.
- **`ctx.sampler()` state leaks across passes.** A bound sampler overrides the texture's own
  settings and does NOT clear on `texture.use()`, so `ctx.clear_samplers(0, N)` is needed per
  pass. Recorded by shadertoy-local as "invisible in a single-pass project, which is what made
  it hard to find" — i.e. precisely the regime ShaderBox is leaving.
- Mesa trims array uniforms to the elements actually indexed while NVIDIA reports full declared
  length; since `make test` runs under MESA overrides, an `iChannelResolution[4]`-style uniform
  is a tripwire. (This is the same class as the already-recorded glyph-table quirk.)
