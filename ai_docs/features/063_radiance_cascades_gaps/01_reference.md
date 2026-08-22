# Reference: what Radiance Cascades actually requires

Primary source: `https://jason.today/rc` + its prerequisite `https://jason.today/gi`
(both read in full from raw HTML; MIT-licensed GLSL+JS ships inline in the page).
Mirrors: gist `0c72d185830685f67d3b8d9c7f330c3a` (rc), `e78bc62874059d64ac9592b6b9e01bb2` (gi).

**Read the shipped source, not the prose.** The article's inline snippets are pedagogical
simplifications that disagree with the demo's actual code on several load-bearing points
(the cascade-count formula, texture formats, per-target filtering). Every claim below is
tagged by provenance: SOURCE = verified in the shipped code, PROSE = article text.

## The pass chain

Per frame, from `doRenderPass()` (SOURCE):

| # | Pass | Reads | Writes |
|---|---|---|---|
| 1 | drawPass | prev draw target + mouse uniforms | drawing target (ping-pong) |
| 2 | seedPass | drawPassTexture | seed target (`vec4(vUv * alpha, 0, 1)`) |
| 3 | jfaPass x N | ping-pong JFA targets | JFA targets |
| 4 | dfPass | JFA output | distance-field target |
| 5 | rcPass x cascadeCount | distanceField + scene + lastTexture | rc ping-pong targets |
| 6 | overlayPass | rc output | screen |

Count is `4 + log2(maxDim) + cascadeCount` — about **19 draw calls** at 512x512, base 4.

**Steady state is much cheaper:** passes 2-4 run only on `frame == 0` (SOURCE). The distance
field is cached and invalidated on draw, not rebuilt per frame.

JFA pass count (SOURCE): `ceil(log2(max(width, height)))` — 9 passes at 512, 10 at 1024,
offset `pow(2, passes - i - 1)`. Ping-pong is mandatory: "We can't just use the same texture
/ render target as this is all happening in parallel, so you'd be modifying pixels that
hadn't been handled yet" (PROSE).

## Cascade hierarchy — the counter-intuitive part

The real formula is hardcoded to base 4, NOT the `uniforms.base` the prose snippet shows (SOURCE):

```js
const angularSize = Math.sqrt(renderWidth**2 + renderHeight**2);
this.radianceCascades = Math.ceil(Math.log(angularSize) / Math.log(4)) + 1.0;
```

512x512 -> diagonal ~724 -> `ceil(4.75)+1` = **6 cascades**. The `+1` is deliberate: with 5,
"our longest interval doesn't reach the edge... Add an extra cascade" (PROSE).

Per level: `rayCount = pow(base, cascadeIndex + 1)`, `spacing = pow(sqrtBase, cascadeIndex)`.

**Every cascade level renders into the SAME two fixed-size targets.** There is no per-level
texture and no per-level resolution; all levels are
`cascadeExtent = (renderWidth, renderHeight) / basePixelsBetweenProbes`. A higher cascade's
coarser probe grid is encoded *within* that fixed texture — freed texels hold more ray
directions per probe. Merge runs coarse->fine (`for i = firstLayer; i >= last; i--`), reading
level N+1 through `lastTexture` while writing N.

This bounds the buffer requirement: **two shared targets, not N.**

## Formats and filtering — both mandatory

HDR is not optional (SOURCE):

```js
type: !document.querySelector("#full-precision")?.checked
      ? THREE.HalfFloatType : THREE.FloatType,
format: THREE.RGBAFormat,
wrapS/wrapT: ClampToEdgeWrapping,
```

The prose never states this; only the source does. Seed/JFA targets force `THREE.FloatType`
outright — they store UV coordinates, and half-float would quantize positions.

Filtering differs **per target** (SOURCE): global default `NearestFilter`; the RC targets
override to `LinearMipmapLinearFilter`/`LinearFilter` with `generateMipmaps: true`; overlay
uses `LinearFilter`. A single global filter setting breaks one or the other.

The linear filter on the upper cascade is load-bearing, not cosmetic (PROSE): "the gpu will
upscale the upper cascade using bilinear interpolation giving us nearly free smoothing."

## Resolution

From URL params (SOURCE):

```js
const dp = urlParams.get('pixelRatio') ?? 2;
const rcScale = urlParams.get('rcScale') ?? dp;
new RC({ width: dp * width / rcScale, height: dp * height / rcScale,
         radius: 4 * dp, dpr: rcScale, canvasScale: rcScale / dp });
```

Default `dp=2, rcScale=2` -> radiance texture at 1x logical size, drawing/final at 2x. Canvas
defaults 512x512, up to 1024x1024.

## Interactivity

Mouse drag paints light and occluders; radius/colour configurable, erase mode, CPU-side
easing with strokes interpolated from the previous point (so last mouse position is retained
state). Controls exposed: Correct sRGB, Naive GI Noise, Ringing Fix, Sun Angle, Reduce Demand,
Sand/Solid/Falling-Sand modes, Pixels Between Base Probes, Interval Split, base ray count
(4 or 16), Cascade Index (debug a single level), Stage To Render (0-3, dumps intermediates).

Implied engine state: a **persistent accumulating draw canvas** plus a cached distance field
invalidated on draw. "Reduce Demand" splits the cascade loop across 2 frames — more
inter-frame persistence.

## Gotchas the article calls out

- **Edge clamping against light leaks**: "it's possible to leak light from one side to the
  other during the merge step" -> `clamp(offset, vec2(0.5), upperSize - 0.5)`.
- **sRGB<->linear is mandatory**: `pow(rgb, 2.2)` on read, `pow(rgb, 1/2.2)` on write.
- **Ringing artifacts are unsolved**: "still an active area of research... many of them incur
  a fair amount of overhead (as in doubling the frame time)".
- **Base-16 is admittedly buggy**: "I either have a bug somewhere or am missing something
  regarding bases other than 4", patched with `modifierHack`.
- **Merge only into empty areas**: `if (cascadeIndex < cascadeCount - 1.0 && nonOpaque)`.
- No temporal accumulation needed (PROSE): "Radiance Cascades doesn't require temporal
  accumulation, which is basically a hack to deal with noise."

## The irreducible floor

Three requirements, each independently disqualifying for a single-pass playground:

1. **~19 sequenced draw calls** across 4 distinct ping-pong target pairs.
2. **HalfFloat/Float RGBA targets** — 8-bit destroys both the UV-encoded JFA seeds and the
   HDR radiance accumulation.
3. **Per-target filtering control** — Nearest for JFA/seed, Linear+mipmaps for RC.

Plus persistent inter-frame state (the draw canvas) and a cached distance field.

The favourable note: cascade levels need only two shared targets, so buffer count is bounded
and modest. Multipass + float targets + per-target sampler state is the floor.
