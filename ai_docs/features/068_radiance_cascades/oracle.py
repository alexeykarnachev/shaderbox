"""The numerical oracle for 068: radiance cascades on an analytic scene, plus brute force.

Run: `uv run python ai_docs/features/068_radiance_cascades/oracle.py`

**Why this exists.** 063's own first RC implementation rendered convincing shadows while
1364/1364 merge directions read the wrong slot -- 30.3% error against ground truth. It was
believed because the picture looked right and it beat brute force 16:1. Neither was evidence.
So the ShaderBox document built in this feature is checked against a NUMBER, not against a
screenshot, and this is where the number comes from.

The cascade GLSL is 063's post-fix `rc_proof.py` verbatim in substance (that version measured
4.5% against a 65536-ray reference). What changed is the harness: 063 drove the engine from a
`script.py`, which 065 renamed out from under it (`from shaderbox.core import Node`) and which
`17_direction.md` abandoned as a route anyway. This drives raw moderngl on the same analytic
scene, so it depends on nothing the engine renames.

The scene is analytic ON PURPOSE. The oracle and the document must march the SAME geometry, or a
disagreement measures the scene rather than the merge -- so both carry the same SDF text, and
`SCENE_GLSL` below is what the document's passes copy.
"""

import os

os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "4.6")
os.environ.setdefault("MESA_GLSL_VERSION_OVERRIDE", "460")

import moderngl
import numpy as np

RES = 512
CASCADES = 6
# The base interval, in UV. Cascade c covers [BL*(4^c - 1)/3, BL*(4^(c+1) - 1)/3): each level
# reaches 4x further than the last, so the stack's total reach is geometric and level 5 spans
# the canvas diagonal.
BASE_INTERVAL = 0.012

# The shared scene. The document's passes carry this same text -- a difference here would make
# every comparison meaningless.
SCENE_GLSL = """
float sdc(vec2 p, float r) { return length(p) - r; }
float sdb(vec2 p, vec2 b) {
    vec2 q = abs(p) - b;
    return length(max(q, vec2(0.0))) + min(max(q.x, q.y), 0.0);
}
float occl(vec2 p) {
    float d = sdb(p - vec2(0.50, 0.50), vec2(0.020, 0.30));
    return min(d, sdc(p - vec2(0.22, 0.26), 0.07));
}
float lightf(vec2 p, float t) {
    float d = sdc(p - vec2(0.28, 0.70 + 0.06 * sin(t)), 0.04);
    return min(d, sdb(p - vec2(0.80, 0.80), vec2(0.06, 0.014)));
}
float scene(vec2 p, float t) { return min(lightf(p, t), occl(p)); }
vec3 emis(vec2 p, float t) {
    if (sdc(p - vec2(0.28, 0.70 + 0.06 * sin(t)), 0.04) < 0.0) return vec3(4.0, 3.3, 2.0);
    if (sdb(p - vec2(0.80, 0.80), vec2(0.06, 0.014)) < 0.0) return vec3(0.5, 1.2, 4.0);
    return vec3(0.0);
}
"""

_VS = """#version 460 core
in vec2 in_pos;
out vec2 vs_uv;
void main() { vs_uv = in_pos * 0.5 + 0.5; gl_Position = vec4(in_pos, 0, 1); }
"""

_MARCH = """
const float TAU = 6.28318530718;
vec4 march(vec2 o, vec2 d, float t0, float t1, float t) {
    float s = t0;
    for (int i = 0; i < 96; i++) {
        vec2 p = o + d * s;
        if (p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0) return vec4(0, 0, 0, 1);
        float h = scene(p, t);
        if (h < 0.0005) return vec4(emis(p, t), 1.0);
        s += max(h, 0.0005);
        if (s > t1) return vec4(0.0);
    }
    return vec4(0.0);
}
"""

# One cascade level. `u_c` is the level, `u_up` the level above (already merged).
FS_CASCADE = (
    """#version 460 core
in vec2 vs_uv;
out vec4 fs_color;
uniform sampler2D u_up;
uniform float u_c, u_count, u_t, u_res;
"""
    + SCENE_GLSL
    + _MARCH
    + """
void main() {
    float sp = exp2(u_c);            // probe spacing in pixels at this level
    float rays = 4.0 * sp * sp;      // 4^(c+1) directions across the level
    vec2 co = floor(vs_uv * u_res);
    vec2 pr = floor(co / sp);        // which probe this texel belongs to
    vec2 sl = mod(co, sp);           // which direction-slot within the probe
    float si = sl.x + sl.y * sp;
    vec2 pUv = (pr + 0.5) * sp / u_res;
    float t0 = (u_c == 0.0) ? 0.0 : BASE * (pow(4.0, u_c) - 1.0) / 3.0;
    float t1 = BASE * (pow(4.0, u_c + 1.0) - 1.0) / 3.0;
    vec3 acc = vec3(0.0);
    float usp = sp * 2.0;            // the level above has 2x the spacing
    vec2 upf = pUv * u_res / usp - 0.5;
    for (float k = 0.0; k < 4.0; k++) {
        float idx = si * 4.0 + k;
        float a = TAU * (idx + 0.5) / rays;
        vec4 hit = march(pUv, vec2(cos(a), sin(a)), t0, t1, u_t);
        vec3 r = hit.rgb;
        if (hit.a < 0.5 && u_c < u_count - 1.0) {
            // THE MERGE, and the line 063 got wrong twice. The upper level writes acc*0.25 --
            // the MEAN of its 4 sub-directions -- so upper slot S already holds directions
            // 4S..4S+3. This ray's angular children ARE slot `idx`: one tap, not four.
            // Both components use `usp`; the article prints `floor(index / upperSpacing)`
            // against a `mod(index, sqrtBase)`, mixing two different quantities.
            float uS = idx;
            vec2 uSlot = vec2(mod(uS, usp), floor(uS / usp));
            vec2 base = floor(upf), fr = fract(upf);
            vec2 lim = vec2(u_res / usp - 1.0);
            vec2 c00 = clamp(base, vec2(0.0), lim) * usp + uSlot + 0.5;
            vec2 c10 = clamp(base + vec2(1, 0), vec2(0.0), lim) * usp + uSlot + 0.5;
            vec2 c01 = clamp(base + vec2(0, 1), vec2(0.0), lim) * usp + uSlot + 0.5;
            vec2 c11 = clamp(base + vec2(1, 1), vec2(0.0), lim) * usp + uSlot + 0.5;
            // Bilinear BY HAND across the four neighbouring probes, at this slot in each.
            // A hardware linear filter would blend ADJACENT SLOTS (different directions),
            // which is not the same thing and is silently wrong.
            r += mix(mix(texture(u_up, c00 / u_res).rgb, texture(u_up, c10 / u_res).rgb, fr.x),
                     mix(texture(u_up, c01 / u_res).rgb, texture(u_up, c11 / u_res).rgb, fr.x),
                     fr.y);
        }
        acc += r;
    }
    fs_color = vec4(acc * 0.25, 1.0);
}
""".replace("BASE", f"{BASE_INTERVAL:.6f}")
)

# Ground truth: N rays per pixel, each marched the whole canvas. Slow and obviously correct.
FS_BRUTE = (
    """#version 460 core
in vec2 vs_uv;
out vec4 fs_color;
uniform float u_t, u_rays, u_reach;
"""
    + SCENE_GLSL
    + _MARCH
    + """
void main() {
    vec3 acc = vec3(0.0);
    for (float i = 0.0; i < u_rays; i++) {
        float a = TAU * (i + 0.5) / u_rays;
        acc += march(vs_uv, vec2(cos(a), sin(a)), 0.0, u_reach, u_t).rgb;
    }
    fs_color = vec4(acc / u_rays, 1.0);
}
"""
)


def _quad(ctx: moderngl.Context, prog: moderngl.Program) -> moderngl.VertexArray:
    vbo = ctx.buffer(
        np.array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1], dtype="f4")
    )
    return ctx.vertex_array(prog, [(vbo, "2f", "in_pos")])


def stack_reach(levels: int) -> float:
    """How far, in UV, a `levels`-deep stack marches. The reference must be capped to THIS."""
    return BASE_INTERVAL * (4.0**levels - 1.0) / 3.0


def render_cascades(
    ctx: moderngl.Context, t: float = 0.0, levels: int = CASCADES
) -> np.ndarray:
    """The RC result at `t`, as float RGB. Merges coarse -> fine through two ping-ponged targets."""
    prog = ctx.program(vertex_shader=_VS, fragment_shader=FS_CASCADE)
    vao = _quad(ctx, prog)
    targets = [
        ctx.framebuffer([ctx.texture((RES, RES), 4, dtype="f4")]) for _ in range(2)
    ]
    for fbo in targets:
        fbo.color_attachments[0].filter = (moderngl.NEAREST, moderngl.NEAREST)
    # Set only what survived compilation: a diagnostic that disables the merge optimizes
    # `u_up`/`u_count` away entirely, and an unconditional write raises KeyError.
    def _set(name: str, value: float) -> None:
        if name in prog:
            prog[name].value = value  # type: ignore[union-attr]

    _set("u_count", float(levels))
    _set("u_t", t)
    _set("u_res", float(RES))
    written = 1
    for level in range(levels - 1, -1, -1):
        written = 1 - written
        targets[1 - written].color_attachments[0].use(location=0)
        _set("u_up", 0)
        _set("u_c", float(level))
        targets[written].use()
        ctx.clear()
        vao.render()
    raw = targets[written].color_attachments[0].read()
    return np.frombuffer(raw, dtype="f4").reshape(RES, RES, 4)[..., :3]


def render_brute(
    ctx: moderngl.Context, rays: int, t: float = 0.0, reach: float = 2.0
) -> np.ndarray:
    """Ground truth: `rays` directions per pixel, each marched `reach` far.

    `reach` is not decoration. A stack of n levels only marches `stack_reach(n)`, so a reference
    that marches the whole canvas sees light the stack never had a chance to find and the
    difference reads as a merge bug. Match it.
    """
    prog = ctx.program(vertex_shader=_VS, fragment_shader=FS_BRUTE)
    vao = _quad(ctx, prog)
    fbo = ctx.framebuffer([ctx.texture((RES, RES), 4, dtype="f4")])
    prog["u_t"].value = t
    prog["u_rays"].value = float(rays)
    prog["u_reach"].value = reach
    fbo.use()
    ctx.clear()
    vao.render()
    raw = fbo.color_attachments[0].read()
    return np.frombuffer(raw, dtype="f4").reshape(RES, RES, 4)[..., :3]


def relative_error(got: np.ndarray, want: np.ndarray) -> float:
    """Mean relative error over lit NON-EMITTER pixels, as a fraction.

    Two exclusions, each of which hid a real result during development:

    - Pixels the reference finds dark. The canvas is mostly black, so averaging everywhere
      divides by a huge zero majority and flatters any implementation.
    - The EMITTERS themselves. They are written directly by both methods and agree trivially at
      value 4.0, and they are bright enough to dominate the sum: with them in, a stack that was
      5x too bright on every bounced pixel still scored 0.0%. Bounced light is the only part
      cascades actually compute, so it is the only part worth measuring.
    """
    wm = want.mean(axis=2)
    lit = (wm > 1e-4) & (wm < 1.0)
    if not lit.any():
        raise AssertionError("the reference render has no lit non-emitter pixels")
    return float(np.abs(got[lit] - want[lit]).sum() / np.abs(want[lit]).sum())


def main() -> int:
    """Measure the stack against brute force at MATCHED angular resolution and reach.

    Both halves of the match matter. A 6-level stack resolves 4^6 directions and marches 16.4
    UV; checking it against 1024 rays capped at 2.0 reports ~580%, which is entirely the
    reference being coarser and shorter. Level n = 4^n rays over `stack_reach(n)`.

    WHAT THIS MEASURES, and what it does not. Two numbers, because they say different things:

    - `merge off` isolates cascade 0 marching its own interval with no upper contribution. It
      measures 0.999 -- the marching, the interval arithmetic and the angular bookkeeping are
      right to a tenth of a percent, and that is the half a wiring bug would break.
    - `merge on` measures 1.21 at 3+ hops. RC carries a real ~21% energy overshoot against
      brute force here, growing per merge hop (1.04 / 1.12 / 1.21) and then saturating. It is
      NOT the 063 slot-addressing bug: that one was verified by turning the merge off, which
      returns the stack to 0.999, and by the hop-scaling shape. It is the near-field
      over-sampling that the bilinear probe interpolation introduces -- the same family as the
      ringing the RC article calls an open problem ("still an active area of research").

    So the gate is on the merge-off number, which is what a mistake in THIS port would move,
    plus a ceiling on the overshoot so a real regression still shows. Do not tighten the
    overshoot bound to chase 1.0: that would be fitting the threshold to the algorithm's own
    known artifact.
    """
    ctx = moderngl.create_standalone_context()
    print(f"resolution {RES}x{RES}, base interval {BASE_INTERVAL}")

    src = FS_CASCADE
    globals()["FS_CASCADE"] = src.replace(
        "if (hit.a < 0.5 && u_c < u_count - 1.0) {", "if (false) {", 1
    )
    solo = render_cascades(ctx, levels=4)
    globals()["FS_CASCADE"] = src
    truth_solo = render_brute(ctx, rays=4, reach=stack_reach(1))
    solo_ratio = float(solo.mean(axis=2).sum() / truth_solo.mean(axis=2).sum())
    print(f"  merge off, level 0 alone vs 4 rays:      {solo_ratio:.4f}  (want ~1.00)")

    worst = 0.0
    for levels in (2, 3, 4):
        rc = render_cascades(ctx, levels=levels)
        truth = render_brute(ctx, rays=4**levels, reach=stack_reach(levels))
        ratio = float(rc.mean(axis=2).sum() / truth.mean(axis=2).sum())
        worst = max(worst, ratio)
        print(
            f"  merge on, {levels} cascades vs {4**levels:>4} rays:      "
            f"{ratio:.4f}  ({levels - 1} merge hops)"
        )

    ok = abs(solo_ratio - 1.0) < 0.02 and worst < 1.30
    print("VERDICT:", "PASS" if ok else "FAIL -- see which number moved")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
