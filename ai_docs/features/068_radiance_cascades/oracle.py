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

The scene is analytic ON PURPOSE: an SDF the brute-force reference can also march exactly, so a
disagreement is about the MERGE rather than about a sampled texture.

**Scope, stated plainly.** This validates a PORT of the cascade merge, not the shipped shader.
The document marches a jump-flooded distance field and reads emission from a texture; this
marches SDFs directly. They share geometry by construction (the same shapes at the same
coordinates) but share no code, so a change to `cascade.frag.glsl` moves nothing here. What
this file protects is the algorithm; `tests/test_radiance_cascades_example.py` protects the
document's wiring, and the two together are the coverage.

**Mutation-verified**, which is the only reason to trust any of the numbers below:

    clean                          3.6%
    merge disabled                98.3%
    upper slot transposed         29.9%   (063 measured 30.3% for this class)
    probe spacing for slot span  117.9%
    063's own four-sub-index bug 120.8%
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

# Brute-force rays for the reference. MEASURED, not assumed: bounced energy moves 0.001% between
# 256 and 32768 rays on this scene, so the reference is converged well below this and the number
# buys margin rather than accuracy. (An earlier revision claimed the energy "was still climbing"
# here -- it was not, and the claim was never checked.)
REFERENCE_RAYS = 16384

# The shared scene. The document's passes carry this same text -- a difference here would make
# every comparison meaningless.
SCENE_GLSL = """
#define HIT_EPS 0.0006
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
// HIT_EPS, not 0.0. `march` reports a hit at h < HIT_EPS -- that is, up to HIT_EPS OUTSIDE the
// surface -- so an emission test for "strictly inside" lands in the gap and returns black for
// every ray that hits a light. That bug cost this file ~99.9% of its direct lighting and
// produced a confident, entirely fictional "21% overshoot" in an earlier revision.
vec3 emis(vec2 p, float t) {
    if (sdc(p - vec2(0.28, 0.70 + 0.06 * sin(t)), 0.04) < HIT_EPS) return vec3(4.0, 3.3, 2.0);
    if (sdb(p - vec2(0.80, 0.80), vec2(0.06, 0.014)) < HIT_EPS) return vec3(0.5, 1.2, 4.0);
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
        // Leaving the canvas is NOT an occlusion: nothing was hit, so the ray must still be
        // allowed to merge the level above. Returning alpha=1 here blocked the merge for every
        // escaping ray, which at short intervals is most of them -- the merge branch then never
        // ran at all and mutating it changed nothing.
        if (p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0) return vec4(0.0);
        float h = scene(p, t);
        if (h < HIT_EPS) return vec4(emis(p, t), 1.0);
        s += max(h, HIT_EPS);
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
    """Relative mean absolute error over BOUNCED texels -- lit, but not an emitter.

    Both exclusions are load-bearing. Dark texels are the huge majority and would flatter any
    implementation; emitters are written directly by both methods, agree exactly, and carry
    99.8% of a whole-frame sum, so including them measures a constant. Bounced light is the only
    thing cascades actually compute.
    """
    wm = want.mean(axis=2)
    lit = (wm > 1e-4) & (wm < 1.0)
    if not lit.any():
        raise AssertionError("the reference has no bounced texels -- the scene or march is broken")
    return float(np.abs(got[lit] - want[lit]).sum() / np.abs(want[lit]).sum())


def main() -> int:
    """Measure the cascade merge against a CONVERGED brute-force reference, on bounced light.

    Three things this gate must do, each of which an earlier revision of this file got wrong:

    1. **Match the emission epsilon to the march epsilon.** `march` reports a hit up to HIT_EPS
       outside a surface; an `emis` testing "strictly inside" returns black there. That cost the
       reference 99.9% of its direct lighting -- the brightest non-emitter texel read 0.0074
       against emitters at 4.0 -- and produced a confident "~21% overshoot, do not investigate"
       docstring describing a defect that was entirely in this harness.
    2. **Score BOUNCED pixels, not the whole frame.** Emitters are written identically by both
       methods and agree exactly, and they carry 99.8% of a whole-frame sum -- so a whole-frame
       ratio is a constant wearing a metric's clothes. Measured: zeroing ALL of RC's bounced
       light moved the old whole-frame number from 1.209 to 0.998, comfortably inside its own
       pass band. A stack computing no global illumination at all would have passed.
    3. **Converge the reference, and check that it IS converged.** Measured here: bounced energy
       moves 0.001% between 256 and 32768 rays, so 16384 is amply converged. Worth measuring
       rather than asserting -- an earlier revision of this docstring claimed the opposite
       without checking.

    Reported: the ratio of total bounced energy, and relative mean absolute error per bounced
    texel. 063's corrected implementation measured 4.5% against a 65536-ray reference and its
    BROKEN one 30.3%, so the band sits between them.
    """
    ctx = moderngl.create_standalone_context()
    print(f"resolution {RES}x{RES}, base interval {BASE_INTERVAL}, {REFERENCE_RAYS} ref rays")

    # Each depth is scored against a reference marched to THAT depth's reach. A shallow stack
    # genuinely cannot see light the full-canvas reference finds, and calling that "error" would
    # be the matched-comparison mistake again, one level down. Depths 2/3/4 rather than 4/5/6:
    # stack_reach(4) is 1.02 UV against a 1.414 diagonal, so 5 and 6 render BIT-IDENTICALLY to 4
    # -- printing them reads as three confirmations and is one measurement repeated.
    worst = 0.0
    for levels in (2, 3, 4):
        truth = render_brute(ctx, rays=REFERENCE_RAYS, reach=stack_reach(levels))
        tm = truth.mean(axis=2)
        bounced = (tm > 1e-4) & (tm < 1.0)
        rc = render_cascades(ctx, levels=levels)
        err = relative_error(rc, truth)
        ratio = float(rc.mean(axis=2)[bounced].sum() / tm[bounced].sum())
        worst = max(worst, err)
        print(
            f"  {levels} cascades (reach {stack_reach(levels):.3f}, "
            f"{int(bounced.sum()):6d} bounced texels): ratio {ratio:.4f}, relMAE {err:6.2%}"
        )

    ok = worst < 0.10
    print("VERDICT:", "PASS" if ok else "FAIL -- the merge disagrees with brute force")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
