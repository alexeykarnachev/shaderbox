# Fidelity: real RC in structure, miswired in the merge — now fixed

A reviewer checked whether `rc_proof.py` is genuinely radiance cascades or a plausible-looking
approximation, against the shipped gist (`0c72d185830685f67d3b8d9c7f330c3a`, fetched and read
directly) and a 65536-ray brute-force ground truth. **This was the single most valuable check in
the whole research wave.**

## The structure is genuinely correct

Verified term-for-term against the reference:

- **Cascade hierarchy: exact.** `rays = 4*sp*sp` with `sp=2^c` is algebraically `4^(c+1)` =
  `pow(base, cascadeIndex+1)` at base 4. Probe grid `res/sp` matches
  `spacing = pow(sqrtBase, cascadeIndex)`. All 6 levels match.
- **Interval partitioning: correct in form.** `t0/t1` is the geometric series
  `BL*(4^c-1)/3` — contiguous, non-overlapping, 4x growth per level.
- **Merge-only-into-non-opaque: present and correct** —
  `if(hit.a<0.5 && u_c<u_count-1.0)` is exactly the reference's
  `nonOpaque && cascadeIndex < cascadeCount-1`.
- **Edge clamping: present**, matching the reference's anti-leak clamp.
- **Bilinear: present, correctly hand-rolled.** The proof uses a **probe-major** layout
  (transposed vs the reference's direction-major), so hardware bilinear cannot interpolate
  neighbouring probes; it emulates with 4 taps and explicit `mix()`. A legitimate equivalent,
  not a shortcut.
- **Penumbra: real.** On a purpose-built disc-light scene, shadow-edge width grows
  0.027 -> 0.199 uv with distance, tracking the 65536-ray ground truth closely. **This is RC's
  signature behaviour and it is genuinely present** — not a gradient dressed up.

## The defect: the merge fetched scrambled directions

The upper cascade writes `acc*0.25` — the **mean** of its 4 directions — so upper slot `S`
already holds directions `4S..4S+3`. The angular children of lower direction `idx` are therefore
**exactly slot `S=idx`, one tap.**

The shipped code instead looped `j=0..3`, built `ui=idx*4+j`, and used `uS=mod(ui,usp*usp)`.
Verified by enumeration here:

```
cascade 0: idx=0 reads slots [0,1,2,3], correct = [0]
           idx=1 reads slots [0,1,2,3], correct = [1]   <- every direction identical
cascade 1: idx=1 reads slots [4,5,6,7], correct = [1]
across cascades 0-4: 1364/1364 directions read the wrong slot(s)
```

**Every single direction was misaddressed.** Cascade 0 is the damning case: all four directions
read the entire circle, so each received an identical, direction-independent term —
**angular information from cascades >=1 was destroyed at the final merge.**

## Measured against ground truth (256x256, GT = 65536 rays/px, converged to 0.02%)

| variant | mean vs GT | rel MAE | corr |
|---|---|---|---|
| proof as shipped | 1.13x | **30.3%** | 0.9546 |
| one-line fix (`uS=idx`, single tap) | **1.02x** | **4.5%** | 0.9933 |
| merge disabled | 0.16x | 85.3% | 0.8585 |

Penumbra scene: 26.1% -> **2.4%**. Darkest 10% of the image: **1.61x too bright -> 1.06x**.

The `nomerge` row is the control proving the merge is load-bearing and doing real work — it was
simply misaddressing.

## The fix, applied

```glsl
// Upper writes acc*0.25 (the MEAN of its 4 directions), so upper slot S already
// holds directions 4S..4S+3. idx's angular children ARE slot idx -- one tap.
float uS=idx;
...
r+=m;        // was r+=m*0.25
```

Node canvas mean **23.1 -> 11.7** after the fix: the excess brightness is gone.

**The corrected render exposes RINGING** — visible concentric rings around each emitter. This is
NOT a regression: it is the real RC artifact the article calls out as unsolved ("there are some
serious ringing artifacts!... this is still an active area of research"). The wrong merge was
averaging over the whole direction circle, which **smeared the rings away — a blur masquerading
as correctness.** `rc_proof.png` is regenerated from the fixed version.

*Reviewer's own caveat, worth keeping:* the sphere-tracer stalls on `min(light, occluder)` near
light surfaces, exhausting its step budget. That had to be fixed in BOTH ground truth and RC
before the comparison was fair — the first ground truth was wrong and made `nomerge` look best.
The same stall remains in `rc_proof.py` and mildly darkens it.

## It grows fine — analytic does not dead-end

The reviewer **prototyped the full painted chain** rather than estimating: draw -> seed -> 9x JFA
-> distance field -> 6x RC, all script-driven, unmodified engine.

- **18 passes at 512x512 in 1.28 ms** (~8% of a 60 fps frame), and JFA is cacheable to frame 0
  as the reference does.
- Accuracy matches: **JFA 5.0% rel MAE vs analytic 4.5%** — visually indistinguishable.
- Painted occluders slot in as *more of the same*: extra `gl.program()` + ping-pong pairs in the
  same `update()`. **No architectural change.**

What analytic loses is only mouse-painting itself. It is arguably **better** for *studying* RC —
exact SDFs mean no JFA quantization, and the interval/cascade parameters stay the object of
study.

Remaining-easy RC features: sun angle, ringing fixes, `base` changes, interval split — all
uniforms or shader edits. **The real constraint is `f4` for JFA seeds** (the proof uses `f2`,
fine for analytic; UV-encoded seeds need `f4`). 3D cascades would need texture arrays/3D
textures — available on the raw context, but the `sys._getframe()` hack grows more painful as
pass count rises.

## Why this check mattered

The proof had already convinced me on two grounds — it rendered shadows, and it beat brute force
16:1. Both were true and **neither was evidence the merge was correct.** A structurally-real
implementation with a fully misaddressed merge still produces a convincing image; it was 30% off
ground truth and leaking light through a wall.

Anything cited as a working reference needs a numerical check against ground truth, not a
plausible render.
