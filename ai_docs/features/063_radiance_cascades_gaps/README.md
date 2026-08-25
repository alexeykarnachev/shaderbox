# 063 — Radiance cascades gap research (CLOSED, research only, no code)

**Question asked:** can ShaderBox host radiance cascades, and what is actually missing?
**Answer:** the GPU capability is entirely present and measured; the engine has no
*representation* of a pass chain. The seam decision is handed to **feature 064**.

**Start here → `00_findings.md`** (its VERDICT block is the whole summary).
**Then → `17_direction.md`** (the maintainer's call).
Everything else is evidence. Don't read it linearly.

## Reading order by question

| If you need... | Read |
|---|---|
| the verdict, in one page | `00_findings.md` |
| what was decided and why | `17_direction.md` |
| what RC actually requires | `01_reference.md` |
| **is 8-bit really fatal? is it fast enough?** | `09_measurements.md` (all measured on this box) |
| **what other tools converged on** | `11_playground_survey.md` |
| **the maintainer's own prior designs** | `08_prior_art.md` (freska, the deleted DAG) |
| why the script route is abandoned | `13_reliability.md`, `16_stress_test.md` |
| what it would FEEL like to work that way | `14_ergonomics.md` |
| whether the proof is really RC | `15_fidelity.md` |
| the history: a DAG was built and deleted | `07_prior_decisions.md` |

## Engine inventories (reference material, read on demand)

`02_input_state.md` (mouse/input) · `03_uniforms.md` · `04_render_pipeline.md` ·
`05_node_model.md` · `06_ui_seams.md` (extension points a feature would use)

## Supersession — read this before trusting any recommendation

Documents were written in order as evidence arrived, and **the conclusion changed twice**:

- `12_it_already_works.md` proved RC runs in the unmodified engine from a `script.py`. Its
  **GPU-capability conclusion stands and survived attack.** Its **"play with it now"
  recommendation is SUPERSEDED** — marked in the file itself.
- `13`–`16` are the five-agent reliability review that overturned it.
- `15_fidelity.md` found the proof's merge was **miswired** (1364/1364 directions read the wrong
  slot, 30.3% error vs a 65536-ray ground truth). **Fixed** — `rc_proof.py` and `rc_proof.png`
  are the corrected versions.

**Any recommendation in `12` is out of date. The final verdict is in `00_findings.md`.**

## Artifacts

- `rc_proof.py` — a working, **merge-corrected** radiance cascades implementation. Run:
  `uv run python ai_docs/features/063_radiance_cascades_gaps/rc_proof.py` from the repo root.
  **Evidence and a correct algorithm reference, NOT a foundation** — it drives GL from a
  `script.py`, which is the route `17_direction.md` abandons. Do not build on it.
- `rc_proof.png` — its render (post-fix; the visible ringing is the real, known-unsolved RC
  artifact, not a bug).
- `rc_bruteforce_control.png` — brute force at 256 rays/px renders essentially black. The control
  that shows why cascades earn their keep.

## Method note worth keeping

Two claims convinced me early and **both were wrong to trust**: the render showed shadows, and it
beat brute force 16:1. Neither was evidence the merge was correct — it was 30% off ground truth
while looking convincing. **A plausible render is not a numerical check.**
