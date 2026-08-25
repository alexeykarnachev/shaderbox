# Direction (maintainer call, after the research wave)

Two decisions taken, and one thing explicitly deferred to design time.

## 1. The script route is abandoned, not adopted

**Do not build on the script-GL approach.** The research proved the GPU capability exists
(`12_it_already_works.md`) and then proved the route is unusable in practice
(`13_reliability.md`, `16_stress_test.md`): save crashes, revert destroys live GL with no
self-heal, `dry_run` corrupts 25% of pixels invisibly, and the copilot's feedback is inverted.

`rc_proof.py` stays as **evidence, not a foundation** — it is what established that multipass
needs no new GPU capability, and its merge fix (`15_fidelity.md`) makes it a correct reference
for the algorithm. Nothing should be built on top of it.

## 2. Scripts should eventually be RESTRICTED to CPU work

Maintainer's call, and the research supports it. Today a script can reach the live GL context
because `__builtins__` is real — recorded as a locked posture ("No sandbox (a personal IDE)").
That posture was reasonable when the only thing a script could do was compute numbers. The
research shows what it actually permits: a script can silently break `Node.release`,
`UINode.save`, `dry_run`, checkpoint/revert, and export isolation, **with no error and no way
for the engine to detect it.**

The engine already states the intended contract in `copilot/prompt.py` — **"A script writes
VALUES only"** — and enforces half of it (`_binding_reject` refuses sampler and block keys). The
gap is that the other half is unenforceable while the GL context is one `import` away.

Direction: once engine-native multipass exists, scripts have no legitimate reason to touch GL,
and the contract can be made real rather than aspirational. **Not now** — it is a separate
decision from the multipass feature, and doing it first would remove the escape hatch before the
sanctioned path exists.

## 3. UI/UX for multipass is an OPEN design question

Deliberately not answered by this research. The research says what a design must *deliver*; it
does not say what it should look like.

**What the research established as requirements** (`14_ergonomics.md` — the friction is
concentrated in authoring and tuning, NOT in the pass chain):

- **Parameters.** Every pass's uniforms must generate controls the same way a node's do today.
  The script route's failure mode was needing a decoy uniform plus a no-op multiply in a second
  file to get one slider. Whatever the design, a cascade-count slider must be as cheap as any
  other uniform.
- **Error locality.** A GLSL error in pass 3 must land in the existing error strip with the
  right file and line, click-to-jump intact. The script route lost `SourceMap`/`#line` remapping
  entirely and froze the node black on a typo.
- **Per-pass visibility.** The maintainer named this explicitly: *how and where to visualise
  different passes.* RC has 6 cascade levels plus intermediates; the reference demo ships a
  "Stage To Render" debug control (0-3) and a "Cascade Index" single-level view for exactly this
  reason (`01_reference.md`). Being able to LOOK at an intermediate pass is not a nicety for
  this class of effect — it is the primary debugging tool.

**Constraints any design inherits** (from `06_ui_seams.md`):

- The node-settings tab bar is a 3-entry registry (`ui.py::_NODE_TABS`); adding a tab is
  mechanical, but `Ctrl+1/2/3` are taken so a fourth needs a chord decision.
- The preview is `imgui.image_with_bg` and submits **no interactive item** — clicking a rendered
  pass reaches nothing today.
- The node grid already renders every node's canvas as a live thumbnail
  (`widgets/node_grid.py`), which is the closest existing precedent for "a grid of live
  textures" and worth reusing rather than reinventing.
- `node.json` is app-written derived state. If pass declarations live there, that changes what
  the file IS — the single biggest seam decision (`11_playground_survey.md`).

**Open, to be decided at spec time:** where pass declarations live (inferred from source vs a
declaration block); whether intermediate passes get thumbnails in the node panel, a dedicated
tab, or a toggle on the main preview; whether a pass is selectable/inspectable; how a
per-pass error is surfaced without a per-pass editor tab.

The survey's two convergences stand as defaults unless a reason appears to break them:
**ratio-of-output sizing**, and **implicit ping-pong** (nobody makes the user manage the pair).
