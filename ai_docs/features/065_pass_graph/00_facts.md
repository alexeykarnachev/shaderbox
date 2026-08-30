# 065 — Facts gathered before the pass-graph rework

Evidence collected by a six-agent fact round, then verified by hand where a claim was
load-bearing. **No design here.** This is the ground the spec is built on.

Feature 064's step system was reverted (`34f6d19`) after the maintainer used it: *"we need the
genuine separate shader; everything clamped inside a single shader is a fucking mess."* Nine bug
fixes it surfaced were kept.

## Corrections to claims made earlier in this work

**1. "N separate shader files get error locality for free" — HALF TRUE, and the half that is false
is the one that matters.**

The MECHANISM is multi-file and always was: `SourceMap` is `dict[int, Path]`, and `resolve_usage`
emits `#line N <file_id>` per contributing source, so a driver error already resolves to the right
path and line. That part is solved.

The CONTAINER is not. Verified by reading:
- `Node` holds exactly ONE `compile_unit`, so there is no slot for a second pass's own unit.
- `resolve_usage` takes ONE root and interns file-ids from 0 per call. Two calls each produce an
  internally-correct map, but nothing today combines or retains more than one — two passes would
  each claim file-id 0 for their own root.
- `tabs/code.py` reads `ui_node.node.compile_unit.errors` — one flat list per node, filtered only
  by path for gutter markers. There is no per-pass grouping and no `pass` tab kind.

So per-pass error locality is **buildable on existing machinery**, not free. What is needed is a
per-pass `CompileUnit`, not a new error system.

**2. "`imgui_node_editor` hard-asserts on BeginChild, BeginListBox and InputTextMultiline" —
OVERSTATED.** Re-probed: the library PRINTS a notice naming all three, but only **`BeginChild`**
actually asserts. `BeginListBox` and `InputTextMultiline` return cleanly and silently do nothing,
which is harder to notice, not easier.

And **`ed.suspend()` / `ed.resume()` is a sanctioned escape** — a `begin_child` wrapped in that pair
runs with no assert (verified). The cost is the canvas's coordinate space: content drawn that way is
an overlay ON the canvas, not a widget inside a node body that pans and zooms with it. So the real
constraint is "no child windows inside a zoomed node body", not "no child windows near a graph".

**3. "The resolver's no-lib fast path is owed a `#line` fix" — ALREADY FIXED** in `d1781b5` and it
survived the revert. A stale spec note said otherwise.

## Measured numbers (RTX 3090, GL 4.6, standalone context)

| What | Measurement |
|---|---|
| Compile, per program | ~1.3-1.4 ms steady state, linear in N |
| Compile, 8 passes | ~10-11 ms total |
| Compile, 16 passes | ~21-22 ms total |
| Draw, 8 passes @512x512 f2 | 0.098 ms total (0.012 ms/draw) |
| Draw, 16 passes @512x512 f2 | 0.185 ms total (0.012 ms/draw) |
| VRAM, 512x512 f2 target | 2 MiB each; 16 targets ~32-34 MiB |

**Reading:** per-frame draw cost is not a design constraint at any plausible pass count. Compile cost
is the one to watch — it is per-edit, not per-frame, but 16 passes is a ~21 ms hitch on a hot reload.
Recompiling ONLY the edited pass is therefore worth designing for from the start.

## Where "one node = one shader" is welded in

Verified singular assumptions, each of which the rework must break:

- **`node.source` / `node.compile_unit`** — reached from across the package, several sites inside
  `copilot/backend.py`.
- **`watch.py::reload_node_if_changed`** — hot reload is per-frame mtime polling over
  `compile_unit.sources`, and index 0 is special-cased as "the root shader". The list shape already
  tolerates N entries; the privileged index does not generalize.
- **`EditorTabKind`** — `shader | script | lib`, string-compared at ~11 sites with no exhaustiveness
  check, so pyright catches nothing when a member is added.
- **`copilot/address.py`** — three address kinds (node id, `lib:`, `example:`), no slot for a pass.
  Every copilot tool taking a bare node id becomes ambiguous.
- **`get_active_uniforms()`** — the union across step programs, and its same-name collision rule,
  is load-bearing for both the panel and `UINode.save`'s row pruning.

## `core.py::Node` — the god object, by responsibility

Eight jobs in one class: GL resource lifecycle; shader compilation; uniform state; per-frame render;
step-chain orchestration (now reverted); script-engine integration; export/media rendering;
persistence loading. The step concern was bolted onto a class that already had seven others.

**This is the central structural finding.** A pass-graph rework is the occasion to split it, because
the split falls out naturally: what is per-PASS (source, program, target, uniforms) versus what is
per-DOCUMENT (the graph, the output, export, scripts).

## What survives any design, and should not be rebuilt

- **`shader_errors.py` + `shader_lib/`** — the resolver, the `SB_*` index, the `#line`/`SourceMap`
  error path. Multi-file by construction; an API-lock test guards drift.
- **`model_salvage.py`** — per-key salvage, with a completeness test enumerating every persisted
  store. New per-pass state will need it immediately.
- **The exporter architecture** — `Exporter` ABC, GL-free artifacts crossing the worker boundary,
  thread affinity enforced by an AST test. Renders *a* result; does not care where it came from.
- **`ui_primitives.py` / `theme.py`** — tokens plus an import-time collision assertion.
- **The `ProjectSession` / `App` split** — `ProjectSession` has no imgui/glfw at construction, which
  is what lets the headless harness drive the real engine. **Preserving this boundary is what keeps
  a new pass-execution engine headlessly testable.**
- **Content-addressed copilot edits** — proven by a 56-finding adversarial review that killed the
  line/anchor alternative. Which FILE a tool addresses will change; the edit mechanics should not.

## Algorithms worth porting from 064 (the code is gone; the reasoning is not)

- **Topological sort with memoization**, self-edge excluded from the cycle check. Pure graph work
  over `dict[str, set[str]]`, no file-layout dependency.
- **Ping-pong** for a pass reading its own previous frame, with the swap tied to a FRAME rather than
  a render CALL (the live loop renders the current node twice per frame; the probe renders twice
  back to back).
- **Per-target allocation with reuse-diffing** on size/dtype/filter/wrap.
- **The Reinhard + sRGB tonemap pass** for viewing a float target. A float target displayed raw is
  pure white, and every pass worth debugging is float.

What was TIED to the single-file model and must be replaced: how a pass's identity, source and read
edges are discovered. In 064 those came from parsing one file and diffing GL-active uniforms across
`#define` variants.

## History's testimony — bug classes that recur here

- **The scripting engine took five feature numbers** (040 -> 041 -> 044 -> 047 -> 048) to converge on
  "one script per node, bound by existence". 048 deleted a whole parallel path, net -1300 lines.
- **Copilot edit addressing took three** (020 -> 036 -> 038 -> 039), ending by deleting the entire
  anchor scheme AND the five guards built on it. `conventions.md` names this the canonical case of
  "structural impossibility over guard-piles: two wasted guard waves".
- **The evaluation-order/memoization bug has now shipped twice** — once in an earlier deleted DAG,
  once in 064 — and was caught neither time by tests. A duplicated pass renders the CORRECT picture,
  just N times, so it reads as "slow", not "wrong". **The new scheduler must assert the invariant on
  every plan its test module builds, not in one test.**

## Prior art: what everyone else calls these things

| Tool | Document | Pass unit | Connection | Feedback |
|---|---|---|---|---|
| Shadertoy | document | tab | positional `iChannel0..3` | bind a buffer to itself |
| ISF | JSON `PASSES` | pass, by `TARGET` | named target | `PERSISTENT: true` |
| KodeLife | XML doc | tree node | typed pass-dropdown | the dropdown, pointed at itself |
| glslViewer | one file | `#ifdef` branch | `u_buffer0` convention | `u_doubleBuffer0` |
| offline-shadertoy | file | file | `#iChannel2 "file://x.glsl"` | `"self"` |
| shadertoy-local | file | file | JSON manifest `channels` | self-reference in manifest |
| SHADERed | `.sprj` | `<pass>` | **positional slot** | manual RT alternation |
| TouchDesigner | patch | graph node | `sTD2DInputs[i]` | a dedicated Feedback TOP |
| VVVV | patch | graph node | pin | unverified |
| freska (maintainer's) | layout only | **node** | **link** | **refused outright** |

**Two convergences.** Every tool makes self-reference IMPLICIT — nobody makes the user manage a
ping-pong pair. And only the spatial-patch tools call the per-draw unit a "node", which is exactly
the word ShaderBox already spends on the document.

**One clear negative finding:** SHADERed binds by positional slot, so the same slot is `posTex` in
one shipped example and `clr` in another. ShaderBox introspects uniforms by name everywhere; binding
by name is a constraint to keep.

## freska, re-read from source

`PinKind { INPUT, OUTPUT, MANUAL }` — the load-bearing idea is that a PARAMETER is a third kind of
pin, not an edge. Uniform binding is by pin NAME. Target size derives from the input's size. A pass
with no ready input is a silent no-op, not an error.

`freska.json` holds ONLY layout — positions, zoom, selection. Zero pins, zero links. Semantics is
rebuilt from code each run.

Its two defects, both with the author's own unactioned TODOs still in place: `Graph::update()`
iterates an `unordered_map` and propagates edges AFTER updating, so an N-node chain lags up to N-1
frames nondeterministically; and inputs are addressed by hard index (`pins[0]`). Both are things to
invert, not adopt. It also BANS same-node links, so it has no feedback concept at all.

## Constraints carried in from the maintainer

- **No migration.** Everything from scratch; nothing on disk needs preserving.
- **No comments carrying semantics.** Engine-level machinery, not comment parsing. This is a fixed
  premise for the spec, not an option to weigh.
- **Shared uniforms are a coincidence, not a generalisation.** Each pass owns its uniforms; no
  sharing mechanism.
- **Rename if beneficial.** "Node" currently means both the document and the render unit, and that
  collision is what pushed back on every attempt to add a second render unit.
