# 100 — Instanced entity rendering

> **Half of this spec was superseded before implementation, by three review rounds whose
> corrections are recorded in the commits `a8976a5..e9eabff`.** What landed, and why it
> differs, is filed in `conventions.md ## Design decisions` under the two feature-100
> entries; read those first. Superseded here: decision 1 (a `PassEntry` discriminated
> union — the mode is per-FRAME, so it cannot be model state), decision 2 (capacity as a
> bounded `graph.json` field — buffers double on growth instead), decision 5 (the script
> "widens the accepted value types by one" — it needed a destination namespace, `@instances`,
> because a returned key must otherwise name a declared uniform), decision 6's *reason*
> (the "solid white" claim was gain-dependent; f2 being the default is the surviving
> argument) and its interleaving note (interleaving measured SLOWER, not faster), decisions
> 8 and 10 (stable slots, generation counters, off-thread simulation and snapshot
> interpolation — deferred to a separate feature by the maintainer), and the Files-touched
> list (five of its entries were never touched). The Goal, the Out-of-scope triggers and
> the gate DISCIPLINE stand.

## Goal

Draw tens of thousands of CPU-simulated entities in ONE draw call, each expanded from a
per-entity record into a quad whose fragment shader the author writes as any other pass's.
The simulation stays CPU-side in numpy; the GPU draws. The target is 50k entities with a
capacity of 65536, at a render rate independent of the simulation rate.

The requirement this serves, in the maintainer's words: "I want to simulate tens of thousands
of them"; "eating, hunting, collisions of course, vision, all this stuff"; "Shouldn't we just
do an instancing or batching or whatever in this case, i.e draw 50k rects in one call".

## Out of scope

- **A per-pass vertex shader** (`<name>.vert.glsl`). One engine-owned instanced vertex shader
  serves arbitrary fragment shaders; the two measured identically. Deferring it keeps a
  feature's worth of reshape out of this one: `SourceMap` has a single root path so a vertex
  error mis-attributes to the fragment file, `watch.py` classifies every non-root path as a
  lib include, and the save sweep, add/rename/delete/import, the editor tab-kind enum and the
  copilot's `<id>#<pass>` addressing all assume one file per pass.
  **Trigger:** an entity shape needs vertex-stage geometry the fragment shader cannot express
  — a trail stretched along velocity, a quad that must grow with speed.
- **`StorageBlock` support in `get_active_uniforms`.** The interleaved per-instance VBO
  measured fastest, so the SSBO route is not needed and `Pass.get_active_uniforms`'s filter
  need not change. **Trigger:** a per-entity record outgrows the vertex-attribute budget
  (`GL_MAX_VERTEX_ATTRIBS`), or a pass needs entity data in the fragment stage by index.
- **Splitting the draw out of `Pass`.** The existing VBO and VAO drive an instanced draw
  unchanged; only the draw call gains arguments. **Trigger:** a third draw shape lands.
- **A no-clear / accumulation flag.** `u_prev` feedback already expresses accumulation
  exactly, including the additive case. **Trigger:** a trail is wanted that a decay pass
  reading `u_prev` cannot express.
- **Multi-threaded simulation** beyond one worker. **Trigger:** one worker cannot hold the
  tick rate the interpolation needs.
- **Entity-state inspection UI** (picking an entity, reading its fields). **Trigger:** the
  first debugging session that cannot answer "why is that one stuck?".

## Design decisions

1. **A pass's draw is a discriminated union on `PassEntry`, not a flag.** `FullscreenDraw`
   (default, today's behaviour) or `InstancedDraw`. A flag plus loose siblings makes nonsense
   states expressible — fullscreen carrying an instance capacity, instanced with none — and
   every guard against those is the pile `conventions.md`'s structural-impossibility law
   forbids. This follows the `RenderShape` precedent: a closed pydantic shape makes the
   invalid combination unrepresentable rather than validated.

2. **Capacity is a bounded model field; the live count is a render argument.** Capacity is an
   allocation decision and lives in `graph.json` beside `scale` and `iterations`, bounded for
   the reason `PassEntry.iterations` states: `graph.json` type-checks nothing and an unbounded
   count is a frame-time bomb — here also a VRAM bomb. The live count varies per frame and
   reaches the draw as `instances=`; it needs no VAO rebuild, so it is not model state.

3. **One engine-owned instanced vertex shader**, beside `default.vert.glsl`. It expands the
   unit quad from the per-instance record and emits the local field coordinate; the author
   writes only the fragment shader, as today. A vertex stage is unavoidable — `gl_InstanceID`
   does not link in a fragment shader.

4. **The entity buffer is engine-owned and never enters `uniform_values`.** The engine
   allocates at capacity, binds per frame and owns the lifetime; the script holds no GL
   handle. This is what the 063 ruling protects — that ruling forbids a script OWNING
   untracked GL, not data flowing from a script to the GPU. Keeping the buffer out of
   `uniform_values` also makes the save defect structurally unreachable rather than avoided
   by care.

5. **The script returns entity rows as a numpy array; the engine performs the write.** A
   numpy array is plain data, not a wrapper type and not a handle, so this widens the
   accepted value types by one and does not revive what 079 deleted. Verified against each
   063 defect: revert and `dry_run` corruption do not return (a returned value cannot bypass
   the sink it routes through); the save blob does return and is closed by decision 4.

6. **The entity target is RGBA16F with a tonemap, not 8-bit.** 8-bit and additive blending
   are each correct alone and jointly produce a white frame: at 32px quads 50k sprites give
   ~25x mean overdraw against a value that clamps at 1.0, and the rendered frame is solid
   white. f2 is also already the `TargetConfig` default, for the same saturation reason 063
   measured.

7. **Additive, no depth, no per-frame sort.** Order-independent and the natural look for
   glowing entities; a sort costs CPU per frame for no GPU gain.

8. **Slot indices are stable; no compaction. Liveness is a field, identity carries a
   generation counter.** Compaction silently repoints a held reference at a different living
   entity, and `alive` alone reports a recycled slot as valid. Both are required by hunting
   (a target held across frames) and by interpolation (`prev[i]` and `cur[i]` must be the
   same entity).

9. **Dead entities collapse in the vertex shader**, not on the CPU. Measured equivalent to
   `discard` and to compaction; the simplest expression wins and CPU compaction would break
   decision 8.

10. **The simulation runs off the render thread above the decoupling threshold, and the
    render interpolates between two snapshots.** Inline, the frame time becomes the tick
    time. A numpy worker holding no GL handle is sanctioned — the rule is that a worker never
    touches moderngl. Export ticks the simulation synchronously, since an export is not
    real-time and must be reproducible.

11. **Simulation parameters are ordinary scalar uniforms.** That is what makes them tunable
    by the app's existing auto-generated controls; a parameter living as a Python constant in
    the script is invisible to the UI.

## Files touched

- `shaderbox/pass_graph.py` — the draw union, capacity bound.
- `shaderbox/core.py` — vertex-shader selection by draw mode, the draw call's arguments,
  entity-buffer ownership and per-frame bind.
- `shaderbox/resources/shaders/instanced.vert.glsl` — new.
- `shaderbox/engine_uniforms.py` — `u_entity_count`, the interpolation alpha.
- `shaderbox/popups/pass_settings.py` — the draw-mode control.
- `shaderbox/document.py` — buffer lifetime with the pass; excluded from the canvas resample.
- `shaderbox/ui_models.py` — the entity buffer must not reach `_uniform_entry`.
- `shaderbox/scripting/` — the entity-row return path (engine performs the write).
- The simulation's own home — see open question 2.
- `tests/` — new gates per the list below.
- `shaderbox/resources/document_examples/<uuid>/` — the demonstrating example.

## Gates, and the break each must survive

A gate is done when the thing it guards has been broken, the gate named it, and the break was
restored. Each line below names the break.

1. One instanced draw regardless of N. Break: remove `instances=`.
2. Instance data reaches per-instance. Break: set `divisor=0` with >=3 entities at distinct
   positions and require three distinct drawn locations — a fixture at one entity cannot see
   this.
3. All-dead draws zero samples, via an occlusion query. Silence is the assertion here, so the
   same fixture must show a living neighbour DID draw.
4. Saving a document with an entity pass does not grow `document.json` by the buffer's size.
   Break: route the buffer through `uniform_values`.
5. A resize leaves entity state bit-identical. Break: put the buffer in the resample loop.
6. Interpolation is built at alpha 0.5, never 0 or 1, where the lerp and its absence are
   indistinguishable.
7. A reference held across a death and a birth into the same slot does not silently repoint.
8. Both the main-thread frame time and the achieved tick rate are asserted; one without the
   other passes on a simulation that is not running.

Measured numbers belong in `probes/` under this directory, printed by the probe, not written
into this spec as prose.

## Open questions for the user

1. **Script hot-reload and simulation state.** Editing `script.py` currently makes a fresh
   instance, so a whitespace change wipes every entity; editing a shader does not. The
   maintainer's answer: "we can restart the full shader when the script is modified. Or we can
   support both modes, I don't know, we need to think about this and about the corner cases."
   Treated as open: the corner cases are what decide it, and they are enumerated in
   `02_hot_reload.md` rather than guessed at here.

2. **Where the simulation lives.** A document script returning entity rows, or a separate
   simulation seam with its own file and lifecycle. Bears on the threading model and on what
   a hot reload means.

3. **Export cost.** A 30-second export at 50k re-simulates every tick before rendering, and
   is not what the live view showed. Accept, cap, or seed.
