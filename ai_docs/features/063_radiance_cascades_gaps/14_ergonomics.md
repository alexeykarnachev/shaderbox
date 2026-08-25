# Ergonomics: the renderer works, the playground does not

A third reviewer assessed the lived experience of developing a cascade effect this way, against
the maintainer's stated goal — **"implement and PLAY WITH"** radiance cascades interactively.
Everything below was run, not read.

**The verdict in one line: ShaderBox's pitch is "edit shader, save, see it change, drag a
slider." A script-driven cascade node keeps *see it change* and loses **the editor** and **the
sliders**. Those are not accessories; they are the product.**

## Blocker 1 — GLSL errors are mislocated, and the node freezes black

A typo on line 12 of an embedded shader string surfaces as:

```
line: 18                       <- the gl.program() call, NOT line 12
message: __init__ raised: Error: GLSL Compiler failed
         0(7) : error C1503: undefined variable "normalze"
```

`0(7)` is line 7 **of the string**, which you must mentally re-add to the string's offset in the
file. The whole machinery that solves this for real shaders — `parse_shader_errors` parsing
that exact format, `SourceMap` + `#line` remapping, click-to-jump in the error strip — **exists
and never fires**, because `_script_errors_for` reads `ScriptError.line`, computed from the
*Python* traceback.

Worse, `__init__ raised:` is a **compile-level freeze**: the node stops rendering entirely and
stays frozen until the file changes. One GLSL typo = a black canvas plus a wrong line number.

**No small fix exists.** The line number is unrecoverable without knowing the string's offset.
This is structural to shaders-as-Python-strings.

## Blocker 2 — one slider costs a decoy uniform in a second file

To get a "cascade count" slider the uniform must be introspected off the **node's** program —
but the driver strips unused uniforms. Verified:

```
active uniforms, u_cascades UNUSED in main(): ['u_c0', 'u_exposure']              -> absent
active uniforms, with the no-op USE hack:     ['u_c0', 'u_cascades', 'u_exposure'] -> present
```

So each tunable parameter needs `+ float(u_cascades)*1e-9` written into the composite shader — a
deliberate no-op arithmetic lie — and then read back through the `sys._getframe()` walk. Cascade
count, ray count, interval length, exposure: **four decoy uniforms, four no-op multiplies.**

`ctx.node` on `EngineContext` helps the read-back but not this: **introspection is the wall.**

## Blocker 3 — saving corrupts the node, silently

Independently reproduced (and matching `13_reliability.md` defects 3 and 6):

```
SAVE FAILED: FileNotFoundError .../textures/u_cascade0.bin      # dir never created
# with the dir pre-created:
RELOAD FAILED: Error data size mismatch 2097152 != 1048576
```

The metadata records **no dtype**, so an `f2` texture writes 2 bytes/component and the load
reads 1. **Any non-uint8 texture is unloadable — and RC needs float targets by definition.**
`App.save()` catches and toasts, so you get "Save failed" and keep working, never noticing the
on-disk node is broken. It also dumps 2 MB of transient cascade data into the node dir per save,
for a texture regenerated in `__init__` anyway.

## The leak is far worse at RC scale

`gc_mode` appears nowhere (verified). 20 hot-reloads of a script allocating 6x 1024^2 f4:

```
glo sequence: [2, 8, 14, ... 116]   distinct: 20 of 20 -> all leaked
VRAM allocated approx: 2013 MB
```

Per export (`fresh_behavior_for` allocates again): **100 MB**. A 2-hour session saving every 2
minutes leaks ~2 GB. `13_reliability.md`'s "80 MB / 40 edits" **understates it ~25x** at
RC-sized buffers.

Mitigating: reload is on explicit save (`flush_current_editor`), **not** per keystroke.

**This one IS a one-line fix** — `ctx.gc_mode = "auto"` — and it is the single highest-value
change on the list.

## Two more, verified

**The copilot's dry-run promise, demonstrated false:**

```
before dry_run, u_c0: Image
after  dry_run, u_c0: Texture
dry_run PROMISE (live node byte-identical) HELD? False
```

**No `SB_*` library in pass shaders.** `resolve_usage` is called only inside `Node.compile()`,
so a script's own `gl.program()` gets no splicing — every SDF is hand-rolled. The lib picker's
bare-name insert still works in a script tab, which is *worse than nothing*: it inserts a name
for a function that will not resolve.

*(Found and fixed in this wave: `rc_proof.py` carried a comment claiming it "uses the real SB_*
lib names so the resolver splices them". The file contains zero `SB_` calls. Corrected in place
— that comment would have been believed on a later read and concealed exactly this gap.)*

## A cascade node cannot ship as an example

`UINode.save` **never touches `scripts/`** (verified: `scripts dir copied? False`).
`create_node_from_example` is `load_node_from_dir` + `save_ui_node`, so "Open a copy" would
silently produce a node whose script is **gone** — a sampler bound to the default image, and no
error anywhere. Shipping one requires teaching the example-copy path about `scripts/`.

## shader-lab survives — better than expected

Tested with a real side-effecting multipass lab node:

```
wrote lab.png (256x256, t=0.5)
wrote lab.mp4 (256x256, 1.0s @ 10fps)
blue-channel mean per frame: [126.1, 137.8, ... 225.5]   ANIMATES? True
```

The MP4 deliverable — the offscreen half of the workflow — **does** survive, because the texture
arrives by the side-effect path rather than the returned dict. Versioning fits fine (a step is a
fresh node dir; the skill already lists `scripts/script.py` as legitimate step content).

Two snags found by running it:

1. **`render_node.py` does not inject `ScriptBehavior`** — it uses raw `importlib`, unlike
   `behavior.py::_build_globals`. First run died with `NameError`. Fix: keep the explicit
   `from shaderbox.scripting import ScriptBehavior` the stub emits. A silent trap for any script
   that drops it, since it works fine in-app. **One-line fix.**
2. **The `sys._getframe()` walk works in all three paths only by coincidence** — the live
   `_tick_script`, `render_node`'s `_pre_render`, and `_apply_script_at` each happen to have a
   local named `node`. Rename that local anywhere and every cascade node breaks with **no
   error**, just a stale texture.

## Ranked against "play with it interactively"

| | Friction | Fixable? |
|---|---|---|
| BLOCKER | GLSL errors mislocated + node freezes black | no — structural to strings-in-Python |
| BLOCKER | a slider costs a decoy uniform + no-op multiply | partial; introspection is the wall |
| BLOCKER | save corrupts the node (no dtype persisted) | yes — skip script-owned samplers on save |
| annoying | 100 MB/export, ~2 GB/20 saves leaked | **yes — one line, `gc_mode="auto"`** |
| annoying | copilot dry-run mutates the live node | no |
| annoying | no `SB_*` in pass shaders | no |
| trivial | no GLSL syntax highlighting | no |
| trivial | `render_node.py` misses `ScriptBehavior` | yes, one line |

## The strategically useful finding

**The friction is concentrated in authoring and tuning, not in the pass chain.** That is real
information about what a first-class design must provide: **parameters and error locality, not
a DAG.**

Two one-line fixes (`gc_mode`, `ScriptBehavior` in `render_node.py`) plus a save guard would
make a week of experimentation *tolerable*. They would not make it **fun** — and the maintainer
said "play with".
