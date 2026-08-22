# Measured: can a script do multipass TODAY, with no engine change?

This is the question that would collapse the whole "architectural addition" framing into
"write a clever script". Tested rather than argued.

## Verdict: the GPU half works. The reach-the-node half is the wall.

### What works — verified end to end

A `script.py` that imports moderngl, builds its own float target, runs an extra pass, and hands
the raw texture to the node's sampler **works under the real `ScriptEngine`**:

```
script errors: {}
canvas after real-engine tick: [64 127 191 255]   (expected ~[64,128,191,255])
```

Every link in that chain is genuinely available today:

- Scripts are **unsandboxed by locked design posture** — `__builtins__` is real, so
  `import moderngl` succeeds (`behavior.py::_build_globals`: "No sandbox (a personal IDE;
  locked posture)").
- `moderngl.get_context()` returns the live context from inside a script.
- A script can create its own `ctx.texture(..., dtype="f2")`, FBO, program, and VAO, and render.
- `Node.render`'s sampler branch **accepts a raw `moderngl.Texture`** — the branch whose only
  producer is `load_from_dir`'s unreachable `textures/*.bin` path. A script can be that
  missing producer.
- Script state persists on `self` across frames, so the textures are built once.

### What blocks it — the script cannot reach its own Node

`EngineContext` is exactly four fields:

```python
t: float
dt: float
frame: int
mouse: MouseState = field(default_factory=lambda: EXPORT_MOUSE)
```

**No node handle, no context handle, no canvas.** In the probe above I had to inject it from
outside the engine (`inst.node = node`) for the test to run at all. A real script has no
sanctioned route to the `Node` whose `uniform_values` it must write.

And the sanctioned route is explicitly closed: a script **cannot return a sampler key** from
`update()`. `ScriptEngine._binding_reject` refuses it —
`"'{name}' is a sampler/block — not a scriptable value"` — so the texture must be delivered as
a **side effect on `node.uniform_values`**, not as a returned value. Structurally, this is
reaching around the engine's contract rather than using it.

A determined script could still get there (walking `gc.get_objects()`, digging through module
globals), but that is not a route to teach, document, or build a feature on.

## What this means

**The attack partially succeeds, and the honest conclusion is narrower than either extreme.**

- "ShaderBox needs a big new rendering architecture before RC is possible" — **too strong.**
  The GPU capability is all present; the engine already accepts the output.
- "Just write a clever script, no feature needed" — **also wrong.** The one missing link is a
  script's access to its own node, and closing it by having scripts mutate `uniform_values`
  as a side effect would make the script engine's contract a lie (it currently promises that
  a script's only effect is its returned dict, which is what makes `dry_run`, export
  isolation, and checkpointing sound).

So the smallest honest framing is not "build a DAG" and not "nothing to build". It is:
**decide the seam by which a pass chain is expressed**, given that every underlying capability
already works. That is a design question about the product's grain, not a feasibility question.

## A caution for any script-based design

The script engine's guarantees rest on `update()` being effect-free apart from its return
value:

- `ScriptEngine.dry_run` steps a **fresh** behaviour through export-clock frames into a
  throwaway sink and promises the live node is "byte-identical afterward". A script that
  renders into GL as a side effect breaks that promise — the dry run would draw.
- `export_isolation` swaps in a fresh behaviour so exports start cold. A side-effecting script
  would allocate a second set of GL resources per export.
- The copilot checkpoint captures `UINode.save()` + `script.py`. Script-owned textures are in
  neither, so they escape both persistence and revert.

Any design that routes multipass through scripts must answer all three. A design that routes
it through the *node model* instead inherits the existing answers.
