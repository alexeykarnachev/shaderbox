# Scripting across passes — design note (finding #30)

Status: research, no code. Written from the code as it is (`scripting/engine.py`, `behavior.py`,
`project_session.py::tick / rename_pass / _scriptable_uniforms_for`) and from 065 D12 / 068 D7.

## The use cases the design must serve

1. **Brush on `paint`** (068 walk): mouse → `u_brush`, `u_brush_color` on ONE non-output pass, plus a
   clear (finding #23). Single pass, single state.
2. **One physics state feeding several passes**: a Verlet cloth whose positions go to `scene` and
   whose velocity magnitude tints `composite`. One state, N passes. This is 065 D12's revisit trigger.
3. **The same uniform name on two passes** (`u_time_scale` on `paint` and on `composite`) driven to
   different values. A flat name map cannot express it.
4. **Rename / delete / add pass while a script exists.** Nothing may fail silently (065 D15).
5. **Export** ticks a fresh instance and must drive every pass identically to live.
6. **Copilot** reads and writes the script (`read_script` / `write_script`) — one file is simpler for
   it than N.

Case 2 and case 6 argue for one file; case 3 argues against a flat dict; case 4 is the constraint.

## Options

### A — one script per pass (065 D12 as decided)

```
documents/<id>/scripts/paint.py       class Behavior: update(ctx) -> {"u_brush": ...}
documents/<id>/scripts/composite.py   class Behavior: update(ctx) -> {"u_exposure": ...}
```

Engine keyed `(document, pass)`; everything else unchanged. Rename = move the file (already how
shaders move). Case 2 needs the state duplicated or pushed into a shader. Case 6 needs pass-aware
copilot tools. Rejected by the maintainer on case 2: the context must be shareable.

### B1 — one script, nested dict

```python
class Behavior(ScriptBehavior):
    def update(self, ctx: Ctx) -> dict:
        self.pos = ctx.mouse.x, ctx.mouse.y
        return {
            "paint": {"u_brush": self.pos},
            "composite": {"u_exposure": 1.2},
        }
```

Smallest engine change and the fewest new concepts. **The maintainer's pick.** What it gives up
against B2: a per-pass method for `Ctrl+R` to jump to, and a `def` line a rename can rewrite
through `ast`. What it avoids: the method-insertion machinery, the pure-reader contract, and the
question of a pass method calling another — a dict has no calls.

### B2 — one script, one method per pass

```python
class Behavior(ScriptBehavior):
    def __init__(self) -> None:
        self.pos = (0.5, 0.5)
        self.down = False

    def update(self, ctx: Ctx) -> None:
        # Shared step: advance state ONCE per frame. Returns nothing.
        self.pos = (ctx.mouse.x, ctx.mouse.y)
        self.down = ctx.mouse.down

    def paint(self, ctx: Ctx) -> dict:
        return {"u_brush": self.pos, "u_brush_down": self.down}

    def composite(self, ctx: Ctx) -> dict:
        return {"u_exposure": 1.0 + 0.5 * ctx.mouse.y}
```

Engine per frame: `update(ctx)`; then for each pass in the document, if the class has a method of
that name, call it and route the dict to THAT pass's `uniform_values`, orphan-checking against that
pass's own active uniforms. A pass with no method drives nothing (manual). A method that names no
pass is a **script error** shown in the strip ("`main` is not a pass; passes: paint, seed, …") —
this is what makes a stale rename visible instead of silently inert. `update` returning a dict is
an error too (the old contract, gone; no compat path).

## The two questions asked

**Rename.** `rename_pass` is transactional by D15: file, every edge, output, open tab move
together, because "any one left behind fails silently". The script method is one more edge.
Rewrite it in the same transaction: parse the script with `ast`, find `def old(` on the `Behavior`
class, replace that one `def` line's identifier by line/column, write back, and let the
existing mtime reload pick it up. Deterministic, one line touched, and the strip error above is
the safety net if the parse fails (a syntax-broken script cannot be rewritten; the rename still
proceeds and the strip reports the orphan method). Not a text `replace` — a pass called `d` must
not rename every `d` in the file.

**A pass method calling another.** `self.seed(ctx)` from inside `paint()` is plain Python and
cannot be forbidden without an interpreter. Make it harmless by contract instead: pass methods are
PURE READERS of `self` — all state advances in `update`, which the engine calls exactly once
before any pass method. Under that rule a nested call reads state twice and returns a dict the
engine never sees; nothing double-steps, nothing is mis-routed. The stub's docstring states the
rule; a test pins that the engine calls `update` once per frame regardless of how many pass
methods exist. Re-entrancy detection is not needed.

## What the spec must pin

- Stopped-uniform keys become `(pass, name)`; today `stopped_uniforms` is a set of names
  (`project_session.py:604`) and would freeze a name on every pass.
- The stub: `update` returning `None`, plus one method per EXISTING pass with that pass's
  scriptable uniforms as commented examples (`script_stub_for` gains the pass list).
- `Ctrl+R` on a pass tab: open the script and put the caret on `def <pass>(`; when the method is
  absent, append it to the class from the stub generator (an `ast`-located insertion at the end of
  the class body). Add pass does NOT touch the script — the method appears on first `Ctrl+R`.
- Delete pass: the method stays and becomes the strip error above; the user deletes it. (Deleting
  user code on their behalf is the one rewrite not worth doing.)
- Copilot: `read_script` / `write_script` unchanged in shape; the prompt's SCRIPT API block
  regenerates from the new stub.
- Export: `dry_run` and the export tick call the same routing.
- Orphan key → strip error, not a `logger.warning`.

## Decision: B1

The maintainer chose the nested dict. Consequences, replacing the B2-specific items above:

- `update(ctx) -> dict` keeps its name and shape; the value is `{pass_name: {uniform: value}}`.
  A top-level key that is not a pass, or a uniform key that is not active on that pass, is a
  **script error in the strip** naming the passes / uniforms that exist. No bare-key shortcut for
  the output pass (a second grammar for the same thing, and the output can change under it).
- **Rename**: no rewrite of user code. The rename lands, the old key becomes the strip error above
  on the next tick, the user edits one string. Explicit and one step; the only thing lost against an
  `ast` rewrite is that one keystroke, and rewriting a string inside an arbitrary expression is the
  kind of edit that goes wrong.
- **Delete pass**: same — the stale key is the strip error.
- **Stub**: `return {}` with one commented block per existing pass, each listing that pass's
  scriptable uniforms:

  ```python
      def update(self, ctx: Ctx) -> dict:
          return {
              # "paint": {
              #     "u_brush": (0.5, 0.5),   # vec2
              # },
              # "composite": {
              #     "u_exposure": 1.0,       # float
              # },
          }
  ```

- `Ctrl+R` opens the one script; no per-pass jump.
- Stopped-uniform keys `(pass, name)`, export/dry_run through the same routing, orphan → strip
  error, copilot tools unchanged in shape and the SCRIPT API prompt block regenerated: as listed
  above, unchanged.

### Bare keys broadcast (maintainer, after B1)

A key whose value is a dict is a pass block; any other key is a uniform name applied to **every
pass that declares it**. Today a bare key reaches the OUTPUT pass only (`tick` binds
`render_pass`), so this is a change, not a preservation: the same script line drives `u_time_scale`
on `paint` and on `composite` at once, and case 1 (brush on `paint`) needs no pass block at all.

```python
    def update(self, ctx: Ctx) -> dict:
        return {
            "u_brush": (ctx.mouse.x, ctx.mouse.y),   # every pass declaring u_brush
            "composite": {"u_exposure": 1.2},        # this pass only
        }
```

Rules the spec pins:

- Value type decides: `dict` → pass block, anything else → broadcast uniform. Unambiguous because
  no uniform value is a dict, whatever the pass is named.
- A broadcast key that NO pass declares is the orphan error; one that some passes declare drives
  those and says nothing about the rest.
- A pass block wins over a broadcast for the same uniform on that pass (specific over general).
- The stub lists each pass's uniforms under its pass block, commented; the docstring states the
  broadcast rule in one line.
