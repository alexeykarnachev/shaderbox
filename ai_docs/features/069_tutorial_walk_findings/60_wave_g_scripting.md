# 069 W-G: Scripting across passes, mouse, clear

Implementation spec for wave W-G of feature 069. The parent spec (`01_spec.md § W-G`) and the design
note (`01_design_scripting.md`) fix the shape; this file fixes the code. Locked decision **D3** and
D7's mouse half apply as CONSTRAINTS and are not re-opened: one document script, `update` returns
`{pass: {uniform: value}}`, a non-dict value under a bare key broadcasts to every pass declaring that
uniform, a pass block wins over a broadcast, an unknown pass or an unknown uniform is a strip error,
and the value-type dispatch rests on the invariant that no uniform value is ever a `dict`.

W-A and W-C land before this wave and touch `ui.py`, `app.py` and `tabs/document.py`. W-E lands the
keybinding moves this wave's new command depends on. Every citation below names a symbol, so a
shifted line does not invalidate it. Line numbers, where given, are read at HEAD `ccd446b`.

## Goal

A document script stops being bound to whichever pass happens to be the output. It addresses passes
by name, so a brush uniform on `paint` is driven whatever the user is looking at, one CPU state can
feed several passes, and the same uniform name on two passes can take two different values. Every
consumer of "is this uniform driven / stopped" moves from a name to a `(pass, name)` pair, so
stopping `u_time_scale` on `composite` no longer freezes the `u_time_scale` on `paint`. A key that
names no pass, or a uniform no pass declares, becomes a soft error the user reads in the editor's
error strip on the tab it concerns, instead of a console line nobody sees. The cursor grows the two
facts a paint script needs (a button state and last frame's position), so a stroke is a capsule
rather than spaced blobs and hovering does not paint. And "clear the canvas" becomes a command with
a chord and a button, instead of a method only export calls.

## Findings folded

Four, quoted verbatim from `00_findings.md`:

- **#22** (ENGINE + UX, Mouse drawing via `ctx.mouse` from a script): "(a) the drawing doesn't
  interpolate between frames — a stroke is a set of blobs, awful; (b) it draws always, on any hover;
  it should draw only while LMB is pressed."
- **#23** (UX feature request, Mouse drawing — clearing): "we need a 'clear all' key or similar.
  Probably just handle this in the script."
- **#29** (ENGINE, unlanded decision, Scripting x passes): "after adding a pass: `key no active
  uniform 'u_brush' (orphan key) — skipped` x3 in the console. Why one script for all passes? I
  asked for the correct approach to be researched and nothing was done."
- **#30** (DESIGN, supersedes #29, Scripting x passes — the shape): "Think in use cases. Maybe a
  proper B: each pass has its own method in the class, so the context is shared. Rename: how do
  methods follow a pass rename? What if a pass calls another pass? Must be a good UX and a robust
  design, no work-arounds, nothing temporary."

#30 supersedes #29's recommendation (per-pass files) and #30's own recommendation (a method per
pass, option B2) is in turn superseded by the maintainer's pick of B1, the nested dict, plus the
bare-key broadcast the maintainer added after B1. Both supersessions are recorded in
`01_design_scripting.md § Decision: B1` and `§ Bare keys broadcast`, and this spec implements only
what survived them.

One change in this wave closes no finding: **the pass-qualified `stopped_uniforms` on disk**. It is
forced by the routing change rather than reported, because a name-keyed stop set under a pass-aware
engine freezes a name on every pass at once. Named here so a later reader does not go looking for
the finding behind it.

## Out of scope

- **Keys in the script context** (`ctx.keys` / `ctx.key_pressed`), the parent's out-of-scope entry
  and #23's option B. This wave ships option A only: an engine command. Trigger, unchanged from the
  parent: a second script-side use for keyboard input after the clear-canvas command exists.
- **A builtin `u_mouse` uniform.** #22's open half, decided by the parent: none. The mouse reaches
  shaders through a script and nothing else, which is what keeps the tutorial's paint step on the
  scripting path (068 D7's lost goal). Trigger to add one: a second mouse-driven example that needs
  no other script state.
- **Rewriting user code on a pass rename.** `01_design_scripting.md § Decision: B1` settles it: the
  rename lands, the stale key becomes the strip error on the next tick, the user edits one string.
  No `ast` rewrite, and `rename_pass` gains nothing in this wave.
- **Deleting a pass's block from the script on pass delete.** Same rule, same reason.
- **A per-pass jump for `OPEN_SCRIPT`.** One script, one tab, no caret placement. `OPEN_SCRIPT` is
  unchanged except for the chord W-E moves it to.
- **Cross-pass CPU state as an engine-provided channel.** `self.*` on the one `Behavior` instance
  already is that channel under B1. The `conventions.md` revisit note about cross-document state is
  untouched.

## Design decisions

### 1. `EngineNode` retires for `ScriptTarget`, a `Document`-shaped protocol

`scripting/engine.py` today declares `EngineNode` (`:95`) as the slice of one pass the engine
touches: a `uniform_values` dict plus `get_active_uniforms()`. Every caller in `project_session.py`
hands it `ui_document.document.render_pass`, which is precisely the defect #29 reports and 068 D7
retracted against.

It is replaced by a protocol shaped like a `Document`, in the same file and with the same headless
rule (no imgui, no glfw, no concrete `Document` import):

```python
class ScriptPass(Protocol):
    # The slice of ONE render pass the engine writes into.
    uniform_values: dict[str, Any]

    def get_active_uniforms(self) -> list[moderngl.Uniform | moderngl.UniformBlock]: ...


class ScriptTarget(Protocol):
    # The slice of a DOCUMENT the engine routes across: its passes by name.
    passes: dict[str, ScriptPass]
```

`ScriptPass` is `EngineNode` verbatim under a name that says what it is; `ScriptTarget` is the new
level. `Document.passes` is already `dict[str, Pass]` and `Pass` already satisfies `ScriptPass`, so
a real `Document` satisfies `ScriptTarget` structurally with no change to `document.py` or
`core.py`. The name `EngineNode` goes entirely rather than being widened: 065 renamed the unit from
node to pass everywhere else, and a protocol still called `Node` while meaning `Document` is the
kind of stale name the walk was full of.

`scripting/__init__.py` drops `EngineNode` from its imports and `__all__` and adds `ScriptPass` and
`ScriptTarget` in its place, alphabetically ordered as the list already is.

The engine never reads `document.graph`, `document.render_pass` or an output notion. Routing is over
`passes` alone, which is what makes "changing what you look at changes what the script drives" (#29)
structurally impossible rather than merely fixed.

### 2. The routing algorithm in `_tick_script`

One pass over the returned dict, in the dict's own iteration order, with per-pass active-uniform maps
built once up front.

```python
active_by_pass: dict[str, dict[str, moderngl.Uniform | moderngl.UniformBlock]] = {
    pass_name: {u.name: u for u in render_pass.get_active_uniforms()}
    for pass_name, render_pass in document.passes.items()
}
```

That is one `get_active_uniforms()` call per pass per tick, which is what the old code did for one
pass. `Pass.get_active_uniforms` compiles a never-attempted pass, so building the map across every
pass would compile every pass on the first tick; 066 D1 forbids that. The map is therefore built
from `render_pass.program` only where a program already exists, and a pass whose program is absent
contributes an EMPTY uniform map for this tick and is retried next tick. Concretely, `ScriptPass`
gains one more member the protocol can express without pulling GL in:

```python
class ScriptPass(Protocol):
    uniform_values: dict[str, Any]
    # False only while the pass has NEVER ATTEMPTED a compile: the engine skips it this tick
    # rather than forcing a compile from inside the script tick (066 D1).
    script_ready: bool

    def get_active_uniforms(self) -> list[moderngl.Uniform | moderngl.UniformBlock]: ...
```

`Pass` satisfies `script_ready` with

```python
    @property
    def script_ready(self) -> bool:
        return self.program is not None or bool(self.compile_unit.error_raw)
```

which is the NEGATION of the pair `get_active_uniforms` tests before compiling (`core.py:236`:
`if self.program is None and not self.compile_unit.error_raw: self.compile()`). That guard is true
exactly when a compile is still owed, so reusing it verbatim as `script_ready` inverts the member:

| Pass state | `program` | `compile_unit.error_raw` | `script_ready` | The engine does |
|---|---|---|---|---|
| never attempted | `None` | empty | **False** | skips it; no compile is forced from the tick (066 D1) |
| compile FAILED | `None` | set | **True** | reads an empty active map; its keys take the orphan path |
| compiled | set | empty | **True** | routes normally against its uniforms |

Read the table, not the expression, if the two ever disagree. The two ways to get this wrong are
both live: the would-compile guard used as-is makes a never-attempted pass read READY, so the engine
calls `get_active_uniforms` on it and compiles it from inside the tick, which is exactly the 066 D1
violation the member exists to prevent; and it makes a successfully compiled pass read NOT ready
forever, so a working document drives nothing at all. **The never-attempted case and the failed case are
different and must not be conflated.** A pass that has never attempted a compile is absent from
`active_by_pass` and its keys are held, not errored: a key naming it records no soft error and drives
nothing this tick. Erroring there would make every first frame of a six-pass document produce five
orphan errors that clear a frame later, which is exactly the console noise #29 reports, moved into
the UI. A pass whose compile FAILED is ready-but-empty: `get_active_uniforms` returns nothing, so its
active map is empty and its keys take the ordinary orphan path, and the user reads
`pass 'paint' has no active uniform 'u_brush' (orphan key)` on the strip beside the compile error
that caused it. That split matters because a failed attempt is never retried (`core.py:232`, "A
FAILED attempt is not retried — its errors stick in `compile_unit` until `invalidate()` resets it"),
so a `script_ready` meaning "the program is built" would hold that pass's keys silently for the life
of the source: no strip row, no driven pair, no play/stop button, which is the same silently-inert
failure the wave exists to remove (#29, and 065 D15's "nothing may fail silently"). Only the
never-attempted state is held, and it lasts frames.

**A held pass's values for that tick are DROPPED, not queued.** The next tick recomputes and writes
them and nothing accumulates. On a cold multi-pass document the wait is one frame per pass, because
`ui.py`'s first-render sweep compiles one never-drawn pass per document per frame (`ui.py:258-267`),
so a six-pass document's last pass starts being driven around frame six and its play/stop button
appears then. This is the same budget every tile in the grid already waits on (066 D2) and needs no
new mechanism, but it is what the maintainer sees on § Manual verification step 1.

Then, per returned key:

```python
for key, value in raw.items():
    if isinstance(value, dict):
        # A pass block: {"paint": {"u_brush": ...}}
        ...
    else:
        # A bare key: broadcast to every pass declaring it.
        ...
```

**The dispatch is the value's type and nothing else.** Not "is the key a pass name", which would make
a document with a pass called `u_brush` ambiguous, and not a syntax marker. D3 states the invariant
this rests on: no uniform value is ever a `dict`. Item 4 makes the engine assert it.

**Pass block.** `key` is looked up in `active_by_pass`. Absent and the pass is not in
`document.passes` at all: one soft error under that key naming the pass and listing the passes that
exist, no writes from the block. Present: each `(uniform, value)` in the block routes to that pass
through the same per-key path the old code ran (engine-owned key dropped silently, `_binding_reject`
for an orphan or a sampler, `coerce_one`, the stopped check, the write), with `(pass, uniform)`
replacing `name` as the key of everything the path records.

**Bare key.** The uniform name is looked up across every ready pass. Every pass whose active map has
it and for which `_binding_reject` returns `None` receives the coerced value. A bare key that NO pass
declares is a soft error, once, keyed on the pass-free form (item 5). A bare key some passes declare
and others do not drives those and says nothing about the rest, per the design note.

**Pass block beats broadcast.** Broadcasts are applied FIRST, then pass blocks, in two passes over
the dict rather than one:

```python
broadcasts = {k: v for k, v in raw.items() if not isinstance(v, dict)}
blocks = {k: v for k, v in raw.items() if isinstance(v, dict)}
```

Two ordered phases, not one pass with a precedence test, because "specific over general" has to hold
regardless of the dict's insertion order and a Python dict preserves the author's, which is not a
precedence the author intended to express. The coercion runs per target pass in both phases (a
uniform's shape is the target's, and the same name can be a `float` on one pass and a `vec2` on
another), so a broadcast whose value fits `paint` and not `composite` drives `paint` and records a
per-key error under `(composite, name)` alone.

A uniform written by both phases coerces twice in a frame. That is the price of the two-phase shape
and it is one extra `coerce_one` on an overridden key, which only happens when the script deliberately
overrides a broadcast.

### 3. The soft error carries the pass, and `ScriptError` gains one field

`ScriptError` (`scripting/errors.py`) is `uniform_name` / `kind` / `message` / `line`. It gains:

```python
@dataclass
class ScriptError:
    uniform_name: str
    kind: Literal["compile", "runtime"]
    message: str
    line: int = -1
    # The pass whose uniform this error is about; "" for a document-level error (the sentinel)
    # and for a bare key no pass declares.
    pass_name: str = ""
```

Defaulted, so every existing construction site (`behavior.py::coerce_one` x2, `PythonBehavior`'s
compile path, `_tick_script`'s three behavior-level sites) compiles unchanged and keeps meaning
"document-level". The engine sets `pass_name` at the two sites that know one: the per-key coercion
failure and the `_binding_reject` orphan inside a pass block.

The three error messages, written out because the strip shows them verbatim:

- Unknown pass block: `no pass named 'paint' in this document (passes: composite, seed)` — the key
  is the pass name, `pass_name` is `""` (there is no such pass to attribute it to), and the listed
  passes are `sorted(document.passes)`.
- Unknown uniform inside a pass block: `pass 'paint' has no active uniform 'u_brsh' (orphan key)` —
  `pass_name` is `paint`.
- Unknown bare key: `no pass declares 'u_brsh' (orphan key)` — `pass_name` is `""`, because the whole
  point is that no pass claims it.

The sampler/block reject keeps its existing wording with the pass prefixed:
`pass 'paint': 'u_prev' is a sampler/block — not a scriptable value`.

`_binding_reject` gains a `pass_name: str` parameter and builds the prefixed forms; the bare-key
orphan is built at its own call site because it is a different sentence, not a prefixed one.

### 4. `coerce_one` asserts the dict-value invariant

D3's dispatch is only safe while no uniform value is a `dict`. Today that holds by inspection of
`normalize_output` (`scripting/outputs.py:169`, which passes anything unknown through unchanged) and
`coerce_uniform_value`. Inspection is not a gate, so `coerce_one` gains an explicit rejection, before
`normalize_output`:

```python
def coerce_one(value: object, uniform: moderngl.Uniform, error_name: str) -> object:
    if isinstance(value, dict):
        raise _RuntimeScriptError(
            ScriptError(
                error_name,
                "runtime",
                "a dict is a PASS BLOCK, not a uniform value — "
                "{'pass': {'u_name': value}} addresses a pass; a bare key drives every pass "
                "declaring it",
            )
        )
    normalized = normalize_output(value)
    ...
```

Two things at once. It makes the invariant a checked property of the coercion atom rather than a
comment, so a future output type that wrapped a mapping fails here instead of silently becoming a
phantom pass block. And it is the error a user gets for a nesting mistake one level too deep
(`{"paint": {"u_brush": {"x": 0.5}}}`), which without it would read as an unhelpful shape hint about
a `float`.

The `pass_name` on this error is set by the caller (the tick knows which pass it was coercing for);
`coerce_one` itself does not take a pass.

### 5. `(pass, name)` keys, end to end

Every keyed structure in the engine moves from `str` to `tuple[str, str]`, with the pass first:

| Structure | Today | After |
|---|---|---|
| `ScriptEngine.errors` | `dict[tuple[str, str], ScriptError]` keyed `(document_id, name)` | `dict[tuple[str, str, str], ScriptError]` keyed `(document_id, pass_name, name)` |
| `DocumentScripts.last_driven` | `set[str]` | `set[tuple[str, str]]` |
| `DocumentScripts.last_skipped` | `set[str]` | `set[tuple[str, str]]` |
| `DocumentScripts.last_good` | `dict[str, Any]` | `dict[tuple[str, str], Any]` |
| `DocumentScripts.warned` | `set[str]` | removed (item 8) |
| `ScriptEngine.script_driven_uniforms` | `-> set[str]` | `-> set[tuple[str, str]]` |
| `ScriptStatus.soft_errors` | `list[tuple[str, ScriptError]]` | `list[tuple[str, str, ScriptError]]`, `(pass, key, error)` |
| `ScriptProbe.driven` | `set[str]` | `set[tuple[str, str]]` |
| `ScriptProbe.per_key_errors` / `.orphan_keys` | `list[tuple[str, ScriptError]]` | `list[tuple[str, str, ScriptError]]` |
| `ScriptProbe.samples` | `list[tuple[float, dict[str, Any]]]` | `list[tuple[float, dict[tuple[str, str], Any]]]` |
| `tick(stopped=)` | `frozenset[str]` | `frozenset[StoppedKey]` (item 6) |

The sentinel key keeps its shape by using `""` as the pass: `(document_id, "", _SCRIPT_FILE)`. A
document-level error belongs to no pass, and `""` is the same absent-pass marker `ScriptError.pass_name`
uses, so the two agree. A bare-key orphan likewise records under `(document_id, "", name)`, which is
what lets the stale-clear pop it when the key stops being returned without knowing which passes it
failed to find.

`drop_document`'s prefix filter (`k[0] == document_id`, `engine.py:629`) is width-agnostic and needs
no change. `_drop_script`'s cleanup loop (`:323`) does: it must UNPACK the pair
(`for pass_name, name in scripts.last_driven | scripts.last_skipped`) and pop
`(document_id, pass_name, name)`. Iterating the pairs as opaque elements would compose
`(document_id, (pass, name))`, a two-tuple whose second element is a tuple, which never equals the
three-tuple key it means to pop, so every per-key error would survive a script deletion and the strip
would keep showing errors for a script that no longer exists.

`_freeze` takes the pair set and resolves each pair to its pass's `uniform_values`, which means it
needs the document rather than one pass; its signature becomes
`_freeze(keys: set[tuple[str, str]], document: ScriptTarget, last_good, sink)`. The `sink` variant
keeps a flat dict keyed by the same pairs, so the dry-run's isolation is unchanged in kind.

### 6. `StoppedKey(pass_name, name)` and every signature that gains a pass

The persisted stop set becomes a list of models:

```python
class StoppedKey(BaseModel):
    # One (pass, uniform) the user has STOPPED. A pair, not a name: the same uniform name on two
    # passes is two independently stoppable rows (069 D3).
    pass_name: str
    name: str
```

placed in `ui_models.py` immediately above `UIDocumentState`, and

```python
    stopped_uniforms: list[StoppedKey] = []
```

replacing `list[str]`. `all_stopped: bool` is unchanged in shape and meaning (whole document, every
pass).

The field name `pass_name` rather than `pass`: `pass` is a Python keyword and cannot be an attribute.
It is spelled `pass_name` everywhere, including on disk, so the JSON and the model read alike.

`StoppedKey` is `frozen=True` and used as a set element in the engine, so `model_config` carries
`frozen=True` (pydantic gives a frozen model `__hash__`). The engine receives
`frozenset[StoppedKey]`, which keeps `ProjectSession._stopped_for`'s existing shape: build fresh per
tick, hand it in as a parameter, and the engine still never learns `UIDocumentState`.

Signatures that gain a pass, all in `project_session.py`:

| Method | After |
|---|---|
| `uniform_is_driven` | `(document_id: str, pass_name: str, name: str) -> bool` |
| `is_uniform_stopped` | `(document_id: str, pass_name: str, name: str) -> bool` |
| `set_uniform_stopped` | `(document_id: str, pass_name: str, name: str, stopped: bool) -> None` |
| `get_script_driven_uniforms` | `(document_id: str) -> set[tuple[str, str]]` |
| `_stopped_for` | `(document_id: str) -> frozenset[StoppedKey]` |
| `_scriptable_uniforms_for` | `(document_id: str) -> dict[str, list[moderngl.Uniform]]` |

`set_document_all_stopped` keeps its `(document_id, stopped)` signature: it is document-wide by
definition and the parent's list names it only because its `stopped_uniforms.clear()` now clears a
list of pairs. It needs no code change at all beyond the type it clears.

`_stopped_for`'s `all_stopped` branch unions `script_driven_uniforms(document_id)`, which is now a set
of pairs, so it becomes `{StoppedKey(pass_name=p, name=n) for p, n in ...}`.

`App` wrappers (`app.py:1489` `set_uniform_stopped`, `app.py:1496` `set_document_all_stopped`) forward
the new arity; the copilot-turn guard in each is unchanged.

`widgets/uniform.py` is where the pass comes from. `draw_ui_uniform` already binds
`panel_pass = app.panel_pass(app.current_document_id)` (`:177`), so the panel's pass NAME is what
every call there passes. `Pass` carries its source path, not its name, so the name is
`pass_name_of(panel_pass.source.path)` (`paths.py:25`), taken once at the top of `draw_ui_uniform`
and threaded into `uniform_is_driven` (`:187`), `is_uniform_stopped` (`:188`), the auto-stop-on-grab
(`:302`) and `_draw_play_stop`, whose signature becomes
`_draw_play_stop(app, pass_name, name, *, driven, playing)`. Its `play_stop_toggle` id becomes
`f"u_{pass_name}_{name}"` so two passes' rows of the same uniform name are two imgui ids rather than
one.

`tabs/document.py` needs no change for this: the panel's uniform rows all draw through
`draw_ui_uniform`, and the document-level stop button it owns is `set_document_all_stopped`, whose
arity is unchanged. (The parent cites `tabs/document.py:253` as "the panel's stop call"; § Verified /
corrected premises records what is actually there.)

### 7. `last_driven` / `last_skipped` keyed by the pair, and what reads them

`script_driven_uniforms` returning pairs has three readers:

- `ProjectSession.uniform_is_driven`, which now asks `(pass_name, name) in ...`.
- `ProjectSession._stopped_for`'s `all_stopped` branch, item 6.
- `CopilotBackend`, through the injected `get_script_driven_uniforms` (`backend.py:406`). Its two
  call shapes resolve oppositely, exactly as the parent says:
  - `_pass_views` (`backend.py:730`, the driven set bound at `:738`) loops every pass of a multi-pass
    document against ONE document-scoped set, so today a uniform driven on `paint` is marked
    `<driven by script.py>` on every pass that declares it. It now filters per pass:
    `{n for p, n in driven if p == name}` inside the loop, so `_format_uniforms` keeps its
    `driven: set[str]` signature and each pass gets its own set. This is the site the change
    genuinely fixes.
  - The `set_uniform` gate (`backend.py:879`) resolves the target against `render_pass` only
    (`:889`), so it stays name-keyed WITHIN that pass. The gate MOVES below the
    `target = self._get_ui_documents()[document_id].document` bind at `:887` and asks
    `(pass_name_of(target.render_pass.source.path), name) in driven`. It has to move: nothing
    naming the output pass is in scope above that bind, so the check as written at `:879` does not
    compile. Moving it past the `uniform is None` resolution is also the better order, because a
    name that is not on the output pass at all should get the "no active uniform" answer rather than
    the script-driven one. Its rejection message keeps its wording; the tool addresses the output
    pass and this wave does not widen it.
  - `_copilot_document_working_view` (`:720`) and the single-pass listing (`:637-647`) both format
    the output pass's uniforms, so both filter to the output pass's name the same way.

`ScriptProbe.driven` reaching the copilot's `write_script` result: `ScriptWriteResult.driven`
(`capabilities.py`, built at `backend.py:1941`) is `list[str]` and becomes a list of `pass.name`
strings, `sorted(f"{p}.{n}" for p, n in probe.driven)`. `per_key_errors` and `orphan_keys` (built at
`:1943` / `:1945`) render as `f"{p}.{n}: {err.message}"` when `p` is set and `f"{n}: {err.message}"`
when it is not. The `pass.uniform` dotted form is display-only, for the agent to read; it is never
parsed back and never becomes a key grammar (the design note rejected pass-qualified string keys
outright).

`_motion_verdict` (`backend.py:308`) and `_uniform_changes` (`:300`) index `probe.samples` by name;
both take the pair and print the dotted form in the `values@t=` lines. Its empty-driven message keeps
its wording, since "drives 0 uniforms" is still what happened.

### 8. The console warning goes; the strip gains the shader tab

`_tick_script`'s `logger.warning` (`engine.py:587`) and the `warned` dedup set that exists only to
throttle it (`DocumentScripts.warned`, `:137`) are both deleted. So is the `warn: bool` parameter on
`_tick_script` and the `warn=False` at its two isolated call sites. This is #29's own last sentence:
the orphan warning becomes a visible error, not a console line.

Where it becomes visible: `tabs/code.py::_script_errors_for` (`:130`) already adapts the script
status into `ShaderError` rows for the shared bottom strip, and `tabs/code.py:517` selects it for a
`script` tab. Two changes:

- `_script_errors_for` renders the pass in the row text for a pass-attributed error:
  `f"{pass_name}.{key}: {message}"` when `pass_name` is set, the existing `f"{key}: {message}"` when
  it is not. The script tab keeps showing EVERY soft error, which is right: the script is one file
  and its author wants all of them.
- A new `_script_errors_for_pass(app, tab, pass_name)` returns the subset of the same status whose
  `pass_name` equals this shader tab's pass, rendered without the redundant pass prefix (the tab
  already says which pass it is). The shader-tab branch of the `errors` expression at `:517`
  concatenates it after the pass's compile errors:

```python
errors = (
    _script_errors_for(app, tab)
    if tab.kind == "script"
    else _to_pass_errors(edited) + _script_errors_for_pass(app, tab, pass_name_of(tab.path))
    if (edited := _pass_for_tab(app, tab)) is not None
    else ui_document.document.render_pass.compile_unit.errors
)
```

Compile errors first: a shader that does not compile is the reason nothing else works, and the strip
caps its visible rows (`_MAX_ERROR_ROWS`), so the ordering decides what a user sees without expanding.

The script errors on a shader tab point at the SCRIPT file, not the shader, because that is where the
fix is: the row carries `script_path_for(tab.document_id)` instead of `tab.path`. That path alone is
necessary and not sufficient. `_draw_error_strip`'s click branch sets
`app.editor_jump_request = JumpRequest(err.path, err.line, 0)` and stops there (`tabs/code.py:218`),
and `_consume_jump` (`:182-193`) DISCARDS a request whose path is not the current tab's ("A request
for a different file is stale (one editor only); clear it"). So the click branch must OPEN the target
before latching the jump, following the two existing cross-file jumps in the codebase
(`widgets/uniform.py:69-72` and `popups/lib_picker/filtering.py:97-98`, both of which open the file
then set the request). For a script row that is `app.open_script_for(tab.document_id)`, then the
`JumpRequest`. Without it the request is discarded as stale on the very next frame, because the
active tab is still the shader.

The markers do not leak the other way: `_apply_markers` (`tabs/code.py:160-165`) filters its
fingerprint to `err.path == current_path`, so a script-path row adds no line-fill to the shader
editor.

The sentinel (a compile or runtime failure of the whole script) stays on the script tab only. It
belongs to no pass, and repeating it on six shader tabs is noise.

### 9. The stub emits one commented block per pass

`_scriptable_uniforms_for` (`project_session.py:632`) reads `.render_pass.get_active_uniforms()`
today, which is the sibling of the 068 D7 defect: the stub lists the output pass's uniforms whatever
the script drives. It returns a per-pass mapping instead:

```python
def _scriptable_uniforms_for(self, document_id: str) -> dict[str, list[moderngl.Uniform]]:
    document = self.ui_documents[document_id].document
    return {
        name: [
            u
            for u in render_pass.get_active_uniforms()
            if is_scriptable(u) and u.name not in ENGINE_DRIVEN_UNIFORMS
        ]
        for name, render_pass in document.passes.items()
    }
```

This one DOES compile every pass, and that is correct here: it runs from `create_script` /
`read_script_source` (a user or agent action), never from the frame loop, so 066 D1's per-frame
prohibition is not in play. Passes are emitted in `document.passes` order, which is insertion order
from disk, so the stub's blocks match the pass strip's order.

`script_stub_for` takes the mapping and emits, per the design note:

```python
    def update(self, ctx: Ctx) -> dict:
        """..."""
        return {
            # A bare key drives that uniform on EVERY pass declaring it:
            #     "u_time_scale": 0.5,
            # A pass block drives one pass only, and wins over a bare key:
            # "paint": {
            #     "u_brush": Vec2(0.0, 0.0),      # vec2
            #     "u_brush_size": 0.0,            # float
            # },
            # "composite": {
            #     "u_exposure": 0.0,              # float
            # },
        }
```

The two-line docstring rule at the top of the body is the whole grammar: value type decides, specific
wins. A pass with no scriptable uniforms emits its block header with a single
`#     (no scriptable uniforms)` line rather than being omitted, so the user sees the pass exists.
A document with no scriptable uniforms anywhere keeps today's bare `return {}`.

`_script_import_line` takes the union of annotations across every pass, so a `Vec2` on one pass and a
`Text` on another both reach the import line.

`_UPDATE_DOC`'s Args block gains the mouse's new fields, which it already builds from
`EXPORT_MOUSE`; item 10 supplies them.

### 10. `MouseState.down` / `prev_x` / `prev_y`, and the `ui.py` fill

```python
@dataclass(frozen=True)
class MouseState:
    x: float = 0.5
    y: float = 0.5
    # Left button held with the cursor over the canvas. False on export and in the headless probe.
    down: bool = False
    # Last frame's position (equal to x/y on the first frame and after re-entering the canvas),
    # so a shader can stamp the CAPSULE from prev to current instead of one disc per frame.
    prev_x: float = 0.5
    prev_y: float = 0.5


EXPORT_MOUSE = MouseState(0.5, 0.5, False, 0.5, 0.5)
```

Field order matters: `x`, `y` stay first so the two positional construct sites (`EXPORT_MOUSE` and
`ui.py`'s) keep reading naturally, and every new field is defaulted so the bare-clock
`EngineContext(t=..., dt=..., frame=...)` sites are untouched.

`EXPORT_MOUSE` keeps `down=False` and `prev == current`, per #22 and the parent: an export is
deterministic, a script gated on `down` paints nothing in an export, and a script reading
`prev - current` sees a zero-length capsule rather than a jump from the origin.

The fill, in `ui.py::_draw_document_image` (`:652-659`, inside the existing hit-test, no second
hit-test):

```python
hit = item_normalized_mouse(img_min, imgui.ImVec2(...))
if hit is not None and hit[2]:
    previous = app.script_mouse
    app.script_mouse = MouseState(
        x=hit[0],
        y=hit[1],
        down=imgui.is_mouse_down(0),
        prev_x=previous.x,
        prev_y=previous.y,
    )
```

Three properties this shape has, each deliberate:

- **`prev` chains from the last IN-BOUNDS sample**, because the branch only runs on a hit. A cursor
  that leaves the canvas and returns gives a capsule from where it left to where it re-entered. That
  is a straight line across a gap the user did not draw, so the re-entry case needs the design note's
  "equal to x/y after re-entry": the else branch of the hit test sets
  `app.script_mouse = replace(app.script_mouse, down=False)` and marks the next in-bounds sample as a
  re-entry, which the fill reads to set `prev = current`. Concretely, `App` carries
  `script_mouse_inside: bool` beside `script_mouse` (`app.py:1140`), set True on a hit and False
  otherwise, and the fill sets `prev_x/prev_y` to `hit[0]/hit[1]` when it was False.
- **`down` is False whenever the cursor is outside**, including mid-drag off the canvas edge. A
  stroke that leaves the canvas ends; it does not resume as one long line when the cursor returns.
- **`is_mouse_down(0)` is read inside the hit branch**, so a press that began on another widget and
  dragged onto the canvas reads as down. That is the imgui-honest answer (the button IS down), and
  gating it on `is_mouse_clicked` instead would drop the common case of a drag that starts on the
  canvas edge.

`item_normalized_mouse` already ANDs `is_window_hovered(child_windows)` into its `inside` flag, so a
popup over the canvas suppresses both the position update and `down`. No change there.

The else branch is NEW: `ui.py:658-659` has no else at HEAD, so `context.py:8-10`'s comment
("Outside the canvas the live value clamps to the last in-bounds position") stops being the whole
truth and is rewritten in the same edit to state that the position clamps, `down` clears, and the
next in-bounds sample restarts `prev`. The else must cover BOTH `hit is None` (imgui's invalid-mouse
sentinel) and `hit[2] == False`; and because `_draw_document_image`'s no-document branch reaches
neither, `script_mouse_inside` and `down` are cleared at the TOP of the with-document branch rather
than only in the hit test's else, so a document closed mid-drag cannot leave `down=True` latched.

### 11. `RESET_FEEDBACK`, its chord, its button, and its callback

`CommandId.RESET_FEEDBACK = auto()` joins the enum, and

```python
    CommandSpec(
        CommandId.RESET_FEEDBACK,
        "Clear canvas",
        _chord(K.f6),
        C.DOCUMENT,
    ),
```

joins `COMMAND_SPECS` beside `TOGGLE_DOCUMENT_PLAY`. **F6**, from `02_keybindings.md` (the table's
last row and its note 3): the verb has to work while the user is looking at the shader they are
painting into, so it must survive editor focus, so the audit's rule 3 puts it on Alt or an F-key; F6
is chosen over an Alt letter because it sits beside F5, which is where W-E moves
`TOGGLE_DOCUMENT_PLAY`, and the two are the document's transport pair. `commands.py::_STANDALONE_KEYS`
already permits F1-F12 unmodified and `chord_needs_modifier` exempts them, so the bare F-key needs no
registry change.

The callback, in `App._build_command_callbacks` beside its siblings:

```python
    CommandId.RESET_FEEDBACK: self.reset_current_document_feedback,
```

```python
    def reset_current_document_feedback(self) -> None:
        # "Clear canvas": drop every feedback history so the next frame starts from black. A
        # document with no feedback pass has nothing to drop and the call is a no-op.
        ui_document = self.ui_documents.get(self.current_document_id)
        if ui_document is not None:
            ui_document.document.reset_feedback()
```

`Document.reset_feedback` (`document.py:357`) is called as-is, with no change: it already releases
every history canvas, clears the generation map and resets `_frame` to -1, which is exactly "as if
the app had just opened" and is what its own docstring promises. The parent names it as the callback
and it is.

The button sits by the preview, as the parent's "a small ghost button by the preview" says. Placement,
concretely: `ui.py::_draw_app_panel` already anchors the FPS chip to the preview's top-RIGHT corner
(`:693-699`, via `fps_overlay(anchor_x=cursor_pos.x + image_width, anchor_y=cursor_pos.y, ...)`). The
clear button anchors to the preview's top-LEFT by the same mechanism: `set_cursor_screen_pos` to
`(cursor_pos.x + SPACE.MD, cursor_pos.y + SPACE.MD)` and a `ghost_button("Clear")`, drawn after the
image so it paints over it. Ghost, so it does not compete with the picture; top-left, so it never
collides with the FPS chip whatever the canvas aspect. The tooltip is the command's label plus its
chord, read through `_hint(app, CommandId.RESET_FEEDBACK)` the same way the menu bar reads every other
one, so a rebind moves the tooltip with it.

Per D1's word budget the label is one word, `Clear`, and the tooltip carries the full name.

The button is drawn only when the current document has at least one feedback pass, because clearing
nothing is a control that does nothing (#4's rule, applied here). `Document` exposes that as
`has_feedback: bool` computed from the GRAPH, never from `_feedback`:
`bool(plan_passes(self.graph)[0].feedback)`. `PassPlan.feedback` (`pass_graph.py:224`, filled at
`:274`) is the set of passes whose own entry names themselves as an input, which is the document's own
declaration of what feeds back and is stable from load. `_feedback` cannot be the source: it is an
allocation cache filled on demand by `_feedback_canvas` during `render()` and emptied by
`release()`, `drop_feedback` and by `reset_feedback` ITSELF, so a property over it would hide the
button on an unrendered document and hide it again the instant the user clicks it. The COMMAND stays live regardless: a chord
that silently no-ops is cheaper than a chord that fires only sometimes, and the palette lists it
either way.

`help_content.py`'s shortcuts section enumerates every bound command, so the new spec appears there
with no edit (`tests/test_help_content.py::test_shortcuts_section_lists_every_bound_command` pins it).

### 12. The prompt block regenerates from `MouseState` and the stub

`scripting/api_doc.py` builds `_MOUSE_FIELDS` from `MouseState.__dataclass_fields__` (`:61`), so
`down`, `prev_x` and `prev_y` reach the rendered block the moment they exist on the dataclass. No edit
is needed for them to APPEAR; two edits are needed for them to read correctly:

- `_CTX_GLOSS["mouse"]`'s prose is position-only today ("x, y in 0..1, y-up"). It becomes:
  `(`, the field list, `; x/y and prev_x/prev_y in 0..1 y-up, down = LMB over the canvas -- FROZEN at
  {at} on export and in the headless probe, where down is False and prev equals x/y)`.
  **Keeping the caveat contiguous is a constraint on the rewrite, not an observation about it**: the
  test that pins it (`test_the_mouse_gloss_carries_the_frozen_at_center_caveat`) asserts the literal
  substring `f"FROZEN at {at} on export and in the headless probe"`, so any wording that inserts
  words between `at 0.5,0.5` and `on export` turns it red. The new-field clauses therefore go AFTER
  the caveat, never inside it.
- The contract bullet (`script_api_summary`'s first bullet, `api_doc.py:120`) states
  `update(self, ctx) -> dict` returning `{uniform_name: value}`. It becomes the D3 grammar in two
  sentences: a bare key drives that uniform on every pass declaring it, a `{pass: {uniform: value}}`
  block drives one pass and wins over a bare key. That is the SCRIPT API block's whole change and it
  is the one the copilot reads before writing a script.

`api_doc.py`'s import discipline holds: both edits are prose over `context.py` types, so the module
still reaches only `scripting.context` and `scripting.outputs`, and
`test_api_doc_reaches_only_for_the_gl_free_half_of_the_package` stays green unedited.

`copilot/prompt_context.py` calls `script_api_summary()` (`:99`) into the `script_api` field and
`copilot/prompt.py` renders it in the RARE tier. Neither changes: the block regenerates itself, which
is the property 059 D3 built it for. The parent lists both as touched; § Verified / corrected
premises records that `prompt_context.py` needs no edit and `prompt.py` needs none either.

`copilot/capabilities.py` changes only in the field COMMENTS on `ScriptWriteResult.driven` /
`orphan_keys` (the dotted `pass.uniform` form, item 7) and on `WorkingSetView.script_listing`; the
tool signatures `read_script` / `write_script` / `apply_script_edit` are unchanged in shape, as the
parent says.

### 13. On-disk shape change: hand edits, no migration code

`stopped_uniforms` changes from `list[str]` to `list[StoppedKey]`. **No migration path, no compat
reader, no one-off script** (`CLAUDE.md ## Hard rules`, `conventions.md ## Design decisions`). The
data is fixed by hand in the same wave, and the loader's existing salvage covers anything that was
not.

What is actually on disk. The tracked set is `git ls-tree -r HEAD --name-only | grep document.json`,
**eleven files across three directories**, not the two the parent names: re-enumerated and each one
opened at `ccd446b`.

| File | `stopped_uniforms` today | Hand edit |
|---|---|---|
| `shaderbox/resources/document_examples/0b0d16bb-.../document.json` | `[]` | none needed; `[]` is valid under both shapes. Verified, not edited. |
| `.../53724dbd-.../document.json` | `[]` | same |
| `.../73ea2431-.../document.json` | `[]` | same |
| `.../8d454b7b-.../document.json` | `[]` | same |
| `.../f90f5ff9-.../document.json` | `[]` | same |
| `.../1c4f8a20-.../document.json` (5 passes) | key absent | none; the field defaults to `[]` |
| `.../77a84d27-.../document.json` (6 passes) | key absent | none; the field defaults to `[]` |
| `projects/dev/documents/e7e00c46-.../document.json` | `[]` | none needed |
| `projects/dev/documents/ec926580-.../document.json` | `[]` | none needed |
| `projects/documents/1901ab60-.../document.json` (5 passes) | `[]` | none needed |
| `projects/documents/307598da-.../document.json` | `[]` | none needed |

So the parent's "five shipped examples are hand-edited" resolves to **eleven files verified and zero
bytes changed**: every persisted value is the empty list (or the key is absent and defaults to one),
which parses as `list[StoppedKey]` unchanged. The wave opens all eleven and states the check, because
the claim that no edit is needed is only worth anything if someone looked, and the two under
`projects/documents/` were missed by the first enumeration precisely because they sit outside the two
directories the parent named. `projects/` is tracked exactly as `projects/dev/` is (only
`projects/*/media/`, `renders/` and the copilot archives are gitignored), and `1901ab60` is a
five-pass document, the shape most likely to hold a real stop set. It does not.

`projects/dev` scripts: there are **none** (neither dev document has a `scripts/` dir), so there is no
script to rewrite to the nested return shape. The parent's "projects/dev scripts hand-edited" has no
subject. If a script appears in the sandbox between this spec and the implementation, the rule is
unchanged: hand-edit it to the new shape and `git add projects/dev` in the same wave.

**The salvage behaviour, verified by running it.** A stale `["u_brush"]` reaching
`ui_models.py:514`'s `drop_invalid(UIDocumentState, ...)` is dropped whole and the field falls back to
`[]`, with one `Ignoring invalid document '<id>'.stopped_uniforms (N error(s))` line. `model_salvage`
needs NO change, confirming the parent. One correction the parent's phrasing hides: the drop is of the
WHOLE list, not element-level. `drop_invalid`'s per-element branch (`model_salvage.py:72-84`) only
descends into elements that are `dict`s; a `str` element is passed through untouched, and the
top-level `validate_assignment` (`:91`) then rejects the list. So a hypothetical MIXED list (some
pairs, some stale strings) loses the valid pairs too. That is the right outcome for a stop set (the
cost is that the user re-stops a uniform) and it needs no code, but it is not what "element-level
salvage" describes, so it is stated rather than assumed.

Because every file on disk holds `[]`, **the first launch after W-G logs no salvage line at all**.
The parent's manual-verification bullet expects one line per stale `projects/dev` document; there are
no stale documents. § Manual verification carries the corrected expectation and how to see the line
deliberately.

### 14. Docs

- `conventions.md`'s scripting entry (the bullet beginning "The CPU-script engine is headless
  `ProjectSession` code", `:289-338`) is rewritten in three places, not replaced: the `EngineNode`
  clause becomes `ScriptTarget` (a document's passes, not one pass's uniform_values); the
  `update(self, ctx) -> dict[str, value]` clause becomes D3's grammar with the dict-value invariant
  named as what the dispatch rests on; and the freeze-granularity sentences move from
  `(document_id, name)` to `(document_id, pass, name)` with the `""` pass for a document-level error.
  The "orphan/typo/sampler key records a soft error + skip" clause drops "+ warn-once" and says the
  strip shows it on the script tab and on the named pass's shader tab.
- `conventions.md`'s PLAY/STOP entry (`:314-338`) is rewritten from "document-scoped + name-keyed" to
  "document-scoped + `(pass, name)`-keyed", with `stopped_uniforms: list[StoppedKey]` and the reason
  (the same name on two passes is two rows). The lazy-row argument, the `list`-not-`set` reason and
  the auto-stop-on-grab clause are unchanged and stay.
- `065_pass_graph/01_spec.md` D12 gains, immediately under its heading:
  `**Superseded by 069 D3.** One script per DOCUMENT, addressing passes by name in the returned dict.
  The addressing hole this decision names is real and is closed by the dict's shape rather than by a
  file per pass; the per-pass file was never implemented.` The body is left intact: its diagnosis is
  correct and is why D3 exists.
- `068_radiance_cascades/01_spec.md` D7 gains, under its heading:
  `**Retraction lifted by 069 (W-G).** The script engine now addresses a named pass and `ctx.mouse`
  carries `down` plus the previous position, which is exactly the trigger this retraction records.
  The tutorial's paint step is rewritten against them in 069 W-H.` Its two stated reasons stay: they
  are the record of why the retraction was right at the time.
- `help_content.py`'s `your_uniforms` section (`:136-139`) says a document "can carry a Python script
  that drives its uniforms". It gains one sentence naming the grammar: a script drives a uniform on
  every pass that declares it, or names a pass to drive only that one. Within D1's budget, so one
  sentence, no snippet change.

## Worked example

A two-pass document. `paint` declares `uniform float u_time_scale;` and `uniform float u_brush;`.
`composite` declares `uniform float u_time_scale;` and no `u_brush`. Both passes are compiled and
ready. The script returns:

```python
{"paint": {"u_brush": 0.5}, "u_time_scale": 0.5}
```

**What happens.** `u_time_scale`'s value is `0.5`, not a `dict`, so it is a broadcast; `"paint"`'s
value is a `dict`, so it is a pass block. Broadcasts run first:

- `u_time_scale` is looked up across both passes. `paint` declares it and `composite` declares it, so
  both are written: `paint.uniform_values["u_time_scale"] = 0.5` and
  `composite.uniform_values["u_time_scale"] = 0.5`. Coercion runs twice, once against each pass's own
  uniform. Both `("paint", "u_time_scale")` and `("composite", "u_time_scale")` enter `driven`.

Then blocks:

- `paint` exists, so `u_brush` is resolved against `paint`'s active map alone.
  `paint.uniform_values["u_brush"] = 0.5`, and `("paint", "u_brush")` enters `driven`.
  `composite` is never consulted for `u_brush` and records nothing about it.

**Result, precisely:** `paint` receives `u_time_scale = 0.5` and `u_brush = 0.5`; `composite`
receives `u_time_scale = 0.5` and nothing else. Three driven pairs. The strip is empty. The Document
tab shows a stop button on `u_brush` and on `u_time_scale` while the panel is on `paint`, and on
`u_time_scale` alone while the panel is on `composite`; stopping `u_time_scale` on `composite`
freezes it there and leaves `paint`'s playing, which is the case a flat name map could not express.

**Misspelled.** The script returns `{"paint": {"u_brsh": 0.5}, "u_time_scale": 0.5}`.

`u_time_scale` broadcasts exactly as above; nothing about the broadcast changes, which is the point of
per-key errors. Inside the `paint` block, `u_brsh` is absent from `paint`'s active map, so
`_binding_reject` returns a reason and the engine records
`ScriptError(uniform_name="u_brsh", kind="runtime", message="pass 'paint' has no active uniform 'u_brsh' (orphan key)", pass_name="paint")`
under `("<doc>", "paint", "u_brsh")`, adds the pair to `skipped`, and writes nothing.

The **script tab's** strip shows one row:

```
paint.u_brsh: pass 'paint' has no active uniform 'u_brsh' (orphan key)
```

The **`paint` shader tab's** strip shows the same error under its own compile errors, without the
`paint.` prefix (the tab already names the pass):

```
u_brsh: pass 'paint' has no active uniform 'u_brsh' (orphan key)
```

Clicking either row opens the script at the line. The **`composite` shader tab** shows nothing: the
error's `pass_name` is `paint`. Fixing the spelling clears both on the next tick, via the existing
stale-clear (a key touched last frame and not this frame has its error popped).

**Contrast, the bare-key misspelling.** `{"u_brsh": 0.5}` with no block. No pass declares `u_brsh`,
so the error is `no pass declares 'u_brsh' (orphan key)` with `pass_name=""`, recorded under
`("<doc>", "", "u_brsh")`. It appears on the script tab and on NO shader tab, because it belongs to no
pass. That asymmetry is deliberate: a bare key is a claim about the whole document, so the document's
own script is where it is reported.

## Files touched

The parent's list, verified against `ccd446b`, one clause each.

| File | Change |
|---|---|
| `shaderbox/scripting/engine.py` | `EngineNode` → `ScriptPass` + `ScriptTarget`; the two-phase routing in `_tick_script`; `(document, pass, name)` error keys and `(pass, name)` driven/skipped/last-good; `script_stub_for` over a per-pass mapping; the `logger.warning`, `warned` and `warn=` parameter deleted; `_freeze` and `_binding_reject` take the pass. |
| `shaderbox/scripting/context.py` | `MouseState` gains `down`, `prev_x`, `prev_y`; `EXPORT_MOUSE` states them. |
| `shaderbox/scripting/behavior.py` | `coerce_one` rejects a `dict` before `normalize_output`, with the pass-block message. |
| `shaderbox/scripting/errors.py` | `ScriptError` gains `pass_name: str = ""`. |
| `shaderbox/scripting/api_doc.py` | the contract bullet states D3's grammar; the mouse gloss covers `down` and the previous position, keeping the freeze caveat verbatim. |
| `shaderbox/scripting/__init__.py` | `EngineNode` out of the imports and `__all__`, `ScriptPass` / `ScriptTarget` in. |
| `shaderbox/scripting/outputs.py` | **no change**, contrary to the parent's file list; `normalize_output` passes an unknown type through and the dict rejection lives in `coerce_one`. |
| `shaderbox/project_session.py` | the **seven** `.render_pass` seams pass the document (`:430`, `:511`, `:555`, `:577`, `:599`, `:683`, `:687`); `_scriptable_uniforms_for` returns a per-pass mapping; `_stopped_for` builds `frozenset[StoppedKey]`; `uniform_is_driven` / `is_uniform_stopped` / `set_uniform_stopped` / `get_script_driven_uniforms` take or return the pass. |
| `shaderbox/ui_models.py` | `StoppedKey` added; `UIDocumentState.stopped_uniforms: list[StoppedKey]`. |
| `shaderbox/widgets/uniform.py` | the panel pass's NAME threaded into the four driven/stopped calls; `_draw_play_stop` takes it; the toggle id is per pass. |
| `shaderbox/ui.py` | the hit-test fills `down` / `prev_x` / `prev_y` and the re-entry flag; the `Clear` ghost button at the preview's top-left. |
| `shaderbox/commands.py` | `CommandId.RESET_FEEDBACK` + its `CommandSpec` on F6 in `C.DOCUMENT`. |
| `shaderbox/app.py` | `reset_current_document_feedback` + its callback binding; `script_mouse_inside`; the two stopped wrappers' arity. |
| `shaderbox/document.py` | `has_feedback` property over `plan_passes(self.graph)[0].feedback` for the button's visibility gate; `reset_feedback` itself unchanged. |
| `shaderbox/tabs/code.py` | `_script_errors_for` renders the pass; `_script_errors_for_pass` added and concatenated into the shader-tab branch. |
| `shaderbox/tabs/document.py` | **no change**; § Verified / corrected premises records why the parent's `:253` cite does not land here. |
| `shaderbox/copilot/backend.py` | `_pass_views` filters the driven set per pass; the three output-pass formatters filter to the output's name; the `set_uniform` gate asks the output pair; the probe's driven/error lists render the dotted form; `_motion_verdict` / `_uniform_changes` take the pair. |
| `shaderbox/copilot/capabilities.py` | field comments only; `read_script` / `write_script` / `apply_script_edit` unchanged in shape. |
| `shaderbox/copilot/prompt.py` | **no change**; the RARE tier renders whatever `script_api_summary()` returns. |
| `shaderbox/copilot/prompt_context.py` | **no change**; it already calls the generator. |
| `shaderbox/help_content.py` | one sentence on the script grammar in `your_uniforms`. |
| `ai_docs/conventions.md` | the scripting entry and the PLAY/STOP entry rewritten per § Design decisions item 14. |
| `ai_docs/features/065_pass_graph/01_spec.md` | D12 gains its "superseded by 069 D3" line. |
| `ai_docs/features/068_radiance_cascades/01_spec.md` | D7 gains its "retraction lifted by 069" line. |
| the eleven tracked `document.json`s | verified, zero bytes changed (item 13): seven under `shaderbox/resources/document_examples/`, two under `projects/documents/`, two under `projects/dev/documents/`. |
| `projects/dev/documents/*/scripts/` | nothing to rewrite; neither dev document has a `scripts/` dir. |
| `scripts/smoke.py` | the 048 stopped-skip canary passes the pass name to `set_uniform_stopped` and compares `("main", "u_a")` against the pair-shaped driven set. **This file is inside `make gates`.** |
| `scripts/dogfood/verify_script_engine.py` | the driven-set assertion becomes `{("main", "u_wave")}`; the `drop_document` assertion compares against the empty set and is shape-agnostic. |
| tests | § Tests. |

## Tests

Each names its falsifier: the mutation that must turn it red.

### `tests/test_script_engine.py` — the routing table

The table is `(bare, nested)` x `(declared in one pass, in two, in none)`, six cases, as one
parametrized test over a two-pass fake document (`paint` declares `u_a` and `u_b`; `composite`
declares `u_a` only):

| Return | Expected |
|---|---|
| `{"u_a": 1.0}` (bare, two passes declare) | both passes written; two driven pairs; no error |
| `{"u_b": 1.0}` (bare, one pass declares) | `paint` written, `composite` untouched; one driven pair; no error |
| `{"u_z": 1.0}` (bare, none declare) | nothing written; one soft error `no pass declares 'u_z'`, `pass_name == ""` |
| `{"paint": {"u_a": 1.0}}` (nested, declared there) | `paint` written, `composite` untouched |
| `{"composite": {"u_b": 1.0}}` (nested, not declared there) | nothing written; soft error `pass 'composite' has no active uniform 'u_b'`, `pass_name == "composite"` |
| `{"nope": {"u_a": 1.0}}` (nested, no such pass) | nothing written; soft error naming the pass and listing `composite, paint` |

**Falsifier:** route every key to `document.passes[next(iter(...))]` (the old output-only behaviour).
Rows 1, 2, 4 and 5 go red on the write side and rows 3 and 6 on the error side, so no single wrong
implementation passes the table.

### `tests/test_script_engine.py::test_a_pass_block_beats_a_broadcast_on_that_pass`

Return `{"u_a": 1.0, "paint": {"u_a": 2.0}}`. Assert `paint.uniform_values["u_a"] == 2.0` and
`composite.uniform_values["u_a"] == 1.0`. Then assert the same with the dict's keys in the other
insertion order (`{"paint": {...}, "u_a": 1.0}`) and the SAME result.

**Falsifier:** collapse the two phases into one loop over `raw.items()`. The second half goes red,
because insertion order then decides the winner. This is the test that pins the two-phase shape
rather than the outcome.

### `tests/test_script_engine.py::test_an_unknown_pass_names_the_pass_and_lists_the_real_ones`

Assert the message contains `'nope'` and both real pass names, and that the OTHER keys in the same
return still drove. One key's failure must not cost its siblings, which is the freeze-granularity
rule the entry in `conventions.md` states.

**Falsifier:** raise on an unknown pass instead of recording. The sibling half goes red (nothing is
driven) and the message half goes red with it.

### `tests/test_script_engine.py::test_a_not_yet_compiled_pass_is_held_not_errored`

A fake pass with `script_ready = False` named by a block. Assert: no error recorded, nothing written,
and after flipping the flag the next tick writes and clears.

**Falsifier:** treat a not-ready pass as absent. The no-error assertion goes red, which is the
first-frame noise this guard exists to prevent.

### `tests/test_script_engine.py::test_coerce_one_rejects_a_dict`

Call `coerce_one({"x": 1}, uniform, "u_a")` directly and assert `_RuntimeScriptError` whose message
names the pass-block grammar. Then, through the engine, `{"paint": {"u_a": {"x": 1}}}` records a
per-key error under `("<doc>", "paint", "u_a")` and freezes `u_a` at last-good.

**Falsifier:** delete the `isinstance(value, dict)` branch. The direct call goes red immediately;
without it the nested case would coerce a `dict` into a murky shape hint instead of the grammar
error, which is the state D3's invariant is unverified in.

### `tests/test_script_engine.py::test_the_stub_has_one_block_per_pass`

`script_stub_for({"paint": [u_a, u_b], "composite": [u_a], "empty": []})` and assert: `"paint"` and
`"composite"` and `"empty"` each appear as a commented block header, `u_b` appears only under
`paint`, the `empty` block says it has no scriptable uniforms, the bare-key rule appears in the
comment, and the whole stub still parses (`ast.parse`) and still returns `{}` when exec'd.

**Falsifier:** emit only the first pass's block. The `composite` and `u_b` assertions go red. The
`ast.parse` half falsifies the separate error of emitting comment text that is not valid Python.

### `tests/test_script_engine.py::test_the_orphan_warning_no_longer_reaches_the_console`

Install a loguru sink for the duration (`handle = logger.add(records.append, level="WARNING")`,
removed in a fixture teardown), drive an orphan key, and assert the sink captured nothing from
`shaderbox.scripting.engine` while `script_status().soft_errors` carries the row.

**The mechanism has to be a real sink.** The repo has no existing log-assertion idiom (zero `caplog`,
zero `loguru` references anywhere under `tests/`, and `conftest.py` sets up no logging), so this test
introduces one, and `caplog` is not it: this repo logs through loguru, which does not propagate into
the stdlib `logging` tree pytest's `caplog` reads, so a `caplog` assertion would pass vacuously with
the `logger.warning` still in place. A test that cannot go red under its own stated falsifier pins
nothing.

**Falsifier:** keep the `logger.warning`. Goes red on the first half. This is the test that pins
#29's own last sentence.

### `tests/test_script_engine.py::test_a_stopped_pair_freezes_only_that_pass`

Both passes declare `u_a`, the script broadcasts it, and `stopped = {StoppedKey("composite", "u_a")}`.
Assert `paint` advances across two ticks and `composite` holds its manual value, and that BOTH pairs
still report as driven (a stopped uniform keeps its play button).

**Falsifier:** key the stopped set by name. `paint` freezes too and the test goes red on its first
assertion. This is the defect the pass-qualified stop set exists to prevent.

### The three `reload` seams and the two script files inside the gates

`project_session.py:430` (`_resolve_scripts`), `:511` (`_load_one_document_from_disk`) and `:577`
(`reload_scripts`) have no dedicated test. Their falsifier is **`make check`**, the first gate step:
`reload`'s parameter becomes `ScriptTarget`, so a site still passing `.render_pass` (a `Pass`) is a
pyright error. That is a real falsifier and it is why a prose-driven sweep that fixes only the three
seams the parent names still cannot ship.

Two non-test files carry the same re-key and are named here because `make gates` runs them:

- `scripts/smoke.py` holds the 048 stopped-skip canary. `:306` and `:322` compare `"u_a"` against
  `script_driven_uniforms(...)`, and `:317` calls `set_uniform_stopped("script_document", "u_a", True)`
  with the old arity. Left alone, `:306`'s assertion fails before `:317`'s `TypeError` is even
  reached, and the smoke step of `make gates` goes red. This is the third place the pair re-key must
  land, beside `test_script_engine.py` and the dogfood check.
- `scripts/dogfood/verify_script_engine.py:97` asserts `driven == {"u_wave"}`, an equality against a
  name set, which goes red the moment `get_script_driven_uniforms` returns pairs. Its sibling at
  `:145` compares against the empty set and would survive by accident, which is worse: it keeps
  passing while `:97` is broken.

**Budget for `test_script_engine.py`.** It is 1088 lines, 61 tests, with 77 engine call sites. The rewrite is
mostly mechanical and concentrated in three helpers: `_FakeDocument` becomes a `passes` mapping
(`_FakeDocument({"main": [u...]})`, with a one-pass default so most constructions read as they do
today), `_engine` and the `tick` / `tick_export` / `dry_run` calls take the document rather than the
pass, and assertions of the form `document.uniform_values["u_x"]` become
`document.passes["main"].uniform_values["u_x"]`. Scripts returning `{"u_x": ...}` need NO edit at all
under the broadcast rule: a single-pass document that declares `u_x` receives it. That is the reason
the broadcast rule keeps this rewrite finite, and it is worth stating as a design consequence, not
only a test convenience. Expected: every test body edited at the assertion line, roughly a dozen
edited more deeply (the play/stop, orphan, status and stub groups), plus the eight new tests above.

### `tests/test_script_engine_gl.py`

Six tests, fifteen engine call sites, all `document.render_pass` (a real `Document`). Since a real
`Document` now satisfies `ScriptTarget` directly, each call drops `.render_pass` and the assertions
gain `.passes["main"]` (or keep `render_pass`, which for a one-pass document IS that pass, so most
assertions read unchanged). One test is ADDED: a real two-pass `Document` where a broadcast reaches
both passes' GPU-visible uniform values, because the whole point of the wave is the multi-pass path
and the GL file is where "the write reaches the GPU" is verified.

**Falsifier for the new one:** route to the output pass only; the non-output pass's pixel does not
change.

### `tests/test_script_dry_run.py`

Ten tests, twelve call sites. `dry_run` takes the document; `probe.driven` is a set of pairs, so the
assertions become pair sets. One test is added: a dry-run over a two-pass document reports the pass in
`orphan_keys`, since the copilot's write feedback is the only place a headless caller learns the
routing verdict.

**Falsifier:** report the bare name; the added assertion goes red.

### `tests/test_export_script_wiring.py`

One test, one call site. It asserts the export pre-render hook ticks a fresh behavior; it now asserts
the hook is handed the DOCUMENT and that a two-pass document's non-output pass receives its value
during an export. Plus `test_export_mouse_is_down_false_and_prev_equals_current`: assert
`EXPORT_MOUSE.down is False` and `(EXPORT_MOUSE.prev_x, EXPORT_MOUSE.prev_y) == (EXPORT_MOUSE.x, EXPORT_MOUSE.y)`,
and that a script gating on `ctx.mouse.down` drives nothing through `tick_export`.

**Falsifier:** default `down=True`, or wire the live cursor into the export context. Both halves go
red, which is what keeps an export deterministic.

### `tests/test_copilot_script_tools.py`

Thirteen tests, no direct engine calls: they drive `write_script` / `read_script` through the backend.
The edits are to the expected strings (the dotted `pass.uniform` form in `driven` and `orphan_keys`)
and one added test: a two-pass document's `_pass_views` marks `<driven by script.py>` on the pass that
declares the driven uniform and NOT on the sibling that declares the same name.

**Falsifier:** keep the document-scoped set in `_pass_views`. The sibling assertion goes red. This is
the test for the one site the parent says the change genuinely fixes.

**It is the FIRST test of that code path, not one added to an existing group.** `_format_uniforms`'s
driven branch (`backend.py:257`) and `_pass_views`'s driven binding (`:738`) have no coverage today in
any file: `tests/test_working_set.py:127` asserts the `<driven by script.py>` marker STRING but builds
`WorkingSetView.uniforms` by hand, and its own comment says why ("the marker itself is built
backend-side (`_format_uniforms`); here we verify the working-set RENDER carries the script fields
it's given"), so it neither exercises nor breaks on the filter and needs no edit. The two-pass fixture is a
`SimpleNamespace` document, **not** `tests/test_document_graph.py`'s `_document`: that helper takes a
`moderngl.Context` and compiles, and its `gl_ctx` fixture skips without a standalone context, which
would put the wave's headline test behind a GL skip in a file that is GL-free by construction (a
skipped test is not a pass, the same rule `CLAUDE.md` states for the smoke step). `_pass_views` reads
only `graph`, `passes`, `source.text`, `compile_unit.errors`, `get_active_uniforms()` and
`uniform_values` (`backend.py:730-757`, `_format_uniforms` at `:241-262`), so a stub supplies all of
them — the same `__get__`-onto-a-light-stub idiom `tests/test_script_driven_reject.py` already uses on
this backend.

### `tests/test_script_driven_reject.py`

Two tests, no engine calls. The injected `get_script_driven_uniforms` fake returns pairs, so both
fakes change shape; the assertions (that `set_uniform` rejects a script-driven uniform) are unchanged
in meaning. One added: `set_uniform` does NOT reject a uniform driven on a NON-output pass, because
the tool addresses the output pass and rejecting there would block a legitimate manual edit.

**The FIXTURE breaks, not only the fake, and this is consequent on § Design decisions item 7's gate
move.** `:22-24` passes a bare `object()` as the document, with the comment "the document only needs
to EXIST; the reject returns before .document" — true today, false once the gate resolves the output
pass name below the `target =` bind. Both stubs therefore grow a real document:
`SimpleNamespace(graph=SimpleNamespace(output_pass="main"), render_pass=SimpleNamespace(source=SimpleNamespace(path=Path("passes/main.frag.glsl"))))`,
and the stale comment goes with it. The added non-output test cannot be written against the current
`_stub` helper either, for the same reason.

**Falsifier:** test the name against the whole document's pairs. The added test goes red.

### `tests/test_script_api_doc.py` — the prompt block pin

`test_summary_lists_every_ctx_field_and_the_mouse_subfields` already loops
`MouseState.__dataclass_fields__`, so it covers `down` / `prev_x` / `prev_y` with **no edit** and goes
red the moment a field is added without the gloss following. `test_the_mouse_gloss_carries_the_frozen_at_center_caveat`
also needs no edit (the caveat substring survives). Two are added:

- `test_the_contract_bullet_states_both_addressing_forms`: the rendered summary contains a bare-key
  sentence and a `{pass: {uniform: value}}` sentence, and states that a pass block wins.
- `test_the_mouse_gloss_states_the_button_and_the_previous_position`: `down` and the previous position
  are described, not merely listed as field names, since a bare field list tells the agent a name and
  not a meaning.

**Falsifier for both:** revert the two `api_doc.py` prose edits. Each goes red on its own sentence.
The falsifier for the no-edit pin is `MouseState` gaining a field with `_CTX_GLOSS` untouched, which
`test_ctx_gloss_keys_are_exactly_the_dataclass_fields` catches at the ctx level.

### `tests/test_ui_models.py::test_a_stale_string_stopped_set_drops_to_empty`

Load a `UIDocumentState` dict carrying `{"stopped_uniforms": ["u_x"], "all_stopped": True}` through
`drop_invalid` + construction. Assert `stopped_uniforms == []` AND `all_stopped is True` (the sibling
key survives, which is the whole point of per-key salvage), and that a well-formed
`[{"pass_name": "paint", "name": "u_x"}]` round-trips.

**Falsifier:** any migration code that reinterprets a bare string as a pair. The first assertion goes
red, and it is the test that makes the no-migration rule mechanical rather than a promise.

### `tests/test_command_routing.py`

No edit for chord uniqueness (it enumerates `COMMAND_SPECS` and F6 collides with nothing). W-E's
disjointness test likewise reads the keymap docs and F6 appears in neither. The one added assertion:
`RESET_FEEDBACK`'s chord is a standalone key that passes `chord_needs_modifier`, so a bare F-key
binding is legal rather than accidentally accepted.

**Falsifier:** bind it to a bare letter; `chord_needs_modifier` returns True and the assertion goes
red.

### `tests/test_persistence_completeness.py`

**No edit.** Its roster covers the four app-data stores and exempts `document.py` explicitly ("loaded
per document with its own skip-and-warn path"), so `UIDocumentState` is outside its battery by design
and the new `test_a_stale_string_stopped_set_drops_to_empty` is where the document path is covered.
The roster-completeness test greps for `json.load` and `ui_models.py` is already rostered, so adding
`StoppedKey` changes nothing it asserts.

## Manual verification (the maintainer, in the app)

The parent's W-G line, expanded:

1. On a two-pass document where both passes declare `u_time_scale` and only `paint` declares
   `u_brush`, write the worked example's script. Both passes' `u_time_scale` move together; `paint`'s
   `u_brush` moves and `composite` is unaffected. Misspell `u_brush` and the strip on the script tab
   reads `paint.u_brsh: pass 'paint' has no active uniform 'u_brsh' (orphan key)`; open the `paint`
   shader tab and the same error is there without the prefix; open `composite`'s and it is not.
   Fix the spelling and both clear on the next frame. On a document opened COLD, expect the play/stop
   buttons to appear one pass per frame rather than all at once: a pass that has never attempted a
   compile is held for that tick and the first-render sweep compiles one per document per frame, so a
   six-pass document is fully driven around frame six. That is the known first-frames behaviour of
   § Design decisions item 2, not a defect.
2. Stop `u_time_scale` while the Document panel is on `composite`; drag it by hand. `paint`'s
   `u_time_scale` keeps moving. Switch the panel to `paint` and its row still shows as playing.
3. Paint: hold LMB and drag over the canvas. The stroke is continuous, not a row of blobs. Release and
   move the cursor over the canvas: nothing is painted. Then drag off the LEFT edge mid-stroke, move
   to the RIGHT edge off-canvas, and re-enter there: the stroke must end at the left edge and restart
   at the right, with no stamp along the line between them. A full-width streak is the
   `script_mouse_inside` flag being unread. This is the only step whose mechanism has no other check,
   which is why the observation is specified as a full-width traverse rather than a short hop.
4. Press F6 (or click `Clear` at the preview's top-left): the canvas empties. Confirm the button is
   absent on a document with no feedback pass, and that F6 there is a silent no-op rather than an
   error.
5. Export the painted document. The video is deterministic: nothing is painted in it, because
   `EXPORT_MOUSE.down` is False. Re-export and the two files match.
6. **The salvage line.** The first launch after W-G logs NOTHING about `stopped_uniforms`, because
   all eleven tracked `document.json` files hold `[]` (or omit the key), which is valid under the new
   shape. This corrects the
   parent's expectation of one line per `projects/dev` document. To see the line deliberately (which
   is worth doing once, to confirm the fail-soft path is live rather than assumed): hand-edit one dev
   document's `stopped_uniforms` to `["u_x"]`, launch, and read
   `Ignoring invalid document '<id>'.stopped_uniforms (1 error(s))` once, with the document otherwise
   intact (its name, its tuned uniforms, its `all_stopped`). Then restore the file and
   `git add projects/dev`.
7. Ask the copilot to write a script driving a uniform on a non-output pass. Its `write_script` result
   names `paint.u_brush` in `driven`, and the pass listing shows `<driven by script.py>` on `paint`'s
   row and a real value on `composite`'s row for the same uniform name.

## Verified / corrected premises

Every citation and claim the parent spec's W-G section, the design note, and findings #22 #23 #29 #30
make, checked against `ccd446b`.

| Parent-spec, design-note or finding citation | Verdict |
|---|---|
| `ScriptEngine.tick` / `tick_export` / `dry_run` / `reload` all take a pass and every caller hands them `.render_pass` | **Confirmed.** `engine.py:370` `tick`, `:397` `tick_export`, `:421` `dry_run`, `:281` `reload`, each typed `document: EngineNode`; the four `project_session.py` callers pass `ui_document.document.render_pass` (`:430`, `:510`, `:555`, `:577`, `:599`, `:683`, `:687` — seven sites, not four; see the next row). |
| "every caller in `project_session.py` — `tick`, the export pre-render closure, `write_script_source`'s reload + dry_run — stops passing `.render_pass`" (three named) | **Corrected: there are seven `.render_pass` arguments to the engine, across five methods.** `_resolve_scripts` (`:430`) and `_load_one_document_from_disk` (`:510`) each pass it to `reload`; `reload_scripts` (`:577`) passes it to `reload`; `_make_export_isolation`'s closure (`:555`) passes it to `tick_export`; `tick` (`:599`) passes it to `tick`; `write_script_source` passes it to `reload` (`:683`) AND to `dry_run` (`:687`). The parent's three named sites are the three the wave's PROSE is about; a mechanical sweep must find all seven or two reload paths keep the old shape and a freshly-loaded document routes against one pass. |
| "the live tick (`project_session.py:598`) is the fourth `.render_pass` seam and the one whose protocol type changes" | **Confirmed as the claim, corrected as the line.** `:598` is `document_id,`; the `.render_pass` argument is `:599`, inside `ProjectSession.tick` whose engine call opens at `:597`. |
| `stopped_uniforms` is a set of names at `project_session.py:604` and would freeze a name on every pass (design note) | **Confirmed as the claim, corrected as the line and the type.** `:604` is `def _stopped_for`, whose body reads `state.stopped_uniforms` at `:613`; the FIELD is `ui_models.py:163`, `stopped_uniforms: list[str] = []` (a list, not a set — the type's own comment explains why: `model_dump` → `json.dump` raises on a Python set). The freeze-every-pass consequence holds exactly. |
| The stub reads the output pass only, `_scriptable_uniforms_for` at `:632` (design note, #29) | **Confirmed.** `:632` is the `def`, and `:640` is `].document.render_pass.get_active_uniforms()`. Its two callers are `create_script` (`:652`) and `read_script_source` (`:669`), both user/agent-triggered, which is what makes the per-pass version's compile cost acceptable. |
| `EngineNode` is the protocol to retire (parent, W-G bullet 1) | **Confirmed.** `engine.py:95`, two members (`uniform_values`, `get_active_uniforms`), referenced as a type at `_freeze` (`:106`), `reload` (`:281`), `tick` (`:373`), `tick_export` (`:400`), `dry_run` (`:423`), `_tick_script` (`:516`), and exported from `scripting/__init__.py`. Seven annotation sites plus the export. |
| The `EngineNode` `stopped: frozenset[str]` parameter type becomes pair-keyed (parent) | **Corrected: `stopped` is not a member of `EngineNode`.** It is a parameter of `ScriptEngine.tick` (`:375`) and of `_tick_script` (`:524`), typed `frozenset[str]`. The protocol has exactly two members and neither is `stopped`. The change the parent wants is real and lands on those two signatures. |
| `DocumentScripts.last_driven` / `last_skipped` become `(pass, name)`-keyed | **Confirmed.** `engine.py:130` and `:134`, both `set[str]`. `last_good` (`:125`) and `warned` (`:137`) are keyed the same way and the parent does not name them; both must move too, or a per-key freeze resolves the wrong pass's last-good. |
| The engine's soft error gains a field recording the pass, "not only the key" | **Confirmed as needed and as absent.** `ScriptError` (`errors.py:9`) has `uniform_name` / `kind` / `message` / `line` and no pass. Adding a defaulted field leaves all six existing construction sites compiling. |
| `tabs/code.py:130` is the script tab's soft-error strip | **Confirmed.** `:130` is `def _script_errors_for`, and its `soft_errors` loop is `:141-142`. The selection that routes a `script` tab to it is `:517-523`, inside `tabs.code.draw`. |
| "today only the script tab shows them" | **Confirmed.** The `errors` expression at `:517` gives a shader tab `edited.compile_unit.errors` and nothing else; `_script_errors_for` is reached only under `tab.kind == "script"`. |
| `copilot/backend.py:740` is `_pass_views`, looping every pass against one document-scoped driven set | **Confirmed as the claim, corrected as the line.** `def _pass_views` is `:730`, the driven set is bound at `:738`, and `:740` is the `for name in sorted(document.passes):` that the claim is really about. The defect is exactly as described: one set, every pass, so a uniform driven on one pass is marked driven on all. |
| `copilot/backend.py:889` is the `set_uniform` gate, resolving against `render_pass` only, and stays name-keyed | **Confirmed as the claim, split across two lines.** The driven-set REJECT is `:879`; `:889` is `(u for u in target.render_pass.get_active_uniforms() if u.name == name)`, the resolution the claim is about. Both are inside `set_uniform` and both are output-pass-scoped, so the verdict (stays name-keyed within that pass) holds for both. |
| `_pass_views` is the only site the change genuinely fixes | **Corrected: there are three more `_format_uniforms` callers, and they need the same filter for a different reason.** `:637-647` (the single-pass member listing) and `:720-721` (`_copilot_document_working_view`) both format the OUTPUT pass with the document-scoped set, so a uniform driven only on `paint` is currently shown as `<driven by script.py>` on the output pass's listing even when the output does not declare it driven. They are not the headline defect but they are wrong by the same mechanism, and filtering to the output pass's name fixes them in the same edit. |
| `app.py:1414-1435` is the `App.set_uniform_stopped` / `set_document_all_stopped` wrappers | **Refuted.** `:1414-1435` is inside `get_current_session_if_exists` and `flush_current_editor` (editor-session code, no stopped state anywhere in it). The wrappers are `app.py:1489` (`set_uniform_stopped`) and `:1496` (`set_document_all_stopped`), under the comment `# --- Script UI (feature 048): thin App-side wrappers over the headless ProjectSession ---` at `:1488`. `toggle_current_document_play` (`:1503`) is a third caller the parent does not name; it calls `set_document_all_stopped`, whose arity is unchanged, so it needs no edit. |
| `tabs/document.py:253` is "the panel's stop call" | **Refuted.** `:253` is a blank line between the `sort_uniform_hashes` call (`:247-252`) and the `begin_child("ui_uniforms")` that draws the rows (`:257`). `tabs/document.py` contains no `set_uniform_stopped` / `is_uniform_stopped` / `uniform_is_driven` call at all. Every per-uniform stop call is in `widgets/uniform.py`: `uniform_is_driven` (`:187`), `is_uniform_stopped` (`:188`), `set_uniform_stopped` in the auto-stop-on-grab (`:302`) and in `_draw_play_stop` (`:169`). The file needs no edit in this wave, and the pass name comes from `app.panel_pass(...)` which `widgets/uniform.py` already binds at `:177`. |
| `widgets/uniform.py`'s play/stop button "passes the panel pass" | **Confirmed as the intent, with the mechanism supplied.** `draw_ui_uniform` binds `panel_pass` at `:177`; `Pass` carries `source.path`, not a name, so the NAME is `pass_name_of(panel_pass.source.path)` (`paths.py:25`). The parent says "pass", and the call sites need a `str`. |
| `widgets/uniform.py`'s "driven-name tint" (parent's file list) | **Corrected: it is a playing tint, not a driven one, and it needs no pass.** `_begin_ctrl(app, name, ..., playing=playing)` (`:191`) takes `playing`, which is already computed pass-correctly once `driven`/`stopped` are. The tint itself reads a bool and never a name, so it changes only by consequence. |
| `UIDocumentState.stopped_uniforms` becomes `list[[pass, name]]` pairs | **Confirmed as the shape, corrected as the encoding.** The parent's prose says "a list of `[pass, name]` pairs" and its next sentence says `list[StoppedKey]` with `StoppedKey` a `BaseModel`. The two are different on disk (a JSON array vs a JSON object) and the model form is the one that carries names, salvages per key and reads correctly in a hand-edited file. This spec takes the model form, which is the parent's own later sentence. |
| "a stale `list[str]` fails `validate_assignment` and drops to `[]` under the existing `drop_invalid` policy (`model_salvage` needs no change)" | **Confirmed by execution, with one clarification.** Run against a `list[StoppedKey]` field, `drop_invalid` leaves the string elements untouched (its element branch descends only into `dict`s, `:72-84`), then the top-level `validate_assignment` (`:91`) rejects the list and pops the key, which falls back to `[]` with one `Ignoring invalid ... (N error(s))` line; the sibling `all_stopped` survives. `model_salvage.py` needs no edit. The clarification: this is WHOLE-LIST salvage, not element-level, so a mixed list loses its valid pairs too. Acceptable (the cost is re-stopping a uniform) but not what "element-level salvage" describes. |
| The document ui_state load path runs `drop_invalid` | **Confirmed.** `ui_models.py:514`, `drop_invalid(UIDocumentState, filtered_ui_state, f"document '{dir_name}'")`, with the unknown-key filter at `:505` above it. So the fail-soft claim applies to `document.json` and not only to the app-data stores. |
| "Seven shipped examples exist; five persist `stopped_uniforms` and are hand-edited; the two that do not are exactly the multi-pass ones" | **Confirmed exactly, and the hand-edit is empty. The enumeration is not the whole tracked set.** Seven example dirs; `0b0d16bb`, `53724dbd`, `73ea2431`, `8d454b7b`, `f90f5ff9` carry `"stopped_uniforms": []` under `ui_state` and each has ONE pass; `1c4f8a20` (5 passes) and `77a84d27` (6 passes) omit the key entirely. But `git ls-tree -r ccd446b --name-only \| grep document.json` returns **eleven** files, not nine: `projects/documents/1901ab60-...` (5 passes) and `projects/documents/307598da-...` sit outside both directories the parent names and were missed by this spec's first draft. Both carry `[]`. Every persisted value across all eleven is `[]` or absent, so zero byte changes. |
| "`projects/dev` scripts + `document.json`s hand-edited to the new shape" | **Refuted on the scripts, and empty on the JSON.** `projects/dev/documents/` holds two documents (`e7e00c46`, `ec926580`), each with one pass and NO `scripts/` directory, so no script exists to rewrite. Both `document.json`s carry `"stopped_uniforms": []`, valid unchanged. The wave changes no bytes under `projects/dev`. |
| "First launch after W-G logs one salvage line per stale `projects/dev` document — expected, and the verification list says so" | **Refuted.** No `projects/dev` document is stale: both hold `[]`, which parses. The first launch logs nothing about `stopped_uniforms`. § Manual verification carries the corrected expectation plus a deliberate way to exercise the salvage path once. |
| `scripting/api_doc.py` "generates it from `MouseState`'s dataclass fields and the stub" | **Confirmed for the fields, refuted for the stub.** `_MOUSE_FIELDS` (`api_doc.py:61`) reads `MouseState.__dataclass_fields__` and `_ctx_fields` (`:96`) reads `EngineContext.__dataclass_fields__`, so both track the dataclasses. But `api_doc.py` does NOT read the stub: it may not, because `script_stub_for` lives in `engine.py` which imports `moderngl`, and `test_api_doc_reaches_only_for_the_gl_free_half_of_the_package` asserts `api_doc`'s imports are exactly `scripting.context` and `scripting.outputs`. The shared surface between them is `_VALUE_SHAPE_GLOSS`'s type names, checked against `_stub_kind`'s returns by AST in `test_every_stub_kind_type_name_has_a_value_shape_gloss` — a test-time join, deliberately not an import. So the contract bullet's new grammar sentence is AUTHORED prose in `api_doc.py`, not generated from the stub, and the pin is a test. |
| "`copilot/prompt_context.py` is its importer — both change" | **Corrected: `prompt_context.py` needs no edit.** `:9` imports `script_api_summary` and `:99` calls it into the `script_api` field; the block regenerates itself. `copilot/prompt.py` likewise renders whatever the field holds. Both are listed as touched by the parent and neither has a line to change. |
| "`copilot/capabilities.py` (read/write_script unchanged in shape)" | **Confirmed.** `read_script(document)` / `write_script(new_text, document)` / `apply_script_edit(...)` (`:386-390`) take no pass and return the same shapes. Only the field COMMENTS on `ScriptWriteResult` change (the dotted display form). |
| `ScriptProbe` / `ScriptStatus` carry the driven set and soft errors the UI and the copilot read | **Confirmed, and both are pair-shaped after the change.** `ScriptStatus` (`engine.py:56`) has `soft_errors: list[tuple[str, ScriptError]]`; `ScriptProbe` (`:66`) has `driven: set[str]`, `per_key_errors`, `orphan_keys` and `samples` keyed by name. The parent names neither dataclass; both are in the blast radius and are listed in § Design decisions item 5. |
| `MouseState` is position-only, sampled once per frame from the preview hit-test at `ui.py:612` (#22) | **Confirmed as the claim, corrected as the line.** `MouseState` is `context.py:7` with `x`/`y` only. The hit-test is `ui.py:654` (`item_normalized_mouse`), assigned at `:659` inside `_draw_document_image`; `:612` is inside that function's docstring at HEAD. The sole `MouseState` construction outside `EXPORT_MOUSE` is that one, and `App.script_mouse` (`app.py:1140`) is what `ProjectSession.tick` reads at `ui.py:241`. |
| `imgui.is_mouse_down(0)` is available for the `down` fill (#22) | **Confirmed.** Used already for the splitter drag path in `ui.py`; the hit-test at `:654` sits inside a window whose hover state `item_normalized_mouse` already consults, so the read needs no extra guard. |
| `Document.reset_feedback` is `document.py:327`, its only caller is export at `:650` (#23) | **Confirmed as the claim, corrected as both lines.** `def reset_feedback` is `document.py:357`; its only call is `render_media` (`:699`, `self.reset_feedback()` inside the `export_isolation` body opened at `:695`). The claim (nothing but export reaches it, no command, chord or button) holds exactly. |
| A `CommandSpec` on a bare F-key is legal with no registry change (`02_keybindings.md`) | **Confirmed.** `commands.py:287` `_STANDALONE_KEYS` and `:313` `chord_needs_modifier` exempt the F-keys; `F1` (`HELP`) and `F8` (`JUMP_NEXT_ERROR`) are existing bare-F-key bindings, so F6 needs no new mechanism. |
| `RESET_FEEDBACK` = F6 (parent open question 4, provisional) | **Confirmed by the audit.** `02_keybindings.md`'s table row reads `RESET_FEEDBACK (W-G, new) | none yet | app | app | app | **new** → F6, see note 3`, and note 3 supplies the reasoning (rule 3 forces Alt or an F-key because the verb must survive editor focus; F6 sits beside F5, which is where W-E moves `TOGGLE_DOCUMENT_PLAY`, making them the document's transport pair). The parent's provisional is not moved. |
| "a small ghost button by the preview" has an existing idiom to follow | **Confirmed.** `ui_primitives.py:97` `ghost_button`, and `ui.py:693` anchors the FPS chip to the preview's top-right via `set_cursor_screen_pos` from `cursor_pos` + `image_width`, which is the same anchoring mechanism the clear button uses on the opposite corner. |
| W-A's checkerboard lands under the preview before this wave | **Confirmed as already present at HEAD.** `ui.py:638` `_draw_canvas_backdrop(app.checker_texture, img_min, image_width, image_height)`, with the 1px border at `:646`. So the button's placement is decided against the final viewer, not a moving one. |
| `Document._feedback` answers "does this document have a feedback pass" | **Refuted, found by the design review.** `document.py:246-250` states it is "Allocated on demand by the first frame that needs one"; the only writer is `_feedback_canvas` (`:410`), reached from `render` (`:478`); `release()` (`:301`) and `drop_feedback` (`:382`) empty it, and `reset_feedback` (`:365`) empties it too, so a button gated on it would hide itself the moment it is clicked. The declared fact is `PassPlan.feedback` (`pass_graph.py:224`, filled at `:274` from an entry naming its own pass), which needs no render. Item 11 takes it. |
| `set_uniform`'s driven gate can be pass-qualified where it stands (`backend.py:879`) | **Refuted, found by the design review.** The document object is first bound at `:887` (`target = ...`), so no output pass name is in scope at `:879` and the change as first written does not compile. The gate moves below the bind (item 7), which also puts it after the `uniform is None` resolution, so a name absent from the output pass gets the "no active uniform" answer rather than the script-driven one. |
| A script-error row on a shader tab jumps to the script by carrying the script path | **Refuted, found by the design review.** `_draw_error_strip` (`tabs/code.py:218`) only latches `editor_jump_request`, and `_consume_jump` (`:187`) discards a request whose path is not the current tab's. Both existing cross-file jumps (`widgets/uniform.py:69-72`, `popups/lib_picker/filtering.py:97-98`) open the file first; item 8 now does the same via `app.open_script_for`. |
| `_drop_script`'s cleanup loop works on a wider tuple with no change | **Refuted, found by the design review.** `engine.py:323` iterates elements and composes `(document_id, stale)`; with pair elements that is a two-tuple containing a tuple, which never equals the three-tuple key, so every per-key error would survive a script deletion. It needs an unpack (item 5). |
| The repo has a `caplog`-based log-assertion idiom to follow | **Refuted, found by the tests review.** Zero `caplog` and zero loguru references anywhere under `tests/`, and `conftest.py` sets up no logging. loguru does not propagate into the stdlib tree `caplog` reads, so a `caplog` assertion would pass vacuously with the `logger.warning` still in place. The orphan-console test installs a real loguru sink and introduces the idiom. |
| `scripts/smoke.py` and `scripts/dogfood/verify_script_engine.py` are outside the blast radius | **Refuted, found by the tests review.** `smoke.py:306`, `:317`, `:322` and `verify_script_engine.py:97` all use the name-keyed driven set or the old `set_uniform_stopped` arity. `smoke.py` runs inside `make gates`, so leaving it out makes a green gate unreachable. Both are now in § Files touched. |
| `test_script_driven_reject.py`'s `object()` document stub survives the gate change | **Refuted, found by the tests review.** `:22-24`'s comment ("the reject returns before .document") is true only while the gate sits above the `target =` bind. Once it moves, both stubs need a real `document.graph.output_pass` and `render_pass.source.path`. |
| `_pass_views`'s driven filter has an existing test group to join | **Refuted, found by the tests review.** `_format_uniforms`'s driven branch (`backend.py:257`) and `_pass_views`'s binding (`:738`) have zero coverage; `tests/test_working_set.py:127` asserts the marker string over a hand-built `uniforms` list and neither exercises nor breaks on the filter. The new test is the first of that path and needs a two-pass `Document` fixture from `tests/test_document_graph.py`. |
| Two passes' same-named uniform rows collide on one imgui id today | **Refuted, and the two reviewers split on it.** `draw_ui_uniform` has exactly one call site (`tabs/document.py:262`), inside a loop over `sorted_hashes` built from `app.panel_pass`'s single pass, so the panel draws one pass per frame and the two rows are never submitted together. Resolved by reading that loop; open question 3 records both readings and which line settles it. |
| The parent's "largest test rewrite in the feature", seven files | **Confirmed as the ranking, quantified.** `test_script_engine.py` 1088 lines / 61 tests / 77 engine call sites; `test_script_engine_gl.py` 321 / 6 / 15; `test_script_dry_run.py` 251 / 10 / 12; `test_export_script_wiring.py` 67 / 1 / 1; `test_copilot_script_tools.py` 193 / 13 / 0 direct; `test_script_driven_reject.py` 64 / 2 / 0 direct; `test_script_api_doc.py` 172 / 10 / 0 direct. The broadcast rule is what keeps it finite: a one-pass fake document plus a `{"u_x": ...}` return needs no script-body edit. |
| `tests/test_persistence_completeness.py` is in the blast radius | **Refuted.** Its roster is the four app-data stores and it exempts `document.py` by name ("loaded per document with its own skip-and-warn path"); `UIDocumentState` is reached through `ui_models.py`, which is rostered as `app_state`, and adding a nested model changes nothing the roster-completeness grep asserts. No edit. |
| 065 D12 decided one script per PASS and was never implemented (#29) | **Confirmed.** `065_pass_graph/01_spec.md` D12 reads "One script per PASS, keyed `(document, pass)`. Corrected during review; the first draft said one script per document and that was wrong", and its files list names `scripting/engine.py` ("keys on a pair per D12 — NOT untouched, as the first draft claimed"). The code still has one `script.py` per document ticked against the output pass, so the decision is unlanded and the "superseded by 069" line is accurate as written. |
| 068 D7 is retracted for exactly the two reasons this wave removes | **Confirmed.** `068_radiance_cascades/01_spec.md` D7 reads "RETRACTED: the engine cannot deliver it", names `ProjectSession.tick` binding to the OUTPUT pass and `ctx.mouse` carrying position only, and states its trigger: "a script engine that can address a named pass rather than only the output". W-G is that trigger, so "lifts the retraction" is accurate. |
| `conventions.md`'s scripting entry says "error-as-data" and the PLAY/STOP entry says "document-scoped + name-keyed" | **Confirmed.** `conventions.md:308` ("A broken script is **error-as-data**") and `:314` ("**PLAY/STOP is document-scoped + name-keyed model state**"). Both bullets are inside one long entry beginning `:289`, so the rewrite is three surgical edits rather than a replacement. |
| The `ProjectSession` callback seam is the idiom for a core mutation with a UI reaction | **Confirmed and NOT needed here.** `conventions.md:279` names the `on_*` idiom for core mutations whose UI reaction touches imgui. `reset_feedback` is the opposite direction (a UI command calling into the core with no reaction), so it needs no callback; `App.reset_current_document_feedback` reaches `Document` directly the way `toggle_current_document_play` reaches `set_document_all_stopped`. |
| The persistence-evolution posture permits reshaping `stopped_uniforms` without migration | **Confirmed.** `conventions.md:711` states the posture is about a model staying LOADABLE, not about migrating data, and the bullet above it (`:705`) forbids migration code outright, naming 048's `u_*.py`→`script.py` migration as the recurring failure. A reshaped field that drops to its default on a stale file is exactly the sanctioned outcome. |
| `Pass.get_active_uniforms` compiles a never-attempted pass | **Confirmed, and it constrains the design in two ways, not one.** `core.py:236` compiles on demand, which is why 066 D1 forbids per-frame compiles and why the per-pass stub scan is acceptable only outside the frame loop (item 9). `core.py:232` adds the second constraint this spec's first draft missed: "A FAILED attempt is not retried — its errors stick in `compile_unit` until `invalidate()` resets it", so `program is None` is permanent after a failure and a `script_ready` meaning "the program is built" would hold a broken pass's keys silently forever. Item 2 splits never-attempted from failed on `self.program is not None or bool(self.compile_unit.error_raw)`, which is the NEGATION of `core.py:236`'s would-compile guard — round 2 caught the spec stating that guard verbatim, which inverts the member. |

Corrected or refuted: **31** of 52 rows (17 corrections, 14 refutations). Nine of those rows came
from the two pre-implementation reviews and say so in the verdict; the rest were derived against the
parent spec before review.

The four that change the work rather than a line number, from the first pass: `app.py:1414-1435` and
`tabs/document.py:253` both point at unrelated code and the real stop-call sites are all in
`widgets/uniform.py`; the `.render_pass` seams are seven, not the three the parent names, and two of
the four unnamed ones are reload paths a prose-driven sweep would miss; the hand-edit is zero bytes,
because every persisted value is already `[]`; and the salvage line the manual verification expects
will not appear.

The three the reviews added that change the work: `_feedback` is an allocation cache, so the Clear
button as first specified would have hidden itself on click; the `set_uniform` gate as first specified
did not compile, because it read a name that is not in scope until eight lines later; and two files
inside `make gates` (`scripts/smoke.py`, `scripts/dogfood/verify_script_engine.py`) carry the
name-keyed driven set, so the wave could not have reached a green gate as written. The instructive one
is `has_feedback`: it was derived from the field that HOLDS feedback rather than from the graph that
DECLARES it, which is the same class of error as reading a canvas texture where `document.canvas_size`
is the authority (W-A #2).

## Open questions

Each carries a robust default, taken; none blocks implementation.

1. **Should a bare key that some passes declare and others do not say anything about the passes it
   skipped?** Default, taken: **no**, per the design note's rule ("one that some passes declare drives
   those and says nothing about the rest"). The alternative is an informational strip row per skipped
   pass, which would fire constantly on the intended use (a brush uniform that only `paint` declares)
   and train the user to ignore the strip. Revisit only if a user reports a broadcast silently not
   reaching a pass they believed declared it; the fix then is a hover readout on the uniform row, not
   a strip row.

2. **Should `script_ready` be a protocol member, or should the engine catch the compile?** Default,
   taken: **a protocol member**, defined as `self.program is not None or bool(self.compile_unit.error_raw)`
   so that it separates never-attempted (False) from failed and compiled (both True), per § Design
   decisions item 2's truth table, and so the engine never triggers a compile and 066 D1 holds by
   construction rather than by exception handling. Note the expression is the NEGATION of
   `core.py:236`'s would-compile guard, not that guard itself. The alternative (call `get_active_uniforms()` and
   swallow a compile error) hides a real shader failure inside the script path and makes the first
   tick of a six-pass document compile all six. Revisit if a second engine-side consumer needs the
   same readiness notion, in which case it belongs on `Pass` as a public property rather than in the
   protocol alone.

3. **Does `_draw_play_stop`'s imgui id change break a user's muscle memory of the row?** Default,
   taken: **no concern**, the id is invisible. **There is no live collision to fix.** The Document
   panel draws exactly one pass per frame: `App.panel_pass` (`app.py:606-617`) returns a single
   `Pass`, and `tabs/document.py:261-262` loops that one pass's `sorted_hashes` through the only
   `draw_ui_uniform` call site in the codebase, so two passes' rows of the same uniform name are never
   submitted in the same frame and `play_stop_toggle`'s `f"{label}##play_stop_u_{name}"`
   (`ui_primitives.py:208`) never sees a duplicate. What the prefix buys is that a row's imgui state
   (held, hovered) does not carry across a panel-pass switch, which is a correctness nicety rather
   than a bug fix. **The two reviewers split on this** — the design reviewer read it as a live state
   bug ("the two toggles share hover/active state"), the tests reviewer as no collision at all. It was
   resolved by reading `tabs/document.py`'s draw loop: the single call site inside the
   `sorted_hashes` loop at `:261-262`, over `app.panel_pass`'s one pass, is the line, and the tests
   reviewer is right. Named here so a reviewer reads the id change as neither cosmetic nor as closing
   a defect.

4. **Should the `Clear` button be hidden or disabled on a document with no feedback pass?** Default,
   taken: **hidden**, because a permanently-disabled control on the primary viewing surface is visual
   noise on every non-feedback document, which is most of them. The command stays live either way.
   Revisit if a user asks why the button appears on some documents and not others; the answer would be
   a tooltip on a disabled button, which costs a permanent control for a one-time question.

## Review history

### Round 1, pre-implementation (two reviewers, both against the spec at `0ce84f8`)

`reviews/wave_g_pre_design.md` (correctness and design) and `reviews/wave_g_pre_tests.md`
(verification and blast radius). Eighteen findings, **all accepted and folded**. Verdicts as filed:

| Reviewer | Dimension | Verdict |
|---|---|---|
| design | D3 fidelity | PASS |
| design | Key propagation `(pass, name)` | PARTIAL (finding 3) |
| design | Persistence | PARTIAL (finding 5) |
| design | Mouse | **FAIL** (finding 1) |
| design | Command | **FAIL** (finding 2) |
| design | Docs | PASS |
| tests | Test-budget accuracy | PASS |
| tests | Falsifiability | PARTIAL (findings 3, 4) |
| tests | Seam coverage | PASS on the seven, FAIL on the Files-touched row (finding 7) |
| tests | On-disk claims | PASS |
| tests | Blast radius | **FAIL** (findings 1, 2) |

**Design review, eight findings.**

1. *The new mouse gloss breaks the freeze-caveat test the spec claimed survives unedited.* Inserting
   `with down=False` between `at 0.5,0.5` and `on export` splits the literal substring
   `test_the_mouse_gloss_carries_the_frozen_at_center_caveat` asserts. Folded into item 12: the
   caveat stays contiguous and the new-field clauses go after it, stated as a constraint on the
   rewrite. The second half of the same claim (the field-list pin needing no edit) was verified
   correct and stands.
2. *`has_feedback` over `_feedback` is false on exactly the documents the button is for.* `_feedback`
   is an allocation cache filled during `render()` and emptied by `reset_feedback` itself, so the
   button would be absent before the first frame and would hide itself on click. Folded into item 11:
   `bool(plan_passes(self.graph)[0].feedback)`, the graph's own declaration. Also added as a premises
   row and named in the § Verified / corrected premises closing paragraph as the instructive error.
3. *The `set_uniform` gate is specified at a line where the output pass name is not in scope.* Folded
   into item 7: the gate moves below the `target =` bind at `backend.py:887`, keyed by
   `pass_name_of(target.render_pass.source.path)`, which also puts it after the `uniform is None`
   resolution. Consequent on tests finding 4.
4. *A permanently-broken pass is held silently forever by `script_ready`.* `core.py:232` never retries
   a failed compile. Folded into item 2: `script_ready` is False only while NEVER ATTEMPTED
   (`program is None and not compile_unit.error_raw`); a failed pass is ready-but-empty and takes the
   ordinary orphan path. The false `first_render_done` claim is dropped; open question 2 and the
   premises row follow.
5. *The disk enumeration misses two tracked `document.json` files.* Folded into item 13: eleven files
   across three directories, with `projects/documents/1901ab60-...` (5 passes) and `307598da-...`
   added. Re-verified independently for this fold (`git ls-tree -r ccd446b`): both carry `[]`, so the
   zero-byte conclusion and the no-salvage-line conclusion survive.
6. *A click on a script-error row in a shader tab does not open the script.* `_consume_jump` discards
   a request whose path is not the current tab's. Folded into item 8: the click branch calls
   `app.open_script_for(tab.document_id)` before latching the `JumpRequest`, following the two
   existing cross-file jumps. The `_apply_markers` non-leak was verified and is now stated.
7. *The first frames of a multi-pass document drop broadcasts silently.* Folded into item 2 (one
   sentence: dropped, not queued, one frame per pass under the first-render sweep) and into manual
   step 1 (what the maintainer will see).
8. *`_drop_script`'s loop needs an unpack, not only a wider tuple.* Folded into item 5.

**Tests review, ten findings.**

1. *`scripts/smoke.py:317` calls `set_uniform_stopped` with the old arity, and smoke is a gate.* Added
   to § Files touched and to § Tests as the third place the pair re-key must land.
2. *`scripts/dogfood/verify_script_engine.py:97` asserts the driven set by name.* Added to § Files
   touched, with the note that its `:145` sibling survives by accident.
3. *The orphan-console test cites a `caplog` idiom the repo does not have.* Folded: the test installs
   a loguru sink, because loguru does not propagate into the stdlib tree `caplog` reads and the test
   would otherwise pass vacuously under its own falsifier.
4. *The gate change contradicts `test_script_driven_reject.py`'s `object()` stub.* Folded: both stubs
   gain a real `document.graph.output_pass` and `render_pass.source.path`, consistent with design
   finding 3's gate move.
5. *Open question 3's claimed imgui id collision does not exist.* The two reviewers split (design
   finding "agree, and it is under-sold" vs this). **Resolved by reading `tabs/document.py`'s draw
   loop for this fold:** `draw_ui_uniform` has one call site, `:262`, inside a loop over
   `sorted_hashes` built from `app.panel_pass`'s single pass, so the panel draws one pass per frame
   and there is no live collision. The tests reviewer is right; open question 3 now records both
   readings, the line that settles it, and what the prefix actually buys (state not carrying across a
   panel-pass switch).
6. Same defect as design finding 2, independently found. Folded once.
7. *The Files-touched row says four `.render_pass` seams; the premises table says seven.* Folded: the
   row now says seven and names them, and § Tests states the three `reload` seams' falsifier, which is
   pyright at `make check` rather than a test.
8. *`_pass_views`'s driven filter has no existing test to edit.* Folded: the added test is named as
   the first of that code path, `tests/test_working_set.py` is accounted for (asserts the marker
   string over a hand-built list, needs no edit), and the two-pass `Document` fixture is sourced from
   `tests/test_document_graph.py`'s `_document` helper.
9. *`test_a_stale_string_stopped_set_drops_to_empty` must construct its own stale file.* Filed as
   verified, not as a defect; no change.
10. *Manual step 3 has no falsifiable observation for the re-entry rule.* Folded: the step is now an
    off-LEFT / re-enter-RIGHT traverse whose failure mode is a full-width streak, and item 10 states
    that the else branch is new and that `context.py`'s "clamps to the last in-bounds position"
    comment is rewritten in the same edit.

Both reviewers independently re-derived the premises table (24 of 40 rows opened by the tests
reviewer, every row re-checked by the design reviewer) and confirmed all six original refutations,
including the two the spec called work-changing. No finding contradicted a locked decision, and D3
fidelity was traced clause by clause against the six routing cases with no hole found.

### Round 2, closure (both reviewers, against the round-1 fold)

Design: seven of eight closed, **finding 4 NOT CLOSED**. Tests: ten of ten closed, with one follow-on
(**8a**) created by the fold itself. Both reversals are recorded here rather than silently corrected,
because each is a case of the fold introducing an error the round-1 finding did not contain.

**Design finding 4, reopened: the `script_ready` expression was folded in inverted.** Round 1 gave the
prose (never-attempted is held, failed is ready-but-empty) and named `core.py:236`'s pair as the place
the two conditions live. The fold wrote that pair verbatim as the property body,
`program is None and not compile_unit.error_raw` — but `:236` is the guard that *triggers* a compile,
so it is true exactly when a compile is still owed, and as `script_ready` it says the opposite of the
prose above it. Both failure modes are live: a never-attempted pass would read READY, so the engine
would call `get_active_uniforms` on it and compile it from inside the tick, which is precisely the
066 D1 violation the member exists to prevent; and a successfully compiled pass would read NOT ready
forever, so a working document would drive nothing. Item 2 now carries
`self.program is not None or bool(self.compile_unit.error_raw)` plus a three-row truth table
(never attempted → False; failed → True; compiled → True), with the instruction to read the table
rather than the expression if the two ever disagree. Open question 2 and the premises row follow, both
naming the expression as the NEGATION of `core.py:236` so it cannot be re-derived from that line
again.

**Tests finding 8a, new: the sourced fixture would have made the headline test GL-gated.** Round 1's
finding 8 asked where the two-pass fixture comes from; the fold answered "`tests/test_document_graph.py`'s
`_document` helper", which takes a `moderngl.Context`, compiles real passes, and draws its context from
a module fixture that calls `pytest.skip` when no standalone context exists. `test_copilot_script_tools.py`
imports no moderngl and drives the tool layer over fake caps, so importing that helper would move the
wave's headline test behind a GL skip. It is also unnecessary: `_pass_views` reads six attributes, all
satisfiable by a `SimpleNamespace`, which is the idiom the sibling test file already uses on this same
backend method. The `test_copilot_script_tools.py` entry now specifies the stub and says why the
GL-backed helper is refused.

The common shape of the two: each was a detail added while folding, not a claim either review made,
and each was derived from a nearby line (`core.py:236`, the `_document` helper) without re-checking
what that line does in its own context. The premises table's own rule covers both — a citation is
verified by opening it — and it was applied to the round-1 claims but not to the round-1 fixes.

The reviewers also re-verified the on-disk enumeration the fold changed (nine files to eleven,
`git ls-files | grep -c "document.json$"` = 11) and confirmed the zero-byte and no-salvage-line
conclusions survive it.
