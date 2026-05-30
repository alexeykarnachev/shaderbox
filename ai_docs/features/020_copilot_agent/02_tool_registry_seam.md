# 020 Copilot Agent — angle 02: the tool registry + the capability seam

> Research report. Angle: **how copilot tools reach the app without importing imgui and
> without an import cycle** — the encapsulation core. Builds on `00_grounding.md`.
> Citations are re-read from source (2026-05-29), not recalled.

---

## Recommendation (top)

1. **Schema source = pydantic `model_json_schema()`** (ovelia's approach), NOT hand-written JSON
   Schema (cc-server). Pydantic is already a dep, the convention is *full type annotations + no
   hand-rolled dicts*, and one model gives us validation **and** the wire schema from a single
   definition. cc-server hand-writes the schema separately from any validation — that's two
   sources of truth we'd have to keep in sync; reject it.

2. **Capability seam = a `CopilotCapabilities` dataclass of bound callables, built in `App`,
   exactly mirroring `App._build_command_callbacks()` (`app.py:302`).** NOT a `Protocol` (over-spec
   for a solo single-impl app — see adversarial §), NOT a new `node_ops` module yet (that's the
   *real* refactor-prep, but it's narrower than the whole seam — see §"papered-over gaps"). The
   dataclass is the narrow surface the `copilot/` package imports instead of `App`; it is also the
   exact seam that makes handlers unit-testable headlessly (build a `CopilotCapabilities` with
   in-memory lambdas, no glfw/GL).

3. **Handler signature = ovelia's `(ok: bool, message_for_llm: str, payload: dict | None)`
   triple**, NOT cc-server's bare `-> str`. The payload is load-bearing for ShaderBox specifically:
   a `create_node` tool returns `{"node_id": ...}` and the chat-widget renders an "open this node"
   affordance / auto-selects it — the references all note this but ShaderBox is the one where the
   payload drives a *visual* side-effect the user wants.

4. **Stay separate from `commands.py`, but rhyme with it.** The copilot registry is a *new* registry
   with the same shape (a frozen spec table + an id→callable map built on `App`). Two of the v1
   tools (`new_node`, `save`) *could* be thin wrappers over `CommandId` callbacks, but most copilot
   tools need **explicit args** (`create_node(template_id)`, `set_uniform(node_id, name, value)`)
   that the zero-arg `Callable[[], None]` command callbacks structurally cannot carry. Sharing infra
   would force the command callbacks to grow args they don't want. Argue below.

5. **`describe_tools()` renders the live registry** → both the system-prompt "what you can do" block
   and a user-facing "Copilot capabilities" cheatsheet surface. One source of truth, like cc-server.

The seam's real justification is **testability + blast-radius**, NOT "imgui-free" purity (App
already imports imgui — see adversarial §). It is worth it: it's ~40 lines, it's the *only* thing
that lets tool handlers run in `pytest` with no window, and it isolates the tool layer from the
1083-line `App` god-object so a tool author touches one narrow surface.

---

## 1. The registry design

### `ToolDefinition` + `ToolRegistry` (the actual shapes)

```python
# shaderbox/copilot/tools.py
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

# (ok, message_for_llm, payload_for_ui). Mirrors ovelia copilot/tools.py:50.
# Sync, NOT async: ShaderBox tool handlers run on the agent worker thread
# synchronously; there is no asyncio loop. (cc-server is sync too.)
ToolHandler = Callable[[dict[str, Any]], tuple[bool, str, dict[str, Any] | None]]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    args_model: type[BaseModel]   # pydantic, extra="forbid" — schema + validation
    handler: ToolHandler
    mutating: bool                # gates "what I did" note + UI confirm (cc-server _MUTATING_TOOLS)
    needs_gl: bool                # True => handler marshals to main thread (see §2)

    def anthropic_spec(self) -> dict[str, Any]:
        # Anthropic tool-use wire format. (Provider = Anthropic per global CLAUDE.)
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.args_model.model_json_schema(),
        }


class ToolRegistry:
    def __init__(self, definitions: list[ToolDefinition]) -> None:
        self._by_name: dict[str, ToolDefinition] = {d.name: d for d in definitions}

    def specs(self) -> list[dict[str, Any]]:
        return [d.anthropic_spec() for d in self._by_name.values()]

    def describe(self) -> str:
        # Single source of truth for the system-prompt capability block AND the
        # user-facing cheatsheet (cc-server describe_tools). One registry, one list.
        return "\n".join(f"- {d.name}: {d.description}" for d in self._by_name.values())

    def execute(self, name: str, raw_args: dict[str, Any]) -> tuple[bool, str, dict[str, Any] | None]:
        tool = self._by_name.get(name)
        if tool is None:
            return False, f"error: unknown tool '{name}'", None
        try:
            args = tool.args_model.model_validate(raw_args)
        except ValidationError as exc:
            return False, _validation_error_message(exc), None       # ovelia pattern
        try:
            return tool.handler(args.model_dump())
        except Exception:
            logger.exception(f"copilot_tool_failed name={name}")
            return False, f"error: tool '{name}' failed", None        # generic — never leak (cc-server)
```

Key decisions and why:

- **`args_model: type[BaseModel]` over `parameters: dict`.** ovelia's `_spec()` (tools.py:763) does
  exactly `args_model.model_json_schema()`. We adopt it wholesale. The model carries
  `Field(description=...)` + `ge/le/max_length` validators, all of which land in the schema *and*
  enforce at `model_validate`. cc-server's `parameters: dict[str, Any]` (tools.py:93) is the schema
  but does *zero* validation — every handler re-reads `args.get("...")` untyped (tools.py:317,499).
  In a fully-typed repo with pydantic already in, the dict approach is strictly worse.

- **`execute()` does validate→dispatch→generic-catch in one place** — combining cc-server's
  defense-in-depth `try/except` returning `"Error: ..."` (tools.py:135-143) with ovelia's
  `ValidationError → friendly message` (tools.py:284,312). The exception text never reaches the LLM
  (could carry an absolute filesystem path / API-key fragment); detail goes to `logger`.

- **`mutating` + `needs_gl` flags** are ShaderBox-specific metadata cc-server only half-has
  (`_MUTATING_TOOLS` frozenset). `mutating` drives the "actions already committed" note if a turn is
  cut off and the UI confirm-gate; `needs_gl` tells the dispatch layer whether the handler must
  marshal to the main thread (§2). Storing it *on the definition* keeps the partition declarative,
  not scattered through handler bodies.

- **No `required_role`.** Single local user — drop cc-server's entire role machinery (grounding §4
  "What ShaderBox does DIFFERENTLY"). A `mutating`/destructive flag is the only gate we keep.

- **Frozen dataclass, not ovelia's `__slots__` class.** Frozen dataclass matches the repo's
  `CommandSpec` (`commands.py:60`), `ShaderError`, `JumpRequest` — house style. ovelia's `__slots__`
  micro-opt is irrelevant at ~15 tools.

### Handlers are closures over the capability seam (consensus pattern)

All three references build handlers as closures capturing their dependencies
(`_make_<tool>_handler(service) -> ToolHandler`, cc-server:316 / ovelia:279 / marginalia:341). We do
the same, but the captured dependency is the **`CopilotCapabilities` bundle** (§2), not raw services:

```python
def _make_set_uniform_handler(caps: CopilotCapabilities) -> ToolHandler:
    def handle(args: dict[str, Any]) -> tuple[bool, str, dict[str, Any] | None]:
        node_id, err = caps.resolve_node_id(args["node_id"])      # ovelia _resolve_id (short ids)
        if node_id is None:
            return False, f"error: {err}", None
        ok, msg = caps.set_uniform_value(node_id, args["name"], args["value"])
        if not ok:
            return False, f"error: {msg}", None
        return True, f"set {args['name']} on node [{node_id[:8]}]", {"node_id": node_id}
    return handle
```

---

## 2. The capability seam (the load-bearing piece)

### Why a seam at all (precise version)

The grounding §3(c) frames it as "reach the verbs without importing imgui." But re-reading
`app.py:1-50`: **`App` itself imports `imgui`, `text_edit`, `imgui_command_palette`,
`portable_file_dialogs`, `GlfwRenderer`.** So `from shaderbox.app import App` inside
`shaderbox/copilot/` transitively pulls in the entire imgui stack and — worse — risks a cycle the
moment anything in the imgui import chain wants to reference copilot state. The repo's hard rule (no
`if TYPE_CHECKING`, "a circular import is a design bug", CLAUDE.md / grounding §6) means we *cannot*
paper a cycle over; we must not import `App` into the tool layer.

`commands.py` already solved this exact problem for feature 018. Re-read its docstring
(`commands.py:1-9`): **"Leaf module: imports `imgui` only, never `App`. The id->callback wiring lives
on `App` (built at init, closing over self) so this stays cycle-free."** The copilot seam is the
same trick, one level narrower: a **dataclass of bound callables built in `App.__init__`**, passed
*into* the copilot package. `shaderbox/copilot/` imports only `CopilotCapabilities` (a leaf module of
plain types), never `App`. `App` imports `copilot` to build the registry. One-directional. No cycle.

### The seam: `CopilotCapabilities`

```python
# shaderbox/copilot/capabilities.py — LEAF MODULE. No imgui, no App, no GL imports.
# Plain types + Callable fields. App builds an instance in __init__; the copilot
# package imports ONLY this. Directly mirrors commands.py's leaf-module discipline
# and ShaderLibFileManager's injected-callbacks pattern (file_ops.py:26-39).
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NodeSummary:        # GL-free value object crossing the seam (like RenderedArtifact)
    node_id: str
    name: str
    uniform_names: list[str]
    has_errors: bool


@dataclass(frozen=True)
class CompileErrorInfo:
    path: str
    line: int
    message: str


@dataclass(frozen=True)
class CopilotCapabilities:
    # ---- read-only / GL-free (safe to call on the worker thread) ----
    list_nodes: Callable[[], list[NodeSummary]]
    get_node_summary: Callable[[str], NodeSummary | None]
    get_shader_source: Callable[[str], str | None]
    get_compile_errors: Callable[[str], list[CompileErrorInfo]]
    list_templates: Callable[[], list[tuple[str, str]]]      # (template_id, name)
    list_lib_functions: Callable[[], list[tuple[str, str]]]  # (name, signature)
    current_node_id: Callable[[], str]
    resolve_node_id: Callable[[str], tuple[str | None, str | None]]  # short-id (ovelia _resolve_id)

    # ---- shader-lib CRUD: GL-free, already explicit-arg (file_ops.py) ----
    create_lib_file: Callable[[Path, str], Path | None]
    rename_lib_file: Callable[[Path, str], Path | None]
    delete_lib_file: Callable[[Path], None]
    read_lib_file: Callable[[Path], str | None]
    write_lib_file: Callable[[Path, str], bool]      # disk write -> mtime watcher reloads

    # ---- mutations the worker REQUESTS but the main thread APPLIES (needs_gl) ----
    #   These return immediately with a queued/applied result; the threading agent
    #   designs the queue. Listed here as the seam's GL-touching half.
    create_node: Callable[[str], tuple[str | None, str]]               # (node_id, msg)
    delete_node: Callable[[str], tuple[bool, str]]
    set_uniform_value: Callable[[str, str, Any], tuple[bool, str]]     # GAP (a) — see §gaps
    set_uniform_input_type: Callable[[str, str, str], tuple[bool, str]]
    edit_shader_source: Callable[[str, str], tuple[bool, str]]         # write node .glsl -> reload
    select_node: Callable[[str], None]
    render_export: Callable[[str, dict[str, Any]], tuple[bool, str, dict[str, Any] | None]]
```

`App` builds it the same way it builds `command_callbacks` (`app.py:302-318`):

```python
def _build_copilot_capabilities(self) -> CopilotCapabilities:
    return CopilotCapabilities(
        list_nodes=self._copilot_list_nodes,            # small adapter methods on App
        get_shader_source=self._copilot_get_shader_source,
        create_node=self._copilot_create_node,          # wraps the refactored create_node(template_id)
        set_uniform_value=self._copilot_set_uniform,     # the NEW headless setter (gap a)
        edit_shader_source=self._copilot_edit_shader,    # writes node .glsl, lets the watcher reload
        create_lib_file=lambda d, n: self.shader_lib_files.create_file_in(d, n),  # already explicit-arg
        ...
    )
```

### The GL-free / GL-touching partition (verified against source)

This is the crux from grounding §3. Re-reading the source confirms the split exactly:

| Capability | GL? | Evidence |
|---|---|---|
| `list_nodes`, `get_node_summary` | GL-free | reads `app.ui_nodes` dict + `UINodeState` (ui_models, pydantic) |
| `get_shader_source` | GL-free | `node.source.text` (a str on a frozen dataclass, `shader_source.py:8`) |
| `get_compile_errors` | GL-free | reads `node.compile_unit.errors: list[ShaderError]` (`core.py:52`) — **but** the list is only fresh after a `compile()` ran on the main thread; the read itself is free |
| `list_lib_functions` | GL-free | `app.shader_lib_index.functions` (`index.py`, "Pure functions, no GL") |
| `create_lib_file` / `rename` / `delete` / `read` | GL-free | `ShaderLibFileManager` — file_ops.py docstring: "does NOT import App ... explicit args so a non-UI caller (e.g. a future agent tool) can drive them" (file_ops.py:1-9) |
| `edit_shader_source` (node) | **GL-free write, GL apply deferred** | write the node's `shader.frag.glsl` to disk; `_reload_if_changed` (ui.py:72) recompiles on the main thread next frame. **The free lunch** (grounding §3 hot-reload). No marshalling. |
| `set_uniform_value` | **GL-touching** | writing `node.uniform_values[name]` is a plain dict write (free), but the value must be *valid for the next render's bind* (`core.py:338-348`), and a texture/buffer value constructs GL objects → main thread |
| `create_node` | **GL-touching** | `load_node_from_dir` → `Node.load_from_dir` → `node.render()` warm-up (`core.py:186`) |
| `delete_node` | **GL-touching** | `node.release()` releases GL program/canvas (`app.py:1042`, `core.py:211`) |
| `render_export` | **GL-touching + worker** | `node.render_media` (`core.py:471`) is the exact thing exporters already marshal — `base.py:97-110` thread-affinity contract |

**Critical insight, re-confirmed:** `edit_shader_source` is the *cleanest* tool seam in the whole
app because the disk-write is GL-free and the existing mtime watcher does the GL recompile on the
main thread with zero new marshalling. A copilot that "edits the shader" should *write the file*,
full stop. The threading agent owns only the `needs_gl=True` minority (create/delete/set_uniform/
render_export). This report just labels the partition via `needs_gl` on each `ToolDefinition`; the
queue mechanism is the threading agent's problem.

### Why a dataclass-of-callables and not a `Protocol`

A `Protocol` would declare ~22 methods and say "App implements this." But: App has 1083 lines and ~80
public methods; a Protocol listing only the 22 copilot ones is a *documentation* win but App still
*is* the implementation, so we still can't import App into the tool layer — the Protocol type itself
would live in a leaf module and `App` would be annotated `App(... ) -> None` implementing it
structurally (no explicit inheritance needed for a Protocol). That's *more* indirection than the
dataclass for the same cycle-avoidance. The dataclass-of-bound-callables:
- is the **exact** pattern `ShaderLibFileManager` already uses for its reach-back into App's editor
  domain (`on_paths_removed`, `on_path_renamed` callbacks, file_ops.py:26-39),
- is the **exact** pattern `command_callbacks` uses (app.py:184,302),
- is trivially constructible with in-memory lambdas in a test (§6),
- doesn't require `App` to nominally satisfy anything.

A Protocol is the better choice *only* if there are genuinely multiple implementations. There aren't.
(Adversarial section revisits.)

---

## 3. Handler signature: the triple, justified

`(ok: bool, message_for_llm: str, payload: dict[str, Any] | None)` — ovelia's shape (tools.py:50).

- **`ok`** drives the agent loop (a `False` lets the loop decide whether to retry / surface) and the
  "what I committed" note for `mutating` tools.
- **`message_for_llm`** is the tool-result string the model reads back. Mutating tools NAME the
  affected entity (cc-server F3: `"set u_color on node [a1b2c3d4] 'Sunset'"`) so a later turn can
  answer "did you change the color?".
- **`payload`** is where ShaderBox diverges hardest and where the triple earns its keep over
  cc-server's bare `-> str`: `create_node` returns `{"node_id": "..."}`; the chat widget reads it and
  **auto-selects + opens that node in the editor** — a visual affordance the user actively wants ("I
  made the node, here it is"). cc-server's payload-less string can't drive a UI side-effect; ovelia's
  payload renders an event card; ShaderBox's payload focuses a node / jumps to a compile error line
  (it can carry `{"jump": {"path": ..., "line": ...}}` reusing the existing `JumpRequest`,
  `editor_types.py:9`). **The payload is the bridge from "the agent did X" to "the app now shows X."**

Bare `-> str` (cc-server) is simpler but throws away the structured side-channel exactly where
ShaderBox needs it most. Take the triple.

---

## 4. Relationship to the existing `commands.py` registry

**Verdict: separate registry, same shape; do NOT merge.** The honest argument both ways:

**For sharing infra:** `commands.py` already is "a registry of callables built on App, leaf module,
cycle-free" (commands.py:1-9). The copilot registry wants the identical property. `new_node`,
`save`, `delete_node`, `open lib picker` are all *already* `CommandId` callbacks
(app.py:303-318). A copilot `save` tool wrapping `command_callbacks[CommandId.SAVE]` is genuinely
3 lines and avoids duplicate wiring.

**Against (decisive):** The command callback type is `Callable[[], None]` (app.py:184) — **zero args,
no return.** It's built for *keyboard chords and the palette* (a chord can't carry arguments). But
the defining feature of a copilot tool is **explicit args**: `create_node(template_id)`,
`set_uniform(node_id, name, value)`, `edit_shader_source(node_id, text)`. You cannot express those
through a zero-arg callback. To share, you'd have to either (a) widen every command callback to take
args it doesn't want, polluting the keyboard-command type for the agent's benefit, or (b) bolt an
arg-carrying variant onto commands.py, at which point it's two registries living in one file. Also:
commands need a `default_chord`, `scope`, `in_palette`, `rebindable` (commands.py:60-71) — all
meaningless to a tool — while tools need `args_model`, `mutating`, `needs_gl` — all meaningless to a
command. The specs barely overlap.

**Resolution:** Two registries, deliberately *rhyming* (frozen spec table + id→callable map built on
App). Where a tool's effect is identical to a command's, the tool handler may *call the same
underlying App method the command callback calls* — but through the `CopilotCapabilities` seam, not
by reaching into `command_callbacks`. E.g. the `save` tool's capability is `caps.save` →
`App.save` (the same method `CommandId.SAVE` dispatches), so the *behavior* is shared at the App
method level (the real single source of truth) without coupling the two registries' *infra*. This is
the right altitude: share the verb, not the registry.

---

## 5. `describe_tools()` — single source of truth

`ToolRegistry.describe()` (shown in §1) renders the live registry to a bullet list. Two consumers,
zero hand-maintained capability lists (cc-server's exact win, tools.py describe_tools):

1. **System prompt**: a "## What you can do" block, assembled least-volatile-first for prompt-cache
   friendliness (ovelia tools.py prompt ordering, grounding §4): static tool list → available lib
   functions (`list_lib_functions`) → current node + its uniforms + compile errors (the volatile
   tail). Only the tail invalidates the prefix cache per turn.
2. **User-facing surface**: a "Copilot capabilities" panel in the chat widget (or a `/?` in chat)
   rendered from the same `describe()`. When a tool is added, both update for free.

If we later want marginalia's per-tool enable/disable (tools.ts:77-89, persisted), it slots in as a
`disabled: set[str]` filter inside `specs()` / `describe()` — additive, not now.

---

## 6. Testability (the seam's primary payoff)

ShaderBox has `tests/` (pytest, GL-free: `test_lib_index.py`, `test_util.py`, etc.) + a
GL-but-headless `scripts/smoke.py` (200 frames in an invisible glfw window). Copilot tools fit **both
tiers**:

**Tier 1 — pure handler unit tests, no glfw/GL (the seam's reason to exist).** Build a
`CopilotCapabilities` from in-memory lambdas and assert handler behavior — exactly marginalia's
`setBookPageProvider(inMemoryImpl)` swap (tools.ts:144). This is *only possible because the seam is a
plain dataclass of callables*; if tools took `app: App`, every test would need a real glfw window:

```python
def test_set_uniform_rejects_unknown_node() -> None:
    caps = CopilotCapabilities(
        resolve_node_id=lambda raw: (None, f"no node {raw!r}"),
        set_uniform_value=lambda *a: (True, "ok"),
        # ... only the fields this tool touches; others = a raising stub
    )
    registry = build_copilot_registry(caps)
    ok, msg, payload = registry.execute("set_uniform", {"node_id": "zzz", "name": "u_x", "value": 1.0})
    assert not ok and "no node" in msg

def test_create_node_returns_node_id_payload() -> None:
    caps = _fake_caps(create_node=lambda tid: ("new-id-123", "created"))
    ok, msg, payload = build_copilot_registry(caps).execute("create_node", {"template_id": "t1"})
    assert ok and payload == {"node_id": "new-id-123"}
```

Also covered free by Tier 1: pydantic `extra="forbid"` rejecting a hallucinated arg, the
`ValidationError → friendly message` path, the generic-error masking (`execute` swallows a raising
handler into `"error: tool failed"`), and `describe()` output stability.

**Tier 2 — smoke integration (the GL half).** Extend `scripts/smoke.py` (smoke.py:73-81 already pokes
`app.cycle_region()` / `focus_node_tab` mid-loop) to build the *real* `CopilotCapabilities` from the
live `App` and drive a `create_node` then `set_uniform` then `edit_shader_source` across a few frames,
asserting no exception + the node appears + its compile errors update. This exercises the
GL-marshalling path that Tier 1 fakes. The threading agent's queue gets exercised here.

The two tiers map cleanly onto the partition: **GL-free capabilities → Tier 1 (most tools, most
coverage); GL-touching capabilities → Tier 2 smoke.** A new `tests/test_copilot_tools.py` holds Tier
1; smoke.py grows a `--copilot` poke block for Tier 2.

---

## 7. The v1 tool catalog (seeds the spec)

Grouped read-only / node-ops / uniform-ops / shader-editing / lib-ops / render-export. `M`=mutating,
`G`=needs_gl (main-thread marshalling). All node addressing accepts a short-id prefix
(ovelia `_resolve_id`, tools.py:56) since node ids are UUIDs (`ui_models.py:282`).

| Tool | M | G | Args (rough) | One-line description |
|---|---|---|---|---|
| `list_nodes` | | | — | List all nodes: id, name, uniform names, has-errors flag. |
| `get_node` | | | `node_id` | One node's name, shader source length, uniforms, compile errors. |
| `get_shader_source` | | | `node_id` | Return the node's full `shader.frag.glsl` text. |
| `get_compile_errors` | | | `node_id` | The node's `compile_unit.errors` (path/line/message) — the "did it work?" signal. |
| `list_templates` | | | — | Available node templates (id, name) for `create_node`. |
| `list_lib_functions` | | | `[filter]` | SB_* lib functions the shader can call (name + signature + doc), from the live index. |
| `read_lib_file` | | | `rel_path` | Read a shader-lib `.glsl` file's contents. |
| `create_node` | M | G | `template_id` | Create a node from a template; returns `node_id` payload → UI auto-selects it. |
| `delete_node` | M | G | `node_id` | Trash a node (recoverable). Names the node in the result. |
| `select_node` | M | | `node_id` | Make a node current / open it in the editor (drives the UI focus). |
| `edit_shader_source` | M | (defer) | `node_id, text` | Replace the node's shader source (writes the file; watcher recompiles main-thread). |
| `set_uniform` | M | G | `node_id, name, value` | Set a scalar/vector/color uniform value (the NEW headless setter — gap a). |
| `set_uniform_input_type` | M | G | `node_id, name, input_type` | Set a uniform's UI input shape, snapped to `valid_input_types()`. |
| `create_lib_file` | M | | `dir_rel, name` | Create a new shader-lib `.glsl` (delegates to `ShaderLibFileManager.create_file_in`). |
| `write_lib_file` | M | (defer) | `rel_path, text` | Overwrite a lib file (watcher rebuilds the index + invalidates dependents). |
| `rename_lib_file` | M | | `old_rel, new_rel` | Rename/move a lib file (path-traversal-guarded by `_validate_target`). |
| `delete_lib_file` | M | | `rel_path` | Trash a lib file (recoverable in `.trash/`). |
| `save_node` | M | | `[node_id]` | Persist the current/given node to disk (wraps `App.save` / `save_ui_node`). |
| `render_export` | M | G | `node_id, {duration, fps, size?, format}` | Render the node to image/video via the export path (worker + GL marshalling). |

Notes:
- `edit_shader_source` / `write_lib_file` marked `(defer)` for GL: the *write* is GL-free; the
  recompile happens on the main thread via the watcher — so they need no explicit marshalling, just
  awareness of the unsaved-editor clobber edge case (grounding §3 hot-reload caveat).
- `render_export` is the one tool that's both worker-bound *and* GL-marshalled — it should reuse the
  exporter thread-affinity machinery (`base.py:97`), not invent a new path.
- v1 deliberately omits: media/texture uniform loads (complex GL value construction), node-rename,
  project open/save-as (destructive, low agent value). Park for v2.

---

## 8. Adversarial section (attack the design)

### "The capability seam is over-engineering — tools should just take `app: App`."

**Strongest honest case:** `App` *already* imports imgui (app.py:9-13), so the "tools must stay
imgui-free" framing is *already violated by App itself*. If a copilot tool calls `app.create_node()`
directly, the only new import is `from shaderbox.app import App` — and App is the orchestrator
anyway. cc-server doesn't have a capability dataclass; its handlers capture *services*
(`IDocumentService`), but those are real separate objects, whereas ShaderBox's "service" is just App.
Building 22 thin adapter methods (`_copilot_list_nodes`, etc.) plus a dataclass to wrap calls that
already exist on App looks like ceremony. The most boring thing that works: `tools.py` imports `App`
for typing only and handlers do `app.create_node(tid)`.

**Why it's still wrong, resolved:**
1. **Cycle risk is real, not theoretical.** `ui.py` imports `App` (ui.py:12); if `copilot/` imports
   `App` and `App.__init__` imports `copilot` to build the registry (it must, to own the registry),
   you have `app → copilot → app`. The repo *forbids* `if TYPE_CHECKING` to break it (CLAUDE.md). The
   seam makes the dependency one-directional by construction — the proven `commands.py` fix.
2. **Testability is the killer.** With `app: App`, a handler unit test needs a live glfw window +
   GL context (App.__init__ calls `glfw.init()`, `glfw.create_window`, `imgui.create_context` —
   app.py:94-141). Tier-1 pure tests become impossible; *every* tool test drops to smoke-only. The
   seam is the single thing enabling `pytest tests/test_copilot_tools.py` with no window. marginalia
   built the whole `ToolRegistrationHelpers` interface for exactly this swap (tools.ts:144).
3. **Blast radius.** A tool author touching a 22-field dataclass can't accidentally reach the other
   ~60 App methods or App's transient imgui focus state (`region_focus_pending`,
   `editor_jump_request`, ...). The seam is a contract: "these are the verbs the agent may use."

The seam costs ~40 lines (one dataclass + the `_build_copilot_capabilities` method, parallel to the
existing `_build_command_callbacks`). For cycle-safety-by-construction + headless testability, it
pays for itself immediately. **Keep the seam; reject `app: App`.**

### "What's the most boring registry that works?"

A module-level `list[ToolDefinition]` and a `dict[str, handler]`, no class (marginalia's
`toolRegistry: ToolDefinition[]` + `executeTool`, tools.ts:59,111). We reject *only* the
module-level-mutable-global part (it fights the seam's per-instance construction + per-test isolation)
but otherwise the boring version *is* what §1 proposes: a small `ToolRegistry` wrapping a dict, a
`build_copilot_registry(caps)` factory (ovelia/cc-server `build_tool_registry()`), and closure
handlers. No agent-loop, no compression, no role logic in *this* layer. The registry is boring on
purpose; the interesting parts (threading, agent loop, prompt) are other agents' angles.

### Which "gaps" are real refactor-prep vs paper-over-in-handler?

- **Gap (a) `set_uniform_value`: REAL refactor-prep.** The only uniform write today is inline in the
  imgui draw loop (`widgets/uniform.py:228-230`: `try_to_release(current_value)` then
  `ui_node.node.uniform_values[name] = new_value`). There is *no* headless setter. A handler cannot
  paper this over cleanly — it would have to reach into `node.uniform_values` itself and re-implement
  the `try_to_release` + validity logic, duplicating it. The right move: extract a real
  `App.set_uniform_value(node_id, name, value)` (or a `node_ops` free function) that the *uniform
  widget also calls*, so there's one writer. This is grounding §6's "missing-half" signal — the read
  path is clean, the write path is the gap. Small, well-scoped extraction; do it as prep.
- **Gap (b) `create_node` reads grid selection: REAL but tiny.** `create_node_from_selected_template`
  reads `app_state.selected_node_template_id` (app.py:1070-1071). The fix is a 2-line refactor:
  `create_node(self, template_id: str)` taking the arg; the existing zero-arg method becomes
  `create_node(self.app_state.selected_node_template_id)`. The palette already does the
  set-then-call dance (app.py:358-363) — so the explicit-arg form is the natural shape and the grid
  selection becomes one caller of it. Do the refactor; it's strictly cleaner even ignoring copilot.
- **input-shape mutation (also inline, `draw_input_type_selector`): REAL, same class as (a).** Same
  fix shape — extract a headless `set_uniform_input_type` honoring `snap_input_type()`
  (`ui_models.py:118`). Bundle with (a).
- **short-id resolution: PAPER-OVER (in a helper, not a refactor).** `resolve_node_id` is a pure
  function over `ui_nodes.keys()` — port ovelia's `_resolve_id` (tools.py:56) into the copilot
  package as a free function. No App change needed.
- **compile-errors freshness: PAPER-OVER.** `get_compile_errors` reads a list that's only fresh after
  a main-thread `compile()`. The handler doesn't fix this; the *agent loop / threading* ensures a
  render frame elapsed before reading (another agent's concern). The capability just exposes the
  read.

---

## Open questions

1. **Does `set_uniform_value` extraction land in `App` or a new headless `node_ops.py`?** The
   grounding floats a future `node_ops` module (§3c). `App` is the boring choice (matches
   `delete_node`); `node_ops` is the GL-free-purity choice but `set_uniform_value` *is* GL-adjacent
   (value validity for bind), so it's not obviously a clean GL-free leaf. Recommend `App` method now,
   extract to `node_ops` only if a second non-UI caller appears. Flag for the synthesis.

2. **Sync vs async handler signature.** I propose **sync** (handlers run on the agent worker thread;
   no asyncio in ShaderBox; cc-server is sync). ovelia is async only because its whole server is.
   Confirm the threading-agent's worker model is a plain `threading.Thread` (not asyncio) so sync
   handlers are correct — this is the one cross-angle dependency.

3. **Where exactly does `needs_gl=True` marshalling happen — in `execute()` or above it?** I kept
   `execute()` pure (validate→dispatch→catch) and left marshalling to the dispatch layer that the
   threading agent designs (the layer reads `tool.needs_gl` and routes accordingly). Need to confirm
   the threading angle agrees the flag lives on `ToolDefinition` (declarative) and the queue routing
   reads it, rather than each handler self-marshalling. This determines whether `render_export`'s
   handler returns a "queued" result or blocks on a main-thread round-trip.
