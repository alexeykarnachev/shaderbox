# 020 Copilot Agent — refactor-prep audit (the leaking-seams pass)

> **Angle:** what gaps / leaking seams must close (or are worth closing) BEFORE the copilot lands,
> prioritized, so the agent's tool layer wires in smoothly. NOT a spec; NOT an implementation.
> Every claim is grounded in source re-read 2026-05-29 (file:line cited). Sibling of `00_grounding.md`.

The headline verdict up front: **almost all of the "prep" is the feature's own tool layer in
disguise.** There is exactly **one** true MUST-do-before, and it is small. `app.py` does **not**
need splitting now. The grounding's 3 named gaps are real; two of them are best absorbed by the
feature, one is a 1-line setter that's a coin-flip (do it or absorb it).

---

## 1. Executive priority list

| # | Finding | Classification | Blast radius |
|---|---------|----------------|--------------|
| F1 | No `set_uniform_value(node_id, name, value)` verb — mutation inline in `widgets/uniform.py:228-230` | **SHOULD-before** (else ugly tool-layer reach-around) | 1 file (+ 1 caller) |
| F2 | `create_node_from_selected_template()` reads grid selection, not a `template_id` arg | CAN-defer-to-feature (trivial param add) | 1 file |
| F3 | GL-free vs GL-touching partition is *implicit* — no single documented contract for the worker | **MUST-before** (this doc IS the deliverable; §2 table) | doc-only |
| F4 | The "reach App without imgui" seam (`CopilotCapabilities` interface) | CAN-defer-to-feature (it's the feature's core design) | new file (feature) |
| F5 | Secret storage for the Anthropic key — extend `IntegrationsStore` | CAN-defer-to-feature (1 model + 2 methods, clean seam exists) | 1 file |
| F6 | `app.py` split (1083 L) before the copilot adds chat-state + worker lifecycle | **PREMATURE** (re-confirmed; copilot lands in its own package) | n/a |
| F7 | `current_node_ui_state_or_default` returns a *throwaway* `UINodeState()` — silent-no-op seam | SHOULD-before (cheap; copilot will read node state by id) | 1 file |
| F8 | `UINodeState` drops a node on a bad known-key value (todo deferral) | CAN-defer (orthogonal; copilot doesn't write `input_type` via load) | 1 file |
| F9 | Editor-session clobber: a disk write while the user has unsaved edits | CAN-defer-to-feature (edge case the tool's write-path owns) | 1 file |
| F10 | No-`TYPE_CHECKING` cycle risk: copilot ↔ App | PREMATURE-to-prep (the interface in F4 *is* the cycle break) | n/a |
| F11 | `ui_primitives` has no chat-widget primitive (multiline log, streaming text) | CAN-defer-to-feature (build the primitive WITH the widget) | feature |
| F12 | Resolution-combo round-trip parse (`tabs/node.py:105`) | CAN-defer (copilot sets resolution via a verb, not the combo) | 1 file |

**The single MUST:** F3 — produce the authoritative GL-free/GL-touching verb table (this document).
Without it the threading agent and the spec re-derive the partition from scratch and will get it
wrong (e.g. assume `set_uniform_value` is GL-free — it is not for samplers/buffers). **F1 and F7 are
the two cheap SHOULDs** worth landing as actual prep commits; everything else is the feature's own
scope by the dev_flow "deferrals get absorbed by the triggering work" rule.

---

## 2. The GL-free / GL-touching verb partition (the critical input)

The hard contract (grounding §2, `exporters/base.py:94-110`): the agent's tool handlers run on a
**worker thread that MUST NOT touch moderngl**. A live GL context exists only on the main (render)
thread. The partition below classifies every mutation/read verb the copilot will want.

**Key insight (verified):** the cleanest GL-touching marshalling primitive *already exists for free* —
the **mtime watcher**. `ui.py::_reload_if_changed` (72-119) + `_maybe_rebuild_lib_index` (122-148)
run every frame on the main thread. A worker that **writes a `.glsl` file to disk** (node shader or
lib file) triggers recompile + editor re-sync on the main thread with **zero extra plumbing**. So
"edit the shader" — the agent's single most important verb — is GL-free at the call site.

### GL-FREE (safe to call directly on the worker thread)

| Verb / op | Where it lives | Notes |
|-----------|----------------|-------|
| Write node shader `.frag.glsl` to disk | filesystem; picked up by `ui.py:72-119` | THE shader-edit seam. Bypasses `flush_current_editor`'s GL path entirely. Caveat F9. |
| Lib file CRUD: `create_file_in` / `create_dir_in` / `rename_file` / `delete_file` / `delete_dir` | `shader_lib/file_ops.py:142-312` | Already explicit-args, App-free, imgui-free. The cleanest agent surface in the repo. Trash + index-rebuild are filesystem ops; the rebuild callback runs `ShaderLibIndex.build` (pure regex/glob, GL-free). |
| Read `node.compile_unit.errors` | `core.py:52` | The agent's "did it compile?" feedback signal. Pure value read (list of `ShaderError`). Stale until the main thread recompiles. |
| Read `node.source.text` / `.path` | `core.py:117` / `shader_source.py` | Pure value. |
| Read `ui_state` (sort keys, ui_name, smoothing) | `ui_models.py:149-159` | Pydantic, GL-free. |
| Read `app_state.*` | `ui_models.py:171-192` | Pure pydantic. |
| Read `UIUniform` metadata + `valid_input_types()` / `snap_input_type()` | `ui_models.py:62-122` | Pure. The agent must respect `valid_input_types()` when proposing an `input_type`. |
| Build context-snapshot text (current node, uniform list, lib functions) | n/a (new, reads the above) | All pure reads. |

### GL-TOUCHING (MUST be marshalled to the main thread — a command queue the frame loop drains)

| Verb / op | Where it lives | Why GL |
|-----------|----------------|--------|
| `set_uniform_value` for **sampler** uniforms | would-be new verb / `widgets/uniform.py:170-205` | Loads `Image`/`Video` → `moderngl.get_context()` lazily (`conventions.md ## Known quirks`). `try_to_release` (`util.py:102`) frees a GL texture. |
| `set_uniform_value` for **buffer/UBO** uniforms | `core.py:294-299` | Value is a `moderngl.Buffer`; `value.write(...)` / `bind_to_uniform_block`. |
| `set_uniform_value` for scalar/vec/array | `widgets/uniform.py:228-230` | The *write* `node.uniform_values[name] = v` is a plain dict write (GL-free), but it's only meaningful once `node.render()` binds it — and `render()` is GL. Treat the whole verb as main-thread. |
| `create_node_from_selected_template` | `app.py:1070` | `load_node_from_dir` → `Node.load_from_dir` → warm-up `node.render()` (`core.py:186`) + texture loads. |
| `delete_node` | `app.py:1028` | `node.release()` frees GL program/vbo/vao/canvas. |
| `node.canvas.set_size((w,h))` (resolution change) | `tabs/node.py:106` / `core.py:84` | Allocates a GL texture+fbo. |
| `node.render_media(...)` (export to image/video) | `core.py:471` / `tabs/render.py:50` | Full render loop. Already has the off-thread precedent via `share_state.render_for` — but the render itself runs render-thread. |
| `flush_current_editor` / `save` | `app.py:815`, `975` | `release_program` + `render` (`app.py:825-826`); `save_ui_node` reads `get_active_uniforms()` (needs live program). |
| `get_active_uniforms()` / `program[...]` | `core.py:215-223` | Needs a compiled program. The agent's introspection of "what uniforms does this node have" depends on a live program → main-thread. |
| `open_shader_lib_file` / `get_session` | `app.py:733`, `756` | Constructs a `TextEditor` (imgui object) — not GL but **imgui-thread-affine**; same restriction. |

**The trap to flag for the threading agent:** "set a uniform" reads as trivially GL-free (a dict
write) but is GL-touching for two of the seven `UniformValue` arms (sampler, buffer) and pointless
without a follow-up render. The verb must be marshalled wholesale. Do not let the spec assume
otherwise.

---

## 3. Per-finding detail

### F1 — No `set_uniform_value` verb (the flagship). SHOULD-before. 1 file.

**Evidence — fully characterized.** `widgets/uniform.py:228-230`:
```python
if new_value is not None:
    try_to_release(current_value)
    ui_node.node.uniform_values[ui_uniform.name] = new_value
```
This is the *only* uniform write in the draw layer. What "set a uniform" actually entails:

1. **The type union is wide.** `UniformValue` (`core.py:93-101`) =
   `int | float | Sequence[int] | Sequence[float] | MediaWithTexture | moderngl.Texture |
   moderngl.Buffer`. A headless setter must accept/validate all seven arms. For an agent, the
   tractable arms are scalar/vec/array/color (plain numbers); sampler (`MediaWithTexture` from a
   file path) and buffer (raw bytes) are GL-constructed (see §2).
2. **The `try_to_release` dance** (`util.py:102-106`): the *old* value, if it's a GL object
   (texture/video/buffer), must be `.release()`d before being overwritten or it leaks GL memory.
   A naive `dict[name] = v` from a tool would leak. The verb MUST replicate this.
3. **The input-shape (`input_type`) is mutated the *same* inline way** —
   `draw_input_type_selector` (`widgets/uniform.py:84-92`) does
   `ui_uniform.input_type = valid[(idx+1) % len(valid)]` with no headless verb. A tool that wants to
   say "show this vec3 as a color picker" has no setter. But note: `input_type` lives on `UIUniform`
   keyed by **hash** in `ui_state.ui_uniforms` (`ui_models.py:153`), while the *value* lives in
   `node.uniform_values` keyed by **name** (`core.py:125`). A by-name setter touches the value dict;
   an input-shape setter must resolve name→hash via `get_uniform_hash` (`util.py:80`). Two different
   key spaces — a seam the agent will trip on.
4. **Render binding happens elsewhere.** Setting the dict does nothing visible until the next
   `node.render()` (`core.py:289-348`) reads `uniform_values.get(name)` and binds it. So the verb is
   "write the dict (+ release old) on the main thread; the existing per-frame render picks it up."
   No extra render call needed for scalars — the frame loop renders the current node every frame
   (`ui.py:176`).

**Why SHOULD not MUST:** the dev_flow says the triggering work absorbs the deferral. A tool author
*could* inline `try_to_release(old); node.uniform_values[name] = v` inside the tool handler. But
that handler runs on the **worker thread** and the write+release touches GL for sampler/buffer arms →
it would have to be marshalled anyway, and duplicating the release-dance in the tool layer is
exactly the leak the convention warns about. **Landing `App.set_uniform_value(node_id, name, value)`
(main-thread, does the release-dance, validates against `valid_input_types`) as a prep commit, then
re-pointing `widgets/uniform.py` at it, is the clean move** — it's the "already-solved-twin / shared
primitive" shape from the maintainer's debugging discipline, and it's ~15 lines. Blast radius: the
new method on `app.py` + one call-site swap in `widgets/uniform.py`.

### F2 — `create_node(template_id)`. CAN-defer-to-feature. 1 file.

**Evidence.** `app.py:1070-1083`:
```python
def create_node_from_selected_template(self) -> None:
    selected_template = self.ui_node_templates[self.app_state.selected_node_template_id]
    ...
```
It reads `app_state.selected_node_template_id` (set by the grid click `node_creator.py:41` and the
palette `app.py:360`). A tool wants `create_node(template_id)` without poking grid state.

**The clean verb.** Extract the body into `create_node(self, template_id: str) -> str` (returns the
new node id — the agent needs the id to address it next turn, mirroring ovelia's "name the affected
entity"), and make the existing method a one-liner:
`create_node_from_selected_template = lambda: create_node(self.app_state.selected_node_template_id)`
(as a real method, not lambda). GL-touching (warm-up render) → main-thread.

**Why defer not prep:** it's a pure-mechanical param hoist with no leak if deferred — the tool can
set `app_state.selected_node_template_id = tid` then call the existing verb (ugly but harmless,
single-threaded-marshalled). The clean refactor is 5 lines and naturally belongs in the commit that
writes the `create_node` tool. Not worth a standalone prep commit. (If F1 is being done as prep
anyway, fold this in — same wave, same file.)

### F3 — The GL-free/GL-touching partition. MUST-before. Doc-only.

This is the one thing that must exist before the spec, because it's the shared input the threading
design and the tool-layer design both consume, and getting it wrong is expensive (a verb wrongly
classified GL-free will crash the worker the first time a user asks it to load a texture). §2 above
is the deliverable. No code. **Status: satisfied by this document.**

### F4 — The `CopilotCapabilities` / `AppHandle` interface. CAN-defer-to-feature. New file.

The grounding (§4 marginalia, gap (c)) already names the design: a narrow injected interface of
GL-free + marshalled verbs that the tool layer imports *instead of* `App`. This is **the feature's
central architectural decision**, not prep — building it before the tools exist is speculative
scaffolding (you can't shape the interface without knowing the tools). Defer. The `ShaderLibFileManager`
(`file_ops.py`) is the working precedent for the pattern (App-free, callback-injected) and the
exporter `OutletUiDeps` (`base.py:36-53`) is a second precedent for "hand over capabilities without
exposing App." The interface will compose those.

### F5 — Secret storage for the Anthropic key. CAN-defer-to-feature. 1 file.

**Evidence.** `integrations.json` (`exporters/integrations.py`) already holds cleartext secrets:
`TelegramIntegration.bot_token` (line 26), `YouTubeIntegration.client_secret` + `token_json` (44-45).
`IntegrationsStore` (51-87) is one pydantic model rooted at `app_data_dir()`, with a thread-safe
`save()` (`_SAVE_LOCK`, 15) — and the worker thread already writes it (the comment at line 13 says
so). There is a standing todo deferral: **"integration credentials stored cleartext"** (todo.md
261-268) which explicitly notes *"One `IntegrationsStore` already centralizes all of it, so the
migration has a single seam."*

**Recommendation:** add `class AnthropicIntegration(BaseModel): api_key: str = ""` to the store
(mirrors the existing two, `extra="forbid"`). It inherits the save-lock (the worker needs to read the
key; the render thread writes it from a settings field) and the cleartext posture is *consistent*
with the existing two — no new security decision, the existing deferral already owns "migrate all
three to a keyring." An env-var fallback (`ANTHROPIC_API_KEY`) is a reasonable additional read-path
for power users but the persisted home is `IntegrationsStore`. **This is the right seam; extending it
is ~10 lines and belongs in the feature** (you want it next to the chat-settings UI that captures the
key). Not prep. Cross-reference: when this lands, the todo cleartext deferral should grow to mention
the third secret.

### F6 — `app.py` split. PREMATURE. (The headline architectural verdict.)

**Re-evaluated honestly given the new load.** `app.py` is 1083 L. The todo deferral
(`todo.md:156-161`) records a 2026-05-15 parallel-agent assessment: extraction would be premature
abstraction (the feature-002 reversed-AppContext shape). The grounding asks: does the copilot's
chat-state + worker-lifecycle push it over?

**Argue the split:** the copilot adds chat history, a worker thread + its lifecycle (start/stop/join),
a job/result queue, the agent loop's pending-command marshalling, streaming-event drain. That's
materially the same surface the Telegram exporter carries (~1257 L with its worker). If that all
lands on `App`, app.py crosses ~1300 L and the deferral's own trigger ("editing app.py feels
painful, lost search-and-replace") plausibly fires.

**Argue against (and this wins):** *none of that should land on `App`.* The exporters are the
precedent — `TelegramExporter` owns its **own** thread, queue, and lifecycle (`telegram.py:188-196`,
`_ensure_worker`/`_worker_main` 859-907); `App` holds only a registry reference. The copilot follows
the identical pattern: a `copilot/` package owns the worker, the queue, the agent loop, the chat
state. `App` gains at most a handful of members (a `copilot` handle, maybe an `is_chat_open` bool
joining the popup booleans at `app.py:174-180`, and the marshalled-command drain hooked into
`update_and_draw`). That is +20–40 lines on `app.py`, not +400. The conventions' three-layer rule
(`conventions.md:55-60`) and the exporter design decision (118-122) both point the new subsystem into
its own module. **Splitting app.py now is solving a problem the copilot won't create.** It also
violates the maintainer's own "small change, watch the blast radius" rule — a pre-emptive multi-file
app.py decomposition is a large diff with no triggering symptom.

**Verdict: do NOT split app.py as prep.** Land the copilot in `copilot/` mirroring `exporters/`.
Re-evaluate only if app.py actually crosses the painful threshold *after* the feature — and if it
does, the split that fires is "extract the editor-session domain" or "extract the shader-lib delegation
wall" (`app.py:496-577`, ~80 lines of pure property delegation), not a copilot-driven split.

### F7 — `current_node_ui_state_or_default` throwaway. SHOULD-before. 1 file.

**Evidence.** `app.py:619-626`:
```python
@property
def current_node_ui_state_or_default(self) -> UINodeState:
    if not node_id:
        return UINodeState()   # a FRESH throwaway — writes to it are silently lost
    return self.ui_nodes[node_id].ui_state
```
`tabs/node.py:84` and `widgets/media_ops.py:15` read this and **write back into it**
(`node_ui_state.uniform_sort_key = ...`, `...smoothing_window = ...`). When no node is selected, those
writes hit a discarded object — a silent no-op. Today harmless (the tabs don't draw with no node).
But the copilot will read/address node state **by id**, and a "set sort key on node X" tool that
routes through any "current/default" accessor inherits this silent-loss seam. **The agent needs
by-id accessors, not current-or-default.** Cheap prep: this isn't a bug to fix so much as a signal
that the agent's verbs must take an explicit `node_id` and resolve it against `ui_nodes` (with a
friendly "no such node" error, à la ovelia's `_resolve_id`), never lean on a "current" fallback.
File it as a design note for the spec; no code change required pre-feature. (Downgrade to a CAN-defer
if you read it as "just don't use the default accessor in tools" — which is the honest minimum.)

### F8 — `UINodeState` drops a node on bad known-key value. CAN-defer. 1 file.

Existing todo deferral (227-237): `load_node_from_dir` (`ui_models.py:359-380`) filters unknown keys
but a known key with an out-of-`Literal` value (a stale `uniform_sort_key` / `input_type`) raises
`ValidationError`, swallowed by `load_nodes_from_dir`'s except (393) → whole node silently lost.
**Orthogonal to the copilot** unless the agent writes node.json directly with a bad enum — which it
shouldn't (it sets state via verbs that snap to valid values). Leave deferred; its trigger (narrowing
a Literal) is unrelated to this feature.

### F9 — Editor-session clobber on disk write with unsaved edits. CAN-defer-to-feature. 1 file.

Grounding §3 (hot-reload) flags it. `_reload_if_changed` (`ui.py:108-119`) re-syncs a lib session
from disk **only if `session.editor.get_text() != new_text`** — but for the *node root* shader
(branch i==0, line 85-98) it calls `sync_editor_from_disk` (`app.py:839-849`) which does an
unconditional `editor.set_text(new_text)`, **clobbering the user's unsaved in-memory edits** if the
agent writes the node shader while the user is mid-edit. This is a real edge case the agent's
shader-write tool must own: check `app.is_current_editor_dirty()` (`app.py:801`) before writing, and
decide a policy (refuse / warn / merge). It's the tool's responsibility, designed *with* the tool —
defer to the feature. (Note: the lib branch already handles it correctly; the asymmetry is the
node-root branch.)

### F10 — No-`TYPE_CHECKING` cycle risk copilot ↔ App. PREMATURE-to-prep.

`conventions.md:38` bans `if TYPE_CHECKING`. The grounding (§6) warns this "will bite the copilot↔App
relationship hard." But the **resolution is F4's injected interface, which is the feature's own
design** — the tool layer imports `CopilotCapabilities` (a narrow Protocol/dataclass in a leaf
module), never `App`; `App` constructs the copilot and passes the capability bundle in (exactly how
`ShaderLibFileManager` and exporters already break the cycle). There is nothing to prep — the cycle
is broken by building the feature correctly. Flag for the spec; no pre-work.

### F11 — Chat-widget UI primitive. CAN-defer-to-feature. Feature.

`conventions.md:44` + the `/imgui-ui` skill: all UI flows through `ui_primitives.py` + `theme.py`.
The chat widget (scrolling message log, streaming-text append, role styling, a send box) has **no
existing primitive** in `ui_primitives.py` (781 L, button tiers + labeled fields + exporter-panel
chrome). It must build one — but building a chat primitive before the chat widget exists is
backwards. Defer; the primitive lands with the widget (and per `todo.md:281` ui_primitives has room
before its ~900 L split trigger). The `/imgui-ui` skill must be read at the start of that UI work
(it's a hard rule).

### F12 — Resolution-combo round-trip parse. CAN-defer. 1 file.

Existing deferral (todo.md:92-101): `tabs/node.py:105` re-parses `(w,h)` out of the human display
label. The copilot would set resolution via a verb wrapping `node.canvas.set_size((w,h))`
(GL-touching, §2) with explicit ints — it never touches the combo's string round-trip. Orthogonal;
leave deferred.

---

## 4. Adversarial: do we even need prep? (Argue it honestly.)

**The strongest case for ZERO prep:** The dev_flow rule is explicit — *deferrals are absorbed by the
triggering work as in-scope when the trigger fires.* The copilot IS the trigger for the
built-in-agent deferral (todo.md:287). Every gap here is reachable from inside the feature:

- F1 (uniform setter): the `set_uniform` tool's commit naturally extracts the verb — that commit is
  *already* touching `widgets/uniform.py`'s territory, so the "shared primitive" lands there for free.
- F2 (create_node arg): 5-line param hoist inside the `create_node` tool's commit.
- F4/F10 (the interface): can't be designed before the tools — doing it as prep is speculative.
- F5 (secret): wants to live next to the settings UI that captures the key — that's feature code.
- F7/F8/F9/F12: orthogonal or tool-owned edge cases.

So the minimum-change answer is: **write the feature; let each tool's commit absorb its gap.** The
maintainer's own rules back this — "what is the minimum change?" and "a small bug producing a large
diff is a signal you're solving the wrong problem." A pre-emptive refactor wave (split app.py, extract
a node_ops module, build a capabilities interface against imagined tools) is precisely the large-diff-
no-symptom anti-pattern.

**Where that case breaks (the gap that WOULD force an ugly workaround):** F3, the partition, is not
absorbable — it's not a code gap, it's *shared knowledge* the threading design needs before any tool
is written, and re-deriving it per-tool guarantees an inconsistency (one tool author assumes
set_uniform is GL-free, the next marshals it). That's why F3 is the lone MUST and why this document
exists. Secondarily, **F1 is the one where deferral risks a real leak**: if the uniform-set tool
inlines the dict-write on the worker thread without the `try_to_release` dance and without
marshalling, it leaks GL textures and races the render thread. The cost of pre-extracting the verb
(15 lines, one call-site swap) is trivially less than the cost of getting that wrong inside a worker.
So F1 earns "SHOULD-before" — not because deferral is impossible, but because the clean version is so
cheap and the dirty version is so easy to get wrong off-thread.

**Net:** the adversarial case is *mostly right*. Prep is one document (F3) + at most one tiny commit
(F1, optionally folding F2). Everything else is feature scope. Resist the urge to "tidy" app.py.

---

## 5. Recommended ordered refactor-prep sequence

1. **F3 — land this document** as the partition contract. (Done.) Feeds the threading agent + spec.
2. **F1 (+ F2) — one small prep commit, optional but recommended:** add
   `App.set_uniform_value(node_id, name, value)` (main-thread; does the `try_to_release` dance;
   validates input_type via `valid_input_types`) and re-point `widgets/uniform.py:228-230` at it;
   in the same commit hoist `create_node(template_id)` out of
   `create_node_from_selected_template`. Run `make check` + `make smoke` (touches UI/lifecycle). This
   is the only code prep; keep it behavior-preserving (the UI must look identical).
   *If the maintainer prefers absorb-into-feature, skip this — it's a clean defer too.*
3. **Everything else → into the feature spec as in-scope notes**, not prep:
   - F4/F10: design the `CopilotCapabilities` interface as the cycle-break (precedents:
     `ShaderLibFileManager`, `OutletUiDeps`).
   - F5: extend `IntegrationsStore` with `AnthropicIntegration`; update the cleartext deferral.
   - F6: copilot ships in a `copilot/` package mirroring `exporters/`; app.py gains only a handle +
     a marshalled-command drain in `update_and_draw`. Do NOT split app.py.
   - F7: agent verbs take explicit `node_id`, resolve against `ui_nodes`, never use a current-or-default
     fallback (ovelia `_resolve_id` shape).
   - F9: the shader-write tool checks `is_current_editor_dirty()` before clobbering.
   - F11: build the chat UI primitive in `ui_primitives.py` *with* the widget; read `/imgui-ui` first.

**One-line bottom line:** the prep that must happen before the copilot is *writing down the GL
partition* (this doc); the only worthwhile code prep is a ~15-line uniform-setter extraction; app.py
does not need splitting.
