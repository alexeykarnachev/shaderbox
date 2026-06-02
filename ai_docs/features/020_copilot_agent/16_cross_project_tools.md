# 020 · 16 — copilot cross-project tools: the whole project as the agent's workspace

The shader-editing vertical (020·12–15) is done + live-verified, but the agent is **trapped in the
current node** — it cannot see or touch other shaders, uniform VALUES, or the shader library. This wave
makes the whole project the agent's workspace: cross-shader read/edit, runtime uniform values, library
read + write, project-wide search — the **basic full copilot tool set** the roadmap banner gates a ship on.

Derived from a two-round request-driven audit (the canonical scenarios in §1 of `11_capability_wave_spec.md`
+ the maintainer's voice-message scenario set + a per-tool adversarial audit grounded against code on
disk). The audit artifacts are not checked in; their conclusions are this spec.

> **Note on `file.py:NN` references below:** line numbers are as-of-plan-lock provenance (what the code
> looked like when this spec was written) — they have since shifted as the wave landed. Treat them as
> "find the named symbol near here", not live coordinates; the symbol name is authoritative.

---

## Background — what's done, what this adds (verified, not assumed)

**Done + verified (the current-node vertical):** `get_current_shader` / `edit_shader` (whitespace-invariant
token match, 020·13) / `replace_lines` / `insert_after` (line-anchored, 020·14) / `get_compile_errors`,
plus the editor lock + a `(node_id, content)` source-freshness guard (020·15) and per-project chat
persistence (022). Default model `x-ai/grok-4-fast`.

**The trap (verified on disk):** the edit closures and the freshness guard are hardcoded to
`self.current_node_id` — `_copilot_apply_shader_edit` (`app.py:641`), `_copilot_apply_line_edit`
(`app.py:696`), `_copilot_current_shader_view` (`app.py:580/587`), `_copilot_freshness_reject`
(`app.py:601/611`). `_copilot_read_revision` is a single scalar `tuple[str, bytes] | None` (`app.py:370`),
reset to `None` at turn start (`app.py:839`). So the agent can edit only the active node, and reading a
non-current node gives it no way to then edit that node.

**The design frame (maintainer-decided):** "current shader" is **passive context**, not a mode — the agent
knows what node is open / rendering / what the user sees, but is never *confined* to it. Every code tool
takes an optional **target** that defaults to current. This dissolves the trap rather than adding a
mode-switch tool (no `select_node`).

---

## Goal

Ship the **8-tool cross-project set** (+ 2 always-in-prompt context blocks) so the agent completes every
now-wave scenario end-to-end: read/compare/search across all nodes + the lib, edit any node or lib file,
set runtime uniform values, create nodes and lib functions. The shipped current-node vertical keeps
working — **behavior-preserving on the current-node (default-arg) path, except two deliberate honesty
fixes**: the `_uniform_type_label` sampler fix (a sampler was mislabeled `float`) and the
`get_current_shader`→`read_shader` tool-name retirement. Not literally byte-for-byte; the freshness
refactor (Decision 2) is the real blast radius and is pinned + tested.

**The locked set:**

| # | tool | signature | needs_gl | gate | eager | one-line role |
|---|------|-----------|----------|------|-------|---------------|
| 1 | `read_shader` | `read_shader(nodes: list[str] = []) -> per-node: numbered source + uniform rows (type+value) + errors` | yes | none | yes | the one read — folds source/uniforms/values/errors over N nodes; `[]` = current |
| 2 | `edit_shader` | `edit_shader(old_str, new_str, target: str = "", replace_all=False)` | yes | none | yes | substring match-replace-recompile, any node or `lib:` file |
| 3 | `replace_lines` | `replace_lines(start, end, new_text, target: str = "")` | yes | none | yes | block rewrite by line range, any node or lib |
| 4 | `insert_after` | `insert_after(line, new_text, target: str = "")` | yes | none | yes | anchor-free insert; auto-creates an absent `lib:` file |
| 5 | `set_uniform` | `set_uniform(name, value, node: str = "")` | yes | none | yes | runtime VALUE write, scalar/vec only, explicit reject on sampler/block/engine-driven |
| 6 | `create_node` | `create_node(name, source="", switch_to=True) -> node_id` | yes | none | yes | the only verb that makes a node; empty source = compiling starter; node-only; `switch_to` controls the tab |
| 7 | `grep` | `grep(query) -> origin-labeled file:line hits across nodes + lib` | no | none | yes | the sole body-content discovery primitive — LOCATE (vs `read_lib` = full body pull) |
| 8 | `read_lib` | `read_lib(names: list[str]) -> full bodies of named lib functions` | no | none | yes | name → full body (the catalogue is in-prompt; `grep` returns hit-lines, this returns whole functions) |

**The current-node sentinel is uniform across the set:** every code tool means current node by the EMPTY default
— `read_shader([])`, `edit_shader(target="")`, `set_uniform(node="")`. No magic `"current"` token (one
convention, not two — a non-frontier model must not learn `["current"]` here and `""` there). The
description of each tool states "omit / empty = the shader you're working on; this NEVER means all."

**Target-kind parse rule (the unified-target discriminator, pinned):** a `target`/`node` arg is a **lib
file IFF it starts with `lib:`**; otherwise it is a **node-id**. A non-existent node-id is an explicit
`error: no node with id <x> — call read_shader or check the node tree`, NEVER a silent fallback to a lib
path. The agent obtains real node-ids from the in-prompt tree and real lib paths from `read_lib` (which
returns each function's `lib:<path>` address) — it copies addresses, never constructs them. An empty
`target`/`node` (`""`/`[]`) resolves to the current node-id **before** anything keys on it (see Decision 2).

**Always-in-prompt (zero tools), assembled per turn into the system context:**
- **Node tree** — `id + name + has_errors + is_current` for every node. **No uniforms** (see Decision 9).
- **Lib catalogue** — `name + signature + doc` for every `SB_*` function. No bodies (those are `read_lib`).
- **Code conventions** — naming/structure rules the agent (not the user) follows, to keep `grep` effective.

**Cut from this wave:** `delete_node` (the gate-UI wave — Decision 7). `list_nodes` / `get_node_summary` /
`list_lib_functions` → folded into the in-prompt trees. A separate input-shape tool → it's a code edit.
`get_shader_source` + `get_compile_errors` → merged into `read_shader`.

---

## Out of scope (each with a trigger)

- **`delete_node` + the gate-UI wave.** The `GateChannel` body is unbuilt (`agent.py:191` is `_ = gate`;
  `take_pending`/`answer` have no caller; the loop never checks `requires_gate`). `delete_node` is the
  first `ALWAYS`-gate tool, so it ships WITH that wave. **Trigger:** the gate-UI wave starts, OR a user
  asks the copilot to delete a node and the honest-decline ("delete it from the node grid") is friction.
- **Eager-recompile-one-caller for lib edits.** A lib edit returns an honest "written; errors surface when
  a caller recompiles" string (Decision 4). The richer variant — eagerly recompile a node that already
  calls the function and surface ITS remapped errors — is deferred; it is also structurally inert for the
  headline case (creating a NEW lib function, which has no caller yet). **Trigger:** lib edits feel
  blind enough in practice that synchronous feedback on *already-referenced* functions is wanted.
- **Uniforms in the in-prompt node tree.** The tree carries names + has_errors only (Decision 9); uniform
  names need a GL read and uniform values bust the prompt cache. The agent fetches uniforms on demand via
  `read_shader`. **Trigger:** practice shows the agent flailing without uniforms in the project map (e.g.
  it grep-spams to discover uniforms it could have seen in the tree).
- **Render / publish / docs tools** (`render_image`/`render_video`, telegram/youtube, `read_doc`) and the
  lazy catalogue-nav (`search_tools`/`list_tools`). Later waves. **Trigger:** the editing set is shipped
  and a render-or-publish scenario is the next priority. (Constraints like the 3s sticker cap will live in
  those tools' arg schemas, not handler logic — design note, not this wave.)
- **Uniform display-shape toggle** ("show u_color as a slider not a color picker"). That writes
  `UIUniform.input_type` (metadata), a different object from the `uniform_values` `set_uniform` writes.
  The agent honest-declines it (named in the prompt's out-of-scope list). **Trigger:** a real workflow
  wants the agent to re-shape a control's widget.
- **The copilot system prompt describing the freetype glyph-atlas text shader.** `todo.md` flags that the
  shipped text template is an SDF segment synth, not the dead `fonts.py` atlas — this wave's prompt rewrite
  must NOT advertise a capability the agent doesn't have, but it does not resolve that doc-drift entry
  (the prompt simply won't mention text-rendering internals). **Trigger:** unchanged — that entry's own.
- **GLSL-aware grep.** `grep` is a substring match — `u_time` in a comment is a false positive, a `#define`
  alias a false negative. Acceptable for rare discovery queries. **Trigger:** false hits make `grep`
  unreliable enough to want a real parser-backed index.

---

## Design decisions (numbered — lock-in only)

**1. "Current shader" is passive context, not a mode; every code tool takes an optional target.**
`read_shader(nodes=[])`, `edit_shader(..., target="")`, `set_uniform(..., node="")`,
`replace_lines`/`insert_after(..., target="")` all default to the current node (the empty default — one
sentinel across the set) and accept an explicit id to act elsewhere. No `select_node` tool — the agent is never confined and never switches the user's tab to
work. This is the alternative to round-1's `select_node` recommendation; the maintainer chose it (voice
note, this session) because it serves "read/edit 2-3 shaders at once" and "edit 3 different nodes in one
turn" without a mode-switch.

**2. Per-node freshness keying — the spine of the wave (touches the DONE vertical).**
`_copilot_read_revision` changes from a single scalar `tuple[str,bytes]|None` (`app.py:370`) to
`dict[node_id -> digest]` (reset to `{}` at turn start, was `None` at `app.py:839`). `read_shader` stamps
**every** node it reads; `create_node` auto-stamps the new node; the freshness reject keys off the edit's
**resolved target node-id**, NOT `self.current_node_id`. Without this, `read_shader([X])` →
`edit_shader(target=X)` silently rejects as stale (the guard at `app.py:601/611` compares against the
current node) and the agent loops forever.

**Three implementation pins the pre-impl review surfaced — all REQUIRED, the "byte-for-byte" claim is
false without them (verified on disk):**

- **(2a) Resolve the empty sentinel FIRST.** `target=""` / `nodes=[]` must resolve to the concrete
  `current_node_id` *before* the digest is keyed or looked up. Keying the dict literally on `""` breaks
  the current-node path (the exact infinite-loop this decision exists to kill). Every stamp and every
  reject lookup operates on a resolved concrete id.
- **(2b) `_copilot_persist_shader` must take and use the TARGET id, not `self.current_node_id`.** Today
  (`app.py:670-683`) it hardcodes `self.current_node_id` in BOTH `sync_editor_from_disk(...)` (`app.py:680`)
  and the chain-edit re-stamp (`app.py:682`). Threading the target *node object* but leaving these two
  reads means a chained non-current edit (a) re-stamps the WRONG dict key → the second same-target edit
  stale-rejects, and (b) **pushes the non-current node's new text into the CURRENT node's editor session**
  — a visible editor corruption, not a cosmetic. The persist tail must thread the resolved target id and
  use it for both the sync and the re-stamp.
- **(2c) Re-derive the "you switched nodes" reject branch.** Today `app.py:601/611` reads
  `nid = self.current_node_id` and the identity branch (`rev[0] != nid`) emits a distinct stale message.
  Under dict keying the reject's first branch becomes a key-presence check (`target not in dict`), and the
  "switched nodes" wording must be preserved for the genuine case (read node A, user switches to B
  mid-turn, agent edits A). Don't let the migration silently collapse the two branches into one message.

**Regression boundary:** the current-node path (`target=""`, `nodes=[]`, resolved to current) is
behavior-preserving — but only with 2a-2c. The manual checks below (E1 + the two new freshness checks)
exercise both the current path AND the non-current/chained path; the dict keying is also a committed
headless unit test (Files-touched), not just a manual check.

**3. Unified target HOLDS as one arg + one edit-tool set, but the RETURN CONTRACT is target-kind-aware.**
A node and a lib file are both "a place that holds GLSL source," edited by identical substring/line ops —
so one `target` arg, no separate `write_lib` tool (which would be a 4th near-identical edit verb,
the exact similar-tool pair grok-4-fast mis-selects). **What differs is feedback:** a node edit recompiles
THAT node and returns its errors synchronously (`_copilot_persist_shader`, `app.py:677-683`); a lib file
has **no standalone compile** (`ShaderLibIndex.build`, `index.py:51-73`, is a pure regex/brace parse — a
lib function compiles only when spliced into a consuming node, `resolver.py`). So the edit tools branch on
target kind for the RETURN STRING only (arg list unchanged): a node target returns compile errors; a lib
target returns the Decision-4 honest string.

**4. Lib-edit feedback is an honest string (no eager recompile this wave).**
A `lib:` edit returns: *"lib file written; it has no standalone compile — errors will surface when a node
that calls SB_foo recompiles."* This is always truthful and neutralizes the silent-success failure mode
(a uniform "ok — compiled clean" on broken lib GLSL is a lie). The eager-recompile-a-caller refinement is
out of scope (and inert for new-function creation — there is no caller yet). **Consequence (stated for the
prompt):** creating or editing a lib function has NO compile validation until a node is edited to call it;
that node-edit's own compile is the only signal. The agent must be told this.

**5. Lib-file CREATION via `insert_after` on an absent `lib:` target — writing LIVE source, not the UI stub.**
"Factor into lib" (the headline scenario) needs a NEW lib file. `insert_after(line=0, new_text=<fn>,
target="lib:foo.glsl")` on a non-existent path creates the file and writes the agent's function as its
body. **It must NOT route through `ShaderLibFileManager.create_file_in`** — that seeds a *commented-out*
stub (`file_ops.py:151`: `/// doc` + `// float SB_my_function...`), and the index extracts functions from
*uncommented* signatures (`_extract_functions` calls `strip_comments` first, `index.py:103`), so a file
created that way contributes ZERO functions until overwritten. The copilot lib-create path writes the
function as live, uncommented source so the index picks it up immediately. (The non-copilot UI
`create_file_in` stub is unchanged — it's a human affordance.) **The copilot lib-create path MUST reuse
`ShaderLibFileManager._validate_target(..., kind="new-file", suffix=".glsl")`** (the same guard
`create_file_in` uses, `file_ops.py:144`) — bypassing `create_file_in` must NOT bypass its path-traversal
validation, or `insert_after(target="lib:../../etc/x")` escapes `shader_lib_root`. Validate, then write
live source.

**6. `set_uniform` writes the runtime value; up-front shape validation is the ONLY feedback channel.**
The value lives in `node.uniform_values: dict[str, Any]` (`core.py:125`), NOT `UIUniform` (metadata only,
`ui_models.py:64`). The closure mirrors the UI write path (`widgets/uniform.py:228-230`: `try_to_release`
the old value, then dict-assign). The render-time shape-mismatch pop (`core.py:343-348`) runs on the MAIN
thread a frame later — DETACHED from the tool's bridge round-trip — so the handler never sees it. Therefore
the handler must validate shape **up front** and return an explicit error; it must reject samplers
(`GL_SAMPLER_2D`) and UBO blocks (`moderngl.UniformBlock`) — not JSON-expressible — with a clear message,
never the silent pop. **`value` JSON shape:** scalar → int/float; vec/color → list. **No min/max metadata
exists anywhere** — so a relative tweak ("20% brighter") works (the agent reads the current value via
`read_shader`, computes, writes — monotonic, no vision needed), but "set it to a *good* value" is
unsupported (a known limit, not a tool gap). **Also reject the engine-driven uniforms `u_time` / `u_aspect`
/ `u_resolution`:** `Node.render()` (`core.py:319-329`) overwrites these every frame from the engine
clock/canvas regardless of `uniform_values`, so a `set_uniform("u_time", 5.0)` is a per-frame silent
no-op — the handler must reject them up front (and the prompt names them as engine-controlled), never
return a phantom "ok".

**7. `set_uniform` needs no freshness keying.** It writes by uniform NAME, immune to source staleness — a
stale source digest is irrelevant to a value write. (Unlike the edit tools, which target source positions.)

**8. `create_node` is node-only, binds the RAW unguarded creator, and takes an explicit `switch_to`.**
`create_node(name, source="", switch_to=True)` makes a node (empty source = the compiling starter from
`_seed_starter_node`, `app.py:1592` — never a blank buffer, which would burn a self-correction loop). It
does NOT create lib files (Decision 5 owns that — keeping `create_node` node-only avoids a name-vs-path arg
ambiguity). It must bind a RAW create body + `set_current_node_id` (`app.py:1115`), NOT
`create_node_from_selected_template` (`app.py:1607`) — that early-returns on `_copilot_busy_blocked`
(`app.py:856`, True for the whole copilot turn) and would refuse the copilot's own call.

- **`switch_to` (default True) is the agent's explicit focus control** — it resolves the tension with
  Decision 1 ("the agent never switches the user's tab" was about EDITING an existing node, not creating
  one). `switch_to=True` (the common "make me a plasma node" case): the new node becomes current, the
  user sees it, and `edit(target="")` lands on it. `switch_to=False` (a background sub-step of a larger
  multi-node task, e.g. authoring three new nodes): the user's view is untouched and the agent edits via
  `target=<returned-id>`. **Either way the new node is freshness-auto-stamped** (`dict[new_id]`, Decision
  2) so the agent can edit it immediately without a re-read; the ONLY difference is whose tab is showing.
  A defaulted boolean the agent ignores in the common case is within the elbow (not arg-overload).
- **Insert order is pinned (mirror `_seed_starter_node`, `app.py:1599-1601`): save → insert into
  `self.ui_nodes` → THEN `set_current_node_id`.** Setting current before the node is in `ui_nodes` opens
  a window where `current_node_id` points at a missing key, which a mid-frame bridge-drain render would
  observe and trip the `scripts/smoke.py` `current_node_id ∈ ui_nodes` invariant.

**9. The in-prompt node tree is lean (id + name + has_errors + is_current) and GL-free.**
`_node_summary` (`app.py:556`) calls `get_active_uniforms()` — a GL read — to get uniform names; and
`build_context` (`context.py:17`) runs on the WORKER thread with no bridge. So a uniform-carrying tree
would force a per-turn `run_on_main` just to build the prompt AND inject render-volatile data into the
cache-warm prefix. The tree stays GL-free: `has_errors` reads the cached `compile_unit.errors` field (no
GL). "Which shaders use u_time?" routes to `grep`, not the tree. Promote uniforms into the tree later only
if practice demands it (Out-of-scope trigger).

**10. `read_lib` is body-fetch only; `grep` drops its scope arg.**
The lib *catalogue* (names + signatures + doc) is in-prompt, so `read_lib`'s catalogue mode is dead —
it reduces to `read_lib(names)` → full bodies. `grep` loses the `{nodes, lib, all}` scope arg: a single
scopeless search over nodes + lib bodies with origin-labeled hits is strictly more universal and removes
the one arg-confusion surface (and the spec's old `docs` scope had no runtime corpus anyway).

**11. Prompt-cache health is preserved by keeping volatile state out of the warm prefix.**
The system prompt is ordered least-volatile → most-volatile for OpenRouter prefix caching (`prompt.py:6-9`).
The node tree is *rare-volatility* (changes on create/delete/rename/compile-flip, NOT per-frame) **only
because it excludes uniform values** (which converge with the GPU only after a render). Keep it value-free
and it stays cacheable. The current shader source stays OUT of the prompt — it enters as a `read_shader`
result AFTER the warm prefix (the shipped §B1a design) and is history-stripped next turn.

---

## Files touched

**New / changed in `copilot/`:**
- `tools/shader.py` — the read tool becomes `read_shader` (list arg, merged source+uniforms+errors); the
  three edit tools gain the `target` arg + target-kind-aware return; `set_uniform` / `create_node` /
  `grep` / `read_lib` added. (The shipped `get_current_shader` / `get_compile_errors` tool names retire
  into `read_shader` — a small tool-name change to the verified vertical, noted in the ledger.)
- `tools/registry.py` — register the new tools.
- `capabilities.py` — new seam closures: list-read, target-addressable apply (node + lib branches),
  `set_uniform`, raw `create_node`, `grep`, `read_lib`, lib-file create/write. Leaf-rule holds.
- `context.py` — `CopilotContext` gains the node tree + lib catalogue + conventions (GL-free build).
- `prompt.py` — `_SYSTEM_PROMPT` rewrite (§ "Prompt changes" below) + render the in-prompt blocks in
  cache-strata order.

**Changed in `app.py` (the capability closures + the freshness machinery):**
- `_copilot_read_revision` scalar → `dict[node_id -> digest]` (reset to `{}` at `app.py:839`); stamp
  per-read under the resolved id; reject keyed off the resolved target id, NOT `current_node_id`
  (`app.py:370/587/601/611/682/839`). See Decision 2 pins 2a-2c.
- The edit apply closures thread the **resolved target id** through the shared persist tail
  (`app.py:641/696`). **`_copilot_persist_shader` (`app.py:670-683`) must take that target id and use it
  for BOTH `sync_editor_from_disk(target_id, ...)` (`app.py:680`) and the chain re-stamp (`app.py:682`)** —
  today both hardcode `self.current_node_id`. With the target id, `sync_editor_from_disk` (which resolves
  by path and no-ops if no session is open, `app.py:1344-1354`) correctly syncs the target node's own
  editor session (or no-ops for a non-current node with no open editor) instead of corrupting the current
  node's editor with the target's text.
- New raw `create_node` body (bypasses `_copilot_busy_blocked`, pins save→insert→set-current order) +
  freshness auto-stamp + the `switch_to` branch (Decision 8).
- New `set_uniform` closure (validate up-front + `try_to_release` + dict-assign, bridge-marshalled).
- `_uniform_type_label` (`app.py:210-216`) fix: label `GL_SAMPLER_2D` / blocks distinctly (today a sampler
  mislabels as `float`) so `read_shader`'s type oracle is honest and `set_uniform`'s reject is consistent
  with what the agent was shown.
- A grep closure over `ui_nodes` text + `shader_lib.active().sources` (GL-free worker).

**`shader_lib/file_ops.py`** — a live-source lib-create path for the copilot (Decision 5, reusing
`_validate_target` for traversal safety), distinct from the human `create_file_in` stub.

**`tests/`** — a committed headless unit test for the freshness dict-keying (the mechanical layer the
manual checks can't reach without an LLM): (a) `read_shader([X])` then `edit(target=X)` succeeds; (b) a
genuinely-stale current edit still rejects; (c) a chained second edit on the same non-current node
re-stamps the right key and succeeds; (d) `target=""`/`[]` resolves to current and the current path
rejects/accepts exactly as the shipped scalar did. This is a deliverable, not just a manual check.

**Docs (same wave):** correct `11_capability_wave_spec.md` §3 stale facts surfaced by the audit — the
`list_nodes` "GL-free with uniform names" contradiction (it's a GL read), the `grep` `scope`/`docs` arg
(dropped), `delete_node` "always gates" (cut to the gate wave), the E4 "drag is the only valid shape"
rationale (vec3/4 also allow `color`). Flip the `020` roadmap row / banner on done.

---

## Manual verification (by hand in the app — most are user-visible)

The headless `make smoke` + `make check` gate the mechanics; these are the agent-behavior checks a real
turn must pass (maintainer-driven, real OpenRouter tokens):

- **E1 (regression — the current-node path):** on the current node, "animate the position uniform";
  introduce a break; confirm the agent reads the source-mapped error and self-corrects. Proves Decision 2
  didn't regress the current-node single-edit path.
- **E1b (chained non-current edit — the freshness blast-radius check, NEW per review):** while on node A,
  ask the agent to fix a compile error in node B that needs TWO consecutive edits to B (a break it
  introduces then fixes), without an intervening read. Confirms the persist re-stamp (Decision 2b) keys
  node B, not A — the second edit must NOT stale-reject, and node A's editor must be untouched (no
  cross-node editor corruption).
- **E1c (node-switch mid-turn, NEW per review):** the agent reads node A, then the user clicks to node B
  before the agent's edit; confirm the agent gets the "you switched nodes" reject (Decision 2c), not the
  "never read this turn" message — the wording must survive the dict migration.
- **E3 (the acid test):** with 3 nodes sharing an inline rotation matrix, "extract the rotation into a lib
  function and call it from all three." Confirm: a NEW lib file is created with LIVE source (Decision 5),
  all three nodes are edited (cross-node, Decision 2), each node recompiles clean, and the lib edit's
  feedback string is honest (Decision 4).
- **R2:** "compare how node A and node B do their lighting" — one `read_shader` call with two ids.
- **U2 (cross-node-safe value tweak):** "make the glow ~20% brighter" — the agent reads the current value
  + type via `read_shader`, computes, sets it; the preview brightens. Then try setting a sampler uniform
  and confirm an explicit reject (Decision 6), not a silent no-op.
- **N1:** "scaffold a basic SDF raymarching shader" — a node is created with compiling starter source and
  the follow-up edit lands on it without a `select_node` (Decision 8).
- **R5 / R4 (zero-call orientation):** "what shaders do I have and which are broken?" answered with NO tool
  call (in-prompt tree); "which shaders use u_time?" routes to `grep` (the tree has no uniforms).
- **Decline:** "set up a new project" / "what GPU am I on" / "undo that" — honest declines, no tool fired.

UN-HEADLESS-ABLE: these need a live LLM + the running app; `scripts/smoke.py` cannot exercise agent
reasoning. The mechanical layer (freshness dict keying, the target thread-through, the lib-create writing
live source, `set_uniform` rejects) IS unit-testable headlessly and should be.

---

## Open questions for the user (resolved with robust defaults — flagged for review)

All three forks the audit surfaced were settled with the maintainer this session; recorded here so a cold
reader sees the decision + the revisit lever:

1. **`delete_node`:** deferred to the gate-UI wave (Q1 — confirmed). The gate body is a whole sub-wave the
   maintainer fenced out; delete returns then. Honest-decline this wave.
2. **Lib-edit feedback:** honest string, no eager recompile (Q2 — confirmed; eager is inert for the
   new-function case anyway).
3. **Uniforms in the node tree:** out for now — lean tree + on-demand `read_shader` (Q3 — "we'll see in
   practice"). Promote later behind the Out-of-scope trigger.

---

## Review history

Drafted from a two-round adversarial tool-usefulness audit (the design decisions already carry their
counter-arguments). Pre-implementation review (3 agents — correctness/design, verification/blast-radius,
spec-fidelity-vs-code).

**Round 1.** Spec-fidelity reviewer (anchored to code on disk, the non-self-authored artifact): **PASS** —
all ~30 file:line claims verified true, zero wrong (the freshness scalar, the hardcoded current-node
edit path, the commented lib stub vs strip-comments extractor, the detached silent uniform pop, the
busy-blocked create guard, the unbuilt gate). Correctness/design + verification/blast-radius: **PARTIAL**,
all findings on the NEW arg/return surface and the freshness refactor's blast radius — none was
decision-drift. Real findings, fixed inline in this spec:
- **[HIGH] persist-tail target-id (verified app.py:680/682):** `_copilot_persist_shader` hardcodes
  `self.current_node_id` for both the editor-sync and the chain re-stamp → a chained non-current edit
  would corrupt the current node's editor + stale-reject the second edit. → Decision 2b + Files-touched.
- **[HIGH] freshness sentinel + branches:** `target=""` must resolve to current before keying; the
  "switched nodes" reject branch must survive the scalar→dict migration. → Decision 2a/2c.
- **[HIGH] current-sentinel mismatch:** `read_shader(["current"])` vs `target=""` elsewhere = two
  conventions for a non-frontier model. → unified to empty-default across the set.
- **[HIGH] target-parse rule unpinned:** lib-iff-`lib:`-prefix; bare unknown node-id = explicit error,
  never silent lib fallback. → pinned in the tool table.
- **[MEDIUM] engine-driven uniforms** (`u_time`/`u_aspect`/`u_resolution`, verified core.py:319-329) are
  per-frame silent no-ops → `set_uniform` rejects them. → Decision 6.
- **[MEDIUM] create_node insert order** (verified vs `_seed_starter_node`) → pinned save→insert→set-current.
- **[MEDIUM] lib-create traversal:** reuse `_validate_target` when bypassing `create_file_in`. → Decision 5.
- **[MEDIUM] "byte-for-byte" overstated** (the label fix + tool rename change the current path's output) →
  reworded to "behavior-preserving except two honesty fixes".
- **[MEDIUM] verification gap:** added E1b (chained non-current edit) + E1c (mid-turn switch) manual checks
  and committed the freshness-keying headless unit test as a deliverable.

**Escalated to the maintainer (resolved this session):** should `create_node` switch the user's tab?
→ added an explicit `switch_to=True` arg (Decision 8) — the agent controls focus per call, resolving the
tension with Decision 1 without a policy baked in.

A re-spawn of the same reviewers against this patched spec is the closing gate before plan-lock.
