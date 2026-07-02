# Slice 2 — Media literacy (the headline)

Give the copilot the ability to put a texture into a shader — the single biggest gap. This slice
introduces the reusable **user-file-pick primitive** (slice 4 reuses it) and is the cleanest possible
actor-model demonstration: the model can't synthesize a filesystem path, so it never types one — it
opens the user's own picker.

## Goal

The copilot can bind an image/video to a `sampler2D` uniform, by opening the native OS file dialog for
the user to choose the file. It can also add a NEW texture input (declare the sampler in source via the
existing `edit_shader`, then bind).

## The user-file-pick primitive (shared with slice 4) — GATE-family, NOT a bridge op

> **Corrected after pre-impl review (round 1).** The first draft routed the pick through
> `bridge.run_on_main` — WRONG on two counts (both code-grounded): (a) `pfd_block` busy-waits on the
> main thread (`util.py:108`), so a single `run_on_main` op would FREEZE the whole frame loop for the
> entire, user-paced, unbounded dialog lifetime; (b) `run_on_main`'s default `bridge_op_timeout_s =
> 5.0` (`config.py:59`) raises before a user can pick. The credential-gate is the correct parent
> precedent *because* it keeps the UI live: the worker blocks on the gate channel while the UI keeps
> drawing. So this primitive is **gate-family**, mirroring `GateKind.CREDENTIAL`.

Shape (a new `GateKind.FILE` on the gate channel) — NOTE it is NOT wired like `GateKind.CREDENTIAL`
(round-2 correction; the round-1 "mirrors CREDENTIAL / reuses the shipped pattern" framing was wrong
and would DEADLOCK):
- The tool handler (worker thread) calls `caps.pick_user_file(kinds)`, which **blocks on the gate
  channel** via `gate.ask` (no timeout — `gate.py`; woken by a pick, a cancel, or `cancel_all` on a
  turn Stop).
- **CREDENTIAL is a PRE-execute gate** — `agent.py` yields `AgentGateOpened` BEFORE `registry.execute`,
  and `session.py`'s `take_pending()` (the only current drive site) runs off that yield. The FILE gate
  is raised INSIDE `execute()` (the `bind_media` handler calls `pick_user_file`), where the worker
  generator is suspended and yields NOTHING → the existing `AgentGateOpened` path never fires → nobody
  calls `take_pending()` → deadlock. So the FILE gate needs a **NEW, independent main-thread per-frame
  poll** for a pending FILE request (separate from the `AgentGateOpened` flow) — this is net-new wiring,
  not a reuse.
- The **UI side** of that poll opens `pfd.open_file(...)` non-blocking and polls `dialog.ready()` /
  `.result()` across LIVE frames. NOTE `widgets/uniform.py:246` uses `pfd_block` (a BLOCKING busy-wait,
  `util.py:108`) — the cross-frame poll is NEW code modeled on `pfd`'s `.ready()`/`.result()` API, not
  a copy of that call site.
- **The FILE `GateRequest` carries `node_id` + `uniform`** so the MAIN side does the load+bind (GL) and
  answers the gate with a path-free `MediaBindResult`. The abs path is created and consumed on main
  (pfd runs on main) — it NEVER crosses back to the worker handler.
- **Late-pick reconciliation (round-2):** the native OS dialog's lifetime is NOT controlled by
  `cancel_all` (unlike the credential gate's in-app imgui widget). If a turn Stop cancels the gate
  mid-dialog, the worker is released but the native picker stays open; when the user later picks, the
  poll site must DROP the result (the gate is no longer pending). Name this in the UI-half.
- `pick_user_file` returns only a small result the model may see:

```python
@dataclass(frozen=True)
class MediaBindResult:
    # What the model is allowed to see. NO absolute path — the path is consumed on the main thread
    # inside the bind and never crosses back to the worker/handler (structural, not a redaction the
    # handler must remember: corollary-1 by construction, per conventions "Structural impossibility
    # over guard-piles").
    ok: bool
    basename: str = ""     # the PERSISTED basename (= media/<uniform>.<ext>, see decision 1) — safe
    width: int = 0
    height: int = 0
    is_video: bool = False
    cancelled: bool = False
    error: str = ""
```

- Why structural, not a `_abs_path` field: a leading-underscore dataclass field is still in `repr()`
  and `asdict()` — any handler that `f"{pick}"`s or builds `payload=asdict(pick)` would leak the path
  into `msg`/trace (`agent.py:816` logs the full `payload`; `trace.py:78` `str()`s it). Keeping the
  path off the returned object entirely is the only guard that can't be forgotten. See the automated
  `test_path_redaction.py` in Manual verification.

## Design decisions (lock-in)

1. **`bind_media(uniform, node="")` — opens the picker, binds the chosen file.**
   - Flow: `_capture_node(node_id)` FIRST (checkpoint register — decision 3) → validate `uniform` is a
     real `sampler2D` on the node (else a truthful reject naming the samplers that DO exist) →
     `pick_user_file(("image","video"))` (blocks on the gate; the abs path is consumed engine-side) →
     on a pick, load `media_class_for(suffix)(path)`, set it as the uniform's value on the main thread,
     persist the node (the existing `ui_models.save` `GL_SAMPLER_2D` branch writes
     `media/<uniform>.<ext>` — NOTE `file_name_wo_ext = uniform.name`, `ui_models.py:290`, so the
     persisted file is `media/u_tex.png`, NOT the picked `fire.png`) → recompile unnecessary (value
     bind) → return `MediaBindResult`.
   - Result facts (on the existing tool-result channel): `bound -> u_tex (512x512, image)` /
     `user cancelled` / the sampler-not-found reject. **The basename echoed is the PERSISTED one
     (`u_tex.png`), never the source `fire.png`** — which also keeps the source path off the channel.
     On success the slice-1 working-set row shows the binding — the model SEES it took.
   - `mutating=True`, `is_edit=False` (not a source edit — must not trip the edit-retry cap),
     `gate_policy=NONE` (the file gate IS the consent; no second imgui confirm).

2. **Adding a NEW texture input needs no new tool.** The model declares `uniform sampler2D u_tex;` via
   `edit_shader` (source), then calls `bind_media("u_tex")`. The prompt teaches this two-step; a
   sampler with no default is the correct GLSL and slice-1 shows it as `(no media bound)`.

3. **Revert semantics: a media bind is fully revertable FOR FREE — no checkpoint code change.**
   > **Corrected after pre-impl review (round 1).** The first draft planned to "expand checkpoint
   > scope to copy `media/`/`textures/`" — that work is REDUNDANT. The checkpoint already captures and
   > restores media: capture is a full `UINode.save` serialize (`checkpoint.py`/`backend.py:539`
   > `snapshot_node` → `n.save(dest.parent, dest.name, rebind=False)`), and `UINode.save`
   > unconditionally writes every sampler into `media/` (`ui_models.py:289-313`); restore is a
   > whole-dir replace (`revert.py:18-32` `_swap_in_snapshot` → rmtree + `staging.replace(dst)`),
   > media/textures included. `30_turn_rollback.md` decision-1 itself lists media/textures as part of
   > the serialized unit; decision-9's "only two text files" describes what the copilot *could reach*,
   > not a snapshot filter.
   - **The ONLY real requirement: `bind_media`/`unbind_media` register with the checkpoint container
     via `_capture_node(node_id)` on first touch** (exactly as `set_uniform` does, `backend.py:929`;
     `30_turn_rollback.md` decision 2: "a NEW mutating tool MUST register or its change escapes the
     net"). Miss this call and a media-only turn takes no snapshot → revert is a silent no-op. It is
     named in decision-1's flow and in Files touched.
   - `30_turn_rollback.md` decision 9's *justification text* is updated (media is now load-bearing
     because bind_media can change it — but the mechanism already handles it). No new `checkpoint.py`
     copytree, no new `app.py` restore code.

4. **`unbind_media(uniform, node="")` — reset a sampler to the DEFAULT image (LOCKED, re-scoped).**
   > **Corrected after pre-impl review (round 1).** There is NO empty/unbound sampler state:
   > `core.py:333` `_default_uniform_value` returns `Image(_DEFAULT_IMAGE_FILE_PATH)` for every
   > sampler, and `ui_models.save` REQUIRES a `MediaWithTexture`/`Texture` (raises otherwise). So
   > "remove the texture" cannot mean an empty slot — it means **reset to the default image**
   > (`RESOURCES_DIR/textures/default.jpeg`, `constants.py`), the same state a never-bound sampler
   > holds. `unbind_media` sets the value back to `Image(_DEFAULT_IMAGE_FILE_PATH)` + persists — and
   > because slice-1's save-skip DOESN'T write a `media/` file for a default sampler, the on-disk state
   > round-trips: no media file, reload re-seeds the default, the row reads `(no media bound)`.
   > `mutating=True`, `gate_policy=NONE`, `_capture_node` first (revertable per decision 3). A complete
   > op — "reset to default" is exactly what the slice-1 `(no media bound)` row detects. (Depends on the
   > slice-1 save-skip; without it, unbind would leave a `media/u_tex.png`=default on disk that reloads
   > as "bound" — the round-2 round-trip bug.)
   > **Reviewer dissent (recorded):** one reviewer argued unbind is low-demand and should be deferred.
   > Kept because a re-scoped "reset to default" is a coherent complete capability and the maintainer
   > prefers completeness; the tool-count cost is paid by lazy loading (slice 0). See Review history.

## Out of scope

- **Binding a render output / another node's output as a texture (feedback / ping-pong).** A real
  ShaderBox capability but a separate feature (multi-pass buffers). Trigger: a user asks for a
  feedback/trail/reaction-diffusion effect. `bind_media` phase-1 is picker-only.
- **A shared project asset pool** (bind the same image to many nodes from one library). Media is
  per-node today; a pool is its own model change. Trigger: a user reuses one texture across ≥3 nodes.
- Video trim/filters via the copilot (the UI has `media_ops`) — the copilot binds the file; trimming
  stays a UI-only control.

## Files touched

- `copilot/capabilities.py` — `MediaBindResult` value object; `pick_user_file` (returns
  `MediaBindResult`, path consumed engine-side) + `bind_media` + `unbind_media` Protocol methods.
- `copilot/backend.py` — `pick_user_file` blocks on the file gate; `bind_media`/`unbind_media` call
  `_capture_node` first, resolve node + sampler, load/reset the media on main, set the value, persist
  (the existing `save_ui_node` path writes `media/`).
- `copilot/gate.py` — a `GateKind.FILE` with `node_id`+`uniform` on the `GateRequest`; the worker-block
  half (`gate.ask`, woken by pick/cancel/`cancel_all`). The bind happens on MAIN and answers a path-free
  `MediaBindResult`; the abs path never returns to the worker handler.
- **A NEW main-thread per-frame FILE-gate poll** (`ui.py`/`session.py` drain area) — independent of the
  `AgentGateOpened` pre-execute path (which does NOT fire for a mid-`execute` gate — round-2). Opens
  `pfd.open_file` non-blocking, polls `.ready()`/`.result()` across LIVE frames (NEW code on `pfd`'s
  API — `widgets/uniform.py:246` uses the BLOCKING `pfd_block`, not this), does the load+bind on pick,
  answers the gate; DROPS a late pick if the gate was already cancelled (turn Stop).
- `copilot/tools/media.py` (new) — `bind_media`, `unbind_media`; registered in
  `registry.build_registry`. Lazy (`eager=False`) once slice 0 lands.
- `copilot/prompt.py` — a MEDIA paragraph in `_SYSTEM_PROMPT` (declare-sampler-then-bind; a picker
  opens for the USER to choose; you never see the path; check the working-set row to confirm).
- `copilot/backend.py` `bind_media`/`unbind_media` call `_capture_node` — NO `checkpoint.py`/`app.py`
  change (decision 3: media revert is already covered by the full-node snapshot). Update
  `30_turn_rollback.md` decision-9 *justification* only.
- `tests/_caps.py` — add a `_FakeCaps` field + a `minimal_caps` default for EACH new Protocol method
  (`pick_user_file`, `bind_media`, `unbind_media`) or `make check` (pyright Protocol conformance)
  fails. (Same obligation for every new capability method across all slices.)
- `tests/test_path_redaction.py` (new) — the automated corollary-1 guard (see Manual verification).
- `constants.py` — reuse `MEDIA_EXTENSIONS`; add a glsl-extensions tuple if slice 4 lands together.

## Manual verification

- **Falsifier (the bind took):** `bind_media("u_tex")` → pick an image → the working-set row flips from
  `(no media bound)` to `<- (WxH, image)` (bound-vs-default detected per slice-1 decision 1) AND a
  `render_image` shows the texture (hand-eye, maintainer). A "bound" claim with an unchanged row = FAIL.
- **AUTOMATED falsifier (path never leaks) — `tests/test_path_redaction.py`, mirroring
  `tests/test_credential_redaction.py`:** fake `pick_user_file` to consume a sentinel abs path; call
  `registry.execute("bind_media", …)`; assert the sentinel appears in NEITHER `msg`, `str(payload)`,
  NOR a rendered trace event. This is the load-bearing corollary-1 guarantee — an automated test, not a
  one-off grep (the credential channel has one; the path channel must too).
- **Falsifier (the UI stays LIVE during the dialog — the whole point of the gate re-shape):** while the
  file dialog is open, the app keeps rendering frames (the frame counter advances; the chat is still
  drawable) — it is NOT frozen. A `run_on_main`/`pfd_block` impl fails this (the loop is stuck in a
  busy-wait). This is the falsifier for the round-1→2 re-shape; without it the reshape is unverified.
- **Cancel path:** dismiss the dialog → `MediaBindResult(cancelled=True)`, no node mutation, no
  `media/` file written, no checkpoint left dangling. **Turn-Stop-mid-dialog:** Stop the turn while the
  picker is open → the worker is released (`cancel_all`); a LATE pick is dropped, not applied.
- **Non-sampler reject:** `bind_media("u_glow")` (a float) → a truthful reject listing the node's real
  samplers.
- **Unbind round-trip (falsifier for decision 4):** bind a texture, then `unbind_media("u_tex")` → row
  reads `(no media bound)` AND no `media/u_tex.<ext>` on disk after save AND a reload still reads
  `(no media bound)`. A row that stays `(WxH, image)` after reload = the round-2 round-trip bug (the
  save-skip wasn't wired). Also revert an unbind → the prior binding returns.
- **Revert (falsifier for decision 3):** bind media in a turn, then Revert the turn → the sampler
  returns to its PRE-turn binding (unbound if it was unbound, or the prior file) AND the on-disk
  `media/<uniform>.<ext>` matches the pre-turn state. Assert the disk file, not just the in-memory
  value — a revert that fixes memory but leaves the new binary on disk is the failure mode.
