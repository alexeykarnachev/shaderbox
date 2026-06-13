"""Headless dogfood harness for the copilot ENGINE (feature 026).

A HAND-DRIVEN driver: a human (Claude) imports this, constructs a real `ProjectSession`
on a standalone EGL context (no App, no glfw window), drives `session.copilot` turn by
turn against a REAL LLM, renders nodes to small PNGs, and EYEBALLS them. The judge is the
human reading the printed events / the trace transcript + looking at the rendered images —
there are NO code assertions and no pass/fail. The point is dogfooding: surface where the
copilot is weak, where context wastes tokens, what's missing from context.

Built on feature 025's `ProjectSession` (the headless project + copilot core). This box is
a display-less Raspberry Pi where glfw can't open a window but a standalone EGL context
reaches the real V3D GPU.

ENV (load-bearing, set at module top BEFORE the shaderbox imports below): `SHADERBOX_DATA_DIR`
redirects the data dir to an isolated run dir (so lib edits + the written `integrations.json`
never touch the real library / creds — `paths.app_data_dir()` reads it at import time); the
MESA overrides give the V3D driver `#version 460` (read at context creation). The OpenRouter
key comes from the `OPENROUTER_API_KEY` env var (export it before running). The model defaults to
`CopilotIntegration.model` (the in-tree default); set `OPENROUTER_MODEL` only to override it.

A caller-provided `SHADERBOX_DATA_DIR` wins (the module-top `setdefault` no-ops) — that is the
RESUME seam (feature 027): set it (+ pass `project_dir`) on the COMMAND LINE before `uv run`, never
in-script after import (the env block runs at import). All run artifacts live under
`scripts/dogfood/runs/`.

THREADING (load-bearing): `CopilotSession` spawns a worker thread for the turn; the worker
marshals GL ops back to the context-owning (main) thread via `bridge.run_on_main`, which
BLOCKS until the main thread drains it. So the harness drive loop MUST pump the bridge +
events from the main thread (the one that created the EGL context), exactly as `App`'s frame
loop does — a synchronous-bridge patch would run GL on the worker thread and corrupt the
context. `drive_until_idle` is that pump. A gate pauses the worker mid-turn; it can ONLY be
answered within the SAME process (the worker is daemon=False, dies on exit, and a gated turn
is never persisted) — there is no answer-a-gate-after-resume.

Usage (from a REPL / chat-driven loop, with OPENROUTER_API_KEY exported):

    from scripts.dogfood import DogfoodHarness
    h = DogfoodHarness.create()                 # fresh run project, EGL context, real copilot
    h.send("Create a shader that draws a filled white circle in the center.")
    h.drive_until_idle()                        # pump; auto-prints events + any gate
    png = h.render()                            # 400x400 PNG of the current node
    # ...open png with Read, eyeball it...

Interactive one-blocking-call-per-turn (feature 027): resume a prior run + emit a structured
JSON turn-result, so a fresh process per turn keeps full state via disk:

    h = DogfoodHarness.create(project_dir=Path("scripts/dogfood/runs/proj-XXXX"))
    h.send("..."); h.drive_until_idle(); h.dump(Path("scripts/dogfood/runs/turn.json"))
"""

import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

# All dogfood run artifacts (data dir, per-run project dirs, JSON dumps, traces, PNGs) live
# under scripts/dogfood/runs/ — one consolidated, gitignored home (feature 027).
_RUNS_DIR = Path(__file__).resolve().parent / "runs"
_RUNS_DIR.mkdir(parents=True, exist_ok=True)

# --- env MUST be set before the shaderbox imports below (paths.app_data_dir reads it at
# --- import time; the MESA overrides are read at EGL context creation). A caller-set
# --- SHADERBOX_DATA_DIR wins (setdefault no-ops) — the resume seam (feature 027). -------
_DATA_DIR = Path(
    os.environ.setdefault(
        "SHADERBOX_DATA_DIR",
        tempfile.mkdtemp(prefix="data-", dir=_RUNS_DIR),
    )
)
os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "4.6")
os.environ.setdefault("MESA_GLSL_VERSION_OVERRIDE", "460")

# Write the OpenRouter creds into the isolated integrations.json (the client reads it live).
# Only override the model when OPENROUTER_MODEL is set — otherwise let CopilotIntegration's
# own default apply (single source of truth; no duplicated model string to go stale).
_INTEGRATIONS = _DATA_DIR / "integrations.json"
if not _INTEGRATIONS.exists():
    _copilot: dict[str, str] = {
        "openrouter_key": os.environ.get("OPENROUTER_API_KEY", "")
    }
    _model = os.environ.get("OPENROUTER_MODEL", "")
    if _model:
        _copilot["model"] = _model
    _INTEGRATIONS.write_text(json.dumps({"copilot": _copilot}), encoding="utf-8")

import glfw  # noqa: E402
import moderngl  # noqa: E402

# core.py's render path reads glfw.get_time() for the default u_time; glfw is never init()'d
# here (we use EGL, not a glfw window), so it warns "not initialized" and returns 0.0 — which
# is exactly the static t=0 frame the dogfood wants. Silence the cosmetic warning.
glfw.set_error_callback(lambda code, desc: None)

from shaderbox.constants import (  # noqa: E402
    NODE_TEMPLATES_DIR,
    STARTER_TEMPLATE_ID,
    TEMPLATE_ORDER,
)
from shaderbox.copilot.capabilities import RenderResult  # noqa: E402
from shaderbox.copilot.persistence import ConversationStore  # noqa: E402
from shaderbox.copilot.session import CopilotSession  # noqa: E402
from shaderbox.copilot.state import Message  # noqa: E402
from shaderbox.exporters.registry import ExporterRegistry  # noqa: E402
from shaderbox.media import texture_to_pil  # noqa: E402
from shaderbox.notifications import Notifications  # noqa: E402
from shaderbox.project_session import ProjectSession  # noqa: E402
from shaderbox.shader_lib.file_ops import ShaderLibFileManager  # noqa: E402


class DogfoodHarness:
    """Owns the EGL context + a real headless `ProjectSession` + the drive loop.

    Construct via `DogfoodHarness.create()`. The context-owning thread is whichever thread
    calls `create()`; all drive methods (`send` / `drive_until_idle` / `render`) must run on
    that thread — the worker marshals GL back to it.
    """

    def __init__(
        self, ctx: moderngl.Context, session: ProjectSession, project_dir: Path
    ) -> None:
        self._ctx = ctx
        self.session = session
        self.project_dir = project_dir
        self._seen_msg_count = 0  # incremental event printing (drive_until_idle)
        self._dumped_msg_count = (
            0  # incremental JSON slice (dump) — separate from printing
        )
        self._last_render_path = ""  # echoed in the dump payload if a turn rendered

    @classmethod
    def create(
        cls, project_dir: Path | None = None, *, seed_templates: bool = True
    ) -> "DogfoodHarness":
        """Build the EGL context + a real `ProjectSession` + restore the conversation if resuming.

        `project_dir=None` -> a fresh mkdtemp'd project (seeded unless `seed_templates=False`).
        `project_dir=<existing run dir>` -> RESUME: skip seeding (the nodes persist from prior
        turns), reload the shaders, and restore the conversation from disk (zero LLM calls) so a
        per-turn process keeps full state. The caller must also point `SHADERBOX_DATA_DIR` at the
        same prior data dir (command-line env — read at import), so the lib + integrations match.
        """
        # Create + leave-current the EGL context on THIS thread (the context owner). No
        # make_current call is needed — create_standalone_context leaves it current, and
        # Node/Canvas pick it up via moderngl.get_context(). (moderngl's stub types **kwargs
        # as a dict, so `backend=` trips pyright — an upstream stub gap.)
        ctx = moderngl.create_standalone_context(backend="egl")  # type: ignore[arg-type]

        resuming = project_dir is not None
        if project_dir is None:
            project_dir = Path(tempfile.mkdtemp(prefix="proj-", dir=_RUNS_DIR))
        nodes_dir = project_dir / "nodes"
        nodes_dir.mkdir(parents=True, exist_ok=True)
        # On resume the nodes already exist on disk; seeding only applies to a fresh project.
        if seed_templates and not resuming:
            for tid in TEMPLATE_ORDER:
                src = NODE_TEMPLATES_DIR / tid
                if src.is_dir():
                    shutil.copytree(src, nodes_dir / tid)

        # The injected imgui-coupled services (App-side in the live app). ExporterRegistry is
        # left EMPTY (publish tools precheck-fail gracefully; registering the real exporters
        # would pull imgui-window code). ShaderLibFileManager is GL/imgui-free in its
        # constructor + write path — only Notifications.push touches imgui, never called here.
        exporters = ExporterRegistry()
        notifications = Notifications()

        # The session needs a lib-files getter, the lib-files manager needs the session's
        # index getter. Build the session first (the getters aren't called during __init__),
        # then the manager (closing over the session), exposed through a mutable slot.
        slot: dict[str, ShaderLibFileManager] = {}
        session = ProjectSession(
            node_templates_dir=NODE_TEMPLATES_DIR,
            starter_template_id=STARTER_TEMPLATE_ID,
            template_order=TEMPLATE_ORDER,
            get_exporter_registry=lambda: exporters,
            get_shader_lib_files=lambda: slot["mgr"],
            # on_* UI-reaction callbacks default to no-ops — the harness has no editor/UI.
        )
        slot["mgr"] = ShaderLibFileManager(
            notifications=notifications,
            rebuild_index=session.rebuild_shader_lib_index,
            index_getter=lambda: session.shader_lib_index,
            on_paths_removed=lambda paths: None,
            on_path_renamed=lambda old, new: None,
        )

        # Load the project (paths/lib-index/nodes/app_state/integrations). The node warm-up
        # compiles run here, so the EGL context must be current — it is (created above on this
        # thread). On resume, load() restores app_state -> current_node_id from disk; only pick a
        # default when it's unset (a fresh project) so a resumed turn keeps its current node.
        session.load(project_dir)
        if session.ui_nodes and not session.current_node_id:
            session.set_current_node_id(next(iter(session.ui_nodes)))

        harness = cls(ctx, session, project_dir)
        if resuming:
            harness._restore_conversation()
        return harness

    @property
    def _copilot(self) -> CopilotSession:
        return self.session.copilot

    # ---- driving a turn ----

    def send(self, user_text: str) -> None:
        """Enqueue a user turn (spawns the worker on first call). Then call drive_until_idle."""
        print(f"\n>>> USER: {user_text}")
        self._copilot.enqueue_turn(user_text)

    def drive_until_idle(self, *, auto_approve_gates: bool = False) -> None:
        """Pump the bridge + events on this (context-owning) thread until the turn completes.

        Prints each new chat message as it lands. On an open gate (a `pending_action` message),
        either auto-approves (if `auto_approve_gates`) or STOPS and returns so the human can
        inspect + answer via `approve()` / `decline()` then call `drive_until_idle` again.
        """
        cop = self._copilot
        while True:
            cop.drain_bridge()
            cop.pump_events()
            self._print_new_messages()
            gate = self._open_gate()
            if gate is not None:
                if auto_approve_gates:
                    print(f"    [gate auto-approved: {gate.text!r}]")
                    cop.answer_gate(True)
                    continue
                print(
                    f"\n??? GATE (answer with h.approve() / h.decline()): {gate.text}"
                )
                return
            if not cop.state.in_flight:
                self._print_turn_footer()
                return
            time.sleep(0.02)  # yield to the worker thread

    def approve(self) -> None:
        self._copilot.answer_gate(True)
        print("    [approved]")

    def decline(self) -> None:
        self._copilot.answer_gate(False)
        print("    [declined]")

    # ---- rendering ----

    def render(self, node_id: str = "", *, size: int = 400) -> str:
        """Render a node to a `size`x`size` PNG; return + print the exact path.

        Forces the size (the node's default canvas is small — too small to eyeball a shape).
        `node_id` empty = the current node. Uses the REAL `render_image` capability — but that
        marshals the GL work through the bridge and BLOCKS on it, so it must run off the
        context-owning thread while THIS thread drains the bridge (the same shape as a copilot
        turn). We run it on a throwaway thread and pump until it returns.
        """
        target = node_id or self.session.current_node_id
        out: dict[str, RenderResult] = {}

        def _do() -> None:
            out["result"] = self.session.copilot_backend.render_image(
                target, size, size
            )

        worker = threading.Thread(target=_do, name="dogfood-render", daemon=True)
        worker.start()
        while worker.is_alive():
            self.session.copilot.drain_bridge()
            time.sleep(0.01)
        worker.join()

        result = out.get("result")
        if result is None or not result.ok:
            err = result.error if result is not None else "no result"
            print(f"    [render FAILED: {err}]")
            return ""
        print(f"    [rendered {result.width}x{result.height} -> {result.path}]")
        self._last_render_path = result.path
        return result.path

    def render_at(self, t: float, node_id: str = "", *, size: int = 400) -> str:
        """Tick the CPU-script engine to `t`, then render the node at that `t` to a PNG (feature
        040 determinism check). Unlike `render` (the copilot tool, fixed at the exporter's t=0),
        this advances the engine clock so a scripted uniform animates — the seam decision 4 needs.
        GL runs on THIS (context-owning) thread; no bridge marshalling (no copilot worker involved).
        """
        target = node_id or self.session.current_node_id
        ui_node = self.session.ui_nodes.get(target)
        if ui_node is None:
            print(f"    [render_at FAILED: no node '{target}']")
            return ""
        node = ui_node.node
        node.canvas.set_size((size, size))
        self.session.tick([target], t, 1.0 / 60.0, 0)
        node.render(u_time=t)
        out_path = self.session.paths.renders_dir / f"{target}_t{t:.3f}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        texture_to_pil(node.canvas.texture).save(out_path)
        print(f"    [rendered {target} @t={t:.3f} -> {out_path}]")
        self._last_render_path = str(out_path)
        return str(out_path)

    def export_at(self, t: float, node_id: str = "", *, size: int = 400) -> str:
        """Render a node at `t` through the EXPORT-ISOLATION seam (feature 041): entering the node's
        `export_isolation()` (the same factory Node.render_media enters) swaps on_pre_render to a FRESH
        per-export behavior set, so a stateful script starts from a clean __init__ regardless of how
        long the live preview ran. Unlike `render_at` (the LIVE tick path), this proves export-state
        isolation. GL on THIS (context-owning) thread."""
        target = node_id or self.session.current_node_id
        ui_node = self.session.ui_nodes.get(target)
        if ui_node is None:
            print(f"    [export_at FAILED: no node '{target}']")
            return ""
        node = ui_node.node
        node.canvas.set_size((size, size))
        with node.export_isolation():
            if node.on_pre_render is not None:
                node.on_pre_render(t, 1.0 / 60.0, 0)
            node.render(u_time=t)
        out_path = self.session.paths.renders_dir / f"{target}_export_t{t:.3f}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        texture_to_pil(node.canvas.texture).save(out_path)
        print(f"    [exported {target} @t={t:.3f} -> {out_path}]")
        self._last_render_path = str(out_path)
        return str(out_path)

    def _latest_render_on_disk(self) -> str:
        # The truthful render pointer: harness renders set _last_render_path, but
        # AGENT-initiated render_image never did (and a bridge-timeout could lose it
        # while the file still landed) — so report the newest file in renders/ (033).
        renders = self.session.paths.renders_dir
        try:
            latest = max(
                (p for p in renders.iterdir() if p.is_file()),
                key=lambda p: p.stat().st_mtime,
                default=None,
            )
        except OSError:
            latest = None
        if latest is not None:
            return str(latest)
        return self._last_render_path

    # ---- inspection ----

    def nodes(self) -> dict[str, str]:
        """node_id -> display name, for picking a target."""
        return {
            nid: ui_node.ui_state.ui_name
            for nid, ui_node in self.session.ui_nodes.items()
        }

    @property
    def trace_path(self) -> Path:
        """The full-fidelity copilot transcript for this session (system prompt + context +
        tools + per-iteration usage/tokens/cost) — the anchor for the dogfood report."""
        return self._copilot.trace._path

    @property
    def session_cost_usd(self) -> float:
        return self._copilot.state.session_cost_usd

    def release(self) -> None:
        # ProjectSession has no release() (App owns lifecycle in the live app); tear down the
        # copilot worker + bridge directly, then the GL context.
        try:
            self._copilot.release()
        finally:
            self._ctx.release()

    # ---- interactive (feature 027): persist + structured turn-result ----

    def dump(self, path: Path) -> dict[str, object]:
        """Persist the conversation, then write a structured JSON turn-result to `path`.

        Persisting lets the NEXT per-turn process resume via `create(project_dir=...)`. The JSON
        is built from structured state (NOT scraped stdout) on its OWN cursor, so it reports only
        the messages new since the last dump even though `drive_until_idle` already advanced the
        print cursor. `project_dir` / `data_dir` echo the two stable paths the next turn reuses.
        """
        cop = self._copilot
        cop.save_conversation(self.session.paths.copilot_conversation_path)
        # Persist app_state too, so a switch_node'd current node survives the next resume (load()
        # restores it; without this the resume falls back to the oldest node).
        self.session.app_state.save(self.session.paths.app_state_file)
        # Persist every node (uniform VALUES live in node.json, written only on save) — without
        # this each per-turn process loses the previous turn's set_uniform values, forcing the
        # agent to re-set them and burn its step budget (033; observed exp-1 turn 3).
        for ui_node in self.session.ui_nodes.values():
            self.session.save_ui_node(ui_node)
        msgs = cop.state.messages
        new = [
            {"role": m.role, "text": (m.text or "").strip()}
            for m in msgs[self._dumped_msg_count :]
            if m.role != "pending_action" and (m.text or "").strip()
        ]
        self._dumped_msg_count = len(msgs)
        stats = cop.state.last_turn
        payload: dict[str, object] = {
            "new_messages": new,
            "assistant_text": next(
                (m["text"] for m in reversed(new) if m["role"] == "assistant"), ""
            ),
            "open_gate": self._open_gate_payload(),
            "last_turn": (
                {
                    "context_tokens": stats.context_tokens,
                    "reply_tokens": stats.reply_tokens,
                    "cost_usd": stats.cost_usd,
                }
                if stats is not None
                else None
            ),
            "session_cost_usd": cop.state.session_cost_usd,
            "last_render_path": self._latest_render_on_disk(),
            "trace_path": str(self.trace_path),
            "project_dir": str(self.project_dir),
            "data_dir": str(_DATA_DIR),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"    [dumped turn-result -> {path}]")
        return payload

    def clear_context(self) -> None:
        """Wipe the conversation — a FRESH agent on the SAME project (the context-wipe technique).

        Archives + resets the chat (via the engine seam `ProjectSession.clear_conversation`), so the
        copilot resumes with ZERO memory of prior turns; only the nodes on disk remain. The next turn
        forces real tool-use (read_shader / grep) because nothing is in history. Resets both message
        cursors since the chat is now empty.
        """
        self.session.clear_conversation()
        self._seen_msg_count = len(self._copilot.state.messages)
        self._dumped_msg_count = len(self._copilot.state.messages)

    def reload(self) -> None:
        """Persist then re-load the conversation in-process — simulates an App restart.

        The literal composition `create(project_dir=...)` uses for resume, exposed for a
        single-process REPL persistence scenario. Must be idle (a mid-turn reload strands the
        worker). `trace_path` CHANGES after this (reset_conversation rotates the trace) — re-read
        it, never cache it.
        """
        cop = self._copilot
        if cop.state.in_flight:
            raise RuntimeError("reload() while a turn is in flight")
        cop.save_conversation(self.session.paths.copilot_conversation_path)
        self.session.app_state.save(self.session.paths.app_state_file)
        cop.reset_conversation()
        self._restore_conversation()

    # ---- internals ----

    def _print_new_messages(self) -> None:
        msgs = self._copilot.state.messages
        for msg in msgs[self._seen_msg_count :]:
            if msg.role == "pending_action":
                continue  # printed by the gate handler
            text = (msg.text or "").strip()
            if text:
                print(f"    [{msg.role}] {text}")
        self._seen_msg_count = len(msgs)

    def _open_gate(self) -> Message | None:
        for msg in self._copilot.state.messages:
            if msg.role == "pending_action" and not msg.resolved:
                return msg
        return None

    def _open_gate_payload(self) -> dict[str, str] | None:
        gate = self._open_gate()
        if gate is None:
            return None
        return {"text": gate.text, "kind": gate.gate_kind.value}

    def _restore_conversation(self) -> None:
        # Restore the persisted conversation onto a quiescent session (zero LLM calls): the
        # NL-only history + chat messages + cost. Both message cursors count the restored chat as
        # already-seen so the next drive/dump reports only NEW messages.
        cop = self._copilot
        store = ConversationStore.load_and_migrate(
            self.session.paths.copilot_conversation_path
        )
        cop.load_conversation(store)
        # A gate dumped mid-turn persists an unresolved pending_action, but no worker is parked on
        # it after a resume (the gated turn died with its process) — mark it resolved so it doesn't
        # read as a live "stuck" gate that drive_until_idle returns on forever.
        for msg in cop.state.messages:
            if msg.role == "pending_action" and not msg.resolved:
                msg.resolved = True
        self._seen_msg_count = len(cop.state.messages)
        self._dumped_msg_count = len(cop.state.messages)

    def _print_turn_footer(self) -> None:
        stats = self._copilot.state.last_turn
        if stats is not None:
            print(
                f"    [turn done · context={stats.context_tokens}tok "
                f"reply={stats.reply_tokens}tok cost=${stats.cost_usd:.5f}]"
            )
