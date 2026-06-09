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
redirects the data dir to an isolated tmp (so lib edits + the written `integrations.json`
never touch the real library / creds — `paths.app_data_dir()` reads it at import time); the
MESA overrides give the V3D driver `#version 460` (read at context creation). The OpenRouter
key comes from the `OPENROUTER_API_KEY` env var (export it before running); the model from
`OPENROUTER_MODEL` (default grok-4-fast).

THREADING (load-bearing): `CopilotSession` spawns a worker thread for the turn; the worker
marshals GL ops back to the context-owning (main) thread via `bridge.run_on_main`, which
BLOCKS until the main thread drains it. So the harness drive loop MUST pump the bridge +
events from the main thread (the one that created the EGL context), exactly as `App`'s frame
loop does — a synchronous-bridge patch would run GL on the worker thread and corrupt the
context. `drive_until_idle` is that pump.

Usage (from a REPL / chat-driven loop, with OPENROUTER_API_KEY exported):

    from scripts.dogfood import DogfoodHarness
    h = DogfoodHarness.create()                 # fresh tmp project, EGL context, real copilot
    h.send("Create a shader that draws a filled white circle in the center.")
    h.drive_until_idle()                        # pump; auto-prints events + any gate
    png = h.render()                            # 400x400 PNG of the current node
    # ...open png with Read, eyeball it...
"""

import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

# --- env MUST be set before the shaderbox imports below (paths.app_data_dir reads it at
# --- import time; the MESA overrides are read at EGL context creation). ----------------
_DATA_DIR = Path(
    os.environ.setdefault(
        "SHADERBOX_DATA_DIR",
        tempfile.mkdtemp(prefix="shaderbox-dogfood-data-"),
    )
)
os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "4.6")
os.environ.setdefault("MESA_GLSL_VERSION_OVERRIDE", "460")

# Write the OpenRouter creds into the isolated integrations.json (the client reads it live).
_INTEGRATIONS = _DATA_DIR / "integrations.json"
if not _INTEGRATIONS.exists():
    _KEY = os.environ.get("OPENROUTER_API_KEY", "")
    _MODEL = os.environ.get("OPENROUTER_MODEL", "x-ai/grok-4-fast")
    _INTEGRATIONS.write_text(
        json.dumps({"copilot": {"openrouter_key": _KEY, "model": _MODEL}}),
        encoding="utf-8",
    )

import glfw  # noqa: E402
import moderngl  # noqa: E402

# core.py's render path reads glfw.get_time() for the default u_time; glfw is never init()'d
# here (we use EGL, not a glfw window), so it warns "not initialized" and returns 0.0 — which
# is exactly the static t=0 frame the dogfood wants. Silence the cosmetic warning.
glfw.set_error_callback(lambda code, desc: None)

from shaderbox.constants import RESOURCES_DIR  # noqa: E402
from shaderbox.copilot.capabilities import RenderResult  # noqa: E402
from shaderbox.copilot.session import CopilotSession  # noqa: E402
from shaderbox.copilot.state import Message  # noqa: E402
from shaderbox.exporters.registry import ExporterRegistry  # noqa: E402
from shaderbox.notifications import Notifications  # noqa: E402
from shaderbox.project_session import ProjectSession  # noqa: E402
from shaderbox.shader_lib.file_ops import ShaderLibFileManager  # noqa: E402

# Authored template order (mirror app.py's list; the first is the starter).
_TEMPLATE_ORDER = [
    "53724dbd-8efb-4c09-8c7d-28d626a066e7",  # UV Mango (the starter)
    "73ea2431-13f6-41e4-b923-04d846b678b0",  # Media Input
    "f90f5ff9-29c6-4bcf-aee7-090f20542353",  # Text Rendering
]
_STARTER_TEMPLATE_ID = _TEMPLATE_ORDER[0]


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
        self._seen_msg_count = 0  # for incremental event printing

    @classmethod
    def create(cls, *, seed_templates: bool = True) -> "DogfoodHarness":
        node_templates_dir = RESOURCES_DIR / "node_templates"

        # Create + leave-current the EGL context on THIS thread (the context owner). No
        # make_current call is needed — create_standalone_context leaves it current, and
        # Node/Canvas pick it up via moderngl.get_context(). (moderngl's stub types **kwargs
        # as a dict, so `backend=` trips pyright — an upstream stub gap.)
        ctx = moderngl.create_standalone_context(backend="egl")  # type: ignore[arg-type]

        # A fresh throwaway project. Seed the shipped templates as nodes so the agent has
        # something to read/edit (seed_templates=False leaves it empty -> create_node only).
        project_dir = Path(tempfile.mkdtemp(prefix="shaderbox-dogfood-proj-"))
        nodes_dir = project_dir / "nodes"
        nodes_dir.mkdir(parents=True)
        if seed_templates:
            for tid in _TEMPLATE_ORDER:
                src = node_templates_dir / tid
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
            node_templates_dir=node_templates_dir,
            starter_template_id=_STARTER_TEMPLATE_ID,
            template_order=_TEMPLATE_ORDER,
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
        # thread).
        session.load(project_dir)
        if session.ui_nodes:
            session.set_current_node_id(next(iter(session.ui_nodes)))

        return cls(ctx, session, project_dir)

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
        return result.path

    # ---- inspection ----

    def nodes(self) -> dict[str, str]:
        """node_id -> display name, for picking a target."""
        return {
            nid: ui_node.ui_state.ui_name
            for nid, ui_node in self.session.ui_nodes.items()
        }

    def release(self) -> None:
        # ProjectSession has no release() (App owns lifecycle in the live app); tear down the
        # copilot worker + bridge directly, then the GL context.
        try:
            self._copilot.release()
        finally:
            self._ctx.release()

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

    def _print_turn_footer(self) -> None:
        stats = self._copilot.state.last_turn
        if stats is not None:
            print(
                f"    [turn done · context={stats.context_tokens}tok "
                f"reply={stats.reply_tokens}tok cost=${stats.cost_usd:.5f}]"
            )
