"""Headless dogfood harness for the copilot ENGINE (feature 026).

A HAND-DRIVEN driver: a human (Claude) imports this, constructs a real `ProjectSession`
on a standalone EGL context (no App, no glfw window), drives `session.copilot` turn by
turn against a REAL LLM, renders documents to small PNGs, and EYEBALLS them. The judge is the
human reading the printed events / the trace transcript + looking at the rendered images —
there are NO code assertions and no pass/fail. The point is dogfooding: surface where the
copilot is weak, where context wastes tokens, what's missing from context.

Built on feature 025's `ProjectSession` (the headless project + copilot core). Runs on ANY
display-less box where glfw can't open a window but a standalone EGL context reaches a GPU or
software GL — a Pi (V3D), a WSL/Linux box (Mesa), a CI runner. Not Pi-specific: all it needs is
EGL + the MESA `#version 460` overrides + `OPENROUTER_API_KEY` + network to OpenRouter.

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
    png = h.render()                            # 400x400 PNG of the current document
    # ...open png with Read, eyeball it...

Interactive one-blocking-call-per-turn (feature 027): resume a prior run + emit a structured
JSON turn-result, so a fresh process per turn keeps full state via disk:

    h = DogfoodHarness.create(project_dir=Path("scripts/dogfood/runs/proj-XXXX"))
    h.send("..."); h.drive_until_idle(); h.dump(Path("scripts/dogfood/runs/turn.json"))

The dogfooding station (feature 075) records a run from the trace-listener seam: after
`h.start_experiment(id, intent=, mode=)` (turn 1 only — a pointer file in the project dir carries it
to every resumed process), each `dump()` appends the turn, its renders and one context breakdown per
LLM request to `dogfood/runs/<id>/events.jsonl` and rebuilds the site. `note` / `end_attempt` /
`start_attempt` are the driver's other three calls.
"""

import json
import os
import shutil
import signal
import tempfile
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# All dogfood run artifacts (data dir, per-run project dirs, JSON dumps, traces, PNGs) live
# under scripts/dogfood/runs/ — one consolidated, gitignored home (feature 027).
_RUNS_DIR = Path(__file__).resolve().parent / "runs"
_RUNS_DIR.mkdir(parents=True, exist_ok=True)

# --- env MUST be set before the shaderbox imports below (paths.app_data_dir reads it at
# --- import time; the MESA overrides are read at EGL context creation). A caller-set
# --- SHADERBOX_DATA_DIR wins — the resume seam (feature 027). The mkdtemp is guarded, not a
# --- setdefault default: a default argument is evaluated EAGERLY, so a resume would still cut
# --- a stray empty runs/data-* dir it never uses. -----------------------------------------
if not os.environ.get("SHADERBOX_DATA_DIR"):
    os.environ["SHADERBOX_DATA_DIR"] = tempfile.mkdtemp(prefix="data-", dir=_RUNS_DIR)
_DATA_DIR = Path(os.environ["SHADERBOX_DATA_DIR"])
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
import numpy as np  # noqa: E402
from PIL import Image as PILImage  # noqa: E402
from PIL import ImageDraw  # noqa: E402

# glfw is never init()'d here (EGL, not a glfw window), so any glfw call the app modules make
# on import warns "not initialized". Silence the cosmetic warning. The default u_time does not
# depend on it: core.py's render path reads the process clock, and every render here passes an
# explicit t anyway.
glfw.set_error_callback(lambda code, desc: None)

from dogfood.report.build import build_site  # noqa: E402
from dogfood.report.station import StationRecorder  # noqa: E402
from shaderbox.constants import (  # noqa: E402
    DOCUMENT_EXAMPLES_DIR,
    EXAMPLE_ORDER,
    STARTER_EXAMPLE_ID,
)
from shaderbox.copilot.capabilities import RenderResult  # noqa: E402
from shaderbox.copilot.gate import GateResponse  # noqa: E402
from shaderbox.copilot.persistence import ConversationStore  # noqa: E402
from shaderbox.copilot.session import CopilotSession  # noqa: E402
from shaderbox.copilot.state import Message  # noqa: E402
from shaderbox.exporters.registry import ExporterRegistry  # noqa: E402
from shaderbox.media import texture_to_pil  # noqa: E402
from shaderbox.notifications import Notifications  # noqa: E402
from shaderbox.project_session import ProjectSession  # noqa: E402
from shaderbox.render_job import render_to  # noqa: E402
from shaderbox.render_preset import (  # noqa: E402
    FitPolicy,
    RenderPreset,
    ResolutionPolicy,
)
from shaderbox.render_shape import RenderShape  # noqa: E402
from shaderbox.scripting import EngineContext  # noqa: E402
from shaderbox.shader_lib.file_ops import ShaderLibFileManager  # noqa: E402


def _fit(canvas: tuple[int, int], size: int) -> tuple[int, int]:
    # The document's aspect kept, its longer side at `size`: a square probe would relayout an
    # aspect-corrected scene (u_aspect) and misreport it.
    w, h = canvas
    scale = size / max(w, h, 1)
    return (max(8, round(w * scale)), max(8, round(h * scale)))


def _strip_cell(texture: moderngl.Texture, t: float, size: int) -> PILImage.Image:
    backdrop = PILImage.new("RGBA", (size, size), (25, 25, 40, 255))
    cell = PILImage.alpha_composite(backdrop, texture_to_pil(texture)).convert("RGB")
    ImageDraw.Draw(cell).text((5, 4), f"t={t:g}s", fill=(255, 255, 255))
    return cell


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
        # The dogfooding station (075): None until start_experiment/start_attempt, or a resume
        # of a project whose pointer file names one. Once set, dump() records the turn.
        self.station: StationRecorder | None = None
        self._media_seen: dict[str, int] = self._media_snapshot()

    @classmethod
    def create(
        cls, project_dir: Path | None = None, *, seed_examples: bool = True
    ) -> "DogfoodHarness":
        """Build the EGL context + a real `ProjectSession` + restore the conversation if resuming.

        `project_dir=None` -> a fresh mkdtemp'd project (seeded unless `seed_examples=False`).
        `project_dir=<existing run dir>` -> RESUME: skip seeding (the documents persist from prior
        turns), reload the shaders, and restore the conversation from disk (zero LLM calls) so a
        per-turn process keeps full state. The caller must also point `SHADERBOX_DATA_DIR` at the
        same prior data dir (command-line env — read at import), so the lib + integrations match.
        """
        # Create + leave-current the EGL context on THIS thread (the context owner). No
        # make_current call is needed — create_standalone_context leaves it current, and
        # Document/Canvas pick it up via moderngl.get_context(). (moderngl's stub types **kwargs
        # as a dict, so `backend=` trips pyright — an upstream stub gap.)
        ctx = moderngl.create_standalone_context(backend="egl")  # type: ignore[arg-type]

        resuming = project_dir is not None
        if project_dir is None:
            project_dir = Path(tempfile.mkdtemp(prefix="proj-", dir=_RUNS_DIR))
        documents_dir = project_dir / "documents"
        documents_dir.mkdir(parents=True, exist_ok=True)
        # On resume the documents already exist on disk; seeding only applies to a fresh project.
        if seed_examples and not resuming:
            for tid in EXAMPLE_ORDER:
                src = DOCUMENT_EXAMPLES_DIR / tid
                if src.is_dir():
                    shutil.copytree(src, documents_dir / tid)

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
            document_examples_dir=DOCUMENT_EXAMPLES_DIR,
            starter_example_id=STARTER_EXAMPLE_ID,
            example_order=EXAMPLE_ORDER,
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

        # Load the project (paths/lib-index/documents/app_state/integrations). The document warm-up
        # compiles run here, so the EGL context must be current — it is (created above on this
        # thread). On resume, load() restores app_state -> current_document_id from disk; only pick a
        # default when it's unset (a fresh project) so a resumed turn keeps its current document.
        session.load(project_dir)
        if session.ui_documents and not session.current_document_id:
            session.set_current_document_id(next(iter(session.ui_documents)))

        harness = cls(ctx, session, project_dir)
        if resuming:
            harness._restore_conversation()
            station = StationRecorder.resume(project_dir)
            if station is not None:
                harness._attach_station(station)
        harness._install_kill_persist()
        return harness

    def _install_kill_persist(self) -> None:
        # An external kill (the command-line `timeout`, Ctrl-C) must not lose the in-flight
        # turn's conversation while its tool edits already landed on disk — that leaves the next
        # resume half-restored (documents changed, history unaware). Persist the conversation before
        # dying; document/app_state saves are skipped (GL state is not signal-safe; edits are already
        # on disk).
        def _persist_and_die(signum: int, frame: object) -> None:
            try:
                self._copilot.save_conversation(
                    self.session.paths.copilot_conversation_path
                )
                if self.station is not None:
                    self.station.flush()
                print(f"    [kill-persist: conversation saved on signal {signum}]")
            finally:
                signal.signal(signum, signal.SIG_DFL)
                signal.raise_signal(signum)

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, _persist_and_die)

    @property
    def _copilot(self) -> CopilotSession:
        return self.session.copilot

    # ---- the dogfooding station (feature 075) ----

    def start_experiment(
        self,
        experiment_id: str,
        *,
        intent: str,
        mode: str,
        criteria: Sequence[str] = (),
    ) -> StationRecorder:
        """Open a new experiment (and its attempt 1) in the station and record every turn of
        this project into it from now on. `mode` is how the driver will play: `end_to_end`,
        `babysat` or `free_run`. The pointer file in the project dir carries the choice to every
        later per-turn process, so only turn 1's command names it."""
        station = StationRecorder.start_experiment(
            self.project_dir,
            experiment_id,
            intent=intent,
            mode=mode,
            model=self._copilot.client.model,
            criteria=list(criteria),
        )
        self._attach_station(station)
        return station

    def start_attempt(self, experiment_id: str) -> StationRecorder:
        """Open the next attempt of an existing experiment on THIS (usually fresh) project; the
        commits landed since the previous attempt started are recorded as its fixes."""
        station = StationRecorder.start_attempt(
            self.project_dir, experiment_id, model=self._copilot.client.model
        )
        self._attach_station(station)
        return station

    def note(self, text: str, *, axis: str = "", turn: int | None = None) -> None:
        """A driver observation into the log: `axis` names one of the report axes (fidelity /
        motion / logic / honesty / process / code, or `verdict`), `turn` pins it to a turn."""
        self._require_station().note(text, axis=axis, turn=turn)

    def end_attempt(self, outcome: str, summary: str = "") -> None:
        self._require_station().end_attempt(outcome, summary)
        build_site()

    def _require_station(self) -> StationRecorder:
        if self.station is None:
            raise RuntimeError(
                "no station attached: start_experiment(...) or start_attempt(...) first"
            )
        return self.station

    def _attach_station(self, station: StationRecorder) -> None:
        self.station = station
        self._copilot.trace_listeners.append(station.on_trace)
        print(
            f"    [station: {station.experiment_id} attempt {station.attempt} "
            f"-> {station.attempt_page}]"
        )

    def _media_snapshot(self) -> dict[str, int]:
        renders = self.session.paths.renders_dir
        try:
            return {
                p.name: p.stat().st_mtime_ns for p in renders.iterdir() if p.is_file()
            }
        except OSError:
            return {}

    def _new_media(self) -> list[Path]:
        # Every file in renders/ that is new or rewritten since the last dump — the harness
        # helpers and the copilot's own render tools all land there.
        now = self._media_snapshot()
        renders = self.session.paths.renders_dir
        fresh = [
            renders / name
            for name, mtime in now.items()
            if self._media_seen.get(name) != mtime
        ]
        self._media_seen = now
        return sorted(fresh, key=lambda p: p.stat().st_mtime_ns)

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
            # Fire the parked deferred render op (probe_render / render_image / render_video use
            # defer=True). The real App does this post-swap in ui.py; without it a deferred render
            # never runs and the worker's run_on_main blocks until it times out (60s).
            cop.bridge.run_deferred_render()
            cop.pump_events()
            self._pump_file_gate()
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

    # ---- FILE gate (feature 052): auto-answer a bind_media/import_document pick ----

    def _pump_file_gate(self) -> None:
        """Auto-answer a mid-turn FILE gate (bind_media / import_document) with a CANNED asset — the
        headless harness has no UI file picker, so this simulates the user's pick (the real app's
        `ui.py::_pump_file_gate` does this with a native dialog). Runs on the context-owning thread
        (the load+bind / read+create are GL/FS on main), same as the app's frame-loop poll.
        """
        gate = self._copilot.gate
        req = gate.take_file_pending()
        if req is None:
            return
        backend = self.session.copilot_backend
        if req.file_action == "import_document":
            result = backend.import_picked_document(self._canned_glsl(), req.switch_to)
            gate.answer_file(GateResponse(import_result=result))
            print(
                f"    [file-gate: import_document <- canned.glsl -> "
                f"{result.document_id or result.error}]"
            )
        else:
            result = backend.bind_picked_media(
                req.document_id, req.uniform, self._canned_image()
            )
            gate.answer_file(GateResponse(media_result=result))
            tag = f"{result.width}x{result.height}" if result.ok else result.error
            print(f"    [file-gate: bind {req.uniform} <- canned.png ({tag})]")

    def _canned_image(self) -> Path:
        # A small gradient PNG in the project dir (generated once) that bind_media binds to a sampler.
        p = self.project_dir / "dogfood_canned.png"
        if not p.exists():
            n = 128
            yy, xx = np.mgrid[0:n, 0:n].astype("float32") / n
            arr = (np.dstack([xx, yy, 1.0 - xx]) * 255).astype("uint8")
            PILImage.fromarray(arr, "RGB").save(p)
        return p

    def _canned_glsl(self) -> Path:
        # A trivially-valid fragment shader on disk (generated once) that import_document creates a document from.
        p = self.project_dir / "dogfood_canned.frag.glsl"
        if not p.exists():
            p.write_text(
                "#version 460 core\nin vec2 vs_uv;\nout vec4 fs_color;\n"
                "void main() { fs_color = vec4(vs_uv, 0.5, 1.0); }\n",
                encoding="utf-8",
            )
        return p

    # ---- rendering ----

    def render(self, document_id: str = "", *, size: int = 400) -> str:
        """Render a document's static (t=0) frame to a PNG with its longer side `size` px, aspect kept
        (the driver's eyeball helper);
        return + print the exact path. `document_id` empty = the current document.

        Uses the DIRECT context-thread render (`render_at` at t=0) — GL on the owning thread, no
        bridge — so it is robust on any GL backend (the bridge-marshalled `render_image` path is
        flaky under WSL software-GL: slow first-draw vs the op timeout). The COPILOT's own gated
        `render_image` tool is exercised separately when the AGENT calls it (see /dogfood §1a) — this
        helper does not need to route through it just to give the driver a PNG to look at.
        """
        return self.render_at(0.0, document_id, size=size)

    def render_video(
        self,
        document_id: str = "",
        *,
        seconds: float = 2.0,
        fps: int = 24,
        shape: RenderShape = RenderShape.NATIVE,
    ) -> str:
        """Render a SHORT WebM (the deliverable a scripted animation produces) and return its
        path so the driver can send it to the maintainer. Uses the REAL `render_video` capability — it
        animates the CPU-script engine frame by frame from t=0 (the same path the agent's render_video
        rides), so a script-driven document moves across the clip. Off-thread + bridge drain like `render`.
        Keep `seconds` small — a video freezes the loop and large frames are slow on V3D, so a
        document with a big canvas wants a small one.
        """
        target = document_id or self.session.current_document_id
        out: dict[str, RenderResult] = {}

        def _do() -> None:
            out["result"] = self.session.copilot_backend.render_video(
                target, seconds, fps, shape
            )

        worker = threading.Thread(target=_do, name="dogfood-video", daemon=True)
        worker.start()
        while worker.is_alive():
            self.session.copilot.drain_bridge()
            time.sleep(0.01)
        worker.join()

        result = out.get("result")
        if result is None or not result.ok:
            err = result.error if result is not None else "no result"
            print(f"    [render_video FAILED: {err}]")
            return ""
        print(
            f"    [rendered {result.width}x{result.height} {result.duration:.1f}s "
            f"video -> {result.path}]"
        )
        self._last_render_path = result.path
        return result.path

    def render_video_mp4(
        self,
        document_id: str = "",
        *,
        seconds: float = 3.0,
        fps: int = 24,
        size: int = 240,
    ) -> str:
        """Render a short H.264 MP4 (for players that don't open WebM — iOS/iPad). GL on THIS
        (context-owning) thread via share_state.render_to, which keys the codec off the .mp4 suffix
        (libx264/yuv420p) and animates the CPU-script engine frame by frame from t=0 through the
        export-isolation seam (a stateful script accumulates from a clean __init__). No bridge — unlike
        the copilot render_video (which is webm-only). Keep seconds/size small on V3D.
        """
        target = document_id or self.session.current_document_id
        ui_document = self.session.ui_documents.get(target)
        if ui_document is None:
            print(f"    [render_video_mp4 FAILED: no document '{target}']")
            return ""
        document = ui_document.document
        saved_size = document.canvas_size
        document.set_canvas_size((size, size))
        # FIXED_DIMS + RENDER_AT_TARGET so (size, size) drives the output (a FREE preset leaves
        # resolution_details at 0 -> ffmpeg gets a stray `-s 0x0` and the pipe breaks).
        preset = RenderPreset(
            is_video=True,
            fps=fps,
            target_w=size,
            target_h=size,
            container=".mp4",
            resolution_policy=ResolutionPolicy.FIXED_DIMS,
            fit=FitPolicy.RENDER_AT_TARGET,
        )
        out = self.session.paths.renders_dir / f"{target}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            art = render_to(document, preset, seconds, out)
        finally:
            document.set_canvas_size(saved_size)
        if art is None:
            print("    [render_video_mp4 FAILED: render error]")
            return ""
        print(
            f"    [rendered {art.size[0]}x{art.size[1]} {art.duration:.1f}s mp4 -> {art.path}]"
        )
        self._last_render_path = str(art.path)
        return str(art.path)

    def render_at(self, t: float, document_id: str = "", *, size: int = 400) -> str:
        """Tick the CPU-script engine to `t`, then render the document at that `t` to a PNG (feature
        040 determinism check). Unlike `render` (the copilot tool, fixed at the exporter's t=0),
        this advances the engine clock so a scripted uniform animates — the seam decision 4 needs.
        GL runs on THIS (context-owning) thread; no bridge marshalling (no copilot worker involved).
        """
        target = document_id or self.session.current_document_id
        ui_document = self.session.ui_documents.get(target)
        if ui_document is None:
            print(f"    [render_at FAILED: no document '{target}']")
            return ""
        document = ui_document.document
        saved_size = document.canvas_size
        document.set_canvas_size(_fit(saved_size, size))
        try:
            self.session.tick([target], t, 1.0 / 60.0, 0)
            document.render(u_time=t)
            out_path = self.session.paths.renders_dir / f"{target}_t{t:.3f}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            texture_to_pil(document.render_pass.canvas.texture).save(out_path)
        finally:
            document.set_canvas_size(saved_size)
        print(f"    [rendered {target} @t={t:.3f} -> {out_path}]")
        self._last_render_path = str(out_path)
        return str(out_path)

    def export_at(self, t: float, document_id: str = "", *, size: int = 400) -> str:
        """Render a document at `t` through the EXPORT-ISOLATION seam (feature 041): entering the document's
        `export_isolation()` (the same factory Document.render_media enters) swaps on_pre_render to a FRESH
        per-export behavior set, so a stateful script starts from a clean __init__ regardless of how
        long the live preview ran. Unlike `render_at` (the LIVE tick path), this proves export-state
        isolation. GL on THIS (context-owning) thread."""
        target = document_id or self.session.current_document_id
        ui_document = self.session.ui_documents.get(target)
        if ui_document is None:
            print(f"    [export_at FAILED: no document '{target}']")
            return ""
        document = ui_document.document
        saved_size = document.canvas_size
        document.set_canvas_size(_fit(saved_size, size))
        try:
            with document.export_isolation():
                if document.on_pre_render is not None:
                    document.on_pre_render(t, 1.0 / 60.0, 0)
                document.render(u_time=t)
            out_path = self.session.paths.renders_dir / f"{target}_export_t{t:.3f}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            texture_to_pil(document.render_pass.canvas.texture).save(out_path)
        finally:
            document.set_canvas_size(saved_size)
        print(f"    [exported {target} @t={t:.3f} -> {out_path}]")
        self._last_render_path = str(out_path)
        return str(out_path)

    def render_strip(
        self,
        times: Sequence[float],
        document_id: str = "",
        *,
        size: int = 300,
        fps: int = 30,
    ) -> str:
        """One horizontal contact sheet of the document at each `t` — the motion-axis measurement.

        Each sample is a REPLAY, never a live tick: a scripted document gets a FRESH behavior per
        sample, stepped through frames `0..round(t*fps)` on the export-isolation seam, so a
        stateful integrator (one that reads `ctx.dt`) samples its real trajectory and two calls
        with the same times give the same sheet. A script-less document renders directly at each `t`.

        Frames alpha-composite onto (25,25,40) — deliberately NOT the eye's (40,40,40), so a
        strip is never mistaken for what the copilot saw — with a 4px gutter and a `t=` label per
        cell. The sheet lands in the project's renders dir so `dump()`'s `last_render_path` finds
        it. Both pieces of state the sampling touches are saved and restored: the document's canvas size (through `set_canvas_size`, the funnel every pass scales from -- resizing the output canvas alone leaves a multi-pass document sampling mismatched targets) and
        `document.uniform_values` (`tick_export` writes the driven uniforms into the LIVE document) — a
        later `dump()` would otherwise persist the last sample's frame into document.json.
        """
        target = document_id or self.session.current_document_id
        ui_document = self.session.ui_documents.get(target)
        if ui_document is None:
            print(f"    [render_strip FAILED: no document '{target}']")
            return ""
        if not times:
            print("    [render_strip FAILED: no sample times]")
            return ""
        document = ui_document.document
        engine = self.session.script_engine
        saved_size = document.canvas_size
        saved_values = dict(document.render_pass.uniform_values)
        dt = 1.0 / fps
        cells: list[PILImage.Image] = []
        document.set_canvas_size((size, size))
        try:
            for t in times:
                behavior = engine.fresh_behavior_for(target)
                if behavior is not None:
                    for frame in range(round(t * fps) + 1):
                        engine.tick_export(
                            target,
                            document,
                            EngineContext(t=frame * dt, dt=dt, frame=frame),
                            behavior,
                        )
                document.render(u_time=t)
                cells.append(_strip_cell(document.render_pass.canvas.texture, t, size))
        finally:
            document.set_canvas_size(saved_size)
            document.render_pass.uniform_values.clear()
            document.render_pass.uniform_values.update(saved_values)

        gutter = 4
        sheet = PILImage.new(
            "RGB",
            (len(cells) * size + (len(cells) - 1) * gutter, size),
            (0, 0, 0),
        )
        for i, cell in enumerate(cells):
            sheet.paste(cell, (i * (size + gutter), 0))
        span = f"{times[0]:g}-{times[-1]:g}" if len(times) > 1 else f"{times[0]:g}"
        out_path = self.session.paths.renders_dir / f"{target}_strip_t{span}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(out_path)
        stamps = ", ".join(f"{t:g}" for t in times)
        print(f"    [strip {target} @t=({stamps}) -> {out_path}]")
        self._last_render_path = str(out_path)
        return str(out_path)

    def script_values(
        self, times: Sequence[float], document_id: str = "", *, fps: int = 30
    ) -> list[tuple[float, dict[tuple[str, str], Any]]]:
        """The logic-axis numeric probe: the script-driven uniform VALUES at each `t`.

        A passthrough to `ScriptEngine.dry_run` — one fresh script stepped continuously through
        the export clock, every write into a throwaway sink, so the live document and engine are
        byte-identical afterwards. Returns `(t, {uniform: value})` per sample.

        Every probe failure is PRINTED (compile, runtime, per-key shape, orphan key): a broken
        probe yields empty sample dicts, which read as "the script drove nothing" — the logic axis
        must never mistake that for a measurement.
        """
        target = document_id or self.session.current_document_id
        ui_document = self.session.ui_documents.get(target)
        if ui_document is None:
            print(f"    [script_values FAILED: no document '{target}']")
            return []
        probe = self.session.script_engine.dry_run(
            target, ui_document.document, tuple(times), fps
        )
        if probe.compile_error is not None:
            print(f"    [script_values: compile error {probe.compile_error.message}]")
        if probe.runtime_error is not None:
            print(f"    [script_values: runtime error {probe.runtime_error.message}]")
        for pass_name, name, err in probe.per_key_errors:
            print(
                f"    [script_values: '{pass_name}.{name}' shape error {err.message}]"
            )
        for pass_name, name in probe.orphan_keys:
            label = f"{pass_name}.{name}" if pass_name else name
            print(f"    [script_values: '{label}' names no active uniform]")
        if not probe.driven:
            print("    [script_values: the script drove NO uniform]")
        for t, values in probe.samples:
            print(f"    [t={t:g}s {values}]")
        return probe.samples

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

    def documents(self) -> dict[str, str]:
        """document_id -> display name, for picking a target."""
        return {
            nid: ui_document.ui_state.ui_name
            for nid, ui_document in self.session.ui_documents.items()
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
        # Persist app_state too, so a switch_document'd current document survives the next resume (load()
        # restores it; without this the resume falls back to the oldest document).
        self.session.app_state.save(self.session.paths.app_state_file)
        # Persist every document (uniform VALUES live in document.json, written only on save) — without
        # this each per-turn process loses the previous turn's set_uniform values, forcing the
        # agent to re-set them and burn its step budget (033; observed exp-1 turn 3).
        for ui_document in self.session.ui_documents.values():
            self.session.save_ui_document(ui_document)
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
        if self.station is not None:
            recorded = self.station.record_turn(
                assistant_text=str(payload["assistant_text"]),
                renders=self._new_media(),
                renders_root=self.session.paths.renders_dir,
            )
            build_site()
            payload["station"] = {
                "experiment_id": self.station.experiment_id,
                "attempt": self.station.attempt,
                "turn": recorded["n"] if recorded else None,
                "page": str(self.station.attempt_page),
            }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"    [dumped turn-result -> {path}]")
        return payload

    def clear_context(self) -> None:
        """Wipe the conversation — a FRESH agent on the SAME project (the context-wipe technique).

        Archives + resets the chat (via the engine seam `ProjectSession.clear_conversation`), so the
        copilot resumes with ZERO memory of prior turns; only the documents on disk remain. The next turn
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
        store = ConversationStore.load(self.session.paths.copilot_conversation_path)
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
