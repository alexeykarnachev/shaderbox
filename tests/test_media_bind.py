"""Media binding (feature 052 slice 2): the FILE-gate slot, the corollary-1 no-path-leak guarantee,
and the bind/unbind behavior. The gate tests are pure (threads, no GL); the bind tests use a
standalone context + a stub (the test_cross_project_tools pattern), so they run without glfw."""

import threading
import time
import types
from collections.abc import Iterator
from pathlib import Path

import moderngl
import pytest
from PIL import Image as PILImage

from shaderbox.constants import NODE_TEMPLATES_DIR, STARTER_TEMPLATE_ID
from shaderbox.copilot.backend import CopilotBackend
from shaderbox.copilot.capabilities import MediaBindResult
from shaderbox.copilot.gate import (
    GateChannel,
    GateKind,
    GateRequest,
    GateResponse,
)
from shaderbox.copilot.tools.registry import build_registry
from shaderbox.core import Node
from shaderbox.media import is_default_image
from shaderbox.ui_models import UINode
from tests._caps import minimal_caps

_SAMPLER_SRC = """#version 460 core
in vec2 vs_uv;
uniform sampler2D u_image;
out vec4 fs_color;
void main() { fs_color = texture(u_image, vs_uv); }
"""


# ---- the FILE-gate slot (pure: threads, no GL) ----


def _wait_for_file_pending(gate: GateChannel) -> GateRequest | None:
    for _ in range(2000):
        req = gate.take_file_pending()
        if req is not None:
            return req
        time.sleep(0.001)
    return None


def test_file_gate_slot_roundtrip() -> None:
    gate = GateChannel()
    out: dict[str, GateResponse] = {}

    def worker() -> None:
        out["r"] = gate.ask_file(
            GateRequest(kind=GateKind.FILE, prompt="p", node_id="n7", uniform="u_image")
        )

    t = threading.Thread(target=worker)
    t.start()
    req = _wait_for_file_pending(gate)
    assert req is not None and req.node_id == "n7" and req.uniform == "u_image"
    gate.answer_file(
        GateResponse(media_result=MediaBindResult(ok=True, basename="fire.png"))
    )
    t.join(timeout=2)
    assert out["r"].media_result is not None
    assert out["r"].media_result.basename == "fire.png"


def test_file_gate_cancel_wakes_worker() -> None:
    gate = GateChannel()
    out: dict[str, GateResponse] = {}

    def worker() -> None:
        out["r"] = gate.ask_file(GateRequest(kind=GateKind.FILE, prompt="p"))

    t = threading.Thread(target=worker)
    t.start()
    assert _wait_for_file_pending(gate) is not None
    gate.cancel_all(reusable=True)
    t.join(timeout=2)
    assert out["r"].cancelled and out["r"].media_result is None


def test_file_and_confirm_slots_are_independent() -> None:
    # A CONFIRM ask enqueues on _pending, a FILE ask on _file_pending — take_pending / take_file_pending
    # never steal from each other (so the App poll can't consume a CONFIRM gate).
    gate = GateChannel()
    threading.Thread(
        target=lambda: gate.ask_file(GateRequest(kind=GateKind.FILE, prompt="f")),
        daemon=True,
    ).start()
    threading.Thread(
        target=lambda: gate.ask(GateRequest(kind=GateKind.CONFIRM, prompt="c")),
        daemon=True,
    ).start()
    time.sleep(0.05)
    assert gate.take_file_pending() is not None
    assert gate.take_pending() is not None
    gate.cancel_all()


# ---- bind/unbind behavior + the corollary-1 no-path-leak guarantee (GL via stub) ----


@pytest.fixture(scope="module")
def gl_ctx() -> Iterator[moderngl.Context]:
    try:
        ctx = moderngl.create_standalone_context()
    except Exception as e:
        pytest.skip(f"no standalone GL context available: {e}")
    yield ctx
    ctx.release()


def _sampler_stub(
    gl: moderngl.Context, project: Path
) -> tuple[types.SimpleNamespace, str]:
    node = Node(gl=gl)
    node.release_program(_SAMPLER_SRC)
    node.compile()
    node.seed_uniform_values()
    ui = UINode(node=node, id="samplernode")
    ui.save(project)
    nodes = {ui.id: ui}
    stub = types.SimpleNamespace(
        _bridge=types.SimpleNamespace(
            run_on_main=lambda fn, timeout=None, defer=False: fn()
        ),
        _get_ui_nodes=lambda: nodes,
        _copilot_resolve_node_id=lambda h: ui.id if h in ("", ui.id) else None,
        _capture_node=lambda nid: None,
        _save_ui_node=lambda un: un.save(project),
    )
    return stub, ui.id


def test_bind_picked_media_binds_and_is_path_free(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    stub, node_id = _sampler_stub(gl_ctx, tmp_path / "proj")
    # A real image under a SENTINEL directory whose name must never surface.
    secret = tmp_path / "SENTINEL_SECRET_DIR"
    secret.mkdir()
    img_path = secret / "fire.png"
    PILImage.new("RGB", (8, 8), (255, 0, 0)).save(img_path)

    outcome = CopilotBackend.bind_picked_media.__get__(stub)(
        node_id, "u_image", img_path
    )
    assert outcome.ok and outcome.basename == "fire.png"
    assert (outcome.width, outcome.height) == (8, 8)
    # The bind took: the sampler no longer holds the default.
    assert not is_default_image(
        stub._get_ui_nodes()[node_id].node.uniform_values["u_image"]
    )
    # Corollary-1: the absolute path / sentinel dir is nowhere in the result.
    assert "SENTINEL_SECRET_DIR" not in str(outcome)
    assert str(img_path) not in str(outcome)

    # And piping the outcome through the tool handler leaves the path out of the model-facing msg.
    ok, msg, payload = build_registry(
        minimal_caps(bind_media=lambda _n, _u: outcome)
    ).execute("bind_media", {"uniform": "u_image", "node": ""})
    assert ok and "SENTINEL_SECRET_DIR" not in msg
    assert payload is None or "SENTINEL_SECRET_DIR" not in str(payload)


def test_unbind_resets_to_default(gl_ctx: moderngl.Context, tmp_path: Path) -> None:
    stub, node_id = _sampler_stub(gl_ctx, tmp_path / "proj")
    img_path = tmp_path / "x.png"
    PILImage.new("RGB", (8, 8), (0, 255, 0)).save(img_path)
    CopilotBackend.bind_picked_media.__get__(stub)(node_id, "u_image", img_path)
    node = stub._get_ui_nodes()[node_id].node
    assert not is_default_image(node.uniform_values["u_image"])

    res = CopilotBackend.unbind_media.__get__(stub)("", "u_image")
    assert res.ok
    assert is_default_image(node.uniform_values["u_image"])

    # A non-sampler / unknown uniform rejects honestly.
    assert not CopilotBackend.unbind_media.__get__(stub)("", "u_nope").ok


_IMPORTABLE_SRC = """#version 460 core
in vec2 vs_uv;
out vec4 fs_color;
void main() { fs_color = vec4(vs_uv, 0.0, 1.0); }
"""


def test_import_picked_node_creates_and_is_path_free(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    nodes: dict[str, object] = {}
    current = {"id": ""}
    stub = types.SimpleNamespace(
        _starter_template_id=STARTER_TEMPLATE_ID,
        _node_templates_dir=NODE_TEMPLATES_DIR,
        _copilot_resolve_template_id=lambda _t: STARTER_TEMPLATE_ID,
        _get_ui_nodes=lambda: nodes,
        _save_ui_node=lambda un: un.save(tmp_path / "proj"),
        _get_active_checkpoint=lambda: None,
        _set_current_node_id=lambda nid: current.__setitem__("id", nid),
        _working_set_add=lambda nid: None,
        _copilot_short_ids=lambda: {i: i for i in nodes},
        _render_facts_for=lambda _node: "facts",
        _last_clean={},
    )
    stub._create_node_on_main = CopilotBackend._create_node_on_main.__get__(stub)
    secret = tmp_path / "SENTINEL_SECRET_DIR"
    secret.mkdir()
    glsl = secret / "cool.glsl"
    glsl.write_text(_IMPORTABLE_SRC)

    result = CopilotBackend.import_picked_node.__get__(stub)(glsl, False)
    assert result.ok and result.basename == "cool.glsl" and not result.errors
    assert len(nodes) == 1
    assert "SENTINEL_SECRET_DIR" not in str(result)

    # The tool handler msg is path-free too.
    ok, msg, _payload = build_registry(
        minimal_caps(import_node=lambda _sw: result)
    ).execute("import_node", {"switch_to": False})
    assert ok and "SENTINEL_SECRET_DIR" not in msg and "cool.glsl" in msg
