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

from shaderbox.constants import DOCUMENT_EXAMPLES_DIR, STARTER_EXAMPLE_ID
from shaderbox.copilot.backend import CopilotBackend
from shaderbox.copilot.capabilities import MediaBindResult
from shaderbox.copilot.gate import (
    GateChannel,
    GateKind,
    GateRequest,
    GateResponse,
)
from shaderbox.copilot.tools.registry import build_registry
from shaderbox.document import Document
from shaderbox.media import MediaWithTexture
from shaderbox.pass_graph import AutoSource
from shaderbox.ui_models import UIDocument
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
            GateRequest(
                kind=GateKind.FILE, prompt="p", document_id="n7", uniform="u_image"
            )
        )

    t = threading.Thread(target=worker)
    t.start()
    req = _wait_for_file_pending(gate)
    assert req is not None and req.document_id == "n7" and req.uniform == "u_image"
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


def test_file_gate_active_flips_on_cancel() -> None:
    # The UI poll gates the bind on file_gate_active(): while a worker is blocked it's True; a turn
    # Stop (cancel_all) flips it False so a dialog still open is abandoned (no post-cancel phantom bind).
    gate = GateChannel()
    assert not gate.file_gate_active()  # idle
    threading.Thread(
        target=lambda: gate.ask_file(GateRequest(kind=GateKind.FILE, prompt="f")),
        daemon=True,
    ).start()
    assert _wait_for_file_pending(gate) is not None
    assert gate.file_gate_active()  # worker parked, dialog would be live
    gate.cancel_all(reusable=True)
    assert not gate.file_gate_active()  # cancelled -> poll abandons the pick, no bind


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
    document = Document(gl=gl)
    document.render_pass.release_program(_SAMPLER_SRC)
    document.render_pass.compile()
    document.render_pass.seed_uniform_values()
    ui = UIDocument(document=document, id="samplernode")
    ui.save(project)
    documents = {ui.id: ui}
    stub = types.SimpleNamespace(
        _bridge=types.SimpleNamespace(
            run_on_main=lambda fn, timeout=None, defer=False: fn()
        ),
        _get_ui_documents=lambda: documents,
        _copilot_resolve_document_id=lambda h: ui.id if h in ("", ui.id) else None,
        _capture_document=lambda nid: None,
        _save_ui_document=lambda un: un.save(project),
    )
    return stub, ui.id


def test_bind_picked_media_binds_and_is_path_free(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    stub, document_id = _sampler_stub(gl_ctx, tmp_path / "proj")
    # A real image under a SENTINEL directory whose name must never surface.
    secret = tmp_path / "SENTINEL_SECRET_DIR"
    secret.mkdir()
    img_path = secret / "fire.png"
    PILImage.new("RGB", (8, 8), (255, 0, 0)).save(img_path)

    outcome = CopilotBackend.bind_picked_media.__get__(stub)(
        document_id, "u_image", img_path
    )
    assert outcome.ok and outcome.basename == "fire.png"
    assert (outcome.width, outcome.height) == (8, 8)
    # The bind took: the sampler holds the media, not a source.
    assert isinstance(
        stub._get_ui_documents()[document_id].document.render_pass.uniform_values[
            "u_image"
        ],
        MediaWithTexture,
    )
    # Corollary-1: the absolute path / sentinel dir is nowhere in the result.
    assert "SENTINEL_SECRET_DIR" not in str(outcome)
    assert str(img_path) not in str(outcome)

    # And piping the outcome through the tool handler leaves the path out of the model-facing msg.
    ok, msg, payload = build_registry(
        minimal_caps(bind_media=lambda _n, _u: outcome)
    ).execute("bind_media", {"uniform": "u_image", "document": ""})
    assert ok and "SENTINEL_SECRET_DIR" not in msg
    assert payload is None or "SENTINEL_SECRET_DIR" not in str(payload)


def test_unbind_returns_the_sampler_to_undecided(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    stub, document_id = _sampler_stub(gl_ctx, tmp_path / "proj")
    img_path = tmp_path / "x.png"
    PILImage.new("RGB", (8, 8), (0, 255, 0)).save(img_path)
    CopilotBackend.bind_picked_media.__get__(stub)(document_id, "u_image", img_path)
    document = stub._get_ui_documents()[document_id].document
    assert isinstance(document.render_pass.uniform_values["u_image"], MediaWithTexture)

    res = CopilotBackend.unbind_media.__get__(stub)("", "u_image")
    assert res.ok
    assert isinstance(document.render_pass.uniform_values["u_image"], AutoSource)

    # A non-sampler / unknown uniform rejects honestly.
    assert not CopilotBackend.unbind_media.__get__(stub)("", "u_nope").ok


_IMPORTABLE_SRC = """#version 460 core
in vec2 vs_uv;
out vec4 fs_color;
void main() { fs_color = vec4(vs_uv, 0.0, 1.0); }
"""


def test_import_picked_document_creates_and_is_path_free(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    documents: dict[str, object] = {}
    current = {"id": ""}
    stub = types.SimpleNamespace(
        _starter_example_id=STARTER_EXAMPLE_ID,
        _document_examples_dir=DOCUMENT_EXAMPLES_DIR,
        _copilot_resolve_example_id=lambda _t: STARTER_EXAMPLE_ID,
        _get_ui_documents=lambda: documents,
        _save_ui_document=lambda un: un.save(tmp_path / "proj"),
        _get_active_checkpoint=lambda: None,
        _set_current_document_id=lambda nid: current.__setitem__("id", nid),
        _working_set_add=lambda nid: None,
        _copilot_short_ids=lambda: {i: i for i in documents},
        _render_facts_for=lambda _document, motion=False, cache_key="": "facts",
        _last_clean={},
    )
    stub._create_document_on_main = CopilotBackend._create_document_on_main.__get__(
        stub
    )
    secret = tmp_path / "SENTINEL_SECRET_DIR"
    secret.mkdir()
    glsl = secret / "cool.glsl"
    glsl.write_text(_IMPORTABLE_SRC)

    result = CopilotBackend.import_picked_document.__get__(stub)(glsl, False)
    assert result.ok and result.basename == "cool.glsl" and not result.errors
    assert len(documents) == 1
    assert "SENTINEL_SECRET_DIR" not in str(result)

    # The tool handler msg is path-free too.
    ok, msg, _payload = build_registry(
        minimal_caps(import_document=lambda _sw: result)
    ).execute("import_document", {"switch_to": False})
    assert ok and "SENTINEL_SECRET_DIR" not in msg and "cool.glsl" in msg


def test_unbind_then_save_removes_the_orphan_media_file(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # save() writes no file for an undecided sampler; the file a PREVIOUS bind wrote must not
    # linger (disk-cleanliness leak that also rode along duplicate_document's copytree).
    stub, document_id = _sampler_stub(gl_ctx, tmp_path / "proj")
    img_path = tmp_path / "x.png"
    PILImage.new("RGB", (8, 8), (0, 255, 0)).save(img_path)
    CopilotBackend.bind_picked_media.__get__(stub)(document_id, "u_image", img_path)
    ui = stub._get_ui_documents()[document_id]
    ui.save(tmp_path / "proj")
    media_dir = tmp_path / "proj" / document_id / "media" / "main"
    assert list(media_dir.glob("u_image.*"))

    CopilotBackend.unbind_media.__get__(stub)("", "u_image")
    ui.save(tmp_path / "proj")
    assert not list(media_dir.glob("u_image.*"))
