"""Feature 053 copilot vision: the model-capability classifier, the /models fetch parsing, and the
probe_render vision cache (frame + intent keyed). All deterministic + headless — the badge RENDERING
is a live maintainer check, but every classifier/wire/cache decision is gated here."""

import types
from collections.abc import Iterator
from typing import Any

import moderngl
import pytest

from shaderbox.constants import NODE_TEMPLATES_DIR, STARTER_TEMPLATE_ID
from shaderbox.copilot import backend as backend_mod
from shaderbox.copilot.backend import CopilotBackend
from shaderbox.copilot.config import COPILOT_CONFIG, CopilotConfig
from shaderbox.copilot.llm import openrouter as openrouter_mod
from shaderbox.copilot.llm.openrouter import fetch_model_image_support
from shaderbox.copilot.vision_probe import (
    VisionModelProbe,
    VisionProbeStatus,
    VisionVerdict,
)
from shaderbox.exporters.integrations import CopilotIntegration, IntegrationsStore
from shaderbox.ui_models import load_node_from_dir


def _ready_probe(support: dict[str, bool]) -> VisionModelProbe:
    probe = VisionModelProbe()
    probe.support = support
    probe.status = VisionProbeStatus.READY
    probe.checked_key = "sk-test"
    return probe


def test_verdict_distinguishes_supported_textonly_unknown() -> None:
    probe = _ready_probe({"vendor/vision": True, "vendor/text": False})
    assert probe.verdict("vendor/vision") == VisionVerdict.SUPPORTED
    assert probe.verdict("vendor/text") == VisionVerdict.UNSUPPORTED
    assert (
        probe.verdict("vendor/typo") == VisionVerdict.UNKNOWN
    )  # absent from the catalogue


def test_verdict_checking_and_unverified_states() -> None:
    checking = VisionModelProbe()
    checking.status = VisionProbeStatus.CHECKING
    assert checking.verdict("anything") == VisionVerdict.CHECKING
    # A transient fetch failure is NOT a hard "no" — it must read as couldn't-verify, not unsupported.
    errored = VisionModelProbe()
    errored.status = VisionProbeStatus.ERROR
    assert errored.verdict("vendor/vision") == VisionVerdict.UNVERIFIED
    assert (
        VisionModelProbe().verdict("x") == VisionVerdict.UNVERIFIED
    )  # IDLE (never checked)


def test_fetch_parses_image_support(monkeypatch: Any) -> None:
    payload = {
        "data": [
            {"id": "a/vision", "architecture": {"input_modalities": ["text", "image"]}},
            {"id": "a/text", "architecture": {"input_modalities": ["text"]}},
            {"id": "a/noarch"},
        ]
    }

    def _fake_get(url: str, headers: dict, timeout: Any) -> Any:
        return types.SimpleNamespace(
            raise_for_status=lambda: None, json=lambda: payload
        )

    monkeypatch.setattr(openrouter_mod.httpx, "get", _fake_get)
    support = fetch_model_image_support("sk-test")
    assert support == {"a/vision": True, "a/text": False, "a/noarch": False}


def test_fetch_returns_none_on_empty_key_and_on_failure(monkeypatch: Any) -> None:
    assert fetch_model_image_support("") is None  # no billed/anon call

    def _boom(url: str, headers: dict, timeout: Any) -> Any:
        raise RuntimeError("offline")

    monkeypatch.setattr(openrouter_mod.httpx, "get", _boom)
    assert (
        fetch_model_image_support("sk-test") is None
    )  # transient != a negative result


def _probe_stub() -> tuple[types.SimpleNamespace, dict[str, int]]:
    # Bind CopilotBackend.probe_render onto a namespace carrying only what it touches (the node-op
    # tests use the same __get__-onto-a-stub trick to avoid the full constructor).
    counters = {"describe": 0, "png": 0}
    ui_node = types.SimpleNamespace(node=object(), id="n1")
    frame = {"hash": 111, "strip": False}

    def _describe(png: bytes, hint: str, is_strip: bool) -> str:
        counters["describe"] += 1
        return f"obs#{counters['describe']} hint={hint!r} strip={is_strip}"

    def _probe_png(node: object, t: float) -> tuple[bytes, bool, int]:
        counters["png"] += 1
        return b"PNG", frame["strip"], frame["hash"]

    stub = types.SimpleNamespace(
        _bridge=types.SimpleNamespace(
            run_on_main=lambda fn, timeout=None, defer=False: fn()
        ),
        _copilot_render_target=lambda node: ui_node,
        _render_facts_for=lambda node, t=0.0: "facts@t",
        _vision_enabled=lambda: True,
        _probe_png=_probe_png,
        _describe_image=_describe,
        _vision_cache={},
        _frame=frame,
    )
    return stub, counters


def test_vision_cache_hits_identical_and_misses_new_intent() -> None:
    stub, counters = _probe_stub()
    run = CopilotBackend.probe_render.__get__(stub)

    out1 = run("n1", 0.0, "reads HELLO")
    assert (
        "obs#1" in out1 and "hint='reads HELLO'" in out1 and counters["describe"] == 1
    )
    # Identical frame + same intent -> cache hit, no second vision call.
    out2 = run("n1", 0.0, "reads HELLO")
    assert out2 == out1 and counters["describe"] == 1
    # SAME frame, DIFFERENT look_for -> MUST miss (the read depends on intent).
    run("n1", 0.0, "is it upside down?")
    assert counters["describe"] == 2
    # A changed frame (new hash) with the first intent again -> miss.
    stub._frame["hash"] = 222
    run("n1", 0.0, "reads HELLO")
    assert counters["describe"] == 3


def test_probe_render_skips_vision_when_disabled() -> None:
    stub, counters = _probe_stub()
    stub._vision_enabled = lambda: False
    out = CopilotBackend.probe_render.__get__(stub)("n1", 0.0, "reads HELLO")
    assert "facts@t" in out and "visual" not in out
    assert counters["png"] == 0 and counters["describe"] == 0


def test_probe_png_returns_strip_flag_to_the_eye() -> None:
    # is_strip flows from _probe_png through to describe_image (the time-strip note).
    stub, _ = _probe_stub()
    stub._frame["strip"] = True
    out = CopilotBackend.probe_render.__get__(stub)("n1", 0.0, "does it pulse?")
    assert "strip=True" in out


def test_integrations_roundtrip_and_old_config_defaults() -> None:
    store = IntegrationsStore()
    store.copilot.vision_enabled = False
    store.copilot.vision_model = "vendor/custom-vision"
    reloaded = IntegrationsStore(**store.model_dump())
    assert reloaded.copilot.vision_enabled is False
    assert reloaded.copilot.vision_model == "vendor/custom-vision"
    # An integrations.json written before 053 lacks the two keys -> pydantic defaults, no store-nuke.
    old = {"copilot": {"openrouter_key": "sk-keep", "model": "vendor/code"}}
    loaded = IntegrationsStore(**old)
    assert loaded.copilot.openrouter_key == "sk-keep"
    assert loaded.copilot.vision_enabled == CopilotConfig.copilot_vision_enabled
    assert loaded.copilot.vision_model == CopilotConfig.copilot_vision_model


def test_backend_module_imports() -> None:
    # Guard the PIL import added for the contact-sheet tiling.
    assert backend_mod.PILImage is not None
    assert CopilotIntegration is not None


@pytest.fixture(scope="module")
def gl_ctx() -> Iterator[moderngl.Context]:
    try:
        ctx = moderngl.create_standalone_context()
    except Exception as e:
        pytest.skip(f"no standalone GL context available: {e}")
    yield ctx
    ctx.release()


def test_probe_png_renders_forward_in_time(
    gl_ctx: moderngl.Context, monkeypatch: Any
) -> None:
    # Regression (post-impl review): the contact sheet must render FORWARD from the AIMED t
    # (t, t+dt, t+2dt), never from an absolute constant. A probe aimed at t=2.5 once produced a
    # reversed 2.5, 1.5, 0.5 strip that describe_image labelled "early -> late". Real GL: capture the
    # u_time of every render and assert it strides forward from t. The stub tests can't see this.
    node = load_node_from_dir(NODE_TEMPLATES_DIR / STARTER_TEMPLATE_ID)
    node.node.compile()
    times: list[float] = []
    orig = node.node.render

    def _spy(u_time: float | None = None, canvas: object = None) -> None:
        times.append(float(u_time or 0.0))
        orig(u_time=u_time, canvas=canvas)

    monkeypatch.setattr(node.node, "render", _spy)
    stub = types.SimpleNamespace(_vision_canvas=None)
    t = 2.5
    dt = COPILOT_CONFIG.render_facts_motion_t
    CopilotBackend._probe_png.__get__(stub)(node.node, t)
    assert len(times) >= 2  # frame0 + frame1 at minimum (a 3rd only if it animates)
    assert times[0] == t
    assert times[1] == t + dt
    assert times == sorted(times) and len(set(times)) == len(times)  # strictly forward
