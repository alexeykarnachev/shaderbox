"""Feature 050: the probe-render clock default (t=0), the probe_render read tool, and the
cause-aware forced turn-end nudge. GL-free - the render is stubbed; we assert the CLOCK and the
tool/nudge wiring, not pixels."""

import inspect
import types

from shaderbox.copilot import backend as backend_mod
from shaderbox.copilot.agent import _final_reply_nudge
from shaderbox.copilot.backend import CopilotBackend
from shaderbox.copilot.config import COPILOT_CONFIG
from shaderbox.copilot.tools.inspect import inspect_tools
from shaderbox.copilot.tools.registry import build_registry
from tests._caps import minimal_caps


class _FakeCanvas:
    def __init__(self) -> None:
        self.texture = types.SimpleNamespace(
            size=(64, 64), read=lambda: b"\x00" * (64 * 64 * 4)
        )

    def set_size(self, _size: tuple[int, int]) -> None:
        pass


def _facts_for(t: float | None) -> str:
    # Run the REAL _render_facts_for bound to a stub backend: record the u_time the probe renders
    # at, and feed render_facts a stand-in so the stamp logic runs without GL. Returns the stamp.
    rendered_at: list[float] = []
    node = types.SimpleNamespace(
        canvas=types.SimpleNamespace(texture=types.SimpleNamespace(size=(64, 64))),
        render=lambda u_time, canvas: rendered_at.append(u_time),
    )
    stub = types.SimpleNamespace(_probe_canvas=_FakeCanvas())
    fn = CopilotBackend._render_facts_for.__get__(stub)
    out = fn(node) if t is None else fn(node, t=t)
    return f"rendered_at={rendered_at[0]}|{out}"


def test_auto_probe_default_clock_is_literal_zero() -> None:
    # The edit auto-probe (no t passed) renders at 0.0 - the export clock - NOT a wall-clock.
    # A behavioral assert can't tell the fix from the bug here: headless glfw.get_time() ALSO
    # returns 0.0, so the old `glfw.get_time() if t is None` would pass the same. Assert the
    # SOURCE instead: the default is the literal 0.0 AND no glfw call reintroduces wall-clock.
    default = (
        inspect.signature(CopilotBackend._render_facts_for).parameters["t"].default
    )
    assert default == 0.0
    src = inspect.getsource(CopilotBackend._render_facts_for)
    assert (
        "glfw" not in src
    )  # a wall-clock reintroduction would break this unconditionally
    assert not hasattr(backend_mod, "glfw")  # the module-level glfw import is gone
    # And the behavioral side still holds: no t -> renders at 0.0.
    assert _facts_for(None).startswith("rendered_at=0.0")


def test_explicit_probe_t_survives_the_default_flip() -> None:
    # A caller passing its own t (the script probe, probe_render) still renders at THAT t after the
    # default became 0.0. Falsifier: the flip clobbers the explicit path.
    assert _facts_for(2.5).startswith("rendered_at=2.5")


def test_probe_render_tool_is_ungated_and_non_mutating() -> None:
    # The aimable look is a FREE read - never gated, never mutating (vs render_image). Falsifier:
    # routed through the gated render path.
    (defn,) = inspect_tools(minimal_caps())
    assert defn.name == "probe_render"
    assert defn.mutating is False
    assert defn.gate_policy.name == "NONE"


def test_probe_render_in_registry_and_reaches_capability() -> None:
    calls: list[tuple[str, float]] = []
    caps = minimal_caps(
        probe_render=lambda n, t: calls.append((n, t)) or "render@t=2.5s: ink 5%"
    )
    reg = build_registry(caps)
    ok, msg, _ = reg.execute("probe_render", {"node": "n1", "t": 2.5}, "")
    assert ok and "render@t=2.5s" in msg
    assert calls == [("n1", 2.5)]


def test_forced_turn_end_nudge_names_engine_cause_and_denies_user_pause() -> None:
    # Feature 050: a forced turn-end must tell the model the ENGINE cause so it owns the stop, not
    # crediting the user with a pause. Falsifier: the cause string / "not a user pause" disclaimer
    # is absent (the generic nudge that produced "you're right to pause now").
    nudge = _final_reply_nudge("You reached the per-turn limit of 16 tool-call steps")
    assert "16 tool-call steps" in nudge
    assert "not a pause the user asked for" in nudge.lower()
    assert nudge.isascii()  # engine-injected chat text stays ASCII


def test_clean_streak_config_defaults_sane() -> None:
    # The hard stop must leave room for the soft advisory to work first.
    assert COPILOT_CONFIG.clean_edit_hard_streak > COPILOT_CONFIG.clean_edit_soft_streak
