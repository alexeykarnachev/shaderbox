"""Reset restarts a document whole: its clock, its script instance, its histories (071 W-C).

Every check here reads the CONSUMER: the pixel a live render produces from `u_time`, the value
the script wrote from `context.t` and from its own `self` state, the command table's callback. The
funnel is `ProjectSession.reset_document`; nothing else knows what a reset consists of.
"""

from typing import Any

import numpy as np

from shaderbox.commands import CommandId
from shaderbox.core import Canvas, process_time
from shaderbox.paths import DOCUMENT_SCRIPT_BASENAME

# Red is the live clock, clamped: a document whose origin lies seconds in the past renders 255,
# one reset an instant ago renders ~0.
_CLOCK_SRC = """#version 460 core
in vec2 vs_uv;
out vec4 fs_color;
uniform float u_time;
void main() { fs_color = vec4(clamp(u_time, 0.0, 1.0), 0.0, 0.0, 1.0); }
"""

_SCRIPT_SRC = """#version 460 core
in vec2 vs_uv;
out vec4 fs_color;
uniform float u_n;
uniform float u_t;
void main() { fs_color = vec4(u_n, u_t, 0.0, 1.0); }
"""

# `self.n` counts ticks since __init__; `u_t` is context.t as the script saw it.
_SCRIPT = (
    "class Behavior(ScriptBehavior):\n"
    "    def __init__(self):\n"
    "        self.n = 0\n"
    "    def update(self, context):\n"
    "        self.n += 1\n"
    "        return {'u_n': float(self.n), 'u_t': float(context.t)}\n"
)


def _install(app: Any, src: str) -> Any:
    document = app.ui_documents[app.current_document_id].document
    document.render_pass.release_program(src)
    document.render_pass.source.path.write_text(src)
    document.render_pass.compile()
    assert document.render_pass.compile_unit.errors == []
    return document


def _red(document: Any) -> int:
    document.render()
    pixels = np.frombuffer(document.render_pass.canvas.texture.read(), dtype=np.uint8)
    return int(pixels[0])


def test_a_live_render_counts_from_the_document_clock(app: Any) -> None:
    document = _install(app, _CLOCK_SRC)
    document.time_origin = process_time() - 5.0
    assert _red(document) == 255  # the control: this document's clock reads ~5 s

    document.reset()
    assert _red(document) <= 3, "a live render right after reset must see u_time near 0"


def test_an_explicit_u_time_ignores_the_document_clock(app: Any) -> None:
    # Export and the probe pass their own clock; a live reset must not bend it. Falsifier: resolve
    # the origin for every render, not only for the live (None) case.
    document = _install(app, _CLOCK_SRC)
    document.reset()
    document.render(u_time=5.0)
    pixels = np.frombuffer(document.render_pass.canvas.texture.read(), dtype=np.uint8)
    assert int(pixels[0]) == 255


def test_reset_restarts_the_script_and_its_clock(app: Any) -> None:
    document_id = app.current_document_id
    document = _install(app, _SCRIPT_SRC)
    scripts_dir = app.session.paths.scripts_dir_for(document_id)
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / DOCUMENT_SCRIPT_BASENAME).write_text(_SCRIPT)
    app.session.reload_scripts()

    document.time_origin = 0.0  # the control: the script sees the clock it was handed
    for frame in range(5):
        app.session.tick([document_id], 10.0 + frame / 60, 1 / 60, frame)
    values = document.render_pass.uniform_values
    assert values["u_n"] == 5.0
    assert values["u_t"] >= 10.0

    app.session.reset_document(document_id)
    app.session.tick([document_id], process_time(), 1 / 60, 5)
    values = document.render_pass.uniform_values
    assert values["u_n"] == 1.0, "the script instance was not re-created"
    assert values["u_t"] < 0.5, (
        "context.t was not re-based on the document's time origin"
    )


def test_reset_drops_the_feedback_histories(app: Any) -> None:
    # The funnel must call the history reset, not only restart the clock. Falsifier: drop the
    # `reset_feedback()` call from `Document.reset`.
    document = app.ui_documents[app.current_document_id].document
    document._feedback["phantom"] = Canvas(size=(4, 4))  # released by the reset
    before = document.time_origin
    app.session.reset_document(app.current_document_id)
    assert document._feedback == {}
    assert document.time_origin > before


def test_the_command_reaches_the_funnel(app: Any, monkeypatch: Any) -> None:
    seen: list[str] = []
    monkeypatch.setattr(
        app.session, "reset_document", lambda document_id: seen.append(document_id)
    )
    app.command_callbacks[CommandId.RESET_DOCUMENT]()
    assert seen == [app.current_document_id]
