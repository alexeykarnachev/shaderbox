"""The copilot's probe renders the document at ITS OWN size and downscales, so `u_resolution`
inside the output pass is the document's, not the probe's. Before this the output pass was drawn
straight into a 64px probe canvas while its inputs stayed native, and an iterated output pass
(a cascade reading its own previous run) sampled off-grid on its last run.

Falsifier: a shader that writes `u_resolution.x` into red reads the native width; a probe-sized
canvas would read ~64."""

import re
from typing import Any

_SHADER = """#version 460 core
in vec2 vs_uv;
out vec4 fs_color;
uniform vec2 u_resolution;
void main() { fs_color = vec4(u_resolution.x / 2048.0, 0.0, 0.0, 1.0); }
"""


def test_probe_facts_carry_the_native_resolution(app: Any) -> None:
    backend = app.copilot_backend
    result = backend.apply_full_rewrite(_SHADER, "")
    assert result.errors == [], result.errors
    document = app.ui_documents[app.current_document_id].document
    native_w = document.render_pass.canvas.texture.size[0]
    facts = backend._render_facts_for(document, t=0.0)
    # A uniform fill reads as the FLAT line, which names its one color.
    m = re.search(r"rgba\((\d+),(\d+),(\d+),\d+\)", facts)
    assert m, facts
    red = int(m.group(1))
    expected = round(255 * native_w / 2048)
    assert abs(red - expected) <= 3, (red, expected, facts)
    # A probe-sized render would have read the 64px probe width instead.
    assert red > round(255 * 64 / 2048) + 10
    # The probe left the document's own canvas at its size.
    assert document.render_pass.canvas.texture.size[0] == native_w
