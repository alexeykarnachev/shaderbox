"""104 D3/D8: the uniforms panel marks an instanced pass with a chip.

A real contrast pair -- one document, two passes, differing ONLY in whether the shader
declares entity fields -- rather than two separate fixtures that could differ for any
number of unrelated reasons. The fullscreen half is a silence assertion, so it carries a
second observable: the fullscreen pass's own name/row must be proven DRAWN in the same
frame, or a fixture that never reached the panel would report "no chip" identically to a
correct one.
"""

from typing import Any

from imgui_bundle import imgui

from shaderbox.shader_source import ShaderSource
from shaderbox.tabs import uniforms as uniforms_tab
from shaderbox.theme import COLOR

_INSTANCED_SOURCE = """#version 460 core
in vec2 vs_quad;
flat in vec2  pos;
flat in float radius;
out vec4 fs_color;
void main() {
    if (length(vs_quad) > 1.0) discard;
    fs_color = vec4(1.0);
}
"""


def _add_instanced_pass(app: Any, document_id: str, name: str) -> None:
    error = app.session.add_pass(document_id, name)
    assert not error, error
    document = app.ui_documents[document_id].document
    render_pass = document.passes[name]
    path = app.session.paths.pass_shader_for(document_id, name)
    path.write_text(_INSTANCED_SOURCE, encoding="utf-8")
    render_pass.source = ShaderSource.load(path)
    render_pass.compile()
    assert render_pass.program is not None, render_pass.compile_unit.error_raw
    assert render_pass.entity_fields, "fixture pass did not compile as instanced"


def _draw_list_command_count() -> int:
    # A cheap, structural "did anything draw" proof: text_chip issues an add_rect_filled
    # plus a text draw, so the command buffer grows. Reading it before/after a call is the
    # contact proof -- the alternative, asserting only that a string is absent, cannot
    # distinguish "correctly not drawn" from "never reached this code at all".
    return imgui.get_window_draw_list().vtx_buffer.size()


def test_an_instanced_pass_is_marked_and_a_fullscreen_pass_beside_it_is_not(
    app: Any,
) -> None:
    document_id = app.current_document_id
    fullscreen_name = next(iter(app.ui_documents[document_id].document.passes))
    _add_instanced_pass(app, document_id, "swarm")

    for _ in range(2):
        imgui.new_frame()
        imgui.set_next_window_size((600.0, 400.0))
        imgui.begin("rig")

        # The fullscreen pass first: prove the panel actually drew ITS row (the contact
        # proof for the "is not marked" half), then read the chip count on this pass.
        app.set_panel_pass(document_id, fullscreen_name)
        before_fullscreen = _draw_list_command_count()
        uniforms_tab._draw_instanced_badge(app, document_id)
        fullscreen_vtx_delta = _draw_list_command_count() - before_fullscreen

        # A row for the fullscreen pass so "the panel drew this frame" is demonstrated,
        # not merely claimed -- its own name really painted pixels.
        imgui.text_colored(COLOR.FG_PRIMARY, fullscreen_name)
        after_name_row = _draw_list_command_count()
        assert after_name_row > before_fullscreen + fullscreen_vtx_delta, (
            "the fullscreen pass's own row never drew -- a fixture that never reached "
            "the panel would report 'no chip' identically to a correct one"
        )

        # Now the instanced pass: the chip must add draw commands of its own.
        app.set_panel_pass(document_id, "swarm")
        before_instanced = _draw_list_command_count()
        uniforms_tab._draw_instanced_badge(app, document_id)
        instanced_vtx_delta = _draw_list_command_count() - before_instanced

        imgui.end()
        imgui.end_frame()

    assert fullscreen_vtx_delta == 0, (
        "the fullscreen pass drew badge geometry -- it declares no entity fields"
    )
    assert instanced_vtx_delta > 0, "the instanced pass drew no badge geometry"
