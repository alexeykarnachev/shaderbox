"""Buffer formatting (078 D9, D11): the shipped formatters and the one-undo-step apply."""

import types
from typing import Any

from shaderbox.commands import SPEC_BY_ID, CommandId
from shaderbox.formatting import (
    _attach_member_access,
    format_glsl,
    format_python,
    formatter_for,
)
from shaderbox.scripting import script_stub_for

_UGLY_GLSL = (
    "void main(){vec3 c=vec3(1.0);\nif(c.x>0.5){c=c*2.0;}\ngl_FragColor=vec4(c,1.0);}\n"
)
_NEAT_GLSL = (
    "void main() {\n"
    "    vec3 c = vec3(1.0);\n"
    "    if (c.x > 0.5) {\n"
    "        c = c * 2.0;\n"
    "    }\n"
    "    gl_FragColor = vec4(c, 1.0);\n"
    "}\n"
)
_UGLY_PY = 'import math\nclass B:\n  def update(self,context):\n      return {"u_x":context.t*2,}\n'
_NEAT_PY = (
    "import math\n\n\nclass B:\n    def update(self, context):\n        return {\n"
    '            "u_x": context.t * 2,\n        }\n'
)


def test_glsl_formats_with_the_nvim_fallback_style() -> None:
    result = format_glsl(_UGLY_GLSL)
    assert result.ok
    assert result.text == _NEAT_GLSL


_HIS_LINE = (
    "void main() {\n"
    "    vec3 light = collect_light(vs_uv, u_n_rays, u_max_n_steps, band_offset, "
    "band_size).rgb;\n"
    "}\n"
)
_HIS_LINE_FORMATTED = (
    "void main() {\n"
    "    vec3 light = collect_light(\n"
    "        vs_uv, u_n_rays, u_max_n_steps, band_offset, band_size\n"
    "    ).rgb;\n"
    "}\n"
)
_FITTING_CALL = (
    "void main() {\n"
    "    vec3 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa = "
    "texture(u_sampler, vs_uv).rgb;\n"
    "}\n"
)
_CHAIN = (
    "void main() {\n"
    "    vec3 light = collect_light(vs_uv, u_n_rays, u_max_n_steps, band_offset, "
    "band_size).bar(x).rgb;\n"
    "}\n"
)
_CHAIN_FORMATTED = (
    "void main() {\n"
    "    vec3 light = collect_light(\n"
    "        vs_uv, u_n_rays, u_max_n_steps, band_offset, band_size\n"
    "    ).bar(x).rgb;\n"
    "}\n"
)


def test_the_maintainers_line_breaks_after_the_bracket_and_keeps_the_member() -> None:
    result = format_glsl(_HIS_LINE)
    assert result.ok
    assert result.text == _HIS_LINE_FORMATTED


def test_a_bare_close_bracket_takes_the_member_access() -> None:
    assert _attach_member_access("    )\n    .rgb;\n") == "    ).rgb;\n"


def test_a_call_that_fits_takes_the_member_access() -> None:
    assert (
        _attach_member_access("    x = texture(u_s, uv)\n        .rgb;\n")
        == "    x = texture(u_s, uv).rgb;\n"
    )


def test_a_chain_folds_one_link_per_line() -> None:
    assert (
        _attach_member_access("    ).bar(x)\n        .rgb;\n") == "    ).bar(x).rgb;\n"
    )


def test_a_continued_expression_is_not_a_member_access() -> None:
    unchanged = "    a = f(x)\n        + 1.0;\n"
    assert _attach_member_access(unchanged) == unchanged


def test_the_member_access_join_is_a_fixed_point_of_format_glsl() -> None:
    for source in (_HIS_LINE, _FITTING_CALL, _CHAIN):
        once = format_glsl(source)
        assert once.ok
        twice = format_glsl(once.text)
        assert twice.ok
        assert twice.text == once.text


def test_a_chain_survives_the_round_trip_through_clang_format() -> None:
    result = format_glsl(_CHAIN)
    assert result.ok
    assert result.text == _CHAIN_FORMATTED


def test_python_formats_with_ruff_at_88() -> None:
    result = format_python(_UGLY_PY)
    assert result.ok
    assert result.text == _NEAT_PY


def test_a_syntax_error_formats_nothing_and_says_why() -> None:
    result = format_python("def f(:\n")
    assert not result.ok
    assert result.text == "def f(:\n"
    assert "parse" in result.error.lower()


def test_every_tab_kind_has_a_formatter() -> None:
    for kind in ("shader", "lib", "script"):
        assert formatter_for(kind) is not None
    assert formatter_for("other") is None


def test_the_chord_is_registered_on_the_editor_scope() -> None:
    spec = SPEC_BY_ID[CommandId.FORMAT_BUFFER]
    assert spec.label == "Format"


def test_format_command_is_one_undo_step_and_keeps_the_caret_line(app: Any) -> None:
    app.ensure_shader_tab(app.current_document_id)
    assert app.active_tab is not None and app.active_tab.kind == "shader"
    session = app.get_session_for_path(app.current_editor_path)
    editor = session.editor
    lines = editor.get_text().split("\n")
    editor.set_selection((0, 0), (len(lines) - 1, len(lines[-1])))
    editor.replace_selection(_UGLY_GLSL)
    editor.set_cursor(1, 0)
    app.format_current_editor()
    assert editor.get_text() == _NEAT_GLSL
    assert editor.get_current_cursor_position().line == 1
    editor.feed("u")
    assert editor.get_text() == _UGLY_GLSL


def test_the_script_stub_is_a_fixed_point_of_the_formatter() -> None:
    # 079 D10: `Ctrl+Shift+I` on a fresh script must change nothing. The stub emitted one blank
    # line between the import block and the class where ruff wants two.
    def _u(name: str, dim: int = 1, n: int = 1) -> Any:
        return types.SimpleNamespace(
            name=name, dimension=dim, array_length=n, gl_type=0x1406, value=0.0
        )

    for uniforms_by_pass in (
        {},
        {"main": []},
        {"main": [_u("u_x"), _u("u_v", dim=3), _u("u_a", n=4)], "blur": [_u("u_r")]},
    ):
        stub = script_stub_for(uniforms_by_pass)
        result = format_python(stub)
        assert result.ok
        assert result.text == stub
