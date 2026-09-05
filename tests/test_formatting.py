"""Buffer formatting (078 D9, D11): the shipped formatters and the one-undo-step apply."""

import types
from typing import Any

from shaderbox.commands import SPEC_BY_ID, CommandId
from shaderbox.formatting import format_glsl, format_python, formatter_for
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
