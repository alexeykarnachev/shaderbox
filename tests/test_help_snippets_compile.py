"""Every shipped help snippet must actually compile.

The Help panel presents these as working examples and offers "Insert at caret", so a
snippet that has rotted teaches the user something false and hands them a broken shader.
Only snippets that are whole shaders (they start with `#version`) are compiled; the
display-only ones (the shortcuts table) are skipped by that same test.
"""

from pathlib import Path

import moderngl
import pytest

from shaderbox.document import Document
from shaderbox.help_content import help_sections
from shaderbox.paths import shader_lib_root
from shaderbox.shader_lib import ShaderLibIndex, set_active
from shaderbox.shader_source import ShaderSource

_WHOLE_SHADERS = [
    s for s in help_sections() if s.snippet.lstrip().startswith("#version")
]


@pytest.fixture(scope="module")
def gl() -> moderngl.Context:
    ctx = moderngl.create_standalone_context(require=460)
    set_active(ShaderLibIndex.build(shader_lib_root()))
    return ctx


def test_there_is_at_least_one_compilable_snippet() -> None:
    # Guards the filter above: if `#version` ever stops being the marker, the
    # parametrized test would silently cover nothing.
    assert _WHOLE_SHADERS


@pytest.mark.parametrize("section", _WHOLE_SHADERS, ids=lambda s: s.key)
def test_a_help_snippet_compiles_and_renders(
    gl: moderngl.Context, tmp_path: Path, section: object
) -> None:
    path = tmp_path / f"{section.key}.frag.glsl"  # type: ignore[attr-defined]
    path.write_text(section.snippet, encoding="utf-8")  # type: ignore[attr-defined]
    document = Document(gl=gl, source=ShaderSource.load(path), canvas_size=(32, 32))
    document.render_pass.compile()
    assert document.render_pass.compile_unit.errors == [], [
        e.message for e in document.render_pass.compile_unit.errors
    ]
    document.render(u_time=0.0)
