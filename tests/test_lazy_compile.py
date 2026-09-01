"""Loading compiles nothing (066 D1): a pass compiles when something first needs its program.

Pins the three halves of the decision: `load_from_dir` leaves every pass's program None while
the tuned uniform VALUES still land in `uniform_values` (they never needed a program), a
never-compiled pass compiles itself the moment `get_active_uniforms()` is asked, and a BROKEN
source gets exactly one attempt — its errors stick until `invalidate()` re-arms the retry.
"""

import shutil
from pathlib import Path

import moderngl
import pytest

from shaderbox.core import ENGINE_DRIVEN_UNIFORMS, Canvas
from shaderbox.paths import shader_lib_root
from shaderbox.shader_lib import ShaderLibIndex, set_active
from shaderbox.ui_models import load_document_from_dir

_EXAMPLES = (
    Path(__file__).resolve().parent.parent / "shaderbox/resources/document_examples"
)
# The tuned single-pass example and the five-pass bloom chain.
_TUNED = _EXAMPLES / "f90f5ff9-29c6-4bcf-aee7-090f20542353"
_BLOOM = _EXAMPLES / "1c4f8a20-7b6e-4d31-9a55-2f0e6b8c31d4"


@pytest.fixture(scope="module")
def gl() -> moderngl.Context:
    ctx = moderngl.create_standalone_context(require=460)
    set_active(ShaderLibIndex.build(shader_lib_root()))
    return ctx


def test_load_compiles_no_pass(gl: moderngl.Context, tmp_path: Path) -> None:
    document_dir = tmp_path / "document"
    shutil.copytree(_BLOOM, document_dir)
    document = load_document_from_dir(document_dir).document
    assert len(document.passes) == 5
    for render_pass in document.passes.values():
        assert render_pass.program is None
        assert render_pass.compile_unit.error_raw == ""


def test_uniform_values_load_without_a_program(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    document_dir = tmp_path / "document"
    shutil.copytree(_TUNED, document_dir)
    render_pass = load_document_from_dir(document_dir).document.render_pass
    assert render_pass.program is None
    assert render_pass.uniform_values, (
        "the example ships tuned values; they must survive load"
    )


def test_get_active_uniforms_compiles_on_demand(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    document_dir = tmp_path / "document"
    shutil.copytree(_TUNED, document_dir)
    render_pass = load_document_from_dir(document_dir).document.render_pass
    assert render_pass.program is None
    uniforms = render_pass.get_active_uniforms()
    assert render_pass.program is not None
    assert "u_zoomout" in {u.name for u in uniforms}
    # Seeding rides the lazy compile: every returned uniform must have a value, or a
    # consumer that indexes uniform_values (the panel's row loop) crashes on a pass that
    # compiled here but never rendered.
    for uniform in uniforms:
        if uniform.name not in ENGINE_DRIVEN_UNIFORMS:
            assert uniform.name in render_pass.uniform_values, uniform.name


def test_a_foreign_canvas_render_leaves_first_render_pending(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    # A probe/export renders into its own canvas and must not consume the live loop's
    # first-render budget — the document's own canvases (what the grid tile shows) are
    # still unwritten, and a consumed budget would leave the tile black for the session.
    document_dir = tmp_path / "document"
    shutil.copytree(_TUNED, document_dir)
    document = load_document_from_dir(document_dir).document
    foreign = Canvas(gl=gl, size=(8, 8))
    document.render(canvas=foreign)
    assert not document.first_render_done
    document.render()
    assert document.first_render_done
    foreign.release()


def test_a_broken_source_is_attempted_once(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    document_dir = tmp_path / "document"
    shutil.copytree(_TUNED, document_dir)
    render_pass = load_document_from_dir(document_dir).document.render_pass
    render_pass.release_program("this is not glsl")

    assert render_pass.get_active_uniforms() == []
    assert render_pass.compile_unit.error_raw
    unit_after_first_attempt = render_pass.compile_unit
    # A second ask must not retry: a retry would replace compile_unit with a fresh object.
    assert render_pass.get_active_uniforms() == []
    assert render_pass.compile_unit is unit_after_first_attempt

    # invalidate() (a source or lib change) re-arms the compile.
    render_pass.release_program((document_dir / "passes/main.frag.glsl").read_text())
    assert render_pass.get_active_uniforms() != []
    assert render_pass.program is not None
