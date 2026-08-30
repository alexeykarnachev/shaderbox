"""UI uniform rows do not outlive the uniforms they describe.

`UIDocumentState.ui_uniforms` is keyed by a hash of the uniform's name AND shape, and rows are
created lazily in the uniform draw loop — so a rename or a retype stranded the old row and
the dict only ever grew. Shipped examples had accumulated rows for uniforms their shaders
dropped long ago.

The prune lives in `UIDocument.save` (the funnel every path reaches, including headless ones
that never draw a row), never in the draw loop.
"""

import json
import shutil
from pathlib import Path

import moderngl
import pytest
from PIL import Image as PILImage

from shaderbox.media import Image
from shaderbox.paths import (
    DOCUMENT_JSON_BASENAME,
    PASSES_DIR_NAME,
    pass_shader_name,
    shader_lib_root,
)
from shaderbox.shader_lib import ShaderLibIndex, set_active
from shaderbox.ui_models import UIUniform, load_document_from_dir
from shaderbox.util import get_uniform_hash

_EXAMPLES = (
    Path(__file__).resolve().parent.parent / "shaderbox/resources/document_examples"
)
_TEXT_EXAMPLE = _EXAMPLES / "f90f5ff9-29c6-4bcf-aee7-090f20542353"


@pytest.fixture(scope="module")
def gl() -> moderngl.Context:
    ctx = moderngl.create_standalone_context(require=460)
    set_active(ShaderLibIndex.build(shader_lib_root()))
    return ctx


def _rows(document_dir: Path) -> dict:
    with (document_dir / DOCUMENT_JSON_BASENAME).open() as f:
        return json.load(f)["ui_state"]["ui_uniforms"]


def _copy(tmp_path: Path, example: Path) -> Path:
    document_dir = tmp_path / "document"
    shutil.copytree(example, document_dir)
    return document_dir


def test_every_surviving_row_names_a_live_uniform(gl, tmp_path: Path) -> None:
    document_dir = _copy(tmp_path, _TEXT_EXAMPLE)
    ui_document = load_document_from_dir(document_dir)
    ui_document.save(document_dir.parent, document_dir.name)

    live = {u.name for u in ui_document.document.render_pass.get_active_uniforms()}
    survivors = {row["name"] for row in _rows(document_dir).values()}
    assert survivors <= live, (
        f"rows describe uniforms the program lacks: {survivors - live}"
    )


def test_a_retype_does_not_strand_the_old_row(gl, tmp_path: Path) -> None:
    document_dir = _copy(tmp_path, _TEXT_EXAMPLE)
    shader = document_dir / PASSES_DIR_NAME / pass_shader_name("main")
    shader.write_text(
        shader.read_text().replace(
            "uniform float u_zoomout = 10.0;", "uniform vec2 u_zoomout = vec2(10.0);"
        )
    )
    ui_document = load_document_from_dir(document_dir)
    # Stand in for the uniform draw loop, which is where rows are actually born.
    for uniform in ui_document.document.render_pass.get_active_uniforms():
        ui_document.ui_state.ui_uniforms.setdefault(
            get_uniform_hash(uniform), UIUniform.from_uniform(uniform)
        )
    ui_document.save(document_dir.parent, document_dir.name)

    rows = [row for row in _rows(document_dir).values() if row["name"] == "u_zoomout"]
    assert len(rows) == 1, f"the retype stranded a row: {rows}"
    assert rows[0]["dimension"] == 2, "the surviving row must be the NEW shape"


def test_pruning_is_a_fixed_point(gl, tmp_path: Path) -> None:
    document_dir = _copy(tmp_path, _TEXT_EXAMPLE)
    load_document_from_dir(document_dir).save(document_dir.parent, document_dir.name)
    once = (document_dir / DOCUMENT_JSON_BASENAME).read_text()
    load_document_from_dir(document_dir).save(document_dir.parent, document_dir.name)
    assert (document_dir / DOCUMENT_JSON_BASENAME).read_text() == once


def test_a_clean_example_loses_nothing(gl, tmp_path: Path) -> None:
    # Falsifier for the prune itself: an over-broad rule would quietly strip live rows, and
    # "rows went away" looks the same as "rows were stale" without this side of the bound.
    document_dir = _copy(tmp_path, _EXAMPLES / "8d454b7b-bd48-49dc-aebe-58b9e31cfc28")
    before = _rows(document_dir)
    load_document_from_dir(document_dir).save(document_dir.parent, document_dir.name)
    assert _rows(document_dir).keys() == before.keys()


def test_no_prune_without_a_live_program(gl, tmp_path: Path) -> None:
    # With no program there is nothing to prune against, so the honest answer is "keep what
    # is there" — pruning would read as "delete every row".
    document_dir = _copy(tmp_path, _TEXT_EXAMPLE)
    ui_document = load_document_from_dir(document_dir)
    before = _rows(document_dir)
    ui_document.document.render_pass.release_program(
        (document_dir / PASSES_DIR_NAME / pass_shader_name("main")).read_text()
    )
    assert ui_document.document.render_pass.program is None

    ui_document.save(document_dir.parent, document_dir.name)
    assert _rows(document_dir).keys() == before.keys()


def _bind_image(tmp_path: Path, name: str) -> Image:
    src = tmp_path / f"{name}_src.png"
    PILImage.new("RGBA", (8, 8), (10, 200, 30, 255)).save(src)
    return Image(src)


_SAMPLER_SHADER = """#version 460 core
in vec2 vs_uv;
uniform sampler2D {name};
out vec4 fs_color;
void main() {{ fs_color = texture({name}, vs_uv); }}
"""


def test_renaming_a_sampler_does_not_orphan_its_media_file(gl, tmp_path: Path) -> None:
    # The unbind cleanup is keyed by the uniform's OWN name, so it only ever visits names
    # the shader still has — a renamed-away sampler's file was never looked at again and
    # stayed on disk forever (riding along duplicate_document).
    document_dir = tmp_path / "document"
    (document_dir / PASSES_DIR_NAME).mkdir(parents=True)
    (document_dir / DOCUMENT_JSON_BASENAME).write_text(
        json.dumps({"canvas_size": [64, 64], "uniforms": {}, "ui_state": {}})
    )

    for name in ("u_tex0", "u_tex1", "u_tex2"):
        (document_dir / PASSES_DIR_NAME / pass_shader_name("main")).write_text(
            _SAMPLER_SHADER.format(name=name)
        )
        ui_document = load_document_from_dir(document_dir)
        ui_document.document.render_pass.uniform_values[name] = _bind_image(
            tmp_path, name
        )
        ui_document.save(document_dir.parent, document_dir.name, rebind=False)

    on_disk = sorted(p.name for p in (document_dir / "media" / "main").iterdir())
    assert on_disk == ["u_tex2.png"], f"orphaned media survived: {on_disk}"

    with (document_dir / DOCUMENT_JSON_BASENAME).open() as f:
        assert sorted(json.load(f)["uniforms"]["main"]) == ["u_tex2"]


def test_a_bound_sampler_keeps_its_file(gl, tmp_path: Path) -> None:
    # The other side of the bound: the sweep must not delete an asset a uniform still uses.
    document_dir = tmp_path / "document"
    (document_dir / PASSES_DIR_NAME).mkdir(parents=True)
    (document_dir / PASSES_DIR_NAME / pass_shader_name("main")).write_text(
        _SAMPLER_SHADER.format(name="u_tex")
    )
    (document_dir / DOCUMENT_JSON_BASENAME).write_text(
        json.dumps({"canvas_size": [64, 64], "uniforms": {}, "ui_state": {}})
    )
    ui_document = load_document_from_dir(document_dir)
    ui_document.document.render_pass.uniform_values["u_tex"] = _bind_image(
        tmp_path, "u_tex"
    )

    ui_document.save(document_dir.parent, document_dir.name, rebind=False)
    ui_document.save(document_dir.parent, document_dir.name, rebind=False)  # idempotent

    assert [p.name for p in (document_dir / "media" / "main").iterdir()] == [
        "u_tex.png"
    ]
