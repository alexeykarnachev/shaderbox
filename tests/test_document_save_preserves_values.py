"""A save with no live program must not write away the document's tuned uniform values.

`UIDocument.save` rebuilds `document.json["uniforms"]` from `get_active_uniforms()`, which is empty
whenever `document.program is None`. That state is ordinary, not exotic: `release_program()`
nulls the program and returns WITHOUT recompiling (the recompile rides the next render), so
an external shader edit picked up by the file watcher followed by a quit — `ui.py` calls
`app.save()` on close — lands exactly there. Before the fix that path wrote `"uniforms": {}`
over every value the user had dialled in, while keeping the cosmetic `ui_uniforms` rows.
"""

import json
import shutil
from pathlib import Path

import moderngl
import pytest

from shaderbox.paths import (
    DOCUMENT_JSON_BASENAME,
    PASSES_DIR_NAME,
    pass_shader_name,
    shader_lib_root,
)
from shaderbox.shader_lib import ShaderLibIndex, set_active
from shaderbox.ui_models import load_document_from_dir

_EXAMPLE = (
    Path(__file__).resolve().parent.parent
    / "shaderbox/resources/document_examples/f90f5ff9-29c6-4bcf-aee7-090f20542353"
)


@pytest.fixture(scope="module")
def gl() -> moderngl.Context:
    ctx = moderngl.create_standalone_context(require=460)
    set_active(ShaderLibIndex.build(shader_lib_root()))
    return ctx


def _values(document_dir: Path) -> dict:
    with (document_dir / DOCUMENT_JSON_BASENAME).open() as f:
        return json.load(f)["uniforms"]["main"]


def test_save_without_a_live_program_keeps_the_values_on_disk(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    document_dir = tmp_path / "document"
    shutil.copytree(_EXAMPLE, document_dir)
    ui_document = load_document_from_dir(document_dir)
    before = _values(document_dir)
    assert before, "the example must ship tuned values for this test to mean anything"

    # What the file watcher does on an external shader edit.
    ui_document.document.render_pass.release_program(
        (document_dir / PASSES_DIR_NAME / pass_shader_name("main")).read_text()
    )
    assert ui_document.document.render_pass.program is None
    assert ui_document.document.render_pass.get_active_uniforms() == []

    ui_document.save(document_dir.parent, document_dir.name)

    assert _values(document_dir) == before


def test_save_with_a_live_program_still_rebuilds_from_the_program(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    # The carry-forward must not become a path that freezes stale values: with a program
    # present, the rebuild-from-live-program behaviour is unchanged.
    document_dir = tmp_path / "document"
    shutil.copytree(_EXAMPLE, document_dir)
    ui_document = load_document_from_dir(document_dir)
    assert ui_document.document.render_pass.program is not None

    ui_document.document.render_pass.uniform_values["u_zoomout"] = 42.0
    ui_document.save(document_dir.parent, document_dir.name)

    assert _values(document_dir)["u_zoomout"] == 42.0
