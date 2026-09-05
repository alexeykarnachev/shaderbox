"""A raw `moderngl.Texture` uniform must survive save -> load, at any dtype.

The branch had demonstrably never executed. Three defects sat in it at once: `UIDocument.save`
wrote `textures/<name>.bin` while only the document dir was mkdir'd (`FileNotFoundError` on first
contact), it recorded no `dtype`, and `Document.load_from_dir` passed none to `gl.texture(...)` --
so anything but `f1` came back as `data size mismatch 512 != 256`. Fixing the loader alone was
impossible; there was nothing on disk to read.

`f2` is the case that matters: 8-bit saturates on the first accumulate pass, so float targets
are a first-class requirement, not an exotic one.
"""

import json
import shutil
from pathlib import Path

import moderngl
import numpy as np
import pytest

from shaderbox.paths import DOCUMENT_JSON_BASENAME, shader_lib_root
from shaderbox.shader_lib import ShaderLibIndex, set_active
from shaderbox.ui_models import load_document_from_dir

# The media example is the one that ships real sampler2D uniforms (u_image, u_video).
_EXAMPLE = (
    Path(__file__).resolve().parent.parent
    / "shaderbox/resources/document_examples/73ea2431-13f6-41e4-b923-04d846b678b0"
)
_SAMPLER = "u_image"


@pytest.fixture(scope="module")
def gl() -> moderngl.Context:
    context = moderngl.create_standalone_context(require=460)
    set_active(ShaderLibIndex.build(shader_lib_root()))
    return context


@pytest.mark.parametrize("dtype", ["f1", "f2", "f4"])
def test_raw_texture_uniform_round_trips(
    gl: moderngl.Context, tmp_path: Path, dtype: str
) -> None:
    document_dir = tmp_path / "document"
    shutil.copytree(_EXAMPLE, document_dir)
    ui_document = load_document_from_dir(document_dir)

    name = _SAMPLER
    assert name in {
        u.name for u in ui_document.document.render_pass.get_active_uniforms()
    }, "the example must declare this sampler for the test to exercise the branch"

    size = (4, 4)
    texture = gl.texture(size, 4, dtype=dtype)
    ui_document.document.render_pass.uniform_values[name] = texture

    ui_document.save(document_dir.parent, document_dir.name)

    with (document_dir / DOCUMENT_JSON_BASENAME).open() as f:
        record = json.load(f)["uniforms"]["main"][name]
    assert record["dtype"] == dtype, (
        "the dtype must be recorded, or the loader cannot recover it"
    )
    assert (document_dir / record["file_path"]).is_file()

    reloaded = load_document_from_dir(document_dir)
    value = reloaded.document.render_pass.uniform_values[name]
    assert isinstance(value, moderngl.Texture)
    assert value.dtype == dtype
    assert tuple(value.size) == size


def test_float_texture_survives_values_outside_unorm(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    # The point of a float target: 8-bit clamps to [0,1] and this is what proves it did not.
    document_dir = tmp_path / "document"
    shutil.copytree(_EXAMPLE, document_dir)
    ui_document = load_document_from_dir(document_dir)

    name = _SAMPLER
    assert name in {
        u.name for u in ui_document.document.render_pass.get_active_uniforms()
    }, "the example must declare this sampler for the test to exercise the branch"

    payload = np.array([7.0, -2.0, 4.5, 1.0] * 4, dtype=np.float32)
    texture = gl.texture((2, 2), 4, data=payload.tobytes(), dtype="f4")
    ui_document.document.render_pass.uniform_values[name] = texture

    ui_document.save(document_dir.parent, document_dir.name)
    reloaded = load_document_from_dir(document_dir)

    value = reloaded.document.render_pass.uniform_values[name]
    assert isinstance(value, moderngl.Texture)
    got = np.frombuffer(value.read(), dtype=np.float32)
    assert got[0] == pytest.approx(7.0)
    assert got[1] == pytest.approx(-2.0)
