"""The uniforms panel's sampler rows show what the sampler reads (071 W-D, D9).

The resolver is the document's; the row draws from its answer, so pinning the answer on the
shipped Radiance Cascades example pins the row's three states."""

import shutil
from pathlib import Path
from typing import Any

import moderngl
import pytest

from shaderbox.constants import DOCUMENT_EXAMPLES_DIR
from shaderbox.paths import shader_lib_root
from shaderbox.shader_lib import ShaderLibIndex, set_active
from shaderbox.ui_models import load_document_from_dir

_RC = DOCUMENT_EXAMPLES_DIR / "77a84d27-2e5b-406d-8011-ee1cb1a9587c"


@pytest.fixture(scope="module")
def gl() -> moderngl.Context:
    ctx = moderngl.create_standalone_context(require=460)
    set_active(ShaderLibIndex.build(shader_lib_root()))
    return ctx


def _online(document: Any) -> None:
    # One never-rendered pass per frame, the way the live loop's sweep does it.
    for frame in range(len(document.passes) + 2):
        document.begin_frame(frame)
        document.render(u_time=frame / 30.0)
        pending = next(
            (n for n, p in document.passes.items() if not p.first_render_done), None
        )
        if pending is not None:
            document.render(target=pending)


def test_a_wired_sampler_names_its_pass_and_an_unwired_one_reads_black(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    shutil.copytree(_RC, tmp_path / "rc")
    document = load_document_from_dir(tmp_path / "rc").document
    _online(document)
    # The walk's case: composite's two samplers read cascade and paint, never the default image.
    assert document.sampler_source("composite", "u_cascade") == "cascade"
    assert document.sampler_source("composite", "u_paint") == "paint"
    # An explicit "(none)" and a stale name both read black, which the renderer binds for them.
    document.graph = document.graph.with_input("composite", "u_paint", "")
    assert document.sampler_source("composite", "u_paint") is None
    document.graph = document.graph.with_input("composite", "u_paint", "gone")
    assert document.sampler_source("composite", "u_paint") is None
    # A pass that declares no such sampler, and a pass that does not exist.
    assert document.sampler_source("paint", "u_paint") is None
    assert document.sampler_source("nope", "u_paint") is None
    document.release()
