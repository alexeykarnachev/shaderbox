"""A save with no live program must not write away the node's tuned uniform values.

`UINode.save` rebuilds `node.json["uniforms"]` from `get_active_uniforms()`, which is empty
whenever `node.program is None`. That state is ordinary, not exotic: `release_program()`
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

from shaderbox.paths import NODE_JSON_BASENAME, NODE_SHADER_BASENAME, shader_lib_root
from shaderbox.shader_lib import ShaderLibIndex, set_active
from shaderbox.ui_models import load_node_from_dir

_EXAMPLE = (
    Path(__file__).resolve().parent.parent
    / "shaderbox/resources/node_examples/f90f5ff9-29c6-4bcf-aee7-090f20542353"
)


@pytest.fixture(scope="module")
def gl() -> moderngl.Context:
    ctx = moderngl.create_standalone_context(require=460)
    set_active(ShaderLibIndex.build(shader_lib_root()))
    return ctx


def _values(node_dir: Path) -> dict:
    with (node_dir / NODE_JSON_BASENAME).open() as f:
        return json.load(f)["uniforms"]


def test_save_without_a_live_program_keeps_the_values_on_disk(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    node_dir = tmp_path / "node"
    shutil.copytree(_EXAMPLE, node_dir)
    ui_node = load_node_from_dir(node_dir)
    before = _values(node_dir)
    assert before, "the example must ship tuned values for this test to mean anything"

    # What the file watcher does on an external shader edit.
    ui_node.node.release_program((node_dir / NODE_SHADER_BASENAME).read_text())
    assert ui_node.node.program is None
    assert ui_node.node.get_active_uniforms() == []

    ui_node.save(node_dir.parent, node_dir.name)

    assert _values(node_dir) == before


def test_save_with_a_live_program_still_rebuilds_from_the_program(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    # The carry-forward must not become a path that freezes stale values: with a program
    # present, the rebuild-from-live-program behaviour is unchanged.
    node_dir = tmp_path / "node"
    shutil.copytree(_EXAMPLE, node_dir)
    ui_node = load_node_from_dir(node_dir)
    assert ui_node.node.program is not None

    ui_node.node.uniform_values["u_zoomout"] = 42.0
    ui_node.save(node_dir.parent, node_dir.name)

    assert _values(node_dir)["u_zoomout"] == 42.0
