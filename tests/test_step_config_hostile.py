"""A malformed `node.json` must cost a setting, never the node.

The rework moved six knobs out of GLSL -- where the compiler checked them -- into
`node.json`, where nothing did. Every defect here was found by a post-impl reviewer
against a suite that was fully green: the tests covered the happy path of the new config
surface and nothing that reached it from disk.
"""

import json
import shutil
from pathlib import Path

import moderngl
import pytest
from pydantic import ValidationError

from shaderbox.constants import NODE_EXAMPLES_DIR
from shaderbox.paths import NODE_JSON_BASENAME, shader_lib_root
from shaderbox.shader_lib import ShaderLibIndex, set_active
from shaderbox.step_spec import DTYPES, StepConfig
from shaderbox.ui_models import load_nodes_from_dir

_EXAMPLE = NODE_EXAMPLES_DIR / "b41e7c9d-2a68-4f53-8e17-6c0d95a3f2b8"


@pytest.fixture(scope="module")
def gl() -> moderngl.Context:
    ctx = moderngl.create_standalone_context(require=460)
    set_active(ShaderLibIndex.build(shader_lib_root()))
    return ctx


def _node_dir_with(tmp_path: Path, blur_patch: object) -> Path:
    root = tmp_path / "root"
    root.mkdir()
    shutil.copytree(_EXAMPLE, root / "n1")
    meta_path = root / "n1" / NODE_JSON_BASENAME
    meta = json.loads(meta_path.read_text())
    meta["ui_state"]["step_configs"]["blur"] = blur_patch
    meta_path.write_text(json.dumps(meta))
    return root


@pytest.mark.parametrize(
    "patch",
    [
        {"scale": "big"},  # wrong type
        {"scale": 5000},  # would fail framebuffer completeness
        {"scale": 0},  # would collapse the target
        {"scale": -2},
        {"dtype": "f8"},  # not a real moderngl dtype
        {"dtype": "u1"},  # real, but outside what the panel can show
        {"filter_linear": "yes"},
        "not even a dict",
    ],
)
def test_a_bad_step_config_costs_the_setting_not_the_node(
    gl: moderngl.Context, tmp_path: Path, patch: object
) -> None:
    # The node must survive with its shader, name and uniforms intact -- losing those to
    # one field is the data-loss class `model_salvage` exists to prevent.
    root = _node_dir_with(tmp_path, patch)
    nodes = load_nodes_from_dir(root)
    assert list(nodes) == ["n1"], "the whole node was dropped"
    node = nodes["n1"].node
    assert node.compile_unit.errors == []
    node.render(u_time=0.0)  # must not raise
    # The offending step falls back to defaults rather than vanishing.
    assert "blur" in {s.name for s in node.steps}


def test_a_good_config_beside_a_bad_one_survives(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    root = _node_dir_with(tmp_path, {"dtype": "f8"})
    nodes = load_nodes_from_dir(root)
    configs = nodes["n1"].ui_state.step_configs
    # `bright` was untouched and keeps its shipped half-size.
    assert configs["bright"].scale == 0.5
    assert "blur" not in configs  # reset to defaults


@pytest.mark.parametrize("scale", [0.0, -1.0, 1.5, 5000.0])
def test_the_model_refuses_a_scale_that_cannot_allocate(scale: float) -> None:
    # A step larger than the canvas has no use case and is the shape that exhausts VRAM;
    # zero or negative fails framebuffer completeness outright.
    with pytest.raises(ValidationError):
        StepConfig(scale=scale)


@pytest.mark.parametrize("dtype", ["f8", "u1", "rgba", ""])
def test_the_model_refuses_a_dtype_the_panel_cannot_show(dtype: str) -> None:
    # `u1` is a REAL moderngl dtype: it would load, render, and then crash the Steps
    # panel's combo, which is worse than failing at load.
    with pytest.raises(ValidationError):
        StepConfig(dtype=dtype)


def test_every_listed_dtype_is_accepted() -> None:
    # Guards the constraint against narrowing below what the panel offers.
    for dtype in DTYPES:
        assert StepConfig(dtype=dtype).dtype == dtype
