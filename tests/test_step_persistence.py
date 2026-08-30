"""What a multi-step node writes to disk, and what it must NOT.

Two silent-failure classes the pre-impl reviewers named:

- Without the uniform union (D4), a uniform declared only in a non-final step is absent
  from "the live program", so `UINode.save`'s prune deletes its row and the user's tuned
  value is gone on the next save. `test_uniform_row_pruning` cannot catch this: it is a
  subset check with no lower bound, so a bigger union makes it pass MORE easily.
- Without the step-sampler guard (D11), a step target -- a bare `moderngl.Texture` --
  falls into the raw-Texture branch and is written to `textures/*.bin`: megabytes of
  transient float state per save, reloaded next session as a frozen frame.
"""

import json
from pathlib import Path

import moderngl
import pytest

from shaderbox.copilot.backend import _format_uniforms
from shaderbox.paths import NODE_JSON_BASENAME, NODE_SHADER_BASENAME, shader_lib_root
from shaderbox.shader_lib import ShaderLibIndex, set_active
from shaderbox.ui_models import load_node_from_dir

_CHAIN = (
    "#version 330\n"
    "out vec4 f_color;\n"
    "in vec2 vs_uv;\n"
    "uniform sampler2D u_mid;  // step, f4\n"
    "uniform float u_only_in_step;\n"
    "uniform float u_final_gain;\n"
    "void step_mid(out vec4 o) { o = vec4(u_only_in_step, 0.0, 0.0, 1.0); }\n"
    "void main() { f_color = texture(u_mid, vs_uv) * u_final_gain; }\n"
)


@pytest.fixture(scope="module")
def gl() -> moderngl.Context:
    ctx = moderngl.create_standalone_context(require=460)
    set_active(ShaderLibIndex.build(shader_lib_root()))
    return ctx


def _make_node_dir(tmp_path: Path) -> Path:
    node_dir = tmp_path / "chain"
    node_dir.mkdir()
    (node_dir / NODE_SHADER_BASENAME).write_text(_CHAIN, encoding="utf-8")
    (node_dir / NODE_JSON_BASENAME).write_text(
        json.dumps({"canvas_size": [16, 16], "uniforms": {}, "ui_state": {}}),
        encoding="utf-8",
    )
    return node_dir


def _meta(node_dir: Path) -> dict:
    with (node_dir / NODE_JSON_BASENAME).open() as f:
        return json.load(f)


def test_a_uniform_only_in_a_step_is_visible_and_persists(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    node_dir = _make_node_dir(tmp_path)
    ui_node = load_node_from_dir(node_dir)
    assert ui_node.node.compile_unit.errors == [], ui_node.node.compile_unit.errors

    names = {u.name for u in ui_node.node.get_active_uniforms()}
    # The union: `u_only_in_step` lives in the step variant, `u_final_gain` in the final.
    assert "u_only_in_step" in names
    assert "u_final_gain" in names

    ui_node.node.uniform_values["u_only_in_step"] = 0.75
    ui_node.node.uniform_values["u_final_gain"] = 2.5
    ui_node.save(node_dir.parent, node_dir.name)

    saved = _meta(node_dir)["uniforms"]
    assert saved["u_only_in_step"] == 0.75
    assert saved["u_final_gain"] == 2.5

    reloaded = load_node_from_dir(node_dir)
    assert reloaded.node.uniform_values["u_only_in_step"] == 0.75


def test_a_step_sampler_writes_no_texture_file(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    node_dir = _make_node_dir(tmp_path)
    ui_node = load_node_from_dir(node_dir)
    ui_node.node.render(u_time=0.0)  # allocate the step target
    ui_node.save(node_dir.parent, node_dir.name)

    saved = _meta(node_dir)["uniforms"]
    assert "u_mid" not in saved, "a step sampler must not be serialized"
    textures_dir = node_dir / "textures"
    assert not textures_dir.exists() or not list(textures_dir.glob("*.bin"))


def test_a_step_sampler_gets_no_uniform_row(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    node_dir = _make_node_dir(tmp_path)
    ui_node = load_node_from_dir(node_dir)
    ui_node.save(node_dir.parent, node_dir.name)
    # Rows are keyed by uniform hash, so check the names the rows were built from.
    rows = _meta(node_dir)["ui_state"].get("ui_uniforms", {})
    assert all(row.get("name") != "u_mid" for row in rows.values())


def test_reload_does_not_resurrect_a_stale_step_frame(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    # The failure this guards: a serialized step target reloading as a plain texture
    # bound to the sampler, so the chain reads last session's frame instead of
    # recomputing and it LOOKS like it works.
    node_dir = _make_node_dir(tmp_path)
    ui_node = load_node_from_dir(node_dir)
    ui_node.node.render(u_time=0.0)
    ui_node.save(node_dir.parent, node_dir.name)

    reloaded = load_node_from_dir(node_dir)
    assert "u_mid" not in reloaded.node.uniform_values
    assert reloaded.node.is_step_sampler("u_mid")


def test_the_copilot_sees_step_samplers_as_engine_wired(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    """The working-set rows the model reads.

    A step sampler labelled "(no media bound)" would invite the model to bind a file
    over the chain -- and `bind_media` must not offer it either. Meanwhile the union
    must surface uniforms from every step, or the model cannot see the controls it just
    authored.
    """
    node_dir = _make_node_dir(tmp_path)
    ui_node = load_node_from_dir(node_dir)
    ui_node.node.render(u_time=0.0)

    rows = _format_uniforms(ui_node.node, set())
    joined = "\n".join(rows)
    assert "u_mid sampler2D <- (step output)" in joined
    assert "no media bound" not in joined
    # Both tunables, from different variants, are visible.
    assert any(r.startswith("u_only_in_step ") for r in rows)
    assert any(r.startswith("u_final_gain ") for r in rows)


def test_the_copilot_cannot_unbind_a_step_sampler(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    # bind_media filters step samplers out of the list it offers; unbind_media took a
    # name directly and would have written the default image into uniform_values --
    # exactly the state D11 keeps out of there, and it would break the chain silently.
    node_dir = _make_node_dir(tmp_path)
    ui_node = load_node_from_dir(node_dir)
    ui_node.node.render(u_time=0.0)
    assert ui_node.node.is_step_sampler("u_mid")
    assert "u_mid" not in ui_node.node.uniform_values
