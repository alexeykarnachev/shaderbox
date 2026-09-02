"""A document round-trips its graph, its passes and its per-pass assets (065 stage 4).

The spec's persistence checks 10-12. `graph.json` is app-written derived state, so the salvage is
per KEY: a malformed pass entry costs that pass's wiring, never the document, and a document that
loads unwired still opens with its shaders intact — which is what makes it fixable.

The GL-backed half needs a real context; the salvage half is pure and always runs.
"""

import json
import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import moderngl
import numpy as np
import pytest
from PIL import Image as PILImage

from shaderbox.constants import DEFAULT_FS_FILE_PATH
from shaderbox.document import DEFAULT_PASS_NAME, Document, load_graph
from shaderbox.media import Image
from shaderbox.pass_graph import PassEntry, PassGraph, TargetConfig
from shaderbox.paths import (
    DOCUMENT_JSON_BASENAME,
    GRAPH_JSON_BASENAME,
    PASSES_DIR_NAME,
    pass_shader_name,
)
from shaderbox.shader_source import ShaderSource
from shaderbox.ui_models import UIDocument, load_document_from_dir

_SAMPLER = """#version 460 core
in vec2 vs_uv;
uniform sampler2D u_tex;
uniform float u_amount;
out vec4 fs_color;
void main() { fs_color = texture(u_tex, vs_uv) * u_amount; }
"""

_PLAIN = """#version 460 core
in vec2 vs_uv;
uniform float u_level;
out vec4 fs_color;
void main() { fs_color = vec4(u_level, 0.0, 0.0, 1.0); }
"""


@pytest.fixture(scope="module")
def gl_ctx() -> Iterator[moderngl.Context]:
    os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "4.6")
    os.environ.setdefault("MESA_GLSL_VERSION_OVERRIDE", "460")
    # Default-backend like every other GL module's fixture — an EXPLICIT backend="egl" context
    # released here poisons the process's EGL display and the NEXT module's first program
    # compile segfaults (module-order-only; one context recipe per process is the rule).
    try:
        ctx = moderngl.create_standalone_context()
    except Exception as e:
        pytest.skip(f"no standalone GL context available: {e}")
    yield ctx
    ctx.release()


def _write_document(
    document_dir: Path, sources: dict[str, str], graph: PassGraph | None = None
) -> None:
    passes = document_dir / PASSES_DIR_NAME
    passes.mkdir(parents=True, exist_ok=True)
    for name, src in sources.items():
        (passes / pass_shader_name(name)).write_text(src)
    (document_dir / DOCUMENT_JSON_BASENAME).write_text(
        json.dumps({"canvas_size": [16, 16], "uniforms": {}, "ui_state": {}})
    )
    if graph is not None:
        (document_dir / GRAPH_JSON_BASENAME).write_text(json.dumps(graph.model_dump()))


def _image(path: Path, color: tuple[int, int, int]) -> Path:
    PILImage.fromarray(np.full((4, 4, 3), color, dtype=np.uint8), "RGB").save(path)
    return path


# ---------------------------------------------------------------- check 10


def test_a_graph_round_trips_every_field(tmp_path: Path) -> None:
    graph = PassGraph(
        version=1,
        output="composite",
        passes={
            "scene": PassEntry(target=TargetConfig(dtype="f4", scale=0.5, wrap=True)),
            "trail": PassEntry(
                inputs={"u_src": "scene", "u_prev": "trail"},
                target=TargetConfig(persist=True, filter_linear=False),
            ),
            "composite": PassEntry(inputs={"u_a": "scene", "u_b": "trail"}),
        },
    )
    path = tmp_path / GRAPH_JSON_BASENAME
    path.write_text(json.dumps(graph.model_dump()))
    assert load_graph(path) == graph


def test_a_document_round_trips_its_graph_and_passes(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    document_dir = tmp_path / "doc"
    graph = PassGraph(
        output="b",
        passes={
            "a": PassEntry(target=TargetConfig(scale=0.5)),
            "b": PassEntry(inputs={"u_tex": "a"}),
        },
    )
    _write_document(document_dir, {"a": _PLAIN, "b": _SAMPLER}, graph)
    loaded = load_document_from_dir(document_dir)
    loaded.document.render_pass.uniform_values["u_amount"] = 0.75
    loaded.save(document_dir.parent, document_dir.name, rebind=False)

    again = load_document_from_dir(document_dir)
    assert sorted(again.document.passes) == ["a", "b"]
    assert again.document.graph.output == "b"
    assert again.document.graph.passes["b"].inputs == {"u_tex": "a"}
    assert again.document.graph.passes["a"].target.scale == 0.5
    assert again.document.render_pass.uniform_values["u_amount"] == 0.75
    again.document.release()
    loaded.document.release()


def test_a_pass_file_with_no_graph_entry_gets_defaults(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # The FILES are the passes: a pass authored on disk without touching graph.json still loads.
    document_dir = tmp_path / "doc"
    _write_document(document_dir, {"solo": _PLAIN})
    loaded = load_document_from_dir(document_dir)
    assert list(loaded.document.passes) == ["solo"]
    assert loaded.document.graph.passes["solo"] == PassEntry()
    assert loaded.document.render_pass is loaded.document.passes["solo"]
    loaded.document.release()


def test_a_removed_pass_does_not_come_back(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # The loader enumerates FILES, so a save that leaves an orphan file resurrects the pass.
    document_dir = tmp_path / "doc"
    _write_document(document_dir, {"a": _PLAIN, "b": _PLAIN})
    loaded = load_document_from_dir(document_dir)
    loaded.document.passes.pop("b").release()
    loaded.document.graph = PassGraph(output="a", passes={"a": PassEntry()})
    loaded.save(document_dir.parent, document_dir.name, rebind=False)
    assert not (document_dir / PASSES_DIR_NAME / pass_shader_name("b")).exists()
    assert list(load_document_from_dir(document_dir).document.passes) == ["a"]
    loaded.document.release()


# ---------------------------------------------------------------- check 11


def test_a_malformed_graph_entry_costs_that_entry_not_the_document(
    tmp_path: Path,
) -> None:
    path = tmp_path / GRAPH_JSON_BASENAME
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "output": "good",
                "passes": {
                    "good": {"inputs": {"u_src": "other"}},
                    "broken": {"target": {"dtype": "f8", "scale": 99.0}},
                    "notadict": "nonsense",
                },
            }
        )
    )
    graph = load_graph(path)
    assert graph.output == "good"
    assert graph.passes["good"].inputs == {"u_src": "other"}
    # `broken`'s own fields were invalid, so it falls back to a default entry rather than
    # taking its siblings with it.
    assert graph.passes["broken"] == PassEntry()
    assert "notadict" not in graph.passes


def test_a_hostile_graph_file_loads_unwired_rather_than_raising(
    tmp_path: Path,
) -> None:
    for body in ("null", "[1, 2]", '"a string"', "{", "42"):
        path = tmp_path / GRAPH_JSON_BASENAME
        path.write_text(body)
        assert load_graph(path) == PassGraph(), f"{body!r} did not degrade cleanly"
    assert load_graph(tmp_path / "absent.json") == PassGraph()


def test_an_unknown_graph_key_is_dropped_and_the_rest_survives(
    tmp_path: Path,
) -> None:
    path = tmp_path / GRAPH_JSON_BASENAME
    path.write_text(
        json.dumps({"output": "a", "passes": {"a": {"inputs": {}}}, "retired_field": 7})
    )
    graph = load_graph(path)
    assert graph.output == "a"
    assert list(graph.passes) == ["a"]


def test_a_malformed_pass_file_costs_that_pass_not_the_document(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # D14. A pass whose SOURCE does not compile is still a pass — it loads and reports its error.
    # The document is only lost when it has no readable pass file at all.
    document_dir = tmp_path / "doc"
    _write_document(document_dir, {"ok": _PLAIN, "broken": "this is not glsl at all"})
    loaded = load_document_from_dir(document_dir)
    assert sorted(loaded.document.passes) == ["broken", "ok"]
    loaded.document.passes["broken"].compile()
    assert loaded.document.passes["broken"].compile_unit.errors
    assert loaded.document.passes["ok"].compile_unit.errors == []
    loaded.document.release()


def test_a_graph_entry_naming_a_missing_file_is_dropped(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    document_dir = tmp_path / "doc"
    _write_document(
        document_dir,
        {"real": _PLAIN},
        PassGraph(
            output="real",
            passes={"real": PassEntry(), "ghost": PassEntry()},
        ),
    )
    loaded = load_document_from_dir(document_dir)
    assert list(loaded.document.passes) == ["real"]
    assert "ghost" not in loaded.document.graph.passes
    loaded.document.release()


# ---------------------------------------------------------------- check 12


def test_two_passes_binding_the_same_sampler_name_keep_separate_media(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # D16: with uniforms pass-scoped, a FLAT asset layout would have the two `u_tex` files
    # overwrite each other and the orphan sweep would delete the survivor's.
    document_dir = tmp_path / "doc"
    _write_document(
        document_dir,
        {"left": _SAMPLER, "right": _SAMPLER},
        PassGraph(output="left", passes={"left": PassEntry(), "right": PassEntry()}),
    )
    loaded = load_document_from_dir(document_dir)
    red = Image(_image(tmp_path / "red.png", (255, 0, 0)))
    blue = Image(_image(tmp_path / "blue.png", (0, 0, 255)))
    loaded.document.passes["left"].uniform_values["u_tex"] = red
    loaded.document.passes["right"].uniform_values["u_tex"] = blue
    loaded.save(document_dir.parent, document_dir.name, rebind=False)

    assert (document_dir / "media" / "left" / "u_tex.png").is_file()
    assert (document_dir / "media" / "right" / "u_tex.png").is_file()
    meta = json.loads((document_dir / DOCUMENT_JSON_BASENAME).read_text())
    assert meta["uniforms"]["left"]["u_tex"]["file_path"] == "media/left/u_tex.png"
    assert meta["uniforms"]["right"]["u_tex"]["file_path"] == "media/right/u_tex.png"

    again = load_document_from_dir(document_dir)
    left_px = again.document.passes["left"].uniform_values["u_tex"].texture.read()[:3]
    right_px = again.document.passes["right"].uniform_values["u_tex"].texture.read()[:3]
    assert tuple(left_px) != tuple(right_px), "the two passes' media collided"
    again.document.release()
    loaded.document.release()


def test_the_orphan_sweep_is_scoped_to_one_pass(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # A save must not reach into a sibling pass's asset dir. Falsifier: a sweep keyed by asset
    # NAME alone deletes `right/u_tex.png` because `left` no longer references that name.
    document_dir = tmp_path / "doc"
    _write_document(
        document_dir,
        {"left": _SAMPLER, "right": _SAMPLER},
        PassGraph(output="left", passes={"left": PassEntry(), "right": PassEntry()}),
    )
    loaded = load_document_from_dir(document_dir)
    loaded.document.passes["right"].uniform_values["u_tex"] = Image(
        _image(tmp_path / "keep.png", (0, 255, 0))
    )
    loaded.save(document_dir.parent, document_dir.name, rebind=False)
    loaded.save(document_dir.parent, document_dir.name, rebind=False)
    assert (document_dir / "media" / "right" / "u_tex.png").is_file()
    loaded.document.release()


def test_a_deleted_passs_assets_are_swept(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    document_dir = tmp_path / "doc"
    _write_document(
        document_dir,
        {"left": _SAMPLER, "right": _SAMPLER},
        PassGraph(output="left", passes={"left": PassEntry(), "right": PassEntry()}),
    )
    loaded = load_document_from_dir(document_dir)
    loaded.document.passes["right"].uniform_values["u_tex"] = Image(
        _image(tmp_path / "gone.png", (0, 255, 0))
    )
    loaded.save(document_dir.parent, document_dir.name, rebind=False)
    assert (document_dir / "media" / "right" / "u_tex.png").is_file()

    loaded.document.passes.pop("right").release()
    loaded.document.graph = PassGraph(output="left", passes={"left": PassEntry()})
    loaded.save(document_dir.parent, document_dir.name, rebind=False)
    assert not (document_dir / "media" / "right" / "u_tex.png").exists()
    loaded.document.release()


def test_a_default_document_saves_and_reloads(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # An empty project must start: the seeded starter has to survive its own save/load.
    doc = Document(
        gl=gl_ctx, source=ShaderSource.load(DEFAULT_FS_FILE_PATH), canvas_size=(16, 16)
    )
    doc.render()
    ui_document = UIDocument(document=doc, id="fresh")
    ui_document.save(tmp_path, "fresh", rebind=False)
    assert (
        tmp_path / "fresh" / PASSES_DIR_NAME / pass_shader_name(DEFAULT_PASS_NAME)
    ).is_file()
    reloaded = load_document_from_dir(tmp_path / "fresh")
    assert list(reloaded.document.passes) == [DEFAULT_PASS_NAME]
    reloaded.document.release()
    doc.release()


def test_a_document_with_no_pass_file_is_skipped_not_crashed(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    document_dir = tmp_path / "empty"
    document_dir.mkdir()
    (document_dir / DOCUMENT_JSON_BASENAME).write_text(
        json.dumps({"canvas_size": [16, 16], "uniforms": {}, "ui_state": {}})
    )
    with pytest.raises(ValueError, match="no readable pass file"):
        Document.load_from_dir(document_dir, gl=gl_ctx)


def test_an_unreadable_asset_costs_that_uniform_not_the_pass(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    document_dir = tmp_path / "doc"
    _write_document(document_dir, {"main": _SAMPLER})
    meta = json.loads((document_dir / DOCUMENT_JSON_BASENAME).read_text())
    meta["uniforms"] = {
        "main": {
            "u_amount": 0.5,
            "u_tex": {
                "file_path": "media/main/missing.png",
                "size": [4, 4],
                "components": 3,
                "dtype": "f1",
            },
        }
    }
    (document_dir / DOCUMENT_JSON_BASENAME).write_text(json.dumps(meta))
    loaded = load_document_from_dir(document_dir)
    assert loaded.document.render_pass.uniform_values["u_amount"] == 0.5
    loaded.document.release()


@pytest.mark.parametrize(
    "corrupt", ["not json at all {", "[1, 2, 3]", "null", "42", ""]
)
def test_a_corrupt_document_json_costs_its_metadata_not_the_document(
    gl_ctx: moderngl.Context, tmp_path: Path, corrupt: str
) -> None:
    # `Document.load_from_dir` runs from the LIVE per-frame sync, so a raise here escapes into
    # the imgui frame loop and takes the app down -- the same shape as the shipped crash where a
    # `relative_to` raised inside a draw call. The app's own saves are not atomic, so it can
    # produce exactly this file by dying mid-write.
    #
    # Falsifier: restore the bare `json.load` and every case below raises.
    source = (
        Path(__file__).resolve().parent.parent
        / "shaderbox"
        / "resources"
        / "document_examples"
        / "53724dbd-8efb-4c09-8c7d-28d626a066e7"
    )
    document_dir = tmp_path / "corrupt"
    shutil.copytree(source, document_dir)
    (document_dir / "document.json").write_text(corrupt, encoding="utf-8")

    document, metadata = Document.load_from_dir(document_dir, gl_ctx)
    assert document.passes, "the shader files survive a broken document.json"
    assert metadata == {}
    document.release()
