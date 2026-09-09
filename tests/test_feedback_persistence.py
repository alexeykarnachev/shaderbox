"""A feedback pass continues from where it was across a restart (feature 089, W-C).

`Document._feedback` starts empty on every load, so before this a self-reading pass came back
black on every app start -- the accumulated canvas the maintainer had drawn into was gone. Now
`UIDocument.save` writes each history's newest frame as `feedback/<pass>.bin` and
`Document.load_from_dir` seeds the history back from it.

Raw bytes throughout, never `texture_to_rgba8`: the state being kept is an accumulator whose
values sit outside [0, 1] on an `f2` target, which a tonemapped 8-bit round trip would clamp and
quantise -- so byte equality against the original's next frame is the assertion, and it is only
meaningful on the raw read.

The fixture is the bloom-chain shape rather than a single pass: the feedback pass sits at
`scale: 0.5` and is NOT the output, which is the case a size read off the live canvas gets wrong
(at load every canvas is at the document's full size; `scale` applies inside `render`).
"""

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import moderngl
import pytest

from shaderbox.document import Document
from shaderbox.paths import (
    DOCUMENT_JSON_BASENAME,
    FEEDBACK_DIR_NAME,
    GRAPH_JSON_BASENAME,
    PASSES_DIR_NAME,
    pass_shader_name,
    shader_lib_root,
)
from shaderbox.shader_lib import ShaderLibIndex, set_active
from shaderbox.ui_models import UIDocument, load_document_from_dir

# Adds a fixed step to its own previous frame, so the value advances once per FRAME and two
# documents that started at different points can never agree by accident.
_ACCUMULATE = """#version 460 core
in vec2 vs_uv;
uniform sampler2D u_prev;
out vec4 fs_color;
void main() { fs_color = vec4(texture(u_prev, vs_uv).r + 0.1, 0.0, 0.0, 1.0); }
"""

# The output: reads the feedback pass, so the graph has a real edge and the feedback pass is not
# the output (whose target is always the full canvas size).
_SHOW = """#version 460 core
in vec2 vs_uv;
uniform sampler2D u_trail;
out vec4 fs_color;
void main() { fs_color = texture(u_trail, vs_uv); }
"""

_BROKEN = """#version 460 core
this is not glsl
"""

_CANVAS = (8, 8)
_SCALED = (4, 4)


@pytest.fixture(scope="module")
def gl_ctx() -> Iterator[moderngl.Context]:
    # Default-backend like every other GL module's fixture (see test_document_graph.py).
    try:
        context = moderngl.create_standalone_context()
    except Exception as e:
        pytest.skip(f"no standalone GL context available: {e}")
    set_active(ShaderLibIndex.build(shader_lib_root()))
    yield context
    context.release()


def _write_document(dir: Path, dtype: str, trail_source: str = _ACCUMULATE) -> Path:
    """A two-pass document on disk: `trail` feeds back at scale 0.5, `show` is the output."""
    passes = dir / PASSES_DIR_NAME
    passes.mkdir(parents=True, exist_ok=True)
    (passes / pass_shader_name("trail")).write_text(trail_source)
    (passes / pass_shader_name("show")).write_text(_SHOW)
    graph = {
        "output": "show",
        "passes": {
            "trail": {"target": {"scale": 0.5, "dtype": dtype}},
            "show": {"target": {"dtype": dtype}},
        },
    }
    (dir / GRAPH_JSON_BASENAME).write_text(json.dumps(graph))
    (dir / DOCUMENT_JSON_BASENAME).write_text(
        json.dumps({"canvas_size": list(_CANVAS), "uniforms": {}, "ui_state": {}})
    )
    return dir


def _advance(ui_document: UIDocument, frames: range) -> None:
    for frame in frames:
        ui_document.document.begin_frame(frame)
        ui_document.document.render(u_time=float(frame))


def _trail_bytes(document: Document) -> bytes:
    return document.passes["trail"].canvas.texture.read()


def _metadata(dir: Path) -> dict[str, Any]:
    with (dir / DOCUMENT_JSON_BASENAME).open() as f:
        return json.load(f)


# --- V5: the round trip -------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["f1", "f2", "f4"])
def test_a_feedback_pass_resumes_where_it_stopped(
    gl_ctx: moderngl.Context, tmp_path: Path, dtype: str
) -> None:
    original_dir = _write_document(tmp_path / "original", dtype)
    original = load_document_from_dir(original_dir)
    _advance(original, range(4))
    original.save(original_dir.parent, original_dir.name)

    reloaded = load_document_from_dir(original_dir)
    seed = reloaded.document._feedback.get("trail")
    assert seed is not None, "the save wrote no history for the scaled feedback pass"
    assert seed.texture.size == _SCALED, (
        "the history was allocated at the live canvas's size, not the graph's"
    )

    # The seed must survive the first frame boundary: a swap here puts it in the LIVE slot,
    # where the first draw overwrites it and the whole round trip reads black.
    reloaded.document.begin_frame(0)
    assert reloaded.document._feedback["trail"] is seed, (
        "begin_frame swapped a never-drawn pass -- the seed left the history slot"
    )
    reloaded.document.render(u_time=0.0)

    # Frame 4 of the original is what the reloaded document's first frame must reproduce.
    _advance(original, range(4, 5))
    assert _trail_bytes(reloaded.document) == _trail_bytes(original.document), (
        "the reloaded document's first frame is not the original's next frame"
    )

    cold_dir = _write_document(tmp_path / "cold", dtype)
    cold = load_document_from_dir(cold_dir)
    _advance(cold, range(1))
    assert _trail_bytes(reloaded.document) != _trail_bytes(cold.document), (
        "the reloaded document matches a cold start -- the seed did nothing"
    )


# --- V7: the write and the sweep ----------------------------------------------------------


def test_a_save_writes_the_newest_frame_and_its_block(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    dir = _write_document(tmp_path / "doc", "f2")
    ui_document = load_document_from_dir(dir)
    _advance(ui_document, range(3))
    ui_document.save(dir.parent, dir.name)

    row = _metadata(dir)["feedback"]["trail"]
    assert row == {
        "file_path": f"{FEEDBACK_DIR_NAME}/trail.bin",
        "size": list(_SCALED),
        "components": 4,
        "dtype": "f2",
    }
    written = (dir / row["file_path"]).read_bytes()
    assert written == _trail_bytes(ui_document.document), (
        "the file holds something other than the pass's newest frame"
    )


def test_a_load_then_save_with_no_render_writes_the_seed(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # The duplicate path: duplicate_document saves the source, loads it, saves the copy -- with
    # no render in between. Without `newest_frame`'s history branch the copy persists the live
    # canvas, which at that point is black.
    source_dir = _write_document(tmp_path / "source", "f2")
    source = load_document_from_dir(source_dir)
    _advance(source, range(3))
    source.save(source_dir.parent, source_dir.name)
    seeded_bytes = (source_dir / FEEDBACK_DIR_NAME / "trail.bin").read_bytes()

    copy_dir = tmp_path / "copy"
    shutil.copytree(source_dir, copy_dir)
    copy = load_document_from_dir(copy_dir)
    copy.save(copy_dir.parent, copy_dir.name)

    assert (copy_dir / FEEDBACK_DIR_NAME / "trail.bin").read_bytes() == seeded_bytes, (
        "the duplicate persisted its blank live canvas instead of the seed it loaded"
    )


def test_a_broken_source_still_carries_its_history(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # The write sits outside the `if live:` guard the asset sweeps use: a document whose every
    # pass fails to compile still holds the history it loaded and must carry it forward, the way
    # its uniform rows are carried forward from disk.
    source_dir = _write_document(tmp_path / "source", "f2")
    source = load_document_from_dir(source_dir)
    _advance(source, range(3))
    source.save(source_dir.parent, source_dir.name)
    seeded_bytes = (source_dir / FEEDBACK_DIR_NAME / "trail.bin").read_bytes()

    (source_dir / PASSES_DIR_NAME / pass_shader_name("trail")).write_text(_BROKEN)
    (source_dir / PASSES_DIR_NAME / pass_shader_name("show")).write_text(_BROKEN)
    broken = load_document_from_dir(source_dir)
    assert all(p.program is None for p in broken.document.passes.values()), (
        "the fixture must have no compiled pass for this to exercise the not-live branch"
    )
    broken.save(source_dir.parent, source_dir.name)

    assert (source_dir / FEEDBACK_DIR_NAME / "trail.bin").read_bytes() == seeded_bytes
    assert "trail" in _metadata(source_dir)["feedback"]


def test_a_reset_drops_the_file_and_the_block(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # Through `Document.reset` rather than `ProjectSession.reset_document`: the session half is
    # the script engine's re-init, and the document half -- `reset_feedback`, which empties
    # `_feedback` -- is the whole of what a save can see.
    dir = _write_document(tmp_path / "doc", "f2")
    ui_document = load_document_from_dir(dir)
    _advance(ui_document, range(3))
    ui_document.save(dir.parent, dir.name)
    assert (dir / FEEDBACK_DIR_NAME / "trail.bin").is_file()

    ui_document.document.reset()
    ui_document.save(dir.parent, dir.name)

    assert not (dir / FEEDBACK_DIR_NAME / "trail.bin").exists(), (
        "the sweep left a frame on disk for a pass with no history"
    )
    assert _metadata(dir)["feedback"] == {}


def test_a_copilot_snapshot_carries_the_feedback_dir(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # The checkpoint's contract: serialize the LIVE object, restore by reload-and-replace. The
    # snapshot is a full `UIDocument.save` with rebind=False, so it carries the canvas along with
    # the source -- and a revert of a copilot turn restores both.
    dir = _write_document(tmp_path / "doc", "f2")
    ui_document = load_document_from_dir(dir)
    _advance(ui_document, range(3))
    source_path_before = ui_document.document.passes["trail"].source.path

    snapshot = tmp_path / "checkpoints" / "turn_1" / "doc"
    snapshot.mkdir(parents=True)
    ui_document.save(snapshot.parent, snapshot.name, rebind=False)

    assert (snapshot / FEEDBACK_DIR_NAME / "trail.bin").read_bytes() == _trail_bytes(
        ui_document.document
    )
    assert ui_document.document.passes["trail"].source.path == source_path_before, (
        "the snapshot repointed the live pass into the checkpoint dir"
    )


# --- V8: the mismatch and the lifetime ----------------------------------------------------


def test_a_feedback_entry_that_does_not_match_is_ignored(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    dir = _write_document(tmp_path / "doc", "f2")
    ui_document = load_document_from_dir(dir)
    _advance(ui_document, range(3))
    ui_document.save(dir.parent, dir.name)

    metadata = _metadata(dir)
    metadata["feedback"]["trail"]["size"] = [16, 16]
    (dir / DOCUMENT_JSON_BASENAME).write_text(json.dumps(metadata))

    reloaded = load_document_from_dir(dir)

    assert "trail" not in reloaded.document._feedback, (
        "a mismatched entry was seeded -- the byte count would have raised on write"
    )
    reloaded.document.begin_frame(0)
    reloaded.document.render(u_time=0.0)
    assert reloaded.document.passes["trail"].program is not None, (
        "the document must still load and render with its history ignored"
    )


def test_a_malformed_feedback_block_costs_only_that_pass(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    dir = _write_document(tmp_path / "doc", "f2")
    ui_document = load_document_from_dir(dir)
    _advance(ui_document, range(3))
    ui_document.save(dir.parent, dir.name)

    metadata = _metadata(dir)
    metadata["feedback"]["trail"] = "not an object"
    (dir / DOCUMENT_JSON_BASENAME).write_text(json.dumps(metadata))

    reloaded = load_document_from_dir(dir)
    assert "trail" not in reloaded.document._feedback
    assert set(reloaded.document.passes) == {"trail", "show"}


def test_a_truncated_feedback_file_is_ignored(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # The branch the block's own `size` cannot reach: the loader allocates from the GRAPH, so a
    # tampered `size` field never reaches the allocation -- only the file's own byte count can
    # disagree with the texture, and `texture.write` raises on it.
    dir = _write_document(tmp_path / "doc", "f2")
    ui_document = load_document_from_dir(dir)
    _advance(ui_document, range(3))
    ui_document.save(dir.parent, dir.name)
    frame = dir / FEEDBACK_DIR_NAME / "trail.bin"
    frame.write_bytes(frame.read_bytes()[:-8])

    reloaded = load_document_from_dir(dir)

    assert "trail" not in reloaded.document._feedback
    reloaded.document.begin_frame(0)
    reloaded.document.render(u_time=0.0)
    assert reloaded.document.passes["trail"].program is not None


def test_a_missing_feedback_file_is_ignored(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    dir = _write_document(tmp_path / "doc", "f2")
    ui_document = load_document_from_dir(dir)
    _advance(ui_document, range(3))
    ui_document.save(dir.parent, dir.name)
    (dir / FEEDBACK_DIR_NAME / "trail.bin").unlink()

    reloaded = load_document_from_dir(dir)
    assert "trail" not in reloaded.document._feedback
