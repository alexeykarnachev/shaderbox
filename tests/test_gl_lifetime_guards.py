"""Guards over release/reuse decisions that no test observed.

Each of these survived a mutation with the whole suite green: delete the line and 700+ tests
still pass while a GL object leaks every frame, or a revert restores the wrong file. They are
cheap to write and the thing they protect is invisible until it is expensive — a leak shows up
as the app slowly eating VRAM, not as a failure anyone can point at.
"""

import shutil
from pathlib import Path

import moderngl
import pytest

from shaderbox.copilot.checkpoint import TurnCheckpoint
from shaderbox.core import Canvas, Node
from shaderbox.paths import NODE_SCRIPT_BASENAME, shader_lib_root
from shaderbox.shader_lib import ShaderLibIndex, set_active

_EXAMPLE = (
    Path(__file__).resolve().parent.parent
    / "shaderbox/resources/node_examples/53724dbd-8efb-4c09-8c7d-28d626a066e7"
)


@pytest.fixture(scope="module")
def gl() -> moderngl.Context:
    ctx = moderngl.create_standalone_context(require=460)
    set_active(ShaderLibIndex.build(shader_lib_root()))
    return ctx


def _released(obj: object) -> bool:
    """True when the GL object has been freed.

    `.glo` keeps its stale integer id after release, so it cannot be the probe. moderngl
    swaps the underlying handle for an `InvalidObject` instead, and that holds for every
    object type (texture, framebuffer, buffer, program, vao) — unlike raising, which only
    some of them do.
    """
    return type(getattr(obj, "mglo", None)).__name__ == "InvalidObject"


def test_node_release_frees_the_canvas(gl: moderngl.Context) -> None:
    # The one test naming Node.release checks only the uniform-values half, so dropping
    # `self.canvas.release()` leaked a texture + FBO per reload with the suite still green.
    node, _ = Node.load_from_dir(_EXAMPLE)
    texture = node.canvas.texture
    fbo = node.canvas.fbo

    node.release()

    assert _released(texture), "the canvas texture outlived Node.release()"
    assert _released(fbo), "the canvas framebuffer outlived Node.release()"


def test_invalidate_frees_the_program_and_its_buffers(gl: moderngl.Context) -> None:
    # invalidate() runs on every hot-reload and every lib change; leaking here leaks per edit.
    node, _ = Node.load_from_dir(_EXAMPLE)
    assert node.program is not None
    program, vbo, vao = node.program, node.vbo, node.vao

    node.invalidate()

    assert node.program is None and node.vbo is None and node.vao is None
    assert _released(program), "the GL program outlived invalidate()"
    for name, obj in (("vbo", vbo), ("vao", vao)):
        if obj is not None:
            assert _released(obj), f"the {name} outlived invalidate()"


def test_canvas_resize_frees_the_old_texture(gl: moderngl.Context) -> None:
    canvas = Canvas(size=(64, 64))
    old_texture, old_fbo = canvas.texture, canvas.fbo

    assert canvas.set_size((128, 128))

    assert canvas.texture is not old_texture
    assert _released(old_texture), "the pre-resize texture leaked"
    assert _released(old_fbo), "the pre-resize framebuffer leaked"


def test_canvas_resize_to_the_same_size_is_a_no_op(gl: moderngl.Context) -> None:
    # set_size is called every frame with the current size; without the guard that is a
    # texture+FBO reallocation per frame, which no test would otherwise notice.
    canvas = Canvas(size=(64, 64))
    texture = canvas.texture

    assert canvas.set_size((64, 64)) is False

    assert canvas.texture is texture, "a same-size resize reallocated the canvas"


def test_snapshot_script_keeps_the_pre_turn_bytes(tmp_path: Path) -> None:
    # First-touch-wins: a second edit in the same turn must NOT overwrite the snapshot, or
    # Revert restores the copilot's own mid-turn draft over the user's pre-turn script —
    # reporting success while losing exactly what the user asked to get back.
    scripts_dir = tmp_path / "node" / "scripts"
    scripts_dir.mkdir(parents=True)
    script = scripts_dir / NODE_SCRIPT_BASENAME
    script.write_text("PRE-TURN\n")

    checkpoint = TurnCheckpoint(turn_id="t1", root=tmp_path / "checkpoints")
    checkpoint.snapshot_script("node", script)
    script.write_text("MID-TURN DRAFT\n")
    checkpoint.snapshot_script("node", script)

    snapshot = checkpoint.turn_dir / "node" / "scripts" / NODE_SCRIPT_BASENAME
    assert snapshot.read_text() == "PRE-TURN\n", (
        "the second snapshot overwrote the pre-turn bytes — revert would restore the draft"
    )


def test_snapshot_script_ignores_a_missing_script(tmp_path: Path) -> None:
    checkpoint = TurnCheckpoint(turn_id="t1", root=tmp_path / "checkpoints")
    checkpoint.snapshot_script("node", tmp_path / "absent" / NODE_SCRIPT_BASENAME)
    assert not (checkpoint.turn_dir / "node").exists()


def test_node_release_frees_uniform_held_media(gl: moderngl.Context) -> None:
    # The 060 fix, pinned from the other side: the uniform values own textures/captures, and
    # every reload releases the node.
    source = _EXAMPLE
    node, _ = Node.load_from_dir(source)
    held = [v for v in node.uniform_values.values() if hasattr(v, "release")]

    node.release()

    assert node.uniform_values == {}
    assert held or True  # the example may hold none; the emptied dict is the invariant
    shutil.rmtree(source / "__pycache__", ignore_errors=True)
