"""Loading compiles nothing (066 D1): a pass compiles when something first needs its program.

Pins the three halves of the decision: `load_from_dir` leaves every pass's program None while
the tuned uniform VALUES still land in `uniform_values` (they never needed a program), a
never-compiled pass compiles itself the moment `get_active_uniforms()` is asked, and a BROKEN
source gets exactly one attempt — its errors stick until `invalidate()` re-arms the retry.
"""

import shutil
from pathlib import Path
from typing import Any

import moderngl
import pytest

from shaderbox.core import ENGINE_DRIVEN_UNIFORMS, Canvas
from shaderbox.pass_graph import evaluation_order
from shaderbox.paths import shader_lib_root
from shaderbox.shader_lib import ShaderLibIndex, set_active
from shaderbox.ui_models import load_document_from_dir

_EXAMPLES = (
    Path(__file__).resolve().parent.parent / "shaderbox/resources/document_examples"
)
# The tuned single-pass example and the five-pass bloom chain.
_TUNED = _EXAMPLES / "f90f5ff9-29c6-4bcf-aee7-090f20542353"
_BLOOM = _EXAMPLES / "1c4f8a20-7b6e-4d31-9a55-2f0e6b8c31d4"


@pytest.fixture(scope="module")
def gl() -> moderngl.Context:
    ctx = moderngl.create_standalone_context(require=460)
    set_active(ShaderLibIndex.build(shader_lib_root()))
    return ctx


def test_load_compiles_no_pass(gl: moderngl.Context, tmp_path: Path) -> None:
    document_dir = tmp_path / "document"
    shutil.copytree(_BLOOM, document_dir)
    document = load_document_from_dir(document_dir).document
    assert len(document.passes) == 5
    for render_pass in document.passes.values():
        assert render_pass.program is None
        assert render_pass.compile_unit.error_raw == ""


def test_uniform_values_load_without_a_program(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    document_dir = tmp_path / "document"
    shutil.copytree(_TUNED, document_dir)
    render_pass = load_document_from_dir(document_dir).document.render_pass
    assert render_pass.program is None
    assert render_pass.uniform_values, (
        "the example ships tuned values; they must survive load"
    )


def test_get_active_uniforms_compiles_on_demand(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    document_dir = tmp_path / "document"
    shutil.copytree(_TUNED, document_dir)
    render_pass = load_document_from_dir(document_dir).document.render_pass
    assert render_pass.program is None
    uniforms = render_pass.get_active_uniforms()
    assert render_pass.program is not None
    assert "u_zoomout" in {u.name for u in uniforms}
    # Seeding rides the lazy compile: every returned uniform must have a value, or a
    # consumer that indexes uniform_values (the panel's row loop) crashes on a pass that
    # compiled here but never rendered.
    for uniform in uniforms:
        if uniform.name not in ENGINE_DRIVEN_UNIFORMS:
            assert uniform.name in render_pass.uniform_values, uniform.name


def test_a_foreign_canvas_render_leaves_first_render_pending(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    # A probe/export renders into its own canvas and must not consume the live loop's
    # first-render budget — the document's own canvases (what the grid tile shows) are
    # still unwritten, and a consumed budget would leave the tile black for the session.
    document_dir = tmp_path / "document"
    shutil.copytree(_TUNED, document_dir)
    document = load_document_from_dir(document_dir).document
    foreign = Canvas(gl=gl, size=(8, 8))
    document.render(canvas=foreign)
    assert not document.first_render_done
    document.render()
    assert document.first_render_done
    foreign.release()


def test_a_broken_source_is_attempted_once(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    document_dir = tmp_path / "document"
    shutil.copytree(_TUNED, document_dir)
    render_pass = load_document_from_dir(document_dir).document.render_pass
    render_pass.release_program("this is not glsl")

    assert render_pass.get_active_uniforms() == []
    assert render_pass.compile_unit.error_raw
    unit_after_first_attempt = render_pass.compile_unit
    # A second ask must not retry: a retry would replace compile_unit with a fresh object.
    assert render_pass.get_active_uniforms() == []
    assert render_pass.compile_unit is unit_after_first_attempt

    # invalidate() (a source or lib change) re-arms the compile.
    render_pass.release_program((document_dir / "passes/main.frag.glsl").read_text())
    assert render_pass.get_active_uniforms() != []
    assert render_pass.program is not None


# ----------------------------------------------------------------
# The first-render sweep (069 W-C): every pass draws once, one pass per frame, and the steady
# state still draws only the output chain.


def _bloom_with_output(tmp_path: Path, output: str) -> Any:
    # `blur` reaches scene+bright+blur, leaving `trail` and `composite` off the output chain —
    # the tiles finding #36 saw as black rectangles until they were clicked.
    document_dir = tmp_path / "document"
    shutil.copytree(_BLOOM, document_dir)
    document = load_document_from_dir(document_dir).document
    document.graph.output = output
    return document


def _count_draws(document: Any) -> dict[str, int]:
    # Per-pass draw counter: the real Pass.render still runs, so the picture is unchanged.
    counts: dict[str, int] = dict.fromkeys(document.passes, 0)
    for name, render_pass in document.passes.items():
        real = render_pass.render

        def counted(
            *args: Any, _name: str = name, _real: Any = real, **kwargs: Any
        ) -> None:
            counts[_name] += 1
            _real(*args, **kwargs)

        render_pass.render = counted
    return counts


def _sweep_frame(document: Any, frame: int) -> str | None:
    # What ui.py's document-render block does: the output chain, then at most one never-drawn
    # pass and its own ancestor chain.
    document.begin_frame(frame)
    document.render()
    pending = next(
        (
            name
            for name, render_pass in document.passes.items()
            if not render_pass.first_render_done
        ),
        None,
    )
    if pending is not None:
        document.render(target=pending)
    return pending


def test_every_pass_renders_once_within_n_frames(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    document = _bloom_with_output(tmp_path, "blur")
    elected: list[str] = []
    for frame in range(len(document.passes)):
        pending = _sweep_frame(document, frame)
        if pending is not None:
            elected.append(pending)
    assert all(p.first_render_done for p in document.passes.values()), {
        name: p.first_render_done for name, p in document.passes.items()
    }
    assert len(elected) == len(set(elected)), f"a pass was elected twice: {elected}"


def test_a_broken_off_chain_pass_is_stamped_on_attempt(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    # The stamp is written by Document.render on ATTEMPT. Inside Pass.render it would be
    # skipped by the early return a failed compile takes, so the sweep would re-elect this pass
    # every frame and never drain.
    document = _bloom_with_output(tmp_path, "blur")
    document.passes["composite"].release_program("this is not glsl")
    elected: list[str] = []
    for frame in range(len(document.passes)):
        pending = _sweep_frame(document, frame)
        if pending is not None:
            elected.append(pending)
    assert document.passes["composite"].first_render_done
    assert all(p.first_render_done for p in document.passes.values())
    assert len(elected) == len(set(elected)), f"a pass was elected twice: {elected}"


def test_the_steady_state_draws_only_the_output_chain(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    document = _bloom_with_output(tmp_path, "blur")
    for frame in range(len(document.passes)):
        _sweep_frame(document, frame)

    counts = _count_draws(document)
    chain = set(evaluation_order(document.graph, "blur"))
    for frame in range(len(document.passes), len(document.passes) + 3):
        before = dict(counts)
        elected = _sweep_frame(document, frame)
        assert elected is None, f"frame {frame} re-elected {elected}"
        drawn = {name for name in counts if counts[name] > before[name]}
        assert drawn == chain, f"frame {frame}: {drawn} != {chain}"
        for name in chain:
            assert counts[name] - before[name] == 1, (
                f"{name} drew twice in frame {frame}"
            )

    # A target render inside a settled frame redraws only what has not drawn yet: the elected
    # pass's ancestors are the ones the output render already stamped, so the skip eats them.
    before = dict(counts)
    document.render(target="composite")
    assert {name for name in counts if counts[name] > before[name]} == {
        "composite",
        "trail",
    }
    for name in chain:
        assert counts[name] == before[name], f"{name} drew twice in one frame"


def test_two_output_renders_in_one_frame_both_draw(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    # ui.py renders the output chain twice per frame — once into the preview canvas, once into
    # each pass's own. An unscoped skip would make the second one draw nothing and every strip
    # thumbnail would go black.
    document = _bloom_with_output(tmp_path, "blur")
    counts = _count_draws(document)
    foreign = Canvas(gl=gl, size=(8, 8))
    document.begin_frame(0)
    document.render(canvas=foreign)
    before = dict(counts)
    document.render()
    chain = set(evaluation_order(document.graph, "blur"))
    drawn = {name for name in counts if counts[name] > before[name]}
    assert drawn == chain, f"the own-canvas render drew {drawn}, not the chain"
    # The target render draws composite plus the ancestors the chain has not already stamped
    # (trail); scene/bright/blur drew twice above and the skip leaves them alone.
    before = dict(counts)
    document.render(target="composite")
    assert {name for name in counts if counts[name] > before[name]} == {
        "composite",
        "trail",
    }
    foreign.release()


def test_a_target_render_does_not_complete_the_document_first_render(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    document = _bloom_with_output(tmp_path, "blur")
    document.render(target="composite")
    assert not document.first_render_done
    document.render()
    assert document.first_render_done


def test_the_skip_does_not_fire_without_a_frame_counter(
    gl: moderngl.Context, tmp_path: Path
) -> None:
    # `_frame` is -1 until begin_frame runs, and the examples popup never calls it. Without the
    # `>= 0` conjunct `-1 == -1` reads as "already drawn this frame" and freezes the document.
    document = _bloom_with_output(tmp_path, "blur")
    counts = _count_draws(document)
    assert document._frame == -1
    document.render(target="composite")
    document.render(target="composite")
    assert counts["composite"] == 2
