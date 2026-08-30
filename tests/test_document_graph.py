"""A document draws its passes in order, feeds each one's inputs, and accumulates feedback (065
stage 3).

These are the spec's engine checks 2-6 and 8-9. Each renders real pixels through a real GL context
rather than inspecting the plan: stage 1 already asserts the ORDER, so what is left to prove is
that the order is what actually draws and that a producer's texture reaches its consumer.

Needs a real GL context. On the display-less dev box use the EGL backend + the MESA version
overrides (set at process top, read at context creation); skips cleanly if no context is available.
"""

import os
from collections.abc import Callable, Iterator
from pathlib import Path

import moderngl
import pytest
from PIL import Image as PILImage

from shaderbox.constants import DEFAULT_FS_FILE_PATH
from shaderbox.core import Canvas, Pass
from shaderbox.document import DEFAULT_PASS_NAME, Document
from shaderbox.media import MediaDetails, texture_to_rgba8
from shaderbox.pass_graph import PassEntry, PassGraph, TargetConfig
from shaderbox.shader_source import ShaderSource

# Writes a constant so a consumer's arithmetic on it is unambiguous.
_CONST = """#version 460 core
in vec2 vs_uv;
out vec4 fs_color;
void main() { fs_color = vec4(%s, 0.0, 0.0, 1.0); }
"""

# Reads one input and halves it: the output proves the producer's texture arrived.
_HALVE = """#version 460 core
in vec2 vs_uv;
uniform sampler2D u_src;
out vec4 fs_color;
void main() { fs_color = vec4(texture(u_src, vs_uv).r * 0.5, 0.0, 0.0, 1.0); }
"""

# Sums two inputs, so a diamond's two branches are distinguishable in the result.
_SUM = """#version 460 core
in vec2 vs_uv;
uniform sampler2D u_a;
uniform sampler2D u_b;
out vec4 fs_color;
void main() {
    fs_color = vec4(texture(u_a, vs_uv).r + texture(u_b, vs_uv).r, 0.0, 0.0, 1.0);
}
"""

# Adds a fixed step to its own previous frame: the value advances once per FRAME.
_ACCUMULATE = """#version 460 core
in vec2 vs_uv;
uniform sampler2D u_prev;
out vec4 fs_color;
void main() { fs_color = vec4(texture(u_prev, vs_uv).r + 0.1, 0.0, 0.0, 1.0); }
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


def _document(
    gl: moderngl.Context,
    sources: dict[str, str],
    graph: PassGraph,
    size: tuple[int, int] = (8, 8),
) -> Document:
    doc = Document(gl=gl, canvas_size=size)
    doc.passes[DEFAULT_PASS_NAME].release()
    doc.passes = {}
    for name, src in sources.items():
        entry = graph.passes.get(name, PassEntry())
        render_pass = Pass(gl=gl, canvas_size=size, target=entry.target)
        render_pass.release_program(src)
        render_pass.compile()
        assert render_pass.compile_unit.errors == [], (
            f"{name}: {render_pass.compile_unit.errors}"
        )
        doc.passes[name] = render_pass
    doc.graph = graph
    return doc


def _red_of(canvas: Canvas) -> int:
    # Through texture_to_rgba8, not a raw byte read: a graph entry's default target is f2 (D9),
    # and reading a float target as uint8 truncates the buffer and yields a plausible wrong value.
    return int(texture_to_rgba8(canvas.texture)[0][0][0])


def _image_details(doc: Document, path: Path) -> MediaDetails:
    details = MediaDetails(is_video=False, duration=1.0)
    details.file_details.path = str(path)
    w, h = doc.render_pass.canvas.texture.size
    details.resolution_details.width = w
    details.resolution_details.height = h
    return details


def _red(doc: Document) -> int:
    return _red_of(doc.render_pass.canvas)


def test_a_two_pass_chain_shows_b_reads_a(gl_ctx: moderngl.Context) -> None:
    # Check 2. Falsifier: the output equals A's own image (0.8 -> 204), meaning B never sampled A.
    doc = _document(
        gl_ctx,
        {"a": _CONST % "0.8", "b": _HALVE},
        PassGraph(
            output="b",
            passes={"a": PassEntry(), "b": PassEntry(inputs={"u_src": "a"})},
        ),
    )
    doc.render(u_time=0.0)
    assert _red(doc) == pytest.approx(102, abs=2)  # 0.8 * 0.5 = 0.4 -> 102
    doc.release()


def test_a_diamond_feeds_both_branches_from_one_ancestor(
    gl_ctx: moderngl.Context,
) -> None:
    # Check 3's rendered half: base -> {left, right} -> out. base draws ONCE (stage 1 asserts the
    # order; this asserts the order is what draws), and both halves reach the sum.
    doc = _document(
        gl_ctx,
        {"base": _CONST % "0.6", "left": _HALVE, "right": _HALVE, "out": _SUM},
        PassGraph(
            output="out",
            passes={
                "base": PassEntry(),
                "left": PassEntry(inputs={"u_src": "base"}),
                "right": PassEntry(inputs={"u_src": "base"}),
                "out": PassEntry(inputs={"u_a": "left", "u_b": "right"}),
            },
        ),
    )
    doc.render(u_time=0.0)
    assert _red(doc) == pytest.approx(153, abs=2)  # (0.6*0.5) + (0.6*0.5) = 0.6 -> 153
    doc.release()


def test_a_feedback_pass_accumulates_once_per_frame(gl_ctx: moderngl.Context) -> None:
    # Check 4. The falsifier is a value advancing at 2x: the live loop renders a document TWICE
    # per frame, so a swap tied to the render CALL rather than the frame doubles the rate.
    doc = _document(
        gl_ctx,
        {"trail": _ACCUMULATE},
        PassGraph(
            output="trail",
            passes={"trail": PassEntry(inputs={"u_prev": "trail"})},
        ),
    )
    for _ in range(4):
        doc.begin_frame()
        doc.render(u_time=0.0)
        doc.render(
            u_time=0.0
        )  # the second draw of the same frame (preview + own canvas)
    # Four frames of +0.1 each, not eight.
    assert _red(doc) == pytest.approx(102, abs=3)
    doc.release()


def test_feedback_starts_black(gl_ctx: moderngl.Context) -> None:
    doc = _document(
        gl_ctx,
        {"trail": _ACCUMULATE},
        PassGraph(
            output="trail", passes={"trail": PassEntry(inputs={"u_prev": "trail"})}
        ),
    )
    doc.begin_frame()
    doc.render(u_time=0.0)
    assert _red(doc) == pytest.approx(26, abs=2)  # 0.0 + 0.1 -> 26
    doc.release()


def test_a_cycle_reports_per_pass_and_still_draws_the_output(
    gl_ctx: moderngl.Context,
) -> None:
    # Check 5: loud, per pass, and NOT a hang. The output still shows its own shader rather than
    # going blank, so a mis-wire is visible as an error rather than as a black preview.
    doc = _document(
        gl_ctx,
        {"a": _HALVE, "b": _HALVE},
        PassGraph(
            output="a",
            passes={
                "a": PassEntry(inputs={"u_src": "b"}),
                "b": PassEntry(inputs={"u_src": "a"}),
            },
        ),
    )
    doc.render(u_time=0.0)
    assert {e.pass_name for e in doc.graph_errors} == {"a", "b"}
    assert all("cycle" in e.message for e in doc.graph_errors)
    doc.release()


def test_an_unfilled_input_reads_black_and_the_document_renders(
    gl_ctx: moderngl.Context,
) -> None:
    # Check 6. Falsifier: an exception, or the DEFAULT IMAGE appearing instead of black.
    doc = _document(
        gl_ctx,
        {"blur": _HALVE},
        PassGraph(output="blur", passes={"blur": PassEntry(inputs={"u_src": "ghost"})}),
    )
    doc.render(u_time=0.0)
    assert _red(doc) == 0
    doc.release()


def test_only_the_passes_the_output_needs_are_drawn(gl_ctx: moderngl.Context) -> None:
    # A branch nothing reads costs nothing: `unused` never draws, so its canvas stays black
    # while the output's chain resolves.
    doc = _document(
        gl_ctx,
        {"a": _CONST % "1.0", "used": _HALVE, "unused": _HALVE},
        PassGraph(
            output="used",
            passes={
                "a": PassEntry(),
                "used": PassEntry(inputs={"u_src": "a"}),
                "unused": PassEntry(inputs={"u_src": "a"}),
            },
        ),
    )
    doc.render(u_time=0.0)
    assert _red(doc) == pytest.approx(128, abs=2)
    assert _red_of(doc.passes["unused"].canvas) == 0
    doc.release()


def test_the_output_pass_is_what_renders(gl_ctx: moderngl.Context) -> None:
    # Check 9: a document whose output is NOT the last-authored pass. Picking the wrong one gives
    # a visibly different image.
    graph = PassGraph(
        output="a",
        passes={"a": PassEntry(), "b": PassEntry()},
    )
    doc = _document(gl_ctx, {"a": _CONST % "0.25", "b": _CONST % "1.0"}, graph)
    doc.render(u_time=0.0)
    assert _red(doc) == pytest.approx(64, abs=2)
    doc.graph = graph.model_copy(update={"output": "b"})
    doc.render(u_time=0.0)
    assert _red(doc) == 255
    doc.release()


def test_an_external_canvas_overrides_only_the_output_target(
    gl_ctx: moderngl.Context,
) -> None:
    # The preview and the probe hand the document their own canvas. An intermediate pass must
    # still draw into ITS OWN target, since that is the texture its consumer samples — routing
    # every pass to the override would make each overwrite the last.
    doc = _document(
        gl_ctx,
        {"a": _CONST % "0.8", "b": _HALVE},
        PassGraph(
            output="b", passes={"a": PassEntry(), "b": PassEntry(inputs={"u_src": "a"})}
        ),
    )
    external = Canvas(gl=gl_ctx, size=(4, 4))
    doc.render(u_time=0.0, canvas=external)
    assert _red_of(external) == pytest.approx(102, abs=2)
    assert _red_of(doc.passes["a"].canvas) == pytest.approx(204, abs=2)
    external.release()
    doc.release()


def test_a_pass_target_config_applies_to_its_own_canvas(
    gl_ctx: moderngl.Context,
) -> None:
    # D9: per-pass target configuration is what makes an accumulate chain work, so a pass built
    # from a graph entry gets that entry's format rather than the document's default.
    graph = PassGraph(
        output="out",
        passes={
            "out": PassEntry(target=TargetConfig(dtype="f2", scale=1.0, persist=True))
        },
    )
    doc = _document(gl_ctx, {"out": _CONST % "1.0"}, graph)
    assert doc.passes["out"].canvas.texture.dtype == "f2"
    doc.release()


def test_a_single_pass_document_still_renders_without_a_graph_edit(
    gl_ctx: moderngl.Context,
) -> None:
    # The ordinary case: a freshly constructed document has one pass, names it the output, and
    # draws without anyone having opened a panel.
    doc = Document(gl=gl_ctx, canvas_size=(8, 8))
    doc.passes[DEFAULT_PASS_NAME].release_program(_CONST % "1.0")
    doc.passes[DEFAULT_PASS_NAME].compile()
    doc.render(u_time=0.0)
    assert _red(doc) == 255
    assert doc.graph_errors == []
    doc.release()


def test_a_documents_source_lands_on_its_only_pass(gl_ctx: moderngl.Context) -> None:
    # `render_pass` resolves to the output, which for a one-pass document is that pass.
    doc = Document(gl=gl_ctx, source=ShaderSource.load(DEFAULT_FS_FILE_PATH))
    assert list(doc.passes) == [DEFAULT_PASS_NAME]
    assert doc.render_pass is doc.passes[DEFAULT_PASS_NAME]
    assert doc.render_pass.source.path == DEFAULT_FS_FILE_PATH
    doc.release()


def test_two_exports_of_a_feedback_document_are_identical(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # Check 8, with live frames rendered in between. D10: a feedback target is export state the
    # same way a stateful script is, so an export starts cold. Falsifier: the second file differs,
    # meaning the export inherited however long the app happened to have been running.
    doc = _document(
        gl_ctx,
        {"trail": _ACCUMULATE},
        PassGraph(
            output="trail", passes={"trail": PassEntry(inputs={"u_prev": "trail"})}
        ),
    )
    first = tmp_path / "a.png"
    second = tmp_path / "b.png"
    doc.render_media(_image_details(doc, first))
    for _ in range(30):  # warm the live history between the two exports
        doc.begin_frame()
        doc.render(u_time=0.0)
    doc.render_media(_image_details(doc, second))
    assert first.read_bytes() == second.read_bytes()
    doc.release()


def test_an_export_does_not_inherit_the_live_history(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # The falsifier for the check above: without the cold start both files would still MATCH each
    # other while both being wrong, so pin the value too. One export frame => one accumulate step.
    doc = _document(
        gl_ctx,
        {"trail": _ACCUMULATE},
        PassGraph(
            output="trail", passes={"trail": PassEntry(inputs={"u_prev": "trail"})}
        ),
    )
    for _ in range(30):
        doc.begin_frame()
        doc.render(u_time=0.0)
    warm = _red(doc)
    assert warm > 200, "the live history did not warm"
    out = tmp_path / "cold.png"
    doc.render_media(_image_details(doc, out))
    exported = PILImage.open(out).convert("RGBA").getpixel((0, 0))[0]
    assert exported == pytest.approx(26, abs=3), (
        f"export inherited live state (got {exported}, live is {warm})"
    )
    doc.release()


def test_begin_frame_is_idempotent_within_one_frame(gl_ctx: moderngl.Context) -> None:
    # Identity, not call count: two callers advancing the same frame must not advance history
    # twice. Without this the correctness of feedback depends on there being exactly one call
    # site, which no type or test can enforce as the app grows.
    doc = _document(
        gl_ctx,
        {"trail": _ACCUMULATE},
        PassGraph(
            output="trail", passes={"trail": PassEntry(inputs={"u_prev": "trail"})}
        ),
    )
    for frame in range(4):
        doc.begin_frame(frame)
        doc.begin_frame(frame)  # a second caller for the SAME frame
        doc.render(u_time=0.0)
    assert _red(doc) == pytest.approx(102, abs=3)  # four steps, not eight
    doc.release()


def test_begin_frame_without_a_number_advances_every_call(
    gl_ctx: moderngl.Context,
) -> None:
    # The export loops own their own sequence and have no frame counter to pass, so the
    # numberless form must still step once per call.
    doc = _document(
        gl_ctx,
        {"trail": _ACCUMULATE},
        PassGraph(
            output="trail", passes={"trail": PassEntry(inputs={"u_prev": "trail"})}
        ),
    )
    for _ in range(3):
        doc.begin_frame()
        doc.render(u_time=0.0)
    assert _red(doc) == pytest.approx(77, abs=3)
    doc.release()


def test_editing_one_pass_recompiles_only_that_pass(
    gl_ctx: moderngl.Context, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Check 7 (D8). The observable is a COUNT, not an eyeball: recompiling every pass on every
    # edit renders the identical picture, and only shows up as a hitch — measured at ~1.3 ms per
    # program, so sixteen passes is a ~21 ms stall on each keystroke-save.
    doc = _document(
        gl_ctx,
        {"a": _CONST % "0.8", "b": _HALVE},
        PassGraph(
            output="b", passes={"a": PassEntry(), "b": PassEntry(inputs={"u_src": "a"})}
        ),
    )
    doc.render(u_time=0.0)

    compiles: dict[str, int] = dict.fromkeys(doc.passes, 0)

    def counting(name: str, render_pass: Pass) -> Callable[[], None]:
        original = render_pass.compile

        def wrapped() -> None:
            compiles[name] += 1
            original()

        return wrapped

    for name, render_pass in doc.passes.items():
        monkeypatch.setattr(render_pass, "compile", counting(name, render_pass))

    doc.passes["b"].release_program(_HALVE.replace("0.5", "0.25"))
    doc.render(u_time=0.0)
    assert compiles == {"a": 0, "b": 1}, f"an edit to b recompiled {compiles}"
    doc.release()


def test_a_pass_scale_shrinks_its_target(gl_ctx: moderngl.Context) -> None:
    # `scale` was DEFINED on TargetConfig and read by nothing — a knob the panel could set, the
    # file could persist, and the renderer ignored. The document applies it, since it is the
    # thing that owns the canvas size.
    graph = PassGraph(
        output="out",
        passes={
            "half": PassEntry(target=TargetConfig(scale=0.5)),
            "out": PassEntry(inputs={"u_src": "half"}),
        },
    )
    doc = _document(
        gl_ctx, {"half": _CONST % "1.0", "out": _HALVE}, graph, size=(64, 64)
    )
    doc.render(u_time=0.0)
    assert doc.passes["half"].canvas.texture.size == (32, 32)
    # The OUTPUT keeps full size whatever its own scale says: it is what the preview and the
    # export read, and a half-size output would silently halve every render.
    assert doc.passes["out"].canvas.texture.size == (64, 64)
    assert _red(doc) == pytest.approx(128, abs=2)  # and the chain still resolves
    doc.release()


def test_a_scaled_pass_keeps_its_size_across_frames(gl_ctx: moderngl.Context) -> None:
    # Falsifier for a resize that runs every frame: a set_size call reallocates, so a canvas that
    # is re-sized each frame drops a feedback pass's history silently.
    graph = PassGraph(
        output="out",
        passes={
            "half": PassEntry(target=TargetConfig(scale=0.5)),
            "out": PassEntry(inputs={"u_src": "half"}),
        },
    )
    doc = _document(
        gl_ctx, {"half": _CONST % "1.0", "out": _HALVE}, graph, size=(64, 64)
    )
    doc.render(u_time=0.0)
    texture = doc.passes["half"].canvas.texture
    doc.render(u_time=1.0)
    assert doc.passes["half"].canvas.texture is texture, "the target was reallocated"
    doc.release()


def test_a_resize_moves_every_pass_together(gl_ctx: moderngl.Context) -> None:
    # `canvas_size` is what every non-output pass scales FROM, so a resize that only touches the
    # output canvas leaves the rest of the graph sizing off the old dimensions. The falsifier is
    # the field, not the textures: render() re-derives the others from it every frame, which is
    # exactly why a stale field is silent.
    graph = PassGraph(
        output="out",
        passes={
            "half": PassEntry(target=TargetConfig(scale=0.5)),
            "full": PassEntry(),
            "out": PassEntry(inputs={"u_a": "half", "u_b": "full"}),
        },
    )
    doc = _document(
        gl_ctx,
        {"half": _CONST % "1.0", "full": _CONST % "0.5", "out": _SUM},
        graph,
        size=(64, 64),
    )
    doc.render(u_time=0.0)
    doc.set_canvas_size((32, 32))
    doc.render(u_time=0.0)
    assert doc.canvas_size == (32, 32)
    assert doc.passes["out"].canvas.texture.size == (32, 32)
    assert doc.passes["full"].canvas.texture.size == (32, 32)
    assert doc.passes["half"].canvas.texture.size == (16, 16)
    assert _red(doc) == pytest.approx(
        255, abs=2
    )  # 1.0 + 0.5, tonemapped by nothing -> clamps
    doc.release()
