"""Editing any pass file on disk reloads that pass (065 stage 6).

The watcher used to poll ONE shader per document and treat `sources[0]` as its root, returning
early. With N pass files that misses every pass but the output's — you would edit a file, see
nothing happen, and have no error to explain it. These drive the real watcher against real files.
"""

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import moderngl
import pytest

from shaderbox.core import Pass
from shaderbox.document import DEFAULT_PASS_NAME, Document
from shaderbox.media import texture_to_rgba8
from shaderbox.pass_graph import PassEntry, PassGraph
from shaderbox.paths import PASSES_DIR_NAME, pass_shader_name
from shaderbox.ui_models import UIDocument, load_document_from_dir
from shaderbox.watch import reload_document_if_changed

_CONST = """#version 460 core
in vec2 vs_uv;
out vec4 fs_color;
void main() { fs_color = vec4(%s, 0.0, 0.0, 1.0); }
"""

_HALVE = """#version 460 core
in vec2 vs_uv;
uniform sampler2D u_src;
out vec4 fs_color;
void main() { fs_color = texture(u_src, vs_uv) * %s; }
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


class _FakeApp:
    """The two things the watcher touches on `app`, and a record of what it was told."""

    def __init__(self) -> None:
        self.editor_sessions: dict[Path, Any] = {}
        self.synced: list[tuple[Path, str]] = []

    def sync_editor_from_disk(self, path: Path, source: str) -> None:
        self.synced.append((path, source))


def _chain_document(gl: moderngl.Context, root: Path) -> UIDocument:
    document_dir = root / "doc"
    passes = document_dir / PASSES_DIR_NAME
    passes.mkdir(parents=True)
    (passes / pass_shader_name("a")).write_text(_CONST % "0.8")
    (passes / pass_shader_name("b")).write_text(_HALVE % "0.5")
    (document_dir / "document.json").write_text(
        '{"canvas_size": [8, 8], "uniforms": {}, "ui_state": {}}'
    )
    graph = PassGraph(
        output="b", passes={"a": PassEntry(), "b": PassEntry(inputs={"u_src": "a"})}
    )
    (document_dir / "graph.json").write_text(graph.model_dump_json())
    ui_document = load_document_from_dir(document_dir)
    ui_document.document.render(u_time=0.0)
    return ui_document


def _red(document: Document) -> int:
    # Through texture_to_rgba8: a graph entry's default target is f2 (D9), and a raw byte read of
    # a float target truncates the buffer to a plausible wrong value.
    return int(texture_to_rgba8(document.render_pass.canvas.texture)[0][0][0])


def test_editing_a_non_output_pass_reloads_it(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # The one the old watcher could not see: `a` feeds `b`, and only `b` is the output.
    ui_document = _chain_document(gl_ctx, tmp_path)
    app = _FakeApp()
    before = _red(ui_document.document)

    source_a = ui_document.document.passes["a"].source.path
    os.utime(source_a, (0, 0))  # make the mtime differ for sure
    source_a.write_text(_CONST % "0.4")
    reload_document_if_changed(app, "doc", ui_document)  # type: ignore[arg-type]
    ui_document.document.render(u_time=0.0)

    assert _red(ui_document.document) != before, "editing pass 'a' changed nothing"
    assert _red(ui_document.document) == pytest.approx(51, abs=2)  # 0.4 * 0.5
    assert [p for p, _ in app.synced] == [source_a]
    ui_document.document.release()


def test_editing_the_output_pass_still_reloads(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    ui_document = _chain_document(gl_ctx, tmp_path)
    app = _FakeApp()
    source_b = ui_document.document.passes["b"].source.path
    os.utime(source_b, (0, 0))
    source_b.write_text(_HALVE % "0.25")
    reload_document_if_changed(app, "doc", ui_document)  # type: ignore[arg-type]
    ui_document.document.render(u_time=0.0)
    assert _red(ui_document.document) == pytest.approx(51, abs=2)  # 0.8 * 0.25
    assert [p for p, _ in app.synced] == [source_b]
    ui_document.document.release()


def test_an_unchanged_document_reloads_nothing(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    ui_document = _chain_document(gl_ctx, tmp_path)
    app = _FakeApp()
    reload_document_if_changed(app, "doc", ui_document)  # type: ignore[arg-type]
    assert app.synced == []
    ui_document.document.release()


def test_the_watcher_identifies_a_root_by_path_not_by_index(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # The old rule was "sources[0] is the root". The position happens to be right, but a
    # positional rule is wrong-by-construction the moment anything reorders that list — so the
    # watcher matches on the pass's own source path.
    ui_document = _chain_document(gl_ctx, tmp_path)
    render_pass = ui_document.document.passes["a"]
    assert render_pass.compile_unit.sources[0].path == render_pass.source.path
    ui_document.document.release()


def test_a_single_pass_document_still_hot_reloads(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    document_dir = tmp_path / "solo"
    passes = document_dir / PASSES_DIR_NAME
    passes.mkdir(parents=True)
    (passes / pass_shader_name(DEFAULT_PASS_NAME)).write_text(_CONST % "1.0")
    (document_dir / "document.json").write_text(
        '{"canvas_size": [8, 8], "uniforms": {}, "ui_state": {}}'
    )
    ui_document = load_document_from_dir(document_dir)
    ui_document.document.render(u_time=0.0)
    assert _red(ui_document.document) == 255

    app = _FakeApp()
    source = ui_document.document.render_pass.source.path
    os.utime(source, (0, 0))
    source.write_text(_CONST % "0.0")
    reload_document_if_changed(app, "solo", ui_document)  # type: ignore[arg-type]
    ui_document.document.render(u_time=0.0)
    assert _red(ui_document.document) == 0
    ui_document.document.release()


def test_a_pass_keeps_its_own_program_across_a_siblings_reload(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # D8's structural half: reloading one pass must not drop another's program, or every edit
    # pays for a full document rebuild.
    ui_document = _chain_document(gl_ctx, tmp_path)
    other: Pass = ui_document.document.passes["b"]
    program_before = other.program
    source_a = ui_document.document.passes["a"].source.path
    os.utime(source_a, (0, 0))
    source_a.write_text(_CONST % "0.2")
    reload_document_if_changed(_FakeApp(), "doc", ui_document)  # type: ignore[arg-type]
    assert other.program is program_before
    ui_document.document.release()
