"""Regression: UIDocument.save must seed uniform_values for the live program, not rely on a prior
render() (the copilot create_document(source=...) path compiles then saves with no render in between).
Pre-fix this raised KeyError on the first non-engine uniform, or ValueError on a sampler (whose
GL default is an int texture-unit, not a usable value). Document.seed_uniform_values is the single home
for the per-type defaults; save calls it. These need a real GL context; skip if none is available."""

import contextlib
from collections.abc import Iterator
from pathlib import Path

import moderngl
import numpy as np
import pytest

from shaderbox.copilot.backend import _format_uniforms
from shaderbox.document import Document
from shaderbox.media import Image, is_default_image
from shaderbox.ui_models import UIDocument

_SCALAR_SRC = """#version 460 core
in vec2 vs_uv;
uniform float u_aspect;
uniform vec3 u_bg_color;
uniform float u_radius;
out vec4 fs_color;
void main() {
    vec2 p = vs_uv - 0.5;
    p.x *= u_aspect;
    float d = length(p) - u_radius;
    fs_color = vec4(u_bg_color * step(0.0, d), 1.0);
}
"""

_SAMPLER_SRC = """#version 460 core
in vec2 vs_uv;
uniform sampler2D u_image;
out vec4 fs_color;
void main() { fs_color = texture(u_image, vs_uv); }
"""


@pytest.fixture(scope="module")
def gl_ctx() -> Iterator[moderngl.Context]:
    try:
        ctx = moderngl.create_standalone_context()
    except Exception as e:
        pytest.skip(f"no standalone GL context available: {e}")
    yield ctx
    ctx.release()


def _document_from_source(gl: moderngl.Context, source: str) -> Document:
    # Mirror create_document's path: swap in source, compile, but DO NOT render.
    document = Document(gl=gl)
    document.render_pass.release_program(source)
    document.render_pass.compile()
    return document


def _teardown(document: Document) -> None:
    with contextlib.suppress(Exception):
        document.release()


def test_save_seeds_scalar_uniforms_without_render(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    document = _document_from_source(gl_ctx, _SCALAR_SRC)
    assert (
        "u_bg_color" not in document.render_pass.uniform_values
    )  # unseeded: no render happened
    ui_document = UIDocument(document=document, id="scalar")
    ui_document.save(tmp_path)  # pre-fix: KeyError: 'u_bg_color'
    reloaded, meta = Document.load_from_dir(tmp_path / "scalar", gl=gl_ctx)
    assert "u_bg_color" in meta["uniforms"]["main"]
    assert "u_radius" in meta["uniforms"]["main"]
    _teardown(document)
    _teardown(reloaded)


def test_save_skips_default_sampler_and_reload_reseeds(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # Feature 052: an UNBOUND sampler (still the shipped default) is NOT persisted — save skips it so
    # it never reads back as "bound" on reload (the round-2 round-trip fix). Load's seed re-establishes
    # the default. Falsifier: without the skip, `u_image` is in meta and reloads from media/ (path !=
    # default) -> would read as bound.
    document = _document_from_source(gl_ctx, _SAMPLER_SRC)
    ui_document = UIDocument(document=document, id="sampler")
    ui_document.save(tmp_path)  # must not raise
    assert not (tmp_path / "sampler" / "media" / "main" / "u_image.png").exists()
    reloaded, meta = Document.load_from_dir(tmp_path / "sampler", gl=gl_ctx)
    assert "u_image" not in meta["uniforms"]["main"]  # default sampler skipped
    assert is_default_image(
        reloaded.render_pass.uniform_values["u_image"]
    )  # re-seeded to default on load
    _teardown(document)
    _teardown(reloaded)


def test_save_persists_user_bound_sampler(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # A user-BOUND sampler (non-default) IS persisted and round-trips as bound.
    document = _document_from_source(gl_ctx, _SAMPLER_SRC)
    document.render_pass.seed_uniform_values()
    document.render_pass.uniform_values["u_image"] = Image(
        np.zeros((8, 8, 3), dtype=np.uint8)
    )  # non-default
    ui_document = UIDocument(document=document, id="bound")
    ui_document.save(tmp_path)
    assert (tmp_path / "bound" / "media" / "main" / "u_image.png").exists()
    reloaded, meta = Document.load_from_dir(tmp_path / "bound", gl=gl_ctx)
    assert "u_image" in meta["uniforms"]["main"]
    assert not is_default_image(reloaded.render_pass.uniform_values["u_image"])
    _teardown(document)
    _teardown(reloaded)


def test_sampler_awareness_row_default_vs_bound(gl_ctx: moderngl.Context) -> None:
    # Feature 052 slice 1: the working-set uniform row shows a sampler's binding, NOT a source path.
    document = _document_from_source(gl_ctx, _SAMPLER_SRC)
    document.render_pass.seed_uniform_values()
    default_rows = _format_uniforms(document.render_pass, set())
    assert any("u_image sampler2D <- (no media bound)" in r for r in default_rows)
    document.render_pass.uniform_values["u_image"] = Image(
        np.zeros((8, 8, 3), dtype=np.uint8)
    )
    bound_rows = _format_uniforms(document.render_pass, set())
    assert any("u_image sampler2D <- (8x8, image)" in r for r in bound_rows)
    # Corollary-1: no absolute path leaks into the row.
    assert all("/" not in r.split("<-")[1] for r in bound_rows if "u_image" in r)
    _teardown(document)


def test_seed_skips_engine_uniforms(gl_ctx: moderngl.Context) -> None:
    document = _document_from_source(gl_ctx, _SCALAR_SRC)
    document.render_pass.seed_uniform_values()
    assert (
        "u_aspect" not in document.render_pass.uniform_values
    )  # engine-driven: valued only in render()
    _teardown(document)


def test_release_frees_uniform_held_resources(gl_ctx: moderngl.Context) -> None:
    # Document.release() used to free only the program + canvas, leaking every texture (and, for a
    # Video, an open capture) parked in uniform_values — one leak per reload, and the file watcher
    # reloads on every external document.json touch. Falsifier: drop the uniform_values loop from
    # Document.release and the sampler's default Image keeps a live texture below.
    document = _document_from_source(gl_ctx, _SAMPLER_SRC)
    document.render_pass.seed_uniform_values()
    image = document.render_pass.uniform_values["u_image"]
    assert (
        image.texture is not None
    )  # touch it: the texture is created lazily on first access

    document.release()
    assert not document.render_pass.uniform_values  # dropped, not merely unreferenced
    assert image._texture is None  # the texture it owned was released, not leaked
