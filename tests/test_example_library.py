"""Example-library mechanics (features 020·22 + 051) — the agent sees/reads/greps/instantiates
shipped examples via the EXISTING read_shader/grep using an `example:` address, and the default
starter is just-an-example. GL-free parts (the catalogue/resolve/edit-reject addressing) are
unit-tested directly; the GL-marshalled read/grep/create paths run against a real headless App with
the bridge patched to execute inline (the worker->main marshalling is what a real turn drives via
the loop).
"""

import json
from collections.abc import Iterator
from typing import Any

import moderngl
import numpy as np
import pytest

from shaderbox.constants import DOCUMENT_EXAMPLES_DIR, EXAMPLE_ORDER
from shaderbox.copilot.capabilities import EditResult
from shaderbox.paths import DOCUMENT_JSON_BASENAME, shader_lib_root
from shaderbox.scripting.context import ScriptContext
from shaderbox.shader_lib import ShaderLibIndex, set_active
from shaderbox.ui_models import load_document_from_dir


def _text_handle(app: Any) -> str:
    return next(
        t.example_id
        for t in app.copilot_backend.example_catalog()
        if t.name == "Text Rendering"
    )


_ENTITY_EXAMPLE_ID = "c1f0a7d2-4b83-4e91-9a52-6d0f3e8b7c14"


def test_catalogue_has_all_prefixed_unique_examples(app: Any) -> None:
    cat = app.copilot_backend.example_catalog()
    assert len(cat) == len(EXAMPLE_ORDER)
    assert all(t.example_id.startswith("example:") for t in cat)
    assert len({t.example_id for t in cat}) == len(
        EXAMPLE_ORDER
    )  # short ids never collide for the shipped set
    assert {t.name for t in cat} == {
        "UV Mango",
        "Media Input",
        "Text Rendering",
        "Fire",
        "Night City",
        "Radiance Cascades",
        "Entity Flock",
    }


def test_resolve_source_distinguishes_example_from_document(app: Any) -> None:
    kind, full = app.copilot_backend._copilot_resolve_source(_text_handle(app))
    assert kind == "example" and full is not None
    # a bare (non-example:) handle is a document
    kind2, _ = app.copilot_backend._copilot_resolve_source("zzzz")
    assert kind2 == "document"


def test_shipped_examples_read_clean_without_joining_working_set(app: Any) -> None:
    for t in app.copilot_backend.example_catalog():
        views = app.copilot_backend.read_shaders([t.example_id])
        # One view per pass (081 D1): a multi-pass example's OUTPUT pass is its presentation
        # step, so returning it alone hid the technique the example exists to show.
        assert views, t.example_id
        for v in views:
            assert v.document_id.split("#", 1)[0] == t.example_id
            assert len(v.errors) == 0, f"{t.name} must compile clean: {v.errors}"
        # read-only: an example read never joins the (editable) working set
        full = app.copilot_backend._copilot_resolve_example_id(t.example_id)
        assert full not in app.session._copilot_working_set
        assert t.example_id not in app.session._copilot_working_set


def test_grep_surfaces_example_origins(app: Any) -> None:
    hits = app.copilot_backend.grep("void main")
    tpl = [h for h in hits if h.origin.startswith("example:")]
    assert tpl, "grep must scan examples"
    assert all(h.location.startswith("example '") for h in tpl)


def test_create_from_example_instantiates_it(app: Any) -> None:
    nid, errors, _ = app.copilot_backend.create_document(
        "My Text", "", _text_handle(app), False
    )
    assert nid and not errors


def test_create_empty_example_uses_default_starter(app: Any) -> None:
    nid, errors, _ = app.copilot_backend.create_document("Blank", "", "", False)
    assert nid and not errors


def test_edit_on_example_target_is_rejected_read_only(app: Any) -> None:
    res = app.copilot_backend._copilot_resolve_target(
        _text_handle(app), allow_create=False
    )
    assert isinstance(res, EditResult)
    assert res.unresolved and "read-only" in res.unresolved_reason


def test_example_description_reads_shipped(app: Any) -> None:
    cat = app.copilot_backend.example_catalog()
    full = app.copilot_backend._copilot_resolve_example_id(cat[0].example_id)
    assert app.example_description(full) == cat[0].description


def test_opening_an_example_leaves_no_tab_outside_the_project(app: Any) -> None:
    # The crash this pins. A document loaded from the read-only examples dir keeps its passes'
    # source paths pointing THERE until it is saved, and `set_current_document_id` opens an
    # editor tab on whatever path the pass currently holds. `code.py::draw_chrome` then does
    # `relative_to(project_dir)` on the active shader tab, which RAISES for a non-descendant --
    # out of a draw call, so it takes the whole frame down rather than showing a bad label.
    # Saving first rebinds every pass into the project, so the tab opens on a project path.
    #
    # Falsifier: swap the two lines in create_document_from_example and the tab below points at
    # shaderbox/resources/document_examples/... instead.
    before = set(app.ui_documents)
    app.create_document_from_example(EXAMPLE_ORDER[-1])
    created = (set(app.ui_documents) - before).pop()
    document = app.ui_documents[created].document

    outside = [
        t.path for t in app.editor_tabs if not t.path.is_relative_to(app.project_dir)
    ]
    assert outside == [], f"editor tabs opened outside the project: {outside}"
    for name, render_pass in document.passes.items():
        assert render_pass.source.path.is_relative_to(app.project_dir), (
            f"pass '{name}' still points outside the project: {render_pass.source.path}"
        )


@pytest.fixture(scope="module")
def gl_ctx() -> Iterator[moderngl.Context]:
    try:
        context = moderngl.create_standalone_context(require=460)
    except Exception as e:
        pytest.skip(f"no standalone GL context available: {e}")
    set_active(ShaderLibIndex.build(shader_lib_root()))
    yield context
    context.release()


@pytest.mark.parametrize("example_id", EXAMPLE_ORDER)
def test_every_shipped_example_loads_compiles_and_renders(
    gl_ctx: moderngl.Context, example_id: str
) -> None:
    """065 check 15, the half that does not need a display: the shipped set is what a new user
    opens first, and it is the highest-probability breakage of any change to loading or the graph.

    Driven from `EXAMPLE_ORDER` rather than a listdir, so an example added to the roster and not
    to disk fails here instead of going unchecked.
    """
    document_dir = DOCUMENT_EXAMPLES_DIR / example_id
    ui_document = load_document_from_dir(document_dir)
    for name, render_pass in ui_document.document.passes.items():
        if render_pass.program is None:
            render_pass.compile()
        assert render_pass.program is not None, f"{example_id}/{name} does not compile"

    ui_document.document.begin_frame(0)
    ui_document.document.render(u_time=0.0)
    assert len(ui_document.document.render_pass.canvas.texture.read()) > 0
    ui_document.document.release()


@pytest.mark.parametrize("example_id", EXAMPLE_ORDER)
def test_a_shipped_example_keeps_every_uniform_row_through_a_save(
    gl_ctx: moderngl.Context, example_id: str, tmp_path: Any
) -> None:
    """The save prunes rows no live uniform claims, and a row's key names its declaring pass --
    so a key that disagrees with what the shaders declare silently empties the panel. Saved to a
    COPY, because the assert is about the rows and not about rewriting the shipped tree.
    """
    import shutil

    document_dir = tmp_path / example_id
    shutil.copytree(DOCUMENT_EXAMPLES_DIR / example_id, document_dir)
    before = set(
        json.loads((document_dir / DOCUMENT_JSON_BASENAME).read_text())["ui_state"].get(
            "ui_uniforms", {}
        )
    )

    ui_document = load_document_from_dir(document_dir)
    ui_document.save(document_dir.parent, document_dir.name)

    after = set(
        json.loads((document_dir / DOCUMENT_JSON_BASENAME).read_text())["ui_state"].get(
            "ui_uniforms", {}
        )
    )
    assert after == before, f"the save dropped {len(before - after)} row(s)"
    ui_document.document.release()


def test_the_entity_example_carries_its_script_and_draws_its_population(
    app: Any,
) -> None:
    """An instanced example is only an example once its simulation travels with it.

    `UIDocument.save` deliberately omits `scripts/` -- the copilot's checkpoint carries
    the script separately, and teaching the save to write one would let a restore
    overwrite a live script with a stale copy. So the copy happens where both directories
    are known, and this is what says it happened.

    The sibling test that renders every shipped example asserts the read is non-empty,
    which an all-black frame satisfies: an instanced pass whose script never arrived
    renders nothing at all and passes it. This one counts ink.
    """
    app.create_document_from_example(_ENTITY_EXAMPLE_ID)
    document_id = app.current_document_id
    script = app.session.paths.document_script_for(document_id)
    assert script.is_file(), "the example's script did not travel with it"
    assert "@instances" in script.read_text()

    document = app.ui_documents[document_id].document
    for frame in range(8):
        app.session.script_engine.tick(
            document_id, document, ScriptContext(t=frame / 60, dt=1 / 60, frame=frame)
        )
        document.begin_frame(frame)
        document.render(u_time=frame / 60)

    pixels = np.frombuffer(
        document.render_pass.canvas.fbo.read(components=4, dtype="f2"), dtype="f2"
    )
    assert float(pixels.max()) > 0.0, "the population drew nothing"


def test_the_entity_example_still_looks_like_a_flock_after_thirty_seconds(
    app: Any,
) -> None:
    """An example is documentation, and a simulation's late state is what a reader sees.

    The first version collapsed: every per-frame increment exceeded the speed cap, so
    the clamp rewrote velocity to pure tangential and the flock converged onto two
    one-pixel rings -- coverage 39% to 2%, peak brightness 6.7 to 219. Frame one looked
    right the whole time, which is why this walks the clock instead.
    """
    app.create_document_from_example(_ENTITY_EXAMPLE_ID)
    document_id = app.current_document_id
    document = app.ui_documents[document_id].document
    # A canvas big enough to resolve individual entities. At the 64x64 default the
    # sprites are sub-pixel and a collapsed ring covers a similar fraction to a healthy
    # disc -- measured, the two are indistinguishable there, so the gate passed with the
    # collapsing parameters restored.
    document.set_canvas_size((480, 270))

    # What  does before the live tick: re-stat each document's scripts dir and
    # bind what it finds. A document created from an example is inserted after load, so
    # without this its script is never bound.
    app.session.reload_scripts()

    coverage: list[float] = []
    for frame in range(0, 1801):
        # Through `session.tick`, which is what the app calls: it resolves and reloads
        # the document's script first. Ticking the engine directly leaves the script
        # unbound, and the pass then draws its fullscreen fallback -- 78.5% coverage at
        # both times, which passed this gate while measuring nothing.
        app.session.tick([document_id], t=frame / 60, dt=1 / 60)
        document.begin_frame(frame)
        document.render(u_time=frame / 60)
        if frame == 60:
            assert document.passes["swarm"].pending_instances, (
                "the script never drove the pass -- this measures the fallback"
            )
        if frame in (60, 1800):
            pixels = np.frombuffer(
                document.render_pass.canvas.fbo.read(components=4, dtype="f2"),
                dtype="f2",
            ).reshape(-1, 4)
            coverage.append(float((pixels[:, 3] > 0.01).mean()))

    early, late = coverage
    # The absolute floor is what matters: a collapsed flock covers a few percent however
    # it started. Compared against a literal rather than against `early`, since a
    # collapse that happened before frame 60 would make the ratio look healthy.
    assert late > 0.08, f"the flock collapsed to {late:.1%} coverage"
    assert late > early * 0.5, f"coverage fell from {early:.1%} to {late:.1%}"
