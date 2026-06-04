"""Template-library mechanics (feature 020·22) — the agent sees/reads/greps/instantiates shipped
templates via the EXISTING read_shader/grep using a `template:` address, and the default starter is
just-a-template. GL-free parts (the sidecar store, the catalogue/resolve/edit-reject addressing) are
unit-tested directly; the GL-marshalled read/grep/create paths run against a real headless App with the
bridge patched to execute inline (the worker->main marshalling is what a real turn drives via the loop).
"""

import contextlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from shaderbox.copilot.capabilities import EditResult
from shaderbox.templates_descriptions import TemplateDescriptionsStore

# ---- the sidecar store (pure, GL-free) ----


def test_descriptions_store_round_trip(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "shaderbox.templates_descriptions.app_data_dir", lambda: tmp_path
    )
    store = TemplateDescriptionsStore.load()
    assert store.get("uuid-a") is None  # no override -> caller falls back to shipped
    store.set("uuid-a", "a text-rendering template")
    reloaded = TemplateDescriptionsStore.load()
    assert reloaded.get("uuid-a") == "a text-rendering template"
    assert reloaded.get("uuid-b") is None


def test_descriptions_store_corrupt_is_empty(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(
        "shaderbox.templates_descriptions.app_data_dir", lambda: tmp_path
    )
    (tmp_path / "template_descriptions.json").write_text("{ not json", encoding="utf-8")
    store = TemplateDescriptionsStore.load()  # must NOT raise
    assert store.get("anything") is None


# ---- the GL-marshalled paths (real headless App, bridge inlined) ----


@pytest.fixture
def app() -> Iterator[Any]:
    glfw = pytest.importorskip("glfw")
    if not glfw.init():
        pytest.skip("no GL")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    from shaderbox.app import App

    a = App(project_dir=None)
    # The bridge latches _shutdown during App-init teardown + needs the main loop to drain ops; in a
    # test there's no loop, so run every marshalled op INLINE (we're already on the GL thread).
    a.copilot.bridge.run_on_main = lambda fn, timeout=None, defer=False: fn()  # type: ignore[method-assign]
    yield a
    with contextlib.suppress(Exception):
        a.release()


def _text_handle(app: Any) -> str:
    return next(
        t.template_id
        for t in app._copilot_template_catalog()
        if t.name == "Text Rendering"
    )


def test_catalogue_has_three_prefixed_unique_templates(app: Any) -> None:
    cat = app._copilot_template_catalog()
    assert len(cat) == 3
    assert all(t.template_id.startswith("template:") for t in cat)
    assert (
        len({t.template_id for t in cat}) == 3
    )  # short ids never collide for the shipped set
    assert {t.name for t in cat} == {"UV Mango", "Media Input", "Text Rendering"}


def test_resolve_source_distinguishes_template_from_node(app: Any) -> None:
    kind, full = app._copilot_resolve_source(_text_handle(app))
    assert kind == "template" and full is not None
    # a bare (non-template:) handle is a node
    kind2, _ = app._copilot_resolve_source("zzzz")
    assert kind2 == "node"


def test_shipped_templates_read_clean_without_freshness_stamp(app: Any) -> None:
    for t in app._copilot_template_catalog():
        views = app._copilot_read_shaders([t.template_id])
        assert len(views) == 1, t.template_id
        v = views[0]
        assert v.node_id == t.template_id
        assert len(v.errors) == 0, f"{t.name} must compile clean: {v.errors}"
        # read-only: a template read never stamps the edit-freshness map
        full = app._copilot_resolve_template_id(t.template_id)
        assert full not in app._copilot_read_revision


def test_grep_surfaces_template_origins(app: Any) -> None:
    hits = app._copilot_grep("void main")
    tpl = [h for h in hits if h.origin.startswith("template:")]
    assert tpl, "grep must scan templates"
    assert all(h.location.startswith("template '") for h in tpl)


def test_create_from_template_instantiates_it(app: Any) -> None:
    nid, errors = app._copilot_create_node("My Text", "", _text_handle(app), False)
    assert nid and not errors


def test_create_empty_template_uses_default_starter(app: Any) -> None:
    nid, errors = app._copilot_create_node("Blank", "", "", False)
    assert nid and not errors


def test_edit_on_template_target_is_rejected_read_only(app: Any) -> None:
    res = app._copilot_resolve_target(_text_handle(app), allow_create=False)
    assert isinstance(res, EditResult)
    assert res.unresolved and "read-only" in res.unresolved_reason


def test_description_override_shadows_shipped(app: Any, monkeypatch: Any) -> None:
    cat = app._copilot_template_catalog()
    full = app._copilot_resolve_template_id(cat[0].template_id)
    app.template_descriptions.descriptions[full] = "MY OVERRIDE"
    assert app.template_description(full) == "MY OVERRIDE"
