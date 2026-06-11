"""Template-library mechanics (feature 020·22) — the agent sees/reads/greps/instantiates shipped
templates via the EXISTING read_shader/grep using a `template:` address, and the default starter is
just-a-template. GL-free parts (the sidecar store, the catalogue/resolve/edit-reject addressing) are
unit-tested directly; the GL-marshalled read/grep/create paths run against a real headless App with the
bridge patched to execute inline (the worker->main marshalling is what a real turn drives via the loop).
"""

from pathlib import Path
from typing import Any

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


def _text_handle(app: Any) -> str:
    return next(
        t.template_id
        for t in app.copilot_backend.template_catalog()
        if t.name == "Text Rendering"
    )


def test_catalogue_has_three_prefixed_unique_templates(app: Any) -> None:
    cat = app.copilot_backend.template_catalog()
    assert len(cat) == 3
    assert all(t.template_id.startswith("template:") for t in cat)
    assert (
        len({t.template_id for t in cat}) == 3
    )  # short ids never collide for the shipped set
    assert {t.name for t in cat} == {"UV Mango", "Media Input", "Text Rendering"}


def test_resolve_source_distinguishes_template_from_node(app: Any) -> None:
    kind, full = app.copilot_backend._copilot_resolve_source(_text_handle(app))
    assert kind == "template" and full is not None
    # a bare (non-template:) handle is a node
    kind2, _ = app.copilot_backend._copilot_resolve_source("zzzz")
    assert kind2 == "node"


def test_shipped_templates_read_clean_without_joining_working_set(app: Any) -> None:
    for t in app.copilot_backend.template_catalog():
        views = app.copilot_backend.read_shaders([t.template_id])
        assert len(views) == 1, t.template_id
        v = views[0]
        assert v.node_id == t.template_id
        assert len(v.errors) == 0, f"{t.name} must compile clean: {v.errors}"
        # read-only: a template read never joins the (editable) working set
        full = app.copilot_backend._copilot_resolve_template_id(t.template_id)
        assert full not in app.session._copilot_working_set
        assert t.template_id not in app.session._copilot_working_set


def test_grep_surfaces_template_origins(app: Any) -> None:
    hits = app.copilot_backend.grep("void main")
    tpl = [h for h in hits if h.origin.startswith("template:")]
    assert tpl, "grep must scan templates"
    assert all(h.location.startswith("template '") for h in tpl)


def test_create_from_template_instantiates_it(app: Any) -> None:
    nid, errors, _ = app.copilot_backend.create_node(
        "My Text", "", _text_handle(app), False
    )
    assert nid and not errors


def test_create_empty_template_uses_default_starter(app: Any) -> None:
    nid, errors, _ = app.copilot_backend.create_node("Blank", "", "", False)
    assert nid and not errors


def test_edit_on_template_target_is_rejected_read_only(app: Any) -> None:
    res = app.copilot_backend._copilot_resolve_target(
        _text_handle(app), allow_create=False
    )
    assert isinstance(res, EditResult)
    assert res.unresolved and "read-only" in res.unresolved_reason


def test_description_override_shadows_shipped(app: Any, monkeypatch: Any) -> None:
    cat = app.copilot_backend.template_catalog()
    full = app.copilot_backend._copilot_resolve_template_id(cat[0].template_id)
    app.template_descriptions.descriptions[full] = "MY OVERRIDE"
    assert app.template_description(full) == "MY OVERRIDE"
