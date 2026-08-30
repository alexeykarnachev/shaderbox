"""Example-library mechanics (features 020·22 + 051) — the agent sees/reads/greps/instantiates
shipped examples via the EXISTING read_shader/grep using an `example:` address, and the default
starter is just-an-example. GL-free parts (the catalogue/resolve/edit-reject addressing) are
unit-tested directly; the GL-marshalled read/grep/create paths run against a real headless App with
the bridge patched to execute inline (the worker->main marshalling is what a real turn drives via
the loop).
"""

from typing import Any

from shaderbox.constants import EXAMPLE_ORDER
from shaderbox.copilot.capabilities import EditResult


def _text_handle(app: Any) -> str:
    return next(
        t.example_id
        for t in app.copilot_backend.example_catalog()
        if t.name == "Text Rendering"
    )


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
    }


def test_resolve_source_distinguishes_example_from_node(app: Any) -> None:
    kind, full = app.copilot_backend._copilot_resolve_source(_text_handle(app))
    assert kind == "example" and full is not None
    # a bare (non-example:) handle is a node
    kind2, _ = app.copilot_backend._copilot_resolve_source("zzzz")
    assert kind2 == "node"


def test_shipped_examples_read_clean_without_joining_working_set(app: Any) -> None:
    for t in app.copilot_backend.example_catalog():
        views = app.copilot_backend.read_shaders([t.example_id])
        assert len(views) == 1, t.example_id
        v = views[0]
        assert v.node_id == t.example_id
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
    nid, errors, _ = app.copilot_backend.create_node(
        "My Text", "", _text_handle(app), False
    )
    assert nid and not errors


def test_create_empty_example_uses_default_starter(app: Any) -> None:
    nid, errors, _ = app.copilot_backend.create_node("Blank", "", "", False)
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
