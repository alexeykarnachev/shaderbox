"""The copilot's pass tools (feature 076): add_pass / set_pass / delete_pass over the session's
pass verbs, so a multi-pass document can be authored without a human at the pass list.

Falsifiers: a pass added by the tool must exist on disk (`passes/<name>.glsl`) and in
`graph.json` with the runs and target given; set_pass must change only the fields given; the
result must carry the pass table the model reads back; a rejected name, an out-of-range run count
and the last pass must come back as errors, not silent no-ops; the registry must expose all three
lazily, delete gated."""

import json
from typing import Any

from shaderbox.copilot.tools.base import GatePolicy
from shaderbox.copilot.tools.registry import build_registry
from shaderbox.pass_graph import MAX_ITERATIONS
from tests._caps import minimal_caps


def _graph(app: Any, document_id: str) -> dict[str, Any]:
    document_dir = app.paths.documents_dir / document_id
    return json.loads((document_dir / "graph.json").read_text())


def test_add_pass_creates_the_file_and_the_graph_entry(app: Any) -> None:
    document_id = app.current_document_id
    backend = app.copilot_backend
    res = backend.add_pass("", "glow", 12, "f4", 0.5, True, None, True)
    assert res.ok, res.error
    document = app.ui_documents[document_id].document
    assert "glow" in document.passes
    assert (app.paths.documents_dir / document_id / "passes" / "glow.frag.glsl").exists()
    entry = _graph(app, document_id)["passes"]["glow"]
    assert entry["iterations"] == 12
    assert entry["target"]["dtype"] == "f4" and entry["target"]["scale"] == 0.5
    assert entry["target"]["filter_linear"] is True
    assert _graph(app, document_id)["output"] == "glow"
    # The echo the model reads back names the pass, its runs, target and the output mark.
    assert "glow [output]: runs 12, target f4 x0.5, linear" in res.table
    assert "#<name>" in res.table


def test_set_pass_changes_only_what_is_given(app: Any) -> None:
    document_id = app.current_document_id
    backend = app.copilot_backend
    assert backend.add_pass("", "blur", 3, "f2", None, None, None, False).ok
    res = backend.set_pass("", "blur", None, None, 0.25, None, None, False, "")
    assert res.ok, res.error
    entry = _graph(app, document_id)["passes"]["blur"]
    assert entry["iterations"] == 3 and entry["target"]["dtype"] == "f2"
    assert entry["target"]["scale"] == 0.25
    # A rename carries the graph entry and the file with it.
    res = backend.set_pass("", "blur", 5, None, None, None, None, True, "smear")
    assert res.ok, res.error
    graph = _graph(app, document_id)
    assert "blur" not in graph["passes"] and graph["passes"]["smear"]["iterations"] == 5
    assert graph["output"] == "smear"
    assert (app.paths.documents_dir / document_id / "passes" / "smear.frag.glsl").exists()


def test_rejections_are_errors_not_silent_no_ops(app: Any) -> None:
    backend = app.copilot_backend
    assert not backend.add_pass("", "9bad", None, None, None, None, None, False).ok
    assert "letter" in backend.add_pass("", "9bad", None, None, None, None, None, False).error
    too_many = backend.add_pass("", "loop", MAX_ITERATIONS + 1, None, None, None, None, False)
    assert not too_many.ok and str(MAX_ITERATIONS) in too_many.error
    bad_dtype = backend.set_pass("", "loop", None, "f8", None, None, None, False, "")
    assert not bad_dtype.ok and "f8" in bad_dtype.error
    assert not backend.set_pass("", "nope", 2, None, None, None, None, False, "").ok
    assert not backend.delete_pass("", "nope").ok
    assert not backend.add_pass("zzzz", "x", None, None, None, None, None, False).ok


def test_delete_pass_removes_it_but_never_the_last(app: Any) -> None:
    document_id = app.current_document_id
    backend = app.copilot_backend
    only = next(iter(app.ui_documents[document_id].document.passes))
    assert not backend.delete_pass("", only).ok
    assert backend.add_pass("", "extra", None, None, None, None, None, False).ok
    res = backend.delete_pass("", "extra")
    assert res.ok, res.error
    assert "extra" not in _graph(app, document_id)["passes"]
    assert "extra" not in res.table


def test_registry_exposes_the_three_lazily_with_delete_gated() -> None:
    registry = build_registry(minimal_caps())
    for name in ("add_pass", "set_pass", "delete_pass"):
        definition = registry.definition_for(name)
        assert definition is not None and not definition.eager, name
        assert registry.is_lazy(name) and registry.is_mutating(name)
    assert registry.definition_for("delete_pass").gate_policy is GatePolicy.ALWAYS
    assert registry.definition_for("add_pass").gate_policy is GatePolicy.NONE
    # The handler is wired: an execute through the registry validates and reaches the capability.
    ok, msg, _ = registry.execute("add_pass", {"name": "glow"}, "")
    assert ok and "added pass 'glow'" in msg
