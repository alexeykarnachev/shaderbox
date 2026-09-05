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
from shaderbox.render_shape import RenderShape
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
    assert (
        app.paths.documents_dir / document_id / "passes" / "glow.frag.glsl"
    ).exists()
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
    assert (
        app.paths.documents_dir / document_id / "passes" / "smear.frag.glsl"
    ).exists()


def test_rejections_are_errors_not_silent_no_ops(app: Any) -> None:
    backend = app.copilot_backend
    assert not backend.add_pass("", "9bad", None, None, None, None, None, False).ok
    assert (
        "letter"
        in backend.add_pass("", "9bad", None, None, None, None, None, False).error
    )
    too_many = backend.add_pass(
        "", "loop", MAX_ITERATIONS + 1, None, None, None, None, False
    )
    assert not too_many.ok and str(MAX_ITERATIONS) in too_many.error
    bad_dtype = backend.set_pass("", "loop", None, "f8", None, None, None, False, "")
    assert not bad_dtype.ok and "f8" in bad_dtype.error
    # An empty dtype is "unchanged", the way a model spells a field it is not setting.
    assert backend.add_pass("", "keep", None, "f4", None, None, None, False).ok
    assert backend.set_pass("", "keep", 2, "", None, None, None, False, "").ok
    assert (
        _graph(app, app.current_document_id)["passes"]["keep"]["target"]["dtype"]
        == "f4"
    )
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


def test_the_pass_verbs_are_mutating_with_delete_lazy_and_gated() -> None:
    # 081 D6: add_pass/set_pass are eager — a lazy load grows the tools array mid-turn, and that
    # array precedes every message, so the whole cached prefix dies for one added schema.
    # delete_pass stays lazy: it is destructive, gated, and rare.
    registry = build_registry(minimal_caps())
    for name in ("add_pass", "set_pass", "delete_pass"):
        definition = registry.definition_for(name)
        assert definition is not None and registry.is_mutating(name), name
    assert registry.definition_for("add_pass").eager
    assert registry.definition_for("set_pass").eager
    assert not registry.definition_for("delete_pass").eager
    assert registry.is_lazy("delete_pass")
    assert registry.definition_for("delete_pass").gate_policy is GatePolicy.ALWAYS
    assert registry.definition_for("add_pass").gate_policy is GatePolicy.NONE
    # The handler is wired: an execute through the registry validates and reaches the capability.
    ok, msg, _ = registry.execute("add_pass", {"name": "glow"}, "")
    assert ok and "added pass 'glow'" in msg


# Every tool whose args carry an address must declare how it reads `<id>#<pass>`. c8960e1 fixed
# probe_render and left a comment claiming "every other tool takes the pass address" — false when
# written: read_shader did not, and four models hit it. A hand-kept list would drift the same way,
# so the domain is enumerated from the registry and a NEW address-taking tool fails this file
# until it is classified.
_PASS_AWARE = frozenset({"edit_shader", "write_shader", "probe_render", "read_shader"})
# A whole-document op (delete/rename/switch/duplicate/canvas/media/script), or a deliverable
# render: "one pass" is not a thing either can mean, so a pass address is a category error and
# must be REFUSED, never silently downgraded to the document (conventions.md: never change a
# destructive op's behavior on a guess the model cannot see).
_DOCUMENT_ONLY = frozenset(
    {
        "add_pass",
        "bind_media",
        "delete_document",
        "delete_pass",
        "duplicate_document",
        "edit_script",
        "read_script",
        "rename_document",
        "render_image",
        "render_video",
        "set_canvas_size",
        "set_pass",
        "set_uniform",
        "switch_document",
        "unbind_media",
        "write_script",
    }
)
_ADDRESS_FIELDS = frozenset({"document", "documents", "target"})


def test_every_address_taking_tool_declares_its_pass_behavior() -> None:
    registry = build_registry(minimal_caps())
    addressed = {
        d.name
        for d in registry.definitions()
        if _ADDRESS_FIELDS & set(d.args_model.model_fields)
    }
    unclassified = addressed - _PASS_AWARE - _DOCUMENT_ONLY
    assert not unclassified, (
        f"address-taking tools with no pass-address classification: {sorted(unclassified)}"
    )
    # The classes name only real tools, so a rename cannot leave a dead entry behind.
    assert addressed >= (_PASS_AWARE | _DOCUMENT_ONLY)


def test_a_pass_aware_tool_resolves_a_pass_address(app: Any) -> None:
    backend = app.copilot_backend
    assert backend.add_pass("", "red", None, None, None, None, None, False).ok
    short = backend._copilot_short_ids()[app.current_document_id]
    red = "#version 460 core\nin vec2 vs_uv;\nout vec4 fs_color;\nvoid main() { fs_color = vec4(1.0, 0.0, 0.0, 1.0); }\n"
    assert backend.apply_full_rewrite(red, f"{short}#red").errors == []
    for name in sorted(_PASS_AWARE):
        assert name in {d.name for d in build_registry(backend).definitions()}, name
    # probe_render and read_shader both take the address; neither says "no such document".
    assert "no such document" not in backend.probe_render(f"{short}#red", 0.0)
    assert backend.read_shaders([f"{short}#red"])[0].document_id.endswith("#red")


def test_a_document_only_tool_refuses_a_pass_address(app: Any) -> None:
    backend = app.copilot_backend
    assert backend.add_pass("", "red", None, None, None, None, None, False).ok
    short = backend._copilot_short_ids()[app.current_document_id]
    result = backend.render_image(f"{short}#red", RenderShape.NATIVE)
    assert not result.ok
    # Named and actionable, not silently downgraded to the whole document: the message says a
    # render takes a document and points at the tool that DOES measure one pass.
    assert "whole document" in (result.error or "")
    assert "probe_render" in (result.error or "")


def test_an_edit_to_a_non_output_pass_is_probed_as_that_pass(app: Any) -> None:
    # hy4-preview on the station: writing paint and canvas while the output was still the
    # stub read "changed NOTHING on screen" six times, and the no-op brake ended the turn.
    backend = app.copilot_backend
    assert backend.add_pass("", "glow", None, None, None, None, None, False).ok
    short = backend._copilot_short_ids()[app.current_document_id]
    red = "#version 460 core\nin vec2 vs_uv;\nout vec4 fs_color;\nvoid main() { fs_color = vec4(1.0, 0.0, 0.0, 1.0); }\n"
    facts = backend.apply_full_rewrite(red, f"{short}#glow").render_facts
    assert "rgba(255,0,0,255)" in facts, facts
    assert "changed NOTHING" not in facts
    # The same frame again IS a no-op, judged on that pass (a new batch: one write per file per batch).
    backend.batch_begin()
    facts = backend.apply_full_rewrite(red + "// again\n", f"{short}#glow").render_facts
    assert "changed NOTHING" in facts


def test_an_invalid_argument_names_the_field_that_was_wrong(app: Any) -> None:
    # 081 D11: pydantic hands the offending field in `loc` and the message threw it away, so
    # "Extra inputs are not permitted" read as a complaint about the VALUE — one model rewrote
    # the same payload's content three times before dropping the extra key.
    registry = build_registry(app.copilot_backend)
    ok, msg, _ = registry.execute(
        "write_shader", {"document": "abcd", "target": "abcd", "new_text": "x"}, ""
    )
    assert not ok
    assert "document" in msg, msg


def test_the_engine_uniform_prose_is_generated_not_retyped(app: Any) -> None:
    # 081 D12: the two prose surfaces named THREE of five engine uniforms, and the two they
    # omitted (the pass counters) are exactly what a model tried to set — 600s of wall clock on
    # three rejected calls. Asserting the rendered text against the table it is generated FROM is
    # a tautology, so this pins the mechanism instead: each surface interpolates the shared list,
    # and neither spells the set out. A hand-written list is what drifted.
    _ = app
    from shaderbox.copilot.prompt_context import _CONVENTIONS
    from shaderbox.copilot.prompt_context import _ENGINE_UNIFORM_LIST as _CONV_LIST
    from shaderbox.copilot.tools.shader import _ENGINE_UNIFORM_LIST, _SET_UNIFORM_DESC
    from shaderbox.engine_uniforms import ENGINE_UNIFORM_TYPES

    for rendered, generated in (
        (_SET_UNIFORM_DESC, _ENGINE_UNIFORM_LIST),
        (_CONVENTIONS, _CONV_LIST),
    ):
        # The generated fragment names the WHOLE table and is what the surface actually carries.
        for name in ENGINE_UNIFORM_TYPES:
            assert name in generated, name
        assert generated in rendered


def test_the_pass_tools_need_no_load_first(app: Any) -> None:
    # 081 D6: the tools array precedes every message, so growing it mid-turn changes byte zero and
    # voids the whole cached prefix — measured 2.9% cache share on requests where it grew against
    # 62.7% where it did not. The two hot pass tools are worth more eager than lazy.
    registry = build_registry(app.copilot_backend)
    eager = {spec.name for spec in registry.assemble_specs(set())}
    assert {"add_pass", "set_pass"} <= eager
