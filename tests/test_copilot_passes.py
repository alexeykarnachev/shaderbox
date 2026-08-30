"""The copilot can see and address a document's passes (065 stage 8, D11).

Three things had to change together, and each is useless alone: the ADDRESS scheme gained a
`<document>#<pass>` kind, the working set shows a document's passes as sub-sections (one member is
one DOCUMENT, so an 8-pass document cannot evict its own passes out of a six-slot cap), and the
project map lists them (a model that is never SHOWN a pass cannot construct an address it has
never seen — corollary 2 of the actor model).

No new tools: every edit tool inherits the address kind through the single resolver.
"""

from typing import Any

import pytest

from shaderbox.copilot.address import (
    is_pass_address,
    pass_address,
    split_pass_address,
)
from shaderbox.copilot.capabilities import DocumentTreeEntry
from shaderbox.copilot.prompt import render_working_set
from shaderbox.copilot.prompt_context import _render_document_tree

_A = """#version 460 core
in vec2 vs_uv;
out vec4 fs_color;
void main() { fs_color = vec4(0.8, 0.0, 0.0, 1.0); }
"""

_B = """#version 460 core
in vec2 vs_uv;
uniform sampler2D u_src;
out vec4 fs_color;
void main() { fs_color = texture(u_src, vs_uv) * 0.5; }
"""


def _two_pass(app: Any) -> str:
    """A document with exactly two passes named `scene` and `composite`.

    The seeded document already has one pass, so it is renamed rather than left beside the new
    ones — a stray third pass would make every count in here mean something else.
    """
    document_id = app.current_document_id
    document = app.ui_documents[document_id].document
    app.session.rename_pass(document_id, next(iter(document.passes)), "scene")
    app.session.add_pass(document_id, "composite")
    document.passes["scene"].release_program(_A)
    document.passes["scene"].compile()
    document.passes["composite"].release_program(_B)
    document.passes["composite"].compile()
    app.session.wire_pass_input(document_id, "composite", "u_src", "scene")
    app.session.set_output_pass(document_id, "composite")
    return document_id


# ---------------------------------------------------------------- the address kind


def test_a_pass_address_round_trips() -> None:
    assert pass_address("ab12", "bright") == "ab12#bright"
    assert split_pass_address("ab12#bright") == ("ab12", "bright")
    assert is_pass_address("ab12#bright")


def test_a_bare_document_address_is_unchanged() -> None:
    # The suffix form is what keeps every tool that predates the graph working: a bare id still
    # means "this document", resolving to its OUTPUT pass.
    assert split_pass_address("ab12") == ("ab12", "")
    assert not is_pass_address("ab12")
    assert split_pass_address("") == ("", "")


def test_a_lib_address_is_never_read_as_a_pass() -> None:
    # A '#' inside a library path is part of the filename.
    assert not is_pass_address("lib:weird#name.glsl")
    assert split_pass_address("lib:weird#name.glsl") == ("lib:weird#name.glsl", "")


# ---------------------------------------------------------------- resolution


def test_an_edit_addressed_to_a_pass_lands_on_that_pass(app: Any) -> None:
    document_id = _two_pass(app)
    short = app.copilot_backend._copilot_short_ids()[document_id]
    result = app.copilot_backend.apply_shader_edit(
        old_str="0.8",
        new_str="0.3",
        replace_all=False,
        target=pass_address(short, "scene"),
    )
    assert result.matches == 1, result
    document = app.ui_documents[document_id].document
    assert "0.3" in document.passes["scene"].source.text
    assert "0.3" not in document.passes["composite"].source.text
    # And it reached DISK, not just the live object.
    assert "0.3" in document.passes["scene"].source.path.read_text()


def test_a_bare_address_still_edits_the_output_pass(app: Any) -> None:
    document_id = _two_pass(app)
    short = app.copilot_backend._copilot_short_ids()[document_id]
    result = app.copilot_backend.apply_shader_edit(
        old_str="0.5", new_str="0.25", replace_all=False, target=short
    )
    assert result.matches == 1, result
    document = app.ui_documents[document_id].document
    assert "0.25" in document.passes["composite"].source.text
    assert "0.25" not in document.passes["scene"].source.text


def test_an_unknown_pass_is_refused_with_the_names_that_exist(app: Any) -> None:
    document_id = _two_pass(app)
    short = app.copilot_backend._copilot_short_ids()[document_id]
    result = app.copilot_backend.apply_shader_edit(
        old_str="x",
        new_str="y",
        replace_all=False,
        target=pass_address(short, "ghost"),
    )
    assert result.unresolved
    assert "no pass 'ghost'" in (result.unresolved_reason or "")
    # Actionable, not merely a refusal: the message names the passes it could have meant.
    assert "composite" in (result.unresolved_reason or "")


# ---------------------------------------------------------------- the working set


def test_the_working_set_shows_a_document_s_passes(app: Any) -> None:
    _two_pass(app)
    views, _ = app.copilot_backend.read_working_set()
    view = next(v for v in views if not v.is_lib)
    assert [p.name for p in view.passes] == ["composite", "scene"]
    text = render_working_set(views, [])[0].content
    assert "PASS scene" in text and "PASS composite" in text
    # The output is marked, and each pass carries the handle that edits it.
    assert "[output]" in text
    assert "#scene" in text and "#composite" in text


def test_a_pass_s_wiring_is_shown_including_what_is_unwired(app: Any) -> None:
    document_id = _two_pass(app)
    views, _ = app.copilot_backend.read_working_set()
    view = next(v for v in views if not v.is_lib)
    composite = next(p for p in view.passes if p.name == "composite")
    assert composite.inputs == ["u_src <- scene"]

    app.session.wire_pass_input(document_id, "composite", "u_src", "")
    views, _ = app.copilot_backend.read_working_set()
    view = next(v for v in views if not v.is_lib)
    composite = next(p for p in view.passes if p.name == "composite")
    # An unwired input reads black (D3), which is silent on screen — so it is stated here.
    assert composite.inputs == ["u_src <- (nothing; reads BLACK)"]
    assert "reads BLACK" in render_working_set(views, [])[0].content


def test_a_single_pass_document_renders_exactly_as_before(app: Any) -> None:
    # The ordinary case must not change: no PASS sub-sections, and the member keeps its own
    # listing / uniforms / errors.
    views, _ = app.copilot_backend.read_working_set()
    view = next(v for v in views if not v.is_lib)
    assert view.passes == []
    text = render_working_set(views, [])[0].content
    assert "PASS " not in text
    assert view.listing and "uniforms:" in text


def test_a_pass_address_in_the_working_set_collapses_to_its_document(app: Any) -> None:
    # One member is one DOCUMENT (D11): an 8-pass document must not be able to evict its own
    # passes out of the six-slot cap. Driven on a NON-current document, since the current one is
    # unioned in regardless and would hide a per-pass member.
    document_id = _two_pass(app)
    short = app.copilot_backend._copilot_short_ids()[document_id]
    other = next(i for i in app.ui_documents if i != document_id)
    app.set_current_document_id(other)

    app.copilot_backend._working_set_add(pass_address(document_id, "scene"))
    app.copilot_backend._working_set_add(pass_address(document_id, "composite"))
    views, _ = app.copilot_backend.read_working_set()
    for_this = [v for v in views if v.address == short]
    assert len(for_this) == 1, (
        f"{len(for_this)} members for one document — its passes are competing for slots"
    )
    assert [p.name for p in for_this[0].passes] == ["composite", "scene"]


def test_a_broken_pass_reports_its_own_errors(app: Any) -> None:
    document = app.ui_documents[_two_pass(app)].document
    document.passes["scene"].release_program("#version 460 core\nnot glsl at all\n")
    document.passes["scene"].compile()
    views, _ = app.copilot_backend.read_working_set()
    view = next(v for v in views if not v.is_lib)
    scene = next(p for p in view.passes if p.name == "scene")
    composite = next(p for p in view.passes if p.name == "composite")
    assert scene.errors, "the broken pass reported no errors"
    assert not composite.errors, "a sibling inherited the broken pass's errors"


# ---------------------------------------------------------------- the project map


def test_the_project_map_lists_passes_and_marks_the_output(app: Any) -> None:
    _two_pass(app)
    entries = app.copilot_backend.document_tree()
    entry = next(e for e in entries if e.passes)
    assert set(entry.passes) == {"scene", "composite"}
    rendered = _render_document_tree(entries)
    assert "passes:" in rendered
    assert "composite*" in rendered  # the star marks the output
    assert "#<pass>" in rendered  # and the map teaches the address form


def test_the_map_says_nothing_about_passes_for_a_single_pass_document(
    app: Any,
) -> None:
    # A single-pass document's row is what it always was — the pass line appears only where there
    # is a graph to describe (the seeded project also ships a multi-pass example, which does get
    # one, so this checks the ROW rather than the whole map).
    entries = app.copilot_backend.document_tree()
    single = [e for e in entries if len(e.passes) < 2]
    assert single, "no single-pass document to check"
    assert "passes:" not in _render_document_tree(single)


def test_a_document_with_a_broken_non_output_pass_reports_errors(app: Any) -> None:
    # has_errors read only the OUTPUT pass, so a broken pass nothing draws reported the document
    # clean — and the model's only project-wide error signal was wrong.
    document_id = _two_pass(app)
    document = app.ui_documents[document_id].document
    document.passes["scene"].release_program("#version 460 core\nbroken\n")
    document.passes["scene"].compile()
    entry = next(
        e
        for e in app.copilot_backend.document_tree()
        if e.document_id == app.copilot_backend._copilot_short_ids()[document_id]
    )
    assert entry.has_errors
    assert "HAS ERRORS" in _render_document_tree([entry])


@pytest.mark.parametrize("verb", ["scene", "composite"])
def test_every_pass_is_addressable(app: Any, verb: str) -> None:
    document_id = _two_pass(app)
    short = app.copilot_backend._copilot_short_ids()[document_id]
    entry = DocumentTreeEntry(
        document_id=short,
        name="x",
        has_errors=False,
        is_current=True,
        passes=("composite", "scene"),
        output_pass="composite",
    )
    assert pass_address(entry.document_id, verb) in {
        pass_address(short, name) for name in entry.passes
    }
