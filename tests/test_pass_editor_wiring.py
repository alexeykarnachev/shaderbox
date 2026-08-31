"""Opening a pass shows THAT pass, and saving it writes THAT pass (065).

The pass list's `open` button opened the right tab all along — the tab carries the pass's own path
— but everything downstream resolved "the current document's shader" as `document.render_pass`,
which is the OUTPUT. That was the same file for a document's whole history until a second pass
existed, so the assumption had never been wrong before and nothing observed it. The symptom the
maintainer saw: every pass row opened the same source.

Each test here names one consumer of the active tab and pins it to the tab's PATH.
"""

from typing import Any

from shaderbox.paths import pass_shader_name

_GREEN = """#version 460 core
in vec2 vs_uv;
out vec4 fs_color;
void main() { fs_color = vec4(0.0, 1.0, 0.0, 1.0); }
"""

_BLUE = """#version 460 core
in vec2 vs_uv;
out vec4 fs_color;
void main() { fs_color = vec4(0.0, 0.0, 1.0, 1.0); }
"""


def _two_pass(app: Any) -> str:
    """A document whose OUTPUT is `main` and which also has a distinguishable `second` pass."""
    document_id = app.current_document_id
    assert app.session.add_pass(document_id, "second") == ""
    document = app.ui_documents[document_id].document
    for name, src in (("main", _GREEN), ("second", _BLUE)):
        document.passes[name].release_program(src)
        document.passes[name].source.path.write_text(src)
        document.passes[name].compile()
    assert app.session.set_output_pass(document_id, "main") == ""
    return document_id


def _edit(app: Any, new_text: str) -> None:
    """Put `new_text` in the active editor and mark it dirty, as a keystroke would.

    `TextEditor.set_text` does NOT advance the undo index, so a programmatic edit reads clean and
    `flush_current_editor` returns early — the dirty flag is what a real edit moves.
    """
    session = app.get_current_session()
    assert session is not None
    session.editor.set_text(new_text)
    session.saved_undo = session.editor.get_undo_index() - 1


def test_opening_a_pass_loads_that_pass_s_source(app: Any) -> None:
    # The reported bug: every row opened the same shader.
    document_id = _two_pass(app)
    app.ensure_shader_tab(document_id, "second")
    session = app.get_current_session()
    assert session is not None
    assert session.source.path.name == pass_shader_name("second")
    assert "0.0, 0.0, 1.0" in session.editor.get_text(), (
        "the editor loaded a different pass's text"
    )


def test_opening_each_pass_in_turn_shows_each_one(app: Any) -> None:
    document_id = _two_pass(app)
    seen: dict[str, str] = {}
    for name in ("main", "second", "main"):
        app.ensure_shader_tab(document_id, name)
        session = app.get_current_session()
        assert session is not None
        seen[name] = session.source.path.name
    assert seen == {
        "main": pass_shader_name("main"),
        "second": pass_shader_name("second"),
    }


def test_the_active_tab_decides_the_session_not_the_output_pass(app: Any) -> None:
    # `main` IS the output here, so resolving through the output would silently agree for it and
    # only disagree for `second` — which is why the bug survived every single-pass document.
    document_id = _two_pass(app)
    app.ensure_shader_tab(document_id, "second")
    document = app.ui_documents[document_id].document
    assert app.current_editor_path == document.passes["second"].source.path
    assert app.current_editor_path != document.render_pass.source.path
    assert app.get_current_session().source.path == app.current_editor_path


def test_saving_a_non_output_pass_updates_that_pass(app: Any) -> None:
    # The mirror of the read bug: flush matched only the output pass, so editing any other pass
    # fell to the plain disk-write branch and its program was never dropped.
    document_id = _two_pass(app)
    app.ensure_shader_tab(document_id, "second")
    _edit(app, _BLUE.replace("0.0, 0.0, 1.0", "1.0, 1.0, 0.0"))
    app.flush_current_editor()

    document = app.ui_documents[document_id].document
    assert "1.0, 1.0, 0.0" in document.passes["second"].source.text
    # The OUTPUT pass is untouched — the edit did not land on the wrong file.
    assert "0.0, 1.0, 0.0" in document.passes["main"].source.text
    # A shader save updates MEMORY and drops the program so the next render recompiles; the disk
    # write rides UIDocument.save (quit / document switch), which is the pre-065 contract.
    assert document.passes["second"].program is None
    app.session.save_ui_document(app.ui_documents[document_id])
    assert "1.0, 1.0, 0.0" in document.passes["second"].source.path.read_text()


def test_saving_the_output_pass_still_works(app: Any) -> None:
    document_id = _two_pass(app)
    app.ensure_shader_tab(document_id, "main")
    _edit(app, _GREEN.replace("0.0, 1.0, 0.0", "1.0, 0.0, 1.0"))
    app.flush_current_editor()
    document = app.ui_documents[document_id].document
    assert "1.0, 0.0, 1.0" in document.passes["main"].source.text
    assert "0.0, 0.0, 1.0" in document.passes["second"].source.text


def test_a_single_pass_document_is_unaffected(app: Any) -> None:
    # The ordinary case, which is what made the bug invisible: with one pass, the tab's path and
    # the output pass's path are the same file.
    document_id = app.current_document_id
    app.ensure_shader_tab(document_id)
    document = app.ui_documents[document_id].document
    assert app.current_editor_path == document.render_pass.source.path
    assert app.get_current_session().source.path == document.render_pass.source.path
