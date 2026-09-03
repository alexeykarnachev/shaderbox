"""Completion policy (073 W-B): the provider table, the auto-trigger, and `K`'s lookup.

The library owns the popup and the word prefix; the host decides what to offer and when.
These pin the decision: `uniform ` offers the builtin declarations, an identifier of two
letters opens by itself, one letter only on Ctrl+N, an unasked popup leaves Enter a newline.
"""

from pathlib import Path
from typing import Any

from shaderbox.completion import (
    CompletionContext,
    builtin_uniform_declarations,
    eligible_providers,
    matches,
    offer,
    symbol_doc,
    word_at,
)
from shaderbox.editor.ffi import Editor, KeyCode, KeyMod, Language, Mode
from shaderbox.editor.input import KeyEvent
from shaderbox.editor_types import EditorTab
from shaderbox.help_content import ENGINE_UNIFORM_DOCS
from shaderbox.hotkeys import _is_lookup_key
from shaderbox.shader_lib.index import ShaderLibFunction
from shaderbox.tabs.code import _consume_lookup_request, _drive_completion


def _context(**overrides: Any) -> CompletionContext:
    base: dict[str, Any] = {
        "tab_kind": "shader",
        "line_before_caret": "",
        "prefix": "",
        "lib_functions": ("SB_hash", "SB_hash2", "SB_noise"),
        "pass_uniforms": ("u_speed",),
        "explicit": False,
    }
    base.update(overrides)
    return CompletionContext(**base)


def test_uniform_context_offers_every_builtin_declaration() -> None:
    found = offer(_context(line_before_caret="uniform "))
    assert found == builtin_uniform_declarations()
    assert all(name in " ".join(found) for name in ENGINE_UNIFORM_DOCS)


def test_uniform_context_filters_by_type_or_name() -> None:
    by_type = offer(_context(line_before_caret="uniform fl", prefix="fl"))
    declarations = [c for c in by_type if " " in c]
    assert declarations and all(c.startswith("float ") for c in declarations)
    assert by_type[: len(declarations)] == declarations, "declarations come first"
    by_name = offer(_context(line_before_caret="uniform u_ti", prefix="u_ti"))
    assert by_name == ["float u_time;"]


def test_an_identifier_opens_by_itself_at_two_letters_and_on_ctrl_n_at_one() -> None:
    assert offer(_context(prefix="S")) == []
    assert offer(_context(prefix="S", explicit=True)) == [
        "SB_hash",
        "SB_hash2",
        "SB_noise",
    ]
    assert offer(_context(prefix="SB_h")) == ["SB_hash", "SB_hash2"]
    assert offer(_context(prefix="u_")) == ["u_speed"]


def test_after_uniform_the_glsl_words_still_come() -> None:
    # The builtin provider fires first but does not shadow the rest: a custom uniform's type
    # is what the user types most often after `uniform`.
    assert offer(
        _context(line_before_caret="uniform sam", prefix="sam", explicit=True)
    ) == ["sampler2D"]
    found = offer(_context(line_before_caret="uniform vec", prefix="vec"))
    assert found[0] == "vec2 u_resolution;"
    assert {"vec2", "vec3", "vec4"} <= set(found)


def test_a_line_comment_offers_nothing() -> None:
    assert offer(_context(line_before_caret="// uniform ")) == []
    assert offer(_context(line_before_caret="x; // SB", prefix="SB")) == []
    assert (
        offer(_context(tab_kind="script", line_before_caret="# wh", prefix="wh")) == []
    )


def test_a_complete_word_is_not_its_own_candidate() -> None:
    assert not matches("SB_hash", "SB_hash")
    assert matches("SB_hash2", "SB_hash")


def test_the_script_tab_offers_python_keywords_only() -> None:
    found = offer(_context(tab_kind="script", prefix="wh"))
    assert found == ["while"]
    assert (
        eligible_providers(_context(tab_kind="script", line_before_caret="uniform "))
        == []
    )


def test_symbol_doc_reads_the_lib_index_then_the_builtin_table() -> None:
    lib = {
        "SB_hash": ShaderLibFunction(
            name="SB_hash",
            signature="float SB_hash(vec2 p)",
            body="float SB_hash(vec2 p) { return 0.0; }",
            file=Path("hash.glsl"),
            line_in_file=0,
            calls=frozenset(),
            doc="a hash",
        )
    }
    assert symbol_doc("SB_hash", lib) == ("float SB_hash(vec2 p)", "a hash")
    assert symbol_doc("u_time", lib) == (
        "uniform float u_time;",
        ENGINE_UNIFORM_DOCS["u_time"][1],
    )
    assert symbol_doc("mix", lib) is None


def _shader_tab(app: Any) -> EditorTab:
    document = app.ui_documents[app.current_document_id].document
    return EditorTab(
        path=document.render_pass.source.path,
        kind="shader",
        document_id=app.current_document_id,
    )


def _drive(app: Any, editor: Editor, tab: EditorTab, keys: str) -> None:
    editor.feed(keys)
    _drive_completion(app, editor, tab)


def test_typing_opens_the_popup_on_the_second_letter(app: Any) -> None:
    editor = Editor("")
    editor.set_language(Language.GLSL)
    editor.set_host_completion(True)
    tab = _shader_tab(app)
    _drive(app, editor, tab, "iS")
    assert not editor.complete_open()
    _drive(app, editor, tab, "B")
    assert editor.complete_open(), "two letters of an SB_ name must open unasked"
    assert app.editor_completion_auto
    # The next frame with no edit leaves it alone; a moving prefix re-filters.
    _drive_completion(app, editor, tab)
    assert editor.complete_open()
    _drive(app, editor, tab, "<Esc>")
    editor.close()


def test_an_accept_does_not_reopen_the_popup(app: Any) -> None:
    editor = Editor("")
    editor.set_language(Language.GLSL)
    editor.set_host_completion(True)
    tab = _shader_tab(app)
    _drive(app, editor, tab, "iSB")
    assert editor.complete_open()
    editor.key(KeyCode.DOWN)
    editor.key(KeyCode.TAB)
    accepted = editor.get_text()
    assert accepted.startswith("SB_") and not editor.complete_open()
    _drive_completion(app, editor, tab)
    assert not editor.complete_open(), "the frame after an accept must not re-offer"
    editor.close()


def test_uniform_space_opens_the_builtin_list(app: Any) -> None:
    editor = Editor("")
    editor.set_language(Language.GLSL)
    editor.set_host_completion(True)
    tab = _shader_tab(app)
    _drive(app, editor, tab, "iuniform")
    _drive(app, editor, tab, " ")
    assert editor.complete_open()
    assert editor.complete_count() == len(builtin_uniform_declarations())
    assert editor.complete_selected() == -1, "unasked: nothing highlighted"
    editor.key(KeyCode.DOWN)
    editor.key(KeyCode.ENTER)
    assert editor.get_text().startswith(
        "uniform float u_time;"
    ) or editor.get_text().startswith("uniform " + builtin_uniform_declarations()[0])
    editor.close()


def test_enter_on_an_unasked_popup_is_a_newline_until_the_user_navigates(
    app: Any,
) -> None:
    # An unasked batch is pushed with nothing highlighted (complete_select(-1)); the
    # library then treats Enter as if no popup were open. Down picks row 0 and Enter accepts.
    editor = Editor("")
    editor.set_language(Language.GLSL)
    editor.set_host_completion(True)
    tab = _shader_tab(app)
    _drive(app, editor, tab, "iSB")
    assert editor.complete_open() and app.editor_completion_auto
    assert editor.complete_selected() == -1
    editor.key(KeyCode.ENTER)
    assert editor.get_text() == "SB\n"
    assert editor.get_mode() == Mode.INSERT
    assert not editor.complete_open()
    _drive_completion(app, editor, tab)
    assert not editor.complete_open()

    _drive(app, editor, tab, "SB")
    assert editor.complete_open() and editor.complete_selected() == -1
    editor.key(KeyCode.DOWN)
    assert editor.complete_selected() == 0
    editor.key(KeyCode.ENTER)
    assert editor.get_text().startswith("SB\nSB_"), editor.get_text()

    # An explicit Ctrl+N batch keeps the library's row-0 highlight.
    app.editor_completion_requested = True
    _drive(app, editor, tab, "<CR>SB")
    assert editor.complete_open() and editor.complete_selected() == 0
    editor.close()


def test_word_at_takes_the_word_under_or_after_the_column() -> None:
    assert word_at("float a = SB_hash(p);", 12) == "SB_hash"
    assert word_at("float a = SB_hash(p);", 9) == "SB_hash"
    assert word_at("float a = SB_hash(p);", 0) == "float"
    assert word_at("   ", 1) == ""
    assert word_at("x;", 2) == ""


def test_k_is_the_lookup_key_in_normal_and_visual_mode_only() -> None:
    editor = Editor("SB_hash(p);")
    editor.set_language(Language.GLSL)
    shift_k = KeyEvent(code=KeyCode.CHAR, mods=KeyMod.SHIFT, text="K")
    assert _is_lookup_key(editor, shift_k)
    assert not editor.key(shift_k.code, shift_k.mods, shift_k.text), "K is unbound"
    editor.feed("v")
    assert _is_lookup_key(editor, shift_k)
    editor.feed("<Esc>i")
    assert not _is_lookup_key(editor, shift_k), "insert mode types the letter"
    editor.close()


def test_a_lookup_request_resolves_the_word_under_the_caret(app: Any) -> None:
    editor = Editor("uniform float u_time;\nfloat a = SB_fbm(p);")
    editor.set_language(Language.GLSL)
    editor.feed("jfS")
    app.editor_lookup_requested = True
    _consume_lookup_request(app, editor)
    assert not app.editor_lookup_requested
    assert app.editor_lookup is not None
    assert app.editor_lookup.word == "SB_fbm"
    assert "SB_fbm(" in app.editor_lookup.signature
    editor.feed("gg0fu")
    app.editor_lookup_requested = True
    _consume_lookup_request(app, editor)
    assert app.editor_lookup is not None
    assert app.editor_lookup.signature == "uniform float u_time;"
    editor.feed("0")
    app.editor_lookup_requested = True
    _consume_lookup_request(app, editor)
    assert app.editor_lookup is None, "`uniform` is a keyword with no doc"
    editor.close()
