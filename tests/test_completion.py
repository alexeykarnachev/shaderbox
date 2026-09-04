"""Completion policy over the intel index (073 W-B, re-based in 078 W-A): what the code
panel offers and when, on the pure providers and through the real driver."""

from typing import Any

from shaderbox.completion import (
    DECLARATION_SITE,
    CompletionContext,
    eligible_providers,
    matches,
    offer,
    word_at,
)
from shaderbox.editor.ffi import Editor, KeyCode, KeyMod, Language, Mode
from shaderbox.editor.input import KeyEvent
from shaderbox.editor_types import EditorTab
from shaderbox.engine_uniforms import ENGINE_UNIFORM_TYPES
from shaderbox.help_content import ENGINE_UNIFORM_DOCS
from shaderbox.hotkeys import _is_lookup_key
from shaderbox.intel.index import GlslContext, GlslIndex, build_glsl_index
from shaderbox.intel.script import ScriptReturn
from shaderbox.intel.symbols import Symbol, SymbolKind
from shaderbox.tabs.code import (
    _consume_lookup_request,
    _drive_completion,
    _glsl_index_for,
)

_TEXT = (
    "uniform float u_time;\nuniform float u_gain;\nvoid main() { float a = u_gain; }\n"
)


def _index(text: str = _TEXT, **overrides: Any) -> GlslIndex:
    context = GlslContext(
        text=text,
        engine_types=ENGINE_UNIFORM_TYPES,
        engine_docs=ENGINE_UNIFORM_DOCS,
        lib_functions={
            "SB_hash": ("float SB_hash(vec2 p)", "a hash"),
            "SB_hash2": ("vec2 SB_hash2(vec2 p)", "a hash pair"),
            "SB_noise": ("float SB_noise(vec2 p)", "value noise"),
        },
        script_returns=(ScriptReturn("u_speed", None, "float", 9),),
        pass_name="main",
        passes=("main", "paint"),
        **overrides,
    )
    return build_glsl_index(context)


def _context(**overrides: Any) -> CompletionContext:
    base: dict[str, Any] = {
        "tab_kind": "shader",
        "line_before_caret": "",
        "prefix": "",
        "explicit": False,
        "index": _index(),
    }
    base.update(overrides)
    return CompletionContext(**base)


def _texts(symbols: list[Symbol]) -> list[str]:
    return [s.inserted for s in symbols]


def test_the_declaration_site_fires_on_the_maintainers_lines_and_not_past_them() -> (
    None
):
    table = {
        "uniform ": True,
        "uniform u_": True,
        "uniform vec4 u_": True,
        "uniform sampler2D u_": True,
        "vec3 color = u_": False,
        "uniform vec4 u_x y": False,
    }
    for line, fires in table.items():
        assert (DECLARATION_SITE.search(line) is not None) == fires, line


def test_after_uniform_the_buffer_lacks_are_offered_type_and_name() -> None:
    found = _texts(offer(_context(line_before_caret="uniform ")))
    assert "float u_time;" not in found, "already declared"
    assert "float u_aspect;" in found
    assert "vec2 u_resolution;" in found
    assert "float u_speed;" in found, "the script's return"
    assert "sampler2D u_paint;" in found and "sampler2D u_prev;" in found
    kinds = {s.inserted: s.kind for s in offer(_context(line_before_caret="uniform "))}
    assert kinds["float u_aspect;"] == SymbolKind.ENGINE_UNIFORM
    assert kinds["float u_speed;"] == SymbolKind.SCRIPT_UNIFORM
    assert kinds["sampler2D u_paint;"] == SymbolKind.WIRABLE_SAMPLER


def test_after_a_typed_type_only_names_of_that_type_come_and_only_the_name_lands() -> (
    None
):
    found = _texts(offer(_context(line_before_caret="uniform vec4 u_", prefix="u_")))
    assert found == [], "no vec4 is missing from this buffer"
    found = _texts(offer(_context(line_before_caret="uniform float u_", prefix="u_")))
    assert found[:3] == ["u_aspect;", "u_pass_iteration;", "u_pass_iterations;"]
    assert "u_speed;" in found
    found = _texts(
        offer(_context(line_before_caret="uniform sampler2D u_", prefix="u_"))
    )
    assert found == ["u_paint;", "u_prev;"]


def test_a_bare_uniform_site_never_offers_a_name_it_offers_whole() -> None:
    # `uniform u_`: the declarations come whole; a bare `u_time` there would land a typeless
    # `uniform u_time`. The type words still come after a bare `uniform`.
    found = _texts(offer(_context(line_before_caret="uniform u_", prefix="u_")))
    assert found and all(" " in text for text in found), found
    assert "float u_aspect;" in found
    found = _texts(
        offer(_context(line_before_caret="uniform sam", prefix="sam", explicit=True))
    )
    assert "sampler2D" in found


def test_an_identifier_site_offers_the_buffers_own_names_first() -> None:
    # Finding 6: `u_gain` is declared and read, `u_time` declared; both are offered, and
    # never only `u_time`. A declared name inserts as itself, an undeclared one as a name.
    found = _texts(offer(_context(line_before_caret="vec3 color = u_", prefix="u_")))
    assert found[:2] == ["u_time", "u_gain"]
    assert "u_aspect" in found and "u_speed" in found and "u_paint" in found


def test_an_identifier_opens_by_itself_at_two_letters_and_on_ctrl_n_at_one() -> None:
    assert offer(_context(prefix="S")) == []
    assert _texts(offer(_context(prefix="S", explicit=True)))[:3] == [
        "SB_hash",
        "SB_hash2",
        "SB_noise",
    ]
    assert _texts(offer(_context(prefix="SB_h"))) == ["SB_hash", "SB_hash2"]


def test_after_uniform_the_glsl_words_still_come() -> None:
    after_uniform = _texts(
        offer(_context(line_before_caret="uniform sam", prefix="sam", explicit=True))
    )
    assert "sampler2D" in after_uniform, after_uniform
    found = _texts(offer(_context(line_before_caret="uniform vec", prefix="vec")))
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


def test_the_script_tab_offers_what_the_worker_answered() -> None:
    answered = (
        Symbol("while", SymbolKind.PY_KEYWORD),
        Symbol("width", SymbolKind.PY_LOCAL),
    )
    found = offer(
        _context(
            tab_kind="script",
            index=None,
            line_before_caret="    wh",
            prefix="wh",
            python_candidates=answered,
        )
    )
    assert _texts(found) == ["while"]
    assert (
        offer(
            _context(tab_kind="script", index=None, line_before_caret="wh", prefix="wh")
        )
        == []
    ), "nothing is guessed while the worker's answer is on its way"
    # A member site opens by itself with nothing typed after the dot; a blank line does not.
    members = (Symbol("t", SymbolKind.PY_MEMBER), Symbol("dt", SymbolKind.PY_MEMBER))
    assert _texts(
        offer(
            _context(
                tab_kind="script",
                index=None,
                line_before_caret="x = ctx.",
                prefix="",
                python_candidates=members,
            )
        )
    ) == ["t", "dt"]
    assert (
        offer(
            _context(
                tab_kind="script",
                index=None,
                line_before_caret="x = ",
                prefix="",
                python_candidates=members,
            )
        )
        == []
    )
    assert (
        eligible_providers(
            _context(tab_kind="script", index=None, line_before_caret="uniform ")
        )
        == []
    )


def test_the_index_explains_the_lib_the_engine_and_the_language() -> None:
    index = _index()
    assert index.lookup("SB_hash").signature == "float SB_hash(vec2 p)"
    assert index.lookup("u_time").doc == ENGINE_UNIFORM_DOCS["u_time"]
    assert index.lookup("u_aspect").signature == "uniform float u_aspect;"
    assert index.lookup("mix") is not None, "GLSL builtins are documented too"
    assert index.lookup("nosuchname") is None


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
    assert editor.complete_count() == len(
        _glsl_index_for(app, editor, tab).declarations
    )
    assert editor.complete_selected() == -1, "unasked: nothing highlighted"
    editor.key(KeyCode.DOWN)
    editor.key(KeyCode.ENTER)
    assert editor.get_text().startswith("uniform float u_time;")
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


def test_a_ctrl_n_batch_keeps_its_highlight_while_the_word_continues(app: Any) -> None:
    # A typed character closes a host-driven popup; the next frame re-offers the SAME batch
    # kind: explicit stays explicit (row 0), unasked stays unasked (nothing highlighted).
    editor = Editor("")
    editor.set_language(Language.GLSL)
    editor.set_host_completion(True)
    tab = _shader_tab(app)
    editor.feed("iS")
    app.editor_completion_requested = True
    _drive_completion(app, editor, tab)
    assert editor.complete_open() and editor.complete_selected() == 0
    for key in "B_f":
        _drive(app, editor, tab, key)
        assert editor.complete_open(), key
        assert editor.complete_selected() == 0, key
    _drive(app, editor, tab, "<BS>")
    assert editor.complete_open() and editor.complete_selected() == 0, "backspace"
    _drive(app, editor, tab, "f")
    assert editor.complete_open() and editor.complete_selected() == 0
    editor.key(KeyCode.ENTER)
    assert editor.get_text().startswith("SB_f") and "\n" not in editor.get_text()
    _drive_completion(app, editor, tab)  # the accept's own frame re-offers nothing
    assert not editor.complete_open()

    _drive(app, editor, tab, "<Esc>oSB")
    assert editor.complete_open() and editor.complete_selected() == -1
    _drive(app, editor, tab, "_")
    assert editor.complete_open() and editor.complete_selected() == -1
    editor.key(KeyCode.ENTER)
    assert editor.get_text().endswith("SB_\n")
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
    tab = _shader_tab(app)
    editor.feed("jfS")
    app.editor_lookup_requested = True
    _consume_lookup_request(app, editor, tab)
    assert not app.editor_lookup_requested
    assert app.editor_lookup is not None
    assert app.editor_lookup.word == "SB_fbm"
    assert "SB_fbm(" in app.editor_lookup.signature
    editor.feed("gg0fu")
    app.editor_lookup_requested = True
    _consume_lookup_request(app, editor, tab)
    assert app.editor_lookup is not None
    assert app.editor_lookup.signature == "uniform float u_time;"
    editor.feed("0")
    app.editor_lookup_requested = True
    _consume_lookup_request(app, editor, tab)
    assert app.editor_lookup is not None
    assert app.editor_lookup.doc == "GLSL keyword"
    editor.close()


def test_every_offered_word_can_be_explained() -> None:
    # Every symbol the index offers carries what `K` shows for it. Enumerated, not sampled.
    index = _index()
    unexplained = [
        s.name
        for s in index.words
        if not (s.signature or s.doc) and s.kind != SymbolKind.BUFFER_SYMBOL
    ]
    assert unexplained == [], unexplained


def test_the_generated_table_covers_the_builtins_a_shader_uses() -> None:
    from shaderbox.glsl_docs import BUILTINS

    for name in ("mix", "smoothstep", "texture", "dot", "clamp", "dFdx", "textureLod"):
        signatures, purpose = BUILTINS[name]
        assert signatures and all(f"{name}(" in s for s in signatures), name
        assert purpose, name
    assert len(BUILTINS["mix"][0]) == 3


def test_the_index_answers_for_a_glsl_builtin() -> None:
    found = _index().lookup("smoothstep")
    assert found is not None and "smoothstep(" in found.signature
    assert "\n" in found.signature, "every overload is shown, one per line"
    assert found.doc
    assert _index().lookup("vec3").doc == "GLSL type"
    assert _index().lookup("discard").doc == "GLSL keyword"
