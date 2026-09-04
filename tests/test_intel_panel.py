"""The intel index through the real code panel (078 W-A): the color feed on a shader tab and
the jedi worker's answers on a script tab, both driven the way a frame drives them."""

import time
from typing import Any

from shaderbox.editor.ffi import ensure_loaded
from shaderbox.editor_types import EditorTab
from shaderbox.intel.worker import PythonRequestKind
from shaderbox.tabs.code import (
    _consume_lookup_request,
    _drive_completion,
    _glsl_index_for,
)


def _shader_session(app: Any) -> tuple[Any, EditorTab]:
    app.ensure_shader_tab(app.current_document_id)
    tab = app.active_tab
    assert tab is not None and tab.kind == "shader"
    return app.get_session_for_path(app.current_editor_path), tab


def _script_session(app: Any) -> tuple[Any, EditorTab]:
    app.open_script_for(app.current_document_id)
    tab = app.active_tab
    assert tab is not None and tab.kind == "script"
    return app.get_session_for_path(app.current_editor_path), tab


def _wait_for(predicate: Any, seconds: float = 10.0) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("the worker did not answer in time")


def test_the_index_colors_engine_uniforms_in_the_text(app: Any) -> None:
    session, tab = _shader_session(app)
    editor = session.editor
    lines = editor.get_text().split("\n")
    editor.set_selection((0, 0), (len(lines) - 1, len(lines[-1])))
    editor.replace_selection(
        "uniform float u_time;\nuniform float u_gain;\nvoid main() { float a = u_gain * u_time; }\n"
    )
    index = _glsl_index_for(app, editor, tab)
    assert index.lookup("u_gain") is not None
    editor.layout((800.0, 600.0), 16.0)
    lib = ensure_loaded()
    assert lib.ed_class_at(editor._h, 0, 14) == 7, "u_time draws in the engine slot"
    assert lib.ed_class_at(editor._h, 1, 14) == 0, (
        "a plain uniform keeps the lexer's class"
    )
    assert lib.ed_class_at(editor._h, 0, 0) == 1, "the keyword keeps its class"


def test_the_script_tab_completes_ctx_members_through_the_worker(app: Any) -> None:
    session, tab = _script_session(app)
    editor = session.editor
    editor.set_host_completion(True)
    editor.feed("Go")
    editor.feed("        x = ctx.")
    _drive_completion(app, editor, tab)
    assert not editor.complete_open(), "the answer is on its way, nothing is guessed"
    assert app.python_last_request is not None
    assert app.python_last_request.kind == PythonRequestKind.COMPLETE

    def answered() -> bool:
        _drive_completion(app, editor, tab)
        return editor.complete_open()

    _wait_for(answered)
    items = [editor.complete_item(i) for i in range(editor.complete_count())]
    assert items[:4] == ["dt", "frame", "mouse", "t"]
    assert app.editor_completion_offered["t"].doc == "seconds"


def test_k_on_the_script_tab_answers_through_the_worker(app: Any) -> None:
    session, tab = _script_session(app)
    editor = session.editor
    editor.feed("Go")
    editor.feed("        y = math.sin")
    editor.feed("<Esc>")
    app.editor_lookup_requested = True
    _consume_lookup_request(app, editor, tab)
    assert app.editor_lookup is None, "answered by the worker, not on this frame"

    def landed() -> bool:
        _drive_completion(app, editor, tab)
        return app.editor_lookup is not None

    _wait_for(landed)
    assert app.editor_lookup is not None
    assert app.editor_lookup.word == "sin"
    assert "sin" in app.editor_lookup.signature
