"""The jedi worker's contract (078 D10): newest request per kind wins, an answer is applied
only to the caret it was asked for, and a dropped answer releases the latch so the site is
asked again."""

import time
from pathlib import Path
from typing import Any

from shaderbox.intel.worker import PythonRequest, PythonRequestKind, PythonWorker
from shaderbox.tabs.code import _drive_completion


def _wait(predicate: Any, seconds: float = 10.0) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("the worker did not answer in time")


def test_the_newest_request_of_a_kind_replaces_an_older_one_waiting() -> None:
    worker = PythonWorker()
    text = "import math\nx = math.s\n"
    for column in (7, 8, 9):
        worker.submit(
            PythonRequest(PythonRequestKind.COMPLETE, Path("s.py"), text, 1, column, 1)
        )
    results: list[Any] = []
    _wait(lambda: results.extend(worker.poll()) or bool(results))
    time.sleep(0.2)
    results.extend(worker.poll())
    answered = sorted({r.request.column for r in results})
    assert answered[-1] == 9, "the newest request is what got answered"
    assert len(answered) <= 2, (
        "a burst costs at most the in-flight call plus the newest"
    )
    worker.close()


def test_a_dropped_answer_releases_the_latch_so_the_site_asks_again(app: Any) -> None:
    app.open_script_for(app.current_document_id)
    tab = app.active_tab
    session = app.get_session_for_path(app.current_editor_path)
    editor = session.editor
    editor.set_host_completion(True)
    editor.feed("Go")
    editor.feed("        x = ctx.")
    _drive_completion(app, editor, tab)
    asked = app.python_last_request
    assert asked is not None and asked.kind == PythonRequestKind.COMPLETE
    # The caret moves while the answer is in flight; the answer must be dropped.
    editor.feed("<Esc>")
    _wait(lambda: bool(app.python_worker._results.qsize()))
    _drive_completion(app, editor, tab)
    assert app.python_candidates is None, "an answer for another caret is dropped"
    assert app.python_last_request is None, "and the latch is released"
    # Back at the same site (same text, same caret), Ctrl+N asks again and the popup opens;
    # with the latch still held nothing would be sent and Ctrl+N would do nothing.
    editor.feed("a")
    assert editor.get_current_cursor_position().column == asked.column
    app.editor_completion_requested = True

    def opened() -> bool:
        _drive_completion(app, editor, tab)
        return editor.complete_open()

    _wait(opened)
    assert app.python_candidates is not None
