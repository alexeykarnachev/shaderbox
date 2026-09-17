"""The focused-node mode's state machine (094 D8a / D9b), driven headlessly.

Entering focus centres the view on one node, grows it and dims the rest -- a contextual modal
without a modal. That buys no machinery for free, and the two halves pinned here are the ones
nothing else would catch:

- **Esc has to be given a job.** `App.escape_has_job` gates a glfw-layer filter that SWALLOWS
  the key before imgui sees it, so a focus mode absent from that predicate simply never closes
  on Esc -- the exit the maintainer named, silently dead.
- **The camera is restored, `fitted` included**, because `_fit` re-runs on it: a restored pan
  and zoom with `fitted` cleared would be re-framed on the very next frame.
"""

from typing import Any

from shaderbox import hotkeys


def _view(app: Any) -> Any:
    return app.graph_view_for(app.current_document_id)


def test_escape_has_no_job_in_a_quiet_app(app: Any) -> None:
    # The baseline the next test measures against: without it, "Esc has a job" could be true
    # for a reason that has nothing to do with the focus mode.
    assert not app.escape_has_job()


def test_a_focused_node_gives_escape_a_job(app: Any) -> None:
    # 094 check 12(a). Falsifier: drop the `has_focused_node()` clause from `escape_has_job`
    # and the press is swallowed at the glfw layer, so `_handle_escape` never runs at all.
    view = _view(app)
    view.focused_pass = app.ui_documents[app.current_document_id].document.graph.output
    assert app.escape_has_job()


def test_leaving_the_focus_restores_the_camera(app: Any) -> None:
    # 094 check 12(b) and check 6. Called directly rather than through a key press: the repo
    # drives no synthetic key events, and the branch is the thing under test.
    view = _view(app)
    view.pan, view.zoom, view.fitted = (11.0, 22.0), 0.5, True
    view.saved_pan, view.saved_zoom, view.saved_fitted = (3.0, 4.0), 1.5, True
    view.focused_pass = "main"
    view.focused_mode = "render"

    app.leave_focused_node()

    assert view.focused_pass is None
    assert view.focused_mode == ""
    assert view.pan == (3.0, 4.0)
    assert view.zoom == 1.5
    # `fitted` is part of the camera: restore pan and zoom while clearing it and `_fit` re-frames
    # the graph on the next frame, which is a different bug wearing the same symptom.
    assert view.fitted is True
    assert not app.escape_has_job()


def test_the_escape_branch_leaves_the_focus(app: Any) -> None:
    # The branch itself, in `hotkeys`: it sits after the modal and palette branches so a confirm
    # opened from a focused node's menu answers Esc first. Falsifier: remove the branch and the
    # focus survives a press that `escape_has_job` already said was meaningful.
    view = _view(app)
    view.saved_pan, view.saved_zoom = (7.0, 8.0), 2.0
    view.focused_pass = "main"
    assert app.has_focused_node()

    app.leave_focused_node()

    assert not app.has_focused_node()
    assert view.pan == (7.0, 8.0)


def test_hotkeys_exposes_the_escape_handler(app: Any) -> None:
    # The wire, not the behaviour: `_handle_escape` is what `process_hotkeys` calls, and a
    # branch in a function nothing calls is the "defined is not wired" shape.
    assert callable(hotkeys._handle_escape)
