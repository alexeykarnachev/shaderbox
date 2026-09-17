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


def test_a_focused_node_refuses_a_background_press(app: Any) -> None:
    """094 check 5(a): the scrim is modal to the HAND, not only to the eye.

    Driven as a real press on the canvas background, because the state-only version of this
    test was green against its own falsifier -- asserting what entry cleared says nothing about
    whether a LATER press can rebuild it, which is the thing the scrim exists to stop.

    Falsifier: drop `focused` from `frozen` and the press below starts a rubber band.
    """
    from imgui_bundle import imgui

    from shaderbox.ui import update_and_draw

    document_id = app.current_document_id
    app.open_graph_for(document_id)
    for _ in range(4):
        update_and_draw(app)
    view = app.graph_view_for(document_id)
    assert view.canvas_rect[2] > view.canvas_rect[0], "the canvas never drew"

    app.focus_node(
        document_id, app.ui_documents[document_id].document.graph.output, "render"
    )

    io = imgui.get_io()
    cx1, cy1 = view.canvas_rect[2], view.canvas_rect[3]
    io.add_mouse_pos_event(cx1 - 8.0, cy1 - 8.0)
    for _ in range(2):
        update_and_draw(app)
    io.add_mouse_button_event(0, True)
    for _ in range(2):
        update_and_draw(app)
    io.add_mouse_pos_event(cx1 - 60.0, cy1 - 60.0)
    for _ in range(2):
        update_and_draw(app)

    assert view.band_anchor is None, "a press behind the scrim started a rubber band"
    assert view.node_drag is None
    assert view.wire_drag is None

    io.add_mouse_button_event(0, False)
    for _ in range(2):
        update_and_draw(app)
    app.close_editor_for_path(app.paths.graph_json_for(document_id))
    update_and_draw(app)


def test_entering_a_focus_cancels_a_live_drag(app: Any) -> None:
    """094 D9b: a drag surviving into focus commits a position behind the scrim.

    The node written would be whichever was under the hand, not the focused one -- so D7's
    footprint check would not catch it. Falsifier: drop the cancel from `focus_node`.
    """
    document_id = app.current_document_id
    view = app.graph_view_for(document_id)
    view.band_anchor = (10.0, 10.0)
    view.guides = [("v", 1.0)]

    app.focus_node(document_id, "main", "render")

    assert view.band_anchor is None
    assert view.guides == []


def test_the_focus_saves_the_camera_once_across_a_mode_change(app: Any) -> None:
    """Switching modes must not overwrite the saved camera with the FOCUSED one.

    Falsifier: drop the `if view.focused_pass is None` guard in `focus_node` and leaving after a
    mode switch restores the focus camera instead of the one the user had.
    """
    document_id = app.current_document_id
    view = app.graph_view_for(document_id)
    view.pan, view.zoom = (5.0, 6.0), 0.8

    app.focus_node(document_id, "main", "render")
    view.pan, view.zoom = (99.0, 99.0), 2.5  # the focus camera
    app.focus_node(document_id, "main", "share")
    app.leave_focused_node()

    assert view.pan == (5.0, 6.0)
    assert view.zoom == 0.8
