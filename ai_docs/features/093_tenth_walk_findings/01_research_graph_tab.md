# 093 — Research: the graph as an editor tab

The maintainer picked mock C (`00_mock_panel.html`): the graph moves into the editor pane as a
third tab KIND beside a document's shader tabs and its script tab, one graph tab per document.
This records what the tab machinery requires, read from the code on 2026-09-13, so the wave
that lands it does not re-derive it. It reverses 092 D2 (the Document-tab toggle, with the
editor-tab route deferred "when the zen mode or pane swap arrives"); the wave records the
reversal in 092's spec.

## Why it fits without tricks

`EditorTab` is `(path, kind, document_id)` and `path` is the tab's identity everywhere: the
imgui id (`tabs/code.py::_tab_id_suffix`), the dedupe (`App._focus_or_add_tab`), the teardown
(`App.close_editor_for_path`). A document's graph is a real file, `graph.json`, one per document
(`paths.graph_json_for`), so the tab is `EditorTab(path=graph.json, kind="graph",
document_id=id)`, exactly as a script tab carries `script.py`. Verified pass-throughs:
`close_editor_for_path` pops the session with a `None` default; `_on_document_deleted` filters
tabs by `document_id`; the rename and file-sync hooks re-key by pass path and never match;
`is_tab_dirty` finds no session and answers False (graph writes save immediately);
`formatter_for("graph")` returns `None`. Tabs are not persisted across restarts (only the
current document's shader tab is reopened on init); the copilot never reads the tab list.

## The one seam and the branches

- `tabs/code.py::draw` is a text-editor body: after `_draw_tab_row` it fetches or creates an
  `EditorSession` for the current path. It dispatches on the active tab's kind right after the
  row: the text body for `shader` / `script` / `lib`, `widgets/pass_graph.draw(app,
  document_id, height)` for `graph`.
- `tab_label`: `<document> (graph)`.
- `App.open_graph_for(document_id)`, the sibling of `open_script_for`, and an `OPEN_GRAPH`
  command beside `OPEN_SHADER` / `OPEN_SCRIPT`.
- The vim chord handler in `hotkeys.py` and `JUMP_NEXT_ERROR` return when the current tab has
  no session; `editor_focused` keeps meaning "the left pane has focus" (right for `Ctrl+W` and
  `Ctrl+Tab`, which cycle onto the graph tab by design).
- The Document tab's `strip | graph` toggle and `UIAppState.passes_view` retire; the tab keeps
  the strip and gets an `open` for the graph like the Script row.
- Tests: a bufferless tab's label, dedupe, close, the format and error-jump guards, the
  dispatch; the smoke keeps its editor-drew assert on the shader tab.

The toolbar consolidation of mock A is separable: with the graph out of the Document tab, that
tab holds the name, the canvas, the script row and the strip.
