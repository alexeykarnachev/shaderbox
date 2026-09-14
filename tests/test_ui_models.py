"""`UIDocumentState`'s persisted shape under the 069 pass-qualified stop set.

The rule this pins is the NO-MIGRATION one: a stale `list[str]` `stopped_uniforms` from before the
reshape must drop to `[]`, costing the user that one setting and nothing else — never be
reinterpreted as a pair by compat code.

The tests drive `_load_ui_state`, the GL-free half `load_document_from_dir` calls once it has the
metadata, rather than the loader itself: the loader builds a real `Document` and so needs a GL
context, and a salvage rule verified behind a GL skip is a rule nothing checks on a display-less box.
"""

import json
from pathlib import Path
from typing import Any

from shaderbox.editor_types import (
    EditorTab,
    TabRecord,
    tab_records,
    tabs_from_records,
)
from shaderbox.scripting import StoppedKey
from shaderbox.ui_models import UIAppState, UIDocumentState, _load_ui_state


def test_a_stale_string_stopped_set_drops_to_empty() -> None:
    # Falsifier: any migration code in this path that reinterprets a bare string as a pair — the
    # first assertion goes red. This is what makes the no-migration rule mechanical rather than a
    # promise.
    state = _load_ui_state({"stopped_uniforms": ["u_x"], "all_stopped": True}, "doc")

    assert state.stopped_uniforms == []
    # The SIBLING key survives — that is the whole point of per-key salvage.
    assert state.all_stopped is True


def test_a_well_formed_pair_round_trips() -> None:
    # Falsifier: a field type that cannot hold the pair — the round trip loses it.
    state = _load_ui_state(
        {"stopped_uniforms": [{"pass_name": "paint", "name": "u_x"}]}, "doc"
    )

    assert state.stopped_uniforms == [StoppedKey(pass_name="paint", name="u_x")]
    assert state.model_dump()["stopped_uniforms"] == [
        {"pass_name": "paint", "name": "u_x"}
    ]


def test_an_empty_stopped_set_is_valid_under_the_pair_shape() -> None:
    # Every tracked document.json on disk holds `[]` (or omits the key), so the reshape changes no
    # bytes and the first launch after it logs no salvage line. Falsifier: a field type that
    # rejects the empty list.
    assert _load_ui_state({"stopped_uniforms": []}, "doc").stopped_uniforms == []
    assert _load_ui_state({}, "doc").stopped_uniforms == []
    assert UIDocumentState().stopped_uniforms == []


def test_an_unknown_key_is_pruned_and_the_rest_survives() -> None:
    # The unknown-key filter and the per-key salvage are ONE path; a change to either must keep the
    # sibling keys. Falsifier: drop the whole state on an unknown key.
    state: Any = _load_ui_state({"gone_field": 1, "all_stopped": True}, "doc")
    assert state.all_stopped is True


# ---- the editor's open tabs, persisted (093 W2-2) -------------------------------------------


def test_the_open_tabs_round_trip_through_the_app_state(tmp_path: Path) -> None:
    # W2-2: the whole point of the field -- what the last session had open is what reopens.
    # Falsifier: drop the field from `UIAppState` and the loaded state carries `[]`.
    state = UIAppState()
    assert state.editor_tabs == []
    assert state.active_tab_index == 0
    state.editor_tabs = [
        TabRecord(path="/p/doc/passes/main.frag.glsl", kind="shader", document_id="d1"),
        TabRecord(path="/p/doc/scripts/script.py", kind="script", document_id="d1"),
        TabRecord(path="/p/doc/graph.json", kind="graph", document_id="d1"),
        TabRecord(path="/lib/sdf.glsl", kind="lib"),
    ]
    state.active_tab_index = 2
    state.save(tmp_path / "app_state.json")
    loaded = UIAppState.load(tmp_path / "app_state.json")
    assert [(r.path, r.kind, r.document_id) for r in loaded.editor_tabs] == [
        ("/p/doc/passes/main.frag.glsl", "shader", "d1"),
        ("/p/doc/scripts/script.py", "script", "d1"),
        ("/p/doc/graph.json", "graph", "d1"),
        ("/lib/sdf.glsl", "lib", ""),
    ]
    assert loaded.active_tab_index == 2


def test_one_malformed_tab_record_costs_only_itself(tmp_path: Path) -> None:
    # The persistence rule this repo learned the expensive way: a bad row costs that row, never
    # the file. `drop_invalid` descends into a list of models for exactly this. Falsifier: let
    # the list validate as a whole and the two good tabs go with the bad one.
    path = tmp_path / "app_state.json"
    path.write_text(
        json.dumps(
            {
                "editor_tabs": [
                    {"path": "/a", "kind": "shader", "document_id": "d1"},
                    {"path": "/b", "kind": "not_a_kind"},
                    {"path": "/c", "kind": "lib"},
                ],
                "active_tab_index": 1,
                "global_target_fps": 90,
            }
        )
    )
    loaded = UIAppState.load(path)
    assert [r.path for r in loaded.editor_tabs] == ["/a", "/c"]
    assert loaded.global_target_fps == 90, "a bad tab row cost an unrelated setting"


def test_a_record_whose_file_or_document_is_gone_is_dropped(tmp_path: Path) -> None:
    """W2-2: a restored tab must still address something.

    A tab at a deleted pass would eat its own edits, and a tab of a document this project no
    longer holds has nothing to draw. A lib tab carries no document id and so passes that
    clause trivially. Falsifier: return the records unfiltered -- a fresh clone of the project
    reopens tabs onto files that were never copied.
    """
    records = [
        TabRecord(path="/live/a.glsl", kind="shader", document_id="d1"),
        TabRecord(path="/gone/b.glsl", kind="shader", document_id="d1"),
        TabRecord(path="/live/c.glsl", kind="shader", document_id="deleted"),
        TabRecord(path="/live/lib.glsl", kind="lib"),
        TabRecord(path="/live/a.glsl", kind="shader", document_id="d1"),
    ]
    kept = tabs_from_records(
        records,
        frozenset({"d1"}),
        lambda p: not str(p).startswith("/gone"),
    )
    assert [(str(t.path), t.kind) for t in kept] == [
        ("/live/a.glsl", "shader"),
        ("/live/lib.glsl", "lib"),
    ], kept


def test_the_live_tabs_mirror_into_records_in_order() -> None:
    # The save half. Falsifier: mirror a set -- the order the user arranged is lost.
    tabs = [
        EditorTab(path=Path("/x.glsl"), kind="shader", document_id="d1"),
        EditorTab(path=Path("/y.json"), kind="graph", document_id="d2"),
        EditorTab(path=Path("/z.glsl"), kind="lib"),
    ]
    assert [(r.path, r.kind, r.document_id) for r in tab_records(tabs)] == [
        ("/x.glsl", "shader", "d1"),
        ("/y.json", "graph", "d2"),
        ("/z.glsl", "lib", ""),
    ]
