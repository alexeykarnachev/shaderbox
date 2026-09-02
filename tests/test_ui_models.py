"""`UIDocumentState`'s persisted shape under the 069 pass-qualified stop set.

The rule this pins is the NO-MIGRATION one: a stale `list[str]` `stopped_uniforms` from before the
reshape must drop to `[]`, costing the user that one setting and nothing else — never be
reinterpreted as a pair by compat code.

The tests drive `_load_ui_state`, the GL-free half `load_document_from_dir` calls once it has the
metadata, rather than the loader itself: the loader builds a real `Document` and so needs a GL
context, and a salvage rule verified behind a GL skip is a rule nothing checks on a display-less box.
"""

from typing import Any

from shaderbox.scripting import StoppedKey
from shaderbox.ui_models import UIDocumentState, _load_ui_state


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
