"""Every tracked `document.json` is in the 090 shape, and carries the mode it is meant to.

The shape is `resolution_mode` + `aspect` + `resolution` (090 revision 1): the aspect is what an
Auto document renders by, the pair is what Fixed renders at and what a switch to Fixed seeds
from, and the two are kept consistent so neither mode opens on a shape the other never had.

Without this gate a document missed by the sweep loads at `DEFAULT_CANVAS_SIZE` = (64, 64) in
SILENCE: `_load_ui_state` fills a missing key from the model's default and never raises, so a
forgotten file renders at a 64-pixel canvas and nothing anywhere says why.

The domain is `git ls-files`, not a directory walk. A local run writes documents under the
untracked `projects/documents/`, and a walk would fail red on the maintainer's box over a file
that does not exist in the repository at all.
"""

import json
import subprocess
from pathlib import Path

import pytest

from shaderbox.render_shape import aspect_of

_ROOT = Path(__file__).resolve().parent.parent

# The mode each tracked document carries. `77a84d27-...` is the one Fixed document: its JFA and
# radiance-cascade PASSES have iteration counts that follow pixel counts, so it is the one
# example that must not be resized to whatever region is showing it.
_FIXED_DOCUMENTS = {"77a84d27-2e5b-406d-8011-ee1cb1a9587c"}


def _tracked_document_jsons() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "*document.json"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return sorted(line for line in out.stdout.split() if line)


def test_git_tracks_the_documents_this_gate_expects() -> None:
    # A census guard on the gate itself: parametrizing over an empty list is a green run that
    # checked nothing, which is the shape a `git` that failed to run would produce.
    paths = _tracked_document_jsons()
    assert len(paths) >= 11, f"only {len(paths)} tracked document.json files: {paths}"
    fixed_seen = {p for p in paths if any(f in p for f in _FIXED_DOCUMENTS)}
    assert fixed_seen, "the Fixed example is not among the tracked documents"


@pytest.mark.parametrize("path", _tracked_document_jsons())
def test_a_tracked_document_carries_the_new_resolution_keys(path: str) -> None:
    # Falsifier: leave one file's top-level `canvas_size` in place, or drop either key from its
    # `ui_state`, and this names that file.
    data = json.loads((_ROOT / path).read_text())
    assert "canvas_size" not in data, (
        f"{path} still carries the top-level canvas_size 090 D1 replaced"
    )
    ui_state = data.get("ui_state")
    assert isinstance(ui_state, dict), f"{path} has no ui_state object"
    assert "resolution_mode" in ui_state, f"{path} carries no resolution_mode"
    for key in ("resolution", "aspect"):
        pair = ui_state.get(key)
        assert isinstance(pair, list) and len(pair) == 2, (
            f"{path} carries no {key} pair"
        )
        assert all(isinstance(n, int) and n > 0 for n in pair), (
            f"{path} carries a malformed {key} {pair}"
        )
    # Every document carries BOTH: the aspect is what Auto renders by, the pair is what a
    # switch to Fixed seeds from, and a document missing either loses one of the two modes.
    assert tuple(ui_state["aspect"]) == aspect_of(tuple(ui_state["resolution"])), (
        f"{path}'s aspect {ui_state['aspect']} is not the reduced ratio of its "
        f"resolution {ui_state['resolution']}"
    )


@pytest.mark.parametrize("path", _tracked_document_jsons())
def test_a_tracked_document_carries_the_mode_it_is_meant_to(path: str) -> None:
    # The mode per file, not merely that the key exists: shipping the pixel-dependent example
    # as Auto renders it at whatever region shows it, and its JFA and cascade iteration counts
    # follow pixel counts. Falsifier: flip one file's mode and this names it.
    ui_state = json.loads((_ROOT / path).read_text())["ui_state"]
    expected = "fixed" if any(f in path for f in _FIXED_DOCUMENTS) else "auto"
    assert ui_state["resolution_mode"] == expected, (
        f"{path} is {ui_state['resolution_mode']}, and it must be {expected}"
    )
