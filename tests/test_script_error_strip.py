"""The script engine's soft errors, adapted into the shared bottom strip (069 W-G).

`tabs/code.py` has two adapters: the SCRIPT tab shows every soft error whatever pass it names (the
script is one file and its author wants all of them), and a SHADER tab shows only the errors naming
its own pass, without the redundant pass prefix. Both are pure over a `ScriptStatus`, so they test
without imgui — only the two free functions are exercised, not the draw.
"""

import types
from pathlib import Path
from typing import Any

from shaderbox.scripting import ScriptError, ScriptStatus
from shaderbox.tabs.code import _script_errors_for, _script_errors_for_pass


def _status(soft: list[tuple[str, str, ScriptError]]) -> ScriptStatus:
    return ScriptStatus(sentinel_error=None, driven_count=0, soft_errors=soft)


def _app(status: ScriptStatus) -> Any:
    return types.SimpleNamespace(
        session=types.SimpleNamespace(
            get_script_status=lambda _id: status,
            script_path_for=lambda _id: Path("documents/d/scripts/script.py"),
        )
    )


def _tab(path: str) -> Any:
    return types.SimpleNamespace(document_id="d", path=Path(path), kind="shader")


_PAINT_ORPHAN = ScriptError(
    "u_brsh",
    "runtime",
    "pass 'paint' has no active uniform 'u_brsh' (orphan key)",
    pass_name="paint",
)
_BARE_ORPHAN = ScriptError(
    "u_brsh", "runtime", "no pass declares 'u_brsh' (orphan key)"
)


def test_the_script_tab_shows_every_soft_error_with_its_pass() -> None:
    # Falsifier: filter the script tab to one pass — the second row disappears.
    rows = _script_errors_for(
        _app(
            _status([("paint", "u_brsh", _PAINT_ORPHAN), ("", "u_brsh", _BARE_ORPHAN)])
        ),
        _tab("documents/d/scripts/script.py"),
    )
    assert [r.message for r in rows] == [
        "paint.u_brsh: pass 'paint' has no active uniform 'u_brsh' (orphan key)",
        "u_brsh: no pass declares 'u_brsh' (orphan key)",
    ]


def test_a_shader_tab_shows_only_its_own_pass_errors() -> None:
    # The asymmetry is deliberate: a bare key is a claim about the whole DOCUMENT, so it belongs on
    # the document's own script tab and on no shader tab. Falsifier: drop the pass filter — the
    # composite tab grows rows that are not about it.
    app = _app(
        _status([("paint", "u_brsh", _PAINT_ORPHAN), ("", "u_brsh", _BARE_ORPHAN)])
    )

    paint = _script_errors_for_pass(app, _tab("passes/paint.frag.glsl"), "paint")
    # No pass prefix — the tab already names the pass.
    assert [r.message for r in paint] == [
        "u_brsh: pass 'paint' has no active uniform 'u_brsh' (orphan key)"
    ]

    assert (
        _script_errors_for_pass(app, _tab("passes/composite.frag.glsl"), "composite")
        == []
    )


def test_a_shader_tab_script_row_points_at_the_script_file() -> None:
    # The row carries the SCRIPT path, not the shader's, because that is where the fix is — which is
    # also why the click branch must open it first. Falsifier: carry `tab.path`; the row would jump
    # into the shader, and `_apply_markers` would line-fill the wrong file.
    rows = _script_errors_for_pass(
        _app(_status([("paint", "u_brsh", _PAINT_ORPHAN)])),
        _tab("passes/paint.frag.glsl"),
        "paint",
    )
    assert [r.path for r in rows] == [Path("documents/d/scripts/script.py")]


def test_no_script_means_no_rows() -> None:
    # A document with no script.py has no status at all. Falsifier: assume a status exists.
    app = types.SimpleNamespace(
        session=types.SimpleNamespace(
            get_script_status=lambda _id: None,
            script_path_for=lambda _id: Path("x"),
        )
    )
    assert _script_errors_for(app, _tab("passes/paint.frag.glsl")) == []
    assert _script_errors_for_pass(app, _tab("passes/paint.frag.glsl"), "paint") == []
