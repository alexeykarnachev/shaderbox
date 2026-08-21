"""The node-dir basenames have exactly one home.

A node dir is `nodes/<id>/{node.json, shader.frag.glsl}`. Both names used to be re-spelled as
literals beside their own constants — so renaming one would break the typed loader loudly and
`sync_nodes_from_disk`'s half-written-node guard SILENTLY (it would skip every dir forever,
and a watcher that reports "nothing changed" looks exactly like a working watcher).

These pin the single home and prove every reader agrees with it.
"""

import ast
from pathlib import Path

import pytest

from shaderbox.paths import (
    NODE_JSON_BASENAME,
    NODE_SCRIPT_BASENAME,
    NODE_SHADER_BASENAME,
    ProjectPaths,
)

_PKG = Path(__file__).resolve().parent.parent / "shaderbox"
_HOME = _PKG / "paths.py"


def _modules_with_literal(literal: str) -> list[str]:
    hits: list[str] = []
    for path in sorted(_PKG.rglob("*.py")):
        if path == _HOME:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == literal:
                hits.append(f"{path.relative_to(_PKG.parent).as_posix()}:{node.lineno}")
    return hits


@pytest.mark.parametrize(
    "literal",
    # EVERY member of the node dir, not the two that happened to be in hand: a guard that
    # advertises a closed class while covering part of it is the defect it exists to catch.
    [NODE_JSON_BASENAME, NODE_SHADER_BASENAME, NODE_SCRIPT_BASENAME],
)
def test_basename_is_never_respelled(literal: str) -> None:
    hits = _modules_with_literal(literal)
    assert hits == [], (
        f"{literal!r} is re-spelled at {hits}. Import it from shaderbox.paths instead — a "
        "second spelling makes a rename break some readers silently."
    )


def test_project_paths_agree_with_the_basenames(tmp_path: Path) -> None:
    paths = ProjectPaths.for_root(tmp_path / "proj")
    assert paths.node_json_for("abc").name == NODE_JSON_BASENAME
    assert paths.node_shader_for("abc").name == NODE_SHADER_BASENAME
    assert paths.node_script_for("abc").name == NODE_SCRIPT_BASENAME
    assert paths.node_json_for("abc").parent == paths.nodes_dir / "abc"
    assert paths.node_shader_for("abc").parent == paths.nodes_dir / "abc"
    assert paths.node_script_for("abc").parent == paths.scripts_dir_for("abc")
