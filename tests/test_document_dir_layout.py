"""The document-dir names have exactly one home.

A document dir is `documents/<id>/{document.json, graph.json, passes/<name>.frag.glsl}`. The names used
to be re-spelled as literals beside their own constants — so renaming one would break the typed
loader loudly and `sync_documents_from_disk`'s half-written-document guard SILENTLY (it would skip
every dir forever, and a watcher that reports "nothing changed" looks exactly like a working
watcher).

These pin the single home and prove every reader agrees with it.
"""

import ast
from pathlib import Path

import pytest

from shaderbox.paths import (
    DOCUMENT_JSON_BASENAME,
    DOCUMENT_SCRIPT_BASENAME,
    GRAPH_JSON_BASENAME,
    PASS_SHADER_SUFFIX,
    PASSES_DIR_NAME,
    ProjectPaths,
    pass_shader_name,
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
    # EVERY member of the document dir, not the two that happened to be in hand: a guard that
    # advertises a closed class while covering part of it is the defect it exists to catch.
    [
        DOCUMENT_JSON_BASENAME,
        GRAPH_JSON_BASENAME,
        DOCUMENT_SCRIPT_BASENAME,
        PASSES_DIR_NAME,
        PASS_SHADER_SUFFIX,
    ],
)
def test_basename_is_never_respelled(literal: str) -> None:
    hits = _modules_with_literal(literal)
    assert hits == [], (
        f"{literal!r} is re-spelled at {hits}. Import it from shaderbox.paths instead — a "
        "second spelling makes a rename break some readers silently."
    )


def test_project_paths_agree_with_the_basenames(tmp_path: Path) -> None:
    paths = ProjectPaths.for_root(tmp_path / "proj")
    assert paths.document_json_for("abc").name == DOCUMENT_JSON_BASENAME
    assert paths.pass_shader_for("abc", "main").name == pass_shader_name("main")
    assert paths.graph_json_for("abc").name == GRAPH_JSON_BASENAME
    assert paths.document_script_for("abc").name == DOCUMENT_SCRIPT_BASENAME
    assert paths.document_json_for("abc").parent == paths.documents_dir / "abc"
    assert paths.pass_shader_for("abc", "main").parent == paths.passes_dir_for("abc")
    assert paths.passes_dir_for("abc").parent == paths.documents_dir / "abc"
    assert paths.document_script_for("abc").parent == paths.scripts_dir_for("abc")
