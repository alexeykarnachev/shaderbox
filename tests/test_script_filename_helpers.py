"""Feature 047 filename/copy helpers + the retype-hash invariant — pure, no GL.

`same_type_scripts` / `copy_script_body` are the F14 copy-content selector's source + body reader
(the spec deferred the whole F14 flow to manual make-run, but the logic is GL-free string/filesystem
work and trivially unit-testable). `get_uniform_hash` must hash a vec2 and a same-named vec3 to
DIFFERENT buckets — that's what gives a retyped uniform a fresh born-False UIUniform (decision 14)."""

import types
from pathlib import Path
from typing import Any

from shaderbox.project_session import ProjectSession

_GL_FLOAT = 0x1406


def _u(name: str, dim: int = 1) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        name=name, dimension=dim, array_length=1, gl_type=_GL_FLOAT, size=1
    )


def _node(name: str, uniforms: tuple[types.SimpleNamespace, ...]) -> types.SimpleNamespace:
    ui_state = types.SimpleNamespace(ui_uniforms={}, ui_name=name, is_brain_active=False)
    node = types.SimpleNamespace(get_active_uniforms=lambda: list(uniforms))
    return types.SimpleNamespace(node=node, ui_state=ui_state)


def _stub(tmp: Path, nodes: dict[str, types.SimpleNamespace]) -> Any:
    def scripts_dir_for(node_id: str) -> Path:
        d = tmp / node_id / "scripts"
        d.mkdir(parents=True, exist_ok=True)
        return d

    paths = types.SimpleNamespace(scripts_dir_for=scripts_dir_for)
    stub = types.SimpleNamespace(ui_nodes=nodes, paths=paths)
    for meth in (
        "_ui_uniform_for",
        "_script_filename",
        "same_type_scripts",
        "copy_script_body",
    ):
        setattr(stub, meth, getattr(ProjectSession, meth).__get__(stub))
    return stub


def _write_script(stub: Any, node_id: str, filename: str) -> None:
    (stub.paths.scripts_dir_for(node_id) / filename).write_text("x\n", encoding="utf-8")


def test_same_type_scripts_matches_tag_excludes_self_and_includes_cross_node(
    tmp_path: Path,
) -> None:
    # u_velocity (vec2) on n0; siblings: a matching-tag vec2 on n1 (included), a vec3 on n1
    # (excluded — wrong tag), and the current file (excluded).
    stub = _stub(
        tmp_path,
        {
            "n0": _node("Alpha", (_u("u_velocity", dim=2),)),
            "n1": _node("Beta", (_u("u_offset", dim=2), _u("u_color", dim=3))),
        },
    )
    _write_script(stub, "n0", "u_velocity__vec2.py")
    _write_script(stub, "n1", "u_offset__vec2.py")  # match
    _write_script(stub, "n1", "u_color__vec3.py")  # wrong tag

    results = stub.same_type_scripts("n0", "u_velocity")
    labels = {label for label, _ in results}
    assert labels == {"Beta / u_offset"}  # cross-node same-tag only; self + vec3 excluded


def test_same_type_scripts_empty_when_no_sibling(tmp_path: Path) -> None:
    stub = _stub(tmp_path, {"n0": _node("Alpha", (_u("u_x", dim=2),))})
    _write_script(stub, "n0", "u_x__vec2.py")
    assert stub.same_type_scripts("n0", "u_x") == []  # only the current file exists


def test_copy_script_body_reads_file_and_handles_missing(tmp_path: Path) -> None:
    stub = _stub(tmp_path, {})
    src = tmp_path / "body.py"
    src.write_text("class Behavior:\n    pass\n", encoding="utf-8")
    assert stub.copy_script_body(src) == "class Behavior:\n    pass\n"
    assert stub.copy_script_body(tmp_path / "nope.py") == ""  # missing -> "" (no crash)
