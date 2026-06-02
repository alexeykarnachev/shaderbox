"""Cross-project copilot tool mechanics (feature 020·16) — pure / GL-free.

Covers the parts the agent-loop fakes don't: uniform-value coercion + the engine-driven
reject set, and the live-source lib-create path (the index must see a function in a
copilot-written file, unlike create_file_in's commented stub). The marshalled GL paths
(read_shader / set_uniform validation / create_node) are exercised in-app + by smoke.
"""

import types
from pathlib import Path

from shaderbox.app import _coerce_uniform_value, _ENGINE_DRIVEN_UNIFORMS
from shaderbox.shader_lib.file_ops import ShaderLibFileManager
from shaderbox.shader_lib.index import ShaderLibIndex


def _uniform(dimension: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(dimension=dimension)


def test_coerce_scalar() -> None:
    assert _coerce_uniform_value(0.5, _uniform(1)) == 0.5
    assert _coerce_uniform_value(3, _uniform(1)) == 3


def test_coerce_vector() -> None:
    assert _coerce_uniform_value([1, 0, 0], _uniform(3)) == (1.0, 0.0, 0.0)
    assert _coerce_uniform_value([0.5, 0.5], _uniform(2)) == (0.5, 0.5)


def test_coerce_shape_mismatch_is_none() -> None:
    # A scalar for a vec3, a vec2 for a vec3, a non-number — all reject (the handler turns
    # None into an explicit error, never the silent render-time pop).
    assert _coerce_uniform_value(0.5, _uniform(3)) is None
    assert _coerce_uniform_value([1, 0], _uniform(3)) is None
    assert _coerce_uniform_value("red", _uniform(3)) is None
    assert _coerce_uniform_value(["a", "b", "c"], _uniform(3)) is None


def test_coerce_rejects_bool() -> None:
    # JSON true/false is not a shader number, even though bool is an int subclass.
    assert _coerce_uniform_value(True, _uniform(1)) is None
    assert _coerce_uniform_value([True, False, True], _uniform(3)) is None


def test_engine_driven_set_is_the_documented_set() -> None:
    # These are overwritten every frame by Node.render() regardless of uniform_values, so
    # set_uniform rejects them (020·16 Decision 6). Pin the set so a new engine uniform
    # added to core.py is consciously added here too.
    assert _ENGINE_DRIVEN_UNIFORMS == {"u_time", "u_aspect", "u_resolution"}


def _lib_manager(root: Path) -> ShaderLibFileManager:
    return ShaderLibFileManager(
        notifications=types.SimpleNamespace(push=lambda *a, **k: None),  # type: ignore[arg-type]
        rebuild_index=lambda: None,
        index_getter=ShaderLibIndex.empty,
        on_paths_removed=lambda _paths: None,
        on_path_renamed=lambda _old, _new: None,
    )


def test_resolve_copilot_path_rejects_traversal(tmp_path: Path, monkeypatch) -> None:
    # The copilot lib path guard reuses shader_lib_root + a relative_to check — a "../" escape
    # is rejected (020·16 Decision 5), a normal name resolves under the root with .glsl.
    monkeypatch.setattr(
        "shaderbox.shader_lib.file_ops.shader_lib_root", lambda: tmp_path
    )
    mgr = _lib_manager(tmp_path)
    assert mgr.resolve_copilot_path("../../etc/passwd") is None
    ok = mgr.resolve_copilot_path("noise")
    assert ok is not None and ok.name == "noise.glsl"
    assert ok.parent == tmp_path


def test_copilot_lib_create_writes_live_source(tmp_path: Path, monkeypatch) -> None:
    # The crux of Decision 5: a copilot-created lib file holds LIVE (uncommented) source, so
    # the index extracts a function from it — unlike create_file_in's commented stub which
    # yields zero functions. We write a real SB_ function and assert ShaderLibIndex sees it.
    monkeypatch.setattr(
        "shaderbox.shader_lib.file_ops.shader_lib_root", lambda: tmp_path
    )
    mgr = _lib_manager(tmp_path)
    path = mgr.resolve_copilot_path("rotate")
    assert path is not None
    fn_source = "/// rotate a 2D vector\nvec2 SB_rotate(vec2 p, float a) {\n    return p;\n}\n"
    assert mgr.write_copilot_lib_file(path, fn_source)

    index = ShaderLibIndex.build(tmp_path)
    assert "SB_rotate" in index.functions
    assert index.functions["SB_rotate"].signature.startswith("vec2 SB_rotate")


def test_create_file_in_stub_yields_no_functions(tmp_path: Path, monkeypatch) -> None:
    # The companion proof: the HUMAN create_file_in stub is commented out, so the index sees
    # ZERO functions in it — which is exactly why the copilot path must NOT route through it.
    monkeypatch.setattr(
        "shaderbox.shader_lib.file_ops.shader_lib_root", lambda: tmp_path
    )
    mgr = _lib_manager(tmp_path)
    created = mgr.create_file_in(Path("."), "stub")
    assert created is not None
    index = ShaderLibIndex.build(tmp_path)
    assert index.functions == {}
