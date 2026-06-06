"""Cross-project copilot tool mechanics (feature 020·16) — pure / GL-free.

Covers the parts the agent-loop fakes don't: uniform-value coercion + the engine-driven
reject set, and the live-source lib-create path (the index must see a function in a
copilot-written file, unlike create_file_in's commented stub). The marshalled GL paths
(read_shader / set_uniform validation / create_node) are exercised in-app + by smoke.
"""

import types
from collections.abc import Iterator
from pathlib import Path

import moderngl
import pytest

from shaderbox.app import _STARTER_TEMPLATE_ID
from shaderbox.copilot.backend import _ENGINE_DRIVEN_UNIFORMS, _coerce_uniform_value
from shaderbox.shader_lib.file_ops import ShaderLibFileManager
from shaderbox.shader_lib.index import ShaderLibIndex
from shaderbox.ui_models import load_node_from_dir


def _uniform(dimension: int) -> types.SimpleNamespace:
    # array_length=1 (a scalar/vec, not an array); gl_type GL_FLOAT — the array branch is exercised
    # by the dedicated coercion tests in test_template_library / test_uniform_arrays.
    return types.SimpleNamespace(dimension=dimension, array_length=1, gl_type=0x1406)


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
    assert {"u_time", "u_aspect", "u_resolution"} == _ENGINE_DRIVEN_UNIFORMS


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
    fn_source = (
        "/// rotate a 2D vector\nvec2 SB_rotate(vec2 p, float a) {\n    return p;\n}\n"
    )
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
    created = mgr.create_file_in(Path(), "stub")
    assert created is not None
    index = ShaderLibIndex.build(tmp_path)
    assert index.functions == {}


@pytest.fixture(scope="module")
def gl_ctx() -> Iterator[moderngl.Context]:
    try:
        ctx = moderngl.create_standalone_context()
    except Exception as e:
        pytest.skip(f"no standalone GL context available: {e}")
    yield ctx
    ctx.release()


def test_create_node_from_source_does_not_touch_starter_template(
    gl_ctx: moderngl.Context, tmp_path: Path
) -> None:
    # Regression (review CRITICAL): _copilot_create_node must NOT write the agent's source
    # through new_node.node.source.path — after load_node_from_dir that path still points at
    # the SHARED starter template. The fix relies on UINode.save rebinding the path + writing
    # the source to the new node's own dir. This pins the contract: the starter file is
    # byte-unchanged, and the new dir holds the agent's source.
    from shaderbox.constants import RESOURCES_DIR

    starter = RESOURCES_DIR / "node_templates" / _STARTER_TEMPLATE_ID
    starter_shader = starter / "shader.frag.glsl"
    before = starter_shader.read_bytes()

    agent_source = "void main() { gl_FragColor = vec4(1.0); }\n"
    new_node = load_node_from_dir(starter)
    new_node.reset_id()
    new_node.node.release_program(agent_source)  # sets source.text, NOT a disk write
    saved_dir = new_node.save(tmp_path)

    assert starter_shader.read_bytes() == before, "starter template was clobbered"
    assert (saved_dir / "shader.frag.glsl").read_text() == agent_source
    assert new_node.node.source.path == saved_dir / "shader.frag.glsl"


def test_create_node_compiles_and_surfaces_errors(gl_ctx: moderngl.Context) -> None:
    # The compile-feedback contract (the test-exposed gap): create_node compiles the new node
    # and returns its errors, so a create-from-broken-source can't report success. Mirrors what
    # _copilot_create_node does (release_program -> compile -> read compile_unit.errors).
    from shaderbox.constants import RESOURCES_DIR

    starter = RESOURCES_DIR / "node_templates" / _STARTER_TEMPLATE_ID

    # Full broken source -> compile surfaces errors.
    broken = load_node_from_dir(starter)
    broken.node.release_program("void main() { this is not glsl }\n")
    broken.node.compile()
    assert (
        broken.node.compile_unit.errors
    ), "broken source should produce compile errors"

    # Empty source -> the starter's own (clean) program compiles clean.
    starter_node = load_node_from_dir(starter)
    starter_node.node.compile()
    assert not starter_node.node.compile_unit.errors, "starter must compile clean"


def _id_stub(ids: list[str]) -> types.SimpleNamespace:
    # Minimal stand-in for CopilotBackend's _copilot_short_ids / _copilot_resolve_node_id (pure
    # dict logic over the injected ui_nodes getter — no GL). dict preserves insertion order.
    nodes = dict.fromkeys(ids)
    return types.SimpleNamespace(_get_ui_nodes=lambda: nodes)


def test_short_ids_are_4_chars_when_no_collision() -> None:
    from shaderbox.copilot.backend import CopilotBackend

    stub = _id_stub(["abcd1111", "ef992222", "12345678"])
    short = CopilotBackend._copilot_short_ids(stub)  # type: ignore[arg-type]
    assert list(short.values()) == ["abcd", "ef99", "1234"]
    assert all(len(s) == 4 for s in short.values())


def test_short_ids_grow_on_collision() -> None:
    from shaderbox.copilot.backend import CopilotBackend

    # Two ids share the first 5 chars -> ALL grow to the shortest disambiguating length (6).
    stub = _id_stub(["abcde1xxx", "abcde2yyy", "ffff0000"])
    short = CopilotBackend._copilot_short_ids(stub)  # type: ignore[arg-type]
    assert short == {"abcde1xxx": "abcde1", "abcde2yyy": "abcde2", "ffff0000": "ffff00"}
    assert len(set(short.values())) == 3


def test_resolve_node_id_accepts_short_full_and_rejects_unknown_ambiguous() -> None:
    from shaderbox.copilot.backend import CopilotBackend

    stub = _id_stub(["abcd1111", "abcd2222", "ef993333"])
    resolve = CopilotBackend._copilot_resolve_node_id.__get__(stub)
    assert resolve("ef993333") == "ef993333"  # exact full id
    assert resolve("ef99") == "ef993333"  # unique short prefix
    assert resolve("abcd") is None  # ambiguous prefix -> None
    assert resolve("abcd1") == "abcd1111"  # disambiguated prefix
    assert resolve("zzzz") is None  # no match
