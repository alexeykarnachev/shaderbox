"""Unit tests for the auto-resolve library — no GL.

Two layers under test:
- `ShaderLibIndex.build` extracts function defs, sigs, bodies, callees, docstrings.
- `resolve_usage` prunes the lib to the transitive set the user actually
  references, emits a topo-sorted preamble + `#line` markers + SourceMap.
"""

from pathlib import Path

from shaderbox.shader_lib import (
    ShaderLibFunction,
    ShaderLibIndex,
    resolve_usage,
)
from shaderbox.shader_source import ShaderSource


def _write(path: Path, text: str) -> ShaderSource:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return ShaderSource.load(path)


def _make_lib(lib: Path, files: dict[str, str]) -> ShaderLibIndex:
    for name, text in files.items():
        (lib / name).parent.mkdir(parents=True, exist_ok=True)
        (lib / name).write_text(text, encoding="utf-8")
    return ShaderLibIndex.build(lib)


# ----------------------------------------------------------------------------
# ShaderLibIndex.build — function extraction
# ----------------------------------------------------------------------------


def test_extracts_single_function(tmp_path: Path) -> None:
    lib = tmp_path / "lib"
    idx = _make_lib(lib, {"hash.glsl": "float SB_hash(vec2 p) {\n    return 0.5;\n}\n"})
    assert set(idx.functions) == {"SB_hash"}
    fn: ShaderLibFunction = idx.functions["SB_hash"]
    assert fn.name == "SB_hash"
    assert "vec2 p" in fn.signature
    assert "return 0.5;" in fn.body
    assert fn.file.name == "hash.glsl"
    assert fn.line_in_file == 0
    assert fn.doc == ""


def test_extracts_multiple_functions_same_file(tmp_path: Path) -> None:
    lib = tmp_path / "lib"
    idx = _make_lib(
        lib,
        {
            "noise.glsl": (
                "float SB_hash(vec2 p) { return 0.0; }\n"
                "float SB_perlin(vec2 p) { return SB_hash(p); }\n"
            )
        },
    )
    assert set(idx.functions) == {"SB_hash", "SB_perlin"}
    assert "SB_hash" in idx.functions["SB_perlin"].calls


def test_extracts_doc_comment(tmp_path: Path) -> None:
    lib = tmp_path / "lib"
    idx = _make_lib(
        lib,
        {
            "hash.glsl": (
                "/// returns a pseudorandom float in [0, 1)\n"
                "/// from a 2D point\n"
                "float SB_hash(vec2 p) { return 0.0; }\n"
            )
        },
    )
    fn = idx.functions["SB_hash"]
    assert "pseudorandom float" in fn.doc
    assert "from a 2D point" in fn.doc


def test_ignores_function_call_inside_body_as_definition(tmp_path: Path) -> None:
    # Only top-level (column 0) declarations count. A line inside a function body
    # that happens to look like `vec3 helper(p)` should NOT be treated as a def.
    lib = tmp_path / "lib"
    idx = _make_lib(
        lib,
        {
            "noise.glsl": (
                "float SB_outer(vec2 p) {\n"
                "    float inner = SB_hash(p);\n"
                "    return inner;\n"
                "}\n"
            )
        },
    )
    assert set(idx.functions) == {"SB_outer"}


def test_brace_counted_body_with_nested_blocks(tmp_path: Path) -> None:
    lib = tmp_path / "lib"
    idx = _make_lib(
        lib,
        {
            "x.glsl": (
                "float SB_branch(float x) {\n"
                "    if (x > 0.0) {\n"
                "        return x;\n"
                "    } else {\n"
                "        return -x;\n"
                "    }\n"
                "}\n"
            )
        },
    )
    fn = idx.functions["SB_branch"]
    # The body should include both branches and end at the outer closing `}`.
    assert "return x" in fn.body
    assert "return -x" in fn.body
    # No stray text after the function.


def test_strips_block_and_line_comments_before_extraction(tmp_path: Path) -> None:
    lib = tmp_path / "lib"
    idx = _make_lib(
        lib,
        {
            "x.glsl": (
                "// a comment with float SB_fake(vec2 p) { in it\n"
                "/* float SB_alsoFake() */ \n"
                "float SB_real(vec2 p) { return 0.0; }\n"
            )
        },
    )
    assert set(idx.functions) == {"SB_real"}


def test_multiple_files_indexed(tmp_path: Path) -> None:
    lib = tmp_path / "lib"
    idx = _make_lib(
        lib,
        {
            "a.glsl": "float SB_a() { return 0.0; }\n",
            "b.glsl": "float SB_b() { return 1.0; }\n",
        },
    )
    assert set(idx.functions) == {"SB_a", "SB_b"}
    assert idx.functions["SB_a"].file.name == "a.glsl"
    assert idx.functions["SB_b"].file.name == "b.glsl"


def test_subdirectory_indexed(tmp_path: Path) -> None:
    lib = tmp_path / "lib"
    idx = _make_lib(lib, {"noise/value.glsl": "float SB_value() { return 0.0; }\n"})
    assert "SB_value" in idx.functions


def test_empty_lib_root_returns_empty_index(tmp_path: Path) -> None:
    lib = tmp_path / "lib"
    lib.mkdir()
    idx = ShaderLibIndex.build(lib)
    assert idx.functions == {}


def test_missing_lib_root_returns_empty_index(tmp_path: Path) -> None:
    idx = ShaderLibIndex.build(tmp_path / "nonexistent")
    assert idx.functions == {}


# ----------------------------------------------------------------------------
# resolve_usage — pruning + topo sort + preamble
# ----------------------------------------------------------------------------


def test_no_usage_passes_through_unchanged(tmp_path: Path) -> None:
    lib = tmp_path / "lib"
    idx = _make_lib(lib, {"hash.glsl": "float SB_hash() { return 0.0; }\n"})
    root = _write(tmp_path / "root.glsl", "void main() {}\n")
    flattened, sources, smap, errors = resolve_usage(root, idx)
    assert errors == []
    assert flattened == "void main() {}\n"
    assert len(sources) == 1
    assert sources[0].path == root.path
    assert smap.file_id_to_path == {0: root.path}


def test_single_function_used(tmp_path: Path) -> None:
    lib = tmp_path / "lib"
    idx = _make_lib(lib, {"hash.glsl": "float SB_hash(vec2 p) { return 0.5; }\n"})
    root = _write(
        tmp_path / "root.glsl",
        "#version 460 core\nvoid main() { float h = SB_hash(vec2(0.0)); }\n",
    )
    flattened, sources, smap, errors = resolve_usage(root, idx)
    assert errors == []
    assert "float SB_hash(vec2 p)" in flattened
    assert "void main()" in flattened
    assert len(sources) == 2
    # File-ids: root=0, hash.glsl=1.
    assert smap.file_id_to_path[0].name == "root.glsl"
    assert smap.file_id_to_path[1].name == "hash.glsl"


def test_transitive_dep_pulled_in(tmp_path: Path) -> None:
    lib = tmp_path / "lib"
    idx = _make_lib(
        lib,
        {
            "noise.glsl": (
                "float SB_hash(vec2 p) { return 0.0; }\n"
                "float SB_fbm(vec2 p) { return SB_hash(p); }\n"
            )
        },
    )
    root = _write(
        tmp_path / "root.glsl",
        "void main() { float n = SB_fbm(vec2(0.0)); }\n",
    )
    flattened, _sources, _smap, errors = resolve_usage(root, idx)
    assert errors == []
    # Both functions are prepended.
    assert "SB_hash" in flattened
    assert "SB_fbm" in flattened
    # SB_hash is declared BEFORE SB_fbm (topo order).
    assert flattened.find("SB_hash(vec2 p)") < flattened.find("SB_fbm(vec2 p)")


def test_unused_lib_functions_excluded(tmp_path: Path) -> None:
    lib = tmp_path / "lib"
    idx = _make_lib(
        lib,
        {
            "a.glsl": "float SB_a() { return 0.0; }\n",
            "b.glsl": "float SB_b() { return 1.0; }\n",
        },
    )
    root = _write(tmp_path / "root.glsl", "void main() { float x = SB_a(); }\n")
    flattened, _sources, _smap, _errors = resolve_usage(root, idx)
    assert "SB_a" in flattened
    assert "SB_b" not in flattened


def test_user_shadowing_suppresses_lib_version(tmp_path: Path) -> None:
    lib = tmp_path / "lib"
    idx = _make_lib(lib, {"hash.glsl": "float SB_hash() { return 0.0; }\n"})
    root = _write(
        tmp_path / "root.glsl",
        "float SB_hash() { return 999.0; }\nvoid main() { float x = SB_hash(); }\n",
    )
    flattened, sources, _smap, _errors = resolve_usage(root, idx)
    # Lib version is NOT prepended (only the user's version exists).
    assert flattened.count("SB_hash()") >= 2  # decl + call site in user code
    # Only one source (root); the lib file was not pulled in.
    assert len(sources) == 1
    # The user's body should be the one defining 999.0; no `return 0.0` from lib.
    assert "return 999.0" in flattened
    assert "return 0.0" not in flattened


def test_call_after_return_keyword_is_not_a_shadowing_def(tmp_path: Path) -> None:
    # Regression: `return SB_foo(...)` must NOT read as a user definition of SB_foo. The old
    # USER_FN_DEF_RE matched `<word> SB_foo(` and so saw `return` as a type, wrongly suppressing
    # the lib splice — the function then failed to compile as "undefined" (a live copilot session
    # hit exactly this with `return SB_palette_sunset(...)`).
    lib = tmp_path / "lib"
    idx = _make_lib(lib, {"palette.glsl": "vec3 SB_pal(float t) { return vec3(t); }\n"})
    root = _write(
        tmp_path / "root.glsl",
        "vec3 animated(float t) { return SB_pal(t); }\n"
        "void main() { vec3 c = animated(0.5); }\n",
    )
    flattened, sources, _smap, errors = resolve_usage(root, idx)
    assert errors == []
    # The lib function MUST be spliced (it was wrongly treated as user-shadowed before the fix).
    assert "vec3 SB_pal(float t)" in flattened
    assert len(sources) == 2  # root + the lib file pulled in
    # And it's spliced BEFORE the call site (GLSL needs def-before-use).
    assert flattened.find("vec3 SB_pal(float t)") < flattened.find("return SB_pal(t)")


def test_preamble_inserted_after_version_directive(tmp_path: Path) -> None:
    lib = tmp_path / "lib"
    idx = _make_lib(lib, {"x.glsl": "float SB_x() { return 0.0; }\n"})
    root = _write(
        tmp_path / "root.glsl",
        "#version 460 core\n#extension GL_ARB_x : enable\nvoid main() { SB_x(); }\n",
    )
    flattened, _sources, _smap, _errors = resolve_usage(root, idx)
    lines = flattened.split("\n")
    # `#version` must be the first non-blank line.
    assert lines[0].startswith("#version")
    # The preamble (`#line ...` + lib body) follows; `void main` comes after.
    version_idx = next(i for i, ln in enumerate(lines) if ln.startswith("#version"))
    extension_idx = next(i for i, ln in enumerate(lines) if ln.startswith("#extension"))
    sb_x_idx = next(
        i for i, ln in enumerate(lines) if "SB_x()" in ln and "return" in ln
    )
    main_idx = next(i for i, ln in enumerate(lines) if "void main()" in ln)
    assert version_idx < extension_idx < sb_x_idx < main_idx


def test_resolve_error_on_lib_cycle(tmp_path: Path) -> None:
    lib = tmp_path / "lib"
    idx = _make_lib(
        lib,
        {
            "loop.glsl": (
                "float SB_a() { return SB_b(); }\nfloat SB_b() { return SB_a(); }\n"
            )
        },
    )
    root = _write(tmp_path / "root.glsl", "void main() { SB_a(); }\n")
    _flattened, _sources, _smap, errors = resolve_usage(root, idx)
    assert len(errors) >= 1
    assert any("cycle" in e.message for e in errors)


def test_typo_silently_passes_through(tmp_path: Path) -> None:
    # User typoed `SB_perln` (no such function); the resolver doesn't prepend
    # anything for it and doesn't emit a ResolveError — the driver will produce
    # "undeclared identifier" at the user's line, which is the right home for it.
    lib = tmp_path / "lib"
    idx = _make_lib(lib, {"x.glsl": "float SB_perlin() { return 0.0; }\n"})
    root = _write(tmp_path / "root.glsl", "void main() { float x = SB_perln(); }\n")
    flattened, _sources, _smap, errors = resolve_usage(root, idx)
    assert errors == []
    assert "SB_perlin" not in flattened  # not requested
    assert "SB_perln" in flattened  # user's typo preserved


def test_line_markers_thread_through_source_map(tmp_path: Path) -> None:
    # A driver error at "file 1, line 1" should resolve to (lib_path, 1).
    lib = tmp_path / "lib"
    idx = _make_lib(lib, {"hash.glsl": "float SB_hash() { return 0.0; }\n"})
    root = _write(tmp_path / "root.glsl", "void main() { SB_hash(); }\n")
    _flattened, _sources, smap, _errors = resolve_usage(root, idx)
    path, line = smap.resolve(1, 1)
    assert path.name == "hash.glsl"
    assert line == 1
    # Driver at file 0 line 1 should resolve to root.
    path, line = smap.resolve(0, 1)
    assert path == root.path
    assert line == 1


def test_subdirectory_function_resolves(tmp_path: Path) -> None:
    lib = tmp_path / "lib"
    idx = _make_lib(lib, {"noise/perlin.glsl": "float SB_perlin() { return 0.0; }\n"})
    root = _write(tmp_path / "root.glsl", "void main() { SB_perlin(); }\n")
    flattened, sources, _smap, errors = resolve_usage(root, idx)
    assert errors == []
    assert "SB_perlin" in flattened
    assert any(s.path.name == "perlin.glsl" for s in sources)


def test_only_SB_prefixed_identifiers_trigger_resolve(tmp_path: Path) -> None:
    # A lib file with a non-SB function: defined, but never auto-resolved into the
    # user's shader unless the user explicitly calls it via SB_-prefixed wrapper.
    lib = tmp_path / "lib"
    idx = _make_lib(
        lib,
        {
            "x.glsl": (
                "float _internal(float x) { return x; }\n"
                "float SB_wrapper(float x) { return _internal(x); }\n"
            )
        },
    )
    # User calls only the SB_-prefixed wrapper.
    root = _write(tmp_path / "root.glsl", "void main() { SB_wrapper(0.0); }\n")
    flattened, _sources, _smap, _errors = resolve_usage(root, idx)
    # Both should appear (SB_wrapper requested; _internal pulled in transitively).
    assert "SB_wrapper" in flattened
    assert "_internal" in flattened

    # A user trying to call `_internal` directly: the lib body is NOT prepended
    # (we only auto-resolve SB_-prefixed identifiers), so the only `_internal`
    # in the flattened text is the user's own call site. The driver will then
    # emit `undeclared identifier _internal` — which is the right home for it.
    root2 = _write(tmp_path / "root2.glsl", "void main() { _internal(0.0); }\n")
    flattened2, sources2, _smap2, _errors2 = resolve_usage(root2, idx)
    # No lib body prepended (just the user's text).
    assert "float _internal" not in flattened2
    assert len(sources2) == 1
