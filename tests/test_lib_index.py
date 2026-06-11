"""Unit tests for the auto-resolve library — no GL.

Two layers under test:
- `ShaderLibIndex.build` extracts function defs, sigs, bodies, callees, docstrings.
- `resolve_usage` prunes the lib to the transitive set the user actually
  references, emits a topo-sorted preamble + `#line` markers + SourceMap.
"""

import re
from pathlib import Path

from shaderbox.glyph_tables import TABLE_UNIFORMS
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


def test_typo_errors_with_closest_name_suggestion(tmp_path: Path) -> None:
    # User typoed `SB_perln` (no such function): the resolver errors at the user's
    # line with the closest catalogue name, instead of letting the driver emit a
    # cryptic "undeclared identifier" (feature 033).
    lib = tmp_path / "lib"
    idx = _make_lib(lib, {"x.glsl": "float SB_perlin() { return 0.0; }\n"})
    root = _write(tmp_path / "root.glsl", "void main() { float x = SB_perln(); }\n")
    _flattened, _sources, _smap, errors = resolve_usage(root, idx)
    assert len(errors) == 1
    assert errors[0].path == root.path
    assert errors[0].line == 0  # 0-based, the ShaderError convention
    assert "unknown library function 'SB_perln'" in errors[0].message
    assert "did you mean 'SB_perlin'" in errors[0].message


def test_unknown_name_errors_even_with_no_known_names_used(tmp_path: Path) -> None:
    # The fast path (no known lib names referenced) must still catch unknowns.
    lib = tmp_path / "lib"
    idx = _make_lib(lib, {"x.glsl": "float SB_perlin() { return 0.0; }\n"})
    root = _write(tmp_path / "root.glsl", "void main() { SB_totally_made_up(); }\n")
    _flattened, _sources, _smap, errors = resolve_usage(root, idx)
    assert len(errors) == 1
    assert "unknown library function 'SB_totally_made_up'" in errors[0].message


def test_non_call_and_non_function_SB_names_are_not_unknown(tmp_path: Path) -> None:
    # A user-side #define / const / uniform with an SB_ name must NOT error (a
    # resolve error blocks the driver compile — false positives are regressions),
    # and a bare non-call mention is left alone too.
    lib = tmp_path / "lib"
    idx = _make_lib(lib, {"x.glsl": "float SB_perlin() { return 0.0; }\n"})
    root = _write(
        tmp_path / "root.glsl",
        "#define SB_SCALE 2.0\n"
        "const float SB_PI = 3.14159;\n"
        "uniform vec2 SB_OFFSET;\n"
        "void main() { float x = SB_PI * SB_SCALE; }\n",
    )
    _flattened, _sources, _smap, errors = resolve_usage(root, idx)
    assert errors == []


def test_unknown_name_line_survives_block_comments(tmp_path: Path) -> None:
    # Multi-line /* */ comments above the call must not shift the reported line.
    lib = tmp_path / "lib"
    idx = _make_lib(lib, {"x.glsl": "float SB_perlin() { return 0.0; }\n"})
    root = _write(
        tmp_path / "root.glsl",
        "/* one\n   two\n   three */\nvoid main() { SB_perln(); }\n",
    )
    _flattened, _sources, _smap, errors = resolve_usage(root, idx)
    assert len(errors) == 1
    assert errors[0].line == 3  # 0-based: the call sits on source line 4


def test_user_defined_SB_function_is_not_unknown(tmp_path: Path) -> None:
    # A user-defined SB_* function (shadowing convention) must not error.
    lib = tmp_path / "lib"
    idx = _make_lib(lib, {"x.glsl": "float SB_perlin() { return 0.0; }\n"})
    root = _write(
        tmp_path / "root.glsl",
        "float SB_mine(vec2 p) { return p.x; }\n"
        "void main() { SB_mine(vec2(0.0)); SB_perlin(); }\n",
    )
    _flattened, _sources, _smap, errors = resolve_usage(root, idx)
    assert errors == []


def test_user_prototype_SB_function_is_not_unknown(tmp_path: Path) -> None:
    # A forward DECLARATION (prototype; the body lands in a later edit) is legal
    # GLSL and must not resolve-error (review cycle 3).
    lib = tmp_path / "lib"
    idx = _make_lib(lib, {"x.glsl": "float SB_perlin() { return 0.0; }\n"})
    root = _write(
        tmp_path / "root.glsl",
        "float SB_later(vec2 p);\nvoid main() { float x = SB_later(vec2(0.0)); }\n",
    )
    _flattened, _sources, _smap, errors = resolve_usage(root, idx)
    assert errors == []


def test_call_statement_is_not_a_prototype(tmp_path: Path) -> None:
    # `return SB_x(...);` and a bare call statement end in `);` too — they must
    # still count as CALLS of an unknown name, not prototypes.
    lib = tmp_path / "lib"
    idx = _make_lib(lib, {"x.glsl": "float SB_perlin() { return 0.0; }\n"})
    root = _write(
        tmp_path / "root.glsl",
        "float f() {\n    return SB_nope(1.0);\n}\n"
        "void main() {\n    SB_also_nope();\n}\n",
    )
    _flattened, _sources, _smap, errors = resolve_usage(root, idx)
    names = " ".join(e.message for e in errors)
    assert "SB_nope" in names
    assert "SB_also_nope" in names


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


# ----------------------------------------------------------------------------
# Top-level const declarations — extraction + resolution
# ----------------------------------------------------------------------------


def test_extracts_top_level_const(tmp_path: Path) -> None:
    lib = tmp_path / "lib"
    idx = _make_lib(
        lib,
        {
            "tables.glsl": (
                "/// stroke table\n"
                "const vec4 SBT_DATA[3] = vec4[](\n"
                "    vec4(0.0),\n"
                "    vec4(1.0),\n"
                "    vec4(2.0)\n"
                ");\n"
                "float SB_read(int i) { return SBT_DATA[i].x; }\n"
            )
        },
    )
    assert set(idx.functions) == {"SBT_DATA", "SB_read"}
    const = idx.functions["SBT_DATA"]
    assert const.signature == "const vec4 SBT_DATA[3]"
    assert const.body.startswith("const vec4 SBT_DATA[3]")
    assert const.body.rstrip().endswith(");")
    assert "vec4(2.0)" in const.body
    assert const.doc == "stroke table"
    assert "SBT_DATA" in idx.functions["SB_read"].calls


def test_const_inside_function_body_not_extracted(tmp_path: Path) -> None:
    lib = tmp_path / "lib"
    idx = _make_lib(
        lib,
        {
            "x.glsl": (
                "float SB_pick(int i) {\n"
                "    const float K[2] = float[](1.0, 2.0);\n"
                "    return K[i];\n"
                "}\n"
            )
        },
    )
    assert set(idx.functions) == {"SB_pick"}


def test_resolver_splices_const_before_its_reader(tmp_path: Path) -> None:
    lib = tmp_path / "lib"
    idx = _make_lib(
        lib,
        {
            "tables.glsl": (
                "const vec4 SBT_DATA[2] = vec4[](vec4(0.0), vec4(1.0));\n"
                "float SB_read(int i) { return SBT_DATA[i].x; }\n"
            )
        },
    )
    root = _write(tmp_path / "root.glsl", "void main() { SB_read(0); }\n")
    flattened, _sources, _smap, errors = resolve_usage(root, idx)
    assert not errors
    assert "const vec4 SBT_DATA[2]" in flattened
    assert flattened.index("const vec4 SBT_DATA[2]") < flattened.index("float SB_read")


def test_user_defined_const_shadows_lib_version(tmp_path: Path) -> None:
    # A user-side `const ... SB_K = ...` must suppress the lib splice — prepending
    # the lib version over the user's redefinition would be a duplicate symbol.
    lib = tmp_path / "lib"
    idx = _make_lib(lib, {"k.glsl": "const float SB_K = 1.0;\n"})
    root = _write(
        tmp_path / "root.glsl",
        "const float SB_K = 2.0;\nvoid main() { float x = SB_K; }\n",
    )
    flattened, _sources, _smap, errors = resolve_usage(root, idx)
    assert not errors
    assert flattened.count("const float SB_K") == 1
    assert "SB_K = 1.0" not in flattened


def test_extracts_top_level_uniform_decl(tmp_path: Path) -> None:
    # Engine-bound table shape (the glyph strokes): a lib `uniform` declaration is
    # a spliceable entry just like a const, so readers can live in lib functions.
    lib = tmp_path / "lib"
    idx = _make_lib(
        lib,
        {
            "tables.glsl": (
                "uniform vec4 SBT_T[4];\n"
                "float SB_read_t(int i) { return SBT_T[i].x; }\n"
            )
        },
    )
    assert set(idx.functions) == {"SBT_T", "SB_read_t"}
    decl = idx.functions["SBT_T"]
    assert decl.signature == "uniform vec4 SBT_T[4]"
    assert decl.body == "uniform vec4 SBT_T[4];"
    root = _write(tmp_path / "root.glsl", "void main() { SB_read_t(0); }\n")
    flattened, _sources, _smap, errors = resolve_usage(root, idx)
    assert not errors
    assert flattened.index("uniform vec4 SBT_T[4];") < flattened.index(
        "float SB_read_t"
    )


# ----------------------------------------------------------------------------
# Depth-0 scanning — a body the signature regex can't match must not leak
# its interior into the index
# ----------------------------------------------------------------------------


_ALLMAN_TEXT: str = (
    "float SB_allman(vec2 p)\n"
    "{\n"
    "    const float K = 2.0;\n"
    "    if (p.x > 0.0) {\n"
    "        return K;\n"
    "    }\n"
    "    else if (p.y > 0.0) {\n"
    "        return -K;\n"
    "    }\n"
    "    return 0.0;\n"
    "}\n"
)


def test_allman_style_body_yields_no_garbage_entries(tmp_path: Path) -> None:
    # Allman braces defeat FN_SIG_RE; the scan must NOT walk into the body and
    # index `else if (...) {` as a function named `if` or the local const `K`.
    # The Allman function itself stays unindexed (one-declarator-per-line
    # convention) — acceptable, as long as it poisons nothing.
    lib = tmp_path / "lib"
    idx = _make_lib(lib, {"allman.glsl": _ALLMAN_TEXT})
    assert idx.functions == {}


def test_allman_file_does_not_poison_sibling_files(tmp_path: Path) -> None:
    # SB_ok's over-collected callee set contains `if`; pre-fix the garbage `if`
    # entry from allman.glsl resolved against it and got spliced into EVERY
    # shader using SB_ok.
    lib = tmp_path / "lib"
    idx = _make_lib(
        lib,
        {
            "allman.glsl": _ALLMAN_TEXT,
            "ok.glsl": (
                "float SB_ok(vec2 p) {\n"
                "    if (p.x > 0.0) { return 1.0; }\n"
                "    return p.x;\n"
                "}\n"
            ),
        },
    )
    assert set(idx.functions) == {"SB_ok"}
    root = _write(tmp_path / "root.glsl", "void main() { SB_ok(vec2(0.0)); }\n")
    flattened, _sources, _smap, errors = resolve_usage(root, idx)
    assert errors == []
    assert "float SB_ok(vec2 p)" in flattened
    assert "else if" not in flattened


def test_unterminated_body_does_not_index_interior(tmp_path: Path) -> None:
    # Mid-edit unbalanced braces: nothing from the broken file gets indexed.
    lib = tmp_path / "lib"
    idx = _make_lib(
        lib,
        {
            "broken.glsl": (
                "float SB_broken(vec2 p) {\n"
                "    const float K = 1.0;\n"
                "    if (p.x > 0.0) {\n"
                "        return K;\n"
            )
        },
    )
    assert idx.functions == {}


def test_top_level_entries_after_allman_body_still_index(tmp_path: Path) -> None:
    # Depth recovers at the body's closing brace; later top-level declarations
    # (incl. a multi-line const initializer) index and resolve normally.
    lib = tmp_path / "lib"
    idx = _make_lib(
        lib,
        {
            "mix.glsl": (
                _ALLMAN_TEXT + "const vec4 SB_TABLE[2] = vec4[](\n"
                "    vec4(0.0),\n"
                "    vec4(1.0)\n"
                ");\n"
                "float SB_after(vec2 p) { return SB_TABLE[0].x; }\n"
            )
        },
    )
    assert set(idx.functions) == {"SB_TABLE", "SB_after"}
    root = _write(tmp_path / "root.glsl", "void main() { SB_after(vec2(0.0)); }\n")
    flattened, _sources, _smap, errors = resolve_usage(root, idx)
    assert errors == []
    assert flattened.index("const vec4 SB_TABLE[2]") < flattened.index("float SB_after")


def test_glyph_tables_sidecar_matches_shipped_glsl() -> None:
    # glyphs.glsl (uniform decls) and glyph_tables.py (the values Node.compile
    # writes) are generated together — a half-regenerated pair must fail loudly.
    glsl = (
        Path(__file__).parent.parent / "shaderbox/resources/shader_lib/text/glyphs.glsl"
    ).read_text(encoding="utf-8")
    decls = dict(re.findall(r"uniform \w+4 (SBT_\w+)\[(\d+)\];", glsl))
    assert set(decls) == set(TABLE_UNIFORMS)
    for name, count in decls.items():
        assert len(TABLE_UNIFORMS[name]) == int(count) * 16  # 16 bytes per (i)vec4
