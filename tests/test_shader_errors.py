"""parse_shader_errors driver-format coverage (033 review fix)."""

from pathlib import Path

from shaderbox.shader_errors import SourceMap, parse_shader_errors


def test_v3d_mesa_ir_format_parses_lines() -> None:
    raw = (
        "GLSL Compiler failed\n"
        "0:57(8): error: `text_sdf' redeclared\n"
        "0:61(8): error: `weight' redeclared"
    )
    smap = SourceMap.identity(Path("/p/shader.frag.glsl"))
    errors = parse_shader_errors(raw, smap)
    assert len(errors) == 2
    assert errors[0].line == 56  # 0-based
    assert "text_sdf" in errors[0].message
    assert errors[1].line == 60


def test_nvidia_and_mesa_formats_still_parse() -> None:
    smap = SourceMap.identity(Path("/p/s.glsl"))
    nv = parse_shader_errors("0(12) : error C1008: bad", smap)
    assert nv[0].line == 11
    mesa = parse_shader_errors("ERROR: 0:3: 'x' : undeclared", smap)
    assert mesa[0].line == 2


def test_unparsable_falls_back_to_blob() -> None:
    smap = SourceMap.identity(Path("/p/s.glsl"))
    out = parse_shader_errors("total nonsense", smap)
    assert len(out) == 1
    assert out[0].line == -1
