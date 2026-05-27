"""Pure unit tests for the text/unicode and resolution-string helpers — no GL.

The unicode round-trip guards the text-uniform crash (a full char[N] with no
trailing zero must decode without raising). The resolution tests pin the display
format that tabs/node.py parses back to (w, h).
"""

from shaderbox.util import (
    ShaderError,
    find_uniform_declaration_line,
    format_auto_value,
    get_resolution_str,
    next_error_line,
    parse_shader_errors,
    str_to_unicode,
    unicode_to_str,
)


def test_unicode_round_trip_partial_fill() -> None:
    encoded = str_to_unicode("abc", 16)
    assert len(encoded) == 16
    assert unicode_to_str(encoded) == "abc"


def test_unicode_full_fill_no_terminator_does_not_raise() -> None:
    encoded = str_to_unicode("0123456789abcdef", 16)
    assert len(encoded) == 16
    assert 0 not in encoded
    assert unicode_to_str(encoded) == "0123456789abcdef"


def test_unicode_overflow_truncates_to_array_length() -> None:
    encoded = str_to_unicode("a" * 32, 16)
    assert len(encoded) == 16
    assert unicode_to_str(encoded) == "a" * 16


def test_unicode_empty_round_trip() -> None:
    encoded = str_to_unicode("", 8)
    assert encoded == [0] * 8
    assert unicode_to_str(encoded) == ""


def test_resolution_str_format_parses_back() -> None:
    # tabs/node.py reconstructs (w, h) via label.split(" ")[0].split("x").
    for name, w, h in [
        (None, 960, 1280),
        ("u_texture", 1920, 1080),
        (None, 1080, 1080),
    ]:
        label = get_resolution_str(name, w, h)
        pw, ph = map(int, label.split(" ")[0].split("x"))
        assert (pw, ph) == (w, h)


def test_resolution_str_name_suffix() -> None:
    assert get_resolution_str("u_tex", 1920, 1080) == "1920x1080 (16:9) - u_tex"
    assert get_resolution_str(None, 1080, 1080) == "1080x1080 (1:1)"


def test_parse_shader_errors_nvidia_format() -> None:
    raw = "0(5) : error C0000: syntax error, unexpected identifier"
    errors = parse_shader_errors(raw)
    assert len(errors) == 1
    # Driver line is 1-based; editor DocPos is 0-based -> 5 becomes 4.
    assert errors[0].line == 4
    assert "syntax error" in errors[0].message


def test_parse_shader_errors_mesa_format() -> None:
    errors = parse_shader_errors("ERROR: 0:20: 'foo' : undeclared identifier")
    assert len(errors) == 1
    assert errors[0].line == 19
    assert "undeclared identifier" in errors[0].message


def test_parse_shader_errors_multi_line_picks_only_error_lines() -> None:
    raw = (
        "GLSL Compiler failed\n\n"
        "fragment_shader\n"
        "===============\n"
        "0(5) : error C0000: missing ';'\n"
        "0(8) : error C1101: undefined variable 'x'\n"
    )
    errors = parse_shader_errors(raw)
    assert [e.line for e in errors] == [4, 7]


def test_parse_shader_errors_unparseable_falls_back_to_raw() -> None:
    raw = "something the regexes don't recognise"
    errors = parse_shader_errors(raw)
    assert len(errors) == 1
    assert errors[0].line == -1
    assert errors[0].message == raw


def test_find_uniform_declaration_line_hit() -> None:
    source = "#version 460 core\nout vec4 c;\nuniform float u_time;\n"
    assert find_uniform_declaration_line(source, "u_time") == 2


def test_find_uniform_declaration_line_miss_returns_none() -> None:
    source = "#version 460 core\nuniform float u_time;\n"
    assert find_uniform_declaration_line(source, "u_aspect") is None


def test_find_uniform_declaration_line_picks_the_right_one() -> None:
    source = (
        "uniform float u_time;\n"
        "uniform vec3 u_color = vec3(0.5);\n"
        "uniform vec2 u_drag_vec2 = vec2(0.5, 0.5);\n"
    )
    assert find_uniform_declaration_line(source, "u_color") == 1
    assert find_uniform_declaration_line(source, "u_drag_vec2") == 2


def test_find_uniform_declaration_line_no_substring_false_match() -> None:
    # u_col must not match a declaration of u_color (word-boundary).
    source = "uniform vec3 u_color;\n"
    assert find_uniform_declaration_line(source, "u_col") is None


def test_find_uniform_declaration_line_array_decl() -> None:
    source = "uniform float u_arr[4];\n"
    assert find_uniform_declaration_line(source, "u_arr") == 0


def test_find_uniform_declaration_line_ignores_name_in_comment() -> None:
    # The name appearing in a comment/initializer must NOT win over the real decl.
    source = "uniform float u_b = 0.0; // tied to u_target\nuniform float u_target;\n"
    assert find_uniform_declaration_line(source, "u_target") == 1


def test_format_auto_value_scalar_and_vector() -> None:
    assert format_auto_value(0.5) == "0.500"
    assert format_auto_value(2) == "2.000"
    assert format_auto_value([0.5, 1.0, 0.25]) == "[0.500, 1.000, 0.250]"
    assert format_auto_value((0.5, 0.5)) == "[0.500, 0.500]"


def test_format_auto_value_non_numeric_falls_back_to_str() -> None:
    assert format_auto_value(None) == "None"


def test_next_error_line_after_and_wrap() -> None:
    errors = [ShaderError(4, "a"), ShaderError(7, "b"), ShaderError(9, "c")]
    assert next_error_line(errors, 0) == 4
    assert next_error_line(errors, 4) == 7  # strictly after
    assert next_error_line(errors, 9) == 4  # wraps to the first
    assert next_error_line(errors, 100) == 4


def test_next_error_line_skips_unparseable_and_dedups() -> None:
    errors = [ShaderError(-1, "raw"), ShaderError(5, "a"), ShaderError(5, "dup")]
    assert next_error_line(errors, 0) == 5
    assert next_error_line([ShaderError(-1, "raw")], 0) is None
    assert next_error_line([], 0) is None
