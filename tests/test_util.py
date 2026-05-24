"""Pure unit tests for the text/unicode and resolution-string helpers — no GL.

The unicode round-trip guards the text-uniform crash (a full char[N] with no
trailing zero must decode without raising). The resolution tests pin the display
format that tabs/node.py parses back to (w, h).
"""

from shaderbox.util import get_resolution_str, str_to_unicode, unicode_to_str


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
    for name, w, h in [(None, 960, 1280), ("u_texture", 1920, 1080), (None, 1080, 1080)]:
        label = get_resolution_str(name, w, h)
        pw, ph = map(int, label.split(" ")[0].split("x"))
        assert (pw, ph) == (w, h)


def test_resolution_str_name_suffix() -> None:
    assert get_resolution_str("u_tex", 1920, 1080) == "1920x1080 (16:9) - u_tex"
    assert get_resolution_str(None, 1080, 1080) == "1080x1080 (1:1)"
