"""Unit tests for the 033 enriched-result facts — pure text/pixel, no GL."""

from shaderbox.copilot.edit_hints import compile_hints, render_facts


def test_redeclared_hint_lists_both_declaration_lines() -> None:
    src = (
        "float weight = 0.1;\n"
        "float x = 1.0;\n"
        "float weight = 0.2;\n"
        "void main() {}\n"
    )
    hints = compile_hints(src, ["`weight' redeclared"])
    assert len(hints) == 1
    assert "'weight' is declared on lines 1, 3" in hints[0]


def test_redefined_function_and_blob_multi_name() -> None:
    # Mesa says `redefined` for duplicate FUNCTIONS, and V3D delivers all errors
    # as one blob — both names must get hints.
    src = (
        "float weight = 0.1;\n"
        "float text_sdf(vec2 p) { return 0.0; }\n"
        "float weight = 0.2;\n"
        "float text_sdf(vec2 p) { return 1.0; }\n"
    )
    blob = "0:3(7): error: `weight' redeclared\n0:4(7): error: `text_sdf' redefined"
    hints = compile_hints(src, [blob])
    assert any("'weight' is declared on lines 1, 3" in h for h in hints)
    assert any("'text_sdf' is declared on lines 2, 4" in h for h in hints)


def test_hint_lines_survive_block_comments() -> None:
    src = (
        "/* a\n   b\n   c */\n"
        "float weight = 0.1;\n"
        "float weight = 0.2;\n"
        "void main() {}\n"
    )
    hints = compile_hints(src, ["`weight' redeclared"])
    assert "'weight' is declared on lines 4, 5" in hints[0]


def test_initializer_count_hint() -> None:
    hints = compile_hints(
        "void main() {}\n",
        [
            "initializer of type uint[73] cannot be assigned to variable of type uint[64]"
        ],
    )
    assert len(hints) == 1
    assert "73 elements, the array wants 64" in hints[0]
    assert "uint[]" in hints[0]


def test_redeclared_hint_covers_inout_qualifiers() -> None:
    src = "out vec4 fs_color;\nuniform float x;\nout vec4 fs_color;\nvoid main() {}\n"
    hints = compile_hints(src, ["`fs_color' redeclared"])
    assert hints and "'fs_color' is declared on lines 1, 3" in hints[0]


def test_brace_imbalance_hint() -> None:
    src = "void main() {\n    float x = 1.0;\n}\n}\n"
    hints = compile_hints(src, ["syntax error, unexpected '}', expecting end of file"])
    assert any("1 '{' vs 2 '}'" in h for h in hints)


def test_no_hints_for_unrelated_error_on_balanced_source() -> None:
    src = "void main() { float x = y; }\n"
    assert compile_hints(src, ["`y' undeclared"]) == []


def _frame(width: int, height: int, painter) -> bytes:
    buf = bytearray(width * height * 4)
    for y in range(height):
        for x in range(width):
            r, g, b = painter(x, y)
            o = (y * width + x) * 4
            buf[o : o + 4] = bytes((r, g, b, 255))
    return bytes(buf)


def test_render_facts_empty_frame() -> None:
    raw = _frame(12, 12, lambda x, y: (10, 10, 10))
    assert "EMPTY" in render_facts(raw, 12, 12)


def test_render_facts_bbox_and_orientation() -> None:
    # A white square in the BOTTOM-left quarter (GL rows are bottom-up, so low y
    # indices = bottom of the screen = vs_uv y near 0).
    def painter(x: int, y: int) -> tuple[int, int, int]:
        if x < 6 and y < 6:
            return (255, 255, 255)
        return (0, 0, 0)

    facts = render_facts(_frame(12, 12, painter), 12, 12)
    assert "ink 25%" in facts
    assert "bbox x 0.00-0.50" in facts
    assert "y 0.00-0.50" in facts  # bottom half, y=0 bottom convention
    # The luma grid prints top row first: the bright cell must be in the LAST row.
    grid_part = facts.split("rows: ")[1]
    top_row, _mid_row, bottom_row = grid_part.split(" / ")
    assert top_row == "0 0 0"
    assert bottom_row.startswith("9")
