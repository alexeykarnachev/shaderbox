"""The intel module's pure sources (078 W-A): the buffer read as text, the script read
statically. Neither touches GL or the App."""

import ast

from shaderbox.editor.ffi import Slot
from shaderbox.intel.glsl import (
    buffer_declarations,
    buffer_words,
    output_declarations,
    uniform_declarations,
)
from shaderbox.intel.script import _literal_type, returned_uniforms
from shaderbox.intel.symbols import SymbolKind, kind_rank
from shaderbox.theme import COLOR, editor_palette, kind_color, kind_slot

_SHADER = """#version 330
// uniform float u_commented;
uniform float u_time;
uniform float u_aspect;
uniform sampler2D u_paint;
uniform vec4 u_colors[4];
const float PI = 3.14159;
#define STEPS 8
/* uniform int u_block; */
vec3 palette(float t, vec3 a) {
    return a + t;
}
void main() {
    vec3 color = palette(u_time, vec3(1.0));
    gl_FragColor = vec4(color, 1.0);
}
"""


def test_uniform_declarations_are_every_declared_uniform_read_or_not() -> None:
    # The finding-6 case: `u_aspect` is declared and never read; the text still declares it.
    found = uniform_declarations(_SHADER)
    assert [(u.name, u.glsl_type, u.line, u.array) for u in found] == [
        ("u_time", "float", 2, ""),
        ("u_aspect", "float", 3, ""),
        ("u_paint", "sampler2D", 4, ""),
        ("u_colors", "vec4", 5, "[4]"),
    ]
    assert found[3].declaration == "uniform vec4 u_colors[4];"


def test_comments_declare_nothing_and_lines_survive_stripping() -> None:
    names = {u.name for u in uniform_declarations(_SHADER)}
    assert "u_commented" not in names
    assert "u_block" not in names
    assert "u_commented" not in buffer_words(_SHADER)


def test_buffer_declarations_are_functions_constants_and_defines() -> None:
    found = buffer_declarations(_SHADER)
    assert [(d.name, d.signature, d.line) for d in found] == [
        ("PI", "const float PI", 6),
        ("STEPS", "#define STEPS", 7),
        ("palette", "vec3 palette(float t, vec3 a)", 9),
        ("main", "void main()", 12),
    ]


def test_buffer_words_are_the_identifiers_in_the_text() -> None:
    words = buffer_words(_SHADER)
    assert {"color", "palette", "u_aspect", "gl_FragColor"} <= words
    assert "330" not in words


_SCRIPT = """import math
from shaderbox.scripting import ScriptBehavior, ScriptContext

class Behavior(ScriptBehavior):
    def __init__(self) -> None:
        self.phase = 0.0

    def helper(self) -> dict:
        return {"u_not_ours": 1.0}

    def update(self, context: ScriptContext) -> dict:
        def inner() -> dict:
            return {"u_nested": 1.0}
        if context.frame == 0:
            return {"u_first": 0}
        return {
            "u_speed": 0.5,
            "u_count": -3,
            "u_on": True,
            "u_tint": [1.0, 0.0, 0.0],
            "u_taps": [0.0] * 6,
            "u_pts": [1.0, 2.0, 3.0, 4.0, 5.0],
            "u_phase": self.phase,
            "u_off": None,
            "paint": {"u_scale": 2.0, "u_dir": [0.0, 1.0]},
        }
"""


def test_returned_uniforms_read_every_return_of_update_only() -> None:
    found = returned_uniforms(_SCRIPT)
    by_name = {(r.pass_name, r.name): r.glsl_type for r in found}
    assert by_name == {
        (None, "u_first"): "int",
        (None, "u_speed"): "float",
        (None, "u_count"): "int",
        (None, "u_on"): "bool",
        (None, "u_tint"): "vec3",
        (None, "u_taps"): "float[6]",
        # 5 numbers is no vector, so it reads as an array; 2-4 read as a vector,
        # which is what a script writing `[x, y]` almost always means.
        (None, "u_pts"): "float[5]",
        (None, "u_phase"): None,
        (None, "u_off"): None,
        ("paint", "u_scale"): "float",
        ("paint", "u_dir"): "vec2",
    }
    names = [r.name for r in found]
    assert "u_not_ours" not in names
    assert "u_nested" not in names
    assert [r.name for r in found[:2]] == ["u_first", "u_speed"]
    assert found[0].line == 14


def test_a_script_that_does_not_parse_returns_nothing() -> None:
    assert returned_uniforms("def update(:\n") == ()
    assert returned_uniforms("x = 1\n") == ()


def test_every_kind_is_a_distinct_string() -> None:
    values = [kind.value for kind in SymbolKind]
    assert len(values) == len(set(values))


def test_every_kind_has_a_color() -> None:
    # The checker-narrowing guard: a kind added to the enum without a color fails here, not
    # at the first frame that draws it.
    palette = editor_palette()
    slots = {
        1: Slot.SYNTAX_1,
        2: Slot.SYNTAX_2,
        3: Slot.SYNTAX_3,
        4: Slot.SYNTAX_4,
        5: Slot.SYNTAX_5,
        6: Slot.SYNTAX_6,
        7: Slot.SYNTAX_7,
        8: Slot.SYNTAX_8,
        9: Slot.SYNTAX_9,
    }
    for kind in SymbolKind:
        assert len(kind_color(kind)) == 4
        assert kind_rank(kind) >= 0  # 079 D2: every kind sorts somewhere
        slot = kind_slot(kind)
        assert 0 <= slot <= 9
        if slot:
            # One color per kind: what the popup and the text draw is what a host surface
            # shows, by the palette rather than by coincidence.
            assert kind_color(kind) == palette[slots[slot]], kind


def test_the_fragment_output_is_scanned_and_the_near_misses_are_not() -> None:
    # 079 D11: the one name a shader WRITES gets its own color, so it has to be found without
    # catching what merely looks like it. Falsifier: match `out` anywhere and `inout` plus the
    # local named `out_thing` come back as fragment outputs.
    text = (
        "layout(location = 0) out vec4 fragColor;\n"
        "out vec3 second;\n"
        "inout vec4 not_an_output;\n"
        "// out vec4 commented;\n"
        "void main() { int out_thing = 1; fragColor = vec4(0.0); }\n"
    )
    found = output_declarations(text)
    assert [(d.name, d.glsl_type, d.line) for d in found] == [
        ("fragColor", "vec4", 0),
        ("second", "vec3", 1),
    ]


def test_the_output_variable_reads_orange_and_sorts_with_the_buffers_own_names() -> (
    None
):
    # The token, not a hex (079 D11's "generalizable across themes"), and slot 9 so the library
    # draws it in the same color the popup does.
    assert kind_color(SymbolKind.OUTPUT_VARIABLE) == COLOR.SYN_OUTPUT
    assert kind_slot(SymbolKind.OUTPUT_VARIABLE) == 9
    assert kind_rank(SymbolKind.OUTPUT_VARIABLE) == kind_rank(SymbolKind.BUFFER_SYMBOL)


def test_an_empty_literal_offers_no_declaration() -> None:
    # `uniform float[0] u_x;` does not compile, and the shader side offers a returned key's
    # declaration as a one-click insertion — so an empty literal must name no shape at all.
    # Falsifier: return `float[0]` again and this hands the user a line GLSL rejects.
    assert _literal_type(ast.parse("[]", mode="eval").body) is None
    assert _literal_type(ast.parse("[1.0]", mode="eval").body) == "float[1]"
