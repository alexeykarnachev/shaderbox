"""The intel module's pure sources (078 W-A): the buffer read as text, the script read
statically. Neither touches GL or the App."""

import ast
from pathlib import Path

import numpy as np

from shaderbox.editor.ffi import Slot
from shaderbox.intel.glsl import (
    buffer_declarations,
    buffer_words,
    output_declarations,
    uniform_declarations,
)
from shaderbox.intel.script import (
    _literal_type,
    glsl_type_of_value,
    returned_uniforms,
)
from shaderbox.intel.symbols import SymbolKind, kind_rank
from shaderbox.paths import DOCUMENT_SCRIPT_BASENAME
from shaderbox.scripting.engine import ScriptEngine
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


def test_a_value_infers_the_type_a_literal_would_have() -> None:
    # The maintainer's case: a script returning a VARIABLE parses to no shape, so `uniform `
    # never offered the name and the uniform read as "does not autocomplete". The value knows.
    assert glsl_type_of_value([0.1, 0.2, 0.3, 0.4]) == "vec4"
    assert glsl_type_of_value(0.5) == "float"
    assert glsl_type_of_value(3) == "int"
    assert glsl_type_of_value(True) == "bool"
    assert glsl_type_of_value((1.0, 2.0)) == "vec2"
    assert glsl_type_of_value([0.0] * 6) == "float[6]"
    # Nothing sensible: an empty sequence names no shape, a dict is a pass block, a string is
    # neither. Each stays None rather than becoming a declaration that cannot compile.
    assert glsl_type_of_value([]) is None
    assert glsl_type_of_value({"u_x": 1.0}) is None
    assert glsl_type_of_value("hello") is None
    # A sequence the COERCION would refuse names no type either, so the completion never seeds a
    # declaration the next tick rejects: bools are not numbers to it, and neither is a float32.
    assert glsl_type_of_value([True, False]) is None
    assert glsl_type_of_value([np.float32(1.0), np.float32(2.0)]) is None
    # A plain numpy array IS accepted -- its elements are float64, which is a real float.
    assert glsl_type_of_value(np.array([1.0, 2.0, 3.0, 4.0])) == "vec4"


def test_the_engine_types_a_key_whose_value_is_a_variable(tmp_path: Path) -> None:
    source = (
        "from shaderbox.scripting import ScriptBehavior, ScriptContext\n"
        "\n"
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, context: ScriptContext) -> dict:\n"
        "        brush_position = [0.1, 0.2, 0.3, 0.4]\n"
        "        return {'paint': {'u_brush_position': brush_position}}\n"
    )
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / DOCUMENT_SCRIPT_BASENAME).write_text(source)
    engine = ScriptEngine()
    engine.reload("doc", scripts, None)
    # The static reader sees no shape here...
    assert returned_uniforms(source)[0].glsl_type is None
    # ...and the tick that produced the value does.
    assert engine.returned_value_types("doc") == {("paint", "u_brush_position"): "vec4"}


def test_a_branching_script_does_not_type_one_pass_from_another(tmp_path: Path) -> None:
    # The tick samples ONE frame, so a script that returns different shapes on different frames
    # reports only what frame 0 produced. A pass-scoped key the tick never saw must stay untyped:
    # taking the broadcast entry's type instead offers `uniform vec2` for a uniform holding four
    # floats, which compiles and then fails every frame in the coercion.
    source = (
        "from shaderbox.scripting import ScriptBehavior, ScriptContext\n"
        "\n"
        "class Behavior(ScriptBehavior):\n"
        "    def update(self, context: ScriptContext) -> dict:\n"
        "        v2 = [0.1, 0.2]\n"
        "        v4 = [0.1, 0.2, 0.3, 0.4]\n"
        "        if context.frame > 0:\n"
        "            return {'paint': {'u_x': v4}}\n"
        "        return {'u_x': v2}\n"
    )
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / DOCUMENT_SCRIPT_BASENAME).write_text(source)
    engine = ScriptEngine()
    engine.reload("doc", scripts, None)
    runtime = engine.returned_value_types("doc")
    # Frame 0 took the broadcast branch, so the paint-scoped key has no entry at all.
    assert runtime == {("", "u_x"): "vec2"}
    assert ("paint", "u_x") not in runtime
