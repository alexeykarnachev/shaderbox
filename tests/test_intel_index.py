"""The GLSL index (078 W-A): the buffer is the source, a sampler follows its value, the
script's returns and the other passes become declarations to offer."""

import subprocess
import sys
from dataclasses import replace

from shaderbox.engine_uniforms import ENGINE_UNIFORM_TYPES
from shaderbox.help_content import ENGINE_UNIFORM_DOCS
from shaderbox.intel.index import GlslContext, build_glsl_index
from shaderbox.intel.script import ScriptReturn
from shaderbox.intel.symbols import SymbolKind
from shaderbox.pass_graph import NoSource, PassSource

_TEXT = """uniform float u_time;
uniform float u_gain;
uniform sampler2D u_paint;
uniform sampler2D u_mask;
uniform float u_speed;
vec3 palette(float t) { return vec3(t); }
void main() { vec3 color = palette(u_time) * u_speed; gl_FragColor = vec4(color, 1.0); }
"""


_BASE = GlslContext(
    text=_TEXT,
    engine_types=ENGINE_UNIFORM_TYPES,
    engine_docs=ENGINE_UNIFORM_DOCS,
    lib_functions={"SB_hash": ("float SB_hash(vec2 p)", "a hash")},
    script_returns=(
        ScriptReturn("u_speed", None, "float", 10),
        ScriptReturn("u_tint", None, "vec3", 11),
        ScriptReturn("u_phase", None, None, 12),
        ScriptReturn("u_other", "other", "float", 13),
    ),
    pass_name="glow",
    passes=("glow", "paint", "other"),
    sampler_values={"u_mask": NoSource()},
)


def _context(**overrides: object) -> GlslContext:
    return replace(_BASE, **overrides)


def test_declared_uniforms_are_classified_from_the_text_and_the_wiring() -> None:
    index = build_glsl_index(_context())
    kinds = {
        name: index.symbols[name].kind
        for name in ("u_time", "u_gain", "u_paint", "u_mask", "u_speed")
    }
    assert kinds == {
        "u_time": SymbolKind.ENGINE_UNIFORM,
        "u_gain": SymbolKind.PASS_UNIFORM,
        "u_paint": SymbolKind.PASS_SAMPLER,
        "u_mask": SymbolKind.PASS_UNIFORM,
        "u_speed": SymbolKind.SCRIPT_UNIFORM,
    }
    assert index.symbols["u_paint"].doc == "reads pass paint"
    # The identifier list opens with the document's own names, the unread u_gain among them.
    assert "u_gain" in [s.name for s in index.words[:8]]


def test_a_sampler_follows_its_value() -> None:
    to_file = build_glsl_index(_context(sampler_values={"u_paint": object()}))
    assert to_file.symbols["u_paint"].kind == SymbolKind.PASS_UNIFORM
    explicit = build_glsl_index(
        _context(sampler_values={"u_mask": PassSource("other")})
    )
    assert explicit.symbols["u_mask"].kind == SymbolKind.PASS_SAMPLER
    assert explicit.symbols["u_mask"].doc == "reads pass other"


def test_declarations_are_what_the_buffer_lacks() -> None:
    index = build_glsl_index(_context())
    offered = {s.name: s for s in index.declarations}
    assert "u_time" not in offered and "u_gain" not in offered
    assert offered["u_aspect"].inserted == "uniform float u_aspect;"
    assert offered["u_resolution"].inserted == "uniform vec2 u_resolution;"
    assert offered["u_resolution"].kind == SymbolKind.ENGINE_UNIFORM
    assert offered["u_tint"].inserted == "uniform vec3 u_tint;"
    assert offered["u_tint"].kind == SymbolKind.SCRIPT_UNIFORM
    assert offered["u_other"].inserted == "uniform sampler2D u_other;"
    assert offered["u_other"].kind == SymbolKind.WIRABLE_SAMPLER
    assert offered["u_prev"].doc == "this pass's previous frame"
    assert "u_paint" not in offered
    # A script uniform with no literal shape is a name, never a guessed declaration.
    # TODO: the maintainer hit this from the other side -- a script returning a VARIABLE
    # (`{"paint": {"u_brush_position": brush_position}}`) gives no inferable type, so the
    # name never reaches a `uniform ` site and reads as "my uniform does not autocomplete".
    # Offering it only where the site ALREADY carries a type (`uniform vec2 |`) inserts the
    # bare name, is compile-safe, and is what `completion.py::_declarations` already says it
    # does ("a name-only script uniform fits any type"). Decide whether that narrowing is
    # wanted before changing this assertion -- it is a decision, not an oversight.
    assert "u_phase" not in offered
    assert index.symbols["u_phase"].inserted == "u_phase"
    assert index.symbols["u_phase"].kind == SymbolKind.SCRIPT_UNIFORM


def test_a_pass_block_scopes_its_uniforms_to_that_pass() -> None:
    glow = build_glsl_index(_context())
    other = build_glsl_index(_context(pass_name="other"))
    assert "u_other" in {
        s.name for s in other.declarations if s.kind == SymbolKind.SCRIPT_UNIFORM
    }
    assert glow.symbols["u_other"].kind == SymbolKind.WIRABLE_SAMPLER


def test_lookup_and_classes_and_the_lib_file_shape() -> None:
    index = build_glsl_index(_context())
    assert index.lookup("palette") is not None
    assert index.lookup("palette").signature == "vec3 palette(float t)"
    assert index.lookup("SB_hash").kind == SymbolKind.LIB_FUNCTION
    assert index.lookup("mix").kind == SymbolKind.GLSL_BUILTIN
    assert index.lookup("vec3").kind == SymbolKind.GLSL_TYPE
    assert index.lookup("color").kind == SymbolKind.BUFFER_SYMBOL
    assert index.classes() == {
        "u_time": SymbolKind.ENGINE_UNIFORM,
        "u_aspect": SymbolKind.ENGINE_UNIFORM,
        "u_resolution": SymbolKind.ENGINE_UNIFORM,
        "u_pass_iteration": SymbolKind.ENGINE_UNIFORM,
        "u_pass_iterations": SymbolKind.ENGINE_UNIFORM,
        "u_paint": SymbolKind.PASS_SAMPLER,
        "u_speed": SymbolKind.SCRIPT_UNIFORM,
        "u_tint": SymbolKind.SCRIPT_UNIFORM,
        "u_phase": SymbolKind.SCRIPT_UNIFORM,
        "SB_hash": SymbolKind.LIB_FUNCTION,
        # The fragment outputs (079 D11), orange in the text since the editor library dropped
        # `gl_FragColor` from its builtins at `38cadbc`.
        "gl_FragColor": SymbolKind.OUTPUT_VARIABLE,
        "gl_FragData": SymbolKind.OUTPUT_VARIABLE,
    }
    lib = build_glsl_index(_context(pass_name=None, passes=(), script_returns=()))
    assert not [s for s in lib.declarations if s.kind == SymbolKind.WIRABLE_SAMPLER]


def test_intel_is_gl_free() -> None:
    # A fresh interpreter, so another test's imports cannot mask a GL pull. The GLSL half and
    # the completion policy; the Python half reaches `shaderbox.scripting` for the API glosses,
    # whose package `__init__` re-exports the engine, so it is App-free but not GL-free.
    probe = (
        "import sys, shaderbox.intel.index, shaderbox.completion; "
        "print(sorted(m for m in sys.modules if m == 'moderngl' or m.split('.')[0] in "
        "('OpenGL', 'glfw', 'imgui_bundle')))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]", out.stdout
