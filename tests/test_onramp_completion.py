"""104 item 4: `vs_quad` is offered to autocomplete. Before this, the one name an author
cannot guess was offered nowhere -- reaching for `vs_uv` instead draws a canvas-wide
vignette with no error, so a silent wrong guess was the only path.
"""

from dataclasses import replace

from shaderbox.engine_uniforms import ENGINE_UNIFORM_TYPES
from shaderbox.help_content import ENGINE_UNIFORM_DOCS
from shaderbox.instanced import QUAD_VARYING
from shaderbox.intel.index import GlslContext, build_glsl_index
from shaderbox.intel.symbols import SymbolKind

_BASE = GlslContext(
    text="uniform float u_time;\nvoid main() { gl_FragColor = vec4(1.0); }\n",
    engine_types=ENGINE_UNIFORM_TYPES,
    engine_docs=ENGINE_UNIFORM_DOCS,
    lib_functions={},
    pass_name="swarm",
    passes=("swarm",),
)


def test_vs_quad_is_offered_as_a_whole_declaration_when_not_yet_declared() -> None:
    index = build_glsl_index(_BASE)
    assert QUAD_VARYING in index.symbols
    symbol = index.symbols[QUAD_VARYING]
    assert symbol.kind == SymbolKind.GLSL_VARIABLE
    assert symbol.inserted == f"in vec2 {QUAD_VARYING};"
    assert QUAD_VARYING in [s.name for s in index.words]


def test_vs_quad_is_not_reoffered_once_the_buffer_declares_it() -> None:
    # Once declared, further completion should insert the bare name like any other symbol
    # already in the buffer -- not re-suggest a second `in vec2 vs_quad;` line.
    text = (
        "in vec2 vs_quad;\nuniform float u_time;\n"
        "void main() { gl_FragColor = vec4(vs_quad, 0.0, 1.0); }\n"
    )
    index = build_glsl_index(replace(_BASE, text=text))
    symbol = index.symbols[QUAD_VARYING]
    assert symbol.inserted == QUAD_VARYING


def test_vs_quad_never_reaches_the_after_uniform_declaration_site() -> None:
    # vs_quad is a VARYING, never a uniform -- it must not be offered through the
    # `declarations` list, which completion.py's DECLARATION_SITE feeds only after the
    # literal word `uniform `. Fed there, _declaration_parts's 3-word assumption would
    # produce the lie `uniform vec2 vs_quad;`.
    index = build_glsl_index(_BASE)
    assert QUAD_VARYING not in {s.name for s in index.declarations}


def test_a_lib_file_offers_no_vertex_stage_vocabulary() -> None:
    # A lib file has no pass and no vertex stage of its own (`if context.pass_name is not
    # None:` is the whole block's guard).
    index = build_glsl_index(replace(_BASE, pass_name=None, passes=()))
    assert QUAD_VARYING not in index.symbols
