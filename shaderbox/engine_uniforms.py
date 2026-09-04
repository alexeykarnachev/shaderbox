"""The engine-driven uniforms, as names and GLSL types: a GL-free leaf so the editor's
intelligence, the help panel and the copilot's tables read them without importing the
renderer."""

from shaderbox.glyph_tables import TABLE_UNIFORMS

# Engine-driven: never pass-intrinsic defaults — seed_uniform_values skips them and
# UIDocument.save excludes them. Two kinds: per-frame values Pass.render() recomputes
# from time/canvas, and the program-resident glyph tables Pass.compile() writes once
# (TABLE_UNIFORMS — render() skips those entirely).
# The GLSL type each per-frame engine uniform must be declared with. The engine writes these
# values itself, and moderngl refuses a value of the wrong shape, so a declaration of another
# type would leave the uniform at zero every frame; compile() rejects it instead.
ENGINE_UNIFORM_TYPES: dict[str, str] = {
    "u_time": "float",
    "u_aspect": "float",
    "u_resolution": "vec2",
    "u_pass_iteration": "float",
    "u_pass_iterations": "float",
}
ENGINE_DRIVEN_UNIFORMS: frozenset[str] = frozenset(
    ENGINE_UNIFORM_TYPES.keys() | TABLE_UNIFORMS.keys()
)
