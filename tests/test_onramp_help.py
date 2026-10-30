"""The pinned false-invariant list for the help content. GL-free.

This file gates "no help text asserts an invariant an instanced pass breaks". A checker
cannot decide "asserts an invariant", so the gate is a PINNED LIST of exact phrases known to
be false for an instanced pass. It covers those phrases and NOTHING ELSE -- a new false
sentence needs its own line added here, not a smarter checker.
"""

from shaderbox.help_content import help_sections

# Every phrase here is a claim about the shader-skeleton section that is false for an
# instanced pass: "ShaderBox draws a full-screen quad
# and runs your main() once per pixel" as an unqualified claim, and "the vs_uv input" listed
# among "three things [that] are fixed" -- an instanced pass gets vs_quad, not vs_uv, and
# draws one quad per entity, not one fragment shader over the canvas.
_FALSE_FOR_AN_INSTANCED_PASS: tuple[str, ...] = (
    "ShaderBox draws a full-screen quad and runs your `main()` once per pixel.",
    "Three things are fixed: the `#version` line (required — nothing is injected for "
    "you), the `vs_uv` input, and a single `vec4` output.",
)


def test_no_help_text_repeats_the_pre_onramp_false_invariants() -> None:
    joined = "\n".join(s.body for s in help_sections())
    stale = [phrase for phrase in _FALSE_FOR_AN_INSTANCED_PASS if phrase in joined]
    assert stale == [], (
        f"help prose restates a phrase false for an instanced pass: {stale}"
    )
