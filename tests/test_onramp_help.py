"""104's gates on the help content: the pinned false-invariant list, and the D4 partition
already covered by tests/test_help_content.py. GL-free.

This file covers the SECOND gate only -- "no help text asserts an invariant an instanced
pass breaks" -- because that gate is the wrong instrument written as prose analysis: a
checker cannot decide "asserts an invariant". Narrowed to a PINNED LIST of exact phrases
known to be false for an instanced pass (the wording that was in the tree before 104 fixed
item 1), this test covers those phrases and NOTHING ELSE -- a new false sentence someone
writes tomorrow needs its own line added here, not a smarter checker.
"""

from shaderbox.help_content import help_sections

# Every phrase here was the shipped text before 104 fixed the shader-skeleton section (item
# 1): both assert something false for an instanced pass. "ShaderBox draws a full-screen quad
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
