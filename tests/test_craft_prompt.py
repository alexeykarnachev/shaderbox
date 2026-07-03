"""Feature 054: guard that the lab-mined craft knowledge is actually WIRED into the copilot prompt +
vision system prompt (a silent revert would quietly regress the whole capability). Marker-string checks
-- cheap, and they fail loudly if the block is dropped or a section is gutted."""

from shaderbox.copilot.llm.openrouter import _VISION_SYSTEM
from shaderbox.copilot.prompt import _SYSTEM_PROMPT


def test_system_prompt_carries_the_visual_craft_block() -> None:
    p = _SYSTEM_PROMPT
    assert "VISUAL CRAFT" in p
    # The load-bearing craft levers (each a distinct failure the flag dogfood exposed).
    for marker in (
        "FIDELITY FIRST",  # match the asked look, not a lazy in-between
        "tonemap",  # never ship a dark muddy frame
        "SHADOWS > AO > normals",  # depth hierarchy
        "EMERGE from feedback",  # motion, not an imposed sine
        "PLAN COMPLEX WORK FIRST",  # decompose before diving into edits
        "BUILD IN STAGES",  # robustness vs the token budget
        "iterate until the render MATCHES",  # convergence discipline
        "aces(",  # an embedded formula the weak model shouldn't recall from memory
    ):
        assert marker in p, marker


def test_scripting_section_teaches_physics_via_script() -> None:
    # Tool-selection: a physics sim belongs in a script pushing an array uniform, not faked per-pixel.
    p = _SYSTEM_PROMPT
    assert "PHYSICS" in p and "Verlet" in p and "ARRAY uniform" in p


def test_vision_system_prompt_has_the_legibility_dimension() -> None:
    # 054: the eye must report a muddy/dark/washed-out subject, not gloss it as "structured, fine".
    assert "readability:" in _VISION_SYSTEM
    assert "muddy" in _VISION_SYSTEM
