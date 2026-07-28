"""Feature 054: guard that the lab-mined craft knowledge is actually WIRED into the copilot prompt
(a silent revert would quietly regress the whole capability). Marker-string checks -- cheap, and
they fail loudly if the block is dropped or a section is gutted."""

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
        "NOT done at first clean compile",  # iterate-until-it-matches discipline
        "aces(",  # an embedded formula the weak model shouldn't recall from memory
    ):
        assert marker in p, marker


def test_scripting_section_teaches_physics_via_script() -> None:
    # Tool-selection: a physics sim belongs in a script pushing an array uniform, not faked per-pixel.
    p = _SYSTEM_PROMPT
    assert "PHYSICS" in p and "Verlet" in p and "ARRAY uniform" in p


def test_system_prompt_states_the_agent_cannot_see_its_render() -> None:
    # 058: the copilot has no eye — the prompt must say the facts line is the only signal and that
    # how it LOOKS is the user's call, or the model narrates visual results it never observed.
    p = _SYSTEM_PROMPT
    assert "You never SEE your render" in p
    assert "let the USER" in p


def test_conventions_carry_the_aspect_correction_rule() -> None:
    # A/B-validated (2026-07-28): without this line a "centered circle" on a 16:9 canvas renders
    # as an ellipse tracking the canvas (aspect 1.78); with it, round (1.00). Guard the wire.
    from shaderbox.copilot.prompt_context import _CONVENTIONS

    assert "aspect-corrected coordinates" in _CONVENTIONS
    assert "NOT square" in _CONVENTIONS


def test_conventions_carry_the_uv_y_direction_rule() -> None:
    # Sweep-validated (2026-07-28): grid runs put the "top row" at the uv-y bottom in 3 of 5
    # runs without this line; 2 of 2 built runs placed rows correctly with it.
    from shaderbox.copilot.prompt_context import _CONVENTIONS

    assert "y=0 is the BOTTOM" in _CONVENTIONS
    assert "top row" in _CONVENTIONS


def test_craft_block_teaches_local_frames() -> None:
    # Maintainer-directed 3D/SDF lesson (2026-07-28), generalized: local frames + surface detail
    # via the dominant-axis face pick. Fixture-validated: the die run applied local-frame carving
    # after this landed (it never did before).
    from shaderbox.copilot.prompt import _SYSTEM_PROMPT

    assert "LOCAL FRAMES" in _SYSTEM_PROMPT
    assert "INVERSE" in _SYSTEM_PROMPT
    assert "DOMINANT axis" in _SYSTEM_PROMPT
