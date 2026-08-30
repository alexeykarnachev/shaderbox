"""Feature 054: guard that the lab-mined craft knowledge is actually WIRED into the copilot prompt
(a silent revert would quietly regress the whole capability). Marker-string checks -- cheap, and
they fail loudly if the block is dropped or a section is gutted."""

from shaderbox.copilot.prompt import _RENDER_FACTS_LEGEND, _SYSTEM_PROMPT
from shaderbox.copilot.prompt_context import _CONVENTIONS
from shaderbox.copilot.tools.publish import (
    _PUBLISH_TELEGRAM_DESC,
    _PUBLISH_YOUTUBE_DESC,
    _RENDER_IMAGE_DESC,
    _RENDER_VIDEO_DESC,
)
from shaderbox.copilot.tools.script import _READ_SCRIPT_DESC, _WRITE_SCRIPT_DESC


def _flat(text: str) -> str:
    # Collapse the prompt's hard wrapping so a phrase pin matches content, not layout.
    return " ".join(text.split())


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


def test_scripting_section_teaches_the_state_watershed() -> None:
    # 059 D1: the routing rule is STATE (this frame needs last frame's value), NOT "changes over
    # time" — a pure function of time is GLSL. The heavy-stateful special case survives.
    p = _SYSTEM_PROMPT
    for marker in (
        "depends on the PREVIOUS frame",
        "PURE FUNCTION OF TIME",
        "needs NO script",
        "Verlet",
        "ARRAY uniform",
    ):
        assert marker in p, marker
    # The deleted routing table's headline: its return means the wrong rule is back.
    assert "VALUES THAT CHANGE" not in p


def test_script_tool_descriptions_carry_the_same_watershed() -> None:
    # 059 D1a: the eager descriptions ride `tools=` on EVERY iteration — the corrected rule and its
    # negation must never share a context window.
    assert "depends on the PREVIOUS frame" in _WRITE_SCRIPT_DESC
    assert "belongs in the shader (u_time)" in _WRITE_SCRIPT_DESC
    assert "ANIMATION" not in _WRITE_SCRIPT_DESC
    # 059 D1a: the stub claim is the one the generator actually emits (an empty `update`).
    assert "ctx.t example" not in _READ_SCRIPT_DESC


def test_script_surface_hides_the_on_disk_path() -> None:
    # 059 D2: the agent gets handles, not implementation detail — no script path in the prompt or
    # in the description that rides every iteration.
    assert "scripts/script.py" not in _SYSTEM_PROMPT
    assert "documents/" not in _READ_SCRIPT_DESC


def test_system_prompt_states_the_agent_cannot_see_its_render() -> None:
    # 058: the copilot has no eye — the prompt must say the facts line is the only signal and that
    # how it LOOKS is the user's call, or the model narrates visual results it never observed.
    p = _SYSTEM_PROMPT
    assert "You never SEE your render" in p
    assert "let the USER" in p


def test_conventions_carry_the_aspect_correction_rule() -> None:
    # A/B-validated (2026-07-28): without this line a "centered circle" on a 16:9 canvas renders
    # as an ellipse tracking the canvas (aspect 1.78); with it, round (1.00). Guard the wire.
    assert "aspect-corrected coordinates" in _CONVENTIONS
    assert "NOT square" in _CONVENTIONS


def test_conventions_carry_the_uv_y_direction_rule() -> None:
    # Sweep-validated (2026-07-28): grid runs put the "top row" at the uv-y bottom in 3 of 5
    # runs without this line; 2 of 2 built runs placed rows correctly with it.
    assert "y=0 is the BOTTOM" in _CONVENTIONS
    assert "top row" in _CONVENTIONS


def test_the_text_const_array_rule_lives_with_the_glsl_domain_rules() -> None:
    # 059 D4: a GLSL domain fact, not editing mechanics — it moves to the RARE conventions block, and
    # names its measured reason (glyphs.glsl above SBT_SPANS: a dynamically indexed const array is
    # demoted to per-thread local memory on NVIDIA, ~100x slower). This is the one 059 cut whose
    # regression the dogfood CANNOT see: the shader compiles clean and just renders slowly.
    assert "NEVER a const array in" in _CONVENTIONS
    assert "~100x" in _CONVENTIONS
    assert "const array" not in _SYSTEM_PROMPT


def test_the_prompt_does_not_restate_what_the_tool_schemas_carry() -> None:
    # 059 D5: every marker below is verbatim-covered by an eager tool description or a lazy tool's
    # catalog_summary. Their return means the duplication is back.
    p = _SYSTEM_PROMPT
    for cut in (
        "`set_uniform(name, value)`",  # _SET_UNIFORM_DESC + _SetUniformArgs.value
        "`create_document(name)`",  # _CREATE_NODE_DESC + its arg descriptions
        "user declined",  # _DELETE_NODE_DESC
        "`switch_document(document)` makes",  # _SWITCH_NODE_DESC
        "`read_lib(names)`",  # _READ_LIB_DESC
        "`grep(query)`",  # _GREP_DESC
        "short_720",  # _SHAPE_DESC on all three render/publish arg models
        "briefly pauses",  # _RENDER_IMAGE_DESC / _RENDER_VIDEO_DESC
        "land edits first",  # ditto (as the corrected "land your edits before rendering")
        "#include",  # triplicated: the RARE catalogue header + _CONVENTIONS bullet 2
        "NEVER means",  # _ReadShaderArgs.documents / _TARGET_DESC
        "lib:` prefix",  # ditto
        "duplicate_document",  # lazy: load_tools' catalogue row
        "set_canvas_size",  # lazy: ditto
        "rename_document",  # lazy: ditto
        "unbind_media",  # lazy: ditto
        "list_telegram_packs",  # lazy: ditto + telegram_precheck's handoff
        "set_youtube_credentials",  # lazy: ditto + youtube_precheck's handoff
    ):
        assert cut not in p, cut
    # What CANNOT live in any single schema survives: the cross-tool order and the reply behaviour.
    assert "Cross-tool order" in p
    assert "never deflect the user to Settings" in p
    assert "never invent a path" in p


def test_render_and_publish_descriptions_do_not_promise_a_path_or_url() -> None:
    # 059 D5's audit rule: overlap resolves in the SCHEMA's favour only once the schema is RIGHT.
    # These four claimed a return the model never gets (the path/URL is payload-only, surfaced to the
    # USER as a button) — cutting the prompt's correction is safe only because they are now correct.
    for desc in (_RENDER_IMAGE_DESC, _RENDER_VIDEO_DESC):
        assert "Reveal render" in desc
        assert "NEITHER the path" in desc
        assert "You render the live source" in desc
    assert "you do NOT get the URL" in _PUBLISH_TELEGRAM_DESC
    assert "you do NOT get the Studio URL" in _PUBLISH_YOUTUBE_DESC
    # The deflection the prompt rule forbids, shipped inside the tool the rule is about.
    assert "must be connected in Settings" not in _PUBLISH_YOUTUBE_DESC
    assert "set_youtube_credentials" in _PUBLISH_YOUTUBE_DESC


def test_craft_block_teaches_local_frames() -> None:
    # Maintainer-directed 3D/SDF lesson (2026-07-28), generalized: local frames + surface detail
    # via the dominant-axis face pick. Fixture-validated: the die run applied local-frame carving
    # after this landed (it never did before).
    assert "LOCAL FRAMES" in _SYSTEM_PROMPT
    assert "INVERSE" in _SYSTEM_PROMPT
    assert "DOMINANT axis" in _SYSTEM_PROMPT


def test_feedback_keeps_the_pre_action_rules_and_sheds_the_legend() -> None:
    # 059 D7: only the POST-action gloss travels with the facts line. The three pre-action rules
    # each fix a recorded regression (618af96 relative-colour verify, c39caac blank-cold-t0-develops,
    # 2ff249f no-op-don't-reapply) and must stay in STATIC — on the turn the user says "make it
    # warmer" there is no facts line yet to carry them.
    # Both texts are hard-wrapped, so a multi-word phrase can straddle a line break — assert
    # against the unwrapped text or a pin fails on layout rather than on content.
    p = _flat(_SYSTEM_PROMPT)
    legend = _flat(_RENDER_FACTS_LEGEND)
    for stays in (
        "ink mean rgb",  # 618af96: verify a relative colour ask against the measurement
        "read it back after the edit",
        "DEVELOPS over time",  # c39caac: blank/cold t=0 + ANIMATES is not a failed edit
        "re-apply the same edit",  # 2ff249f: a no-op means find the cause, don't repeat it
        "It MEASURES, it does not judge",
    ):
        assert stays in p, stays
    # The self-non-glossing terms ride the facts line instead (option C) — including the one
    # DIAGNOSTIC the emitted verdict cannot state (STATIC reports an unchanged frame; that the
    # u_time wiring is missing is the reading). None of them may survive in STATIC too.
    for moved in (
        "how to read the line above",
        "share of pixels differing",
        "corner-sampled",
        "alpha-weighted",
        "3x3 brightness",
        "motion: STATIC when you meant it to move",
    ):
        assert moved not in p, moved
        assert moved in legend, moved
