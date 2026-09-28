"""The contract features 102-105 are built against: the shared vocabulary, gated.

Every check here guards a type three or more flows read. They are in one file rather than
spread across the feature test modules because the thing under test is the CONTRACT -- a
flow that changes one of these changes what every other flow compiled against.
"""

from typing import get_args

from shaderbox.blend import blend_func_for
from shaderbox.instanced import (
    ENGINE_INTERNAL_NAMES,
    RESERVED_NAMES,
    USER_FACING_NAMES,
)
from shaderbox.instanced_outcome import (
    HEALTHY_INSTANCED_STATES,
    INSTANCED_STATES,
    InstancedOutcome,
)
from shaderbox.pass_graph import BLEND_MODES, TargetConfig
from shaderbox.scripting.keys import (
    KEY_FAIL_REASONS,
    SILENT_KEY_FAIL_REASONS,
    WARNING_KEY_FAIL_REASONS,
)


def test_every_target_field_is_classified_by_allocates_same_as() -> None:
    """`allocates_same_as` must mention every field of the model, or a field added later is
    silently treated as allocation-irrelevant -- which for a real one means a stale canvas
    of the wrong format, and for a draw-state one means a destroyed feedback trail.

    The domain is the MODEL's fields, not a list written here: a hand-written list is the
    domain-narrowing this exists to prevent. The contact proof is the assertion below that
    the classification actually discriminates; a method returning a constant would satisfy
    the coverage check alone.
    """
    allocation_relevant = {"scale", "dtype", "filter_linear", "wrap"}
    draw_state_only = {"blend"}
    assert allocation_relevant | draw_state_only == set(TargetConfig.model_fields), (
        "a TargetConfig field is in neither tier -- classify it in allocates_same_as"
    )

    base = TargetConfig()
    # Each allocation-relevant field, changed one at a time, must force a reallocation.
    for field, other in (
        ("scale", 0.5),
        ("dtype", "f4"),
        ("filter_linear", not base.filter_linear),
        ("wrap", not base.wrap),
    ):
        changed = base.model_copy(update={field: other})
        assert not base.allocates_same_as(changed), (
            f"changing {field} must reallocate the canvas"
        )

    # And every draw-state field must NOT, or picking it from a combo drops the trail.
    for mode in BLEND_MODES:
        changed = base.model_copy(update={"blend": mode})
        assert base.allocates_same_as(changed), (
            f"a blend-only change to {mode} must not reallocate"
        )
        # The pair differs ONLY in blend, so `==` disagreeing is the whole point of the
        # method existing: the two questions have different answers.
        if mode != base.blend:
            assert base != changed


def test_every_blend_mode_has_a_gl_mapping() -> None:
    # Break: delete a mode's entry in `blend.py` and this fails at import.
    for mode in BLEND_MODES:
        src, dst = blend_func_for(mode)
        assert isinstance(src, int) and isinstance(dst, int)
    # Opaque is the mode that must not combine with what is already there; if it maps to
    # the additive pair, the measurement that made I6 real (an opaque overlap must not
    # double) cannot be taken.
    assert blend_func_for("opaque") != blend_func_for("additive")


def test_key_fail_reasons_partition_into_warning_and_silent() -> None:
    """Every reason is in exactly one tier. A member added without a tier decision falls
    into the warning tier by construction, which is the safe direction -- but the count
    assertion below makes the addition visible rather than silent."""
    assert set(WARNING_KEY_FAIL_REASONS) | SILENT_KEY_FAIL_REASONS == set(
        KEY_FAIL_REASONS
    )
    assert not set(WARNING_KEY_FAIL_REASONS) & SILENT_KEY_FAIL_REASONS
    # 102 D1: exactly two exemptions, and a THIRD means the rule shape wants re-deriving
    # rather than another member here.
    assert len(SILENT_KEY_FAIL_REASONS) == 2, (
        "a third silent reason means 'warn unless...' is the wrong rule shape (102 D1)"
    )


def test_instanced_states_partition_into_healthy_and_reportable() -> None:
    assert set(INSTANCED_STATES) >= HEALTHY_INSTANCED_STATES
    reportable = set(INSTANCED_STATES) - HEALTHY_INSTANCED_STATES
    # The five states the research found that nothing in the app could say (101 I1-I5).
    assert reportable == {
        "empty",
        "refused",
        "no_fields",
        "compile_failed",
        "stale_program",
    }


def test_every_outcome_state_describes_itself_distinctly() -> None:
    """A surface renders `describe()`, so two states reading identically would be two
    pictures the user cannot tell apart -- which is the defect class 102 exists to end.

    Presence of each is not distinctness of all: the set comparison is what catches a
    collision, and checking each string is non-empty would pass one.
    """
    lines = {
        state: InstancedOutcome(
            "blur", state, count=0 if state == "empty" else 7
        ).describe()
        for state in INSTANCED_STATES
    }
    assert len(set(lines.values())) == len(INSTANCED_STATES), (
        f"two states describe identically: {lines}"
    )
    # The count must survive into the text: 104 D3 reads a live entity count off this.
    assert "7" in lines["drew"]


def test_reserved_names_partition_covers_every_name() -> None:
    """104 D4: the engine must be able to say which reserved names are documentable.
    `RESERVED_NAMES` is derived from the two halves, so they cannot drift -- break it by
    making the union a literal again and dropping a name from one half."""
    assert USER_FACING_NAMES | ENGINE_INTERNAL_NAMES == RESERVED_NAMES
    assert not USER_FACING_NAMES & ENGINE_INTERNAL_NAMES
    # The two that must never be documented, named so a widening argues with a test.
    assert {"sb_instanced", "a_corner"} == ENGINE_INTERNAL_NAMES
    assert "vs_quad" in USER_FACING_NAMES


def test_the_literals_and_their_tuples_agree() -> None:
    """Each `get_args` tuple is the domain a gate walks. A tuple written by hand beside its
    `Literal` drifts the first time a member is added to one and not the other."""
    from shaderbox.instanced_outcome import InstancedState
    from shaderbox.pass_graph import BlendMode
    from shaderbox.scripting.keys import KeyFailReason

    assert get_args(BlendMode) == BLEND_MODES
    assert get_args(KeyFailReason) == KEY_FAIL_REASONS
    assert get_args(InstancedState) == INSTANCED_STATES
