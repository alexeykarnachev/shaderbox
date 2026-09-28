"""The seams where one flow's output is another's input, through the real system.

Each test here exercises a path no single flow could build or gate alone: the script
engine decides a state the draw cannot see, the draw decides states the engine cannot see,
and a surface reads both out of one dict. A flow's own tests run against its half with the
other half stubbed, so these are the first place the two meet.
"""

import numpy as np
import pytest

from shaderbox.pass_graph import BLEND_MODES, TargetConfig


def test_no_fields_survives_the_frame_that_follows_it() -> None:
    """The engine decides `no_fields` during the tick; the draw runs afterwards and must
    not erase it.

    This is the joint that had a real defect: `Pass.render` wrote `last_outcome`
    unconditionally, so a population offered to a pass with no `flat in` was marked
    dropped by the engine and then overwritten with `fullscreen` by the very next draw --
    which is the ORIGINAL silence (101 I1) restored by the fix for it. The engine's
    verdict is about a decision already taken and outranks the draw's description of what
    it then did.
    """
    import inspect

    from shaderbox.core import Pass

    # `_upload_instances` only RETURNS an outcome; `render` is what assigns
    # `last_outcome`. A fixture calling the former never reaches the overwrite and reports
    # green whether or not the defect is present, so the assertion is made against
    # `render`'s own source: the write must be conditioned on the engine's verdict, not
    # only on staleness.
    source = inspect.getsource(Pass.render)
    assert "self.last_outcome = outcome" in source, (
        "render no longer assigns last_outcome -- this test is aimed at nothing"
    )
    guard_line = next(
        line for line in source.splitlines() if "self.last_outcome = outcome" in line
    )
    preceding = source[: source.index(guard_line)].splitlines()[-4:]
    assert any("no_fields" in line for line in preceding), (
        "render overwrites last_outcome without checking for the engine's `no_fields` "
        "verdict -- a population dropped by a pass with no `flat in` is marked by the "
        "engine and erased by the very next draw, which is I1's silence restored"
    )


def test_every_blend_mode_round_trips_through_a_target_config() -> None:
    # F3 gates this against graph.json; here it is the type itself, so a mode that cannot
    # survive validation is caught even if no example document uses it yet.
    for mode in BLEND_MODES:
        config = TargetConfig(blend=mode)
        assert TargetConfig.model_validate(config.model_dump()).blend == mode


def test_a_blend_change_alone_never_reallocates() -> None:
    """The two-path feedback wipe (102 D5), asserted where both paths read it.

    `Document.set_pass_target` and `Pass.set_target` both decide reallocation, and both
    asked `==` before this wave. A blend-only change must be invisible to both.
    """
    base = TargetConfig()
    for mode in BLEND_MODES:
        changed = base.model_copy(update={"blend": mode})
        assert base.allocates_same_as(changed)


@pytest.mark.parametrize("dtype", ["f1", "f2", "f4"])
def test_a_population_column_must_be_f4_i4_or_u4(dtype: str) -> None:
    """The dtype contract is CHECKED, never cast (100). The copilot's prompt block states
    this rule, so it is gated here rather than only in the text that teaches it."""
    from shaderbox.instanced import EntityField, validate_population

    fields = (EntityField(name="pos", glsl_type="vec2", line=0),)
    numpy_dtype = {"f1": np.uint8, "f2": np.float16, "f4": np.float32}[dtype]
    column = np.zeros((4, 2), dtype=numpy_dtype)
    _, problem = validate_population(fields, {"pos": column})
    if dtype == "f4":
        assert problem is None
    else:
        assert problem is not None, (
            f"{dtype} column accepted -- the check was cast away"
        )


def test_the_no_fields_verdict_clears_when_the_pass_gains_fields() -> None:
    """A sticky verdict is as wrong as a missing one.

    `no_fields` now outranks the draw, which is what makes it visible at all -- but that
    same precedence would pin it forever once set. The author's fix for it is to declare
    the `flat in`, and a successful compile is where the engine learns that happened. A
    FAILED compile already resets the outcome (`_fail_compile`); a successful one must
    too, or the strip keeps reporting a defect the author has already corrected.
    """
    import inspect

    from shaderbox.core import Pass

    source = inspect.getsource(Pass.compile)
    assert "last_outcome" in source, (
        "a successful compile does not reset last_outcome -- the `no_fields` verdict "
        "outranks the draw and so survives the edit that fixes it, reporting a defect "
        "that no longer exists"
    )


def test_the_warn_callback_is_wired_from_the_app_to_notifications() -> None:
    """102 D-F says a non-landing key warns "in the logs AND in the notifications", and
    D3a routes it through an injected callback because the engine imports no imgui.

    The flow that built the engine half could not wire the app half -- `app.py` was not its
    file -- so the seam existed end to end and nothing in the shipped app passed a real
    callback. Every test passed, and the notification half of the headline behaviour
    reached nobody. This asserts the wiring, which is the part no engine-side test can see.
    """
    import inspect

    from shaderbox.app import App

    construction = inspect.getsource(App.__init__)
    assert "on_key_warning=" in construction, (
        "ProjectSession is built without a warn callback -- the engine fires into the "
        "default no-op and 102's notifications never happen"
    )
    handler = inspect.getsource(App._on_script_key_warning)
    assert "notifications.push" in handler, (
        "the warn handler does not reach the notification stack"
    )
