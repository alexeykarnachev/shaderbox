"""The load bands: which state hue a measurement gets for its share of the frame budget
(feature 089 D7), and the THROTTLE bands beside them (090 D9c).

GL-free -- both are pure functions of a ratio, and the plan they feed is pure too.
The bands are pinned at their EDGES, because a band is exactly its two thresholds and a
test in the middle of one passes whichever way a threshold moved.

The two functions have deliberately different knees, which is the whole reason the second
exists: `load_color` is red at 1.0, and 1.0 is where a converged throttled document sits.
"""

from shaderbox.theme import (
    _ACCENTS,
    _GROUP_TINT_EXCLUSIONS,
    COLOR,
    group_tint,
    load_color,
    throttle_color,
)


def test_the_bands_meet_at_their_thresholds() -> None:
    # Falsifier: move either literal in `theme.py` and one of the four flips.
    assert load_color(0.49) is COLOR.STATE_OK
    assert load_color(0.5) is COLOR.STATE_WARN
    assert load_color(0.99) is COLOR.STATE_WARN
    assert load_color(1.0) is COLOR.STATE_ERROR


def test_throttle_color_reads_a_converged_document_as_healthy() -> None:
    # `throttle_color` exists because `load_color`'s knees are WRONG for a throttled document's
    # row (090 D9c, closing correctness F10): `load_color` turns red at ratio 1.0, which is
    # exactly where a converged document sits -- at its allowance, which is the state the
    # throttle aims for. Falsifier: `return load_color(share_ratio)` and the first assertion
    # goes red, reintroducing F10 verbatim.
    assert throttle_color(1.0, False) is COLOR.STATE_OK
    assert throttle_color(1.2, False) is COLOR.STATE_WARN
    assert throttle_color(1.6, False) is COLOR.STATE_ERROR


def test_a_missed_frame_reddens_a_document_inside_its_allowance() -> None:
    # The clause the spec singles out: a document within its share while the UI frame still
    # misses its target is what the reader must see. It is also the half a later edit is most
    # likely to drop, since every share-only case passes without it. Falsifier: drop
    # `frame_over_budget` from the condition and this returns STATE_OK.
    assert throttle_color(0.2, True) is COLOR.STATE_ERROR


def test_group_tints_are_stable_and_collide_with_nothing() -> None:
    # 091 D8. (a) pinned by VALUE at crc32 indices: `hash()` is salted per process, so under
    # it the pin is red on essentially every run rather than flaky green.
    assert group_tint("bloom") is COLOR.GROUP_TINTS[3]
    assert group_tint("radiance") is COLOR.GROUP_TINTS[2]
    assert group_tint("fx") is COLOR.GROUP_TINTS[0]
    # (b) disjoint from every outline and chip these tiles carry, and no duplicate. Falsifier:
    # put `purple_b` (COLOR.SELECT) in the tuple -- a check over accent primaries and state
    # hues alone lets it through.
    excluded = {primary for primary, _active, _alpha in _ACCENTS.values()} | {
        COLOR.STATE_OK,
        COLOR.STATE_WARN,
        COLOR.STATE_ERROR,
        COLOR.STATE_INFO,
        COLOR.SELECT,
        COLOR.TAG,
        COLOR.FAVS,
        # 092: a wire is drawn against a box border and beside a STATE_ERROR wire.
        # Falsifier: `GRAPH_EDGE = COLOR.GROUP_TINTS[2]` imported clean before it joined.
        COLOR.GRAPH_EDGE,
        # 093: the hover hue is drawn on the same wire and the same border.
        COLOR.GRAPH_HOVER,
    }
    assert not set(COLOR.GROUP_TINTS) & excluded
    assert len(set(COLOR.GROUP_TINTS)) == len(COLOR.GROUP_TINTS)
    assert COLOR.GRAPH_EDGE != COLOR.STATE_ERROR
    # (c) 093 S1: the hover hue meets three other cues on one wire and, as a node's halo,
    # sits beside the output node's accent border. Break to try: the design record's own
    # `blue_b` -- it IS the blue accent's primary, so the accent clause goes red while the
    # three `!=` clauses all stay green, which is why they alone were not enough.
    assert COLOR.GRAPH_HOVER in _GROUP_TINT_EXCLUSIONS
    assert COLOR.GRAPH_HOVER not in {
        primary for primary, _active, _alpha in _ACCENTS.values()
    }
    assert COLOR.GRAPH_HOVER not in {
        COLOR.SELECT,
        COLOR.STATE_ERROR,
        COLOR.GRAPH_EDGE,
    }
