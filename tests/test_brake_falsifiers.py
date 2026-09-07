"""The copilot brakes react to their config value.

A knob with a reader but no falsifier is indistinguishable from a dead one: the suite stays
green whatever it is set to, so a later change to its semantics has nothing to catch it.
These drive each brake's reader rather than merely reading the field back — a test that
builds its input FROM the cap can only prove the reader agrees with itself.
"""

from shaderbox.copilot.gate import GateKind, SourceLock
from shaderbox.copilot.state import ChatState
from shaderbox.copilot.tools.base import GatePolicy
from shaderbox.copilot.tools.registry import build_registry
from tests._caps import minimal_caps


def test_gating_is_a_two_state_decision() -> None:
    # GatePolicy.BULK ("confirm when a list arg exceeds bulk_gate_threshold") was built in
    # 020 and never adopted: no tool ever declared it, so requires_gate's BULK branch was
    # unreachable and its threshold unfalsifiable. Removed in 061 along with the knob. What
    # remains is a two-state decision, and this pins that — a third policy would need a
    # reachable reader and a test that drives it.
    assert {p.name for p in GatePolicy} == {"NONE", "ALWAYS"}

    registry = build_registry(minimal_caps())
    policies = {d.gate_policy for d in registry.definitions()}
    assert policies <= {GatePolicy.NONE, GatePolicy.ALWAYS}
    # Both states are actually in use — a registry where every tool gated (or none did)
    # would pass a subset check while meaning the gate had stopped discriminating.
    assert policies == {GatePolicy.NONE, GatePolicy.ALWAYS}

    gated = [d.name for d in registry.definitions() if registry.requires_gate(d.name)]
    assert "delete_document" in gated
    assert "read_shader" not in gated


# The source lock's domain (083). The lock EXCLUDES two ungated mutators that already block on a
# native file picker as their first act: locking them would put a three-button card in front of a
# file dialog, two prompts for one call. `telegram_connect` mutates but touches no project source.
_LOCK_EXCLUDED: set[str] = {"telegram_connect", "bind_media", "import_document"}


def test_the_source_lock_domain_is_enumerated_not_asserted() -> None:
    # A set EQUALITY, not an implication. `locks_source => mutating` passes while a NEW mutating,
    # ungated tool silently omits the flag — which is the drift the field exists to prevent, and
    # the domain-narrowing class that costs the most here. Computing the domain from the registry
    # means a tool added tomorrow either declares its answer or turns this red.
    registry = build_registry(minimal_caps())
    ungated_mutators = {
        d.name
        for d in registry.definitions()
        if d.mutating and d.gate_policy is GatePolicy.NONE
    }
    locked = {d.name for d in registry.definitions() if d.locks_source}
    assert ungated_mutators - _LOCK_EXCLUDED == locked
    # The exclusions must still BE ungated mutators; a stale name here would silently widen the
    # subtraction and let a real tool slip out of the roster.
    assert ungated_mutators >= _LOCK_EXCLUDED


def test_a_locked_tool_never_carries_a_second_prompt() -> None:
    # Two ways a tool would ask twice for one call: its own ALWAYS policy, or a FILE gate raised
    # inside its handler. An ALWAYS-only check passes while the FILE double-ask ships — that is
    # exactly how bind_media and import_document got caught, so both disjuncts are pinned.
    registry = build_registry(minimal_caps())
    for d in registry.definitions():
        if not d.locks_source:
            continue
        assert d.gate_policy is not GatePolicy.ALWAYS, d.name
        assert d.gate_kind is not GateKind.FILE, d.name
        assert d.mutating, d.name


def test_the_lock_widens_must_confirm_and_leaves_requires_gate_alone() -> None:
    # requires_gate is ALSO the irreversibility predicate `_RunLog.summary_lines` reads to decide
    # whether a ledger line carries its identity verbatim and uncapped. Widening it would push
    # every source edit of a locked session into that branch. must_confirm is the loop's question;
    # requires_gate stays the ledger's.
    registry = build_registry(minimal_caps())
    assert not registry.requires_gate("edit_shader")
    assert not registry.must_confirm("edit_shader")

    registry.source_lock = SourceLock.ASK
    assert registry.must_confirm("edit_shader")
    assert not registry.requires_gate("edit_shader"), (
        "the lock must not reclassify a source edit as irreversible"
    )
    # DENY claims the call just as ASK does -- must_confirm stays a bool and says "this needs
    # permission", never "and here is how we get it" (085 D2a). The loop is what turns a DENY mode
    # into a refusal, and that refusal lives INSIDE the gate block, so a must_confirm that went
    # False here would route the call straight past it and RUN the edit the mode forbids.
    registry.source_lock = SourceLock.DENY
    assert registry.must_confirm("edit_shader")
    assert not registry.requires_gate("edit_shader")
    # A read is never confirmed, locked or not.
    assert not registry.must_confirm("read_shader")
    # An always-gated tool is confirmed either way, and stays irreversible.
    assert registry.must_confirm("delete_document")
    assert registry.requires_gate("delete_document")


def test_a_bare_registry_is_inert_where_a_session_asks() -> None:
    # The two defaults DIFFER, and asserting the difference is the point: a bare registry is an
    # inert catalogue (ALLOW), while a session carries the project's mode, which defaults ASK. A
    # test naming only the registry's own default would assert a constant equals itself and keep
    # passing if the two were ever unified -- at which point the gate-discriminates test above
    # would silently gain twelve entries.
    assert build_registry(minimal_caps()).source_lock is SourceLock.ALLOW
    assert ChatState().source_lock is SourceLock.ASK
    assert build_registry(minimal_caps()).source_lock is not ChatState().source_lock
