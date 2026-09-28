"""The (pass, uniform) key the script engine and the persisted stop set share (069 D3).

It lives in the engine's own package rather than beside `UIDocumentState`, because the engine takes
a `frozenset[StoppedKey]` per tick and `ui_models` imports the concrete `Document` — the engine may
not. The persisted field imports it from here, so the on-disk shape and the engine's key are one
type rather than two that must be kept in step.
"""

from typing import Any, Literal, get_args

from pydantic import BaseModel

# A population the engine REFUSED, as distinct from one the script did not send. An
# absent population legitimately means "draw fullscreen this frame", so a rejected one
# needs its own value or a dtype slip is answered by painting the entity shader over the
# whole canvas. Compared by IDENTITY, which is why it is a module-level singleton.
REFUSED_POPULATION: dict[str, Any] = {}


class StoppedKey(BaseModel, frozen=True):
    # One (pass, uniform) the user has STOPPED. A pair, not a name: the same uniform name on two
    # passes is two independently stoppable rows (069 D3). `frozen=True` in the CLASS ARGS, not in a
    # `model_config`, because the engine holds these in a set and only this form makes the generated
    # `__hash__` visible to the type checker. `pass_name`, not `pass`, on disk as in code — `pass` is
    # a Python keyword and cannot be an attribute.
    pass_name: str
    name: str


# Why a script key did not reach a uniform. ONE path in the engine decides this and ONE
# rule acts on it (102 D1): a key naming a uniform no pass declares, a key naming a pass
# that does not exist, `@instances` reaching a compiled pass with no `flat in`, a
# sampler/block key and any sibling not yet enumerated are the SAME case, and the code must
# not be able to tell them apart at the point it decides what to do.
#
# It is a `Literal` with a `get_args` tuple rather than an `Enum` so a gate can walk its
# domain the way `TARGET_DTYPES` is walked (`pass_graph.py:44-45`) -- which is the whole
# reason it exists as a type instead of as seventeen branches in `_tick_script`. A gate
# enumerating from a hand-written list narrows its own domain silently; one enumerating
# from here cannot.
#
# The two NON-warning members are named here rather than left implicit, because a rule
# stated as "warn unless..." has nowhere to put its exceptions and they end up as comments:
#   - `held_uncompiled`: the pass has never attempted a compile. On frame one of every
#     document every pass is in this state, so warning here fires on every open.
#   - `engine_owned`: the engine owns the slot (`u_time`...) and a script cannot be
#     expected to avoid naming it.
# Revisit if a THIRD non-warning case appears -- at which point the rule shape is wrong and
# wants re-deriving rather than a third exemption (102 D1).
KeyFailReason = Literal[
    "no_such_pass",
    "no_such_uniform",
    "not_scriptable",
    "instances_without_fields",
    "unknown_engine_key",
    "engine_key_misplaced",
    "bad_population",
    "held_uncompiled",
    "engine_owned",
]
KEY_FAIL_REASONS: tuple[KeyFailReason, ...] = get_args(KeyFailReason)

# The two members that are NOT warnings, and are exempt because their correct behaviour is
# provably not a warning rather than because they are special. Derived-from, not
# parallel-to, `KEY_FAIL_REASONS`: a gate walks the full domain and asserts every member
# outside this set warns, so adding a member without deciding its tier fails the gate.
SILENT_KEY_FAIL_REASONS: frozenset[KeyFailReason] = frozenset(
    {"held_uncompiled", "engine_owned"}
)
WARNING_KEY_FAIL_REASONS: tuple[KeyFailReason, ...] = tuple(
    reason for reason in KEY_FAIL_REASONS if reason not in SILENT_KEY_FAIL_REASONS
)
