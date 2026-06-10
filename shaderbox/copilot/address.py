from pathlib import Path

# The copilot working-set address scheme: a node is a bare id, a library file is
# "lib:<rel-path>", a shipped template is "template:<short-handle>". This module is the
# single round-trip parse/build point for the prefixed kinds.
LIB_PREFIX = "lib:"
TEMPLATE_PREFIX = "template:"

# Templates are addressed by a fixed short handle (never the uuid) — unlike node short-ids
# there is no collision-growth (the shipped set is tiny; the resolver prefix-matches).
_TEMPLATE_HANDLE_LEN = 4


def is_lib_address(address: str) -> bool:
    return address.startswith(LIB_PREFIX)


def strip_lib_prefix(address: str) -> str:
    # Conditional: returns the rel path for a "lib:" address, else the input unchanged — safe to
    # call on an address already known to be a lib target.
    return address[len(LIB_PREFIX) :] if is_lib_address(address) else address


def lib_address(rel: Path | str) -> str:
    rel_str = rel.as_posix() if isinstance(rel, Path) else rel
    return f"{LIB_PREFIX}{rel_str}"


def is_template_address(address: str) -> bool:
    return address.startswith(TEMPLATE_PREFIX)


def strip_template_prefix(address: str) -> str:
    # Conditional, mirroring strip_lib_prefix.
    return address[len(TEMPLATE_PREFIX) :] if is_template_address(address) else address


def template_address(full_id: str) -> str:
    return f"{TEMPLATE_PREFIX}{full_id[:_TEMPLATE_HANDLE_LEN]}"
