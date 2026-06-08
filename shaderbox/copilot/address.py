from pathlib import Path

# The copilot working-set address scheme: a node is a bare id, a library file is "lib:<rel-path>".
# This module is the single round-trip parse/build point for the lib: kind.
LIB_PREFIX = "lib:"


def is_lib_address(address: str) -> bool:
    return address.startswith(LIB_PREFIX)


def strip_lib_prefix(address: str) -> str:
    # Conditional: returns the rel path for a "lib:" address, else the input unchanged — safe to
    # call on an address already known to be a lib target.
    return address[len(LIB_PREFIX) :] if is_lib_address(address) else address


def lib_address(rel: Path | str) -> str:
    rel_str = rel.as_posix() if isinstance(rel, Path) else rel
    return f"{LIB_PREFIX}{rel_str}"
