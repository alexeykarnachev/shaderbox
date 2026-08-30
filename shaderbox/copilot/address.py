from pathlib import Path

# The copilot working-set address scheme: a document is a bare id, ONE PASS of a document is
# "<id>#<pass>", a library file is "lib:<rel-path>", a shipped example is
# "example:<short-handle>". This module is the single round-trip parse/build point, so a new kind
# is one change every tool inherits rather than a new tool per kind.
LIB_PREFIX = "lib:"
EXAMPLE_PREFIX = "example:"

# Examples are addressed by a fixed short handle (never the uuid) — unlike document short-ids
# there is no collision-growth (the shipped set is tiny; the resolver prefix-matches).
_EXAMPLE_HANDLE_LEN = 4

# Document short-id floor: the id length the agent sees. The backend grows ALL ids past it together
# on collision (CopilotBackend._copilot_short_ids).
DOCUMENT_SHORT_ID_LEN = 4


def is_lib_address(address: str) -> bool:
    return address.startswith(LIB_PREFIX)


def strip_lib_prefix(address: str) -> str:
    # Conditional: returns the rel path for a "lib:" address, else the input unchanged — safe to
    # call on an address already known to be a lib target.
    return address[len(LIB_PREFIX) :] if is_lib_address(address) else address


def lib_address(rel: Path | str) -> str:
    rel_str = rel.as_posix() if isinstance(rel, Path) else rel
    return f"{LIB_PREFIX}{rel_str}"


def is_example_address(address: str) -> bool:
    return address.startswith(EXAMPLE_PREFIX)


def strip_example_prefix(address: str) -> str:
    # Conditional, mirroring strip_lib_prefix.
    return address[len(EXAMPLE_PREFIX) :] if is_example_address(address) else address


def example_address(full_id: str) -> str:
    return f"{EXAMPLE_PREFIX}{full_id[:_EXAMPLE_HANDLE_LEN]}"


# A pass of a document: "<document-id>#<pass-name>". A SUFFIX rather than a prefix, so a bare
# document id stays a valid address (it means the document's OUTPUT pass) and every tool that
# takes a document keeps working unchanged.
PASS_SEPARATOR = "#"


def is_pass_address(address: str) -> bool:
    return PASS_SEPARATOR in address and not is_lib_address(address)


def split_pass_address(address: str) -> tuple[str, str]:
    """`(document_address, pass_name)`; the pass name is "" when none is given.

    A lib address is returned untouched — a `#` inside a path is part of the filename.
    """
    if not is_pass_address(address):
        return address, ""
    document, _, pass_name = address.partition(PASS_SEPARATOR)
    return document, pass_name


def pass_address(document_address: str, pass_name: str) -> str:
    return f"{document_address}{PASS_SEPARATOR}{pass_name}"
