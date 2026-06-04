"""Pure GLSL text machinery for the shader library — regexes + brace-matching.

No index types, no GL, no imgui. `index._extract_functions` (the build path) and
`resolver._collect_used_lib_names` (the resolve path) both lean on these helpers,
so they live in this leaf to keep the index/resolver split acyclic.
"""

import re

# Top-level function signature: `<type> <name>(<args>)` followed by `{`. The type
# permits a trailing `[]` (rare in lib code but cheap to allow). Captures: 1=type,
# 2=name, 3=args. Anchored to start-of-line so we don't pick up function calls
# inside other functions.
FN_SIG_RE = re.compile(
    r"^\s*(\w+(?:\s*\[\s*\d*\s*\])?)\s+(\w+)\s*\(([^)]*)\)\s*\{",
    re.MULTILINE,
)

# Identifier references inside a function body (used to build the call graph).
IDENT_RE = re.compile(r"\b([A-Za-z_]\w*)\b")

# A `SB_` identifier — what the user types in their shader to call a lib function.
SB_IDENT_RE = re.compile(r"\bSB_\w+\b")

# A function DEFINITION in the user's text (so we can suppress the lib version when the
# user shadows it). Must match the full signature `<type> SB_foo(args) {` — the trailing
# `)` + `{` body is what distinguishes a definition from a CALL that happens to follow a
# keyword (`return SB_foo(...)` reads `return` as a type otherwise, and the lib function is
# wrongly treated as shadowed and never spliced). Args are non-capturing so findall yields
# just the name.
USER_FN_DEF_RE = re.compile(
    r"\b\w+(?:\s*\[\s*\d*\s*\])?\s+(SB_\w+)\s*\((?:[^)]*)\)\s*\{", re.MULTILINE
)


def strip_comments(text: str) -> str:
    # GLSL has `//` line comments and `/* ... */` block comments. No strings, no
    # nested block comments. Block-strip is non-greedy + multi-line.
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return text


def split_root_header(text: str) -> tuple[int, list[str], list[str]]:
    # Return (header_end_index, header_lines, body_lines).
    # Header = any prefix of: blank lines, line-comments (kept verbatim), and
    # `#version` / `#extension` / `#pragma` directives. First line that's not one
    # of those starts the body.
    lines = text.splitlines()
    header_end = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("//"):
            continue
        if stripped.startswith(("#version", "#extension", "#pragma")):
            header_end = i + 1
            continue
        # First non-header line.
        header_end = i
        break
    else:
        # All lines were header.
        header_end = len(lines)
    return header_end, lines[:header_end], lines[header_end:]


def find_body_end(lines: list[str], start: int) -> int | None:
    # Starting at `start` (line with the function signature + opening `{`), walk
    # forward counting braces; return the 0-based line index containing the
    # matching `}`.
    depth = 0
    in_string = False  # GLSL has no strings, but keep the toggle harmless
    for i in range(start, len(lines)):
        for ch in lines[i]:
            if ch == "{" and not in_string:
                depth += 1
            elif ch == "}" and not in_string:
                depth -= 1
                if depth == 0:
                    return i
    return None


def extract_doc(lines: list[str], sig_line: int) -> str:
    # Walk backwards from `sig_line - 1` collecting contiguous `///` lines.
    doc_lines: list[str] = []
    j = sig_line - 1
    while j >= 0:
        stripped = lines[j].lstrip()
        if stripped.startswith("///"):
            doc_lines.append(stripped[3:].strip())
            j -= 1
        else:
            break
    doc_lines.reverse()
    return "\n".join(doc_lines).strip()
