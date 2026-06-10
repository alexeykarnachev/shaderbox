"""Pure GLSL text machinery for the shader library — regexes + brace-matching.

No index types, no GL, no imgui — a leaf so the index/resolver split stays acyclic.
"""

import re

# Top-level function signature `<type> <name>(<args>) {`. Captures 1=type, 2=name,
# 3=args. Anchored to start-of-line so nested calls aren't matched.
FN_SIG_RE = re.compile(
    r"^\s*(\w+(?:\s*\[\s*\d*\s*\])?)\s+(\w+)\s*\(([^)]*)\)\s*\{",
    re.MULTILINE,
)

# Identifier references inside a function body (used to build the call graph).
IDENT_RE = re.compile(r"\b([A-Za-z_]\w*)\b")

# A `SB_` identifier — what the user types in their shader to call a lib function.
SB_IDENT_RE = re.compile(r"\bSB_\w+\b")

# A function DEFINITION in the user's text — matches it to suppress the shadowed lib
# version. Must require the `) {` body: without it a CALL after a keyword
# (`return SB_foo(...)`) reads as a definition and wrongly suppresses the lib function.
# Args are non-capturing so findall yields just the name.
USER_FN_DEF_RE = re.compile(
    r"\b\w+(?:\s*\[\s*\d*\s*\])?\s+(SB_\w+)\s*\((?:[^)]*)\)\s*\{", re.MULTILINE
)


def strip_comments(text: str) -> str:
    # GLSL has no strings and no nested block comments, so a non-greedy block-strip
    # then a line-strip is safe.
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return text


def strip_comments_keep_lines(text: str) -> str:
    # Like strip_comments but PRESERVES line numbering: a multi-line block comment
    # collapses to an equal number of newlines. Use wherever line numbers are
    # reported off the stripped text.
    text = re.sub(
        r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.DOTALL
    )
    return re.sub(r"//[^\n]*", "", text)


def split_root_header(text: str) -> tuple[int, list[str], list[str]]:
    # Header = leading prefix of blank lines, `//` comments, and
    # `#version`/`#extension`/`#pragma` directives. First other line starts the body.
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
        header_end = i
        break
    else:
        header_end = len(lines)
    return header_end, lines[:header_end], lines[header_end:]


def find_body_end(lines: list[str], start: int) -> int | None:
    # `start` holds the signature's opening `{`; count braces forward, return the
    # line index of the matching `}`.
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
