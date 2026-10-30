"""Fail when a comment or docstring carries development-process references.

Comments describe the code as it is. Planning numbers, review history, document paths
and process markers belong in commit messages, so this reads every comment and every
docstring (string literals elsewhere are product text and are not read) and names each
line that matches one of the patterns below.

Usage: check_comments.py [paths...]   (default: shaderbox tests scripts dogfood)
"""

import ast
import io
import re
import sys
import tokenize
from pathlib import Path

_DEFAULT_ROOTS = ("shaderbox", "tests", "scripts", "dogfood")
_SKIP = ("shaderbox/resources/editor/abi_probe.py", "scripts/dogfood/runs/")

_PATTERNS: dict[str, re.Pattern[str]] = {
    "feature number": re.compile(
        r"\b[Ff]eatures? \d{3}\b|\(\d{3}[a-z]?[,)]|(?<![\w.])\d{3}[a-z]?'s\b"
        r"|\b\d{3}[a-z]? (?:D|W|R)\d|\b(?:0\d\d|1[0-4]\d) (?:wave|round|review|walk|spec)\b"
    ),
    "planning number": re.compile(
        r"(?<![\w.,#-])0\d\d(?![\w.%])|\b1[0-4]\d [A-Z]{1,2}-?[A-Z0-9]"
        r"|\b1[0-4]\d (?:item|shape|seam|stage)\b|\u00a7"
    ),
    "commit hash": re.compile(
        r"\b(?:editor|upstream|commit|library|at|graph_canvas)\s+`?[0-9a-f]{7,12}\b"
    ),
    "decision or wave id": re.compile(r"\b(?:D\d{1,2}[a-z]?|W-?[A-H]\d?|W\d-\d)\b"),
    "process document": re.compile(
        r"ai_docs|CLAUDE\.md|conventions\.md|roadmap|todo\.md|dev_flow|SKILL\.md"
    ),
    "process role": re.compile(
        r"\b[Mm]aintainer|\b[Cc]laude\b|\bskill\b"
        r"|(?<![\w/])/(?:imgui-ui|dogfood|sanitize|ship|shader-lab)\b"
    ),
    "mutation note": re.compile(r"\b[Ff]alsifier|\bBreak:|\bBroken by\b"),
    "planning id": re.compile(
        r"\b[Gg]ates? \d{1,2}\b|(?<![\w&])#\d{1,3}\b|\bslice \d\b|\b[Ff]inding #?\d"
    ),
    "review history": re.compile(
        r"\b[Rr]eview round|\b[Rr]ound \d\b|post-implementation review|\bsweep wave\b"
        r"|\bre-vendor|\b(?:[a-z]+th|first|second|third|maintainer's) walk\b"
        r"|\b[Dd]ecisions? \d|\bused to be\b|\bit used to\b|\bbefore the fix\b"
        r"|\ban earlier version\b|\b[Hh]is (?:call|finding|to)\b"
    ),
}


# Only for the shipped package: the repo's own dev tools legitimately carry these names.
_SHIPPED_ONLY: dict[str, re.Pattern[str]] = {
    "dev tooling": re.compile(r"(?i)\bdogfood"),
}


def _docstring_spans(tree: ast.AST) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                end = body[0].end_lineno or body[0].lineno
                spans.append((body[0].lineno, end))
    return spans


def _prose_lines(path: Path) -> list[tuple[int, str]]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    out: list[tuple[int, str]] = []
    if path.suffix != ".py":
        for number, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith(("#", "//", "/*", "*")):
                out.append((number, line))
        return out
    for start, end in _docstring_spans(ast.parse(text)):
        out.extend((n, lines[n - 1]) for n in range(start, end + 1))
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == tokenize.COMMENT:
            out.append((token.start[0], token.string))
    return out


def check(paths: list[Path]) -> list[str]:
    findings: list[str] = []
    for path in paths:
        posix = path.as_posix()
        if any(skip in posix for skip in _SKIP):
            continue
        patterns = _PATTERNS
        if posix.startswith("shaderbox/") or "/shaderbox/shaderbox/" in posix:
            patterns = {**_PATTERNS, **_SHIPPED_ONLY}
        for number, line in _prose_lines(path):
            for name, pattern in patterns.items():
                if pattern.search(line):
                    findings.append(f"{posix}:{number}: {name}: {line.strip()}")
                    break
    return findings


def _files(roots: list[str]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        base = Path(root)
        if base.is_file():
            files.append(base)
            continue
        for suffix in ("*.py", "*.glsl", "*.sh"):
            files.extend(sorted(base.rglob(suffix)))
    return files


def main() -> int:
    findings = check(_files(sys.argv[1:] or list(_DEFAULT_ROOTS)))
    for finding in findings:
        print(finding)
    if findings:
        print(f"{len(findings)} comment line(s) carry process references")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
