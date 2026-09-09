"""Buffer formatting (078 D9, D11): one formatter per tab kind, the maintainer's own nvim
settings, shipped with the app.

Python is `ruff format --line-length 88`; GLSL and the lib files are `clang-format` with the
style the nvim config falls back to when no `.clang-format` file is near. Both are binary
wheels the app depends on, called by subprocess over stdin; a formatter that is missing at
runtime raises, it never quietly returns the text unchanged. A leaf: no GL, no App.
"""

import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import clang_format

PYTHON_LINE_LENGTH = 88
GLSL_STYLE = (
    "{BasedOnStyle: LLVM, IndentWidth: 4, TabWidth: 4, UseTab: Never, "
    "AlignAfterOpenBracket: false, BreakAfterOpenBracketFunction: true, "
    "BreakBeforeCloseBracketFunction: true, BinPackArguments: true, "
    "PenaltyBreakAssignment: 1000}"
)

# A line beginning with `.` is only ever a member access in GLSL, and clang-format writes
# `);`, `) {` and `),` for every other close.
_MEMBER_ACCESS = re.compile(r"^\s*\.[A-Za-z_]")
_COMMENT_MARKER = re.compile(r"//|/\*|\*/")


@dataclass(frozen=True)
class FormatResult:
    # `error` is the formatter's first complaint line when the text could not be formatted (a
    # syntax error); `text` is then the input, untouched.
    text: str
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def _ruff_executable() -> Path:
    # The venv's own `ruff` console script, beside the interpreter running the app.
    return Path(sys.executable).parent / "ruff"


def _clang_format_executable() -> Path:
    return Path(clang_format.get_executable("clang-format"))


def _run(command: list[str], text: str) -> FormatResult:
    completed = subprocess.run(
        command, input=text, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        first = (completed.stderr.strip() or "formatter failed").splitlines()[0]
        return FormatResult(text=text, error=first)
    return FormatResult(text=completed.stdout)


def format_python(text: str) -> FormatResult:
    return _run(
        [
            str(_ruff_executable()),
            "format",
            "--line-length",
            str(PYTHON_LINE_LENGTH),
            "-",
        ],
        text,
    )


def _is_anchor(line: str) -> bool:
    # A `)` inside a comment or a preprocessor directive closes prose, not a call: joining
    # onto that line would move the member access there and leave the statement unterminated.
    return (
        line.endswith(")")
        and not _COMMENT_MARKER.search(line)
        and not line.lstrip().startswith("#")
    )


def _attach_member_access(text: str) -> str:
    lines: list[str] = text.split("\n")
    joined: list[str] = []
    for line in lines:
        if joined and _is_anchor(joined[-1]) and _MEMBER_ACCESS.match(line):
            joined[-1] += line.lstrip()
        else:
            joined.append(line)
    return "\n".join(joined)


def format_glsl(text: str) -> FormatResult:
    result = _run(
        [str(_clang_format_executable()), f"--style={GLSL_STYLE}", "-"],
        text,
    )
    if not result.ok:
        return result
    return FormatResult(text=_attach_member_access(result.text))


# Tab kind -> formatter. Every kind the code panel opens has one; `formatter_for` answering
# None for a kind is what the panel treats as "nothing to format here".
_FORMATTERS: dict[str, Callable[[str], FormatResult]] = {
    "shader": format_glsl,
    "lib": format_glsl,
    "script": format_python,
}


def formatter_for(tab_kind: str) -> Callable[[str], FormatResult] | None:
    return _FORMATTERS.get(tab_kind)
