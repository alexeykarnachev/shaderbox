"""A shader buffer read as text: what it declares (uniforms, functions, constants, defines)
and every other word in it. The buffer, never the compiled program, is what the editor knows
about -- a declaration the body does not read is still a declaration."""

import re
from dataclasses import dataclass

_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
_UNIFORM = re.compile(r"\buniform\s+(\w+)\s+(\w+)\s*(\[[^\]]*\])?\s*;")
_FUNCTION = re.compile(r"^[ \t]*(\w+)\s+(\w+)\s*\(([^)]*)\)\s*\{", re.MULTILINE)
_CONST = re.compile(r"\bconst\s+(\w+)\s+(\w+)\s*(\[[^\]]*\])?\s*=")
_DEFINE = re.compile(r"^[ \t]*#\s*define\s+(\w+)(\([^)]*\))?", re.MULTILINE)
_WORD = re.compile(r"\b[A-Za-z_]\w*\b")


@dataclass(frozen=True)
class UniformDeclaration:
    name: str
    glsl_type: str
    # 0-based line of the declaration.
    line: int
    array: str = ""

    @property
    def declaration(self) -> str:
        return f"uniform {self.glsl_type} {self.name}{self.array};"


@dataclass(frozen=True)
class BufferDeclaration:
    name: str
    # The declaration head: `vec3 palette(float t)`, `const float PI`, `#define STEPS`.
    signature: str
    line: int


def _strip_comments(text: str) -> str:
    # Comments become spaces of the same length so every line number and offset survives.
    return _COMMENT.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset)


def uniform_declarations(text: str) -> tuple[UniformDeclaration, ...]:
    """Every `uniform <type> <name>;` in the buffer, comments excluded, in source order."""
    code = _strip_comments(text)
    return tuple(
        UniformDeclaration(
            m.group(2), m.group(1), _line_of(code, m.start()), m.group(3) or ""
        )
        for m in _UNIFORM.finditer(code)
    )


def buffer_declarations(text: str) -> tuple[BufferDeclaration, ...]:
    """Functions, constants and defines the buffer itself declares."""
    code = _strip_comments(text)
    found: list[BufferDeclaration] = []
    for m in _FUNCTION.finditer(code):
        params = " ".join(m.group(3).split())
        found.append(
            BufferDeclaration(
                m.group(2),
                f"{m.group(1)} {m.group(2)}({params})",
                _line_of(code, m.start(2)),
            )
        )
    for m in _CONST.finditer(code):
        found.append(
            BufferDeclaration(
                m.group(2),
                f"const {m.group(1)} {m.group(2)}{m.group(3) or ''}",
                _line_of(code, m.start(2)),
            )
        )
    for m in _DEFINE.finditer(code):
        found.append(
            BufferDeclaration(
                m.group(1),
                f"#define {m.group(1)}{m.group(2) or ''}",
                _line_of(code, m.start(1)),
            )
        )
    found.sort(key=lambda d: d.line)
    return tuple(found)


def buffer_words(text: str) -> frozenset[str]:
    """Every identifier-shaped word in the buffer, comments excluded."""
    return frozenset(_WORD.findall(_strip_comments(text)))
