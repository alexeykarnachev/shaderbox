import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ShaderError:
    # `line` is 0-based editor line; -1 = unparsable driver string (fallback, not clickable).
    path: Path
    line: int
    message: str


@dataclass(frozen=True)
class SourceMap:
    # Maps a driver-emitted `(file_id, line)` back to source path + 1-based line.
    # Drivers honour the `#line N <id>` directives emitted during flatten, so the
    # reported line is already the source line; only the path needs recovery. Root = file-id 0.
    file_id_to_path: dict[int, Path]

    @property
    def root_path(self) -> Path:
        return self.file_id_to_path[0]

    @classmethod
    def identity(cls, root_path: Path) -> "SourceMap":
        return cls(file_id_to_path={0: root_path})

    def resolve(self, file_id: int, line: int) -> tuple[Path, int]:
        # Unknown file_id falls back to root rather than dropping the error.
        path = self.file_id_to_path.get(file_id, self.file_id_to_path[0])
        return (path, line)


# NVIDIA `0(LINE) : error CXXXX: msg`; Mesa/Intel/AMD `ERROR: 0:LINE: msg`;
# Mesa GLSL-IR (V3D et al) `FILE:LINE(COL): error: msg`.
# Leading number = file-id from our `#line` directives (0 = root). Driver lines are
# 1-based; resolved via SourceMap then shifted to 0-based.
_NVIDIA_ERROR_RE = re.compile(r"^\s*(\d+)\((\d+)\)\s*:\s*(.+)$")
_MESA_ERROR_RE = re.compile(r"^\s*(?:ERROR|WARNING):\s*(\d+):(\d+):\s*(.+)$")
_MESA_IR_ERROR_RE = re.compile(r"^\s*(\d+):(\d+)\(\d+\):\s*(.+)$")


def parse_shader_errors(raw: str, source_map: SourceMap) -> list[ShaderError]:
    errors: list[ShaderError] = []
    for raw_line in raw.splitlines():
        match = (
            _NVIDIA_ERROR_RE.match(raw_line)
            or _MESA_ERROR_RE.match(raw_line)
            or _MESA_IR_ERROR_RE.match(raw_line)
        )
        if match:
            file_id = int(match.group(1))
            driver_line = int(match.group(2))
            path, source_line = source_map.resolve(file_id, driver_line)
            errors.append(ShaderError(path, source_line - 1, match.group(3).strip()))

    if not errors:
        return [ShaderError(source_map.root_path, -1, raw.strip())]
    return errors


def next_error_line(errors: list[ShaderError], after_line: int) -> int | None:
    # First markable line (line >= 0) strictly after `after_line`, wrapping to the first.
    lines = sorted({e.line for e in errors if e.line >= 0})
    if not lines:
        return None
    for line in lines:
        if line > after_line:
            return line
    return lines[0]


def find_uniform_declaration_line(source: str, name: str) -> int | None:
    # Name must be the declared identifier (immediately before `[`/`=`/`;`/`{`), not
    # just mentioned on a `uniform` line. `{` covers UBO blocks where the name is the
    # block-type; an optional `layout(...)` may precede `uniform`.
    pattern = re.compile(
        rf"^\s*(?:layout\s*\([^)]*\)\s*)?uniform\b[^=;{{]*\b{re.escape(name)}\s*(?:\[|=|;|\{{)"
    )
    for i, line in enumerate(source.splitlines()):
        if pattern.match(line.split("//")[0]):
            return i
    return None
