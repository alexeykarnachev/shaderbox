import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ShaderError:
    # `path` identifies which source file the error belongs to (the root today; an
    # included file once 015 lands). `line` is 0-based editor line; -1 when the raw
    # driver string didn't match either error regex (fallback row, not clickable).
    path: Path
    line: int
    message: str


@dataclass(frozen=True)
class SourceMap:
    # Maps a driver-emitted `(file_id, line)` pair back to a source file + 1-based
    # line in that file. The include resolver populates `file_id_to_path` and emits
    # `#line N <id>` directives during flatten — both NVIDIA `0(LINE) : msg` and Mesa
    # `ERROR: 0:LINE: msg` honour `#line`, so the line they report is already the
    # source line; we just need the file-id table to recover the path. The root
    # file is always file-id 0.
    file_id_to_path: dict[int, Path]

    @property
    def root_path(self) -> Path:
        return self.file_id_to_path[0]

    @classmethod
    def identity(cls, root_path: Path) -> "SourceMap":
        return cls(file_id_to_path={0: root_path})

    def resolve(self, file_id: int, line: int) -> tuple[Path, int]:
        # Unknown file_id (driver emitted an id we never declared) → fall back to
        # root file; better than dropping the error entirely.
        path = self.file_id_to_path.get(file_id, self.file_id_to_path[0])
        return (path, line)


# NVIDIA `0(LINE) : error CXXXX: msg`; Mesa/Intel/AMD `ERROR: 0:LINE: msg`.
# The leading number is the source-string file-id set by our `#line N M` directives
# (0 for the root). Driver lines are 1-based; we resolve via SourceMap then shift
# 1-based -> 0-based DocPos.
_NVIDIA_ERROR_RE = re.compile(r"^\s*(\d+)\((\d+)\)\s*:\s*(.+)$")
_MESA_ERROR_RE = re.compile(r"^\s*(?:ERROR|WARNING):\s*(\d+):(\d+):\s*(.+)$")


def parse_shader_errors(raw: str, source_map: SourceMap) -> list[ShaderError]:
    errors: list[ShaderError] = []
    for raw_line in raw.splitlines():
        match = _NVIDIA_ERROR_RE.match(raw_line) or _MESA_ERROR_RE.match(raw_line)
        if match:
            file_id = int(match.group(1))
            driver_line = int(match.group(2))
            path, source_line = source_map.resolve(file_id, driver_line)
            errors.append(ShaderError(path, source_line - 1, match.group(3).strip()))

    if not errors:
        return [ShaderError(source_map.root_path, -1, raw.strip())]
    return errors


def next_error_line(errors: list[ShaderError], after_line: int) -> int | None:
    # Markable lines (line >= 0), sorted; the first strictly after `after_line`,
    # wrapping to the first. None when there are no markable errors.
    lines = sorted({e.line for e in errors if e.line >= 0})
    if not lines:
        return None
    for line in lines:
        if line > after_line:
            return line
    return lines[0]


def find_uniform_declaration_line(source: str, name: str) -> int | None:
    # The name must be the declared identifier (immediately before `[`/`=`/`;`/`{`),
    # not merely mentioned on a `uniform` line (a comment / another's initializer).
    # `{` covers UBO blocks (`layout(std140) uniform u_params { ... }`) where the
    # name is the block-type, terminated by `{`, and the line may carry an optional
    # `layout(...)` prefix before `uniform`.
    pattern = re.compile(
        rf"^\s*(?:layout\s*\([^)]*\)\s*)?uniform\b[^=;{{]*\b{re.escape(name)}\s*(?:\[|=|;|\{{)"
    )
    for i, line in enumerate(source.splitlines()):
        if pattern.match(line.split("//")[0]):
            return i
    return None
