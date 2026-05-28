import hashlib
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import moderngl

K = TypeVar("K")


def adjust_size(
    size: tuple[int, int],
    width: int | None = None,
    height: int | None = None,
    aspect: float | None = None,
    max_size: int | None = None,
) -> tuple[int, int]:
    if (width, height, aspect, max_size).count(None) != 3:
        return size

    original_width, original_height = size

    if width is not None:
        new_height = round(width * original_height / original_width)
        return (width, new_height)
    elif height is not None:
        new_width = round(height * original_width / original_height)
        return (new_width, height)
    elif aspect is not None:
        current_aspect = original_width / original_height
        if aspect > current_aspect:
            new_width = round(original_height * aspect)
            return (new_width, original_height)
        else:
            new_height = round(original_width / aspect)
            return (original_width, new_height)
    elif max_size is not None:
        if original_width >= original_height:
            new_width = max_size
            new_height = round(max_size * original_height / original_width)
            return (new_width, new_height)
        else:
            new_height = max_size
            new_width = round(max_size * original_width / original_height)
            return (new_width, new_height)
    else:
        return size


def select_next_value(
    values: Sequence[K],
    current_value: K | None,
    default_value: K,
    step: int = 1,
) -> K:
    if not values:
        return default_value

    idx = (
        0
        if not current_value or current_value not in values
        else values.index(current_value)
    )

    return values[(idx + step) % len(values)]


def get_resolution_str(name: str | None, w: int, h: int) -> str:
    g = math.gcd(w, h)
    w_ratio, h_ratio = w // g, h // g
    label = f"{w}x{h} ({w_ratio}:{h_ratio})"
    if name:
        label += f" - {name}"
    return label


def get_uniform_hash(u: moderngl.Uniform | moderngl.UniformBlock) -> int:
    if isinstance(u, moderngl.Uniform):
        key = f"{u.name}_{u.array_length}_{u.dimension}_{u.gl_type}"  # type: ignore
    else:
        key = f"{u.name}_{u.size}"

    hash = hashlib.md5(key.encode()).digest()
    return int.from_bytes(hash, "big")


def unicode_to_str(char_inds: list[int]) -> str:
    eos_idx = char_inds.index(0) if 0 in char_inds else len(char_inds)
    return "".join(chr(char_inds[i]) for i in range(eos_idx))


def str_to_unicode(text: str, array_length: int) -> list[int]:
    text = text[:array_length]
    char_inds = [ord(c) for c in text]
    char_inds += [0] * (array_length - len(char_inds))
    return char_inds


def try_to_release(value: Any) -> bool:
    if release := getattr(value, "release", None):
        release()
        return True
    return False


def pfd_block(dialog: Any) -> Any:
    while not dialog.ready(20):
        pass
    return dialog.result()


def format_auto_value(value: object) -> str:
    if isinstance(value, float | int):
        return f"{value:.3f}"
    if isinstance(value, Iterable):
        return "[" + ", ".join(f"{v:.3f}" for v in value) + "]"
    return str(value)


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
    # (driver_line, source_path, source_line) — all 1-based. The future include
    # resolver populates this from `#line` directives it injects during flatten;
    # today there's a single source so the mapping is the identity.
    entries: tuple[tuple[int, Path, int], ...]
    root_path: Path

    @classmethod
    def identity(cls, root_path: Path) -> "SourceMap":
        return cls(entries=(), root_path=root_path)

    def resolve(self, driver_line: int) -> tuple[Path, int]:
        # Identity map (no entries): driver line == source line, root file.
        if not self.entries:
            return (self.root_path, driver_line)
        # Walk entries (sorted ascending by driver_line); the matching span is the
        # last entry whose driver_line <= the queried line.
        match = self.entries[0]
        for entry in self.entries:
            if entry[0] <= driver_line:
                match = entry
            else:
                break
        span_driver, span_path, span_source = match
        return (span_path, span_source + (driver_line - span_driver))


# NVIDIA `0(LINE) : error CXXXX: msg`; Mesa/Intel/AMD `ERROR: 0:LINE: msg`.
# Driver lines are 1-based; we resolve via SourceMap (identity today) then shift
# 1-based -> 0-based DocPos.
_NVIDIA_ERROR_RE = re.compile(r"^\s*\d+\((\d+)\)\s*:\s*(.+)$")
_MESA_ERROR_RE = re.compile(r"^\s*(?:ERROR|WARNING):\s*\d+:(\d+):\s*(.+)$")


def parse_shader_errors(raw: str, source_map: SourceMap) -> list[ShaderError]:
    errors: list[ShaderError] = []
    for raw_line in raw.splitlines():
        match = _NVIDIA_ERROR_RE.match(raw_line) or _MESA_ERROR_RE.match(raw_line)
        if match:
            driver_line = int(match.group(1))
            path, source_line = source_map.resolve(driver_line)
            errors.append(ShaderError(path, source_line - 1, match.group(2).strip()))

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
