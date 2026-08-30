"""Parsing of `// step` riders on sampler declarations, and the step model.

A node declares an extra render step by riding a comment on the sampler that reads it:

    uniform sampler2D u_blur;   // step, scale: 0.5, f2, linear

The rider configures the declaration it sits on, so the two cannot desync. The step's body
is `void step_blur(out vec4 o)` in the same file.

GL-free by design: the parser runs on source text, so it is unit-testable without a context
and importable from anywhere without a cycle.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from shaderbox.shader_errors import ShaderError

STEP_MARKER = "step"
STEP_FN_PREFIX = "step_"
STEP_OUT_NAME = "sb_step_out"
USER_MAIN_ALIAS = "sb_user_main"

_DTYPES = ("f1", "f2", "f4")
_FILTERS = ("linear", "nearest")
_WRAPS = ("clamp", "repeat")

# `uniform sampler2D u_name;  // rider`
_SAMPLER_RE = re.compile(
    r"^\s*uniform\s+sampler2D\s+(?P<name>\w+)\s*;\s*(?://\s*(?P<rider>.*))?$"
)
_STEP_FN_RE = re.compile(r"^\s*void\s+(?P<name>step_\w+)\s*\(", re.MULTILINE)


# f2, not f1: 063 measured f1 saturating at 255 on the FIRST accumulate pass where f2
# reached exactly 7.0, so the safe value is the default and f1 is the opt-in. clamp
# inverts moderngl's repeat_x/y=True, which is wrong for a feedback border.
DEFAULT_DTYPE = "f2"
DEFAULT_FILTER_LINEAR = True
DEFAULT_WRAP = False


@dataclass(frozen=True)
class StepSpec:
    """One declared step: the sampler that reads it plus its target's format."""

    name: str  # the bare name, e.g. "blur" for `u_blur` / `step_blur`
    sampler: str  # the uniform that reads this step's output
    scale: float = 1.0
    size: tuple[int, int] | None = None  # absolute, wins over scale
    dtype: str = DEFAULT_DTYPE
    filter_linear: bool = DEFAULT_FILTER_LINEAR
    wrap: bool = DEFAULT_WRAP
    persist: bool = False

    @property
    def fn_name(self) -> str:
        return f"{STEP_FN_PREFIX}{self.name}"

    def target_size(self, canvas_size: tuple[int, int]) -> tuple[int, int]:
        if self.size is not None:
            return self.size
        return (
            max(1, round(canvas_size[0] * self.scale)),
            max(1, round(canvas_size[1] * self.scale)),
        )


@dataclass
class StepParseResult:
    steps: list[StepSpec] = field(default_factory=list)
    errors: list[ShaderError] = field(default_factory=list)


def _is_transposition(a: str, b: str) -> bool:
    if len(a) != len(b):
        return False
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    if len(diff) != 2:
        return False
    i, j = diff
    return j == i + 1 and a[i] == b[j] and a[j] == b[i]


def _looks_like_marker(token: str) -> bool:
    """A typo of `step`, near enough that silence would be the wrong answer.

    Transpositions count: `setp` is two substitutions away but is the likeliest typo
    of all, and it is exactly the case a distance-1 check misses.
    """
    return _levenshtein_le_1(token, STEP_MARKER) or _is_transposition(
        token, STEP_MARKER
    )


def _levenshtein_le_1(a: str, b: str) -> bool:
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        return sum(x != y for x, y in zip(a, b)) == 1
    short, long = (a, b) if la < lb else (b, a)
    for i in range(len(long)):
        if short == long[:i] + long[i + 1 :]:
            return True
    return False


def _sampler_to_step_name(sampler: str) -> str:
    return sampler[2:] if sampler.startswith("u_") else sampler


def parse_steps(source: str, path: Path) -> StepParseResult:
    """Read every `// step` rider out of `source`.

    A malformed rider is an error, never a silently-ignored comment: a rider that merely
    looks like a step would otherwise leave an ordinary sampler bound to the default
    image, so the user gets a picture and it is the wrong one.
    """
    result = StepParseResult()
    declared_fns = {m.group("name") for m in _STEP_FN_RE.finditer(source)}
    seen: dict[str, int] = {}
    # A near-miss marker already reports itself; its body would otherwise be reported a
    # second time as an orphan, which is one typo wearing two errors.
    near_missed: set[str] = set()

    for line_idx, line in enumerate(source.splitlines()):
        match = _SAMPLER_RE.match(line)
        if match is None:
            continue
        rider = (match.group("rider") or "").strip()
        if not rider:
            continue

        tokens = [t.strip() for t in rider.split(",") if t.strip()]
        if not tokens:
            continue
        head = tokens[0].lower()

        if head != STEP_MARKER:
            # Only complain about a near-miss; an ordinary comment is left alone.
            if _looks_like_marker(head):
                near_missed.add(_sampler_to_step_name(match.group("name")))
                result.errors.append(
                    ShaderError(
                        path,
                        line_idx,
                        f"did you mean '// {STEP_MARKER}'? "
                        f"'{tokens[0]}' is not a step marker, so this sampler stays an "
                        f"ordinary texture input",
                    )
                )
            continue

        sampler = match.group("name")
        name = _sampler_to_step_name(sampler)
        spec = _parse_rider_tokens(tokens[1:], name, sampler, path, line_idx, result)
        if spec is None:
            # The rider is already reported; don't also report its body as an orphan.
            near_missed.add(name)
            continue

        if name in seen:
            result.errors.append(
                ShaderError(
                    path,
                    line_idx,
                    f"step '{name}' is already declared on line {seen[name] + 1}",
                )
            )
            continue
        if spec.fn_name not in declared_fns:
            result.errors.append(
                ShaderError(
                    path,
                    line_idx,
                    f"step '{name}' has no body: add "
                    f"`void {spec.fn_name}(out vec4 o)`",
                )
            )
            continue

        seen[name] = line_idx
        result.steps.append(spec)

    _report_orphan_bodies(source, declared_fns, seen | dict.fromkeys(near_missed, -1), path, result)
    return result


def _parse_rider_tokens(
    tokens: list[str],
    name: str,
    sampler: str,
    path: Path,
    line_idx: int,
    result: StepParseResult,
) -> StepSpec | None:
    scale = 1.0
    size: tuple[int, int] | None = None
    dtype = DEFAULT_DTYPE
    filter_linear = DEFAULT_FILTER_LINEAR
    wrap = DEFAULT_WRAP
    persist = False

    for raw in tokens:
        token = raw.lower()
        if token.startswith("scale:"):
            value = token.split(":", 1)[1].strip()
            try:
                scale = float(value)
            except ValueError:
                result.errors.append(
                    ShaderError(path, line_idx, f"step '{name}': bad scale '{value}'")
                )
                return None
            if scale <= 0.0:
                result.errors.append(
                    ShaderError(
                        path, line_idx, f"step '{name}': scale must be > 0, got {scale}"
                    )
                )
                return None
        elif token.startswith("size:"):
            value = token.split(":", 1)[1].strip()
            parts = value.split("x")
            if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
                result.errors.append(
                    ShaderError(
                        path, line_idx, f"step '{name}': bad size '{value}', want WxH"
                    )
                )
                return None
            size = (int(parts[0]), int(parts[1]))
            if min(size) < 1:
                result.errors.append(
                    ShaderError(path, line_idx, f"step '{name}': size must be >= 1x1")
                )
                return None
        elif token in _DTYPES:
            dtype = token
        elif token in _FILTERS:
            filter_linear = token == "linear"
        elif token in _WRAPS:
            wrap = token == "repeat"
        elif token == "persist":
            persist = True
        else:
            result.errors.append(
                ShaderError(path, line_idx, f"step '{name}': unknown option '{raw}'")
            )
            return None

    return StepSpec(
        name=name,
        sampler=sampler,
        scale=scale,
        size=size,
        dtype=dtype,
        filter_linear=filter_linear,
        wrap=wrap,
        persist=persist,
    )


def _report_orphan_bodies(
    source: str,
    declared_fns: set[str],
    seen: dict[str, int],
    path: Path,
    result: StepParseResult,
) -> None:
    # A `step_x` body with no `// step` rider would never run; that is a typo, not intent.
    for fn_name in sorted(declared_fns):
        name = fn_name[len(STEP_FN_PREFIX) :]
        if name in seen:
            continue
        line_idx = 0
        for i, line in enumerate(source.splitlines()):
            if re.match(rf"^\s*void\s+{re.escape(fn_name)}\s*\(", line):
                line_idx = i
                break
        result.errors.append(
            ShaderError(
                path,
                line_idx,
                f"'{fn_name}' is never run: no sampler declares it. Add "
                f"`uniform sampler2D u_{name};  // step`",
            )
        )
