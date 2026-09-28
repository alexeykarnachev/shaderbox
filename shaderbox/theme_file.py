"""Reads a theme file: a named palette, vim base groups, and treesitter captures.

The format is the one vim colorschemes use, because that is what themes are
already written in -- a palette of hues, a set of base groups over it, and
captures that LINK into those groups. Resolving a link chain is the whole
mechanism; everything a theme decides, it decides by where its links point.

Nothing in the Python names a colour. `shaderbox/resources/themes/*.theme`
are the colours, and swapping the file swaps the app.
"""

import re
from dataclasses import dataclass
from pathlib import Path

Color = tuple[float, float, float, float]

_SECTION = re.compile(r"^\[(\w+)\]$")
_ENTRY = re.compile(r"^([@\w.]+)\s*=\s*(.+?)$")
_HEX = re.compile(r"^#([0-9a-fA-F]{6})$")

# What a capture name IS: `@` then dot-separated segments of lowercase letters, digits and
# underscores. Taken from the names nvim-treesitter's own queries emit -- a sweep of every
# `highlights.scm` it ships finds no other character in one.
_CAPTURE_NAME = re.compile(r"^@[a-z0-9_]+(\.[a-z0-9_]+)*$")

# A capture falls back to its parent by dropping the last dotted segment, which is
# treesitter's own rule: `@function.method` with no line of its own draws as
# `@function`, and `@function` as `@variable`. A theme therefore states only what it
# wants to differ, and a capture this app asks for that the theme never heard of still
# resolves instead of failing.
_ROOT_FALLBACK = "@variable"


class ThemeError(Exception):
    """A theme file that cannot be resolved, named with the line that broke it."""


@dataclass(frozen=True)
class Theme:
    """A resolved theme: every name mapped to a concrete colour."""

    name: str
    palette: dict[str, Color]
    groups: dict[str, Color]
    captures: dict[str, Color]
    # The editor's own furniture -- background, caret, gutter, status row. A theme that
    # names none keeps the app's chrome tokens, which is what the dark theme does: it was
    # BUILT to match them. A light theme cannot, since a light foreground on the app's
    # near-black ground is unreadable, so it carries its own.
    chrome: dict[str, Color]

    def capture(self, name: str) -> Color:
        """The colour a capture draws in, falling back along its dotted parents.

        A name the theme cannot account for is a typo rather than a fallback case, and
        raises: a capture resolving quietly to a plausible colour is indistinguishable
        from a correct mapping, so every gate downstream passes on it. Two shapes qualify
        -- a root the theme never declares, and a name that is not a capture name at all.

        The second is stated as what a capture name IS (`_CAPTURE_NAME`) rather than as a
        list of malformed shapes, because the list is never complete: an earlier version
        rejected an EMPTY segment and still let `@variable.parameter ` through, whose
        trailing space is not empty, so the walk missed every parent and returned the
        root's colour -- `#ebdbb2` where `#83a598` was asked for.
        """
        if _CAPTURE_NAME.fullmatch(name) is None:
            raise ThemeError(f"{self.name}: {name!r} is not a capture name")
        probe = name
        while probe:
            if probe in self.captures:
                return self.captures[probe]
            head, _, _ = probe.rpartition(".")
            probe = head
        root = name.split(".")[0]
        if root not in self.captures:
            raise ThemeError(
                f"{self.name}: no capture {name!r} and no {root!r} to fall back to"
            )
        return self.captures[_ROOT_FALLBACK]


def _parse_hex(text: str) -> Color:
    match = _HEX.match(text)
    if match is None:
        raise ThemeError(f"not a #rrggbb colour: {text!r}")
    value = match.group(1)
    return (
        int(value[0:2], 16) / 255.0,
        int(value[2:4], 16) / 255.0,
        int(value[4:6], 16) / 255.0,
        1.0,
    )


def parse_theme(text: str, name: str) -> Theme:
    """Parse a theme file's text. Raises `ThemeError` naming the offending line."""
    sections: dict[str, dict[str, str]] = {
        "palette": {},
        "groups": {},
        "captures": {},
        "chrome": {},
    }
    current: str = ""
    for lineno, raw in enumerate(text.splitlines(), start=1):
        # A `#` opens a comment only at the start of a line: everywhere else it opens a
        # colour, which is most of what this file contains.
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        header = _SECTION.match(line)
        if header is not None:
            current = header.group(1)
            if current not in sections:
                raise ThemeError(f"{name}:{lineno}: unknown section [{current}]")
            continue
        entry = _ENTRY.match(line)
        if entry is None or not current:
            raise ThemeError(f"{name}:{lineno}: cannot read {raw.strip()!r}")
        key, value = entry.group(1), entry.group(2).strip()
        if key in sections[current]:
            raise ThemeError(f"{name}:{lineno}: {key} defined twice")
        sections[current][key] = value

    palette = {key: _parse_hex(value) for key, value in sections["palette"].items()}
    groups = _resolve(sections["groups"], palette, {}, name)
    captures = _resolve(sections["captures"], palette, groups, name)
    chrome = _resolve(sections["chrome"], palette, groups, name)
    if _ROOT_FALLBACK not in captures:
        raise ThemeError(
            f"{name}: no {_ROOT_FALLBACK}, so a capture can fall back nowhere"
        )
    return Theme(
        name=name,
        palette=palette,
        groups=groups,
        captures=captures,
        chrome=chrome,
    )


def _resolve(
    entries: dict[str, str],
    palette: dict[str, Color],
    outer: dict[str, Color],
    name: str,
) -> dict[str, Color]:
    """Turn each entry into a colour, following `link` chains until one lands.

    A chain is followed through this section first and the outer one second, so a
    capture may link to another capture or to a base group with the same word.
    """
    resolved: dict[str, Color] = {}
    for key in entries:
        seen: list[str] = []
        probe = key
        while True:
            if probe in seen:
                raise ThemeError(f"{name}: link cycle {' -> '.join([*seen, probe])}")
            seen.append(probe)
            value = entries.get(probe)
            if value is None:
                if probe in outer:
                    resolved[key] = outer[probe]
                    break
                raise ThemeError(f"{name}: {key} links to unknown {probe!r}")
            if value.startswith("link "):
                probe = value[len("link ") :].strip()
                continue
            if value in palette:
                resolved[key] = palette[value]
                break
            resolved[key] = _parse_hex(value)
            break
    return resolved


_THEME_DIR = Path(__file__).resolve().parent / "resources" / "themes"
DEFAULT_THEME = "gruvbox_dark"


def theme_path(name: str) -> Path:
    return _THEME_DIR / f"{name}.theme"


def available_themes() -> list[str]:
    return sorted(path.stem for path in _THEME_DIR.glob("*.theme"))


def load_theme(name: str = DEFAULT_THEME) -> Theme:
    path = theme_path(name)
    if not path.is_file():
        raise ThemeError(
            f"no theme {name!r} in {_THEME_DIR}; have {available_themes()}"
        )
    return parse_theme(path.read_text(), name)
