"""A theme is a FILE: swapping it re-colours the app, and nothing in the code names a hue.

The gate behind "changing the theme is easy". Its contact proof is that a swapped file
actually moves every syntax colour -- a swap that never happened reports the same clean
result as one that carried perfectly, so each test here checks it landed before checking
what it landed on.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

from shaderbox.intel.symbols import SymbolKind
from shaderbox.syntax_colors import kind_capture
from shaderbox.theme_file import (
    DEFAULT_THEME,
    Theme,
    ThemeError,
    load_theme,
    parse_theme,
    theme_path,
)

_REPO = Path(__file__).resolve().parent.parent

# The colours the user's own nvim reports for these captures, read with
# `nvim_get_hl(0, {name=..., link=false})` under the same gruvbox.nvim the editor loads.
# This is the whole point of naming captures after treesitter's: the two sides are
# comparable, so "it doesn't match my editor" is a question with an answer.
_NVIM_GRUVBOX: dict[str, str] = {
    "@constructor": "#fe8019",
    "@variable.parameter": "#83a598",
    "@variable.member": "#83a598",
    "@function.method": "#b8bb26",
    "@function": "#b8bb26",
    "@function.builtin": "#fe8019",
    "@type": "#fabd2f",
    "@variable.builtin": "#fe8019",
    "@variable": "#ebdbb2",
    "@module": "#ebdbb2",
    "@keyword": "#fb4934",
    "@keyword.import": "#8ec07c",
    "@attribute": "#8ec07c",
    "@constant": "#d3869b",
    "@number": "#d3869b",
    "@operator": "#fe8019",
    "@punctuation.bracket": "#fe8019",
    "@string": "#b8bb26",
    "@comment": "#928374",
}


def _hex(colour: tuple[float, float, float, float]) -> str:
    return "#%02x%02x%02x" % (
        round(colour[0] * 255),
        round(colour[1] * 255),
        round(colour[2] * 255),
    )


def test_the_shipped_theme_matches_the_editor_it_names() -> None:
    """Every capture resolves to what nvim's gruvbox resolves it to.

    Falsifier: point `@constructor` at `Function` instead of `Special` and this fails
    naming it -- which is the mismatch that took four rounds to find by eye.
    """
    theme = load_theme()
    wrong = {
        capture: (_hex(theme.capture(capture)), expected)
        for capture, expected in _NVIM_GRUVBOX.items()
        if _hex(theme.capture(capture)) != expected
    }
    assert not wrong, f"captures disagreeing with nvim gruvbox (got, want): {wrong}"


def test_every_kind_resolves_through_the_theme_file() -> None:
    """The enum's whole domain, so a kind added without a capture fails here."""
    theme = load_theme()
    for kind in SymbolKind:
        capture = kind_capture(kind)
        assert capture.startswith("@"), f"{kind} maps to {capture!r}, not a capture"
        assert theme.capture(capture), f"{kind}'s capture {capture} resolves to nothing"


def test_a_capture_falls_back_along_its_dotted_parents() -> None:
    """Treesitter's own rule, so a theme states only what it wants to differ.

    `@function.method` has no line of its own in the shipped file, and must therefore draw
    as `@function` rather than failing or falling to plain text.
    """
    theme = load_theme()
    # A child the shipped file does not mention: it must land on its parent's colour, not
    # on plain text and not on an error.
    child = "@function.call.chained"
    assert child not in theme.captures
    assert theme.capture(child) == theme.capture("@function.call")
    assert theme.capture(child) != theme.capture("@variable"), (
        "the fallback reached the root, so it skipped the parent it should have found"
    )
    # But a capture whose ROOT the theme never declares is a typo, not a fallback case:
    # resolving it quietly to plain text is indistinguishable from a correct mapping, so
    # it must raise instead.
    with pytest.raises(ThemeError, match="fall back to"):
        theme.capture("@nonsense.deeply.nested")


def _swapped_theme(tmp_path: Path) -> Theme:
    """The shipped file with every palette hue replaced, parsed back."""
    source = theme_path(DEFAULT_THEME).read_text()
    out: list[str] = []
    section = ""
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            section = stripped
        if section == "[palette]" and "=" in stripped and not stripped.startswith("#"):
            name, _, value = stripped.partition("=")
            # A deterministic but totally different hue per entry: distinct from each
            # other so a collision cannot make two captures agree by accident.
            digest = abs(hash(name.strip())) % 0xFFFFFF
            out.append(f"{name.strip()} = #{digest:06x}")
            continue
        out.append(line)
    return parse_theme("\n".join(out), "swapped")


def test_swapping_the_file_moves_every_syntax_colour(tmp_path: Path) -> None:
    """The property: no code names a hue, so a new file re-themes the lot.

    Contact proof first -- if the rewrite missed, every capture is still gruvbox and the
    assertion below passes for the wrong reason.
    """
    shipped = load_theme()
    swapped = _swapped_theme(tmp_path)

    gruvbox = set(_NVIM_GRUVBOX.values())
    survivors = {
        capture: _hex(swapped.capture(capture))
        for capture in shipped.captures
        if _hex(swapped.capture(capture)) in gruvbox
    }
    assert not survivors, (
        f"these captures kept a gruvbox hue through a full palette swap, so a colour is "
        f"written somewhere other than the palette: {survivors}"
    )
    # Contact: the swap moved things at all, rather than producing an empty theme.
    moved = [
        capture
        for capture in shipped.captures
        if swapped.capture(capture) != shipped.capture(capture)
    ]
    assert len(moved) > 30, f"only {len(moved)} captures moved -- the swap did not land"


def test_a_second_theme_file_is_all_it_takes(tmp_path: Path) -> None:
    """A whole theme, written from scratch, in the format a colorscheme is already in.

    This is the user-facing claim: re-theming is a file, not a patch. The file below is
    the minimum one -- a palette, the groups the captures link through, and a root.
    """
    text = """
[palette]
ink   = #101010
paper = #f0f0f0
rose  = #ff0066

[groups]
Normal     = paper
Keyword    = rose
Identifier = ink

[captures]
@variable           = link Normal
@keyword            = link Keyword
@variable.parameter = link Identifier
"""
    theme = parse_theme(text, "minimal")
    assert _hex(theme.capture("@keyword")) == "#ff0066"
    assert _hex(theme.capture("@variable.parameter")) == "#101010"
    # Fallback still works in a file that never mentions the child capture.
    assert _hex(theme.capture("@keyword.return")) == "#ff0066"
    # A root this minimal file never declares raises rather than guessing.
    with pytest.raises(ThemeError, match="fall back to"):
        theme.capture("@function.method")


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("[palette]\nx = #zzzzzz\n", "not a #rrggbb"),
        ("[nope]\nx = #ffffff\n", "unknown section"),
        ("[palette]\nx = #ffffff\nx = #000000\n", "defined twice"),
        (
            "[palette]\np = #ffffff\n[captures]\n@variable = link @nowhere\n",
            "unknown",
        ),
        (
            "[palette]\np = #ffffff\n[captures]\n@a = link @b\n@b = link @a\n",
            "cycle",
        ),
        ("[palette]\np = #ffffff\n[captures]\n@keyword = p\n", "fall back nowhere"),
    ],
)
def test_a_broken_theme_file_raises_rather_than_keeping_the_old_one(
    text: str, fragment: str
) -> None:
    """A theme that cannot be resolved must fail LOUDLY.

    The failure this prevents is the expensive one: a typo that leaves the app running in
    the previous theme, so the file says one thing and the screen says another.
    """
    with pytest.raises(ThemeError, match=fragment):
        parse_theme(text, "broken")


def test_no_syntax_colour_is_written_in_the_python() -> None:
    """The structural half: a hue written in the code is the defect to prevent.

    Scoped to what it can actually decide -- a six-digit hex literal in a string. It does
    NOT cover a colour spelled as floats, which no regular expression separates from any
    other three numbers; `test_swapping_the_file_moves_every_syntax_colour` covers that
    case behaviourally, by checking no capture survives a palette swap.

    `theme.py` keeps the app's CHROME palette, a separate question gated by
    `test_theme_swap.py`.
    """
    hex_literal = re.compile(r"""["']#[0-9a-fA-F]{6}["']""")
    for name in ("syntax_colors.py", "theme_file.py"):
        source = (_REPO / "shaderbox" / name).read_text()
        found = hex_literal.findall(source)
        assert not found, f"{name} writes colour literals: {found}"


def test_the_shipped_theme_loads_in_a_clean_process() -> None:
    """Import-time resolution, where a broken file would take the app down at start."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from shaderbox.theme_file import load_theme;"
            "t = load_theme();"
            "print(len(t.captures))",
        ],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert int(result.stdout.strip()) > 30
