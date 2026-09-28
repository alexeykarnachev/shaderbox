"""Replacing the palette re-themes the app: the gate behind "easy to re-theme".

`_P` in `theme.py` is the named palette and the only place a literal colour lives. Every
role colour, every chrome token and every syntax class resolves from it, so a new theme is
a new `_P` and nothing else.

That is a property nothing else checks, and the obvious way to check it does not work.
`_P` -> `COLOR` -> the role table is THREE import-time copies: `_P` is a dict literal,
`_ColorBag` copies values into class attributes when the class body executes, and the role
table copies those at module import. Mutating `_P` in a running process changes nothing,
and `importlib.reload` re-executes the literal and discards the swap. A monkeypatch version
of this test passes while proving nothing.

So it rewrites the source into a temporary tree and imports it in a SUBPROCESS.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent

# Distinct, unmistakable, and nothing like gruvbox: a colour that survives the swap is
# visible as one rather than hiding among near-neighbours.
_PROBE = [
    f"#{r:02x}{g:02x}{b:02x}"
    for r, g, b in [
        (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0),
        (255, 0, 0), (0, 0, 255), (128, 0, 255), (255, 128, 0),
        (0, 128, 255), (128, 255, 0), (255, 0, 128), (0, 255, 128),
        (200, 100, 50), (50, 200, 100), (100, 50, 200), (220, 20, 60),
        (20, 220, 60), (60, 20, 220), (240, 240, 40), (40, 240, 240),
        (240, 40, 240), (10, 90, 170), (170, 10, 90), (90, 170, 10),
        (200, 200, 200), (30, 30, 30), (140, 140, 140),
    ]
]

_REPORT = """
import json
from shaderbox.theme import COLOR, ROLE_COLOR

def hx(c):
    return "#%02x%02x%02x" % (round(c[0]*255), round(c[1]*255), round(c[2]*255))

out = {}
for name in dir(COLOR):
    if name.startswith("_"):
        continue
    value = getattr(COLOR, name)
    if (
        isinstance(value, tuple)
        and len(value) == 4
        and all(isinstance(c, float) for c in value)
    ):
        out["COLOR." + name] = hx(value)
for role, colour in ROLE_COLOR.items():
    out["ROLE." + role] = hx(colour)
print(json.dumps(out))
"""


@pytest.fixture(scope="module")
def swapped(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """Every colour the app reports, with the palette replaced wholesale."""
    tree = tmp_path_factory.mktemp("swap")
    shutil.copytree(_REPO / "shaderbox", tree / "shaderbox", symlinks=True)

    theme = tree / "shaderbox" / "theme.py"
    source = theme.read_text()
    hexes = re.findall(r'_hex\("(#[0-9a-fA-F]{6})"\)', source)
    palette_hexes = list(dict.fromkeys(hexes))
    assert len(palette_hexes) <= len(_PROBE), (
        f"the palette has {len(palette_hexes)} distinct colours, the probe has "
        f"{len(_PROBE)} -- extend the probe"
    )
    for original, probe in zip(palette_hexes, _PROBE, strict=False):
        source = source.replace(f'_hex("{original}")', f'_hex("{probe}")')
    theme.write_text(source)

    result = subprocess.run(
        [sys.executable, "-c", _REPORT],
        cwd=tree,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    import json

    return json.loads(result.stdout)


def test_the_probe_actually_reached_the_app(swapped: dict[str, str]) -> None:
    """Contact proof. If the rewrite missed, every colour below is still gruvbox and every
    other assertion in this file passes for the wrong reason -- a swap that never happened
    reports the same clean result as one that carried perfectly."""
    assert swapped, "the subprocess reported no colours at all"
    assert len(swapped) > 40, f"only {len(swapped)} colours reported"
    probe = {c.lower() for c in _PROBE}
    moved = [name for name, colour in swapped.items() if colour.lower() in probe]
    assert len(moved) > 20, (
        f"only {len(moved)} colours came from the probe palette -- the rewrite did not land"
    )


def test_no_colour_survives_the_palette_swap(swapped: dict[str, str]) -> None:
    """The property itself: nothing the app draws is outside the palette.

    A survivor is a literal written somewhere other than `_P` -- which is how four
    hand-typed alpha tuples sat beside the `_P` entries they duplicated, each one the same
    colour at 0.18 alpha, silently keeping the old theme's accent after a swap.
    """
    probe = {c.lower() for c in _PROBE}
    # Derived colours are legitimately outside the palette: `_muted` keeps a `_P` hue and
    # substitutes saturation and lightness, so its output is palette-DEPENDENT without
    # being a palette entry. They are checked by the hue test below instead.
    neutral = {"#000000", "#ffffff"}
    survivors = {
        name: colour
        for name, colour in swapped.items()
        if colour.lower() not in probe and colour.lower() not in neutral
    }
    # Anything left must be derived, not literal. `_muted` is the only deriver.
    assert not survivors, (
        f"these colours survived a full palette swap, so they are literals outside `_P`: "
        f"{survivors}"
    )
