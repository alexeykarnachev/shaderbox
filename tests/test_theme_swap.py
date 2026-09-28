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
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 0),
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (128, 0, 255),
        (255, 128, 0),
        (0, 128, 255),
        (128, 255, 0),
        (255, 0, 128),
        (0, 255, 128),
        (200, 100, 50),
        (50, 200, 100),
        (100, 50, 200),
        (220, 20, 60),
        (20, 220, 60),
        (60, 20, 220),
        (240, 240, 40),
        (40, 240, 240),
        (240, 40, 240),
        (10, 90, 170),
        (170, 10, 90),
        (90, 170, 10),
        (200, 200, 200),
        (30, 30, 30),
        (140, 140, 140),
    ]
]

_REPORT = """
import json
from shaderbox.theme import COLOR

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
    assert len(swapped) > 30, f"only {len(swapped)} colours reported"
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
    # Pure black and white are not theme colours -- they are the neutral ends every
    # palette shares, so a swap leaves them alone by construction.
    neutral = {"#000000", "#ffffff"}
    survivors = {
        name: colour
        for name, colour in swapped.items()
        if colour.lower() not in probe and colour.lower() not in neutral
    }
    assert not survivors, (
        f"these colours survived a full palette swap, so they are literals outside `_P`: "
        f"{survivors}"
    )


def test_the_canvas_palette_follows_a_swap(swapped: dict[str, str]) -> None:
    """`canvas.theme` is a SECOND palette the app draws from, and it must move too.

    It holds the node graph's colours and the library parses it, so before its fields
    named `_P` entries a swapped theme left the whole graph gruvbox while every panel
    around it changed. The file still decides WHICH entry each field takes -- those are
    measured choices whose reasoning is in its own comments -- and only the colour behind
    the name moves.
    """
    import json

    probe_theme = _REPO / "shaderbox" / "resources" / "graph_canvas" / "canvas.theme"
    source = probe_theme.read_text()
    named = [
        line.split("=")[1].strip().split()[0]
        for line in source.splitlines()
        if "=" in line
        and not line.lstrip().startswith("#")
        and line.split("=")[1].strip()[:1].isalpha()
    ]
    # Count the RAW colour fields, not the named ones: asserting "some are named" passes
    # when one reverts to floats, which is the drift this gate exists to catch. Exactly one
    # field is allowed to hold literals -- `control`, which its own comment explains as a
    # deliberate departure from every palette entry.
    raw = [
        line.split("=")[0].strip()
        for line in source.splitlines()
        if "=" in line
        and not line.lstrip().startswith("#")
        and re.match(r"^[\d.]+ [\d.]+ [\d.]+", line.split("=")[1].strip())
    ]
    assert raw == ["control"], (
        f"these fields hold literal colours a palette swap cannot reach: {raw}"
    )
    assert named, "no field names a palette entry -- the file is still raw floats"

    # Every name it uses must exist, or the resolver raises at load and the canvas is
    # black. Cheap here, expensive in a frame.
    report = subprocess.run(
        [
            sys.executable,
            "-c",
            "from shaderbox.theme import _P;"
            f"import json;print(json.dumps([n for n in {named!r} if n not in _P]))",
        ],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert report.returncode == 0, report.stderr
    assert json.loads(report.stdout) == [], (
        "theme names a palette entry that does not exist"
    )


def test_a_theme_field_can_override_the_palette_entrys_alpha() -> None:
    """`name alpha` in a `.theme` file means that entry's colour at that transparency.

    Gated because nothing exercised it. `canvas.theme` uses the syntax for `wire_outline`
    and `pin_ring`, both naming `bg_0h`, whose own alpha is 1.0 -- so dropping the
    override makes two translucent washes draw fully opaque, and every existing test
    still passed: the theme tests compare `parse_theme(resolve_palette_refs(text))`
    against `canvas_theme()`, and BOTH sides route through the same function, so they
    agree with each other while disagreeing with the file.

    The lesson generalises past this field: a comparison whose two sides share the
    machinery under test cannot see that machinery fail.
    """
    from shaderbox.theme import _P, resolve_palette_refs

    entry_alpha = _P["bg_0h"][3]
    assert entry_alpha != 0.5, "pick a probe alpha the entry does not already have"

    resolved = resolve_palette_refs("wire_outline = bg_0h 0.5\n")
    assert resolved.strip().endswith("0.5"), (
        f"the explicit alpha did not reach the parsed value: {resolved!r}"
    )

    # And the RGB still comes from the entry, so an override changes only transparency.
    r, g, b, _ = _P["bg_0h"]
    assert resolved.strip().split("=")[1].split()[:3] == [
        f"{r:.4f}",
        f"{g:.4f}",
        f"{b:.4f}",
    ]

    # Without an alpha the entry's own is kept, which is the other half of the branch.
    assert resolve_palette_refs("canvas = bg_0h\n").strip().endswith(str(entry_alpha))


def test_the_shipped_theme_keeps_its_translucent_washes() -> None:
    """The two fields that use the override, read back through the real loader.

    A unit test on the resolver alone would pass with the loader wired to something
    else, so this asserts the value the CANVAS receives.
    """
    from shaderbox.widgets.pass_graph import canvas_theme

    theme = canvas_theme()
    assert theme.wire_outline[3] == pytest.approx(0.85), (
        "the wire outline lost its transparency and will draw as an opaque band"
    )
    assert theme.pin_ring[3] == pytest.approx(0.55)
