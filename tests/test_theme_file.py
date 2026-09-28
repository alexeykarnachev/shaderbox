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
from typing import Any

import pytest

from shaderbox import syntax_colors
from shaderbox.editor.ffi import Slot
from shaderbox.intel.symbols import SymbolKind
from shaderbox.syntax_colors import (
    _KIND_CAPTURE,
    _LEXER_CLASS_CAPTURE,
    editor_palette,
    kind_capture,
    kind_color,
    kind_slot,
    popup_slot,
    set_theme,
)
from shaderbox.theme import COLOR
from shaderbox.theme_file import (
    _CAPTURE_NAME,
    DEFAULT_THEME,
    Theme,
    ThemeError,
    available_themes,
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
    r, g, b = (round(channel * 255) for channel in colour[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


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
            name, _, _value = stripped.partition("=")
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
    other three numbers. That case is covered by
    `test_intel_sources.py::test_every_kind_has_a_color`, which compares what the palette
    draws against what `kind_color` returns and so fails on any hue the Python invents.

    The credit used to point at `test_swapping_the_file_moves_every_syntax_colour`, which
    does not cover it: that test builds a `Theme` and interrogates the object, never
    importing `syntax_colors`, so no change to the Python can fail it. Measured by
    returning a float tuple from `kind_color` -- the swap test passed, four others did
    not. A wrong credit is worse than none, because it invites deleting the gate that
    works.

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


@pytest.mark.parametrize(
    "name",
    [
        "@variable.",
        "@type.",
        "@function..builtin",
        "@keyword..",
        "@.x",
        "@",
        "variable",
        "",
        # Whitespace inside a segment. `@variable.parameter ` is the one that got past an
        # earlier guard: its segments are not EMPTY, so an empty-segment check waves it
        # through, and the walk then matches no parent and returns the ROOT's colour --
        # `#ebdbb2` where `#83a598` was asked for. A plausible colour, and the wrong one.
        "@variable.parameter ",
        " @variable.parameter",
        "@variable .parameter",
        "@variable.parameter\t",
        # A trailing NEWLINE, which `re.match` and a `$` anchor both tolerate -- the same
        # shape as the trailing space, and it needs `fullmatch` rather than a wider
        # pattern. Two guards in two rounds missed a whitespace shape, so the whole class
        # is enumerated here rather than the instance that happened to be found.
        "@variable.parameter\n",
        "@keyword\n",
        "@vari able",
        # Uppercase and non-ascii: treesitter capture names are lowercase ascii, so these
        # are typos rather than captures this theme happens not to carry.
        "@Variable",
        "@VARIABLE",
    ],
)
def test_a_malformed_capture_name_raises_rather_than_resolving(name: str) -> None:
    """A name that is not a capture name must not return a colour.

    The walk up the dotted parents cannot tell a typo from a fallback: it strips segments
    until something matches, so a malformed name lands on an ancestor and returns a colour
    that looks entirely legitimate. `@variable.parameter ` returning the root's `#ebdbb2`
    instead of `#83a598` is the live shape -- one trailing space, a plausible answer, and
    nothing to distinguish it from a correct mapping.

    Falsifier: drop the guard in `Theme.capture` and every name here resolves silently.
    """
    theme = load_theme()
    with pytest.raises(ThemeError, match="not a capture name"):
        theme.capture(name)


def test_the_guard_admits_every_capture_name_treesitter_emits() -> None:
    """The other side of the guard: it must reject typos without rejecting real captures.

    Anchored to nvim-treesitter's own queries rather than to this theme -- the app must
    survive meeting a capture from a grammar it has never seen, and a guard tightened
    against a typo is exactly the change that would break that.

    Names appearing only inside comments or string literals (`@Nullable` in java, Zig's
    `@cImport`) are not captures and are expected to be rejected; the assertion is over
    names a grammar actually emits, which is why it reads the `@x` of a capture line.
    """
    queries = Path.home() / ".local/share/nvim/lazy/nvim-treesitter/queries"
    if not queries.is_dir():
        pytest.skip("nvim-treesitter queries not installed on this machine")
    emitted: set[str] = set()
    for scm in queries.rglob("highlights.scm"):
        for line in scm.read_text().splitlines():
            # A capture is emitted by a line that is not a comment; the name follows `@`.
            if line.lstrip().startswith(";"):
                continue
            emitted.update(re.findall(r"@[a-z_][a-z0-9_.]*", line))
    assert len(emitted) > 100, f"only {len(emitted)} capture names found -- bad sweep"
    rejected = sorted(n for n in emitted if _CAPTURE_NAME.match(n) is None)
    assert not rejected, f"the guard rejects real capture names: {rejected}"


@pytest.mark.parametrize(
    "name",
    ["@variable", "@constructor", "@variable.parameter", "@function.call.chained"],
)
def test_a_well_formed_capture_still_resolves(name: str) -> None:
    """The other half of the guard: rejecting typos must not reject the real thing.

    `@function.call.chained` is the case that matters -- it is NOT in the theme file, so
    it exercises the dotted-parent fallback the guard sits in front of.

    Asserts the VALUE, not merely that one came back. `assert theme.capture(name)` passes
    on any colour, so a guard that quietly resolved `@variable.parameter` to the root's
    hue satisfied it -- the shape this file exists to prevent, in the test meant to prevent
    it. Each name is checked against what the file itself states for it, or for the nearest
    parent it declares.
    """
    theme = load_theme()
    # The colour the FILE gives this name: its own line if it has one, else the nearest
    # declared parent. Derived from `theme.captures` rather than from `capture()`, so the
    # method under test is not its own reference.
    probe = name
    while probe and probe not in theme.captures:
        probe, _, _ = probe.rpartition(".")
    assert probe, f"{name} has no declared ancestor -- pick a name the theme covers"
    assert theme.capture(name) == theme.captures[probe], (
        f"{name} resolved to something other than {probe}'s colour"
    )


def test_a_kind_draws_one_colour_in_the_buffer_and_the_popup() -> None:
    """The same symbol must not be two colours on two surfaces.

    Class 0 means a DIFFERENT thing in each: in the buffer it falls through to the
    library's TEXT slot, which carries `@variable`'s colour, while in the completion popup
    it means the popup's own plain text -- deliberately dimmer so unselected rows recede.
    So a kind resolving to `@variable` drew `#ebdbb2` in the buffer and `#a89984` in the
    popup, and neither surface was obviously wrong on its own.

    The comparison is between the two SURFACES for one kind, which differ only in the
    property under test; comparing either against a constant would pass whichever way the
    bug fell.

    Falsifier: push `kind_slot` in the popup instead of `popup_slot` and the three plain
    kinds go back to drawing the popup's grey.
    """
    palette = editor_palette()
    theme = load_theme()

    def drawn(cls: int, plain: tuple[float, float, float, float]) -> tuple[float, ...]:
        # Class 0 is not a slot: it means "whatever this surface calls plain".
        return plain if cls == 0 else palette[getattr(Slot, f"SYNTAX_{cls}")]

    buffer_plain = palette[Slot.TEXT]
    popup_plain = palette[Slot.POPUP_TEXT]
    # Contact: the two surfaces really do disagree about class 0, or this test is vacuous
    # and would pass with `popup_slot` deleted.
    assert buffer_plain != popup_plain, (
        "the two surfaces' plain colours are equal, so this fixture cannot see the bug"
    )

    for kind in SymbolKind:
        in_buffer = drawn(kind_slot(kind), buffer_plain)
        in_popup = drawn(popup_slot(kind), popup_plain)
        assert in_buffer == in_popup, (
            f"{kind.name} draws {in_buffer} in the buffer and {in_popup} in the popup"
        )
        # And both are what the theme file asks for, so agreeing on a wrong colour fails.
        assert in_buffer == theme.capture(kind_capture(kind)), (
            f"{kind.name} draws {in_buffer}, not its capture's colour"
        )


# What each kind draws as, restated independently of the table under test. Written out by
# hand from the maintainer's nvim rather than generated from `_KIND_CAPTURE`, because a
# copy of that table would agree with it however it changed.
_EXPECTED_CAPTURE: dict[SymbolKind, str] = {
    SymbolKind.GLSL_KEYWORD: "@keyword",
    SymbolKind.GLSL_TYPE: "@type",
    SymbolKind.GLSL_BUILTIN: "@function.builtin",
    SymbolKind.GLSL_VARIABLE: "@variable.builtin",
    SymbolKind.LIB_FUNCTION: "@function",
    SymbolKind.ENGINE_UNIFORM: "@uniform.engine",
    SymbolKind.PASS_UNIFORM: "@variable",
    SymbolKind.PASS_SAMPLER: "@uniform.sampler",
    SymbolKind.WIRABLE_SAMPLER: "@uniform.sampler",
    SymbolKind.SCRIPT_UNIFORM: "@uniform.script",
    SymbolKind.OUTPUT_VARIABLE: "@output",
    SymbolKind.BUFFER_SYMBOL: "@variable",
    SymbolKind.PY_KEYWORD: "@keyword",
    SymbolKind.PY_BUILTIN: "@function.builtin",
    SymbolKind.PY_API: "@type",
    SymbolKind.PY_MEMBER: "@variable.member",
    SymbolKind.PY_PARAMETER: "@variable.parameter",
    SymbolKind.PY_LOCAL: "@variable",
    SymbolKind.GLSL_MEMBER: "@variable.member",
    SymbolKind.PY_SELF: "@variable.builtin",
    SymbolKind.PY_DUNDER: "@function.builtin",
    SymbolKind.PY_CLASS: "@type",
    SymbolKind.PY_CONSTRUCTOR: "@constructor",
    SymbolKind.PY_DEFINITION: "@function",
    SymbolKind.PY_DECORATOR: "@attribute",
    SymbolKind.PY_ANNOTATION: "@type",
}


def test_every_kind_draws_as_the_capture_it_is_meant_to() -> None:
    """The kind->capture table pinned over the enum's whole domain.

    This is the one hand-written table in the colour system and the thing these commits
    exist to correct, and it was almost entirely unpinned: repointing `GLSL_KEYWORD` at
    `@string` turned every GLSL keyword from red to green with the full suite green. Only
    four captures were named anywhere.

    The expected values are written out by hand rather than generated from the table, so
    this compares two independent statements of the same fact; generating them would
    produce a fixture that agrees with the table however it changes.

    Falsifier: repoint any kind and this names it.
    """
    assert set(_EXPECTED_CAPTURE) == set(SymbolKind), (
        "a kind was added or removed without updating this table: "
        f"{set(SymbolKind) ^ set(_EXPECTED_CAPTURE)}"
    )
    wrong = {
        kind.name: (kind_capture(kind), expected)
        for kind, expected in _EXPECTED_CAPTURE.items()
        if kind_capture(kind) != expected
    }
    assert not wrong, f"kinds drawing as the wrong capture (got, want): {wrong}"


def test_the_app_only_invents_captures_treesitter_does_not_define() -> None:
    """The app's own captures must stay DISJOINT from treesitter's vocabulary.

    `@uniform.engine` and kin describe a shader document, which no grammar has a name for.
    Nothing enforced that they stay outside the namespace they borrow from, so a grammar
    shipping `@uniform.*` would silently give one name two meanings.

    Anchored to nvim-treesitter's queries, which this repo does not author.
    """
    queries = Path.home() / ".local/share/nvim/lazy/nvim-treesitter/queries"
    if not queries.is_dir():
        pytest.skip("nvim-treesitter queries not installed on this machine")
    emitted: set[str] = set()
    for scm in queries.rglob("highlights.scm"):
        for line in scm.read_text().splitlines():
            if line.lstrip().startswith(";"):
                continue
            emitted.update(re.findall(r"@[a-z_][a-z0-9_.]*", line))

    invented = {"@uniform.engine", "@uniform.script", "@uniform.sampler", "@output"}
    assert invented <= set(_KIND_CAPTURE.values()), (
        "this test names captures the app no longer uses; update it"
    )
    collisions = invented & emitted
    assert not collisions, (
        f"these app-invented captures are also treesitter's, so one name now means two "
        f"things: {collisions}"
    )
    # The other direction: every capture the app uses that is NOT invented must be one
    # treesitter actually emits, so a typo cannot masquerade as an app concept.
    borrowed = set(_KIND_CAPTURE.values()) - invented
    unknown = sorted(name for name in borrowed if name not in emitted)
    assert not unknown, (
        f"these look like treesitter captures but none is emitted: {unknown}"
    )


def test_the_class_budget_counts_the_plain_class_too() -> None:
    """The ceiling assert must cover `_PLAIN_CLASS`, which is spent LAST and so is the
    class most likely to overflow.

    An assert over the capture table alone left it unchecked: a theme needing 15 capture
    classes plus a distinct plain colour imported cleanly and then died inside
    `editor_palette` with `AttributeError: 'Slot' has no attribute 'SYNTAX_16'` -- a
    message naming a slot rather than the budget that was exceeded.

    Driven through the REAL `set_theme`, so the module's own assert is what has to fire.
    An earlier version re-derived the assignment inline and asserted against its own
    arithmetic; it passed with `_assert_class_budget` reduced to `pass`, because it never
    called it -- a test of the test.

    Built so the capture table lands ON the ceiling and only plain crosses it: a theme
    that overflows the table too is caught either way and would pass with the fix reverted.
    """
    shipped = theme_path(DEFAULT_THEME).read_text()
    used = sorted(set(_KIND_CAPTURE.values()))
    fresh = [c for c in used if c != "@variable"][:10]
    out: list[str] = []
    section = ""
    hue = 0
    for line in shipped.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            section = stripped
        if section == "[captures]" and stripped.startswith("@"):
            name = stripped.split("=")[0].strip()
            if name == "@variable":
                out.append(f"{name} = #010203")
                continue
            if name in fresh:
                hue += 1
                out.append(f"{name} = #{hue:02x}00{hue:02x}")
                continue
            if name in used:
                out.append(f"{name} = link Keyword")
                continue
        out.append(line)
    text = "\n".join(out)

    # The fixture is only meaningful at the boundary: the capture TABLE must fit while
    # plain does not, or it cannot tell the two asserts apart.
    classes, plain_class = _classes_for(parse_theme(text, "budget"))
    ceiling = sum(1 for slot in Slot if slot.name.startswith("SYNTAX_"))
    assert max(classes.values()) <= ceiling, (
        f"fixture drifted: the capture table needs {max(classes.values())} classes, so "
        f"either assert catches it and this proves nothing"
    )
    assert plain_class > ceiling, (
        f"fixture drifted: plain landed on {plain_class}, inside the ceiling {ceiling}"
    )

    # Now make the MODULE resolve it, so its own assert is the thing under test.
    written = theme_path("budget_probe")
    written.write_text(text)
    try:
        with pytest.raises(AssertionError, match="syntax classes"):
            set_theme("budget_probe")
    finally:
        written.unlink()
        set_theme(DEFAULT_THEME)

    # And the app is left on a working theme rather than half-switched.
    assert kind_color(SymbolKind.PY_KEYWORD) == load_theme().capture("@keyword")


def _classes_for(theme: Theme) -> tuple[dict[str, int], int]:
    """The class assignment a theme implies, derived here rather than read from the module.

    A second statement of the rule, so comparing it against `_CAPTURE_CLASS` compares two
    derivations instead of comparing the module with itself.
    """
    by_colour = {
        theme.capture(capture): cls for cls, capture in _LEXER_CLASS_CAPTURE.items()
    }
    plain = theme.capture("@variable")
    assigned: dict[str, int] = {}
    host = 7
    for capture in sorted(set(_KIND_CAPTURE.values())):
        colour = theme.capture(capture)
        if colour == plain:
            assigned[capture] = 0
            continue
        if colour not in by_colour:
            by_colour[colour] = host
            host += 1
        assigned[capture] = by_colour[colour]
    return assigned, by_colour.get(plain, host)


def test_switching_the_theme_repaints_every_kind() -> None:
    """Selecting another theme is what "changing the theme is easy" actually means.

    Writing a second file was already possible; USING it was not -- `load_theme()` ran once
    at import with no argument, so a second theme required editing Python. This asserts
    the switch reaches the colours.

    Falsifier: make `set_theme` rebind `_THEME` without rebuilding `_CAPTURE_CLASS`, and
    the class table keeps the old theme's grouping.
    """
    themes = available_themes()
    assert len(themes) >= 2, f"only {themes} shipped -- nothing to switch between"
    other = next(name for name in themes if name != DEFAULT_THEME)

    before = {kind: kind_color(kind) for kind in SymbolKind}
    try:
        set_theme(other)
        after = {kind: kind_color(kind) for kind in SymbolKind}
        moved = [kind for kind in SymbolKind if before[kind] != after[kind]]
        assert len(moved) > 10, (
            f"only {len(moved)} kinds changed colour -- the switch did not reach them"
        )
        # Every kind still draws what the NEW theme asks for, so the switch rebuilt the
        # derivation rather than leaving a stale table behind.
        switched = load_theme(other)
        for kind in SymbolKind:
            assert after[kind] == switched.capture(kind_capture(kind)), (
                f"{kind.name} kept a stale colour through the switch"
            )
        # The palette follows too, or the editor would draw the old theme's classes.
        palette = editor_palette()
        for kind in SymbolKind:
            cls = kind_slot(kind)
            if cls:
                assert palette[getattr(Slot, f"SYNTAX_{cls}")] == after[kind]

        # And the class TABLE itself is rebuilt, not just re-read. Asserting colours alone
        # cannot see this: `editor_palette` regenerates from whatever table is current, so
        # a stale table paints the right colour at the wrong class number and both sides
        # agree with each other while disagreeing with the theme. The two shipped themes
        # group `@uniform.sampler` and `@uniform.script` differently, which is what makes
        # the difference observable at all.
        expected_classes, _ = _classes_for(switched)
        # Read through the MODULE: `from ... import _CAPTURE_CLASS` binds the dict at
        # import and would never see `set_theme` rebind the name.
        live_classes = syntax_colors._CAPTURE_CLASS
        assert live_classes == expected_classes, (
            "the class table kept the previous theme's grouping: "
            f"{ {k: (v, expected_classes[k]) for k, v in live_classes.items() if expected_classes[k] != v} }"
        )
    finally:
        set_theme(DEFAULT_THEME)

    assert kind_color(SymbolKind.PY_KEYWORD) == before[SymbolKind.PY_KEYWORD], (
        "the theme did not switch back, so this test leaked into the rest of the suite"
    )


def test_the_shipped_light_theme_matches_nvim_in_light_mode() -> None:
    """The second theme is checked against the same external source as the first.

    Values read from nvim with `background=light`, so the light file is pinned to the
    upstream colorscheme rather than to my transcription of its Lua.
    """
    expected = {
        "@variable": "#3c3836",
        "@keyword": "#9d0006",
        "@string": "#79740e",
        "@type": "#b57614",
        "@variable.parameter": "#076678",
        "@constructor": "#af3a03",
    }
    theme = load_theme("gruvbox_light")
    wrong = {
        capture: (_hex(theme.capture(capture)), want)
        for capture, want in expected.items()
        if _hex(theme.capture(capture)) != want
    }
    assert not wrong, f"light theme disagreeing with nvim (got, want): {wrong}"


def _contrast(fg: tuple[float, ...], bg: tuple[float, ...]) -> float:
    """WCAG relative-contrast ratio, so a theme's readability is a number not an opinion."""

    def luminance(colour: tuple[float, ...]) -> float:
        channels = [
            value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4
            for value in colour[:3]
        ]
        return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]

    high, low = sorted((luminance(fg), luminance(bg)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def test_every_theme_draws_its_text_on_its_own_background() -> None:
    """A theme must supply a ground its foregrounds are legible on.

    The light theme shipped selectable and unusable: `editor_palette` took BACKGROUND from
    `theme.py`, which is near-black, so light foregrounds landed on it at 1.54 -- 22 of 26
    kinds under the 4.5 floor, and the colour most text is drawn in invisible.

    The bound here is 3.3, not 4.5, and the reason matters: gruvbox light's own worst pair
    measures 3.33 IN NVIM (`@comment` on `Normal`), so 4.5 would fail the upstream theme
    this one is a faithful port of. The floor exists to catch a theme drawing on the WRONG
    ground -- a whole-theme mistake, which lands near 1.5 -- not to second-guess a
    colorscheme's own taste.
    """
    for name in available_themes():
        set_theme(name)
        try:
            background = editor_palette()[Slot.BACKGROUND]
            worst = min(
                (_contrast(kind_color(kind), background), kind.name)
                for kind in SymbolKind
            )
            assert worst[0] >= 3.3, (
                f"{name}: {worst[1]} draws at {worst[0]:.2f} contrast on this theme's own "
                f"background -- the theme is probably painting on the other theme's ground"
            )
        finally:
            set_theme(DEFAULT_THEME)


def test_a_theme_that_names_no_chrome_keeps_the_apps_own() -> None:
    """The fallback half: the dark theme names no `[chrome]` and must be unaffected.

    Its colours were chosen against `theme.py`'s tokens, so a chrome section it does not
    have must not change what it draws.
    """
    assert not load_theme(DEFAULT_THEME).chrome, (
        f"{DEFAULT_THEME} now names chrome; this test no longer covers the fallback"
    )
    assert editor_palette()[Slot.BACKGROUND] == COLOR.BG_SURFACE


def test_a_chrome_key_that_is_not_a_slot_is_refused() -> None:
    """A misspelled chrome key must fail rather than be dropped in silence.

    A slot the editor does not have is a typo, and a typo that vanishes leaves the theme
    saying one thing and the screen another -- the same failure the capture guard exists
    for.
    """
    text = theme_path(DEFAULT_THEME).read_text() + "\n[chrome]\nnot_a_slot = bg0\n"
    written = theme_path("chrome_probe")
    written.write_text(text)
    try:
        with pytest.raises(ThemeError, match="not an editor slot"):
            set_theme("chrome_probe")
    finally:
        written.unlink()
        set_theme(DEFAULT_THEME)


def test_the_chrome_palette_and_the_dark_theme_hold_the_same_hues() -> None:
    """`theme.py`'s `_P` and the dark theme file's `[palette]` are the same 27 hues, twice.

    They are separate on purpose -- one themes the app's chrome, the other the editor's
    syntax -- but the dark theme was built to sit inside that chrome, so a hue edited on
    one side and not the other silently breaks the pairing they were tuned as.

    Scoped to what it can decide: this compares VALUE SETS. The two use different names
    for the same hue (`aqua_b` against `bright_aqua`, `bg_0h` against `bg0_h`), so it
    cannot catch a rename, and widening it toward a name map would be inventing a
    correspondence neither file declares. It catches the case that matters -- a palette
    edited in one place.
    """
    from shaderbox.theme import _P

    chrome = {colour[:3] for colour in _P.values()}
    syntax = {colour[:3] for colour in load_theme(DEFAULT_THEME).palette.values()}
    assert chrome == syntax, (
        f"the two palettes have drifted; only in theme.py: {sorted(chrome - syntax)}, "
        f"only in {DEFAULT_THEME}: {sorted(syntax - chrome)}"
    )


def test_a_theme_that_will_not_load_tells_the_user(
    app: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The picker is a combo, so a failure the user is not told about reads as a dead UI.

    `set_theme` is careful -- it restores the default and re-raises rather than leaving
    the app half-switched -- and the caller then caught, logged, and returned. The engine
    was right and the report was missing, which is the same shape as a failed copy or a
    blocked save: the failure reaches a log and never a person.

    Falsifier: drop the `notifications.push` from `App.apply_syntax_theme`.
    """
    from shaderbox import app as app_module

    def refuse(_name: str) -> None:
        raise ThemeError("over the class budget")

    monkeypatch.setattr(app_module, "set_theme", refuse)
    app.app_state.editor_settings.syntax_theme = "nonesuch"
    app.notifications._stack.clear()

    app.apply_syntax_theme()

    texts = [n.text for n in app.notifications._stack]
    assert texts, "the failed theme switch said nothing at all"
    assert any("nonesuch" in text for text in texts), (
        f"the report did not name the theme the user picked: {texts}"
    )
    assert any("budget" in text for text in texts), (
        f"the report did not carry the reason: {texts}"
    )
