"""American spelling on every surface a user or the copilot reads (071 D3): `color`, never
`colour`.

The roster is every tracked text file except three kinds, each excluded for a reason a reader
can check: the vendored editor set (`shaderbox/resources/editor/`, upstream's files, re-copied
whole on every re-vendor), the feature records under `ai_docs/features/` other than the living
tutorial body (they quote what was said at the time, the maintainer's own words included), and
dogfood run transcripts (records of what a model wrote), and this file, which has to name the
word it bans. Identifiers already use `color`, so a hit is prose: a comment, a docstring, a
string, a doc.
"""

import re
import subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_BRITISH = re.compile(r"colour", re.IGNORECASE)
_EXCLUDED_PREFIXES = (
    "shaderbox/resources/editor/",
    "ai_docs/features/",
    "scripts/dogfood/runs/",
    "tests/test_prose_spelling.py",
)
_INCLUDED_UNDER_EXCLUDED = (
    "ai_docs/features/068_radiance_cascades/tutorial_body.html",
)
_TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".html",
    ".glsl",
    ".json",
    ".toml",
    ".txt",
    ".sh",
    ".yaml",
    ".yml",
}


def _roster() -> list[Path]:
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=_ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    files: list[Path] = []
    for rel in tracked:
        excluded = (
            rel.startswith(_EXCLUDED_PREFIXES) and rel not in _INCLUDED_UNDER_EXCLUDED
        )
        if excluded or Path(rel).suffix not in _TEXT_SUFFIXES:
            continue
        files.append(_ROOT / rel)
    return files


def test_no_surface_a_reader_sees_uses_the_british_spelling() -> None:
    hits: list[str] = []
    for path in _roster():
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if _BRITISH.search(line):
                hits.append(f"{path.relative_to(_ROOT)}:{number}: {line.strip()[:80]}")
    assert not hits, "British spelling on a reader-facing surface:\n" + "\n".join(hits)


def test_the_roster_is_not_empty_and_covers_the_package() -> None:
    # The gate is only a gate while it walks something: a roster that silently shrank to nothing
    # would pass forever.
    roster = {p.relative_to(_ROOT).as_posix() for p in _roster()}
    assert "shaderbox/ui.py" in roster
    assert "ai_docs/conventions.md" in roster
    assert "ai_docs/features/068_radiance_cascades/tutorial_body.html" in roster
    assert not any(r.startswith("shaderbox/resources/editor/") for r in roster)
