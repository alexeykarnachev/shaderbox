"""The roadmap's own stated shape, enforced (082 review).

`roadmap.md` states its Active-context and row shapes in HTML comments and nothing checked
them, so both drifted: a banner edit appended instead of replacing and ran to ~580 words
against a stated 200, and the same edit left a fixed bug described as open. A convention with
no gate is a wish -- these are the parts of the stated shape a check can decide.
"""

import re
from pathlib import Path

ROADMAP = Path("ai_docs/roadmap.md")

STATUSES = frozenset({"pending", "in progress", "done", "partial", "superseded"})

# The stated cap is 200; the check allows headroom so a banner is not rewritten for one word,
# while still failing the append pattern that produced 580.
BANNER_WORD_CEILING = 260


def _banner() -> str:
    text = ROADMAP.read_text()
    start = text.index("## Active context")
    return text[start : text.index("## Features")]


def _rows() -> list[list[str]]:
    text = ROADMAP.read_text()
    table = text[text.index("## Features") :]
    found: list[list[str]] = []
    for line in table.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 4 and cells[0] not in ("#",) and not cells[0].startswith("--"):
            found.append(cells)
    return found


def test_the_banner_stays_within_its_stated_budget() -> None:
    # The failure this catches is an APPEND: a new block written above the old one keeps every
    # word of both. Broken (paste the previous banner back under the current one), this fails.
    words = len(re.sub(r"<!--.*?-->", "", _banner(), flags=re.DOTALL).split())
    assert words <= BANNER_WORD_CEILING, f"Active context is {words} words"


def test_the_banner_carries_its_own_date_stamp() -> None:
    assert re.search(r"<!-- As of \d{4}-\d{2}-\d{2},", _banner()) is not None


def test_every_feature_row_has_a_status_from_the_vocabulary() -> None:
    rows = _rows()
    assert rows, "no feature rows parsed -- the table shape changed"
    for cells in rows:
        assert cells[2] in STATUSES, f"{cells[1]}: status {cells[2]!r}"


def test_every_feature_row_points_at_a_spec_that_exists() -> None:
    # A row's Spec pointer is the one thing a cold start follows out of the table, so a path
    # that no longer resolves is worse than no pointer. Commit shas are not checked here --
    # `git log` is the authority on those; a path is this repo's to keep true. A brace
    # expression names several files at once (`ai_docs/{a,b}.md`) and every branch is checked.
    for cells in _rows():
        for found in re.findall(r"`(ai_docs/[^`]+\.md)`", cells[3]):
            brace = re.search(r"\{([^}]*)\}", found)
            paths = (
                [found]
                if brace is None
                else [
                    found[: brace.start()] + name + found[brace.end() :]
                    for name in brace.group(1).split(",")
                ]
            )
            for path in paths:
                assert Path(path).exists(), f"{cells[1]}: {path} does not exist"
