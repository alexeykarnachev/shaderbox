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

# The number roadmap.md's own comment states. A gate that allows more than the rule it
# enforces makes the rule the weaker of the two, so this is the stated cap, not a ceiling
# with headroom -- a banner one word over is a banner to trim.
BANNER_WORD_CEILING = 200


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
    assert words <= BANNER_WORD_CEILING, (
        f"Active context is {words} words, over the {BANNER_WORD_CEILING} it states"
    )


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


def test_both_intel_rosters_name_every_module() -> None:
    """`conventions.md` and `dev_flow.md` each enumerate `intel/`, and both read as complete.

    `members.py` landed and neither roster gained it, so a session following either would put
    member logic in `glsl.py` or `index.py` -- splitting a concern that already had a home. A
    roster that reads complete and is not is worse than no roster, so the domain here is the
    package's, not the prose's.
    """
    modules = {
        path.stem
        for path in Path("shaderbox/intel").glob("*.py")
        if path.stem != "__init__"
    }
    for doc, anchor, span in (
        (Path("ai_docs/conventions.md"), "lives in `shaderbox/intel/`", 1600),
        (Path("ai_docs/dev_flow.md"), "**`intel/`**", 600),
    ):
        text = doc.read_text()
        roster = text[text.index(anchor) :][:span]
        # Matched as a backticked module name, not as a bare word: `members` also occurs in
        # the surrounding prose, so a substring test passed with the module itself dropped.
        named = set(re.findall(r"`(\w+)(?:\.py)?`", roster))
        missing = sorted(modules - named)
        assert not missing, f"{doc}: roster omits {missing}"
