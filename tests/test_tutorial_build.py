"""The Radiance Cascades tutorial's build (069 W-H).

The tutorial's pass cards and shader listings are GENERATED from the shipped example rather than
typed, so what it teaches cannot drift from what the example is. These tests gate the generation
(every pass has a card and a code block, no marker survives, a card states every row, a code block
is the whole file) and the hand-written prose generation cannot reach: a fragment naming a uniform
no shipped shader declares, a script instruction with no chord, a chord the command table does not
have, and the `jfa` run count against the canvas clamp.

GL-free and display-free. The generator lives under `ai_docs/`, which is not a package, so it is
imported by path.
"""

import html
import importlib.util
import json
import math
import pathlib
import re
from typing import Any

import pytest

from shaderbox.commands import COMMAND_SPECS, chord_to_str
from shaderbox.help_content import help_sections
from shaderbox.pass_graph import (
    DEFAULT_DTYPE,
    DEFAULT_FILTER_LINEAR,
    DEFAULT_SCALE,
    DEFAULT_WRAP,
    DTYPES,
    MAX_CANVAS_PX,
    PassEntry,
)
from shaderbox.tabs.document import _SQUARE_PRESETS

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_TUTORIAL_DIR = _REPO_ROOT / "ai_docs" / "features" / "068_radiance_cascades"


def _load_generator() -> Any:
    spec = importlib.util.spec_from_file_location(
        "build_tutorial", _TUTORIAL_DIR / "build_tutorial.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_BUILD = _load_generator()


@pytest.fixture(scope="module")
def graph() -> dict[str, Any]:
    raw: dict[str, Any] = json.loads(
        (_BUILD.EXAMPLE_DIR / "graph.json").read_text(encoding="utf-8")
    )
    return raw


@pytest.fixture(scope="module")
def body() -> str:
    return (_TUTORIAL_DIR / "tutorial_body.html").read_text(encoding="utf-8")


def test_every_pass_has_a_card_and_a_code_block(
    graph: dict[str, Any], body: str
) -> None:
    for name in graph["passes"]:
        assert f"{{{{CARD:{name}}}}}" in body, f"no card marker for pass {name}"
        assert f"{{{{CODE:{name}}}}}" in body, f"no code marker for pass {name}"


def test_no_marker_survives_the_build(tmp_path: pathlib.Path) -> None:
    out = tmp_path / "tutorial.html"
    _BUILD.build(out)
    rendered = out.read_text(encoding="utf-8")
    assert "{{" not in rendered, "an unreplaced marker survived the build"


def test_a_card_states_every_row_and_marks_the_defaults(graph: dict[str, Any]) -> None:
    card = _BUILD._card_html("jfa", graph)
    labels = re.findall(r"<tr><td>([a-z]+)</td>", card)
    assert labels == ["name", "reads", "format", "size", "smooth", "repeat", "runs"]

    rows = dict(re.findall(r"<tr><td>([a-z]+)</td><td>(.*?)</td></tr>", card, re.S))
    assert "32-bit float" in rows["format"] and "dfl" not in rows["format"]
    assert "100%" in rows["size"] and "dfl" in rows["size"]
    assert rows["smooth"].strip() == "off"
    assert "off" in rows["repeat"] and "dfl" in rows["repeat"]

    composite = _BUILD._card_html("composite", graph)
    comp_rows = dict(
        re.findall(r"<tr><td>([a-z]+)</td><td>(.*?)</td></tr>", composite, re.S)
    )
    assert "on" in comp_rows["smooth"] and "dfl" in comp_rows["smooth"]


def test_a_code_block_is_the_whole_file() -> None:
    source = (_BUILD.EXAMPLE_DIR / "passes" / "cascade.frag.glsl").read_text(
        encoding="utf-8"
    )
    block = _BUILD._code_html("cascade")
    assert "#version 460 core" in block
    inner = block[len("<pre><code>") : -len("</code></pre>")]
    assert html.unescape(inner) == source.rstrip()
    assert inner.rstrip().endswith("}")
    assert len(inner.splitlines()) == len(source.rstrip().splitlines())


def test_the_code_block_escapes_html() -> None:
    block = _BUILD._code_html("jfa")
    assert "&lt;" in block
    inner = block[len("<pre><code>") : -len("</code></pre>")]
    assert "<" not in inner


_BRUSH_UNIFORMS = frozenset({"u_brush", "u_brush_prev", "u_brush_down"})


def _hand_written_code_blocks(body: str) -> list[str]:
    blocks = re.findall(r"<pre><code>(.*?)</code></pre>", body, re.S)
    return [b for b in blocks if "{{CODE:" not in b]


def test_no_hand_written_fragment_names_an_absent_uniform(body: str) -> None:
    shipped: set[str] = set()
    for path in sorted((_BUILD.EXAMPLE_DIR / "passes").glob("*.frag.glsl")):
        shipped.update(re.findall(r"\bu_[a-z_]+", path.read_text(encoding="utf-8")))
    allowed = shipped | _BRUSH_UNIFORMS
    for block in _hand_written_code_blocks(body):
        for token in re.findall(r"\bu_[a-z_]+", html.unescape(block)):
            assert token in allowed, f"hand-written fragment names absent uniform {token}"


_INSTRUCTION_VERBS = (
    "add",
    "adds",
    "adding",
    "added",
    "create",
    "creates",
    "creating",
    "created",
    "make",
    "makes",
    "making",
    "open",
    "opens",
    "opening",
    "opened",
    "write",
    "writes",
    "writing",
    "hit",
    "hits",
    "hitting",
    "press",
    "presses",
    "pressing",
    "pressed",
)


def _prose_sentences(body: str) -> list[str]:
    prose = re.sub(r"<pre><code>.*?</code></pre>", " ", body, flags=re.S)
    prose = re.sub(r"</p>|</li>", ".", prose)
    prose = re.sub(r"<[^>]+>", " ", prose)
    prose = html.unescape(prose)
    return [s.strip() for s in re.split(r"[.!?]", prose) if s.strip()]


def _is_script_instruction(sentence: str) -> bool:
    lowered = sentence.lower()
    if "script" not in lowered:
        return False
    return any(re.search(rf"\b{verb}\b", lowered) for verb in _INSTRUCTION_VERBS)


def test_every_script_instruction_carries_the_chord(body: str) -> None:
    for sentence in _prose_sentences(body):
        if not _is_script_instruction(sentence):
            continue
        assert "Alt+R" in sentence or "Script → open" in sentence, (
            f"script instruction with no chord: {sentence!r}"
        )


def test_no_add_the_script_instruction_anywhere(body: str) -> None:
    surfaces = {"tutorial_body.html": body}
    for section in help_sections():
        surfaces[f"help:{section.key}"] = section.body
    add_script = re.compile(r"add[^.]{0,40}script", re.I)
    creation = re.compile(
        r"\b(add|adds|create|creates|make|makes)\b[^.]{0,40}scripts/script\.py", re.I
    )
    for where, text in surfaces.items():
        assert not add_script.search(text), f"{where} instructs adding a script"
        assert not creation.search(text), f"{where} instructs creating scripts/script.py"


def test_the_generator_defaults_match_the_engine() -> None:
    assert _BUILD._DEFAULT_SCALE == DEFAULT_SCALE
    assert _BUILD._DEFAULT_DTYPE == DEFAULT_DTYPE
    assert _BUILD._DEFAULT_FILTER_LINEAR == DEFAULT_FILTER_LINEAR
    assert _BUILD._DEFAULT_WRAP == DEFAULT_WRAP
    assert _BUILD._DEFAULT_ITERATIONS == PassEntry().iterations
    assert set(_BUILD._DTYPE_LABELS) == set(DTYPES)


def test_the_jfa_run_count_covers_every_reachable_canvas(graph: dict[str, Any]) -> None:
    runs: int = graph["passes"]["jfa"]["iterations"]
    assert runs >= math.ceil(math.log2(max(_SQUARE_PRESETS)))
    assert runs >= math.ceil(math.log2(MAX_CANVAS_PX)), (
        f"jfa runs {runs} is short of ceil(log2(MAX_CANVAS_PX={MAX_CANVAS_PX})); "
        "the canvas fields reach the clamp, not only the presets"
    )


def test_the_tutorial_names_no_chord_the_command_table_does_not_have(body: str) -> None:
    known = {chord_to_str(s.default_chord) for s in COMMAND_SPECS if s.default_chord}
    prose = html.unescape(re.sub(r"<[^>]+>", " ", body))
    quoted = set(re.findall(r"\b(?:Ctrl|Alt|Shift)(?:\+(?:Ctrl|Alt|Shift))*\+\w+", prose))
    quoted |= set(re.findall(r"\bF(?:[1-9]|1[0-2])\b", prose))
    for chord in sorted(quoted):
        assert chord in known, f"tutorial quotes {chord}, which no CommandSpec binds"
