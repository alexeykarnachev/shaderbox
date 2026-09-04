"""The station's static site (075 W-2): built from a fixture log with no network, every turn on
the attempt page, media by relative path (images as <img>, videos as <video>), a meta-refresh
only while an attempt is live, and no external resource anywhere."""

import re
from pathlib import Path

from PIL import Image

from dogfood.report.build import STORE_DIR_NAME, build_site
from dogfood.report.log import LOG_NAME, EventLog
from scripts.dogfood.analyze import REACHABLE_TOOLS


def _fixture(root: Path) -> Path:
    store = root / STORE_DIR_NAME
    exp_dir = store / "rc_build"
    media = exp_dir / "media" / "1"
    media.mkdir(parents=True)
    Image.new("RGB", (8, 8), (200, 40, 40)).save(media / "frame.png")
    (media / "clip.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")
    log = EventLog(exp_dir / LOG_NAME, "rc_build")
    log.append(
        "experiment_start",
        0,
        {
            "intent": "a working radiance cascades build",
            "mode": "babysat",
            "criteria": ["it merges"],
        },
    )
    log.append("attempt_start", 1, {"model": "openai/test", "sha": "abcdef1234567"})
    log.append(
        "context",
        1,
        {
            "turn": 1,
            "iteration": 0,
            "blocks": [
                {
                    "name": "static",
                    "volatility": "STATIC",
                    "messages": 1,
                    "chars": 400,
                    "est_tokens": 100,
                    "text": "SYSTEM PROMPT TEXT",
                },
                {
                    "name": "dialogue",
                    "volatility": "DIALOGUE",
                    "messages": 2,
                    "chars": 40,
                    "est_tokens": 10,
                    "text": "hi",
                    "trimmed": True,
                    "dropped_messages": 4,
                },
                {
                    "name": "working_set",
                    "volatility": "PER_TURN",
                    "messages": 1,
                    "chars": 800,
                    "est_tokens": 200,
                    "text": "WORKING SET -- void main() {}",
                },
            ],
            "tools": ["edit_shader"],
            "tools_chars": 400,
            "tools_est_tokens": 100,
            "tools_text": '[{"name": "edit_shader"}]',
            "est_total_tokens": 410,
            "billed": {"input_tokens": 450, "cached_tokens": 300},
        },
    )
    log.append(
        "turn",
        1,
        {
            "n": 1,
            "user_text": "make the <first> pass red",
            "assistant_text": "Made it red & probed it.",
            "iterations": [
                {
                    "iteration": 0,
                    "input_tokens": 450,
                    "output_tokens": 30,
                    "cached_tokens": 300,
                    "cost_usd": 0.01,
                }
            ],
            "tool_calls": [
                {
                    "n": 1,
                    "name": "edit_shader",
                    "args": {"old_str": "a"},
                    "ok": True,
                    "result": "applied",
                }
            ],
            "usage": {
                "input_tokens": 450,
                "output_tokens": 30,
                "cached_tokens": 300,
                "cost_usd": 0.01,
            },
            "renders": [
                {"path": "media/1/frame.png", "label": "t=0"},
                {"path": "media/1/clip.mp4", "label": "3s clip"},
            ],
            "gates": [],
            "terminal": "turn_done",
            "cutoff": "",
        },
    )
    log.append(
        "turn",
        1,
        {
            "n": 2,
            "user_text": "now animate it",
            "assistant_text": "",
            "iterations": [
                {
                    "iteration": 0,
                    "input_tokens": 500,
                    "output_tokens": 0,
                    "cost_usd": 0.02,
                }
            ],
            "tool_calls": [],
            "usage": {"input_tokens": 500, "output_tokens": 0, "cost_usd": 0.02},
            "renders": [],
            "terminal": "turn_done",
            "cutoff": "max_iterations",
        },
    )
    log.append("note", 1, {"text": "the red is there", "axis": "fidelity", "turn": 1})
    log.append(
        "context", 1, {"turn": 3, "iteration": 0, "blocks": [], "est_total_tokens": 5}
    )
    # A second, finished experiment.
    other = EventLog(store / "older" / LOG_NAME, "older")
    other.append("experiment_start", 0, {"intent": "o", "mode": "free_run"})
    other.append("attempt_start", 1, {"model": "m"})
    other.append("attempt_end", 1, {"outcome": "success", "summary": "fine"})
    return root


def test_site_builds_every_page_from_the_log(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "dogfood")
    written = build_site(root)
    names = {p.relative_to(root).as_posix() for p in written}
    assert names == {
        "index.html",
        "runs/rc_build/index.html",
        "runs/rc_build/attempt_1.html",
        "runs/older/index.html",
        "runs/older/attempt_1.html",
    }
    index = (root / "index.html").read_text()
    assert 'href="runs/rc_build/index.html"' in index and "LIVE" in index
    assert "a working radiance cascades build" in index


def test_attempt_page_carries_every_turn_media_and_context(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "dogfood")
    build_site(root)
    page = (root / "runs/rc_build/attempt_1.html").read_text()
    # Every turn, escaped.
    assert "make the &lt;first&gt; pass red" in page and "now animate it" in page
    assert "Made it red &amp; probed it." in page
    # Media by relative path: image as <img>, video as <video>.
    assert '<img src="media/1/frame.png"' in page
    assert '<video src="media/1/clip.mp4"' in page
    for m in re.finditer(r'(?:src|href)="([^"]+)"', page):
        ref = m.group(1)
        if ref.startswith("#") or ref.endswith(".html"):
            continue
        assert (root / "runs/rc_build" / ref).exists(), f"dangling media ref {ref}"
    # The context panel: proportional bar, expandable block text, the trim flag, the cache join.
    assert "SYSTEM PROMPT TEXT" in page and "WORKING SET -- void main() {}" in page
    assert "TRIMMED, 4 messages dropped" in page
    assert "450 billed, 300 cached (67%)" in page
    assert 'class="bar"' in page and "<details>" in page
    # The honesty AUTO half and the process axis.
    assert "turn 2 (max_iterations)" in page
    for tool in REACHABLE_TOOLS:
        assert f"<code>{tool}</code>" in page
    assert "the red is there" in page
    # A context that arrived for a turn with no turn record shows as in progress.
    assert "Turn 3" in page and "in progress" in page
    # Live attempt: meta refresh; and nothing external.
    assert 'http-equiv="refresh"' in page
    assert "http://" not in page and "https://" not in page


def test_finished_attempt_has_no_refresh(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "dogfood")
    build_site(root)
    page = (root / "runs/older/attempt_1.html").read_text()
    assert 'http-equiv="refresh"' not in page and "success" in page
    exp_page = (root / "runs/older/index.html").read_text()
    assert 'http-equiv="refresh"' not in exp_page


def test_index_links_prior_reports_when_present(tmp_path: Path) -> None:
    root = _fixture(tmp_path / "dogfood")
    features = tmp_path / "ai_docs" / "features"
    features.mkdir(parents=True)
    (features / "035_dogfood_report_mega.md").write_text("# old")
    build_site(root)
    index = (root / "index.html").read_text()
    assert 'href="../ai_docs/features/035_dogfood_report_mega.md"' in index


def test_empty_store_still_builds_an_index(tmp_path: Path) -> None:
    written = build_site(tmp_path)
    assert [p.name for p in written] == ["index.html"]
    assert "No experiments recorded yet" in (tmp_path / "index.html").read_text()
