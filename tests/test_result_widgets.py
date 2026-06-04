"""The structured-result-widget channel (feature 020·21): tool producers emit a TERSE fact (no
url/path in the model-facing message) plus a structured widget in the payload; the agent extracts
the widget without the raw target ever reaching the model. These pin the single most important
property of the feature — the URL/path must NOT leak into the LLM-visible message string."""

from shaderbox.copilot.agent import _widget_from_payload
from shaderbox.copilot.capabilities import PublishResult, RenderResult
from shaderbox.copilot.state import ResultWidget
from shaderbox.copilot.tools.publish import _publish_result, _render_result

_URL = "https://studio.youtube.com/video/abc123/edit"
_PATH = "/home/u/proj/renders/spiral_472c_0.png"


def test_render_result_message_has_no_path() -> None:
    ok, msg, payload = _render_result(
        RenderResult(ok=True, path=_PATH, width=512, height=512), "image"
    )
    assert ok
    assert _PATH not in msg  # the path must never reach the model-facing message
    assert "512x512" in msg
    assert "button is shown" in msg  # the agent is told a widget was surfaced
    assert payload is not None
    assert payload["widget"] == {
        "kind": "open_path",
        "label": "Reveal render",
        "target": _PATH,
    }


def test_publish_result_message_has_no_url() -> None:
    ok, msg, payload = _publish_result(PublishResult(ok=True, url=_URL), "YouTube")
    assert ok
    assert _URL not in msg and "http" not in msg  # the link must never reach the model
    assert "button is shown" in msg
    assert payload is not None
    assert payload["widget"]["kind"] == "open_url"
    assert payload["widget"]["target"] == _URL


def test_render_without_path_emits_no_widget_and_no_promise() -> None:
    # The degraded path (render ok but the file path wasn't captured): no widget, and the message
    # must NOT promise a button that isn't there.
    ok, msg, payload = _render_result(
        RenderResult(ok=True, path="", width=512, height=512), "image"
    )
    assert ok and payload is None
    assert "button" not in msg


def test_publish_without_url_emits_no_widget_and_no_promise() -> None:
    ok, msg, payload = _publish_result(PublishResult(ok=True, url=""), "YouTube")
    assert ok and payload is None
    assert "button" not in msg and "http" not in msg


def test_failed_publish_has_no_widget() -> None:
    ok, msg, payload = _publish_result(PublishResult(ok=False, error="boom"), "YouTube")
    assert not ok and payload is None and msg == "error: boom"


def test_widget_from_payload_is_defensive() -> None:
    # A well-formed widget round-trips; malformed / unknown / empty drops to None (no dead button).
    good = {"widget": {"kind": "open_url", "label": "Open in YouTube", "target": _URL}}
    assert _widget_from_payload(good) == ResultWidget(
        kind="open_url", label="Open in YouTube", target=_URL
    )
    assert _widget_from_payload(None) is None
    assert _widget_from_payload({}) is None
    assert _widget_from_payload({"widget": {"kind": "bogus", "target": "t"}}) is None
    assert _widget_from_payload({"widget": {"kind": "open_url", "target": ""}}) is None
