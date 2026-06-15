"""Pure unit tests for the YouTube helpers — no GL, no network, no Google libs."""

import json

import pytest

from shaderbox.exporters.base import ExporterValueError
from shaderbox.exporters.youtube_util import (
    DEFAULT_CATEGORY_ID,
    SHORTS_TAG,
    build_client_config,
    build_insert_body,
    decorate_short,
    parse_client_secret_json,
    parse_tags,
    studio_edit_url,
)

_INSTALLED = {
    "installed": {
        "client_id": "cid.apps.googleusercontent.com",
        "client_secret": "secret-xyz",
        "redirect_uris": ["http://localhost"],
    }
}


def test_parse_client_secret_valid_installed() -> None:
    cid, secret = parse_client_secret_json(json.dumps(_INSTALLED))
    assert cid == "cid.apps.googleusercontent.com"
    assert secret == "secret-xyz"


def test_parse_client_secret_web_rejected() -> None:
    blob = json.dumps({"web": {"client_id": "x", "client_secret": "y"}})
    with pytest.raises(ExporterValueError, match="Desktop"):
        parse_client_secret_json(blob)


def test_parse_client_secret_malformed_json() -> None:
    with pytest.raises(ExporterValueError, match="valid JSON"):
        parse_client_secret_json("{not json")


def test_parse_client_secret_empty() -> None:
    with pytest.raises(ExporterValueError):
        parse_client_secret_json("   ")


def test_parse_client_secret_missing_fields() -> None:
    blob = json.dumps({"installed": {"client_id": "x"}})  # no secret
    with pytest.raises(ExporterValueError, match="missing"):
        parse_client_secret_json(blob)


def test_build_client_config_round_trips() -> None:
    cfg = build_client_config("cid", "secret")
    assert cfg["installed"]["client_id"] == "cid"
    assert cfg["installed"]["client_secret"] == "secret"
    assert cfg["installed"]["redirect_uris"] == ["http://localhost"]


def test_studio_edit_url() -> None:
    assert studio_edit_url("ABC123") == "https://studio.youtube.com/video/ABC123/edit"


def test_decorate_short_adds_tag_to_description() -> None:
    title, desc = decorate_short("My clip", "a cool shader")
    assert title == "My clip"
    assert desc.endswith(SHORTS_TAG)


def test_decorate_short_empty_description() -> None:
    _, desc = decorate_short("t", "")
    assert desc == SHORTS_TAG


def test_decorate_short_no_duplicate_when_present_in_title() -> None:
    title, desc = decorate_short("clip #shorts", "body")
    assert title == "clip #shorts"
    assert desc == "body"  # not appended again


def test_decorate_short_no_duplicate_when_present_in_description() -> None:
    _title, desc = decorate_short("clip", "body #Shorts")
    assert desc == "body #Shorts"


def test_build_insert_body_private_and_kids_flag() -> None:
    body = build_insert_body("t", "d", ["a", "b"], "22", is_short=False)
    assert body["status"]["privacyStatus"] == "private"
    assert body["status"]["selfDeclaredMadeForKids"] is False
    assert body["snippet"]["categoryId"] == "22"
    assert body["snippet"]["tags"] == ["a", "b"]


def test_build_insert_body_short_injects_tag() -> None:
    body = build_insert_body("t", "d", [], "22", is_short=True)
    assert SHORTS_TAG in body["snippet"]["description"]


def test_build_insert_body_defaults() -> None:
    body = build_insert_body("", "", [], "", is_short=False)
    assert body["snippet"]["title"] == "Untitled"
    assert body["snippet"]["categoryId"] == DEFAULT_CATEGORY_ID
    assert "tags" not in body["snippet"]  # empty tags omitted


def test_parse_tags() -> None:
    assert parse_tags("a, b ,, c ") == ["a", "b", "c"]
    assert parse_tags("") == []
    assert parse_tags("  ") == []
