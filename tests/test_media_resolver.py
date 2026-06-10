"""Pins the media suffix->class resolver: every advertised MEDIA_EXTENSIONS suffix
must resolve (the picker offers all of them and Video.save preserves the source
suffix, so an unresolvable advertised suffix kills the node on reload)."""

import pytest

from shaderbox.constants import IMAGE_EXTENSIONS, MEDIA_EXTENSIONS
from shaderbox.media import Image, Video, media_class_for


def test_every_advertised_suffix_resolves() -> None:
    for suffix in MEDIA_EXTENSIONS:
        cls = media_class_for(suffix)
        expected = Image if suffix in IMAGE_EXTENSIONS else Video
        assert cls is expected, suffix
    assert media_class_for(".webm") is Video


def test_unknown_suffix_raises() -> None:
    with pytest.raises(ValueError, match="suffix"):
        media_class_for(".gif")
