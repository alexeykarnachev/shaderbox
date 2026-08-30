"""A video uniform closes a small gap by decoding forward, not by seeking.

`Video.update` used to demand an exact +1 gap and take a random-access seek for anything
else — including a +2. A seek costs ~50x a forward grab, and the cost is self-reinforcing:
the seek lengthens the frame, the longer frame skips a video frame, and the wider gap forces
the next seek. Below the video's own fps that tips from ~0% to ~100% of updates, and the
editor lurches whenever a video-bearing document is on screen.

The frame that ends up on the texture must be the SAME one a seek would have produced —
that is the half a performance fix can quietly break.
"""

import cv2
import moderngl
import numpy as np
import pytest

from shaderbox import media
from shaderbox.media import Video

_VIDEO = "shaderbox/resources/document_examples/73ea2431-13f6-41e4-b923-04d846b678b0/media/main/u_video.mp4"


@pytest.fixture(scope="module")
def gl() -> moderngl.Context:
    return moderngl.create_standalone_context(require=460)


class _CountingCapture:
    """Wraps the real capture to count random-access seeks."""

    def __init__(self, cap: cv2.VideoCapture) -> None:
        self._cap = cap
        self.seeks = 0

    def __getattr__(self, name: str) -> object:
        return getattr(self._cap, name)

    def set(self, prop: int, value: float) -> bool:
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self.seeks += 1
        return self._cap.set(prop, value)

    def grab(self) -> bool:
        return self._cap.grab()

    def read(self) -> tuple[bool, object]:
        return self._cap.read()


@pytest.mark.parametrize("gap", [1, 2, 3, 5, 8, 9, 20])
def test_the_frame_shown_matches_what_a_seek_would_give(gl, gap: int) -> None:
    # Covers both sides of _MAX_GRAB_AHEAD: gaps <= 8 walk forward, 9 and 20 still seek.
    video = Video(_VIDEO)
    video.restart()
    video.update(0.0)
    index = gap % video._n_frames
    video.update(index / video._fps)
    got = video.texture.read()

    reference = cv2.VideoCapture(_VIDEO)
    reference.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = reference.read()
    assert ok
    want = np.ascontiguousarray(
        np.flipud(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
    ).tobytes()

    assert got == want, f"gap={gap} put the wrong frame on the texture"


def test_a_small_gap_does_not_seek(gl) -> None:
    # The regime that used to spiral: the app running at half the video's frame rate, so
    # every update lands a +2 gap.
    video = Video(_VIDEO)
    counting = _CountingCapture(video._cap)
    video._cap = counting
    video.update(0.0)
    baseline = counting.seeks

    steps = 60
    for i in range(1, steps + 1):
        video.update((i * 2) / video._fps)

    seeks = counting.seeks - baseline
    # Only the wrap-arounds may seek; a +2 gap never should.
    assert seeks <= steps * 2 // video._n_frames + 1, (
        f"{seeks} seeks over {steps} updates at a +2 gap — the forward-decode path is not "
        "being taken, which is the seek death-spiral"
    )


def test_a_large_gap_still_seeks(gl) -> None:
    # The other side of the bound: forward-decoding a long jump would be slower than seeking,
    # so a scrub must stay on the seek path.
    video = Video(_VIDEO)
    counting = _CountingCapture(video._cap)
    video._cap = counting
    video.update(0.0)
    baseline = counting.seeks

    video.update((media._MAX_GRAB_AHEAD + 10) / video._fps)

    assert counting.seeks == baseline + 1


def test_a_backwards_gap_seeks(gl) -> None:
    video = Video(_VIDEO)
    video.update(20 / video._fps)
    counting = _CountingCapture(video._cap)
    video._cap = counting

    video.update(2 / video._fps)

    assert counting.seeks == 1, "a backwards jump cannot be reached by decoding forward"


def test_first_texture_access_survives_an_end_of_stream_capture(gl) -> None:
    # `texture` ignored grab()'s return value, so a capture parked past the last frame
    # retrieved None and crashed inside cvtColor — taking the whole document load with it
    # (load_from_dir warm-renders, which touches .texture). Forward-decoding makes an
    # end-of-stream position reachable, so this went from latent to a ~25% flake.
    video = Video(_VIDEO)
    video._cap.set(cv2.CAP_PROP_POS_FRAMES, video._n_frames)
    assert not video._cap.grab(), "the capture must really be at end-of-stream"

    fresh = Video(_VIDEO)
    fresh._cap.set(cv2.CAP_PROP_POS_FRAMES, fresh._n_frames)

    assert fresh.texture is not None  # rewinds and shows frame 0 rather than crashing
