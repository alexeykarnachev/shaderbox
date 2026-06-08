from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class RenderDefer:
    # A Render-tab render held one frame so the "Rendering..." cue paints before the synchronous
    # main-thread encode freezes the loop. `shown` records that the cue frame was presented; the
    # encode fires only once it has (ready_to_fire). The GL calls (swap + glFinish) stay in the
    # caller — this owns only the request + the one-frame latch.
    request: Callable[[], None] | None = None
    shown: bool = False

    def submit(self, fn: Callable[[], None]) -> None:
        self.request = fn
        self.shown = False

    def has_request(self) -> bool:
        return self.request is not None

    def ready_to_fire(self) -> bool:
        return self.request is not None and self.shown

    def mark_shown(self) -> None:
        self.shown = True

    def fire_and_clear(self) -> Callable[[], None] | None:
        request = self.request
        self.request = None
        self.shown = False
        return request
