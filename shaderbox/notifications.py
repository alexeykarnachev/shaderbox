from collections import deque

from imgui_bundle import imgui

from shaderbox.theme import COLOR

_DEFAULT_COLOR: tuple[float, float, float] = COLOR.STATE_OK[:3]


class Notifications:
    def __init__(self, stack_size: int = 5) -> None:
        self._stack = deque(maxlen=stack_size)  # type: ignore

    def push(self, text: str, color=_DEFAULT_COLOR, ttl=5.0) -> None:  # type: ignore
        self._stack.appendleft([text, color, ttl])

    def update(self) -> None:
        """Age the stack one frame. Discards expired toasts. Call once per frame."""
        alive_inds = []
        for i in range(len(self._stack)):
            self._stack[i][2] -= imgui.get_io().delta_time
            if self._stack[i][2] > 0.0:
                alive_inds.append(i)

        if len(alive_inds) != len(self._stack):
            self._stack = deque(
                [self._stack[i] for i in alive_inds], maxlen=self._stack.maxlen
            )

    @property
    def head(self) -> tuple[str, tuple[float, float, float]] | None:
        """Most-recent toast (text, color), or None if the stack is empty."""
        if not self._stack:
            return None
        text, color, _ttl = self._stack[0]
        return text, color
