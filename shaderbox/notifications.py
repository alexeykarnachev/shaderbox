from collections import deque
from dataclasses import dataclass, field

from imgui_bundle import imgui

from shaderbox.theme import COLOR, SPACE

_DEFAULT_COLOR: tuple[float, float, float] = COLOR.STATE_OK[:3]


@dataclass
class _Notification:
    text: str
    color: tuple[float, float, float] = field(default=_DEFAULT_COLOR)
    ttl: float = 5.0


class Notifications:
    def __init__(self, stack_size: int = 5) -> None:
        self._stack: deque[_Notification] = deque(maxlen=stack_size)

    def push(
        self,
        text: str,
        color: tuple[float, float, float] = _DEFAULT_COLOR,
        ttl: float = 5.0,
    ) -> None:
        self._stack.appendleft(_Notification(text, color, ttl))

    def update_and_draw(self) -> None:
        # ----------------------------------------------------------------
        # Update
        delta_time = imgui.get_io().delta_time
        for notification in self._stack:
            notification.ttl -= delta_time

        alive = [n for n in self._stack if n.ttl > 0.0]
        if len(alive) != len(self._stack):
            self._stack = deque(alive, maxlen=self._stack.maxlen)

        if not self._stack:
            return

        # ----------------------------------------------------------------
        # Draw
        pad = float(SPACE.MD)
        gap = float(SPACE.SM)

        window_size = imgui.get_window_size()
        current_y = pad

        for notification in self._stack:
            text_size = imgui.calc_text_size(notification.text)
            x = window_size.x - text_size.x - pad
            imgui.set_cursor_pos((x, current_y))
            imgui.text_colored((*notification.color, 1.0), notification.text)
            current_y += text_size.y + gap
