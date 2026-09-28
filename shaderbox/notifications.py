from collections import deque
from dataclasses import dataclass

from imgui_bundle import imgui
from loguru import logger

from shaderbox.theme import COLOR, SPACE


@dataclass
class _Notification:
    text: str
    # `None` rather than a module-level snapshot of `COLOR.STATE_OK`: a default argument is
    # evaluated once at import, so the old constant froze whatever the palette was when
    # this module first loaded and no later theme change reached it.
    color: tuple[float, float, float] | None = None
    ttl: float = 5.0


class Notifications:
    def __init__(self, stack_size: int = 5) -> None:
        self._stack: deque[_Notification] = deque(maxlen=stack_size)

    def push(
        self,
        text: str,
        color: tuple[float, float, float] | None = None,
        ttl: float = 5.0,
    ) -> None:
        logger.debug(f"[notification] {text}")
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
        # Draw — toasts rise from the BOTTOM-right of the window. The top is
        # occupied by the tab bar / menu chrome, where notifications were drawn
        # behind it and barely visible; the bottom is clear.
        pad = float(SPACE.MD)
        gap = float(SPACE.SM)

        window_size = imgui.get_window_size()
        line_h = imgui.get_text_line_height_with_spacing()
        current_y = window_size.y - pad - line_h

        for notification in self._stack:
            text_size = imgui.calc_text_size(notification.text)
            x = window_size.x - text_size.x - pad
            imgui.set_cursor_pos((x, current_y))
            # Resolved HERE, not at construction: the palette is read when the row is
            # drawn, so a theme change reaches a notification already on the stack.
            colour = notification.color or COLOR.STATE_OK[:3]
            imgui.text_colored((*colour, 1.0), notification.text)
            current_y -= text_size.y + gap
