from collections import deque

from imgui_bundle import imgui


class Notifications:
    def __init__(self, stack_size: int = 5) -> None:
        self._stack = deque(maxlen=stack_size)  # type: ignore

    def push(self, text: str, color=(0.0, 1.0, 0.0), ttl=5.0) -> None:  # type: ignore
        self._stack.appendleft([text, color, ttl])

    def update_and_draw(self) -> None:
        # ----------------------------------------------------------------
        # Update
        alive_inds = []
        for i in range(len(self._stack)):
            self._stack[i][2] -= imgui.get_io().delta_time
            if self._stack[i][2] > 0.0:
                alive_inds.append(i)

        if len(alive_inds) != len(self._stack):
            self._stack = deque(
                [self._stack[i] for i in alive_inds], maxlen=self._stack.maxlen
            )

        if not self._stack:
            return

        # ----------------------------------------------------------------
        # Draw
        pad = 10.0
        gap = 7.0

        window_size = imgui.get_window_size()
        current_y = pad

        for text, color, _ in self._stack:
            text_size = imgui.calc_text_size(text)
            x = window_size.x - text_size.x - pad
            imgui.set_cursor_pos((x, current_y))
            imgui.text_colored((*color, 1.0), text)
            current_y += text_size.y + gap
