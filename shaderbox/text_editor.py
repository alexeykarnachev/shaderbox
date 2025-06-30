import contextlib
import re
from pathlib import Path

import moderngl
import numpy as np
from numpy.typing import NDArray

from shaderbox.core import RESOURCES_DIR, Font, KeyEvent, Node


class TextEditor:
    def __init__(self, gl: moderngl.Context | None = None) -> None:
        self._gl = gl or moderngl.get_context()

        self._node = Node(
            fs_source=Path(RESOURCES_DIR / "shaders" / "editor.frag.glsl").read_text()
        )
        self._font = Font(
            file_path=str(
                RESOURCES_DIR / "fonts" / "Anonymous_Pro" / "AnonymousPro-Regular.ttf"
            ),
            size=24,
        )

        self._lines: list[str] = []
        self._top_line_idx: int = 0

        self._grid_size: tuple[int, int] = (1, 1)

        self._grid_uvs_data: NDArray[np.float32] | None = None
        self._grid_metrics_data: NDArray[np.float32] | None = None

        self._grid_uvs_texture: moderngl.Texture | None = None
        self._grid_metrics_texture: moderngl.Texture | None = None

        self._cursor_line_pos: tuple[int, int] = (0, 0)
        self._desired_cursor_char_idx = 0
        self._line_to_grid: dict[tuple[int, int], tuple[int, int]] = {}
        self._grid_to_line: dict[tuple[int, int], tuple[int, int]] = {}

    def release(self) -> None:
        self._node.release()

        if self._grid_uvs_texture is not None:
            self._grid_uvs_texture.release()

        if self._grid_metrics_texture is not None:
            self._grid_metrics_texture.release()

    @property
    def canvas_texture(self) -> moderngl.Texture:
        return self._node.canvas.texture

    def set_canvas_size(self, size: tuple[int, int]) -> None:
        width = int(size[0])
        height = int(size[1])
        if self._node.canvas.set_size((width, height)):
            self._update()

    def set_text(self, text: str) -> None:
        self._lines = text.split("\n")

        self._top_line_idx = 0

        self._update()

    def process_mouse_wheel(self, y_offset: int) -> None:
        if y_offset == 0:
            return

        self._inc_top_line_idx(-int(y_offset))

        self._update()

    def process_key_event(self, event: KeyEvent) -> None:
        if event.key == -1:
            return

        print(event)

        # if imgui.is_key_pressed(glfw.KEY_LEFT, repeat=True):
        #     self.move_cursor_horizontally(-1)
        # if imgui.is_key_pressed(glfw.KEY_RIGHT, repeat=True):
        #     self.move_cursor_horizontally(+1)
        # if imgui.is_key_pressed(glfw.KEY_UP, repeat=True):
        #     self.move_cursor_vertically(+1)
        # if imgui.is_key_pressed(glfw.KEY_DOWN, repeat=True):
        #     self.move_cursor_vertically(-1)

        self._update()

    def render(self) -> None:
        cursor_grid_pos = self._line_to_grid.get(self._cursor_line_pos, (-1, -1))

        self._node.uniform_values["u_grid_size"] = self._grid_size
        self._node.uniform_values["u_glyph_size_px"] = self._font.glyph_size_px
        self._node.uniform_values["u_glyph_atlas"] = self._font.atlas_texture
        self._node.uniform_values["u_grid_uvs"] = self._grid_uvs_texture
        self._node.uniform_values["u_grid_metrics"] = self._grid_metrics_texture
        self._node.uniform_values["u_cursor_grid_pos"] = cursor_grid_pos
        self._node.render()

    def _update(self) -> None:
        grid_n_rows = int(
            self._node.canvas.texture.size[1] / self._font.glyph_size_px[1]
        )
        grid_n_cols = int(
            self._node.canvas.texture.size[0] / self._font.glyph_size_px[0]
        )

        data_size = (grid_n_rows, grid_n_cols)
        self._grid_size = (grid_n_cols, grid_n_rows)

        if self._grid_uvs_data is None or self._grid_uvs_data.size != data_size:
            self._grid_uvs_data = np.zeros(
                (grid_n_rows, grid_n_cols, 4), dtype=np.float32
            )
        else:
            self._grid_uvs_data.fill(0.0)

        if self._grid_metrics_data is None or self._grid_metrics_data.size != data_size:
            self._grid_metrics_data = np.zeros(
                (grid_n_rows, grid_n_cols, 4), dtype=np.float32
            )
        else:
            self._grid_metrics_data.fill(0.0)

        i_row = 0
        i_line = self._top_line_idx
        bot_line_idx = self._top_line_idx

        self._line_to_grid.clear()
        self._grid_to_line.clear()

        while i_row < grid_n_rows and i_line < len(self._lines):
            line = self._lines[i_line]

            line_pad = 0
            with contextlib.suppress(StopIteration):
                first_printable_match = next(re.finditer(r"\S", line))
                line_pad = first_printable_match.span()[0]

            i_col = 0
            for i_ch, ch in enumerate(line):
                self._line_to_grid[(i_line, i_ch)] = (i_row, i_col)
                self._grid_to_line[(i_row, i_col)] = (i_line, i_ch)

                ch_ord = ord(ch)
                self._grid_uvs_data[i_row][i_col] = self._font.glyphs[ch_ord]["uv"]
                self._grid_metrics_data[i_row][i_col] = self._font.glyphs[ch_ord][
                    "metrics"
                ]

                i_col += 1

                if i_col >= grid_n_cols:
                    i_col = line_pad
                    i_row += 1

                if i_row >= grid_n_rows:
                    break

            # Handles empty lines and also rightmost cursor position
            self._line_to_grid[(i_line, len(line))] = (i_row, i_col)
            self._grid_to_line[(i_row, i_col)] = (i_line, len(line))

            bot_line_idx = i_line
            i_row += 1
            i_line += 1

        # ----------------------------------------------------------------
        # Ensure cursor visible
        if self._cursor_line_pos[0] < self._top_line_idx:
            self._move_cursor_vertically(self._cursor_line_pos[0] - self._top_line_idx)
        elif self._cursor_line_pos[0] > bot_line_idx:
            self._move_cursor_vertically(self._cursor_line_pos[0] - bot_line_idx)

        # ----------------------------------------------------------------
        # Update textures
        if self._grid_uvs_texture is None:
            self._grid_uvs_texture = self._gl.texture(
                size=self._grid_size,
                components=4,
                data=self._grid_uvs_data,
                dtype="f4",
            )

        if self._grid_uvs_texture.size != self._grid_size:
            self._grid_uvs_texture.release()
            self._grid_uvs_texture = self._gl.texture(
                size=self._grid_size,
                components=4,
                data=self._grid_uvs_data,
                dtype="f4",
            )
        else:
            self._grid_uvs_texture.write(self._grid_uvs_data)

        if self._grid_metrics_texture is None:
            self._grid_metrics_texture = self._gl.texture(
                size=self._grid_size,
                components=4,
                data=self._grid_metrics_data,
                dtype="f4",
            )

        if self._grid_metrics_texture.size != self._grid_size:
            self._grid_metrics_texture.release()
            self._grid_metrics_texture = self._gl.texture(
                size=self._grid_size,
                components=4,
                data=self._grid_metrics_data,
                dtype="f4",
            )
        else:
            self._grid_metrics_texture.write(self._grid_metrics_data)

    def _inc_top_line_idx(self, step: int) -> None:
        if step == 0:
            return

        self._top_line_idx = max(0, min(self._top_line_idx + step, len(self._lines)))

    def _move_cursor_vertically(self, step: int = +1) -> None:
        new_line_idx = max(
            0, min(len(self._lines) - 1, self._cursor_line_pos[0] - step)
        )
        line_length = len(self._lines[new_line_idx])
        new_char_idx = min(line_length, self._desired_cursor_char_idx)

        self._cursor_line_pos = (new_line_idx, new_char_idx)

    def _move_cursor_horizontally(self, step: int = +1) -> None:
        line_length = len(self._lines[self._cursor_line_pos[0]])
        new_char_idx = max(0, min(line_length, self._cursor_line_pos[1] + step))

        self._desired_cursor_char_idx = new_char_idx
        self._cursor_line_pos = (self._cursor_line_pos[0], new_char_idx)
