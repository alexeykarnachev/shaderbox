from pathlib import Path

import freetype
import moderngl
import numpy as np

from shaderbox.constants import (
    FONT_ATLAS_PADDING,
    FONT_ATLAS_SIZE,
    PRINTABLE_ASCII_END,
    PRINTABLE_ASCII_START,
    SPACE_CHAR_CODE,
)


class Font:
    def __init__(self, file_path: Path | str, size: int) -> None:
        face = freetype.Face(str(file_path))
        face.set_pixel_sizes(0, size)  # Set font height to 'size' pixels

        texture_width, texture_height = FONT_ATLAS_SIZE
        padding = FONT_ATLAS_PADDING
        current_x, current_y = padding, padding
        max_row_height = 0

        texture_data = np.zeros((texture_height, texture_width, 1), dtype=np.uint8)

        face.load_char("M", freetype.FT_LOAD_RENDER)  # type: ignore
        cell_width = face.glyph.advance.x / 64.0
        cell_height = size

        glyphs = {}

        # Special case for the space
        glyphs[SPACE_CHAR_CODE] = {
            "uv": [0.0, 0.0, 0.0, 0.0],
            "metrics": [face.glyph.advance.x / 64.0 / cell_width, 0.0, 0.0, 0.0],
        }

        # Printable ASCII characters
        for char_code in range(PRINTABLE_ASCII_START, PRINTABLE_ASCII_END):
            face.load_char(char_code, freetype.FT_LOAD_RENDER)  # type: ignore
            bitmap = face.glyph.bitmap

            if bitmap.width == 0 or bitmap.rows == 0:
                continue

            if current_x + bitmap.width + padding > texture_width:
                current_x = padding
                current_y += max_row_height + padding
                max_row_height = 0

            if current_y + bitmap.rows + padding > texture_height:
                raise ValueError("Glyph atlas texture too small")

            bitmap_array = np.array(bitmap.buffer, dtype=np.uint8).reshape(
                bitmap.rows, bitmap.pitch
            )[:, : bitmap.width]
            bitmap_array = np.flipud(bitmap_array)  # Flip for OpenGL

            target_slice = texture_data[
                current_y : current_y + bitmap.rows,
                current_x : current_x + bitmap.width,
            ]
            target_slice[:, :, 0] = bitmap_array

            u0 = current_x / texture_width
            v0 = current_y / texture_height
            u1 = (current_x + bitmap.width) / texture_width
            v1 = (current_y + bitmap.rows) / texture_height

            glyph_width = bitmap.width / cell_width
            glyph_height = bitmap.rows / cell_height
            bearing_x = face.glyph.bitmap_left / cell_width
            bearing_y = (face.glyph.bitmap_top - bitmap.rows) / cell_height

            glyphs[char_code] = {
                "uv": [u0, v0, u1, v1],
                "metrics": [glyph_width, glyph_height, bearing_x, bearing_y],
            }

            current_x += bitmap.width + padding
            max_row_height = max(max_row_height, bitmap.rows)

        min_bearing_y = min(g["metrics"][3] for g in glyphs.values())
        baseline_y = -min_bearing_y if min_bearing_y < 0 else 0
        for char_code in glyphs:
            glyphs[char_code]["metrics"][3] += baseline_y

        gl = moderngl.get_context()
        atlas_texture = gl.texture(
            size=(texture_width, texture_height),
            components=1,
            data=texture_data.tobytes(),
            dtype="f1",
        )
        atlas_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

        self.glyphs = glyphs
        self.glyph_size_px: tuple[float, float] = (cell_width, cell_height)
        self.atlas_texture = atlas_texture

    def release(self) -> None:
        self.atlas_texture.release()
