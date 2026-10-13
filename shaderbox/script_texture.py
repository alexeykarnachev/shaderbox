"""What a script may return for a `sampler2D`, decided once (feature 110).

The engine refuses with it, the draw allocates from it and completion offers a declaration
from it, so all three agree by construction. GL-free: the script engine and the intel
package both import it.
"""

import numpy as np

# The GL_MAX_TEXTURE_SIZE every OpenGL 4.x context guarantees. A constant, because the
# engine that refuses a field never holds a context to ask.
MAX_SCRIPT_TEXTURE_EXTENT = 16384

# numpy dtype -> moderngl texture dtype. `uint8` is NORMALIZED (`f1`, 255 reads 1.0):
# moderngl's `u1` is an integer texture, which a `sampler2D` cannot read.
TEXTURE_DTYPES: dict[np.dtype, str] = {
    np.dtype(np.float32): "f4",
    np.dtype(np.uint8): "f1",
}

# moderngl texture dtype -> the numpy name a script wrote, for a caption.
TEXTURE_DTYPE_NAMES: dict[str, str] = {
    moderngl_dtype: str(numpy_dtype)
    for numpy_dtype, moderngl_dtype in TEXTURE_DTYPES.items()
}

# A field the engine refused this tick, as distinct from one the script did not send: an
# absent field binds the sampler's own source, a refused one holds the last frame that
# validated. Compared by IDENTITY.
REFUSED_TEXTURE: np.ndarray = np.empty(0, dtype=np.float32)


def texture_problem(value: object) -> str | None:
    """Why `value` cannot be a sampler's texture, or None when it can."""
    if not isinstance(value, np.ndarray):
        return (
            f"a sampler2D takes a numpy array of shape (H, W) or (H, W, C), "
            f"got {type(value).__name__}"
        )
    if value.ndim not in (2, 3):
        return f"a sampler2D takes shape (H, W) or (H, W, C), got shape {value.shape}"
    if value.ndim == 3 and not 1 <= value.shape[2] <= 4:
        return f"a sampler2D takes 1 to 4 channels, got {value.shape[2]}"
    height, width = value.shape[0], value.shape[1]
    if not (
        1 <= height <= MAX_SCRIPT_TEXTURE_EXTENT
        and 1 <= width <= MAX_SCRIPT_TEXTURE_EXTENT
    ):
        return (
            f"a sampler2D field is 1 to {MAX_SCRIPT_TEXTURE_EXTENT} texels on each side, "
            f"got {height}x{width}"
        )
    if value.dtype not in TEXTURE_DTYPES:
        return (
            f"a sampler2D takes float32 or uint8, got {value.dtype} -- "
            f"use .astype(np.float32)"
        )
    if not value.flags.c_contiguous:
        return "a sampler2D field must be C-contiguous -- use np.ascontiguousarray(...)"
    return None


def texture_layout(field: np.ndarray) -> tuple[tuple[int, int], int, str]:
    """The (width, height), channel count and moderngl dtype of an accepted field."""
    height, width = field.shape[0], field.shape[1]
    channels = field.shape[2] if field.ndim == 3 else 1
    return (width, height), channels, TEXTURE_DTYPES[field.dtype]
