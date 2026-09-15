"""The popups layer's one shared type: the `Modal` spec each popup module declares.

It lives in the package root rather than in `registry.py` because every popup module builds
one and the registry imports every popup module -- the two would otherwise cycle. The
registry stays the layer's leaf: it imports `App` and every popup, and nothing imports it
back.
"""

from collections.abc import Callable
from dataclasses import dataclass

from shaderbox.app import App, ModalId
from shaderbox.ui_primitives import ModalSizing


@dataclass(frozen=True)
class Modal:
    """One modal's whole specification.

    `size` is a callable because the Examples browser computes its own from the grid it
    holds; `sizing` says how the window takes it (`ModalSizing`). `on_close` is the
    per-modal cleanup, in ONE place, and `owns_esc` reports that an inline input inside the
    body has claimed Esc for itself.
    """

    id: ModalId
    label: str
    size: Callable[[App], tuple[float, float]]
    body: Callable[[App], bool]
    sizing: ModalSizing = ModalSizing.RESIZABLE
    flags: int = 0
    on_close: Callable[[App], None] | None = None
    owns_esc: Callable[[App], bool] | None = None
