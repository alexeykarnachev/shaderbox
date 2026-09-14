"""The popups layer's one shared type: the `Modal` spec each popup module declares.

It lives in the package root rather than in `registry.py` because every popup module builds
one and the registry imports every popup module -- the two would otherwise cycle. The
registry stays the layer's leaf: it imports `App` and every popup, and nothing imports it
back.
"""

from collections.abc import Callable
from dataclasses import dataclass

from shaderbox.app import App, ModalId


@dataclass(frozen=True)
class Modal:
    """One modal's whole specification.

    `size` is a callable because the Examples browser computes its own from the grid it
    holds. `before` runs outside the popup scope (the pass-settings size constraints must
    precede `begin_popup_modal`). `on_close` is the per-modal cleanup, in ONE place, and
    `owns_esc` reports that an inline input inside the body has claimed Esc for itself.
    """

    id: ModalId
    label: str
    size: Callable[[App], tuple[float, float]]
    body: Callable[[App], bool]
    flags: int = 0
    fixed_size: bool = False
    before: Callable[[App], None] | None = None
    on_close: Callable[[App], None] | None = None
    owns_esc: Callable[[App], bool] | None = None
