"""The one read-only object every behavior's `update` receives (feature 041)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EngineContext:
    # The clock — all v1 carries. mouse is feature 042 (system input); state lives in the
    # behavior instance (self.*), not here. `Ctx` is the name in scope inside a script.
    t: float
    dt: float
    frame: int


Ctx = EngineContext
