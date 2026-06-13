"""The one object every behavior script receives — the read-only world (feature 040)."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EngineContext:
    # The clock is the only field v1 populates (live: glfw.get_time; export: i/fps).
    # mouse/state/uniforms are RESERVED for 041 (system input) / 042 (state scripts) —
    # present so the Behavior protocol is stable across those features, left empty in v1.
    t: float
    dt: float
    frame: int
    mouse: tuple[float, float] = (0.0, 0.0)
    state: dict[str, Any] = field(default_factory=dict)
    uniforms: dict[str, Any] = field(default_factory=dict)
