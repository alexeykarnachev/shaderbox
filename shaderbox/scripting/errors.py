"""Script failure as data — mirrors `shader_errors.ShaderError` (feature 040). A broken
script never raises into the frame loop; it freezes the uniform at last-good and records
one of these for 041's UI to surface."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class ScriptError:
    uniform_name: str
    kind: Literal["compile", "runtime"]
    message: str
    line: int = -1
