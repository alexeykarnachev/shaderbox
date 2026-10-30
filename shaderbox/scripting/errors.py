"""Script failure as data — mirrors `shader_errors.ShaderError`. A broken
script never raises into the frame loop; it freezes the uniform at last-good and records
one of these for the UI to surface."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class ScriptError:
    uniform_name: str
    kind: Literal["compile", "runtime"]
    message: str
    # 0-based line in the user's script, matching the editor; -1 when unmappable.
    line: int = -1
    # The pass whose uniform this error is about; "" for a document-level error (the sentinel)
    # and for a bare key no pass declares.
    pass_name: str = ""
