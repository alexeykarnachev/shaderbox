from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ShaderSource:
    path: Path
    text: str
    mtime: float

    @classmethod
    def load(cls, path: Path) -> "ShaderSource":
        return cls(
            path=path,
            text=path.read_text(encoding="utf-8"),
            mtime=path.lstat().st_mtime if path.exists() else 0.0,
        )
