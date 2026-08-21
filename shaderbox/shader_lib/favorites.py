"""Cross-project favorites for library functions.

A flat set of function names the user has starred in the shader-library picker.
Persisted to `<app_data_dir>/shader_lib_favorites.json` (same posture as the
shader library itself + the integrations store — cross-project, lives outside
any user project).
"""

import json
from dataclasses import dataclass, field
from typing import Self

from loguru import logger

from shaderbox.paths import app_data_dir

_STORE_FILE = "shader_lib_favorites.json"


@dataclass
class ShaderLibFavoritesStore:
    favorites: set[str] = field(default_factory=set)

    @property
    def file_path(self) -> "object":
        return app_data_dir() / _STORE_FILE

    def is_favorite(self, name: str) -> bool:
        return name in self.favorites

    def toggle(self, name: str) -> None:
        if name in self.favorites:
            self.favorites.remove(name)
        else:
            self.favorites.add(name)
        self.save()

    def save(self) -> None:
        path = app_data_dir() / _STORE_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump({"favorites": sorted(self.favorites)}, f, indent=4)

    @classmethod
    def load(cls) -> Self:
        path = app_data_dir() / _STORE_FILE
        if not path.exists():
            return cls()
        try:
            with path.open() as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Unreadable {_STORE_FILE} ({e}); falling back to empty")
            return cls()

        # Salvage PER ENTRY — same reasoning as the tags store: no backup, and loaded
        # unguarded at startup, so a malformed entry must not take the app down with it.
        entries = data.get("favorites") if isinstance(data, dict) else None
        if not isinstance(entries, list):
            if entries is not None:
                logger.warning(f"Malformed {_STORE_FILE}; falling back to empty")
            return cls()
        return cls(favorites={e for e in entries if isinstance(e, str) and e.strip()})
