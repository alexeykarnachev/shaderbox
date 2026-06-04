"""User-editable overrides for node-template descriptions (feature 020·22).

A shipped template's description lives in its `node.json` (read-only in a bundle). When the user edits
a description in the node-creator popup, the edit is stored HERE, keyed by the template's full dir-uuid,
so it survives app updates (the sidecar at `<app_data_dir>` is untouched by a new bundle). Lookup
precedence (applied at the consumption site, never by mutating the in-memory template): this override if
present, else the shipped `ui_state.description`. Cross-project — templates are global, not per-project
(same posture as the shader-lib favorites/tags stores + integrations).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

from loguru import logger

from shaderbox.paths import app_data_dir

_STORE_FILE = "template_descriptions.json"


@dataclass
class TemplateDescriptionsStore:
    # {full_template_uuid: description}. Keyed by the FULL uuid (stable) — short ids are display-only.
    descriptions: dict[str, str] = field(default_factory=dict)

    @property
    def file_path(self) -> Path:
        return app_data_dir() / _STORE_FILE

    def get(self, template_uuid: str) -> str | None:
        # The user override, or None if the user never edited this template (caller falls back to the
        # shipped default).
        return self.descriptions.get(template_uuid)

    def set(self, template_uuid: str, description: str) -> None:
        # No-op if unchanged — the popup persists ON CHANGE every frame the editor is focused, so a
        # value check keeps a held-focus frame from rewriting the same JSON 60x/sec (feature 020·22).
        if self.descriptions.get(template_uuid) == description:
            return
        self.descriptions[template_uuid] = description
        self.save()

    def save(self) -> None:
        path = self.file_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump({"descriptions": self.descriptions}, f, indent=4)

    @classmethod
    def load(cls) -> Self:
        path = app_data_dir() / _STORE_FILE
        if not path.exists():
            return cls()
        try:
            with path.open() as f:
                data = json.load(f)
            raw = data.get("descriptions", {})
            descriptions = {str(k): str(v) for k, v in raw.items()}
            return cls(descriptions=descriptions)
        except (OSError, json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Unreadable {_STORE_FILE} ({e}); falling back to empty")
            return cls()
