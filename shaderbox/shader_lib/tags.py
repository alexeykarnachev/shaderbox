"""Cross-project tags for library functions.

A `function_name -> set[tag]` map persisted to `<app_data_dir>/shader_lib_tags.json`
(same posture as `shader_lib_favorites.json` and `integrations.json` — cross-project
metadata that lives outside any user project).

Tags are metadata, not intrinsic to the function. `ShaderLibFunction` carries the
function's source-derived fields (signature/body/doc/calls); a separate store
carries the user-applied categorization. Lookups go `app.shader_lib_tags.tags_for(name)`.
"""

import json
from dataclasses import dataclass, field
from typing import Self

from loguru import logger

from shaderbox.paths import app_data_dir

_STORE_FILE = "shader_lib_tags.json"


@dataclass
class ShaderLibTagsStore:
    tags: dict[str, set[str]] = field(default_factory=dict)

    def tags_for(self, name: str) -> frozenset[str]:
        return frozenset(self.tags.get(name, set()))

    def has_tag(self, name: str, tag: str) -> bool:
        return tag in self.tags.get(name, set())

    def add(self, name: str, tag: str) -> None:
        tag = tag.strip().lstrip("#").lower()
        if not tag:
            return
        self.tags.setdefault(name, set()).add(tag)
        self.save()

    def remove(self, name: str, tag: str) -> None:
        if name in self.tags and tag in self.tags[name]:
            self.tags[name].remove(tag)
            if not self.tags[name]:
                del self.tags[name]
            self.save()

    def toggle(self, name: str, tag: str) -> None:
        if self.has_tag(name, tag):
            self.remove(name, tag)
        else:
            self.add(name, tag)

    def all_tags(self) -> frozenset[str]:
        # Union across every function — what the global tag bar shows.
        out: set[str] = set()
        for tag_set in self.tags.values():
            out.update(tag_set)
        return frozenset(out)

    def save(self) -> None:
        path = app_data_dir() / _STORE_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            # Serialize as sorted lists for stable diffs.
            data = {name: sorted(tags) for name, tags in sorted(self.tags.items())}
            json.dump(data, f, indent=4)

    @classmethod
    def load(cls) -> Self:
        path = app_data_dir() / _STORE_FILE
        if not path.exists():
            return cls()
        try:
            with path.open() as f:
                raw = json.load(f)
            return cls(tags={name: set(tags) for name, tags in raw.items()})
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Unreadable {_STORE_FILE} ({e}); falling back to empty")
            return cls()
