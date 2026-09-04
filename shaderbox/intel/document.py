"""Per-buffer index cache (078 W-A): an index is rebuilt only when its fingerprint -- the
cheap inputs it is a function of -- changes, and the caller learns whether it did, so the
color feed re-pushes exactly then."""

from collections.abc import Callable, Hashable
from pathlib import Path

from shaderbox.intel.index import GlslIndex


class IntelCache:
    def __init__(self) -> None:
        self._entries: dict[Path, tuple[Hashable, GlslIndex]] = {}

    def index_for(
        self, key: Path, fingerprint: Hashable, build: Callable[[], GlslIndex]
    ) -> tuple[GlslIndex, bool]:
        """The index for `key` at `fingerprint`, and whether it was (re)built this call."""
        entry = self._entries.get(key)
        if entry is not None and entry[0] == fingerprint:
            return entry[1], False
        index = build()
        self._entries[key] = (fingerprint, index)
        return index, True

    def drop(self, key: Path) -> None:
        self._entries.pop(key, None)
