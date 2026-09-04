"""Per-buffer index cache (078 W-A): an index is rebuilt only when its fingerprint -- the
cheap inputs it is a function of -- changes, and the caller learns whether it did, so the
color feed re-pushes exactly then. An entry belongs to one editor HANDLE: the library's
word-class table is per handle, so a handle recreated for the same path must be fed again,
which a cache keyed on the path alone would skip."""

from collections.abc import Callable, Hashable
from dataclasses import dataclass
from pathlib import Path

from shaderbox.editor.ffi import Editor
from shaderbox.intel.index import GlslIndex


@dataclass(frozen=True)
class _Entry:
    editor: Editor
    fingerprint: Hashable
    index: GlslIndex


class IntelCache:
    def __init__(self) -> None:
        self._entries: dict[Path, _Entry] = {}

    def index_for(
        self,
        key: Path,
        editor: Editor,
        fingerprint: Hashable,
        build: Callable[[], GlslIndex],
    ) -> tuple[GlslIndex, bool]:
        """The index for `key` at `fingerprint` on `editor`, and whether it was (re)built
        this call."""
        entry = self._entries.get(key)
        if (
            entry is not None
            and entry.editor is editor
            and entry.fingerprint == fingerprint
        ):
            return entry.index, False
        index = build()
        self._entries[key] = _Entry(editor, fingerprint, index)
        return index, True

    def drop(self, key: Path) -> None:
        self._entries.pop(key, None)
