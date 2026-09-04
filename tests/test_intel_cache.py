"""The per-handle index cache (078 W-A): an entry belongs to the editor handle it was fed
into, so a handle recreated for the same path rebuilds and is fed again."""

from pathlib import Path

from shaderbox.editor.ffi import Editor
from shaderbox.intel.document import IntelCache
from shaderbox.intel.index import GlslIndex


def _index() -> GlslIndex:
    return GlslIndex(symbols={}, declarations=(), words=())


def test_the_cache_entry_belongs_to_its_handle() -> None:
    cache = IntelCache()
    first = Editor("")
    second = Editor("")
    path = Path("x.glsl")
    _, changed = cache.index_for(path, first, 1, _index)
    assert changed
    _, changed = cache.index_for(path, first, 1, _index)
    assert not changed, "same handle, same fingerprint: a hit"
    _, changed = cache.index_for(path, second, 1, _index)
    assert changed, (
        "a new handle at the same path must rebuild, its class table is empty"
    )
    _, changed = cache.index_for(path, second, 2, _index)
    assert changed, "a moved fingerprint rebuilds"
    cache.drop(path)
    _, changed = cache.index_for(path, second, 2, _index)
    assert changed, "a dropped entry rebuilds"
    first.close()
    second.close()
