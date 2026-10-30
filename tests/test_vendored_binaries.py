"""Both platform builds of each vendored library export the same API.

Read from the files, so a Linux host catches a `.dll` left behind by a rebuild that
replaced only the `.so`. Export NAMES only: two builds of different commits that kept
every name pass here, and the per-platform binding tests are what load and drive them.
"""

import re

import pytest

from shaderbox.constants import RESOURCES_DIR
from tests._exports import exported_functions

_LIBRARIES = [
    ("editor/libeditor", "ed_"),
    ("graph_canvas/libgraph_canvas", "gc_"),
]


@pytest.mark.parametrize(("stem", "prefix"), _LIBRARIES)
def test_the_linux_and_windows_builds_export_the_same_api(
    stem: str, prefix: str
) -> None:
    api = re.compile(rf"^{prefix}\w+$")
    so, dll = (
        {
            name
            for name in exported_functions(RESOURCES_DIR / f"{stem}{ext}")
            if api.match(name)
        }
        for ext in (".so", ".dll")
    )
    assert so, f"{stem}.so exports nothing named {prefix}*"
    assert so == dll, (
        f"only in the .so: {sorted(so - dll)}; only in the .dll: {sorted(dll - so)}"
    )
