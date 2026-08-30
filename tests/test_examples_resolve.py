"""Examples <-> shipped-lib coherence (feature 051). Both tests are GL-free and run against the
SEED lib (`resources/shader_lib/`), never the user's live root — the live root can carry helpers
that never shipped (the one-way-seed divergence), and an example must resolve on a FRESH install.

`test_shader_lib_api_lock` pins `{name: signature}` of every shipped `SB_*` entry; changing,
removing, or adding one fails until the snapshot is deliberately regenerated:

    uv run python -c "import json; from shaderbox.constants import SHADER_LIB_SEED_DIR; \\
from shaderbox.shader_lib import ShaderLibIndex; \\
i = ShaderLibIndex.build(SHADER_LIB_SEED_DIR); \\
open('tests/shader_lib_api_lock.json', 'w').write(json.dumps( \\
{n: f.signature for n, f in sorted(i.functions.items())}, indent=2) + '\\n')"

Shipped `SB_*` is supersede-don't-mutate (conventions.md): regenerating over a signature CHANGE
needs a conscious reason; regenerating over an ADDITION is routine.
"""

import json
from pathlib import Path

from shaderbox.constants import DOCUMENT_EXAMPLES_DIR, SHADER_LIB_SEED_DIR
from shaderbox.shader_lib import ShaderLibIndex
from shaderbox.shader_lib.resolver import resolve_usage
from shaderbox.shader_source import ShaderSource

_LOCK_FILE = Path(__file__).parent / "shader_lib_api_lock.json"


def test_examples_resolve_clean() -> None:
    index = ShaderLibIndex.build(SHADER_LIB_SEED_DIR)
    assert index.functions, "seed lib index is empty — wrong root?"
    shaders = sorted(DOCUMENT_EXAMPLES_DIR.glob("*/passes/*.frag.glsl"))
    assert shaders, "no shipped example shaders found"
    for path in shaders:
        _, _, _, errors = resolve_usage(ShaderSource.load(path), index)
        assert not errors, f"{path.parent.name}: {[e.message for e in errors]}"


def test_shader_lib_api_lock() -> None:
    index = ShaderLibIndex.build(SHADER_LIB_SEED_DIR)
    live = {name: fn.signature for name, fn in sorted(index.functions.items())}
    locked = json.loads(_LOCK_FILE.read_text(encoding="utf-8"))
    assert live == locked, (
        "shipped SB_* API drifted from tests/shader_lib_api_lock.json — if the change is "
        "deliberate (supersede-don't-mutate!), regenerate the snapshot per this module's "
        "docstring and fix any example the change breaks"
    )
