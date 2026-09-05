"""A generated file still matches its generator (079 final review).

`scripts/gen_glyphs.py` writes two committed artifacts. Nothing re-ran it after the 065
`Node` -> `Pass` rename, so the shipped `glyphs.glsl` kept describing `Node.compile()` for
several features — and it is library text the copilot reads, not a comment nobody sees. The
generator is the source of truth; this asserts the committed bytes are what it produces.

Import-and-call rather than subprocess: the generator's `main()` WRITES, and a test that
writes into the repo is a test that can corrupt it on a failure.
"""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))

import gen_glyphs  # noqa: E402


@pytest.mark.parametrize(
    ("artifact", "render"),
    [
        ("shaderbox/resources/shader_lib/text/glyphs.glsl", gen_glyphs.generate),
        ("shaderbox/glyph_tables.py", gen_glyphs.generate_tables_py),
    ],
    ids=("glyphs.glsl", "glyph_tables.py"),
)
def test_a_generated_artifact_matches_its_generator(
    artifact: str, render: object
) -> None:
    # Falsifier: edit either committed file by hand, or change the generator without re-running
    # it, and this names the file and the command that fixes it.
    assert callable(render)
    path = _ROOT / artifact
    assert path.read_text(encoding="utf-8") == render(), (
        f"{artifact} has drifted from scripts/gen_glyphs.py — "
        "run `uv run python scripts/gen_glyphs.py` and commit the result"
    )
