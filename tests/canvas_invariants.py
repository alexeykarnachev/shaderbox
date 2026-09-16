"""The post-condition every canvas reachable from a Document must satisfy (096 W-0).

Six canvas defects in one week were one class: a canvas's configuration -- size, dtype, filter,
wrap -- is decided in several places, and each place knows about some canvases but not all. This
is the check that decides the class, written as a helper so the operation battery in
`test_canvas_ownership.py` can assert it after every state-changing call.

Two rules the callers must keep, both learned the hard way:

- **Assert BEFORE any render.** `Document.render` fixes up a non-output pass's size lazily, so a
  test that renders first passes whether or not the operation under test did anything.
- **Drive it over the NON-DEFAULT corner.** `DEFAULT_FILTER_LINEAR` is True and `DEFAULT_WRAP` is
  False, so a check built on defaults cannot fail. `NON_DEFAULT` below is that corner.
"""

import moderngl

from shaderbox.core import Canvas
from shaderbox.document import Document
from shaderbox.pass_graph import PassEntry, TargetConfig

# The corner every default is the opposite of: NEAREST against a LINEAR default, wrap against
# clamp, f4 against f2, a half-size scale against 1.0.
NON_DEFAULT = TargetConfig(scale=0.5, dtype="f4", filter_linear=False, wrap=True)

_NEAREST = (moderngl.NEAREST, moderngl.NEAREST)
_LINEAR = (moderngl.LINEAR, moderngl.LINEAR)


def filter_of(target: TargetConfig) -> tuple[int, int]:
    return _LINEAR if target.filter_linear else _NEAREST


def _config_of(canvas: Canvas) -> tuple[tuple[int, int], str, tuple[int, int], bool]:
    return (canvas.texture.size, canvas.dtype, canvas.filter, canvas.wrap)


def _implied(
    document: Document, name: str
) -> tuple[tuple[int, int], str, tuple[int, int], bool]:
    """What `name`'s graph entry says its canvas must be. The output is sized to the DOCUMENT,
    never to its own scale -- the output pass ignores `scale` by decision."""
    entry = document.graph.passes.get(name, PassEntry())
    output = document.graph.output_pass
    size = (
        document.canvas_size
        if output is not None and name == output
        else entry.target.target_size(document.canvas_size)
    )
    return (size, entry.target.dtype, filter_of(entry.target), entry.target.wrap)


def canvas_violations(document: Document) -> list[str]:
    """Every way `document`'s canvases disagree with what its graph implies, as readable lines.

    A list rather than an assert so a caller can report all of them at once: the defects this
    exists for arrive in groups, and a checker that stops at the first one hides the rest.
    """
    violations: list[str] = []
    for name, render_pass in document.passes.items():
        live = _config_of(render_pass.canvas)
        implied = _implied(document, name)
        if live != implied:
            violations.append(
                f"pass '{name}': canvas {live} but graph implies {implied}"
            )
        history = document._feedback.get(name)
        if history is not None and _config_of(history) != live:
            violations.append(
                f"pass '{name}': history {_config_of(history)} but live canvas {live}"
            )
    return violations


def assert_canvases_agree(document: Document) -> None:
    violations = canvas_violations(document)
    assert not violations, (
        "canvas configuration disagrees with the graph:\n" + "\n".join(violations)
    )


def assert_blit_filter_follows(blit_canvas: Canvas, source: moderngl.Texture) -> None:
    """A channel blit's canvas carries the filter of the texture it shows.

    The viewer magnifies the BLIT's texture, so a blit that keeps its own default makes the
    output pass's `smooth` setting invisible in every view but COLOR.
    """
    assert blit_canvas.texture.filter == source.filter, (
        f"blit filter {blit_canvas.texture.filter} does not follow source {source.filter}"
    )
