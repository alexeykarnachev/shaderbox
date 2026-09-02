"""Geometry the Document tab and the pass-settings gear depend on (069 W-B).

Both facts here are measurable only inside a real imgui frame with the app font loaded,
which is why they live beside the frame rig rather than in the AST-only prose gate.

`always_auto_resize` makes imgui size the window to its content every frame and IGNORE
`set_next_window_size`, so the width token only holds through a size CONSTRAINT. Both
facts are geometry a headless frame can measure, and both were wrong in the first
implementation: the width silently became 323 while the token said 440.
"""

from typing import Any

from imgui_bundle import imgui

from shaderbox.core import ENGINE_DRIVEN_UNIFORMS
from shaderbox.glyph_tables import TABLE_UNIFORMS
from shaderbox.popups import pass_settings
from shaderbox.theme import SIZE
from shaderbox.ui_primitives import _ellipsize


def _gear_sizes(app: Any, frames: int = 6) -> list[tuple[float, float]]:
    """Open the gear on the document's first pass and record its rect per frame."""
    document_id = app.current_document_id
    name = sorted(app.ui_documents[document_id].document.passes)[0]
    app.open_pass_settings(name)

    sizes: list[tuple[float, float]] = []
    original = pass_settings._draw_body

    def measured(inner: Any) -> bool:
        keep: bool = original(inner)
        size = imgui.get_window_size()
        sizes.append((round(size.x, 1), round(size.y, 1)))
        return keep

    pass_settings._draw_body = measured
    try:
        for _ in range(frames):
            imgui.new_frame()
            pass_settings.draw_pass_settings(app)
            imgui.end_frame()
    finally:
        pass_settings._draw_body = original
    return sizes


def test_the_gear_keeps_its_width_token_under_auto_resize(app: Any) -> None:
    # Falsifier: drop the `set_next_window_size_constraints` call and the settled width is
    # 323 — auto-resize discards the seeded size, so the token would be inert and the next
    # reader would change 440 and see nothing move.
    sizes = _gear_sizes(app)
    settled = sizes[-1]
    assert settled[0] == float(SIZE.PASS_SETTINGS_W), (
        f"the gear settled at {settled[0]}px wide, not the {SIZE.PASS_SETTINGS_W} token; "
        "always_auto_resize ignores set_next_window_size, so the constraint is what pins it."
    )


def test_the_gear_height_follows_its_content(app: Any) -> None:
    # The height is NOT the width's fixed token: it is whatever the body needs, which is
    # what finding #7 asked for (a settings popup that scrolls is itself the defect).
    sizes = _gear_sizes(app)
    settled = sizes[-1]
    assert settled[1] != float(SIZE.PASS_SETTINGS_W)
    assert 0.0 < settled[1] <= imgui.get_io().display_size.y


def test_the_auto_name_column_fits_every_engine_uniform(app: Any) -> None:
    """Every engine uniform's name renders inside `SIZE.AUTO_NAME_W` without ellipsis.

    The Document tab draws these through `clickable_label`, which ELLIPSIZES rather than
    overflowing, so an over-wide name is a SILENT truncation. Measured against the real
    rasterized face inside a frame, never a hard-coded em ratio: the first version of this
    check assumed 6.5508px per character where the 12px face advances 7.0, so it passed a
    19-character name that visibly truncates.

    Falsifier: add a 20-character name to `ENGINE_DRIVEN_UNIFORMS` and this goes red.
    """
    names = sorted(set(ENGINE_DRIVEN_UNIFORMS) - set(TABLE_UNIFORMS))
    assert names, "the engine-uniform set is empty; the walk found nothing to measure"

    widths: dict[str, float] = {}
    kept: dict[str, str] = {}
    imgui.new_frame()
    imgui.begin("rig")
    imgui.push_font(app.font_12, app.font_12.legacy_size)
    for name in names:
        widths[name] = imgui.calc_text_size(name).x
        kept[name] = _ellipsize(name, float(SIZE.AUTO_NAME_W))
    imgui.pop_font()
    imgui.end()
    imgui.end_frame()

    for name in names:
        assert widths[name] <= float(SIZE.AUTO_NAME_W), (
            f"{name} renders {widths[name]}px wide against a {SIZE.AUTO_NAME_W}px column"
        )
        assert kept[name] == name, (
            f"{name} is ellipsized to {kept[name]!r} in the block's name column"
        )
