"""The menu bar and the one primitive every command-bearing menu item goes through (093/17).

The bar is a RENDER of `COMMAND_SPECS`: one top-level menu per `CommandCategory` in
`CATEGORY_ORDER`, one item per spec carrying `in_menu`, in table order, its label the spec's
and its hint the chord currently bound. Nothing here authors a label, so the bar, the palette
and the cheatsheet cannot drift.

This module imports `App`, which is why it is not in `ui_primitives.py` (that layer is
`App`-free by the three-layer rule). `commands.command_label` stays in the leaf.
"""

from imgui_bundle import imgui, imgui_ctx

from shaderbox.app import App
from shaderbox.commands import (
    CATEGORY_ORDER,
    COMMAND_SPECS,
    SPEC_BY_ID,
    CommandId,
    CommandScope,
    CommandSpec,
    chord_to_str,
)
from shaderbox.theme import COLOR, SPACE


def menu_enabled(app: App, spec: CommandSpec) -> bool:
    """Whether `spec`'s menu item accepts a click right now.

    Per ITEM, never per category: a `begin_menu` wrapped in `begin_disabled` does not open at
    all, so a whole greyed category would hide what is in it.
    """
    if spec.scope is CommandScope.EDITOR:
        return app.active_tab is not None
    if spec.scope is CommandScope.COPILOT:
        return app.is_copilot_open
    return True


def command_menu_item(app: App, command_id: CommandId) -> bool:
    """One command as a menu item: its label, its bound chord as the hint, its scope as the
    enabled test. Fires the command's callback on a click and returns whether it fired."""
    spec = SPEC_BY_ID[command_id]
    chord = app.effective_bindings.get(command_id, spec.default_chord)
    hint = chord_to_str(chord) if chord else ""
    fired = imgui.menu_item(spec.label, hint, False, enabled=menu_enabled(app, spec))[0]
    if fired:
        app.command_callbacks[command_id]()
    return fired


def draw_menu_bar(app: App) -> None:
    """The main menu bar: every `in_menu` command under its category, then the open project's
    name right-aligned."""
    with imgui_ctx.begin_menu_bar() as bar:
        if not bar:
            return
        for category in CATEGORY_ORDER:
            specs = [
                spec
                for spec in COMMAND_SPECS
                if spec.category is category and spec.in_menu
            ]
            if not specs:
                continue
            with imgui_ctx.begin_menu(category.value) as menu:
                if not menu:
                    continue
                for spec in specs:
                    if spec.separator_before:
                        imgui.separator()
                    command_menu_item(app, spec.id)
        # The open project's name, right-aligned and dim (084 D8): the one piece of chrome no
        # modal covers, and the only place the app says which project it is in. Text, not a
        # button — the File menu's `Projects` already owns the click.
        label = f"project {app.project_dir.name}"
        # SPACE.LG off the right edge: flush against it, the last glyph touches the window
        # border.
        imgui.same_line(
            imgui.get_content_region_avail().x
            - imgui.calc_text_size(label).x
            + imgui.get_cursor_pos_x()
            - float(SPACE.LG)
        )
        imgui.text_colored(COLOR.FG_DIM, label)
