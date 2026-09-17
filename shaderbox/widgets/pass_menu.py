"""One pass's context-menu items, shared by every surface that offers them (092 D10).

The item-set function for the PASS kind: the graph node has it, and the deleted pass strip had
it too, so the two could not drift. It lives in its own module because its former home was that
strip -- a surface, where an item set that outlives it should not sit.

The caller owns the popup itself: the canvas anchors it with the previous item (an explicit id
would fire on a right-click anywhere in the shared child), while a surface whose rows are each
their own child window can pass an id safely.
"""

from collections.abc import Callable

from imgui_bundle import imgui

from shaderbox.app import App
from shaderbox.commands import CommandId
from shaderbox.menus import target_menu_item
from shaderbox.pass_graph import PassEntry


def pass_menu_items(
    app: App,
    document_id: str,
    name: str,
    slot: Callable[[], None] | None = None,
) -> None:
    """The items of one pass's context menu, shared by the strip's tile and the graph's node
    (092 D10) so the two surfaces cannot drift. The caller owns the popup itself: the strip
    anchors it with an explicit id (safe there, each tile is its own window), the canvas with
    the previous item.

    `slot` draws a caller's own items after the pass-settings item — the graph node's `Group`,
    which the strip has no selection to seed."""
    document = app.ui_documents[document_id].document
    # The two items that ARE commands read their label and their chord hint from the command
    # table (`command_hint`), so a rebind reaches this menu too. They act on the clicked pass
    # rather than on `panel_pass`, which is why they call the verbs instead of the callbacks.
    if target_menu_item(app, CommandId.OPEN_SHADER):
        app.ensure_shader_tab(document_id, name, focus_editor=True)
    if target_menu_item(app, CommandId.OPEN_PASS_SETTINGS, "Settings"):
        app.open_pass_settings(name)
    if slot is not None:
        slot()
    grouped = bool(document.graph.passes.get(name, PassEntry()).group)
    if grouped:
        imgui.separator()
        if imgui.menu_item_simple("Leave group"):
            app.leave_group(document_id, name)
    imgui.separator()
    # The last pass of a document cannot go: the delete would leave no output to draw. The
    # disabled submenu does not open, so nothing inside it is reachable.
    if imgui.menu_item_simple("Delete", enabled=len(document.passes) > 1):
        app.delete_pass_confirmed(document_id, name)
