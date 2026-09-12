"""The import dialog (091 D10): another document's passes, copied into this one as a group.

One modal in the `PopupState` mutex. Pick a source on one of two tabs (the project's other
documents, the shipped examples), name the group (it also prefixes the passes), and decide
each ENTRY POINT of the source -- a pass that reads no other pass of it: kept as it is, or fed
by one of this document's passes, whose own readers are then handed to the bundle's output
(D6). The plan is recomputed every frame the body draws, so the button and its message never
lag the field.

No field takes keyboard focus on open or on selection: a focused `input_text` writes its own
buffer back over an external write on the next frame, which would defeat any programmatic
prefill.
"""

from imgui_bundle import imgui

from shaderbox.app import App, PopupState
from shaderbox.document import document_dir_of, offered_entry_points
from shaderbox.pass_import import ImportPlan, plan_import
from shaderbox.paths import DOCUMENT_SCRIPT_BASENAME, SCRIPTS_DIR_NAME
from shaderbox.theme import COLOR, SIZE, SPACE
from shaderbox.ui_models import ImportDraft, UIDocument
from shaderbox.ui_primitives import (
    caption_text,
    help_marker,
    label_row,
    modal_window,
    primary_button,
    standard_button,
)
from shaderbox.widgets.document_grid import draw_document_preview_button

_LABEL = "Import passes##popup"
_POPUP_W = 720.0
_POPUP_H = 600.0
_GRID_COLS = 4
_GRID_ROWS = 2
_ROW_LABEL_W = 110.0
_CTRL_W = 168.0
_COMBO_W = 200.0
_KEEP = "theirs"


def draw_import_passes(app: App) -> None:
    if app.popup_state != PopupState.IMPORT_PASSES:
        return
    with modal_window(_LABEL, (_POPUP_W, _POPUP_H)) as visible:
        if not visible:
            return
        if not _draw_body(app):
            app.close_import_passes()
            imgui.close_current_popup()


def _draw_body(app: App) -> bool:
    draft = app.import_draft
    host = app.ui_documents.get(app.current_document_id)
    if draft is None or host is None:
        return False

    _draw_tabs(app, draft)
    sources = app.import_sources(draft.examples_tab)
    _draw_grid(app, draft, sources)
    source = sources.get(draft.source_id)

    imgui.dummy((0.0, float(SPACE.SM)))
    plan: ImportPlan | str = "pick a document"
    if source is None:
        caption_text("Pick a document")
    else:
        _draw_description(source)
        imgui.dummy((0.0, float(SPACE.SM)))
        label_row(app.font_12, "group", _CTRL_W, _ROW_LABEL_W)
        _, draft.group_buf = imgui.input_text("##import_group", draft.group_buf)
        imgui.same_line()
        help_marker("marks the tiles and prefixes the passes")
        plan = _plan(draft, source, host)
        _draw_entry_points(app, draft, source, host, plan)
    draft.rejection = plan if isinstance(plan, str) else ""

    imgui.dummy((0.0, float(SPACE.MD)))
    keep_open = True
    imgui.begin_disabled(bool(draft.rejection))
    label = (
        f"Import {len(plan.renames)} passes"
        if isinstance(plan, ImportPlan)
        else "Import"
    )
    if primary_button(label) and app.import_passes_from_draft():
        keep_open = False
    imgui.end_disabled()
    imgui.same_line()
    if standard_button("Cancel"):
        keep_open = False
    if draft.rejection and source is not None:
        imgui.same_line()
        imgui.text_colored(COLOR.STATE_ERROR, draft.rejection)
    return keep_open


def _draw_tabs(app: App, draft: ImportDraft) -> None:
    # imgui owns the selection and takes a programmatic one only through `set_selected`, a
    # frame after the request; until then the read-back would write the old tab straight
    # back over the draft (/imgui-ui §8).
    pending = draft.tab_select_pending
    if imgui.begin_tab_bar("##import_tabs"):
        for label, examples_tab in (("This project", False), ("Examples", True)):
            flags = (
                imgui.TabItemFlags_.set_selected
                if pending and examples_tab == draft.examples_tab
                else imgui.TabItemFlags_.none
            )
            if imgui.begin_tab_item(label, flags=flags)[0]:
                if not pending and draft.examples_tab != examples_tab:
                    app.select_import_source("", examples_tab)
                imgui.end_tab_item()
        imgui.end_tab_bar()
    draft.tab_select_pending = False


def _draw_grid(app: App, draft: ImportDraft, sources: dict[str, UIDocument]) -> None:
    style = imgui.get_style()
    cell_h = float(SIZE.THUMB_LG) + imgui.get_text_line_height_with_spacing()
    grid_h = (
        _GRID_ROWS * cell_h
        + (_GRID_ROWS - 1) * style.item_spacing.y
        + 2.0 * style.window_padding.y
    )
    if imgui.begin_child("##import_grid", size=(0.0, grid_h)):
        if not sources:
            caption_text("No other documents")
        for i, (source_id, ui_document) in enumerate(sources.items()):
            border = COLOR.SELECT if source_id == draft.source_id else None
            result = draw_document_preview_button(
                ui_document,
                border,
                float(SIZE.THUMB_LG),
                stale=not ui_document.document.first_render_done,
            )
            if result.clicked:
                app.select_import_source(source_id, draft.examples_tab)
            if (i + 1) % _GRID_COLS != 0 and i != len(sources) - 1:
                imgui.same_line()
            else:
                imgui.spacing()
    imgui.end_child()


def _draw_description(source: UIDocument) -> None:
    description = source.ui_state.description
    if description:
        imgui.push_text_wrap_pos(0.0)
        imgui.text_colored(COLOR.FG_DIM, description)
        imgui.pop_text_wrap_pos()


def _plan(draft: ImportDraft, source: UIDocument, host: UIDocument) -> ImportPlan | str:
    return plan_import(
        source.document.effective_wiring(),
        source.document.graph.output_pass or "",
        draft.group_buf.strip(),
        draft.substitutions,
        draft.handovers,
        host.document.effective_wiring(),
        host.document.graph.output,
    )


def _draw_entry_points(
    app: App,
    draft: ImportDraft,
    source: UIDocument,
    host: UIDocument,
    plan: ImportPlan | str,
) -> None:
    document = source.document
    wiring = document.effective_wiring()
    output = document.graph.output_pass or ""
    broken = {name for name, p in document.passes.items() if p.program is None}
    roots = offered_entry_points(document)
    host_passes = sorted(host.document.passes)
    host_output = host.document.graph.output

    imgui.separator_text("Entry points")
    for root in roots:
        fed = draft.substitutions.get(root, "")
        readers = sorted(
            name for name, reads in wiring.items() if root in reads.values()
        )
        imgui.align_text_to_frame_padding()
        imgui.text_colored(COLOR.FG_SECONDARY, root)
        imgui.same_line(_ROW_LABEL_W + float(SPACE.MD))
        imgui.set_next_item_width(_COMBO_W)
        is_output = root == output
        imgui.begin_disabled(is_output)
        preview = f"{fed} (mine)" if fed else _KEEP
        if imgui.begin_combo(f"##entry_{root}", preview):
            if imgui.selectable(_KEEP, not fed)[0]:
                app.set_import_substitution(root, "")
            for name in host_passes:
                if imgui.selectable(f"{name} (mine)", name == fed)[0]:
                    app.set_import_substitution(root, name)
            imgui.end_combo()
        imgui.end_disabled()
        imgui.same_line()
        if is_output:
            caption_text("the output, stays")
        elif readers:
            caption_text(f"readers: {', '.join(readers)}")
        if fed:
            _draw_handovers(app, draft, fed, host_output, plan)

    bundle_output = plan.output if isinstance(plan, ImportPlan) else output
    imgui.dummy((0.0, float(SPACE.SM)))
    caption_text(f"output: {bundle_output}")
    if isinstance(plan, ImportPlan) and plan.becomes_output:
        caption_text(f"{bundle_output} becomes the output")
    for name in sorted(broken):
        imgui.text_colored(COLOR.STATE_WARN, f"{name} does not compile; copied as is")
    if _has_script(source):
        caption_text("script not imported")


def _draw_handovers(
    app: App,
    draft: ImportDraft,
    fed: str,
    host_output: str,
    plan: ImportPlan | str,
) -> None:
    readers = sorted(app.host_readers_of(fed))
    bundle_output = plan.output if isinstance(plan, ImportPlan) else "the bundle"
    imgui.indent(_ROW_LABEL_W + float(SPACE.MD))
    if readers:
        caption_text(f"{fed} -> {bundle_output}:")
        for pair in readers:
            imgui.same_line()
            on = pair in draft.handovers
            changed, on = imgui.checkbox(f"{pair[0]}.{pair[1]}##handover", on)
            if changed:
                if on:
                    draft.handovers.add(pair)
                else:
                    draft.handovers.discard(pair)
    elif fed != host_output:
        caption_text(f"nothing reads {fed}")
    imgui.unindent(_ROW_LABEL_W + float(SPACE.MD))


def _has_script(source: UIDocument) -> bool:
    return (
        document_dir_of(source.document) / SCRIPTS_DIR_NAME / DOCUMENT_SCRIPT_BASENAME
    ).is_file()
