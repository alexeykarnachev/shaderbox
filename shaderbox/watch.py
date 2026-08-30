from dataclasses import replace

from loguru import logger

from shaderbox.app import App
from shaderbox.core import Pass
from shaderbox.paths import shader_lib_root
from shaderbox.shader_lib import is_shader_lib_path
from shaderbox.ui_models import UIDocument


def reload_document_if_changed(app: App, name: str, ui_document: UIDocument) -> None:
    # Every pass, not just the output: a document is N files now, and the one you are editing is
    # often not the one that draws last.
    for render_pass in list(ui_document.document.passes.values()):
        _reload_pass_if_changed(app, name, render_pass)


def _reload_pass_if_changed(app: App, name: str, render_pass: Pass) -> None:
    # `sources` is this pass's own root plus the lib files it includes. The root is identified by
    # PATH rather than by index 0 — the position happens to be right, but a positional rule is
    # wrong-by-construction the moment anything reorders the list.
    root_path = render_pass.source.path
    for i, src in enumerate(render_pass.compile_unit.sources):
        path = src.path
        if not path.exists():
            continue
        disk_mtime = path.lstat().st_mtime
        if disk_mtime == src.mtime:
            continue

        if path == root_path:
            logger.debug(f"Reloading document {name} ({path.name} changed)")
            try:
                new_text = path.read_text()
                render_pass.release_program(new_text)
                render_pass.source = replace(render_pass.source, mtime=disk_mtime)
                app.sync_editor_from_disk(path, new_text)
            except Exception as e:
                logger.error(f"Failed to reload document {name}: {e}")
                render_pass.source = replace(render_pass.source, mtime=disk_mtime)
            # release_program() rebuilt `sources` — stop iterating the stale list.
            return

        # Lib reload: bump cached mtime + invalidate so the next compile re-resolves the
        # include. If an open session's text diverges from disk, re-sync (external edit);
        # if it matches, the user saved in-app — don't clobber their undo history.
        logger.debug(f"Reloading document {name} (lib changed: {path.name})")
        render_pass.compile_unit.sources[i] = replace(src, mtime=disk_mtime)
        render_pass.invalidate()
        session = app.editor_sessions.get(path)
        if session is not None:
            try:
                new_text = path.read_text()
                if session.editor.get_text() != new_text:
                    session.editor.set_text(new_text)
                    session.saved_undo = session.editor.get_undo_index()
                session.source = replace(
                    session.source, text=new_text, mtime=disk_mtime
                )
            except Exception as e:
                logger.error(f"Failed to sync lib editor for {path}: {e}")


def maybe_rebuild_lib_index(app: App) -> bool:
    # Detect lib-root changes (add / remove / mtime) and rebuild the index. One glob + N
    # stats per frame. `is_shader_lib_path` MUST match the filter ShaderLibIndex.build
    # applies, else current vs cached diverge every frame on trashed files and loop forever.
    root = shader_lib_root()
    current: dict[str, float] = {}
    for path in root.glob("**/*.glsl"):
        if not is_shader_lib_path(path, root):
            continue
        try:
            current[str(path)] = path.lstat().st_mtime
        except OSError:
            continue
    cached = {str(p): s.mtime for p, s in app.shader_lib_index.sources.items()}
    if current == cached:
        return False
    app.rebuild_shader_lib_index()
    # Invalidate every pass that pulled in a lib file so its next render recompiles against the
    # new index (a referenced function may have changed or disappeared).
    for ui_document in app.ui_documents.values():
        for render_pass in ui_document.document.passes.values():
            if len(render_pass.compile_unit.sources) > 1:
                render_pass.invalidate()
    return True
