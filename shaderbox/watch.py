from dataclasses import replace

from loguru import logger

from shaderbox.app import App
from shaderbox.paths import shader_lib_root
from shaderbox.shader_lib import is_shader_lib_path
from shaderbox.ui_models import UINode


def reload_node_if_changed(app: App, name: str, ui_node: UINode) -> None:
    # sources[0] is the root (resolve_includes seeds it first); sources[1:] are lib
    # files in first-seen order. Root and lib reloads differ — see inline.
    for i, src in enumerate(ui_node.node.compile_unit.sources):
        path = src.path
        if not path.exists():
            continue
        disk_mtime = path.lstat().st_mtime
        if disk_mtime == src.mtime:
            continue

        if i == 0:
            # Root reload: re-sync the open editor session from disk.
            logger.debug(f"Reloading node {name} (root shader changed)")
            try:
                new_text = path.read_text()
                ui_node.node.release_program(new_text)
                ui_node.node.source = replace(ui_node.node.source, mtime=disk_mtime)
                app.sync_editor_from_disk(name, new_text)
            except Exception as e:
                logger.error(f"Failed to reload node {name}: {e}")
                ui_node.node.source = replace(ui_node.node.source, mtime=disk_mtime)
            # release_program() rebuilt `sources` — stop iterating the stale list.
            return

        # Lib reload: bump cached mtime + invalidate so the next compile re-resolves the
        # include. If an open session's text diverges from disk, re-sync (external edit);
        # if it matches, the user saved in-app — don't clobber their undo history.
        logger.debug(f"Reloading node {name} (lib changed: {path.name})")
        ui_node.node.compile_unit.sources[i] = replace(src, mtime=disk_mtime)
        ui_node.node.invalidate()
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
    # Invalidate every node that pulled in a lib file so its next render recompiles
    # against the new index (a referenced function may have changed or disappeared).
    for ui_node in app.ui_nodes.values():
        if len(ui_node.node.compile_unit.sources) > 1:
            ui_node.node.invalidate()
    return True
