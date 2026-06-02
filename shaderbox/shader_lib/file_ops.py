"""Shader-library file management: create / rename / delete (trash) / reveal.

`ShaderLibFileManager` owns the picker's inline-input + delete-armed + filter
state and the file/dir CRUD against `shader_lib_root()`. It does NOT import
`App` — editor-session cleanup (which is App's editor domain) flows back through
injected callbacks (`on_paths_removed`, `on_path_renamed`). The mutating verbs
take explicit args (path / name) so a non-UI caller (e.g. a future agent tool)
can drive them; the picker keeps thin inline-input shims that read the buffer
then call the explicit core (`create_file_in` / `create_dir_in` / `rename_file`).
"""

import shutil
from collections.abc import Callable
from pathlib import Path

from loguru import logger

from shaderbox.editor_types import InlineInput
from shaderbox.notifications import Notifications
from shaderbox.paths import shader_lib_root, shader_lib_trash_dir
from shaderbox.shader_lib.index import ShaderLibIndex
from shaderbox.theme import COLOR


class ShaderLibFileManager:
    def __init__(
        self,
        *,
        notifications: Notifications,
        rebuild_index: Callable[[], None],
        index_getter: Callable[[], ShaderLibIndex],
        on_paths_removed: Callable[[list[Path]], None],
        on_path_renamed: Callable[[Path, Path], None],
    ) -> None:
        self._notifications = notifications
        self._rebuild_index = rebuild_index
        self._index = index_getter
        self._on_paths_removed = on_paths_removed
        self._on_path_renamed = on_path_renamed

        # Picker filter / selection state.
        self.picker_query: str = ""
        self.picker_selected_function: str = ""
        self.picker_just_opened: bool = False
        self.picker_favs_only: bool = False
        self.picker_disabled_tags: set[str] = set()
        self.picker_new_tag_buf: str = ""
        self.picker_tag_input_focused: bool = False

        # Three mutually-exclusive inline inputs (only one open at a time;
        # `reset_inline_state` enforces the mutex). Rename's `target` is the file
        # being renamed; new-file/new-dir's `target` is the parent directory.
        self.file_rename = InlineInput()
        self.file_new = InlineInput()
        self.dir_new = InlineInput()
        # A single file/dir path armed for delete-confirm at a time. Root never armed.
        self.file_delete_armed: Path | None = None
        self.dir_delete_armed: Path | None = None

    # ----------------------------------------------------------------
    # Inline-input mutex + openers (UI shims; read buffers, call explicit cores)
    # ----------------------------------------------------------------

    def reset_inline_state(self) -> None:
        # Single source of mutual-exclusion: every `begin_*` / `arm_*` opener
        # calls this first to clear ALL sibling inline-input + armed state.
        self.file_rename.close()
        self.file_new.close()
        self.dir_new.close()
        self.file_delete_armed = None
        self.dir_delete_armed = None

    def arm_file_delete(self, path: Path | None) -> None:
        # Disarm-by-passing-None is the cancel; otherwise replace mutex state.
        if path is None:
            self.file_delete_armed = None
            return
        self.reset_inline_state()
        self.file_delete_armed = path

    def arm_dir_delete(self, path: Path | None) -> None:
        if path is None:
            self.dir_delete_armed = None
            return
        self.reset_inline_state()
        self.dir_delete_armed = path

    def begin_file_rename(self, path: Path) -> None:
        # Pre-fill the buffer with the current relative path so the user can
        # edit, not retype.
        try:
            rel = path.relative_to(shader_lib_root())
        except ValueError:
            rel = Path(path.name)
        self.reset_inline_state()
        self.file_rename.open(path, buf=str(rel))

    def cancel_file_rename(self) -> None:
        self.file_rename.close()

    def begin_file_new_in(self, dir_rel: Path) -> None:
        # Open the inline new-file input under the given subdir (Path("") for
        # the lib root). `target` is the parent dir; `buf` is the new filename.
        self.reset_inline_state()
        self.file_new.open(dir_rel)

    def cancel_file_new(self) -> None:
        self.file_new.close()

    def begin_dir_new_in(self, parent_rel: Path) -> None:
        self.reset_inline_state()
        self.dir_new.open(parent_rel)

    def cancel_dir_new(self) -> None:
        self.dir_new.close()

    def commit_file_new(self) -> Path | None:
        dir_rel = self.file_new.target
        if dir_rel is None or not self.file_new.buf.strip():
            return None
        created = self.create_file_in(dir_rel, self.file_new.buf.strip())
        if created is not None:
            self.cancel_file_new()
        return created

    def commit_dir_new(self) -> Path | None:
        parent_rel = self.dir_new.target
        if parent_rel is None:
            return None
        name = self.dir_new.buf.strip().strip("/")
        if not name:
            return None
        created = self.create_dir_in(parent_rel, name)
        if created is not None:
            self.cancel_dir_new()
        return created

    # ----------------------------------------------------------------
    # Explicit CRUD cores (agent-callable — explicit args, no buffer reads)
    # ----------------------------------------------------------------

    def create_file_in(self, dir_rel: Path, name: str) -> Path | None:
        # Create the new file at `<shader_lib_root>/<dir_rel>/<name>(.glsl)`.
        target = self._validate_target(
            str(dir_rel / name), kind="new-file", suffix=".glsl"
        )
        if target is None:
            return None
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                "/// new SB_* function — write a doc here\n"
                "// float SB_my_function(float x) {\n"
                "//     return x;\n"
                "// }\n",
                encoding="utf-8",
            )
        except OSError as e:
            logger.error(f"Failed to create lib file {target}: {e}")
            self._notifications.push(
                f"Create failed: {e!s}", color=COLOR.STATE_ERROR[:3]
            )
            return None
        logger.info(f"Created new lib file: {target}")
        self._rebuild_index()
        return target

    def resolve_copilot_path(self, rel_path: str) -> Path | None:
        # Resolve a copilot "lib:<rel>" address to an absolute path UNDER shader_lib_root,
        # rejecting path traversal (feature 020·16 Decision 5 — the same guard create_file_in
        # uses, but allowing an EXISTING file since the copilot edits existing lib files too).
        # Returns None if the path escapes the root or is empty/non-.glsl.
        cleaned = rel_path.strip().lstrip("/")
        if not cleaned:
            return None
        if not cleaned.endswith(".glsl"):
            cleaned += ".glsl"
        root = shader_lib_root().resolve()
        target = (root / cleaned).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            logger.warning(f"Copilot lib path rejected (escapes root): {cleaned}")
            return None
        return target

    def write_copilot_lib_file(self, path: Path, source: str) -> bool:
        # Write LIVE (uncommented) source to a lib file for the copilot (feature 020·16
        # Decision 5) — distinct from create_file_in's commented stub, which the index would
        # see as zero functions. Creates parent dirs, writes, rebuilds the index so the new
        # function is immediately resolvable. Returns False on an OS error.
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")
        except OSError as e:
            logger.error(f"Copilot lib write failed for {path}: {e}")
            return False
        logger.info(f"Copilot wrote lib file: {path}")
        self._rebuild_index()
        return True

    def create_dir_in(self, parent_rel: Path, name: str) -> Path | None:
        # Create `<shader_lib_root>/<parent_rel>/<name>` as a real directory + a
        # starter `placeholder.glsl` inside (so the dir survives the
        # ShaderLibIndex.build glob, which only walks `.glsl`).
        target = self._validate_target(str(parent_rel / name), kind="new-dir")
        if target is None:
            return None
        # Emit a real SB_ stub (not a commented-out one) so the file shows up as a
        # function leaf the user can immediately rename/edit — an empty placeholder
        # would render as an un-expandable leaf and look like a bug.
        starter = target / "placeholder.glsl"
        sanitized_name = "".join(c if c.isalnum() else "_" for c in target.name)
        stub_fn_name = f"SB_{sanitized_name}_placeholder"
        try:
            target.mkdir(parents=True, exist_ok=False)
            starter.write_text(
                f"/// rename me — placeholder stub for the {target.name}/ subdir\n"
                f"float {stub_fn_name}(float x) {{\n"
                "    return x;\n"
                "}\n",
                encoding="utf-8",
            )
        except OSError as e:
            logger.error(f"Failed to create lib dir {target}: {e}")
            self._notifications.push(
                f"Create failed: {e!s}", color=COLOR.STATE_ERROR[:3]
            )
            return None
        logger.info(f"Created new lib dir: {target}")
        self._rebuild_index()
        return target

    def delete_file(self, path: Path) -> None:
        # Move into `.trash/` (basename + numeric suffix on collision). The mtime
        # watcher rebuilds the index next frame. Armed state cleared on every exit.
        self.file_delete_armed = None
        if not path.exists():
            logger.warning(f"Lib file no longer exists: {path}")
            return
        trash = shader_lib_trash_dir()
        dest = trash / path.name
        i = 1
        while dest.exists():
            dest = trash / f"{path.stem}_{i}{path.suffix}"
            i += 1
        try:
            shutil.move(str(path), str(dest))
        except OSError as e:
            logger.error(f"Failed to delete lib file {path}: {e}")
            self._notifications.push(
                f"Delete failed: {e!s}", color=COLOR.STATE_ERROR[:3]
            )
            return
        logger.info(f"Trashed lib file: {path} -> {dest}")
        self._on_paths_removed([path])
        self._clear_selection_for_files([path])
        msg = f"Deleted {path.name}"
        if dest.name != path.name:
            msg += f" (as {dest.name})"
        msg += " - recoverable in .trash/"
        self._notifications.push(msg)

    def delete_dir(self, path: Path) -> None:
        # Recursive move of an entire subdir into .trash/. EVERY file under the
        # dir lands in `.trash/<basename>` with the numeric-suffix collision rule
        # — nothing is silently rmtree'd. Refuses symlinked dirs so a
        # `shader_lib/external -> /other/repo/` link can't trash files outside
        # shader_lib_root. Always clears armed state on any exit.
        self.dir_delete_armed = None
        if not path.exists() or not path.is_dir():
            logger.warning(f"Lib dir no longer exists or not a dir: {path}")
            return
        root = shader_lib_root().resolve()
        resolved = path.resolve()
        if resolved == root:
            logger.warning("Refusing to delete the shader_lib_root itself.")
            return
        if path.is_symlink() or not resolved.is_relative_to(root):
            logger.warning(f"Refusing to delete symlinked/escaping lib dir: {path}")
            self._notifications.push(
                "Delete refused: symlinked or outside shader_lib_root",
                color=COLOR.STATE_ERROR[:3],
            )
            return
        trash = shader_lib_trash_dir()
        moved_files: list[Path] = []
        try:
            # Walk EVERY file (glsl + non-glsl) so the user's `notes.md` next to
            # their shader isn't silently destroyed by the rmtree below. Skip
            # symlinks to avoid following an escape route mid-walk.
            for f in sorted(path.rglob("*")):
                if not f.is_file() or f.is_symlink():
                    continue
                if not f.resolve().is_relative_to(root):
                    # Defensive: a file here should never resolve outside root (we
                    # refused symlinked dirs above) — log and skip.
                    logger.debug(f"Skipping out-of-root file during dir delete: {f}")
                    continue
                dest = trash / f.name
                i = 1
                while dest.exists():
                    dest = trash / f"{f.stem}_{i}{f.suffix}"
                    i += 1
                shutil.move(str(f), str(dest))
                moved_files.append(f)
            shutil.rmtree(path)
        except OSError as e:
            logger.error(f"Failed to delete lib dir {path}: {e}")
            self._notifications.push(
                f"Delete dir failed: {e!s}", color=COLOR.STATE_ERROR[:3]
            )
            return
        logger.info(f"Trashed lib dir {path} ({len(moved_files)} files)")
        self._on_paths_removed(moved_files)
        self._clear_selection_for_files(moved_files)
        self._notifications.push(
            f"Deleted {path.name}/ ({len(moved_files)} files) - recoverable in .trash/"
        )

    def rename_file(self, old: Path, new_rel: str) -> Path | None:
        # Returns the new resolved path on success, None on rejection or no-op.
        target = self._validate_target(
            new_rel, kind="rename", suffix=".glsl", allow_existing=old
        )
        if target is None:
            return None
        if target == old.resolve():
            # Self-rename — silent no-op, close the input.
            self.cancel_file_rename()
            return None
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            old.rename(target)
        except OSError as e:
            logger.error(f"Failed to rename lib file {old} -> {target}: {e}")
            self._notifications.push(
                f"Rename failed: {e!s}", color=COLOR.STATE_ERROR[:3]
            )
            return None
        logger.info(f"Renamed lib file: {old} -> {target}")
        self._on_path_renamed(old, target)
        # Function names don't change on rename, so picker_selected_function stays
        # valid; the next frame's tree walk re-discovers it at its new file path.
        self.cancel_file_rename()
        return target

    def _clear_selection_for_files(self, files: list[Path]) -> None:
        deleted_fn_names = {
            fn.name
            for fn in self._index().functions.values()
            if any(fn.file == f for f in files)
        }
        if self.picker_selected_function in deleted_fn_names:
            self.picker_selected_function = ""

    def _validate_target(
        self,
        rel_path: str,
        *,
        kind: str,
        suffix: str | None = None,
        allow_existing: Path | None = None,
    ) -> Path | None:
        # Centralized validation for new-file / new-dir / rename: strip, append
        # `suffix` (e.g. ".glsl") if missing, resolve under shader_lib_root, reject
        # path traversal, reject collisions (unless the target equals
        # `allow_existing` — used by rename's self-rename short-circuit).
        cleaned = rel_path.strip()
        if not cleaned:
            return None
        if suffix and not cleaned.endswith(suffix):
            cleaned += suffix
        root = shader_lib_root().resolve()
        target = (root / cleaned).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            logger.warning(f"Lib {kind} rejected (outside shader_lib_root): {cleaned}")
            self._notifications.push(
                f"{kind.capitalize()} rejected: path escapes shader_lib_root",
                color=COLOR.STATE_ERROR[:3],
            )
            return None
        if allow_existing is not None and target == allow_existing.resolve():
            return target
        if target.exists():
            logger.warning(f"Lib {kind} rejected (target exists): {target}")
            self._notifications.push(
                f"{kind.capitalize()} rejected: target exists",
                color=COLOR.STATE_ERROR[:3],
            )
            return None
        return target
