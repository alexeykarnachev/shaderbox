"""Shipped shader-library seeding, update sync, and factory reset.

The canonical library ships in `shaderbox/resources/shader_lib/`; the live root
(`shader_lib_root()`) is user-writable. A dot-file manifest in the live root maps
each seeded rel-path to the sha1 of the shipped version it came from, which makes
user intent decidable per file: a pristine file (disk hash == manifest) tracks
shipped updates, an edited file is never touched, a deleted file stays deleted,
and a user-authored file (unknown to the manifest) is never touched by anything.
`reset_to_shipped` force-restores every shipped file (edited copies are moved to
`.trash/` first) and still leaves user-authored files alone.
"""

import hashlib
import json
import shutil
from pathlib import Path

from loguru import logger

_MANIFEST_NAME = ".seed_manifest.json"


def _sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def _load_manifest(root: Path) -> dict[str, str]:
    try:
        raw = json.loads((root / _MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items()}


def _save_manifest(root: Path, manifest: dict[str, str]) -> None:
    (root / _MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _seed_files(seed_dir: Path) -> dict[str, bytes]:
    if not seed_dir.is_dir():
        return {}
    return {
        p.relative_to(seed_dir).as_posix(): p.read_bytes()
        for p in sorted(seed_dir.rglob("*.glsl"))
    }


def _trash_file(root: Path, path: Path) -> None:
    # Mirror of file_ops soft-delete: basename into `.trash/`, numeric suffix on collision.
    trash = root / ".trash"
    trash.mkdir(parents=True, exist_ok=True)
    dest = trash / path.name
    i = 1
    while dest.exists():
        dest = trash / f"{path.stem}_{i}{path.suffix}"
        i += 1
    shutil.move(str(path), str(dest))


def sync_shipped_lib(seed_dir: Path, root: Path) -> int:
    """Per-file shipped->live sync, run once at startup (before the first index build).

    Returns the number of files written. Cases per shipped rel-path:
    missing + unknown to the manifest -> seed it (new install or new shipped file);
    missing + in the manifest -> the user deleted it, stays deleted;
    pristine (disk hash == manifest) -> follow the shipped update;
    edited -> never touched. A manifest entry whose rel-path left the seed is a
    rename/removal upstream: a still-pristine copy is deleted with it, an edited
    copy becomes user-owned (entry dropped). A file on disk that the manifest does
    not know is user-authored: adopted only if byte-identical to the shipped one.
    """
    seed = _seed_files(seed_dir)
    if not seed:
        return 0
    manifest = _load_manifest(root)
    written = 0
    for rel, seed_bytes in seed.items():
        target = root / rel
        seed_hash = _sha1(seed_bytes)
        if not target.exists():
            if rel in manifest:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(seed_bytes)
            manifest[rel] = seed_hash
            written += 1
            continue
        disk_hash = _sha1(target.read_bytes())
        if rel not in manifest:
            if disk_hash == seed_hash:
                manifest[rel] = seed_hash
            else:
                logger.info(f"shader-lib seed: user file shadows shipped '{rel}'")
            continue
        if disk_hash == manifest[rel] and disk_hash != seed_hash:
            target.write_bytes(seed_bytes)
            manifest[rel] = seed_hash
            written += 1
        elif disk_hash == seed_hash:
            manifest[rel] = seed_hash
    for rel in [r for r in manifest if r not in seed]:
        stale = root / rel
        if stale.exists() and _sha1(stale.read_bytes()) == manifest[rel]:
            stale.unlink()
            logger.info(f"shader-lib seed: removed stale shipped file '{rel}'")
        del manifest[rel]
    _save_manifest(root, manifest)
    if written:
        logger.info(f"shader-lib seed: {written} shipped file(s) written")
    return written


def reset_to_shipped(seed_dir: Path, root: Path) -> tuple[int, int]:
    """Factory reset: every shipped file back to its shipped content.

    Edited copies are moved to `.trash/` before being overwritten; deleted shipped
    files are restored; user-authored files are untouched. Returns
    (files written, edited copies trashed). The caller does NOT need to rebuild
    the lib index — the mtime watcher picks the writes up next frame.
    """
    seed = _seed_files(seed_dir)
    manifest: dict[str, str] = {}
    written = 0
    trashed = 0
    for rel, seed_bytes in seed.items():
        target = root / rel
        seed_hash = _sha1(seed_bytes)
        if target.exists():
            disk_bytes = target.read_bytes()
            if _sha1(disk_bytes) == seed_hash:
                manifest[rel] = seed_hash
                continue
            _trash_file(root, target)
            trashed += 1
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(seed_bytes)
        manifest[rel] = seed_hash
        written += 1
    _save_manifest(root, manifest)
    logger.info(f"shader-lib reset: {written} restored, {trashed} trashed")
    return written, trashed
