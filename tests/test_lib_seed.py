"""Shipped shader-library seeding / sync / factory reset — pure file ops, no GL."""

from pathlib import Path

from shaderbox.shader_lib.seed import reset_to_shipped, sync_shipped_lib


def _mk(d: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return d


def _roots(tmp_path: Path, seed_files: dict[str, str]) -> tuple[Path, Path]:
    seed = _mk(tmp_path / "seed", seed_files)
    root = tmp_path / "root"
    root.mkdir()
    return seed, root


def test_fresh_root_gets_seeded(tmp_path: Path) -> None:
    seed, root = _roots(tmp_path, {"draw/render.glsl": "v1\n", "noise/n.glsl": "n\n"})
    assert sync_shipped_lib(seed, root) == 2
    assert (root / "draw/render.glsl").read_text() == "v1\n"
    assert (root / ".seed_manifest.json").exists()


def test_pristine_file_follows_shipped_update(tmp_path: Path) -> None:
    seed, root = _roots(tmp_path, {"a.glsl": "v1\n"})
    sync_shipped_lib(seed, root)
    (seed / "a.glsl").write_text("v2\n", encoding="utf-8")
    assert sync_shipped_lib(seed, root) == 1
    assert (root / "a.glsl").read_text() == "v2\n"


def test_edited_file_is_never_touched(tmp_path: Path) -> None:
    seed, root = _roots(tmp_path, {"a.glsl": "v1\n"})
    sync_shipped_lib(seed, root)
    (root / "a.glsl").write_text("my version\n", encoding="utf-8")
    (seed / "a.glsl").write_text("v2\n", encoding="utf-8")
    assert sync_shipped_lib(seed, root) == 0
    assert (root / "a.glsl").read_text() == "my version\n"


def test_deleted_shipped_file_stays_deleted(tmp_path: Path) -> None:
    seed, root = _roots(tmp_path, {"a.glsl": "v1\n", "b.glsl": "b\n"})
    sync_shipped_lib(seed, root)
    (root / "a.glsl").unlink()
    assert sync_shipped_lib(seed, root) == 0
    assert not (root / "a.glsl").exists()
    # ... even when the shipped content changes later.
    (seed / "a.glsl").write_text("v2\n", encoding="utf-8")
    assert sync_shipped_lib(seed, root) == 0
    assert not (root / "a.glsl").exists()


def test_new_shipped_file_arrives_later(tmp_path: Path) -> None:
    seed, root = _roots(tmp_path, {"a.glsl": "a\n"})
    sync_shipped_lib(seed, root)
    _mk(seed, {"sdf2d/new.glsl": "new\n"})
    assert sync_shipped_lib(seed, root) == 1
    assert (root / "sdf2d/new.glsl").read_text() == "new\n"


def test_file_dropped_from_seed_pristine_removed_edited_kept(tmp_path: Path) -> None:
    seed, root = _roots(
        tmp_path, {"old.glsl": "o\n", "edited.glsl": "e\n", "keep.glsl": "k\n"}
    )
    sync_shipped_lib(seed, root)
    (root / "edited.glsl").write_text("mine\n", encoding="utf-8")
    (seed / "old.glsl").unlink()
    (seed / "edited.glsl").unlink()
    sync_shipped_lib(seed, root)
    assert not (root / "old.glsl").exists()  # pristine -> removed with the seed
    assert (root / "edited.glsl").read_text() == "mine\n"  # edited -> user-owned
    assert (root / "keep.glsl").read_text() == "k\n"


def test_fully_empty_seed_never_deletes(tmp_path: Path) -> None:
    # A zero-file seed dir means a broken install, not "the library is gone" —
    # the guard must skip the sync (incl. stale cleanup) rather than wipe pristine files.
    seed, root = _roots(tmp_path, {"a.glsl": "a\n"})
    sync_shipped_lib(seed, root)
    (seed / "a.glsl").unlink()
    assert sync_shipped_lib(seed, root) == 0
    assert (root / "a.glsl").read_text() == "a\n"


def test_user_file_shadowing_a_new_shipped_name_wins(tmp_path: Path) -> None:
    seed, root = _roots(tmp_path, {"a.glsl": "a\n"})
    sync_shipped_lib(seed, root)
    _mk(root, {"mine.glsl": "user content\n"})
    _mk(seed, {"mine.glsl": "shipped content\n"})
    assert sync_shipped_lib(seed, root) == 0
    assert (root / "mine.glsl").read_text() == "user content\n"


def test_identical_unknown_file_is_adopted_as_pristine(tmp_path: Path) -> None:
    # A pre-manifest install (or a hand-copied seed) has the shipped bytes but no
    # manifest entry: adopt it, so later shipped updates flow.
    seed, root = _roots(tmp_path, {"a.glsl": "v1\n"})
    _mk(root, {"a.glsl": "v1\n"})
    assert sync_shipped_lib(seed, root) == 0
    (seed / "a.glsl").write_text("v2\n", encoding="utf-8")
    assert sync_shipped_lib(seed, root) == 1
    assert (root / "a.glsl").read_text() == "v2\n"


def test_reset_restores_trashes_and_keeps_user_files(tmp_path: Path) -> None:
    seed, root = _roots(tmp_path, {"a.glsl": "a1\n", "b.glsl": "b1\n"})
    sync_shipped_lib(seed, root)
    (root / "a.glsl").write_text("edited\n", encoding="utf-8")  # edited -> trash
    (root / "b.glsl").unlink()  # deleted -> restored
    _mk(root, {"mine.glsl": "user\n"})  # user-authored -> kept
    written, trashed = reset_to_shipped(seed, root)
    assert (written, trashed) == (2, 1)
    assert (root / "a.glsl").read_text() == "a1\n"
    assert (root / "b.glsl").read_text() == "b1\n"
    assert (root / "mine.glsl").read_text() == "user\n"
    assert (root / ".trash" / "a.glsl").read_text() == "edited\n"
    # After a reset everything shipped is pristine again: a sync is a no-op.
    assert sync_shipped_lib(seed, root) == 0


def test_reset_trash_name_collision_gets_suffix(tmp_path: Path) -> None:
    seed, root = _roots(tmp_path, {"a.glsl": "a1\n"})
    sync_shipped_lib(seed, root)
    for marker in ("first\n", "second\n"):
        (root / "a.glsl").write_text(marker, encoding="utf-8")
        reset_to_shipped(seed, root)
    assert (root / ".trash" / "a.glsl").read_text() == "first\n"
    assert (root / ".trash" / "a_1.glsl").read_text() == "second\n"


def test_empty_seed_dir_is_a_noop(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    assert sync_shipped_lib(tmp_path / "missing", root) == 0
    assert not (root / ".seed_manifest.json").exists()
