"""No committed text file carries a stray control byte.

A bulk rewrite that protects words by swapping in placeholder tokens leaks one the moment a token
survives the restore pass — and what lands is a file that reads correctly in a diff, renders
plausibly in most viewers, and is silently corrupt. Feature 065's `node` -> `document` sweep did
exactly that to a `dev_flow.md` sentence: the path `projects/_lab/` became two NULs around a
placeholder id, shipped, and was found by a reader rather than by any gate.

Tabs, newlines and carriage returns are ordinary text; everything else below 0x20 is not.
"""

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_ALLOWED = {0x09, 0x0A, 0x0D}
_SUFFIXES = {".py", ".md", ".json", ".toml", ".yaml", ".yml", ".glsl", ".sh", ".txt"}
_SKIP_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    ".ruff_cache",
    ".pytest_cache",
    "build",
    "dist",
}


def _text_files() -> list[Path]:
    files: list[Path] = []
    for path in _ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in _SUFFIXES:
            continue
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def _control_bytes(path: Path) -> list[str]:
    return sorted(
        hex(b) for b in {b for b in path.read_bytes() if b < 0x20 and b not in _ALLOWED}
    )


def test_no_committed_text_file_holds_a_control_byte() -> None:
    # One case over every file rather than one case each: the suite gains a test, not a thousand,
    # and a failure names every offender at once instead of stopping at the first.
    offenders = {
        str(path.relative_to(_ROOT)): bad
        for path in _text_files()
        if (bad := _control_bytes(path))
    }
    assert not offenders, (
        f"{offenders} — a bulk rewrite most likely leaked a placeholder token"
    )


def test_the_guard_sees_a_corrupt_file(tmp_path: Path) -> None:
    # The guard's own falsifier: a file shaped like the one that shipped reads fine in a diff.
    corrupt = tmp_path / "sample.md"
    corrupt.write_bytes(b"a line with a \x00placeholder\x00 in it\n")
    assert _control_bytes(corrupt) == ["0x0"]
    clean = tmp_path / "ok.md"
    clean.write_text("a line\twith a tab\r\n")
    assert _control_bytes(clean) == []
