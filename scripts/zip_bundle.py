"""Zip one staged bundle directory for `build.sh`.

Python's zipfile rather than `zip`: Git Bash on Windows ships no `zip`, and a Windows
filesystem carries no exec bit, so the modes are written here explicitly -- `*.sh`
gets 0755 and everything else 0644 -- and the archive is the same whichever host
built it.

Usage: zip_bundle.py <staged dir> <out.zip>
"""

import sys
import zipfile
from pathlib import Path

_EXEC_MODE = 0o755
_FILE_MODE = 0o644
_DIR_MODE = 0o755
_S_IFREG = 0o100000
_S_IFDIR = 0o040000


def _info(arcname: str, mode: int, file_type: int) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(arcname)
    info.create_system = 3  # unix, so extractors honour the mode bits
    info.external_attr = (file_type | mode) << 16
    return info


def zip_bundle(stage: Path, out: Path) -> None:
    root = stage.name
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(_info(f"{root}/", _DIR_MODE, _S_IFDIR), b"")
        for path in sorted(stage.rglob("*")):
            arcname = f"{root}/{path.relative_to(stage).as_posix()}"
            if path.is_dir():
                archive.writestr(_info(f"{arcname}/", _DIR_MODE, _S_IFDIR), b"")
                continue
            mode = _EXEC_MODE if path.suffix == ".sh" else _FILE_MODE
            info = _info(arcname, mode, _S_IFREG)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    zip_bundle(Path(sys.argv[1]), Path(sys.argv[2]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
