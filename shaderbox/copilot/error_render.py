from pathlib import Path

from shaderbox.copilot.address import DOCUMENT_SHORT_ID_LEN, lib_address
from shaderbox.copilot.capabilities import CompileErrorInfo
from shaderbox.paths import shader_lib_root

# The ONE model-facing renderer for compile errors. `CompileErrorInfo.path` carries the real
# absolute path (the backend's own-file vs cross-file guards resolve it); the agent only ever
# sees a label in ITS address space — `lib:<rel>` for a library file, the document's short id for a
# document source, and any non-absolute label (`script.py`, "") verbatim. Every renderer routes
# through here so a new one cannot re-leak the filesystem by omission.
# Under a short-id collision the map grows every id past DOCUMENT_SHORT_ID_LEN while this label stays
# 4 chars — acceptable: the agent addresses documents from the map, the label only names the breakage.


def _error_label(path: str, root: Path) -> str:
    if not path or not Path(path).is_absolute():
        return path
    p = Path(path)
    try:
        return lib_address(p.relative_to(root))
    except ValueError:
        pass
    try:
        return lib_address(p.resolve().relative_to(root.resolve()))
    except ValueError:
        return p.parent.name[:DOCUMENT_SHORT_ID_LEN]


def format_compile_errors(errors: list[CompileErrorInfo]) -> str:
    # `line` 0 = unmapped (an unparsed compiler message, a script error the mapper couldn't place):
    # print no location rather than a ":0:" the agent would try to edit at.
    root = shader_lib_root()
    lines: list[str] = []
    for e in errors:
        label = _error_label(e.path, root)
        loc = f"{label}:{e.line}" if e.line > 0 else label
        lines.append(f"{loc}: {e.message}" if loc else e.message)
    return "\n".join(lines)
