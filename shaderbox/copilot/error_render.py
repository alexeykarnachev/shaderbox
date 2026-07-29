from pathlib import Path

from shaderbox.copilot.address import NODE_SHORT_ID_LEN, lib_address
from shaderbox.copilot.capabilities import CompileErrorInfo
from shaderbox.paths import shader_lib_root

# The ONE model-facing renderer for compile errors. `CompileErrorInfo.path` carries the real
# absolute path (the backend's own-file vs cross-file guards resolve it); the agent only ever
# sees a label in ITS address space — `lib:<rel>` for a library file, the node's short id for a
# node source, and any non-absolute label (`script.py`, "") verbatim. Every renderer routes
# through here so a new one cannot re-leak the filesystem by omission.


def _error_label(path: str) -> str:
    if not path or not Path(path).is_absolute():
        return path
    p = Path(path)
    root = shader_lib_root()
    try:
        return lib_address(p.relative_to(root))
    except ValueError:
        pass
    try:
        return lib_address(p.resolve().relative_to(root.resolve()))
    except ValueError:
        return p.parent.name[:NODE_SHORT_ID_LEN]


def format_compile_errors(errors: list[CompileErrorInfo]) -> str:
    return "\n".join(f"{_error_label(e.path)}:{e.line}: {e.message}" for e in errors)
