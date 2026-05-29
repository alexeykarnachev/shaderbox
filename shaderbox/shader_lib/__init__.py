"""Shader library — auto-resolved GLSL helpers.

Public surface (import from `shaderbox.shader_lib`, not the submodules):
  - `ShaderLibIndex` / `ShaderLibFunction` — the index snapshot + a single function.
  - `is_shader_lib_path` — the dot-dir filter shared by build + the mtime watcher.
  - `active` / `set_active` — the module-level active-index accessor.
  - `resolve_usage` / `ResolveError` — the per-compile usage pruner.

Submodules: `index` (types + build + singleton), `resolver` (resolve_usage),
`parser` (regex/brace text machinery).
"""

from shaderbox.shader_lib.index import (
    ShaderLibFunction,
    ShaderLibIndex,
    active,
    is_shader_lib_path,
    set_active,
)
from shaderbox.shader_lib.resolver import ResolveError, resolve_usage

__all__ = [
    "ResolveError",
    "ShaderLibFunction",
    "ShaderLibIndex",
    "active",
    "is_shader_lib_path",
    "resolve_usage",
    "set_active",
]
