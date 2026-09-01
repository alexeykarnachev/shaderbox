"""The embedded code editor (feature 067): ctypes binding over libeditor.so.

`ffi` is the leaf binding (no imgui/moderngl), `render` the moderngl MTSDF pass,
`input` the glfw->ed_key pump. The vendored binary + atlas live in
`shaderbox/resources/editor/` with the editor-repo commit sha in `VERSION`.
"""

from shaderbox.editor.ffi import Editor, Kind, Language, Mode, Slot, language_for_path

__all__ = [
    "Editor",
    "Kind",
    "Language",
    "Mode",
    "Slot",
    "language_for_path",
]
