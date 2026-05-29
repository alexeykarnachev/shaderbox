from dataclasses import dataclass
from pathlib import Path

from imgui_bundle import imgui_color_text_edit as text_edit

from shaderbox.shader_source import ShaderSource


@dataclass(frozen=True)
class JumpRequest:
    path: Path
    line: int
    column: int


@dataclass(frozen=True)
class HoverMark:
    path: Path
    line: int


@dataclass
class EditorSession:
    # A live TextEditor instance bound to a specific on-disk file. `source` is
    # the snapshot used to seed the editor; the editor's current text may diverge
    # from `source.text` until the next flush. `saved_undo` is the editor's
    # undo-index at last save — anything beyond that is unsaved.
    editor: text_edit.TextEditor
    source: ShaderSource
    saved_undo: int


@dataclass
class InlineInput:
    # One of the three shader-lib-picker inline inputs (file rename / file new /
    # dir new). `target` is the path the input is bound to (file path for rename,
    # parent dir for new). `buf` is the user-edited text. `needs_focus` is a
    # one-shot the first draw consumes to grab keyboard focus. `target is None`
    # = the input is closed.
    target: Path | None = None
    buf: str = ""
    needs_focus: bool = False

    def open(self, target: Path, buf: str = "") -> None:
        self.target = target
        self.buf = buf
        self.needs_focus = True

    def close(self) -> None:
        self.target = None
        self.buf = ""
        self.needs_focus = False

    @property
    def is_open(self) -> bool:
        return self.target is not None
