from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from shaderbox.editor.ffi import Editor
from shaderbox.shader_source import ShaderSource

EditorTabKind = Literal["shader", "script", "lib"]


@dataclass(frozen=True)
class EditorTab:
    # One open file in the code-editor's tab bar (feature 045; 048 collapsed to one script per document).
    # `path` is the on-disk file (the EditorSession key); the tab LABEL is derived from the document name
    # (tab_label), but the imgui `##id` keys on the stable path. `kind` selects the semantic
    # label + the error tint. For a script/shader/graph tab, `document_id` addresses the document; "" for
    # lib tabs. A `graph` tab has NO EditorSession: its path is the document's `graph.json`, which keys
    # every path-keyed pass-through, and nothing edits it as text (093 T1).
    path: Path
    kind: EditorTabKind
    document_id: str = ""


class TabRecord(BaseModel):
    """One open tab as it persists (093 W2-2), beside the live `EditorTab` it mirrors.

    A separate shape rather than making `EditorTab` a model: the live tab is a frozen
    dataclass used as a dict-free value all over the draw layer, and `path` on disk is a
    string so a saved state is readable and portable.
    """

    path: str
    kind: EditorTabKind
    document_id: str = ""


def tab_records(tabs: Iterable[EditorTab]) -> list[TabRecord]:
    """The persisted shape of the live tab list."""
    return [
        TabRecord(path=str(tab.path), kind=tab.kind, document_id=tab.document_id)
        for tab in tabs
    ]


def tabs_from_records(
    records: Sequence[TabRecord],
    document_ids: frozenset[str],
    exists: Callable[[Path], bool],
) -> list[EditorTab]:
    """The tabs to reopen, dropping every record that no longer addresses anything.

    A record survives only while its file is still on disk AND its document is loaded -- a
    lib tab carries `""` and so answers the document clause trivially. Dropping rather than
    repairing is the point: a tab pointing at a deleted pass would eat its own edits, and a
    tab of a document this project no longer has has nothing to draw.
    """
    kept: list[EditorTab] = []
    for record in records:
        path = Path(record.path)
        if not exists(path):
            continue
        if record.document_id and record.document_id not in document_ids:
            continue
        if any(tab.path == path for tab in kept):
            continue
        kept.append(
            EditorTab(path=path, kind=record.kind, document_id=record.document_id)
        )
    return kept


@dataclass(frozen=True)
class JumpRequest:
    path: Path
    line: int
    column: int


@dataclass(frozen=True)
class HoverMark:
    path: Path
    line: int


@dataclass(frozen=True)
class LookupPopup:
    # What `K` found for the word under the caret: a signature or declaration, and its doc.
    word: str
    signature: str
    doc: str


@dataclass
class EditorSession:
    # A live libeditor instance bound to a specific on-disk file. `source` is
    # the snapshot used to seed the editor; the editor's current text may diverge
    # from `source.text` until the next flush. `saved_undo` is the editor's
    # revision at last save — anything beyond it is unsaved. The revision RISES
    # across set_text, so a re-baseline reads it AFTER the set, never before.
    editor: Editor
    source: ShaderSource
    saved_undo: int
