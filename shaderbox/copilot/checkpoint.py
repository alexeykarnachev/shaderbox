"""Per-turn rollback checkpoints for the copilot (feature 020·30).

A `TurnCheckpoint` records, for one user turn, what the copilot mutated so a later Revert can
restore the pre-turn state. The capture is BEST-EFFORT (a capture failure never fails the edit —
feature 020·30 decision 10) and runs main-thread inside the backend's bridge `_on_main` blocks,
keyed on the active turn id.

This module owns only the DATA (the index + the per-node serialized snapshots on disk under
`<checkpoints_root>/<turn_id>/`). The RESTORE orchestration is App-side (`App.restore_checkpoint`)
because reload-and-replace touches live `Node` / GL / editor state the backend can't reach
(decisions 1-2). A captured node snapshot is a full `save_ui_node` serialize of the LIVE node — NOT
a copy of the possibly-stale on-disk dir — because `set_uniform` writes only in-memory
`uniform_values`, never `node.json` (decision 2).
"""

import hashlib
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from shaderbox.ui_models import UINode

# A node snapshot subdir name is the node id; a lib snapshot is stored by a sanitized address.
_LIB_SNAPSHOT_SUBDIR = "_lib"
_INDEX_BASENAME = (
    "checkpoint.json"  # the self-describing index beside a turn's snapshots
)


def _lib_snapshot_name(ws_address: str) -> str:
    # "lib:math/transform.glsl" -> a flat, collision-free name under the turn's _lib subdir. The
    # address hash disambiguates distinct addresses that sanitize to the same readable stem
    # (e.g. "a/b.glsl" vs "a_b.glsl").
    rel = ws_address[len("lib:") :] if ws_address.startswith("lib:") else ws_address
    stem = "".join(c if c.isalnum() or c in "-_." else "_" for c in rel)
    digest = hashlib.sha1(ws_address.encode("utf-8")).hexdigest()[:8]
    return f"{stem}.{digest}"


@dataclass
class RevertResult:
    """Outcome of a restore, for the chat notice + the confirm modal's preview."""

    restored_nodes: list[str] = field(default_factory=list)  # names
    deleted_nodes: list[str] = field(default_factory=list)  # names of reverted-creates
    recovered_nodes: list[str] = field(
        default_factory=list
    )  # names of reverted-deletes
    reverted_libs: list[str] = field(default_factory=list)  # addresses
    unrestorable: list[str] = field(
        default_factory=list
    )  # nodes with no/failed snapshot

    @property
    def touched_anything(self) -> bool:
        return bool(
            self.restored_nodes
            or self.deleted_nodes
            or self.recovered_nodes
            or self.reverted_libs
        )

    def as_notice(self) -> str:
        # The NL note shown in chat + handed to the agent after a revert.
        if not self.touched_anything and not self.unrestorable:
            return "Reverted: nothing to restore for that turn."
        parts: list[str] = []
        if self.restored_nodes:
            parts.append(f"restored {', '.join(self.restored_nodes)}")
        if self.recovered_nodes:
            parts.append(f"recovered {', '.join(self.recovered_nodes)}")
        if self.deleted_nodes:
            parts.append(f"removed {', '.join(self.deleted_nodes)}")
        if self.reverted_libs:
            parts.append(f"reverted library {', '.join(self.reverted_libs)}")
        notice = (
            "Reverted that turn's changes: " + "; ".join(parts) + "." if parts else ""
        )
        if self.unrestorable:
            gap = " " if notice else ""
            notice += (
                f"{gap}Could not restore {', '.join(self.unrestorable)} (no snapshot)."
            )
        return notice


@dataclass
class TurnCheckpoint:
    """One user turn's rollback record. `turn_id` is the stable key tying it to its user Message.
    The snapshot files live under `root/turn_id/`; this object is the index (persisted)."""

    turn_id: str
    root: Path  # <project>/copilot/checkpoints
    user_excerpt: str = (
        ""  # first line of the user text, for the confirm modal + notice
    )
    # node_id -> the node's CURRENT name at capture (for the modal); its snapshot is at
    # root/turn_id/<node_id>/ (a full save_ui_node serialize of the pre-edit LIVE node).
    snapshotted_nodes: dict[str, str] = field(default_factory=dict)
    # node ids this turn CREATED (reverse = delete-to-trash; no snapshot exists).
    created_nodes: list[str] = field(default_factory=list)
    # node_id -> trash_name for nodes this turn DELETED (reverse = restore from trash).
    deleted_nodes: dict[str, str] = field(default_factory=dict)
    # lib ws_address -> pre-edit bytes captured (snapshot at root/turn_id/_lib/<safe-name>).
    snapshotted_libs: dict[str, str] = field(default_factory=dict)
    # lib ws_addresses this turn CREATED (reverse = delete the file; no pre-edit bytes exist).
    created_libs: list[str] = field(default_factory=list)
    # The current-node id before this turn's first switch_node (reverse = switch back). "" = unset.
    pre_switch_node_id: str | None = None
    # NAMES of nodes whose capture FAILED (decision 10) — Revert reports them un-restorable.
    failed_nodes: list[str] = field(default_factory=list)

    @property
    def turn_dir(self) -> Path:
        return self.root / self.turn_id

    def has_changes(self) -> bool:
        return bool(
            self.snapshotted_nodes
            or self.created_nodes
            or self.deleted_nodes
            or self.snapshotted_libs
            or self.created_libs
            or self.failed_nodes
            or self.pre_switch_node_id is not None
        )

    def snapshot_node(
        self, node_id: str, node: UINode, save_into: Callable[[UINode, Path], object]
    ) -> None:
        # Capture the LIVE node once per turn (first touch wins — later edits don't re-snapshot).
        # Best-effort: a failure is logged + recorded, never raised, so the edit proceeds.
        if node_id in self.snapshotted_nodes or node_id in self.created_nodes:
            return
        try:
            dest = self.turn_dir / node_id
            dest.mkdir(parents=True, exist_ok=True)
            save_into(node, dest)
            self.snapshotted_nodes[node_id] = node.ui_state.ui_name
        except Exception as e:
            logger.warning(
                f"copilot checkpoint: failed to snapshot node {node_id}: {e}"
            )
            name = node.ui_state.ui_name
            if name not in self.failed_nodes:
                self.failed_nodes.append(name)

    def mark_created(self, node_id: str) -> None:
        if node_id not in self.created_nodes:
            self.created_nodes.append(node_id)

    def record_deleted(self, node_id: str, trash_name: str) -> None:
        # A node this turn created-then-deleted nets to "create" (reverse = stay deleted); else
        # record the trash_name so revert restores it.
        if node_id in self.created_nodes:
            return
        self.deleted_nodes.setdefault(node_id, trash_name)

    def mark_created_lib(self, ws_address: str) -> None:
        if ws_address not in self.created_libs:
            self.created_libs.append(ws_address)

    def snapshot_lib(self, ws_address: str, pre_edit_source: str) -> None:
        if ws_address in self.snapshotted_libs or ws_address in self.created_libs:
            return
        try:
            lib_dir = self.turn_dir / _LIB_SNAPSHOT_SUBDIR
            lib_dir.mkdir(parents=True, exist_ok=True)
            snap = lib_dir / _lib_snapshot_name(ws_address)
            snap.write_text(pre_edit_source, encoding="utf-8")
            self.snapshotted_libs[ws_address] = snap.name
        except Exception as e:
            logger.warning(
                f"copilot checkpoint: failed to snapshot lib {ws_address}: {e}"
            )

    def record_pre_switch(self, current_node_id: str) -> None:
        # Only the FIRST switch of the turn matters (reverse = the node current before the turn).
        if self.pre_switch_node_id is None:
            self.pre_switch_node_id = current_node_id

    def lib_snapshot_text(self, ws_address: str) -> str | None:
        name = self.snapshotted_libs.get(ws_address)
        if name is None:
            return None
        snap = self.turn_dir / _LIB_SNAPSHOT_SUBDIR / name
        return snap.read_text(encoding="utf-8") if snap.exists() else None

    def node_snapshot_dir(self, node_id: str) -> Path | None:
        d = self.turn_dir / node_id
        return d if d.is_dir() else None

    def save_index(self) -> None:
        # Persist the self-describing index beside the snapshots, so a restart rehydrates without
        # a parallel structure in ConversationStore that could desync from the on-disk dirs.
        try:
            self.turn_dir.mkdir(parents=True, exist_ok=True)
            data = {
                "turn_id": self.turn_id,
                "user_excerpt": self.user_excerpt,
                "snapshotted_nodes": self.snapshotted_nodes,
                "created_nodes": self.created_nodes,
                "deleted_nodes": self.deleted_nodes,
                "snapshotted_libs": self.snapshotted_libs,
                "created_libs": self.created_libs,
                "pre_switch_node_id": self.pre_switch_node_id,
                "failed_nodes": self.failed_nodes,
            }
            (self.turn_dir / _INDEX_BASENAME).write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
        except Exception as e:
            logger.warning(
                f"copilot checkpoint: failed to save index {self.turn_id}: {e}"
            )

    @classmethod
    def load(cls, turn_dir: Path, root: Path) -> "TurnCheckpoint | None":
        # Rehydrate from a turn dir's index. Returns None (skipped) on a missing/corrupt index.
        index = turn_dir / _INDEX_BASENAME
        if not index.exists():
            return None
        try:
            data = json.loads(index.read_text(encoding="utf-8"))
            return cls(
                turn_id=str(data["turn_id"]),
                root=root,
                user_excerpt=str(data.get("user_excerpt", "")),
                snapshotted_nodes=dict(data.get("snapshotted_nodes", {})),
                created_nodes=list(data.get("created_nodes", [])),
                deleted_nodes=dict(data.get("deleted_nodes", {})),
                snapshotted_libs=dict(data.get("snapshotted_libs", {})),
                created_libs=list(data.get("created_libs", [])),
                pre_switch_node_id=data.get("pre_switch_node_id"),
                failed_nodes=list(data.get("failed_nodes", [])),
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"copilot checkpoint: unreadable index at {turn_dir}: {e}")
            return None

    def delete_files(self) -> None:
        # Drop the on-disk snapshots (the index entry is dropped by the owner).
        shutil.rmtree(self.turn_dir, ignore_errors=True)


class CheckpointStore:
    """Owns the active turn's checkpoint + the index of sealed ones, keyed by turn id. The session
    opens a checkpoint at turn start; the backend writes captures into the ACTIVE one (main-thread,
    inside the bridge `_on_main` blocks); `ui.py` seals it at turn-done. A captured edit's own turn
    is the only one writable — a deferred op from a prior turn can't leak in (it addresses no active
    checkpoint). App reads sealed checkpoints by turn id for restore."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._active: TurnCheckpoint | None = None
        self._sealed: dict[str, TurnCheckpoint] = self._rehydrate()

    def _rehydrate(self) -> dict[str, TurnCheckpoint]:
        # Load persisted checkpoints from disk on session init, so Revert works after a restart.
        out: dict[str, TurnCheckpoint] = {}
        if not self._root.is_dir():
            return out
        for turn_dir in self._root.iterdir():
            if not turn_dir.is_dir():
                continue
            cp = TurnCheckpoint.load(turn_dir, self._root)
            if cp is not None:
                out[cp.turn_id] = cp
        return out

    def open(self, turn_id: str, user_excerpt: str) -> None:
        self._active = TurnCheckpoint(
            turn_id=turn_id, root=self._root, user_excerpt=user_excerpt
        )

    @property
    def active(self) -> TurnCheckpoint | None:
        return self._active

    def seal(self) -> None:
        # Keep only a checkpoint that recorded a real change; an empty one (a read-only turn) is
        # collected immediately so it never shows a dead Revert button.
        cp = self._active
        self._active = None
        if cp is None:
            return
        if cp.has_changes():
            cp.save_index()
            self._sealed[cp.turn_id] = cp
        else:
            cp.delete_files()

    def get(self, turn_id: str) -> TurnCheckpoint | None:
        return self._sealed.get(turn_id)

    def drop(self, turn_id: str) -> None:
        # After a successful revert (or a prune): forget + delete the snapshots.
        cp = self._sealed.pop(turn_id, None)
        if cp is not None:
            cp.delete_files()

    def prune_to(self, live_turn_ids: set[str]) -> None:
        # Retention: drop any sealed checkpoint whose user Message is no longer in the live
        # conversation (decision 14).
        for tid in [t for t in self._sealed if t not in live_turn_ids]:
            self.drop(tid)

    def clear(self) -> None:
        # Clear-the-chat (decision 4): delete every snapshot outright, no archive.
        for cp in self._sealed.values():
            cp.delete_files()
        self._sealed.clear()
        if self._active is not None:
            self._active.delete_files()
            self._active = None
        shutil.rmtree(self._root, ignore_errors=True)

    def sealed_ids(self) -> list[str]:
        return list(self._sealed)
