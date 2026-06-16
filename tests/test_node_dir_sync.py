"""Per-frame node-dir auto-sync (sync_nodes_from_disk) — disk is the source of truth.

The watcher reconciles ui_nodes to nodes/ each frame: a dir added/removed/edited OUTSIDE the app
shows up without a manual reload. Driven through the real headless `app` fixture (GL context + live
editor sessions), since the value type is the reconciliation against the live App state, not pure
logic. Shader-text hot-reload is NOT covered here (it rides reload_node_if_changed); this owns dir
add/remove + node.json edits.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any

from shaderbox.constants import STARTER_TEMPLATE_ID, TEMPLATE_ORDER


def _bump_node_json(node_dir: Path) -> None:
    # Rewrite node.json (canvas_size tweak) with a guaranteed-newer mtime so the mtime diff fires
    # even on a coarse filesystem clock.
    meta_path = node_dir / "node.json"
    meta = json.loads(meta_path.read_text())
    meta["canvas_size"] = [123, 123]
    meta_path.write_text(json.dumps(meta, indent=4))
    future = meta_path.lstat().st_mtime + 100.0
    os.utime(meta_path, (future, future))


def test_added_dir_appears(app: Any) -> None:
    nodes_dir = app.paths.nodes_dir
    new_id = "externally-added-node"
    shutil.copytree(nodes_dir / STARTER_TEMPLATE_ID, nodes_dir / new_id)
    assert new_id not in app.ui_nodes

    app.session.sync_nodes_from_disk()

    assert new_id in app.ui_nodes
    assert app.ui_nodes[new_id].node.program is not None  # warm-compiled


def test_removed_dir_drops_node_and_editor(app: Any) -> None:
    # Open a tab for a non-current node, then delete its dir on disk: node + its editor tab go.
    victim = TEMPLATE_ORDER[1]
    app.ensure_shader_tab(victim)
    assert any(t.node_id == victim for t in app.editor_tabs)

    shutil.rmtree(app.paths.nodes_dir / victim)
    app.session.sync_nodes_from_disk()

    assert victim not in app.ui_nodes
    assert not any(t.node_id == victim for t in app.editor_tabs)


def test_removed_current_dir_reselects(app: Any) -> None:
    assert app.current_node_id == STARTER_TEMPLATE_ID
    shutil.rmtree(app.paths.nodes_dir / STARTER_TEMPLATE_ID)

    app.session.sync_nodes_from_disk()

    assert STARTER_TEMPLATE_ID not in app.ui_nodes
    assert app.current_node_id in app.ui_nodes  # fell back to a surviving node


def test_changed_node_json_reloads(app: Any) -> None:
    target = TEMPLATE_ORDER[1]
    assert tuple(app.ui_nodes[target].node.canvas.texture.size) != (123, 123)
    _bump_node_json(app.paths.nodes_dir / target)

    app.session.sync_nodes_from_disk()

    assert tuple(app.ui_nodes[target].node.canvas.texture.size) == (123, 123)


def test_quiet_frame_is_a_noop(app: Any) -> None:
    # No disk change → ui_nodes identity is untouched (no needless reload/release churn).
    before = {nid: id(n.node) for nid, n in app.ui_nodes.items()}
    app.session.sync_nodes_from_disk()
    after = {nid: id(n.node) for nid, n in app.ui_nodes.items()}
    assert before == after


def test_own_save_does_not_self_trigger(app: Any) -> None:
    # save_ui_node rebaselines the mtime cache, so the next sync must NOT read our own write back
    # as an external change and reload (which would churn the live node object).
    app.save_ui_node(app.ui_nodes[STARTER_TEMPLATE_ID])
    node_obj_id = id(app.ui_nodes[STARTER_TEMPLATE_ID].node)

    app.session.sync_nodes_from_disk()

    assert id(app.ui_nodes[STARTER_TEMPLATE_ID].node) == node_obj_id
