"""Shared test fixtures. The `app` fixture builds a real headless App against a THROWAWAY tmp
project (never the tracked projects/dev sandbox — tests must not read or mutate it), seeded with
the three shipped template nodes so there is always a loadable current node."""

import contextlib
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from shaderbox.constants import NODE_TEMPLATES_DIR, STARTER_TEMPLATE_ID, TEMPLATE_ORDER


def seed_tmp_project(tmp_path: Path) -> Path:
    # A throwaway project dir seeded with the shipped template nodes copied out of resources.
    project = tmp_path / "project"
    nodes = project / "nodes"
    nodes.mkdir(parents=True)
    for tid in TEMPLATE_ORDER:
        shutil.copytree(NODE_TEMPLATES_DIR / tid, nodes / tid)
    return project


@pytest.fixture
def app(tmp_path: Path) -> Iterator[Any]:
    glfw = pytest.importorskip("glfw")
    if not glfw.init():
        pytest.skip("no GL")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    from shaderbox.app import App

    project = seed_tmp_project(tmp_path)
    a = App(project_dir=project)
    # No main loop in a test: run every marshalled bridge op INLINE (already on the GL thread).
    a.copilot.bridge.run_on_main = lambda fn, timeout=None, defer=False: fn()  # type: ignore[method-assign]
    a.set_current_node_id(STARTER_TEMPLATE_ID)
    a.ui_nodes[
        STARTER_TEMPLATE_ID
    ].node.render()  # warm the GL program (matches the live loop)
    yield a
    with contextlib.suppress(Exception):
        a.release()
