"""Shared test fixtures. The `app` fixture builds a real headless App against a THROWAWAY tmp
project (never the tracked projects/dev sandbox — tests must not read or mutate it), seeded with
ONLY the starter document (066 D4 — the fixture diet): most tests need one loadable current
document, and the example library still loads from resources regardless of the project seed. A
test that needs a second project document calls `seed_extra_document`."""

import contextlib
import copy
import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import moderngl
import pytest

from shaderbox.constants import DOCUMENT_EXAMPLES_DIR, STARTER_EXAMPLE_ID
from shaderbox.copilot.config import COPILOT_CONFIG

# LOAD-BEARING, read at GL-context creation (not at import): compiling this repo's #version 460
# shaders on a bare llvmpipe 4.5 context SEGFAULTS Mesa — see the Makefile's `test` note. `make
# test` exports them; these setdefaults make a bare `uv run pytest tests/` safe too.
os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "4.6")
os.environ.setdefault("MESA_GLSL_VERSION_OVERRIDE", "460")


def seed_tmp_project(tmp_path: Path) -> Path:
    # A throwaway project dir seeded with the starter document copied out of resources.
    project = tmp_path / "project"
    documents = project / "documents"
    documents.mkdir(parents=True)
    shutil.copytree(
        DOCUMENT_EXAMPLES_DIR / STARTER_EXAMPLE_ID, documents / STARTER_EXAMPLE_ID
    )
    return project


def seed_extra_document(app: Any, new_id: str) -> str:
    # Copy the starter document dir under a new id and sync it in — for tests that need a
    # second project document beside the starter-only default seed.
    documents = app.paths.documents_dir
    shutil.copytree(documents / STARTER_EXAMPLE_ID, documents / new_id)
    app.session.sync_documents_from_disk()
    assert new_id in app.ui_documents
    return new_id


@pytest.fixture(scope="module")
def gl_ctx() -> Iterator["moderngl.Context"]:
    """A standalone GL context, one per module.

    Default-backend on purpose: an EXPLICIT backend="egl" context released here poisons the
    process's EGL display, and the NEXT module's first program compile segfaults. The failure is
    module-order-only, so it survives a single-module run and appears in a full suite —
    one context recipe per process is the rule.

    Module-scoped rather than session-scoped because the modules that use it run in their own
    xdist processes; a wider scope would share a context across files that were partitioned
    apart deliberately.
    """
    try:
        context = moderngl.create_standalone_context()
    except Exception as e:
        pytest.skip(f"no standalone GL context available: {e}")
    yield context
    context.release()


def restart_app(app: Any) -> Any:
    """Reopen `app`'s own project the way a project switch does, and return the same App.

    A test asking "does this survive a restart?" used to tear the App down and build a second
    one, which a per-test fixture could afford and a reused one cannot — `shutdown()` destroys
    the process's imgui context and every later test in that worker dies on it. `_init` is the
    restore path either way: it releases the outgoing project, re-reads `app_state.json` and
    rebuilds the tabs from the records, which is precisely what the second App was being built
    to prove.
    """
    app._init(app.project_dir, persist_pointer=False)
    app.copilot.bridge.run_on_main = lambda fn, timeout=None, defer=False: fn()
    return app


@pytest.fixture(scope="session")
def _app_process(tmp_path_factory: Any) -> Iterator[Any]:
    """ONE App per xdist worker. Its window, GL context and imgui context are built once.

    The per-process constraint is the imgui font atlas: its GL texture dies with the App that
    built it, so a second App in one interpreter inherits a dead texture (`App.shutdown`'s
    docstring). That has always forced one App per process; what it never forced is one App per
    TEST. `App._init` is the project-switch path the running app takes — it releases the outgoing
    project and loads the incoming one, leaving the window and the imgui context alone — so the
    `app` fixture below reuses this one through that path and each test still gets a project
    nothing else has touched.
    """
    glfw = pytest.importorskip("glfw")
    if not glfw.init():
        pytest.skip("no GL")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    # app_data_dir() is where the shader library, the favorites and integrations.json live, and
    # App.__init__ SEEDS the library there. Without the override every test that builds an App
    # writes into the developer's own ~/.local/share/shaderbox. Set before the App import, since
    # the paths are read at call time but the seed runs in the constructor.
    data_dir = tmp_path_factory.mktemp("data")
    os.environ["SHADERBOX_DATA_DIR"] = str(data_dir)
    from shaderbox.app import App

    a = App(project_dir=seed_tmp_project(tmp_path_factory.mktemp("boot")))
    yield a
    with contextlib.suppress(Exception):
        a.shutdown()


# Fields the baseline leaves alone, for two reasons.
#
# `_init` rebuilds the first four for the incoming project, and a shallow copy of one is a
# DIFFERENT object holding the boot project's contents — restoring it hands `_init` a session
# whose documents are not the ones the App then reads.
#
# The rest are built once and handed to collaborators that keep the reference: replacing
# `notifications` with a copy leaves `shader_lib_files` pushing into the object the test is no
# longer watching. A field like this is shared by identity, so copying it breaks the sharing.
_APP_INIT_OWNS = frozenset(
    {
        "session",
        "checker_texture",
        "alpha_view",
        "rgb_view",
        "notifications",
        "shader_lib_files",
        "exporter_registry",
        "profiler",
        "python_worker",
    }
)


@pytest.fixture(scope="session")
def _app_baseline(_app_process: Any) -> dict[str, Any]:
    """The App's UI state as `__init__` left it, one shallow copy per field.

    Restoring this before each test is what makes ONE App behave like a fresh one. The
    alternative — resetting the handful of fields a failure happens to name — was tried and
    grew: an open modal, a queued project switch, a turn marked in flight, a graph view's
    selection, a copilot working set. They are all the same shape (state `__init__` sets and no
    project reload clears), so the fixture restores the whole class rather than its members.

    A field whose value will not copy is one of the built-once objects — the window, the fonts,
    the renderer, the worker — and those are what the reuse EXISTS to keep, so failing to copy
    is the right answer for them rather than an error.
    """
    baseline: dict[str, Any] = {}
    for field, value in vars(_app_process).items():
        if field in _APP_INIT_OWNS:
            continue
        try:
            baseline[field] = copy.copy(value)
        except Exception:
            continue
    return baseline


@pytest.fixture
def app(
    _app_process: Any, _app_baseline: dict[str, Any], monkeypatch: Any, tmp_path: Path
) -> Iterator[Any]:
    glfw = pytest.importorskip("glfw")
    a = _app_process
    monkeypatch.setenv("SHADERBOX_DATA_DIR", os.environ["SHADERBOX_DATA_DIR"])
    # A module that also takes `gl_ctx` leaves ITS standalone context current, and every GL
    # allocation below would land there (or fail, once it is released). `App.__init__` used to
    # re-make the window current on every build; reusing one App means the fixture does it.
    glfw.make_context_current(a.window)
    moderngl.init_context()
    # NOT "auto", which the app uses. Under auto, moderngl frees a dropped GL object from its
    # `__del__`, so the release runs on whatever thread Python happens to collect on. The app
    # only ever collects on its GL thread; here one App outlives every test in the worker, so
    # its objects survive until a collection that xdist's receiver thread can trigger -- and
    # freeing a GL object there, with no current context, segfaults the worker mid-test,
    # including in tests that touch no GL at all. `context_gc` queues the drops instead, and
    # the `gc()` below frees the queue on this thread, where the context is current.
    moderngl.get_context().gc_mode = "context_gc"
    moderngl.get_context().gc()
    for field, value in _app_baseline.items():
        setattr(a, field, copy.copy(value))
    a._init(seed_tmp_project(tmp_path), persist_pointer=False)
    # No main loop in a test: run every marshalled bridge op INLINE (already on the GL thread).
    a.copilot.bridge.run_on_main = lambda fn, timeout=None, defer=False: fn()
    # A turn opens a tool batch before its first edit; a test that calls an edit tool directly
    # never does, so the intra-batch rewrite guard would carry one test's target into the next
    # and refuse its rewrite as a stale duplicate.
    a.copilot_backend.batch_begin()
    a.set_current_document_id(STARTER_EXAMPLE_ID)
    a.ensure_shader_tab(STARTER_EXAMPLE_ID)
    a.ui_documents[
        STARTER_EXAMPLE_ID
    ].document.render()  # warm the GL program (matches the live loop)
    yield a


@pytest.fixture(autouse=True)
def _restore_copilot_config() -> Iterator[None]:
    # COPILOT_CONFIG is a process-wide mutable singleton, and loading ANY project pushes the
    # persisted per-user limits onto it (ProjectSession -> IntegrationsStore.apply_limits).
    # Nothing restores it, so a test that builds an App silently rewrites the config every
    # later test reads — which lets an assertion about a config default pass because an
    # earlier test repaired the value, and go red only when run alone.
    fields = [f for f in dir(COPILOT_CONFIG) if not f.startswith("_")]
    before = {f: getattr(COPILOT_CONFIG, f) for f in fields}
    yield
    for field, value in before.items():
        setattr(COPILOT_CONFIG, field, value)
