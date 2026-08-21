"""Every worker↔main thread the app spawns must be abandonable at shutdown.

`conventions.md` states the teardown contract: `cancel_all()`/STOP to release blocked
waiters, then `join(timeout)`, and on a timeout you ABANDON the survivor. Abandoning only
works if the thread is a daemon — a non-daemon survivor is re-joined by interpreter
`_shutdown` and blocks process exit for as long as it stays blocked (the 043 headless hang).

The check enumerates the spawn sites from the source rather than from a hand-listed roster,
so a NEW worker defaults INTO the denominator instead of being silently outside it.
"""

import ast
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parent.parent / "shaderbox"


def _thread_spawns() -> list[tuple[str, int, bool | None]]:
    """(module, lineno, daemon-literal) for every `threading.Thread(...)` in the package."""
    found: list[tuple[str, int, bool | None]] = []
    for path in sorted(_PKG.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_thread = (
                isinstance(func, ast.Attribute) and func.attr == "Thread"
            ) or (isinstance(func, ast.Name) and func.id == "Thread")
            if not is_thread:
                continue
            daemon: bool | None = None
            for kw in node.keywords:
                if kw.arg == "daemon" and isinstance(kw.value, ast.Constant):
                    daemon = bool(kw.value.value)
            rel = path.relative_to(_PKG.parent).as_posix()
            found.append((rel, node.lineno, daemon))
    return found


def test_the_enumeration_actually_finds_the_known_workers() -> None:
    # Falsifier for the check itself: if the AST walk silently matched nothing, every
    # assertion below would pass vacuously (the "checker that narrows its own domain" family).
    # Both exporters spawn through the shared ExporterWorker, so worker.py is their site.
    spawns = _thread_spawns()
    modules = {module for module, _, _ in spawns}
    assert "shaderbox/copilot/session.py" in modules
    assert "shaderbox/exporters/worker.py" in modules


@pytest.mark.parametrize("module,lineno,daemon", _thread_spawns())
def test_worker_threads_are_daemon(module: str, lineno: int, daemon: bool | None) -> None:
    assert daemon is True, (
        f"{module}:{lineno} spawns a non-daemon worker. A worker abandoned after a "
        "join timeout is re-joined by interpreter shutdown and hangs process exit; "
        "the teardown contract in conventions.md requires daemon=True."
    )
