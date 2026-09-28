"""No test permanently patches an object the `app` fixture shares by identity.

`conftest._APP_INIT_OWNS` names the fields the per-test baseline restore deliberately
leaves alone, because collaborators hold the reference and replacing one breaks the
sharing. `notifications` is one of them. A bare `app.notifications.push = <lambda>` in a
test therefore outlives that test and silently swallows every later test's notifications
-- the suite stays green, because a test that asserts a notification ARRIVED is rare and
the ones that do run earlier.

Three tests had done it, and the cost was paid by a fourth: a new test asserting that a
failed save reports itself passed alone and failed in the full suite, which reads as a
bug in the new code rather than in the old fixture.

`monkeypatch.setattr` is the fix -- it undoes the patch at teardown.
"""

import ast
from pathlib import Path

_TESTS = Path(__file__).resolve().parent


def test_no_test_assigns_over_a_shared_app_collaborator() -> None:
    """Falsifier: write `app.notifications.push = lambda ...` in any test file.

    Matches an assignment whose target is an attribute reached through one of the shared
    objects, which is the shape that leaks. `monkeypatch.setattr(app.notifications, ...)`
    is a CALL, not an assignment, so it does not match.
    """
    shared = {"notifications", "shader_lib_files", "exporter_registry", "profiler"}
    offenders: list[str] = []
    for path in sorted(_TESTS.rglob("test_*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Attribute)
                    and target.value.attr in shared
                ):
                    offenders.append(
                        f"{path.name}:{node.lineno} "
                        f"{target.value.attr}.{target.attr} = ..."
                    )
    assert not offenders, (
        "a permanent patch on a session-shared object -- use monkeypatch.setattr:\n  "
        + "\n  ".join(offenders)
    )
