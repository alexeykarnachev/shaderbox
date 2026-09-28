"""The UI asks `ProjectSession`, never the collaborator it owns.

`dev_flow.md`'s module map says the UI reads the script engine VIA `ProjectSession`, and
`ProjectSession` forwards every script verb -- `script_status`, `has_script`,
`script_path_for`, `read_script_source` and the rest. Three sites in `tabs/code.py`
reached two attributes deep to `app.session.script_engine.*` instead, while the same file
used the forwarders correctly seven lines away. The inconsistency inside one file is what
made it a gap rather than a convention.

Nothing enforced it, so this does. A rule with no check is a wish.
"""

import ast
from pathlib import Path

_UI_DIRS = ("tabs", "widgets", "popups")
_SRC = Path(__file__).resolve().parent.parent / "shaderbox"

# What `ProjectSession` OWNS and forwards. Reaching through the session to one of these
# binds a UI module to a collaborator's interface that the session is free to reshape.
_OWNED = frozenset({"script_engine", "copilot_backend", "app_state"})


def test_no_ui_module_reaches_through_the_session_to_what_it_owns() -> None:
    """Falsifier: write `app.session.script_engine.anything(...)` in a UI module.

    Matches `<...>.session.<owned>.<...>`, the two-deep reach. A UI module naming
    `app.session.<verb>()` is the sanctioned form and does not match, and so does
    `ProjectSession`'s own `self.script_engine.*`, which is not a UI module at all.
    """
    files = [
        path
        for directory in _UI_DIRS
        for path in (_SRC / directory).rglob("*.py")
    ] + [_SRC / "ui.py"]
    assert files, "the fixture found no UI modules -- it never looked at anything"

    offenders: list[str] = []
    for path in sorted(files):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr in _OWNED
                and isinstance(node.value.value, ast.Attribute)
                and node.value.value.attr == "session"
            ):
                offenders.append(
                    f"{path.relative_to(_SRC)}:{node.lineno} "
                    f"session.{node.value.attr}.{node.attr}"
                )

    assert not offenders, (
        "a UI module reaches past ProjectSession to what it owns; add a forwarding "
        "method on ProjectSession and call that:\n  " + "\n  ".join(offenders)
    )
