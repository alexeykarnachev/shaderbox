"""The UI kit imports without the app behind it.

`theme`, `ui_primitives` and `notifications` are reused by other projects
(a themed window, the button tiers, toasts). That only works while they
depend on nothing the app happens to have installed -- and the drift is
invisible from inside this repo, where every dependency is present and
every module imports.

Measured in a SUBPROCESS: this process has already imported the app, so
`sys.modules` here answers about the test run rather than about the kit.
"""

import subprocess
import sys

# What a host installing the bare core does NOT get. An import of one of
# these from the kit's own closure is the regression: it means the widgets
# now need a dependency that lives behind an extra.
BEHIND_AN_EXTRA = (
    "openai",
    "telegram",
    "cv2",
    "jedi",
    "googleapiclient",
    "google_auth_oauthlib",
    "clang_format",
)

# The surface another project consumes. Named one by one rather than as a
# module import, so deleting or renaming one fails here rather than in the
# consumer.
KIT = """
from shaderbox.theme import apply_theme, COLOR, SIZE, SPACE, fade
from shaderbox.ui_primitives import (
    standard_button, primary_button, toggle_button, danger_button,
    chip_button, play_stop_toggle, caption_text, wrapped_caption,
    help_marker, text_tab_row, label_row,
)
from shaderbox.notifications import Notifications
"""


def _kit_closure() -> set[str]:
    """Every module a fresh interpreter loads to import the kit."""
    probe = (
        KIT + "\nimport sys\nprint(' '.join(sorted(m for m in sys.modules "
        "if not m.startswith('_'))))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, f"the kit did not import alone:\n{done.stderr}"
    return set(done.stdout.split())


def test_the_kit_pulls_nothing_that_lives_behind_an_extra() -> None:
    loaded = _kit_closure()
    tops = {name.split(".")[0] for name in loaded}
    intruders = sorted(tops & set(BEHIND_AN_EXTRA))
    assert not intruders, (
        f"importing the kit loaded {intruders}, which a host installing the "
        "bare core does not have -- the widgets grew a dependency that lives "
        "behind an optional extra"
    )


def test_the_kit_pulls_no_app_module() -> None:
    """The kit is three modules. Anything else is the app leaking into it.

    A shaderbox module with no third-party dependency still costs the
    consumer: it is the path by which one arrives later. `theme` importing
    the editor and `ui_primitives` importing the render plan is how the
    closure reached nine modules before this was split.
    """
    ours = sorted(m for m in _kit_closure() if m.startswith("shaderbox"))
    assert ours == [
        "shaderbox",
        "shaderbox.notifications",
        "shaderbox.theme",
        "shaderbox.ui_primitives",
    ], f"the kit's closure grew past its three modules: {ours}"
