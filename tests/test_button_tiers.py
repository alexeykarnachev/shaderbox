"""The button system's two invariants (079 D12).

The maintainer's finding was that the app's buttons did not read as one family: a text-only
tier blurred into the prose around it, and enough sites bypassed the primitives that fixing
the tier would not have fixed the app. So the tier set is capped and the bypass is gated.
"""

import ast
import inspect
import textwrap
from pathlib import Path

from shaderbox import ui_primitives

_PACKAGE = Path(ui_primitives.__file__).parent

# The four tiers every labelled verb in the app takes (079 D12). Chips and pills are a
# different thing — a tag, a filter, a mode selector — and are counted separately.
_TIERS: frozenset[str] = frozenset(
    {"standard_button", "primary_button", "danger_button", "toggle_button"}
)

# Raw imgui button calls that are NOT a labelled verb, each with why. A new entry needs a
# reason of the same kind: the call draws something that is not a button with a word on it.
_NOT_A_VERB: dict[tuple[str, str], str] = {
    (
        "tabs/code.py",
        "invisible_button",
    ): "the editor's interaction surface — a hit rect, no label",
    (
        "ui.py",
        "invisible_button",
    ): "the editor/panel splitter's drag handle",
    (
        "widgets/copilot_chat.py",
        "invisible_button",
    ): "the chat's resize handle",
    (
        "popups/emoji_picker.py",
        "button",
    ): "one emoji cell in the glyph grid; the label IS the glyph",
    (
        "popups/lib_picker/tree.py",
        "small_button",
    ): "the per-row favorite star, which imgui-ui 7.4 keeps inline",
    (
        "exporters/telegram.py",
        "button",
    ): "an empty button the glyph font draws over (imgui-ui 5) and the carousel arrows",
}


def _raw_button_calls() -> dict[tuple[str, str], list[int]]:
    """Every `imgui.*button*` call under `shaderbox/`, keyed by (module, call name)."""
    found: dict[tuple[str, str], list[int]] = {}
    for path in sorted(_PACKAGE.rglob("*.py")):
        if path.name == "ui_primitives.py":
            continue  # the primitives are where a raw call belongs
        module = path.relative_to(_PACKAGE).as_posix()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            call = node.func
            if not isinstance(call, ast.Attribute) or "button" not in call.attr:
                continue
            if not isinstance(call.value, ast.Name) or call.value.id != "imgui":
                continue
            found.setdefault((module, call.attr), []).append(node.lineno)
    return found


def test_no_site_hand_rolls_a_button_outside_the_primitives() -> None:
    # Falsifier: put an `imgui.button("Save")` in any panel — it lands here with no reason
    # beside it, which is the point: a bypass is what let the app drift out of one family.
    unlisted = {
        where: lines
        for where, lines in _raw_button_calls().items()
        if where not in _NOT_A_VERB
    }
    assert not unlisted, (
        "raw imgui button calls outside ui_primitives.py: "
        + "; ".join(f"{m}::{c} at {lines}" for (m, c), lines in sorted(unlisted.items()))
        + " — use a button tier (079 D12), or list it in _NOT_A_VERB with why"
    )


def test_every_listed_exception_still_exists() -> None:
    # An allowlist that outlives its site is a rule nobody is following any more.
    present = set(_raw_button_calls())
    stale = sorted(where for where in _NOT_A_VERB if where not in present)
    assert not stale, f"delete these _NOT_A_VERB entries: {stale}"


def test_the_tier_set_stays_at_four() -> None:
    # 079 D12 caps it: a site that needs a fifth is a design question, not a new primitive.
    # Every labelled button primitive is either one of the four tiers, one of the two chip/pill
    # shapes (a tag or a filter, not a verb), or a thin wrapper that forwards to a tier.
    # Falsifier: add a fifth labelled primitive with its own styling and it lands in `extra`.
    labelled = {
        name: function
        for name, function in vars(ui_primitives).items()
        if name.endswith("_button")
        and not name.startswith("_")
        and "label" in inspect.signature(function).parameters
    }
    extra = {
        name
        for name, function in labelled.items()
        if name not in _TIERS
        and name not in {"chip_button", "pill_button"}
        and not _forwards_to_a_tier(function)
    }
    assert not extra, (
        f"{sorted(extra)} style their own buttons outside the four tiers "
        f"{sorted(_TIERS)} — pick a tier, or make the case for changing the tier set"
    )
    assert _TIERS <= set(labelled), (
        f"a tier went missing: {sorted(_TIERS - set(labelled))}"
    )


def _forwards_to_a_tier(function: object) -> bool:
    # A wrapper (`open_url_button`, `open_path_button`) calls a tier by name rather than
    # pushing its own colors, so it is a call site in primitive clothing.
    source = textwrap.dedent(inspect.getsource(function))
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _TIERS
        for node in ast.walk(ast.parse(source))
    )
