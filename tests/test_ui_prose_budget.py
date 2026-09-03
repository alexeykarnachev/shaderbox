"""Every fixed UI string in `shaderbox/` fits the word budget D1 states.

The budgets live in `.claude/skills/imgui-ui/SKILL.md` § 2 and in
`ai_docs/conventions.md ## Code rules`: a label is 1-2 words, an icon tooltip is the
control's name, a `help_marker` is one clause of at most 8 words, an empty state is at
most 4. This module walks the package's AST, scores every call that carries authored copy,
and fails the ones over budget or joining a second clause.

The domain is DERIVED, not hand-listed: every public `ui_primitives` function whose
signature carries a copy-bearing parameter is scored at that parameter, with its positional
index read from the signature, so a new helper defaults INTO the gate. Only the three
`imgui.*` calls and `text_colored` are explicit rows — imgui's own surface has no signature
this module can reflect over.

A string the walk cannot read (a name bound to a call, a helper forwarding its caller's
text) is UNMEASURABLE and must be written into `_UNMEASURABLE` with a reason; a string it
CAN read that stays over budget must be written into `_OVER_BUDGET` with its measured
count, which is what makes that exemption self-invalidating. Neither list can rot: every
entry must still name a site the walk finds.
"""

import ast
import inspect
from pathlib import Path
from types import FunctionType

import pytest

from shaderbox import ui_primitives
from shaderbox.popups.pass_settings import _FORMATS
from shaderbox.ui_primitives import label_row, row_label, small_caption

_PKG = Path(__file__).resolve().parent.parent / "shaderbox"

UNMEASURABLE = -1

# What a copy-bearing parameter NAME is worth, from § 2's budget table. A parameter named
# here in any `ui_primitives` signature puts that function into the gate's domain.
_PARAMETER_BUDGETS: dict[str, int] = {
    "tooltip": 5,
    "label": 2,
    "title": 2,
    "footer": 2,
    "text": 4,
    "caption": 4,
    "message": 4,
    "hint": 8,
}

# A BUTTON's label is an action phrase, not the noun a control label is: § 1's own tier
# examples are "Add" / "Create" / "Re-render", and the app's read as "Add to pack" /
# "Open a copy". § 2's table has no row for one, so the budget is stated here at 3 — long
# enough for a verb and its object, short enough to reject a sentence on a button.
_BUTTON_LABEL_BUDGET = 3
_BUTTON_TIERS: frozenset[str] = frozenset(
    {
        "button",
        "primary_button",
        "ghost_button",
        "danger_button",
        "toggle_button",
        "pill_button",
        "chip_button",
        "open_path_button",
        "open_url_button",
    }
)

# `ui_primitives` functions whose copy-bearing parameter is NOT authored UI copy. Each is
# excluded by name with its reason, and `test_every_copy_bearing_helper_is_in_the_domain`
# fails on a name here that no longer takes such a parameter, so this cannot rot either.
_NOT_UI_COPY: dict[str, str] = {
    "parse_markdown_lines": "a pure parser over a Help section's body, draws nothing",
    "markdown_text": "renders a Help section's body, which § 2 exempts as documentation",
    "modal_window": "an imgui popup ID, not a heading the user reads",
    "help_marker": "scored at 8 by an explicit row: a marker is one clause, not a readout",
}


def _derived_rows() -> list[tuple[str, str, int | None, int]]:
    """(call, parameter, positional index, budget) for every copy-bearing helper.

    The index comes from the real signature, so a parameter that is positionally
    reachable is scored positionally too — a keyword-only row is the only `None`.
    """
    rows: list[tuple[str, str, int | None, int]] = []
    for name, function in sorted(vars(ui_primitives).items()):
        if name.startswith("_") or name in _NOT_UI_COPY:
            continue
        if not isinstance(function, FunctionType):
            continue
        if function.__module__ != ui_primitives.__name__:
            continue
        for index, parameter in enumerate(
            inspect.signature(function).parameters.values()
        ):
            budget = _PARAMETER_BUDGETS.get(parameter.name)
            if budget is None:
                continue
            if name in _BUTTON_TIERS and parameter.name == "label":
                budget = _BUTTON_LABEL_BUDGET
            positional = None if parameter.kind is parameter.KEYWORD_ONLY else index
            rows.append((name, parameter.name, positional, budget))
    return rows


# imgui's own calls carry no signature this module can reflect over, so they stay explicit.
# `text_colored` is scored only when its color is `COLOR.FG_DIM` — the dim readout.
_IMGUI_ROWS: list[tuple[str, str, int | None, int]] = [
    # `help_marker`'s own budget is 8, not the 4 its `text` parameter name would give it:
    # § 2 grants the (?) marker one clause where a dim readout gets a fragment.
    ("help_marker", "text", 0, 8),
    ("set_tooltip", "text", 0, 5),
    ("separator_text", "label", 0, 2),
    ("text_colored", "text", 1, 4),
]

_SCORED: list[tuple[str, str, int | None, int]] = _IMGUI_ROWS + _derived_rows()

_CLAUSE_JOINERS: tuple[str, ...] = (";", " — ", " -- ")

# Sites the gate CANNOT measure, each with why. A `ui_primitives` entry is a shared helper
# forwarding a caller's text -- the CALLERS are the measured sites.
_UNMEASURABLE: dict[tuple[str, str], str] = {
    (
        "shaderbox/tabs/code.py",
        "_draw_lookup_popup",
    ): "the `K` lookup's signature and doc, read from the lib index and ENGINE_UNIFORM_DOCS",
    (
        "shaderbox/tabs/code.py",
        "_draw_candidate_doc",
    ): "the highlighted candidate's signature and doc, from the same tables as `K`",
    (
        "shaderbox/ui.py",
        "_draw_app_panel",
    ): "the channel view's name from CHANNEL_VIEW_LABELS; "
    "test_channel_view.py::test_every_label_is_within_the_control_budget measures it",
    ("shaderbox/popups/emoji_picker.py", "_draw_body"): "an emoji group/entry name",
    ("shaderbox/widgets/uniform.py", "_draw_pass_source"): "the producing pass's name",
    ("shaderbox/popups/lib_picker/preview.py", "draw_preview"): "the file's own path",
    (
        "shaderbox/popups/lib_picker/search.py",
        "draw_search_row",
    ): "the matched tag list, built per frame",
    (
        "shaderbox/popups/lib_picker/tree.py",
        "_draw_inline_new_input",
    ): "the caller's inline-input label",
    (
        "shaderbox/popups/lib_picker/tree.py",
        "_draw_function_leaf",
    ): "the function's own doc line, ellipsized per frame",
    (
        "shaderbox/popups/pass_settings.py",
        "_draw_target",
    ): "the format table's tooltip, reached by subscript; "
    "test_the_format_tooltips_are_within_the_help_budget measures it directly",
    (
        "shaderbox/popups/settings.py",
        "_draw_copilot_config",
    ): "the copilot limits table's label and hint; a cost the reader is spending "
    "real money on, and copilot-llm-agent-design owns the wording",
    (
        "shaderbox/tabs/code.py",
        "draw_chrome",
    ): "the open file's path, tab label or error",
    ("shaderbox/tabs/document.py", "_draw_auto_block"): "the uniform's live value",
    ("shaderbox/tabs/document.py", "_entry_row_label"): "the caller's row label",
    (
        "shaderbox/exporters/telegram.py",
        "draw_config_ui",
    ): "the integration's own auth message, built per attempt",
    (
        "shaderbox/exporters/youtube.py",
        "draw_config_ui",
    ): "the integration's auth message and the paste button's state-dependent label",
    (
        "shaderbox/popups/settings.py",
        "_draw_keybindings",
    ): "each command spec's own label, from the command table",
    ("shaderbox/ui_primitives.py", "primary_button"): "forwards the caller's label",
    ("shaderbox/ui_primitives.py", "button"): "forwards the caller's label",
    ("shaderbox/ui_primitives.py", "ghost_button"): "forwards the caller's label",
    ("shaderbox/ui_primitives.py", "toggle_button"): "forwards the caller's label",
    ("shaderbox/ui_primitives.py", "danger_button"): "forwards the caller's label",
    ("shaderbox/ui_primitives.py", "chip_button"): "forwards the caller's label",
    ("shaderbox/ui_primitives.py", "pill_button"): "forwards the caller's label",
    (
        "shaderbox/ui_primitives.py",
        "labeled_text_input",
    ): "forwards the caller's field caption",
    (
        "shaderbox/ui_primitives.py",
        "labeled_multiline_input",
    ): "forwards the caller's field caption",
    (
        "shaderbox/ui_primitives.py",
        "labeled_drag_float",
    ): "forwards the caller's field caption",
    (
        "shaderbox/ui_primitives.py",
        "labeled_combo",
    ): "forwards the caller's field caption",
    (
        "shaderbox/ui_primitives.py",
        "unconnected_gate",
    ): "forwards the caller's hint and action label",
    ("shaderbox/ui_primitives.py", "row_label"): "forwards the caller's label",
    ("shaderbox/ui_primitives.py", "_code_chip"): "a code span of the Help body",
    ("shaderbox/ui_primitives.py", "_chip_row"): "the caller's chips (a pass name)",
    (
        "shaderbox/widgets/copilot_chat.py",
        "_draw_result_widget",
    ): "the tool payload's own button label",
    (
        "shaderbox/widgets/copilot_chat.py",
        "_draw_message",
    ): "the conversation's own text",
    ("shaderbox/widgets/uniform.py", "_begin_ctrl"): "the uniform's live count suffix",
    ("shaderbox/ui_primitives.py", "play_stop_toggle"): "forwards the caller's tooltip",
    ("shaderbox/ui_primitives.py", "clipped_caption"): "forwards the caller's text",
    ("shaderbox/ui_primitives.py", "setup_steps"): "forwards the step's own url",
    ("shaderbox/ui_primitives.py", "small_caption"): "forwards the caller's text",
    ("shaderbox/ui_primitives.py", "gauge_bar"): "forwards the caller's tooltip",
    ("shaderbox/ui_primitives.py", "label_row"): "forwards the caller's label",
    (
        "shaderbox/ui_primitives.py",
        "draw_copyable_text",
    ): "forwards the caller's tooltip",
    ("shaderbox/ui_primitives.py", "clickable_label"): "forwards the caller's tooltip",
    (
        "shaderbox/widgets/copilot_chat.py",
        "_tooltip_stat_row",
    ): "the caller's stat label",
    (
        "shaderbox/widgets/copilot_chat.py",
        "_draw_snippet_tooltip",
    ): "the turn's own token numbers",
    (
        "shaderbox/widgets/copilot_chat.py",
        "_draw_top_bar",
    ): "the context gauge's readout",
    ("shaderbox/widgets/details.py", "draw_file_details"): "the file's own path",
    (
        "shaderbox/widgets/document_grid.py",
        "draw_document_preview_button",
    ): "the document's own name",
    ("shaderbox/widgets/pass_list.py", "_draw_pass_tile"): "the pass's own name",
    ("shaderbox/widgets/uniform.py", "uniform_name_label"): "the uniform's own name",
    ("shaderbox/widgets/uniform.py", "draw_ui_uniform"): "the uniform's live value",
}

# Sites the gate CAN measure, that are over budget, and that stay. Each entry carries the
# measured word count, so a rewrite that changes the string changes this line too.
_OVER_BUDGET: dict[tuple[str, str, int], str] = {
    (
        "shaderbox/exporters/telegram.py",
        "_draw_status_slot",
        9,
    ): "a derived stat line: five interpolations around four authored words; "
    "revisit if a third stat joins it",
    (
        "shaderbox/exporters/youtube.py",
        "_draw_controls",
        4,
    ): "a link's destination name, not a control label",
    (
        "shaderbox/popups/help.py",
        "_draw_body",
        15,
    ): "a disabled-state reason; a control's name cannot carry why it is greyed",
    (
        "shaderbox/popups/lib_picker/__init__.py",
        "_draw_body",
        11,
    ): "the same disabled state on the picker's Insert button",
    (
        "shaderbox/popups/settings.py",
        "_draw_body",
        3,
    ): "derived: an exporter's name joined to its own unavailable reason",
    (
        "shaderbox/widgets/copilot_chat.py",
        "_draw_revert_modal",
        20,
    ): "a destructive-confirm body: what a revert will undo, which § 2's table has no "
    "row for and a 4-word fragment cannot state without misleading",
    (
        "shaderbox/widgets/copilot_chat.py",
        "_draw_turn_snippet",
        5,
    ): "a derived turn readout: the waiting clock around two authored words",
    (
        "shaderbox/widgets/copilot_chat.py",
        "_draw_turn_snippet",
        9,
    ): "a derived turn readout: tool/token/cost figures around three authored words",
}


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _enclosing_function(
    node: ast.AST, parents: dict[ast.AST, ast.AST]
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    current: ast.AST | None = node
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
        current = parents.get(current)
    return None


def _visible(text: str) -> str:
    """The part of a widget string a person reads: imgui treats `##` and after as the ID."""
    return text.split("##", 1)[0]


def _score(node: ast.expr) -> int:
    """Word count of an authored-copy expression, or UNMEASURABLE."""
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, str):
            return UNMEASURABLE
        return len(_visible(node.value).split())
    if isinstance(node, ast.JoinedStr):
        return _score_joined(node)
    if isinstance(node, ast.IfExp):
        return _worst([_score(node.body), _score(node.orelse)])
    if isinstance(node, (ast.List, ast.Tuple)):
        return _worst([_score(element) for element in node.elts])
    return UNMEASURABLE


def _score_joined(node: ast.JoinedStr) -> int:
    """An f-string's words, counting each interpolation as one and stopping at `##`."""
    total = 0
    for part in node.values:
        if isinstance(part, ast.FormattedValue):
            total += 1
        elif isinstance(part, ast.Constant) and isinstance(part.value, str):
            visible = _visible(part.value)
            total += len(visible.split())
            if visible != part.value:
                break
        else:
            return UNMEASURABLE
    return total


def _worst(scores: list[int]) -> int:
    if not scores or UNMEASURABLE in scores:
        return UNMEASURABLE
    return max(scores)


def _text_of(node: ast.expr) -> str:
    """The concatenated authored text of a scorable expression (for the clause check)."""
    if isinstance(node, ast.Constant):
        return _visible(node.value) if isinstance(node.value, str) else ""
    if isinstance(node, ast.JoinedStr):
        return _visible(
            "".join(
                part.value
                for part in node.values
                if isinstance(part, ast.Constant) and isinstance(part.value, str)
            )
        )
    if isinstance(node, ast.IfExp):
        return f"{_text_of(node.body)}\n{_text_of(node.orelse)}"
    if isinstance(node, (ast.List, ast.Tuple)):
        return "\n".join(_text_of(element) for element in node.elts)
    return ""


def _resolve(
    arg: ast.expr, enclosing: ast.FunctionDef | ast.AsyncFunctionDef | None
) -> ast.expr | None:
    """A `Name` argument resolved to its string assignment in the enclosing function.

    Returns None when the name cannot be resolved to authored copy — including a name the
    function ever appends to, which is longer than any one binding shows.
    """
    if not isinstance(arg, ast.Name):
        return arg
    if enclosing is None:
        return None
    for node in ast.walk(enclosing):
        if (
            isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == arg.id
        ):
            return None
    candidates: list[ast.expr] = [
        node.value
        for node in ast.walk(enclosing)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == arg.id
        and isinstance(node.value, (ast.Constant, ast.JoinedStr, ast.IfExp))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda node: (_score(node), 0))


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _is_dim(call: ast.Call) -> bool:
    """`text_colored`'s first argument is `COLOR.FG_DIM` — the dim-readout budget."""
    if not call.args:
        return False
    first = call.args[0]
    return isinstance(first, ast.Attribute) and first.attr == "FG_DIM"


class Site:
    def __init__(
        self,
        module: str,
        function: str,
        lineno: int,
        call: str,
        parameter: str,
        budget: int,
        words: int,
        text: str,
        expression: str,
        is_label: bool,
        has_interpolation: bool,
    ) -> None:
        self.module: str = module
        self.function: str = function
        self.lineno: int = lineno
        self.call: str = call
        self.parameter: str = parameter
        self.budget: int = budget
        self.words: int = words
        self.text: str = text
        self.expression: str = expression
        self.is_label: bool = is_label
        self.has_interpolation: bool = has_interpolation

    @property
    def key(self) -> tuple[str, str]:
        return (self.module, self.function)

    def __repr__(self) -> str:
        return (
            f"{self.module}::{self.function}:{self.lineno} {self.call}.{self.parameter}"
        )


def _has_interpolation(node: ast.expr) -> bool:
    if isinstance(node, ast.JoinedStr):
        return any(isinstance(part, ast.FormattedValue) for part in node.values)
    if isinstance(node, ast.IfExp):
        return _has_interpolation(node.body) or _has_interpolation(node.orelse)
    if isinstance(node, (ast.List, ast.Tuple)):
        return any(_has_interpolation(element) for element in node.elts)
    return False


def _collect() -> list[Site]:
    sites: list[Site] = []
    for path in sorted(_PKG.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        parents = _parents(tree)
        module = path.relative_to(_PKG.parent).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            if name is None:
                continue
            for call, parameter, index, budget in _SCORED:
                if call != name:
                    continue
                if call == "text_colored" and not _is_dim(node):
                    continue
                arg: ast.expr | None = None
                if index is not None and len(node.args) > index:
                    arg = node.args[index]
                else:
                    for keyword in node.keywords:
                        if keyword.arg == parameter:
                            arg = keyword.value
                    # A call that omits an optional parameter carries no string.
                    if arg is None and not _supplies(node, parameter, index):
                        continue
                enclosing = _enclosing_function(node, parents)
                function = enclosing.name if enclosing is not None else "<module>"
                resolved = _resolve(arg, enclosing) if arg is not None else None
                words = _score(resolved) if resolved is not None else UNMEASURABLE
                sites.append(
                    Site(
                        module=module,
                        function=function,
                        lineno=node.lineno,
                        call=call,
                        parameter=parameter,
                        budget=budget,
                        words=words,
                        text=_text_of(resolved) if resolved is not None else "",
                        expression=ast.unparse(arg) if arg is not None else "**kwargs",
                        is_label=call in ("label_row", "row_label"),
                        has_interpolation=(
                            _has_interpolation(resolved)
                            if resolved is not None
                            else False
                        ),
                    )
                )
    return sites


def _supplies(call: ast.Call, parameter: str, index: int | None) -> bool:
    """Whether the call supplies `parameter` at all — by position, keyword, or `**kwargs`."""
    if index is not None and len(call.args) > index:
        return True
    for keyword in call.keywords:
        if keyword.arg == parameter or keyword.arg is None:
            return True
    return False


_SITES: list[Site] = _collect()
_MEASURABLE: list[Site] = [s for s in _SITES if s.words != UNMEASURABLE]
_UNREADABLE: list[Site] = [s for s in _SITES if s.words == UNMEASURABLE]
_EXEMPT: set[tuple[str, str, int]] = set(_OVER_BUDGET)


def _budgeted() -> list[Site]:
    return [s for s in _MEASURABLE if (s.module, s.function, s.words) not in _EXEMPT]


def _ids(sites: list[Site]) -> list[str]:
    return [repr(s) for s in sites]


def test_the_walk_finds_the_known_call_sites() -> None:
    # The collector's own falsifier: without it every parametrized assertion below would
    # pass vacuously on an empty collection — the "checker that narrows its own domain"
    # family. The floor sits well under today's count so a legitimate new string never
    # trips it; only a collector that broke does.
    modules = {site.module for site in _SITES}
    assert "shaderbox/popups/pass_settings.py" in modules
    assert "shaderbox/widgets/pass_list.py" in modules
    assert "shaderbox/tabs/document.py" in modules
    assert "shaderbox/ui_primitives.py" in modules
    assert len(_SITES) >= 60, f"the walk found only {len(_SITES)} sites"


@pytest.mark.parametrize("site", _budgeted(), ids=_ids(_budgeted()))
def test_every_measured_site_is_within_budget(site: Site) -> None:
    assert site.words <= site.budget, (
        f"{site.module}::{site.function}:{site.lineno} {site.call}({site.parameter}=) "
        f"is {site.words} words against a budget of {site.budget}: {site.text!r}"
    )


@pytest.mark.parametrize("site", _budgeted(), ids=_ids(_budgeted()))
def test_no_scored_string_joins_a_second_clause(site: Site) -> None:
    for joiner in _CLAUSE_JOINERS:
        assert joiner not in site.text, (
            f"{site.module}::{site.function}:{site.lineno} joins a second clause with "
            f"{joiner!r}: {site.text!r}. D1 is one clause, not one sentence."
        )


@pytest.mark.parametrize(
    "site",
    [s for s in _MEASURABLE if s.is_label],
    ids=_ids([s for s in _MEASURABLE if s.is_label]),
)
def test_no_label_carries_an_interpolation(site: Site) -> None:
    assert not site.has_interpolation, (
        f"{site.module}::{site.function}:{site.lineno} puts an interpolation in a label: "
        f"{site.expression}. A label column is fixed-width; derived values go in the control."
    )


@pytest.mark.parametrize("site", _UNREADABLE, ids=_ids(_UNREADABLE))
def test_every_unmeasurable_site_is_listed(site: Site) -> None:
    assert site.key in _UNMEASURABLE, (
        f"{site.module}::{site.function}:{site.lineno} {site.call}({site.parameter}=) "
        f"passes {site.expression}, which the gate cannot read. Make it a literal, or add "
        f"{site.key} to _UNMEASURABLE with the reason."
    )


def test_no_site_is_both_measured_and_unmeasurable_listed() -> None:
    measured_keys = {site.key for site in _MEASURABLE}
    unreadable_keys = {site.key for site in _UNREADABLE}
    overlap = set(_UNMEASURABLE) & measured_keys - unreadable_keys
    assert not overlap, (
        f"{sorted(overlap)} are listed unmeasurable but every site there was read. "
        "An entry that suppresses a measurable function is a hole, not an exemption."
    )


@pytest.mark.parametrize("key", sorted(_UNMEASURABLE), ids=lambda k: f"{k[0]}::{k[1]}")
def test_every_unmeasurable_entry_still_names_a_real_site(key: tuple[str, str]) -> None:
    assert key in {site.key for site in _UNREADABLE}, (
        f"{key} no longer names an unmeasurable site; delete the entry."
    )


@pytest.mark.parametrize(
    "key", sorted(_OVER_BUDGET), ids=lambda k: f"{k[0]}::{k[1]}:{k[2]}"
)
def test_every_over_budget_entry_still_names_a_real_site(
    key: tuple[str, str, int],
) -> None:
    module, function, words = key
    matches = [
        site
        for site in _MEASURABLE
        if site.module == module and site.function == function
    ]
    assert matches, f"{module}::{function} has no measurable site; delete the entry."
    assert any(site.words == words for site in matches), (
        f"{module}::{function} no longer has a site at {words} words "
        f"(found {sorted({site.words for site in matches})}); update or delete the entry."
    )


def test_a_keyword_supplied_argument_is_scored() -> None:
    sites = _sites_of('help_marker(text="a b c d e f g h i")')
    assert [site.words for site in sites] == [9]


def test_an_argument_supplied_by_neither_position_nor_keyword_is_unmeasurable() -> None:
    sites = _sites_of("help_marker(**kwargs)")
    assert [site.words for site in sites] == [UNMEASURABLE]


def test_an_ifexp_argument_scores_its_worst_branch() -> None:
    sites = _sites_of('help_marker("a" if p else "b c d e f g h i j")')
    assert [site.words for site in sites] == [9]


def test_a_name_rebound_by_augassign_is_not_resolved() -> None:
    source = 'def f():\n    text = "a"\n    text += " b c"\n    help_marker(text)\n'
    sites = _sites_of(source, wrap=False)
    assert [site.words for site in sites] == [UNMEASURABLE]


def _sites_of(source: str, wrap: bool = True) -> list[Site]:
    """Run the collector's per-call logic over a source fixture."""
    tree = ast.parse(f"def f():\n    {source}\n" if wrap else source)
    parents = _parents(tree)
    sites: list[Site] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        for call, parameter, index, budget in _SCORED:
            if call != name:
                continue
            arg: ast.expr | None = None
            if index is not None and len(node.args) > index:
                arg = node.args[index]
            else:
                for keyword in node.keywords:
                    if keyword.arg == parameter:
                        arg = keyword.value
                if arg is None and not _supplies(node, parameter, index):
                    continue
            enclosing = _enclosing_function(node, parents)
            resolved = _resolve(arg, enclosing) if arg is not None else None
            words = _score(resolved) if resolved is not None else UNMEASURABLE
            sites.append(
                Site(
                    module="<fixture>",
                    function=enclosing.name if enclosing is not None else "<module>",
                    lineno=node.lineno,
                    call=call,
                    parameter=parameter,
                    budget=budget,
                    words=words,
                    text=_text_of(resolved) if resolved is not None else "",
                    expression=ast.unparse(arg) if arg is not None else "**kwargs",
                    is_label=call in ("label_row", "row_label"),
                    has_interpolation=(
                        _has_interpolation(resolved) if resolved is not None else False
                    ),
                )
            )
    return sites


def _copy_bearing_helpers() -> list[tuple[str, str]]:
    """(function, parameter) for every `ui_primitives` helper taking authored copy."""
    found: list[tuple[str, str]] = []
    for name, function in sorted(vars(ui_primitives).items()):
        if name.startswith("_") or not isinstance(function, FunctionType):
            continue
        if function.__module__ != ui_primitives.__name__:
            continue
        for parameter in inspect.signature(function).parameters.values():
            if parameter.name in _PARAMETER_BUDGETS:
                found.append((name, parameter.name))
    return found


@pytest.mark.parametrize(
    "helper", _copy_bearing_helpers(), ids=lambda h: f"{h[0]}.{h[1]}"
)
def test_every_copy_bearing_helper_is_in_the_domain(helper: tuple[str, str]) -> None:
    # A new `ui_primitives` helper taking a tooltip/label/text must default INTO the gate.
    # The whole table is derived, so this can only fail when a name is exempted in
    # `_NOT_UI_COPY` — which also means a stale exemption there goes red rather than
    # silently widening the hole it was written for.
    name, parameter = helper
    if name in _NOT_UI_COPY:
        pytest.skip(f"{name} is exempted: {_NOT_UI_COPY[name]}")
    assert any(
        call == name and scored == parameter for call, scored, _, _ in _SCORED
    ), (
        f"{name}({parameter}=) takes authored copy but is outside _SCORED, so every "
        "caller's string there is unmeasured. The table is derived — check _NOT_UI_COPY."
    )


@pytest.mark.parametrize("name", sorted(_NOT_UI_COPY), ids=sorted(_NOT_UI_COPY))
def test_every_domain_exemption_still_names_a_copy_bearing_helper(name: str) -> None:
    assert name in {helper for helper, _ in _copy_bearing_helpers()}, (
        f"{name} no longer takes a copy-bearing parameter; delete its _NOT_UI_COPY entry."
    )


def test_a_helper_drawing_its_own_tooltip_is_still_measured_at_its_callers() -> None:
    # The shape the domain fix exists for: a helper that renders copy through
    # `begin_tooltip`/`text_unformatted` rather than a scored call. `help_marker` IS that
    # shape — its body draws the tooltip itself — and reflection puts it in the domain by
    # its SIGNATURE, so its callers are scored no matter how it draws.
    source = (_PKG / "ui_primitives.py").read_text()
    body = source[source.index("def help_marker(") :]
    body = body[: body.index("\ndef ")]
    assert "begin_tooltip" in body and "text_unformatted" in body, (
        "help_marker no longer draws its own tooltip; pick another anchor for this test"
    )
    assert "set_tooltip(" not in body, (
        "help_marker now forwards to set_tooltip, so it is no longer this shape"
    )
    assert any(call == "help_marker" for call, _, _, _ in _SCORED)
    callers = [
        site for site in _SITES if site.call == "help_marker" and site.words >= 0
    ]
    assert callers, "no help_marker caller was scored"


def test_every_scored_row_matches_the_real_signature() -> None:
    # Generalizes the label-helper check to all sixteen derived rows: a row with an index
    # must name the parameter AT that index, and an index of None must name a genuinely
    # keyword-only parameter — otherwise a positional caller produces no row at all.
    for call, parameter, index, _ in _SCORED:
        function = getattr(ui_primitives, call, None)
        if not isinstance(function, FunctionType):
            continue
        parameters = list(inspect.signature(function).parameters.values())
        names = [p.name for p in parameters]
        assert parameter in names, f"{call} no longer takes {parameter}"
        actual = names.index(parameter)
        if index is None:
            assert parameters[actual].kind is parameters[actual].KEYWORD_ONLY, (
                f"{call}.{parameter} is reachable positionally at {actual} but the row "
                "reads it by keyword only, so a positional caller is skipped."
            )
        else:
            assert index == actual, (
                f"{call}.{parameter} sits at argument {actual}, not {index}"
            )


def test_the_label_helpers_are_read_at_the_right_argument() -> None:
    # The gate reads a label at argument 1 because the font comes first. A reorder would
    # move every label out of the measured position and the gate would go green measuring
    # nothing, which is how this wave's own first census produced 17 false unmeasurables.
    for function, index, expected in (
        (label_row, 1, "label"),
        (row_label, 1, "label"),
        (small_caption, 1, "text"),
    ):
        parameters = list(inspect.signature(function).parameters)
        assert parameters[0] == "font", (
            f"{function.__name__} no longer takes font first"
        )
        assert parameters[index] == expected, (
            f"{function.__name__}'s argument {index} is {parameters[index]}, not {expected}"
        )


def test_the_format_tooltips_are_within_the_help_budget() -> None:
    # `_FORMATS` is reached through a Subscript, so no call-site walk can read it. Any
    # future table of UI strings needs its own direct assertion for the same reason.
    for code, label, tooltip in _FORMATS:
        assert len(label.split()) <= 2, f"{code}'s menu label is over budget: {label!r}"
        assert len(tooltip.split()) <= 8, (
            f"{code}'s tooltip is over budget: {tooltip!r}"
        )
        for joiner in _CLAUSE_JOINERS:
            assert joiner not in tooltip, f"{code}'s tooltip joins a second clause"
