"""The 019 region system is gone (069 W-E D4), and the flags that outlived it stayed.

Two halves. The NEGATIVE one greps for retired names, which pyright cannot see inside a
comment or a string. The POSITIVE one asserts every container hosting a focusable widget
still carries `no_nav_inputs`: imgui runs basic Tab traversal regardless of
`nav_enable_keyboard`, so that flag is what keeps Tab off the panel's sliders and the grid's
tiles, and a later reader who reads "nav is off" as "nav flags are dead" would ship focus
walking the document tiles with every other gate green.
"""

import ast
from pathlib import Path

from imgui_bundle import imgui

# `no_nav_inputs` and `nav_enable_keyboard` are deliberately NOT banned: the first
# STAYS at every focusable container (it stops Tab, which nav-off does not), and the
# second is named by smoke.py's inverted assertion.
# `region_` as a bare prefix is also unsafe and is not banned: `get_content_region_avail`
# appears about fifteen times and `scripts/dogfood/judge.py` defines `region_diff`.
# Do not "tighten" this tuple to a prefix.
_BANNED = (
    "ActiveRegion",
    "active_region",
    "region_focus_pending",
    "cycle_region",
    "CYCLE_REGION",
    "_set_region",
    "region_derive_allowed",
    "region_outline_visible",
    "focus_move_in_flight",
    "_yield_editor_to_region",
    "active_region_outline",
    "nav_flatten",
    "nav_flattened",
    "config_nav_escape_clear_focus_item",
)

_FOCUSABLE = (
    "input_text",
    "input_text_multiline",
    "input_int",
    "input_float",
    "drag_int",
    "drag_float",
    "slider_int",
    "slider_float",
    "checkbox",
    "combo",
    "selectable",
    "button",
)

_MODULES = ("ui.py", "widgets/copilot_chat.py", "widgets/document_grid.py")

_CONTAINERS = ("begin_child", "begin")

_MIN_CONTAINERS = 8


def test_no_source_file_mentions_the_region_system() -> None:
    hits: list[str] = []
    for path in (*Path("shaderbox").rglob("*.py"), Path("scripts/smoke.py")):
        text = path.read_text()
        for name in _BANNED:
            if name in text:
                hits.append(f"{path}: {name}")
    assert not hits, "\n".join(hits)


def _is_container_call(node: ast.AST) -> str | None:
    """The string-literal id of a `begin_child` / `begin` call, else None."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr not in _CONTAINERS:
        return None
    if not node.args:
        return None
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return None


def _resolve_flags(
    node: ast.AST, assignments: dict[str, ast.AST], depth: int = 0
) -> str:
    """Flatten a flags expression to source-ish text, following a Name through the
    module's own assignments — three of the eight containers reach their flag through a
    variable (`panel_flags`, `grid_flags`, the chat's `flags = _WINDOW_FLAGS | ...`), so a
    check reading only inline text scores correct code as unflagged."""
    if depth > 4:
        return ""
    if isinstance(node, ast.Name):
        target = assignments.get(node.id)
        if target is None:
            return node.id
        return node.id + " " + _resolve_flags(target, assignments, depth + 1)
    parts = [ast.dump(node)]
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in assignments:
            parts.append(_resolve_flags(child, assignments, depth + 1))
    return " ".join(parts)


def _module_assignments(tree: ast.Module) -> dict[str, ast.AST]:
    out: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
        ):
            out[node.target.id] = node.value
    return out


def _container_withs(tree: ast.Module) -> list[tuple[ast.With, str, int, ast.Call]]:
    """Every `with`-statement whose item is a container call with a literal id."""
    found: list[tuple[ast.With, str, int, ast.Call]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        for item in node.items:
            name = _is_container_call(item.context_expr)
            if name is not None and isinstance(item.context_expr, ast.Call):
                found.append((node, name, item.context_expr.lineno, item.context_expr))
    return found


def _has_flag(call: ast.Call, assignments: dict[str, ast.AST]) -> bool:
    for kw in call.keywords:
        if kw.arg not in ("window_flags", "flags"):
            continue
        if "no_nav_inputs" in _resolve_flags(kw.value, assignments):
            return True
    return False


def _calls_focusable(body: list[ast.stmt], excluded: set[int]) -> bool:
    for stmt in body:
        for node in ast.walk(stmt):
            if id(node) in excluded:
                continue
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in _FOCUSABLE:
                return True
    return False


def test_the_focusable_widget_names_are_real() -> None:
    missing = [name for name in _FOCUSABLE if not hasattr(imgui, name)]
    assert not missing, f"not imgui functions (typo or rename?): {missing}"


def _descendant_nodes(body: list[ast.stmt]) -> set[int]:
    out: set[int] = set()
    for stmt in body:
        for node in ast.walk(stmt):
            out.add(id(node))
    return out


def test_every_child_hosting_a_focusable_widget_blocks_tab() -> None:
    violations: list[str] = []
    total = 0
    for rel in _MODULES:
        path = Path("shaderbox") / rel
        tree = ast.parse(path.read_text())
        assignments = _module_assignments(tree)
        containers = _container_withs(tree)
        total += len(containers)
        # Parsed ONCE per module: a re-parse inside the check gives every node a fresh
        # identity, so the ancestry test never matches and every container is skipped.
        body_nodes = {
            id(with_node): _descendant_nodes(with_node.body)
            for with_node, _n, _l, _c in containers
        }
        flagged = {
            id(with_node)
            for with_node, _n, _l, call in containers
            if _has_flag(call, assignments)
        }
        for with_node, name, line, _call in containers:
            key = id(with_node)
            # An enclosing flagged container covers this one.
            covered = key in flagged or any(
                other_key in flagged and key in body_nodes[other_key]
                for other_key in body_nodes
                if other_key != key
            )
            if covered:
                continue
            # A flagged DESCENDANT answers for its own widgets, not this container.
            excluded: set[int] = set()
            for other, _n, _l, _c in containers:
                other_key = id(other)
                if other_key == key or other_key not in flagged:
                    continue
                if other_key in body_nodes[key]:
                    excluded |= body_nodes[other_key] | {other_key}
            if _calls_focusable(with_node.body, excluded):
                violations.append(f"{rel}:{line} {name!r}")
    assert total >= _MIN_CONTAINERS, (
        f"the container walk found {total} containers, expected at least "
        f"{_MIN_CONTAINERS} — the walk narrowed its own domain"
    )
    assert not violations, (
        "these containers host a focusable widget with no no_nav_inputs on themselves "
        f"or an enclosing container, so Tab traverses them: {violations}"
    )
