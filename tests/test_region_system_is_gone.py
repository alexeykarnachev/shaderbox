"""The 019 region system is gone (069 W-E D4), and the flags that outlived it stayed.

Two halves. The NEGATIVE one greps for retired names, which pyright cannot see inside a
comment or a string. The POSITIVE one asserts that every container reaching a Tab stop still
carries `no_nav_inputs`: imgui runs basic Tab traversal regardless of `nav_enable_keyboard`,
so that flag is what keeps Tab off the panel's sliders, and a later reader who reads "nav is
off" as "nav flags are dead" would ship focus walking them with every other gate green.

The positive half FOLLOWS CALLS, transitively, because none of `ui.py`'s six containers writes
a widget call inline -- every one draws through a free function, often several deep
(`document_settings` -> `_NODE_TABS` -> `tabs/document.py::draw` -> the uniform rows), so a
check reading only each `with` body scores all six clean and the three `ui.py` flags are
unguarded. It also follows `ui_primitives` wrappers, derived from that module by AST rather
than listed, and callees parked in a module-level table, which is the only route from
`document_settings` to the sliders. The walk stops at a callee that opens its own top-level
window or popup: that is a separate Tab ring the caller's flag cannot reach.

WHAT IT COVERS, measured rather than claimed: of the five `no_nav_inputs` sites, deleting the
flag from `ui.py`'s `document_settings` or from `copilot_chat.py`'s `_WINDOW_FLAGS` turns this
test red. The other three -- `code_editor`, `copilot_bar`, `document_preview_grid` -- stay
green, because none of them contains a Tab stop TODAY: the editor is a rendered image, the bar
holds only buttons, and `preview_cell` holds a `selectable`. That is not a hole in the walk,
it is the truth about those containers, and it follows from a second measurement: with
`nav_enable_keyboard` OFF, Tab lands only on text-entry widgets (`_FOCUSABLE`), never on
`button` / `checkbox` / `combo` / `selectable`. So those three flags are defensive -- they
cost nothing and they pre-empt the day someone adds an input there, at which point this test
starts guarding them automatically. Manual verification covers them meanwhile.
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

# The imgui widgets Tab actually lands on with `nav_enable_keyboard` OFF. Measured on a
# headless rig rather than assumed: an anchor input focused, one Tab, does the candidate take
# focus? Every text-entry widget does; `checkbox`, `combo`, `selectable` and `button` do NOT
# (they are nav stops only under nav-on, which this app does not run). So the grid's flag is
# earned by the panel's sliders, not by `preview_cell`'s selectable tiles.
_FOCUSABLE = (
    "input_text",
    "input_text_multiline",
    "input_int",
    "input_float",
    "drag_int",
    "drag_float",
    "slider_int",
    "slider_float",
)

# Widgets that are focusable under nav-ON only. Not part of the assertion, but named so a
# later reader does not "restore" them without re-running the measurement above.
_NAV_ONLY_FOCUSABLE = ("checkbox", "combo", "selectable", "button")

_PKG = Path("shaderbox")
_MODULES = ("ui.py", "widgets/copilot_chat.py", "widgets/document_grid.py")
_CONTAINERS = ("begin_child", "begin")
_MIN_CONTAINERS = 8


def test_no_source_file_mentions_the_region_system() -> None:
    hits: list[str] = []
    for path in (*_PKG.rglob("*.py"), Path("scripts/smoke.py")):
        text = path.read_text()
        for name in _BANNED:
            if name in text:
                hits.append(f"{path}: {name}")
    assert not hits, "\n".join(hits)


def test_the_focusable_widget_names_are_real() -> None:
    # imgui-bundle exports no __all__, so hasattr is the check: a typo or an upstream
    # rename must fail here rather than silently matching nothing.
    missing = [name for name in _FOCUSABLE if not hasattr(imgui, name)]
    assert missing == [], f"not imgui functions (typo or rename?): {missing}"


# --------------------------------------------------------------------------- AST helpers


def _parse(rel: str) -> ast.Module:
    return ast.parse((_PKG / rel).read_text())


def _called_names(node: ast.AST) -> set[str]:
    """Every callee under `node`, as a bare name (`draw_x`) or a dotted tail
    (`code_tab.draw` -> `code_tab.draw`, `imgui.button` -> `imgui.button`)."""
    out: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Name):
            out.add(func.id)
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            out.add(f"{func.value.id}.{func.attr}")
            out.add(func.attr)
        elif isinstance(func, ast.Attribute):
            out.add(func.attr)
    return out


def _module_imports(tree: ast.Module) -> dict[str, str]:
    """Local name -> the `shaderbox/`-relative module file it came from, for both
    `from shaderbox.x import f` and `from shaderbox.tabs import code as code_tab`."""
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module is None or not node.module.startswith("shaderbox"):
                continue
            base = node.module.removeprefix("shaderbox").strip(".").replace(".", "/")
            for alias in node.names:
                local = alias.asname or alias.name
                as_module = f"{base}/{alias.name}.py" if base else f"{alias.name}.py"
                if (_PKG / as_module).is_file():
                    out[local] = as_module
                elif base and (_PKG / f"{base}.py").is_file():
                    out[local] = f"{base}.py"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if not alias.name.startswith("shaderbox"):
                    continue
                rel = alias.name.removeprefix("shaderbox").strip(".").replace(".", "/")
                if (_PKG / f"{rel}.py").is_file():
                    out[alias.asname or alias.name] = f"{rel}.py"
    return out


def _functions(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }


def _ui_primitives_focusables() -> set[str]:
    """Every `ui_primitives` function that reaches a bare imgui focusable, directly or
    through a sibling. Derived, never listed: a new wrapper defaults INTO the domain."""
    tree = _parse("ui_primitives.py")
    funcs = _functions(tree)
    calls = {name: _called_names(node) for name, node in funcs.items()}
    focusable = {
        name
        for name, called in calls.items()
        if called & {f"imgui.{f}" for f in _FOCUSABLE}
    }
    while True:
        grown = {
            name
            for name, called in calls.items()
            if name not in focusable and called & focusable
        }
        if not grown:
            return focusable
        focusable |= grown


def _resolve_flags(
    node: ast.AST, assignments: dict[str, ast.AST], depth: int = 0
) -> str:
    """Flatten a flags expression to text, following a Name through the module's own
    assignments -- the chat reaches its flag through `flags = _WINDOW_FLAGS | ...`, so a
    check reading only inline text scores correct code as unflagged."""
    if depth > 4:
        return ""
    if isinstance(node, ast.Name):
        target = assignments.get(node.id)
        if target is None:
            return node.id
        return f"{node.id} {_resolve_flags(target, assignments, depth + 1)}"
    parts = [ast.dump(node)]
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in assignments:
            parts.append(_resolve_flags(child, assignments, depth + 1))
    return " ".join(parts)


def _table_callees(tree: ast.Module) -> dict[str, set[str]]:
    """Table name -> the function references parked in it. `ui.py`'s `_NODE_TABS` holds
    `document_tab.draw` / `render_tab.draw` / `share_tab.draw`, which the panel invokes as
    `draw_tab(app)` after unpacking the table -- a name the walk can never resolve by itself,
    and the only route from `document_settings` to the uniform sliders. Keyed by table so a
    container inherits only the tables it actually iterates."""
    out: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = [x.id for x in targets if isinstance(x, ast.Name)]
        if not names or node.value is None:
            continue
        refs: set[str] = set()
        for element in ast.walk(node.value):
            if isinstance(element, ast.Attribute) and isinstance(
                element.value, ast.Name
            ):
                refs.add(f"{element.value.id}.{element.attr}")
        if refs:
            for name in names:
                out[name] = refs
    return out


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
    found: list[tuple[ast.With, str, int, ast.Call]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        for item in node.items:
            call = item.context_expr
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            if not isinstance(func, ast.Attribute) or func.attr not in _CONTAINERS:
                continue
            if call.args and isinstance(call.args[0], ast.Constant):
                value = call.args[0].value
                if isinstance(value, str):
                    found.append((node, value, call.lineno, call))
    return found


def _has_flag(call: ast.Call, assignments: dict[str, ast.AST]) -> bool:
    return any(
        kw.arg in ("window_flags", "flags")
        and "no_nav_inputs" in _resolve_flags(kw.value, assignments)
        for kw in call.keywords
    )


def _names_used(body: list[ast.stmt], excluded: set[int]) -> set[str]:
    return {
        node.id
        for stmt in body
        for node in ast.walk(stmt)
        if isinstance(node, ast.Name) and id(node) not in excluded
    }


def _descendants(body: list[ast.stmt]) -> set[int]:
    return {id(node) for stmt in body for node in ast.walk(stmt)}


# --------------------------------------------------------------- the reachability walk

_MAX_DEPTH = 6

# A callee that opens its own TOP-LEVEL window or popup starts a new Tab ring, so the walk
# stops there: the main window "reaches" the emoji picker's input only through
# `draw_emoji_picker`, whose `modal_window` is a separate popup the caller's flag cannot
# reach. A nested `begin_child` is NOT a boundary -- it inherits the ancestor's
# `no_nav_inputs`, which is the whole reason the ancestry rule below exists.
_WINDOW_OPENERS = ("begin", "modal_window", "begin_popup", "begin_popup_modal")


def _opens_a_window(node: ast.AST) -> bool:
    return bool(_called_names(node) & set(_WINDOW_OPENERS))


def _module_cache() -> dict[str, ast.Module]:
    return {}


def _resolve_callee(
    name: str, own: str, imports: dict[str, str], cache: dict[str, ast.Module]
) -> tuple[str, str] | None:
    """(module file, function name) for a callee, or None. A bare name resolves in the
    calling module first, then through its `from shaderbox.x import f` map; a dotted
    `code_tab.draw` resolves through the module-alias half of the same map."""
    head, _, tail = name.rpartition(".")
    for module, func in (
        (own, name),
        (imports.get(name, ""), tail),
        (imports.get(head, ""), tail),
    ):
        if not module or not func:
            continue
        tree = _load(module, cache)
        if tree is not None and func in _functions(tree):
            return module, func
    return None


def _load(rel: str, cache: dict[str, ast.Module]) -> ast.Module | None:
    if rel not in cache:
        path = _PKG / rel
        if not path.is_file():
            return None
        cache[rel] = ast.parse(path.read_text())
    return cache[rel]


def _reaches_focusable(
    called: set[str],
    module: str,
    primitives: set[str],
    cache: dict[str, ast.Module],
) -> str:
    """The first Tab stop this call set reaches, following calls transitively into any
    module under `shaderbox/`, or ''. None of `ui.py`'s containers writes a widget call
    inline, so without this the three flags in that file are unguarded."""
    bare = {f"imgui.{name}" for name in _FOCUSABLE}
    queue: list[tuple[str, set[str], str, int]] = [(module, called, "", 0)]
    seen: set[tuple[str, str]] = set()
    while queue:
        mod, names, trail, depth = queue.pop(0)
        imports = _module_imports(_load(mod, cache) or ast.Module([], []))
        for name in sorted(names):
            if name in bare:
                return f"{trail}{name}" if trail else name
            if name.rpartition(".")[2] in primitives:
                return f"{trail}ui_primitives.{name.rpartition('.')[2]}"
            if depth >= _MAX_DEPTH:
                continue
            found = _resolve_callee(name, mod, imports, cache)
            if found is None or found in seen:
                continue
            seen.add(found)
            target_mod, func = found
            tree = _load(target_mod, cache)
            if tree is None:
                continue
            body = _functions(tree)[func]
            if _opens_a_window(body):
                continue
            step = (
                f"{trail}{func} -> "
                if target_mod == mod
                else f"{trail}{target_mod}::{func} -> "
            )
            queue.append((target_mod, _called_names(body), step, depth + 1))
    return ""


def test_every_child_hosting_a_focusable_widget_blocks_tab() -> None:
    primitives = _ui_primitives_focusables()
    assert len(primitives) >= 3, (
        f"only {len(primitives)} ui_primitives wrappers reach a focusable widget; "
        "the derivation narrowed its own domain"
    )
    violations: list[str] = []
    total = 0
    cache: dict[str, ast.Module] = _module_cache()
    for rel in _MODULES:
        tree = _parse(rel)
        assignments = _module_assignments(tree)
        tables = _table_callees(tree)
        containers = _container_withs(tree)
        total += len(containers)
        # Parsed ONCE per module: a re-parse gives every node a fresh identity, so the
        # ancestry test never matches and every container is silently skipped.
        bodies = {id(w): _descendants(w.body) for w, _n, _l, _c in containers}
        flagged = {id(w) for w, _n, _l, c in containers if _has_flag(c, assignments)}
        for with_node, name, line, _call in containers:
            key = id(with_node)
            covered = key in flagged or any(
                other in flagged and key in bodies[other]
                for other in bodies
                if other != key
            )
            if covered:
                continue
            # A flagged DESCENDANT answers for its own widgets, not this container's.
            excluded: set[int] = set()
            for other, _n, _l, _c in containers:
                other_key = id(other)
                if (
                    other_key != key
                    and other_key in flagged
                    and other_key in bodies[key]
                ):
                    excluded |= bodies[other_key] | {other_key}
            called: set[str] = set()
            for stmt in with_node.body:
                for node in ast.walk(stmt):
                    if id(node) in excluded or not isinstance(node, ast.Call):
                        continue
                    called |= _called_names(ast.Expr(value=node))
            # A table this container iterates contributes its parked callees.
            from_tables = {
                ref
                for table, refs in tables.items()
                if table in _names_used(with_node.body, excluded)
                for ref in refs
            }
            hit = _reaches_focusable(called | from_tables, rel, primitives, cache)
            if hit:
                violations.append(f'{rel}:{line} "{name}" reaches {hit}')
    assert total >= _MIN_CONTAINERS, (
        f"the container walk found {total} containers, expected at least "
        f"{_MIN_CONTAINERS} -- the walk narrowed its own domain"
    )
    assert violations == [], (
        "these containers reach a focusable widget with no no_nav_inputs on themselves "
        f"or an enclosing container, so Tab traverses them: {violations}"
    )
