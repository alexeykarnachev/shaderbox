"""The document script read statically: the uniforms its `update` returns, with the GLSL type
each literal value implies -- the source of `SCRIPT_UNIFORM` candidates in a shader."""

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class ScriptReturn:
    name: str
    # The pass block the key sits in, or None for a bare key (every pass declaring it).
    pass_name: str | None
    # The GLSL type a literal value implies; None when the value's shape is not literal
    # (`self.phase`, a call, an expression) -- the name is still known.
    glsl_type: str | None
    # 0-based line of the key, for a jump.
    line: int


def _literal_length(value: ast.expr) -> int | None:
    # How many numbers a literal sequence holds: a list/tuple of them, or `[0.0] * N`. None when
    # the expression is anything else, so nothing is guessed from a computed value.
    if isinstance(value, ast.List | ast.Tuple):
        return len(value.elts)
    if (
        isinstance(value, ast.BinOp)
        and isinstance(value.op, ast.Mult)
        and isinstance(value.left, ast.List)
        and isinstance(value.right, ast.Constant)
        and isinstance(value.right.value, int)
    ):
        return len(value.left.elts) * value.right.value
    return None


def _literal_type(value: ast.expr) -> str | None:
    # The GLSL type a returned literal implies, so the shader side can offer the declaration.
    # A script returns plain Python, so the shape IS the literal: 2-4 numbers read as a vector,
    # any other length as an array, a str as a text array.
    if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub | ast.UAdd):
        return _literal_type(value.operand)
    if isinstance(value, ast.Constant):
        if isinstance(value.value, bool):
            return "bool"
        if isinstance(value.value, int):
            return "int"
        if isinstance(value.value, float):
            return "float"
        return None
    length = _literal_length(value)
    if length is None:
        return None
    # An empty literal names no shape, and `uniform float[0] u_x;` does not compile — offering it
    # as a one-click declaration hands the user a broken line.
    if length == 0:
        return None
    # A vector and a 2-4 element array are the same literal; the vector is what a script writing
    # `[x, y]` almost always means, and the array form spells its length out.
    if 2 <= length <= 4 and isinstance(value, ast.List | ast.Tuple):
        return f"vec{length}"
    # TODO: rows read as a vector by their OUTER length -- `[[1,2],[3,4]]` offers `vec2` where
    # `coerce_array` drives it as `vec2[2]`. This inference is meant to mirror that coercion;
    # aligning them wants the rule rethought against it, not a special case here.
    return f"float[{length}]"


def _update_function(tree: ast.Module) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "update":
            return node
    return None


def _returns_of(function: ast.FunctionDef) -> list[ast.Return]:
    # `update`'s own returns, not those of a function nested inside it.
    found: list[ast.Return] = []
    stack: list[ast.AST] = list(function.body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            continue
        if isinstance(node, ast.Return):
            found.append(node)
        stack.extend(ast.iter_child_nodes(node))
    found.sort(key=lambda r: r.lineno)
    return found


def returned_uniforms(text: str) -> tuple[ScriptReturn, ...]:
    """Every uniform name `update` returns, in source order; empty for a script that does
    not parse or has no `update`. A dict-valued entry is a pass block: its keys are the
    uniforms, scoped to that pass."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return ()
    update = _update_function(tree)
    if update is None:
        return ()
    found: list[ScriptReturn] = []
    for ret in _returns_of(update):
        if not isinstance(ret.value, ast.Dict):
            continue
        for key, value in zip(ret.value.keys, ret.value.values, strict=True):
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                continue
            if isinstance(value, ast.Dict):
                for inner_key, inner_value in zip(
                    value.keys, value.values, strict=True
                ):
                    if isinstance(inner_key, ast.Constant) and isinstance(
                        inner_key.value, str
                    ):
                        found.append(
                            ScriptReturn(
                                inner_key.value,
                                key.value,
                                _literal_type(inner_value),
                                inner_key.lineno - 1,
                            )
                        )
            else:
                found.append(
                    ScriptReturn(key.value, None, _literal_type(value), key.lineno - 1)
                )
    return tuple(found)
