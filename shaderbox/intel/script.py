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


_VEC_TYPES: dict[str, str] = {"Vec2": "vec2", "Vec3": "vec3", "Vec4": "vec4"}


def _literal_type(value: ast.expr) -> str | None:
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
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
        if value.func.id in _VEC_TYPES:
            return _VEC_TYPES[value.func.id]
        if value.func.id == "Array" and value.args:
            first = value.args[0]
            if isinstance(first, ast.List | ast.Tuple):
                return f"float[{len(first.elts)}]"
            if (
                isinstance(first, ast.BinOp)
                and isinstance(first.op, ast.Mult)
                and isinstance(first.left, ast.List)
                and isinstance(first.right, ast.Constant)
                and isinstance(first.right.value, int)
            ):
                return f"float[{len(first.left.elts) * first.right.value}]"
    return None


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
