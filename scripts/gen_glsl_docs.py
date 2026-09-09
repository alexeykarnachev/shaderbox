"""Generate shaderbox/glsl_docs.py from the Khronos OpenGL-Refpages XML.

The GLSL builtins a shader author calls -- their FULL overload sets and the spec's own
one-line purpose -- are data Khronos publishes, not something to type from memory. This
reads the gl4 refpages (one DocBook XML per page, a `funcprototype` per overload), which
is the vocabulary a `#version 460` shader may name, and emits a table the code panel reads
for `K` and for the completion popup's detail note.

An entry is named from its PROTOTYPE, never from the page's refname: a family page
(`packUnorm.xml`, `noise.xml`) carries a refname no prototype matches, and keying on the
refname drops such a page whole. A page that yields no entry at all is reported the way an
unparsable one is, so the table cannot shrink in silence.

A builtin VARIABLE (`gl_FragCoord` and kin) is declared in a `fieldsynopsis` rather than a
prototype; those pages fill a second table, filtered to the fragment stage by
`_FRAGMENT_VARIABLES` -- the refpages do not encode the stage, so that list is held here
with its citation while the declarations and the purpose still come from the page.

The keyword and type vocabulary comes from the editor library's own GLSL lexer
(`src/lex_glsl.odin` in the editor repo), which is the list that actually colors this
editor's text -- so completion offers exactly the words the highlighter knows. Neither half
is typed from memory.

Usage:
    git clone --depth 1 --filter=blob:none --sparse \\
        https://github.com/KhronosGroup/OpenGL-Refpages.git /tmp/refpages
    cd /tmp/refpages && git sparse-checkout set gl4
    uv run python scripts/gen_glsl_docs.py /tmp/refpages/gl4 ~/src/editor/src/lex_glsl.odin

Output (repo-anchored): shaderbox/glsl_docs.py
"""

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "shaderbox" / "glsl_docs.py"
DOCBOOK = "{http://docbook.org/ns/docbook}"
# The repo's ruff line length, which the emitted table is already formatted to.
_LINE_LENGTH = 88

# Pages in the refpages tree that document the API, not the shading language. The GLSL
# pages are the ones whose refentry carries a funcsynopsis with a funcprototype; a gl*
# entry point is documented in the same directory and must not reach a shader author's
# completion list.
_API_PREFIX = re.compile(r"^gl[A-Z]")

# The math entities the refpages use, as plain text. Any entity NOT listed here surfaces as
# a reported skip rather than a silent drop, so the table can never quietly shrink.
_ENTITIES: dict[str, str] = {
    # Enumerated from the corpus itself:
    #   grep -ohE '&[a-zA-Z][a-zA-Z0-9]*;' gl4/*.xml | sort -u
    # A page using an entity absent here fails to parse and is REPORTED, never dropped in
    # silence -- the generator exits non-zero if any page is skipped.
    "af": "",
    "amp": "&#38;",
    "delta": "delta",
    "Delta": "Delta",
    "ge": "&#8805;",
    "gt": "&#62;",
    "infin": "&#8734;",
    "it": "",
    "lambda": "lambda",
    "lceil": "ceil(",
    "lcub": "{",
    "le": "&#8804;",
    "lfloor": "floor(",
    "lt": "&#60;",
    "minus": "-",
    "nbsp": " ",
    "ne": "&#8800;",
    "plus": "+",
    "quot": "&#34;",
    "rceil": ")",
    "rfloor": ")",
    "sdot": "*",
    "times": "x",
    "VerticalBar": "|",
    "VerticalLine": "|",
}

# Stage-only verbs a fragment shader may not call. The gl4 tree documents the whole
# pipeline; each name here is excluded with the stage that owns it as the reason. Image,
# atomic and barrier functions are NOT excluded: the 4.60 spec's 7.1.5 states their
# behavior inside helper invocations, which is a fragment-stage fact.
_STAGE_ONLY: dict[str, str] = {
    "EmitVertex": "geometry",
    "EndPrimitive": "geometry",
    "EmitStreamVertex": "geometry",
    "EndStreamPrimitive": "geometry",
}

# Pages that document no name a shader can write, each with what it documents instead. A
# page absent here that yields neither a prototype nor a declaration is REPORTED, so a
# family page whose entries stop arriving cannot pass as one of these.
_NAMES_NOTHING: dict[str, str] = {
    "gl_PointSize.xml": "a gl_PerVertex member, declared in a programlisting block",
    "gl_Position.xml": "a gl_PerVertex member, declared in a programlisting block",
    "removedTypes.xml": "the API types removed in OpenGL 4.2, not a shader name",
}

# The fragment stage's builtin variables, the declaration block of the OpenGL Shading
# Language 4.60 specification 7.1.5 ("Fragment Shader Special Variables"). The refpages
# carry a variable's declarations and purpose but not the stage it belongs to, which is why
# this list is held here rather than read; every name is looked up in the corpus, and one
# that finds no page is reported.
_FRAGMENT_VARIABLES: tuple[str, ...] = (
    "gl_FragCoord",
    "gl_FrontFacing",
    "gl_ClipDistance",
    "gl_CullDistance",
    "gl_PointCoord",
    "gl_PrimitiveID",
    "gl_SampleID",
    "gl_SamplePosition",
    "gl_SampleMaskIn",
    "gl_Layer",
    "gl_ViewportIndex",
    "gl_HelperInvocation",
    "gl_FragDepth",
    "gl_SampleMask",
)


class NotARefpage(Exception):
    """The file is a shared include fragment, not a function page."""


def _load(path: Path) -> ET.Element | None:
    """Parse a refpage, resolving the DocBook math entities it declares.

    The pages reference `&sdot;` and friends through a SYSTEM entity file. Python's parser
    does not fetch it, so the entities are declared inline before parsing; without this,
    every page carrying a formula (mix and smoothstep among them) raises ParseError and
    silently vanishes from the table.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    if "<refentry" not in raw:
        # A shared include fragment (a table header, the version block), not a refpage. It
        # documents no function, so it is not a hole in the table.
        raise NotARefpage(path.name)
    # Strip the DOCTYPE that points at math.ent, then declare the entities we need as
    # their plain-text meaning; the prose is collapsed to text anyway.
    raw = re.sub(r"<!DOCTYPE[^>]*\[.*?\]>", "", raw, count=1, flags=re.S)
    raw = re.sub(r"<!DOCTYPE[^>]*>", "", raw, count=1)
    entities = "".join(
        f'<!ENTITY {name} "{value}">' for name, value in _ENTITIES.items()
    )
    doc = f"<!DOCTYPE refentry [{entities}]>{raw}"
    try:
        return ET.fromstring(doc)
    except ET.ParseError as exc:
        print(f"  skipped {path.name}: {exc}", file=sys.stderr)
        return None


def _text(node: ET.Element) -> str:
    """All text under `node`, whitespace collapsed."""
    return re.sub(r"\s+", " ", "".join(node.itertext())).strip()


def _prototypes(entry: ET.Element) -> list[str]:
    """Every overload in the page, as `<return> <name>(<type> <param>, ...)`."""
    out: list[str] = []
    for proto in entry.iter(f"{DOCBOOK}funcprototype"):
        funcdef = proto.find(f"{DOCBOOK}funcdef")
        if funcdef is None:
            continue
        name_node = funcdef.find(f"{DOCBOOK}function")
        if name_node is None or not name_node.text:
            continue
        name = name_node.text.strip()
        # funcdef's own text is the return type, with the function name as a child.
        return_type = (funcdef.text or "").strip()
        params: list[str] = []
        for pdef in proto.findall(f"{DOCBOOK}paramdef"):
            param = pdef.find(f"{DOCBOOK}parameter")
            ptype = (pdef.text or "").strip()
            pname = (param.text or "").strip() if param is not None else ""
            # A `void` parameter list carries no parameter element.
            params.append(f"{ptype} {pname}".strip() if pname else ptype)
        signature = f"{return_type} {name}({', '.join(params)})".strip()
        # The spec marks an optional trailing parameter with a bracket that opens in one
        # paramdef and closes outside the prototype; balance it so the rendered signature
        # reads as GLSL rather than as a truncated one.
        if signature.count("[") > signature.count("]"):
            signature = signature.replace("(", "(", 1)
            signature = re.sub(r"\[\s*", "[", signature)
            signature = signature[:-1] + "])" if signature.endswith(")") else signature
        signature = re.sub(r",\s*\[", " [, ", signature)
        if signature not in out:
            out.append(signature)
    return out


def _purpose(entry: ET.Element) -> str:
    node = entry.find(f".//{DOCBOOK}refpurpose")
    return _text(node) if node is not None else ""


def _declarations(entry: ET.Element) -> list[tuple[str, str]]:
    """Every `fieldsynopsis` in the page, as (variable name, `<modifier> <type> <name>`).

    A builtin variable is declared rather than called, so its page carries no prototype. A
    page may carry two declarations for one name (an `in` form and an `out` form, one per
    stage); both are kept, in page order.
    """
    out: list[tuple[str, str]] = []
    for field in entry.iter(f"{DOCBOOK}fieldsynopsis"):
        varname = field.find(f"{DOCBOOK}varname")
        type_node = field.find(f"{DOCBOOK}type")
        if varname is None or not varname.text or type_node is None:
            continue
        # `gl_ClipDistance[]` declares the name `gl_ClipDistance`.
        name = varname.text.strip().split("[")[0]
        modifier = field.find(f"{DOCBOOK}modifier")
        parts = [
            (modifier.text or "").strip() if modifier is not None else "",
            (type_node.text or "").strip(),
            varname.text.strip(),
        ]
        declaration = " ".join(part for part in parts if part)
        if (name, declaration) not in out:
            out.append((name, declaration))
    return out


def parse_refpages(
    root: Path,
) -> tuple[
    dict[str, tuple[list[str], str]], dict[str, tuple[list[str], str]], list[str]
]:
    """The function table, the variable table, and the pages that yielded neither.

    A function entry is named from its own prototype: a page whose refname is a family name
    (`packUnorm`, `noise`) documents several functions and matches none of them, so keying
    on the refname loses the page whole. A variable entry is named from its
    `fieldsynopsis`, filtered to `_FRAGMENT_VARIABLES`.

    The third value is every page that produced no entry at all -- a hole in the table, not
    a curiosity. A caller that ignores it ships a partial table.
    """
    functions: dict[str, tuple[list[str], str]] = {}
    variables: dict[str, tuple[list[str], str]] = {}
    empty: list[str] = []
    for path in sorted(root.glob("*.xml")):
        if _API_PREFIX.match(path.stem) or path.name in _NAMES_NOTHING:
            continue
        try:
            entry = _load(path)
        except NotARefpage:
            continue
        if entry is None:
            empty.append(path.name)
            continue
        purpose = _purpose(entry)
        yielded = False
        # One entry per function the page's prototypes name, carrying only the overloads
        # whose own name matches it: a page may document several (dFdx/dFdy share one).
        by_name: dict[str, list[str]] = {}
        for signature in _prototypes(entry):
            call = re.search(r"\b(\w+)\s*\(", signature)
            if call is None or _API_PREFIX.match(call.group(1)):
                continue
            by_name.setdefault(call.group(1), []).append(signature)
        for name, signatures in by_name.items():
            yielded = True
            if name not in _STAGE_ONLY:
                functions[name] = (signatures, purpose)
        for name, declaration in _declarations(entry):
            yielded = True
            if name in _FRAGMENT_VARIABLES:
                held, _ = variables.get(name, ([], purpose))
                if declaration not in held:
                    held.append(declaration)
                variables[name] = (held, purpose)
        if not yielded:
            empty.append(path.name)
    return functions, variables, empty


def parse_lexer_words(path: Path) -> tuple[list[str], list[str]]:
    """The GLSL keyword and type lists out of the editor library's lexer source."""
    text = path.read_text(encoding="utf-8")

    def block(name: str) -> list[str]:
        match = re.search(rf"{name} :: \[\]string \{{(.*?)\}}", text, re.S)
        if match is None:
            raise SystemExit(f"{path}: no {name} block")
        return sorted(set(re.findall(r'"([^"]+)"', match.group(1))))

    return block("GLSL_KEYWORDS"), block("GLSL_TYPES")


def _q(text: str) -> str:
    """A double-quoted Python string literal, which is what ruff-format emits."""
    body = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{body}"'


def _table(entries: dict[str, tuple[list[str], str]]) -> str:
    """The rows of a `name -> (forms, purpose)` table, in the shape ruff-format emits."""
    rows: list[str] = []
    for name in sorted(entries):
        forms, purpose = entries[name]
        # Double quotes and a trailing comma: the repo formats with ruff, and a generated
        # file it wants to rewrite produces a spurious diff on every run.
        collapsed = f"        ({_q(forms[0])},),"
        if len(forms) == 1 and len(collapsed) <= _LINE_LENGTH:
            # ruff-format collapses a one-element tuple onto its own line while it fits;
            # emit the shape it would, so the generated file is already formatted.
            form_block = collapsed
        else:
            form_lines = "\n".join(f"            {_q(form)}," for form in forms)
            form_block = f"        (\n{form_lines}\n        ),"
        rows.append(f"    {_q(name)}: (\n{form_block}\n        {_q(purpose)},\n    ),")
    return "\n".join(rows)


def render(
    builtins: dict[str, tuple[list[str], str]],
    variables: dict[str, tuple[list[str], str]],
    source: str,
    keywords: list[str],
    types: list[str],
    lexer_source: str,
) -> str:
    body = _table(builtins)
    variable_body = _table(variables)
    return f'''"""GLSL builtin functions and variables: every form and the spec's own one-line purpose.

GENERATED by scripts/gen_glsl_docs.py from the Khronos OpenGL-Refpages ({source}) --
do not hand-edit. The code panel reads this for `K` over a builtin and for the detail
note beside a completion candidate, so what a shader author is told about `mix` is what
Khronos publishes rather than anything typed from memory.

`genType` is the spec's notation for "float, vec2, vec3 or vec4, the same throughout";
`genIType`, `genUType` and `genBType` are its int, uint and bool counterparts.

`VARIABLES` holds the fragment stage's builtin variables. A name carries every declaration
its page states, because three of them are declared once per stage and picking one would
be arbitrary.
"""

# name -> (overload signatures, one-line purpose)
BUILTINS: dict[str, tuple[tuple[str, ...], str]] = {{
{body}
}}

# name -> (declarations, one-line purpose)
VARIABLES: dict[str, tuple[tuple[str, ...], str]] = {{
{variable_body}
}}

# The language's reserved words and type names, from the editor library's GLSL lexer
# ({lexer_source}) -- the same list that colors the text, so completion offers exactly what
# the highlighter knows.
KEYWORDS: tuple[str, ...] = (
{chr(10).join(f"    {_q(w)}," for w in keywords)}
)

TYPES: tuple[str, ...] = (
{chr(10).join(f"    {_q(w)}," for w in types)}
)
'''


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            f"usage: {sys.argv[0]} <refpages gl4 dir> <editor src/lex_glsl.odin>"
        )
    root = Path(sys.argv[1])
    lexer = Path(sys.argv[2])
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")
    if not lexer.is_file():
        raise SystemExit(f"not a file: {lexer}")
    builtins, variables, empty = parse_refpages(root)
    if not builtins:
        raise SystemExit(f"no builtin pages parsed from {root}")
    if empty:
        # A page that yields no entry is a HOLE in the table, not a curiosity: whether it
        # would not parse or its refname matched no prototype, the name it documents
        # silently has no doc. Fix the entity list or the naming rule and re-run.
        raise SystemExit(
            f"{len(empty)} page(s) yielded no entry: {', '.join(sorted(empty))}"
        )
    missing = [name for name in _FRAGMENT_VARIABLES if name not in variables]
    if missing:
        raise SystemExit(f"no page for: {', '.join(missing)}")
    keywords, types = parse_lexer_words(lexer)
    OUT_PATH.write_text(
        render(builtins, variables, root.name, keywords, types, lexer.name),
        encoding="utf-8",
    )
    print(
        f"{OUT_PATH.relative_to(REPO_ROOT)}: {len(builtins)} builtins, "
        f"{len(variables)} variables, {len(keywords)} keywords, {len(types)} types"
    )


if __name__ == "__main__":
    main()
