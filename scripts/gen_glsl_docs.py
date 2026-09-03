"""Generate shaderbox/glsl_docs.py from the Khronos OpenGL-Refpages XML.

The GLSL builtins a shader author calls -- their FULL overload sets and the spec's own
one-line purpose -- are data Khronos publishes, not something to type from memory. This
reads the ES 3.0 refpages (one DocBook XML per function, a `funcprototype` per overload)
and emits a table the code panel reads for `K` and for the completion popup's detail note.

The keyword and type vocabulary comes from the editor library's own GLSL lexer
(`src/lex_glsl.odin` in the editor repo), which is the list that actually colors this
editor's text -- so completion offers exactly the words the highlighter knows. Neither half
is typed from memory.

Usage:
    git clone --depth 1 --filter=blob:none --sparse \\
        https://github.com/KhronosGroup/OpenGL-Refpages.git /tmp/refpages
    cd /tmp/refpages && git sparse-checkout set es3.0
    uv run python scripts/gen_glsl_docs.py /tmp/refpages/es3.0 ~/src/editor/src/lex_glsl.odin

Output (repo-anchored): shaderbox/glsl_docs.py
"""

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "shaderbox" / "glsl_docs.py"
DOCBOOK = "{http://docbook.org/ns/docbook}"

# Pages in the refpages tree that document the API, not the shading language. The GLSL
# pages are the ones whose refentry carries a funcsynopsis with a funcprototype; a gl*
# entry point is documented in the same directory and must not reach a shader author's
# completion list.
_API_PREFIX = re.compile(r"^gl[A-Z]")

# The math entities the refpages use, as plain text. Any entity NOT listed here surfaces as
# a reported skip rather than a silent drop, so the table can never quietly shrink.
_ENTITIES: dict[str, str] = {
    # Enumerated from the corpus itself:
    #   grep -ohE '&[a-zA-Z][a-zA-Z0-9]*;' es3.0/*.xml | sort -u
    # A page using an entity absent here fails to parse and is REPORTED, never dropped in
    # silence -- the generator exits non-zero if any page is skipped.
    "af": "",
    "amp": "&#38;",
    "delta": "delta",
    "Delta": "Delta",
    "ge": "&#8805;",
    "gt": "&#62;",
    "it": "",
    "lambda": "lambda",
    "lceil": "ceil(",
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
    "VerticalLine": "|",
}


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


def parse_refpages(root: Path) -> tuple[dict[str, tuple[list[str], str]], list[str]]:
    """name -> (overload signatures, purpose) for every GLSL builtin page, plus the pages
    that failed to parse. A caller that ignores the second half ships a partial table."""
    found: dict[str, tuple[list[str], str]] = {}
    unparsed: list[str] = []
    for path in sorted(root.glob("*.xml")):
        if _API_PREFIX.match(path.stem):
            continue
        try:
            entry = _load(path)
        except NotARefpage:
            continue
        if entry is None:
            unparsed.append(path.name)
            continue
        signatures = _prototypes(entry)
        if not signatures:
            continue
        # The page's own name, not the filename: `abs.xml` documents `abs`, but a few
        # pages carry a different refname.
        # A page may document several functions (dFdx/dFdy share one). Each gets its own
        # entry, carrying only the overloads whose own name matches it.
        purpose = _purpose(entry)
        # A refname may list several functions in one string ("dFdx, dFdy"), and a page may
        # carry several refname elements. Both forms split into one entry per function.
        names = [
            part.strip()
            for n in entry.findall(f".//{DOCBOOK}refname")
            for part in (n.text or "").split(",")
            if part.strip()
        ] or [path.stem]
        for name in names:
            if _API_PREFIX.match(name):
                continue
            own = [s for s in signatures if re.search(rf"\b{re.escape(name)}\s*\(", s)]
            if own:
                found[name] = (own, purpose)
    return found, unparsed


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


def render(
    builtins: dict[str, tuple[list[str], str]],
    source: str,
    keywords: list[str],
    types: list[str],
    lexer_source: str,
) -> str:
    rows: list[str] = []
    for name in sorted(builtins):
        signatures, purpose = builtins[name]
        # Double quotes and a trailing comma: the repo formats with ruff, and a generated
        # file it wants to rewrite produces a spurious diff on every run.
        if len(signatures) == 1:
            # ruff-format collapses a one-element tuple onto its own line; emit that shape
            # so the generated file is already formatted.
            sig_block = f"        ({_q(signatures[0])},),"
        else:
            sig_lines = "\n".join(f"            {_q(sig)}," for sig in signatures)
            sig_block = f"        (\n{sig_lines}\n        ),"
        rows.append(f"    {_q(name)}: (\n{sig_block}\n        {_q(purpose)},\n    ),")
    body = "\n".join(rows)
    return f'''"""GLSL builtin functions: every overload and the spec's own one-line purpose.

GENERATED by scripts/gen_glsl_docs.py from the Khronos OpenGL-Refpages ({source}) --
do not hand-edit. The code panel reads this for `K` over a builtin and for the detail
note beside a completion candidate, so what a shader author is told about `mix` is what
Khronos publishes rather than anything typed from memory.

`genType` is the spec's notation for "float, vec2, vec3 or vec4, the same throughout";
`genIType`, `genUType` and `genBType` are its int, uint and bool counterparts.
"""

# name -> (overload signatures, one-line purpose)
BUILTINS: dict[str, tuple[tuple[str, ...], str]] = {{
{body}
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
            f"usage: {sys.argv[0]} <refpages es3.0 dir> <editor src/lex_glsl.odin>"
        )
    root = Path(sys.argv[1])
    lexer = Path(sys.argv[2])
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")
    if not lexer.is_file():
        raise SystemExit(f"not a file: {lexer}")
    builtins, unparsed = parse_refpages(root)
    if not builtins:
        raise SystemExit(f"no builtin pages parsed from {root}")
    if unparsed:
        # A page that will not parse is a HOLE in the table, not a curiosity: the builtin it
        # documents would silently have no doc. Fix the entity list and re-run.
        raise SystemExit(
            f"{len(unparsed)} page(s) failed to parse: {', '.join(sorted(unparsed))}"
        )
    keywords, types = parse_lexer_words(lexer)
    OUT_PATH.write_text(
        render(builtins, root.name, keywords, types, lexer.name), encoding="utf-8"
    )
    print(
        f"{OUT_PATH.relative_to(REPO_ROOT)}: {len(builtins)} builtins, "
        f"{len(keywords)} keywords, {len(types)} types"
    )


if __name__ == "__main__":
    main()
