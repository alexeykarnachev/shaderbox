"""Assemble tutorial.html from its parts.

The tutorial embeds its stage renders as data URIs so the file is one self-contained
thing you can open anywhere. Regenerate the images with the snippet in `oracle.py`'s
sibling notes, then run this. Kept as a script rather than a hand-edited HTML file
because the base64 blobs are thousands of characters and would make the document
uneditable.

Three markers in `tutorial_body.html` are spliced here, all read out of the shipped
Radiance Cascades example so a card or a shader listing cannot drift from what the
example actually is: `{{CARD:<pass>}}` (the pass's gear settings as a table),
`{{CODE:<pass>}}` (the complete shader), `{{IMG:<pass>}}` (the stage render). This
module never imports `shaderbox`: it runs from `ai_docs/`, outside the package tree,
and an import would make a documentation build depend on the app's import graph.
`tests/test_tutorial_build.py` pins the duplicated defaults below to the engine's own.
"""

import base64
import html
import json
import pathlib
from typing import Any

HERE = pathlib.Path(__file__).resolve().parent

EXAMPLE_ID = "77a84d27-2e5b-406d-8011-ee1cb1a9587c"
EXAMPLE_DIR = (
    pathlib.Path(__file__).resolve().parents[3]
    / "shaderbox"
    / "resources"
    / "document_examples"
    / EXAMPLE_ID
)

# Mirror `pass_graph.py`'s DEFAULT_* and `PassEntry.iterations`; the build test
# compares them.
_DEFAULT_SCALE: float = 1.0
_DEFAULT_DTYPE: str = "f2"
_DEFAULT_FILTER_LINEAR: bool = True
_DEFAULT_WRAP: bool = False
_DEFAULT_ITERATIONS: int = 1

# The `_FORMATS` labels in `popups/pass_settings.py` — the words the combo shows.
_DTYPE_LABELS: dict[str, str] = {
    "f1": "8-bit",
    "f2": "16-bit float",
    "f4": "32-bit float",
}


def _data_uri(name: str) -> str:
    raw = (HERE / "img" / f"{name}.png").read_bytes()
    return "data:image/png;base64," + base64.b64encode(raw).decode()


def _value_cell(rendered: str, is_default: bool) -> str:
    if is_default:
        return f'{rendered} <span class="dfl">default</span>'
    return rendered


def _reads_html(name: str, entry: dict[str, Any]) -> str:
    inputs: dict[str, str] = entry.get("inputs", {})
    if not inputs:
        return "nothing"
    lines: list[str] = []
    for uniform in sorted(inputs):
        source: str = inputs[uniform]
        line = f"<code>{html.escape(uniform)}</code> from <b>{html.escape(source)}</b>"
        if source == name:
            iterations: int = entry.get("iterations", _DEFAULT_ITERATIONS)
            when = "last run" if iterations > 1 else "last frame"
            line += f" (itself, {when})"
        lines.append(line)
    return "<br>\n      ".join(lines)


def _card_html(name: str, graph: dict[str, Any]) -> str:
    entry: dict[str, Any] = graph["passes"][name]
    target: dict[str, Any] = entry.get("target", {})
    scale: float = target.get("scale", _DEFAULT_SCALE)
    dtype: str = target.get("dtype", _DEFAULT_DTYPE)
    filter_linear: bool = target.get("filter_linear", _DEFAULT_FILTER_LINEAR)
    wrap: bool = target.get("wrap", _DEFAULT_WRAP)
    iterations: int = entry.get("iterations", _DEFAULT_ITERATIONS)
    rows: list[tuple[str, str]] = [
        ("name", f"<code>{html.escape(name)}</code>"),
        ("reads", _reads_html(name, entry)),
        ("format", _value_cell(_DTYPE_LABELS[dtype], dtype == _DEFAULT_DTYPE)),
        ("size", _value_cell(f"{scale * 100:.0f}%", scale == _DEFAULT_SCALE)),
        (
            "smooth",
            _value_cell(
                "on" if filter_linear else "off",
                filter_linear == _DEFAULT_FILTER_LINEAR,
            ),
        ),
        ("repeat", _value_cell("on" if wrap else "off", wrap == _DEFAULT_WRAP)),
        ("runs", _value_cell(str(iterations), iterations == _DEFAULT_ITERATIONS)),
    ]
    body = "\n".join(
        f"  <tr><td>{label}</td><td>{value}</td></tr>" for label, value in rows
    )
    return f'<table class="card">\n{body}\n</table>'


def _code_html(name: str) -> str:
    source = (EXAMPLE_DIR / "passes" / f"{name}.frag.glsl").read_text(encoding="utf-8")
    return f"<pre><code>{html.escape(source.rstrip())}</code></pre>"


def build(out: pathlib.Path | None = None) -> None:
    graph: dict[str, Any] = json.loads(
        (EXAMPLE_DIR / "graph.json").read_text(encoding="utf-8")
    )
    body = (HERE / "tutorial_body.html").read_text(encoding="utf-8")
    for name in graph["passes"]:
        body = body.replace(f"{{{{CARD:{name}}}}}", _card_html(name, graph))
        body = body.replace(f"{{{{CODE:{name}}}}}", _code_html(name))
    for name in graph["passes"]:
        body = body.replace(f"{{{{IMG:{name}}}}}", _data_uri(name))
    target = out if out is not None else HERE / "tutorial.html"
    target.write_text(body, encoding="utf-8")
    print(f"wrote {target}")


if __name__ == "__main__":
    build()
