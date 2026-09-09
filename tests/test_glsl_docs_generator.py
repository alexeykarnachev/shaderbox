"""The refpage reader reports a page that names nothing (089 W-B review).

`_NAMES_NOTHING` excuses three pages from the empty-page report. Skipping them BEFORE the
parse made the excuse unconditional: a page listed there could never be reported however it
changed, and its entries could never arrive either. The membership test belongs where the
report is decided, which is what these pin -- along with the line between a hole (a page
that names nothing) and a page whose names a filter deliberately drops.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))

import gen_glsl_docs  # noqa: E402

_A_REAL_PAGE = """<refentry xmlns="http://docbook.org/ns/docbook" version="5.0">
    <refnamediv>
        <refname>abs</refname>
        <refpurpose>return the absolute value of the parameter</refpurpose>
    </refnamediv>
    <refsynopsisdiv>
        <funcsynopsis>
            <funcprototype>
                <funcdef>genType <function>abs</function></funcdef>
                <paramdef>genType <parameter>x</parameter></paramdef>
            </funcprototype>
        </funcsynopsis>
    </refsynopsisdiv>
</refentry>
"""
# A page whose only name is a stage-only verb: it parses, and the filter that drops it is
# the record of why, so it is not a hole.
_A_STAGE_ONLY_PAGE = _A_REAL_PAGE.replace("abs", "EmitVertex")
# A page that names nothing at all -- the hole the report exists for.
_A_PAGE_THAT_NAMES_NOTHING = """<refentry xmlns="http://docbook.org/ns/docbook" version="5.0">
    <refnamediv>
        <refname>packUnorm</refname>
        <refpurpose>pack floating-point values into an unsigned integer</refpurpose>
    </refnamediv>
</refentry>
"""


def _corpus(tmp_path: Path, pages: dict[str, str]) -> Path:
    root = tmp_path / "gl4"
    root.mkdir()
    for name, text in pages.items():
        (root / name).write_text(text, encoding="utf-8")
    return root


def test_a_page_that_names_nothing_still_contributes_what_it_yields(
    tmp_path: Path,
) -> None:
    excused = sorted(gen_glsl_docs._NAMES_NOTHING)[0]
    root = _corpus(tmp_path, {excused: _A_REAL_PAGE})
    functions, _, empty = gen_glsl_docs.parse_refpages(root)
    assert "abs" in functions
    assert empty == []


def test_a_page_that_names_nothing_is_the_hole_the_report_names(tmp_path: Path) -> None:
    root = _corpus(
        tmp_path,
        {"abs.xml": _A_REAL_PAGE, "packUnorm.xml": _A_PAGE_THAT_NAMES_NOTHING},
    )
    functions, _, empty = gen_glsl_docs.parse_refpages(root)
    assert "abs" in functions
    assert empty == ["packUnorm.xml"]


def test_a_page_a_named_filter_drops_is_not_a_hole(tmp_path: Path) -> None:
    root = _corpus(
        tmp_path,
        {"abs.xml": _A_REAL_PAGE, "EmitVertex.xml": _A_STAGE_ONLY_PAGE},
    )
    functions, _, empty = gen_glsl_docs.parse_refpages(root)
    assert "EmitVertex" not in functions
    assert empty == []


def test_an_excused_page_that_names_nothing_stays_out_of_the_report(
    tmp_path: Path,
) -> None:
    excused = sorted(gen_glsl_docs._NAMES_NOTHING)[0]
    root = _corpus(
        tmp_path,
        {"abs.xml": _A_REAL_PAGE, excused: _A_PAGE_THAT_NAMES_NOTHING},
    )
    _, _, empty = gen_glsl_docs.parse_refpages(root)
    assert empty == []
