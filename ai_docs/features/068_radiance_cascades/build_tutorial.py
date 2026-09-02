"""Assemble tutorial.html from its parts.

The tutorial embeds its stage renders as data URIs so the file is one self-contained thing you
can open anywhere. Regenerate the images with the snippet in `oracle.py`'s sibling notes, then
run this. Kept as a script rather than a hand-edited HTML file because the base64 blobs are
thousands of characters and would make the document uneditable.
"""

import base64
import pathlib

HERE = pathlib.Path(__file__).resolve().parent


def _data_uri(name: str) -> str:
    raw = (HERE / "img" / f"{name}.png").read_bytes()
    return "data:image/png;base64," + base64.b64encode(raw).decode()


def build() -> None:
    body = (HERE / "tutorial_body.html").read_text(encoding="utf-8")
    for name in ("paint", "seed", "jfa", "df", "cascade", "composite"):
        body = body.replace(f"{{{{IMG:{name}}}}}", _data_uri(name))
    (HERE / "tutorial.html").write_text(body, encoding="utf-8")
    print(f"wrote {HERE / 'tutorial.html'}")


if __name__ == "__main__":
    build()
