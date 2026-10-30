"""What the editor understands about the document it edits.

One index of symbols per (document, tab), built from the buffer text, the engine's tables, the
script's returns and the pass graph, read three ways: completion candidates, `K` lookup, and
the identifier classes that color the text. GL-free; imports nothing from `App`.
"""
