"""The `flat in` parser: what an instanced pass declares as its per-entity fields.

Read from SOURCE, never from the linked program -- the driver dead-strips an attribute the
body does not read yet, and an introspection-derived record is then the wrong width with no
error (measured: energy 0.3 rendered where 0.8999 belonged).
"""

from shaderbox.intel.glsl import entity_fields


def test_both_qualifier_orders_are_found() -> None:
    # GLSL accepts `flat in` and `in flat`. A line-anchored `flat\s+in` finds ZERO fields
    # in a shader written the second way -- a hard break on legal code, so both are pinned.
    assert [f.name for f in entity_fields("flat in vec2 pos;")] == ["pos"]
    assert [f.name for f in entity_fields("in flat vec2 pos;")] == ["pos"]


def test_one_declaration_may_carry_several_names() -> None:
    # `flat in float radius, energy;` is TWO fields. A pattern taking only the first
    # silently drops the rest, and the dropped field then has no attribute to bind.
    fields = entity_fields("flat in float radius, energy, mass;")
    assert [f.name for f in fields] == ["radius", "energy", "mass"]
    assert {f.glsl_type for f in fields} == {"float"}


def test_qualifiers_in_front_do_not_hide_the_field() -> None:
    assert [f.name for f in entity_fields("layout(location=3) flat in vec3 col;")] == ["col"]
    assert [f.name for f in entity_fields("flat in highp float t;")] == ["t"]


def test_a_commented_declaration_is_not_a_field() -> None:
    # The fixture is a BLOCK comment whose inner line starts at column 0, because that is
    # the only shape that reaches the comment stripper: a `//` or same-line `/* */` fails
    # the pattern's own start-of-line anchor, so a fixture built from one passes with the
    # stripping deleted and tests nothing. `pos` is here to prove the parser still ran.
    source = "/*\nflat in vec2 ghost;\n*/\nflat in vec2 pos;"
    assert [f.name for f in entity_fields(source)] == ["pos"]


def test_an_interpolated_varying_is_not_an_entity_field() -> None:
    # `vs_uv` is the per-PIXEL coordinate every pass already has. Only `flat` -- a value
    # with nothing to interpolate -- marks a per-ENTITY field, which is what keeps the two
    # kinds of `in` distinguishable in one file.
    assert entity_fields("in vec2 vs_uv;") == ()


def test_the_line_is_the_declaration_s_own() -> None:
    fields = entity_fields("#version 460 core\nin vec2 vs_uv;\nflat in float energy;\n")
    assert [(f.name, f.line) for f in fields] == [("energy", 2)]
