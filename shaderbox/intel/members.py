"""What a dot reaches: the members of a GLSL type.

Only vectors have members. A matrix is indexed (`m[0]`), never dotted, and a sampler is
opaque -- so a dot after either offers nothing rather than offering something wrong. GLSL has
no user structs in any shader this app ships, and none in its library; if one appears, its
members belong here beside the vectors.
"""

from shaderbox.intel.symbols import Symbol, SymbolKind

# The three component sets a vector's members come from: position, color, texture coordinate.
# GLSL forbids MIXING them in one swizzle -- `v.xg` does not compile -- so each set generates
# its own candidates and no candidate ever spans two.
_SWIZZLE_SETS: tuple[str, ...] = ("xyzw", "rgba", "stpq")

# How many components each vector type has. A swizzle may only name components the source
# HAS, so a vec2 draws from the first two letters of each set and `uv.z` is never offered.
_VECTOR_WIDTH: dict[str, int] = {
    "vec2": 2,
    "vec3": 3,
    "vec4": 4,
    "bvec2": 2,
    "bvec3": 3,
    "bvec4": 4,
    "ivec2": 2,
    "ivec3": 3,
    "ivec4": 4,
    "uvec2": 2,
    "uvec3": 3,
    "uvec4": 4,
}

# The component type each vector's swizzle of length 1 yields; longer swizzles yield the
# vector of that width, which `_swizzle_type` assembles.
_COMPONENT_TYPE: dict[str, str] = {
    "vec": "float",
    "bvec": "bool",
    "ivec": "int",
    "uvec": "uint",
}


def _family(glsl_type: str) -> str:
    return glsl_type[:-1]


def _swizzle_type(glsl_type: str, length: int) -> str:
    family = _family(glsl_type)
    return _COMPONENT_TYPE[family] if length == 1 else f"{family}{length}"


def members_of(glsl_type: str) -> tuple[Symbol, ...]:
    """Every swizzle worth offering for a type, or empty when the type has no members.

    Every single component, plus the leading runs (`xy`, `xyz`, `rgba`, ...) -- not the full
    cartesian product, which for a vec4 is 340 swizzles across the three sets. That product
    would bury every other candidate under the offer cap to spell out reaches nobody types;
    a user who wants `v.zyx` types it. The singles are NOT an identity prefix, though:
    `uv.y` and `col.g` are the common reach, so all of them come, in component order.
    """
    width = _VECTOR_WIDTH.get(glsl_type)
    if width is None:
        return ()
    found: list[Symbol] = []
    for components in _SWIZZLE_SETS:
        available = components[:width]
        swizzles = list(available) + [available[:n] for n in range(2, width + 1)]
        for swizzle in swizzles:
            found.append(
                Symbol(
                    swizzle,
                    SymbolKind.GLSL_MEMBER,
                    signature=f"{_swizzle_type(glsl_type, len(swizzle))} {glsl_type}.{swizzle}",
                    doc=f"component{'s' if len(swizzle) > 1 else ''} of a {glsl_type}",
                )
            )
    return tuple(found)
