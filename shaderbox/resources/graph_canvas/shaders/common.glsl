// Shared by every shader here; prepended at load time.
//
// Attribute locations are PINNED rather than queried. An attribute a program
// does not read is optimized out and reports location -1, so a queried binding
// would leave a VAO silently unbound while the program still drew -- with
// whatever was last in that slot.
//
// They are also declared HERE rather than as literals at each use, because the
// Odin side pins the same numbers and nothing but a shared name keeps the two
// in agreement. tools/check.sh compares these defines against the ATTR_
// constants in ui/, and a collision or a drift fails the gate: two attributes
// on one slot renders every shape with another's parameters, which no test and
// no build catches.
//
// Slots 0..3 are raylib's fixed rlgl set (0=position, 1=texcoord, 2=normal,
// 3=color); 4 and 5 are its tangent and texcoord2. This project has neither
// normals nor a second UV set, so those slots carry our own parameters. 6 and
// 7 are past raylib's set and are ours alone.
#define ATTR_POSITION   0
#define ATTR_TEXCOORD   1
#define ATTR_COLOR      3
#define ATTR_USER0      4
#define ATTR_USER1      5
#define ATTR_USER2      2
#define ATTR_UV_BOUNDS  2
#define ATTR_ROTATION   6

// The SDF shape program's own names for the same slots.
#define ATTR_CORNER     0
#define ATTR_RECT       1
#define ATTR_FILL_TOP   3
#define ATTR_SHAPE      4
#define ATTR_FILL_BOT   5
#define ATTR_EDGE       2
#define ATTR_FIELD      7
#define ATTR_UV         8
#define ATTR_BORDER     9

float median3(vec3 c) { return max(min(c.r, c.g), min(max(c.r, c.g), c.b)); }
