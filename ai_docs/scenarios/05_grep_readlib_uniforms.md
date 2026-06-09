# Scenario: grep → read_lib two-step + set_uniform edge cases

**Probes:** two core navigation/value capabilities — the grep→read_lib code-discovery pattern, and
set_uniform type handling (scalar / vector / text-string / the engine-driven reject). No `todo.md`
deferral targeted; this is a "do the basics actually work" sweep.

**Setup:** `h = DogfoodHarness.create()` (templates seeded — the Text node uses the SDF glyph machinery
and engine uniforms; UV Mango is simple).

---

## Part A — grep → read_lib

Step A1 — locate a uniform across the project
- User: "Which shaders use u_time? Search the whole project."
- Expect: grep("u_time") → origin-labeled hits (node ids / lib: addresses). The agent should report where.
- Human check: does grep return hits with locations, and does the agent summarize them correctly (not
  hallucinate files that don't match)?

Step A2 — read a library function body (if the dogfood lib has any)
- Note: the isolated dogfood `SHADERBOX_DATA_DIR` starts with an EMPTY shader library (0 functions), so
  read_lib will likely miss. First make the agent create one:
  User: "Create a library function SB_circle(vec2 uv, float r) that returns a smooth circle mask, then
  use it in the current shader."
- Expect: insert_after on a `lib:...` address (creates the lib file) → edit the node to call SB_circle →
  compile. Then a follow-up read:
  User: "Show me the body of SB_circle."
- Expect: read_lib("SB_circle") → the full body.
- Human check: did the lib function get created (grep/read_lib find it), and does the node compile using
  it (the auto-resolve preamble splice working headless)?

## Part B — set_uniform edge cases

Step B1 — a scalar
- User: "Set u_time to 2.0." → Expect: REJECTED ("engine-driven — read it, never set it"). Human check:
  does it refuse u_time/u_aspect/u_resolution with a clear reason, not silently accept?

Step B2 — a real scalar/vector uniform
- First ensure one exists: User: "Add a uniform u_color (vec3) and a uniform u_scale (float) to the
  current shader and use them."
- User: "Set u_color to red." → Expect: set_uniform(u_color, [1,0,0]) (a 3-vector). Render → eyeball red.
- User: "Set u_scale to 0.5." → Expect: set_uniform(u_scale, 0.5) (a scalar). Render → eyeball the change.
- Human check: does the agent shape the value correctly (list for the vec3, number for the float), and
  does the render reflect it? (This also re-tests visual-blindness: does it claim the color changed, and
  did it?)

Step B3 — a text-string uniform (if you switch to the Text node)
- `h.send("Switch to the Text node.")` then User: "Set the text to 'HELLO'."
- Expect: set_uniform on the text uint-array uniform with a STRING "HELLO" (the harness/engine converts to
  codepoints). Render → eyeball "HELLO".
- Human check: does the string→codepoint path work, and does the glyph SDF render the word?

**What to record:** any tool that mis-shapes a value, accepts an engine uniform, or fails the grep→read_lib
round-trip. These are the "table stakes" — if they're shaky, fix before chasing the fancier weak spots.
