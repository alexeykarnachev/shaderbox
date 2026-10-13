# 110 — A script drives a sampler2D

A document script can return an array for a `sampler2D` uniform, and the pass samples it
as a texture that frame. This closes the gap 109's spec deferred under "A script returning a
texture" (trigger: "the first document that needs a CPU-computed field drawn as an image"),
which the multi-fluid showcase for 109 is. It is useful without a native library at all: a
numpy field from plain Python takes the same path.

Research measured on the dev box (RTX 3090, Linux, standalone GL context, median of 200
frames; throwaway scripts kept out of the repo). The field was filled by a 109 library
through the real `native()` path, then uploaded with `Texture.write` straight from the
library's buffer:

| field (RGBA f4) | bytes | native call, empty export | upload, cost seen by a draw |
|---|---|---|---|
| 256×256 | 1 MB | 0.043 ms | ~0.18 ms |
| 512×512 | 4 MB | 0.043 ms | ~0.73 ms |
| 1024×1024 | 16 MB | 0.043 ms | ~2.7 ms |

"Cost seen by a draw" is (write + a draw sampling it + `finish`) minus (the draw alone + `finish`).
The call does not grow with the buffer: a 109 buffer is an `mmap` the parent and the host share,
so nothing is copied between processes, and the upload is the one copy on the way to the GPU
(about 6 GB/s). Writing through a pixel buffer object first was no faster (slower at 16 MB);
converting to half floats on the CPU first was 3-4x slower, dominated by `astype`. A
`texture.read()` after the write was bit-identical to the library's buffer.

## Goal

`update` returns `{"fluid": {"u_density": field}}` (or the bare `{"u_density": field}`),
where `field` is a numpy array -- a 109 buffer is one -- and the pass `fluid` samples
`u_density` as a texture holding exactly those values this frame. A wrong shape or dtype is
an ordinary script warning naming the problem and its fix; the app keeps running. Nothing
about the sampler's saved source changes: when the script stops sending, is stopped or is
removed, the sampler reads what the user chose for it.

## Out of scope

- **Samplers other than a single `sampler2D`** (`sampler3D`, `samplerCube`, integer
  samplers, sampler arrays). They become refused keys (D1), not drivable ones.
  **Trigger:** a document that needs one.
- **Integer textures.** `u1` is accepted as NORMALIZED 8-bit (D2); `i4`/`u4` data would need
  an integer sampler, the item above. **Trigger:** as above.
- **Mipmaps and per-texture filter/wrap choice.** One fixed filter and wrap (D3).
  **Trigger:** a document whose field is drawn minified enough to alias.
- **Reading a pass's canvas BACK into a script** (GPU to CPU). **Trigger:** a document whose
  script needs to read what a pass drew.
- **Asynchronous or PBO upload.** Measured no faster above. **Trigger:** an upload measured as
  the frame's bottleneck in the running app.
- **Converting `f8` (or any other dtype) for the author.** D2 refuses it with the fix named.
  **Trigger:** a maintainer request.
- **The copilot's corroborating render showing the probe's field.**
  `backend.py::_script_render_line` merges the probe's value samples into the live document;
  a field never enters `samples`, so the render shows whatever the live slot holds, which is
  what it already does for a population. **Trigger:** a copilot trace where that render
  misdescribes a script-fed sampler.
- **Hover in the editor naming a script-fed sampler.** `intel/index.py` labels every sampler
  `PASS_SAMPLER` from source alone; the intel package is GL-free and has no view of a frame's
  slots. **Trigger:** a report that the hover misleads.

## Design decisions

- **D1. The uniform's TYPE decides the path, per pass, and the key grammar does not change.**
  A script addresses a sampler exactly as it addresses a scalar: a bare key broadcasts to
  every pass declaring that name, a pass block targets one pass (069 D3), a `None` value
  leaves it manual. Targeting is decided PER PASS: for each pass declaring the name, a value
  uniform takes the coercion path and a drivable sampler takes the texture path, each
  succeeding or failing on its own. Today a bare key a value uniform somewhere binds never
  examines the passes where the same name is a sampler (they are looked at only when no pass
  binds), so both loops of `_tick_script` -- broadcasts and blocks -- are restructured to ask
  each pass. A drivable sampler is `is_script_texture(uniform)`: `gl_type == GL_SAMPLER_2D`
  and `array_length == 1`. `is_scriptable` keeps meaning "a VALUE uniform the coercion
  drives" and becomes an ALLOWLIST: `gl_type in VALUE_GL_TYPES`, a set in
  `uniform_coerce.py` built from the float, double, int, uint and bool scalar, vector and
  matrix types the coercion shapes today (every one of them coerces and writes, measured
  with a `#version 460` shader declaring each) (it already holds the int/uint/bool subsets). Today it is a denylist of
  `sampler2D` alone, and `sampler3D`, `samplerCube` and `isampler2D` pass it (measured), so a
  float aimed at one is coerced into a texture unit. An allowlist refuses a type nobody
  listed, so the sampler and image types need no enumeration of their own to stay out. Those,
  sampler arrays and uniform blocks are `not_scriptable`.
  No `@texture` engine key: the `@` namespace is for engine vocabulary with no uniform to
  name (`@instances`), and a sampler has a name.

- **D2. What a script may return for a sampler, decided in one GL-free function.** A
  `numpy.ndarray`, C-contiguous, of shape `(H, W)` (one channel) or `(H, W, C)` with
  `C` in 1..4, `H` and `W` each at least 1 and at most `MAX_SCRIPT_TEXTURE_EXTENT`, dtype
  equal to native-endian `float32` (a float texture, moderngl `"f4"`, values as given) or
  `uint8` (moderngl `"f1"`, normalized: 255 reads 1.0 -- never moderngl `"u1"`, which is an
  integer texture a `sampler2D` cannot read). Anything else is refused with reason
  `bad_texture` (D6), and the message names what arrived and the fix (`.astype(np.float32)`,
  `np.ascontiguousarray(...)`). Refusing rather than converting, because a conversion is a
  per-frame cost the author cannot see: the `f2` measurement above is what one costs.
  Non-finite values pass through: scanning a field for them costs a full read per frame, and
  a shader may use them deliberately; the value path's NaN refusal is about one number in a
  panel, not a field. Row 0 is the BOTTOM row, `field[y, x]` with `y` up, which is the
  texture's own `v = 0` and matches `vs_uv`, so no flip is made. `MAX_SCRIPT_TEXTURE_EXTENT`
  is 16384, the `GL_MAX_TEXTURE_SIZE` minimum OpenGL 4.x requires (ShaderBox's shaders are
  `#version 460`); it is a constant because the engine is GL-free (025). The function lives
  in a new GL-free module `shaderbox/script_texture.py` (`texture_problem(value) -> str |
  None`, the dtype map, the extent, `REFUSED_TEXTURE`), because the engine (refuse), the draw
  (allocate) and completion (D9) must agree; it imports numpy only, which the intel package's
  GL-free import gate checks.

- **D3. The engine hands over DATA; the draw owns the GPU texture.** The same split as
  `@instances` (`ScriptPass.pending_instances`): the engine writes
  `render_pass.pending_textures: dict[str, np.ndarray]`, holding the script's own array
  object (no copy) or `REFUSED_TEXTURE`, a module-level zero-size array compared by `is`
  like `REFUSED_POPULATION`. The slot is rebuilt on every live tick that RAN the script, and
  holds unchanged when the behavior failed (compile error, raise, non-dict return): that is
  the freeze-as-data rule (048) applied to the slot. It is never written by `dry_run`.
  Texture pairs are tracked in their own set, `last_texture_driven`, beside `last_driven`
  (threaded through `tick`, `tick_export` and `dry_run` the way `last_driven` is, fresh sets
  for the last two). `last_driven` keeps meaning "pairs `_freeze` may restore", so `_freeze`
  never sees a texture pair and needs no type lookup on the failure path, where
  `active_by_pass` has not been built; `script_driven_uniforms` returns the union.
  Everywhere the engine resets or stashes `last_driven`, it does the same for the texture
  set: `_drop_script` clears it (or a removed script keeps reporting its samplers as driven,
  and the copilot's `set_uniform` keeps refusing them), and `_dry_run_with` stashes it beside
  `last_driven`, which is the driven set's only source on the headless copilot path.
  `ScriptStatus.driven_count` stays the count of VALUE pairs; its readers are tests.
  `Pass.render` keeps `script_textures: dict[str, moderngl.Texture]` by sampler name,
  reallocates one only when `(W, H)`, `C` or dtype change (a size check alone accepts an f4
  one-channel field followed by a u1 four-channel one of equal bytes), writes the array with
  `Texture.write`, and binds it. Filter linear, `repeat_x = repeat_y = False` (moderngl's
  clamp to edge): a field has no meaning past its edge, and repeat would bleed the opposite
  edge into the border texels. A failed allocation logs once per sampler and binds black.
  The textures are released in `Pass.release`, not on recompile: they belong to the data,
  not the program. The plain write is the measured fastest of the three tried.

- **D4. A texture the script sends this frame outranks the sampler's source at draw time,
  and never touches it.** For one draw, `Pass.render` binds, in order: the script's texture,
  then `inputs` (the pass the wiring names), then `uniform_values` (media, a bound texture,
  or a source that reads black). The texture path writes neither `uniform_values` nor
  `last_good`, and `_freeze` never sees texture pairs (D3's separate set), so no path can put an array in the
  sampler's source: the saved source (072) is unchanged on disk and in the panel, and binds
  again when the script stops sending the key, the user STOPS it (048: a stopped key still
  counts as driven, but its texture is not put in the slot), or the script is removed. The
  pair joins the texture-driven set (D3), so `script_driven_uniforms` reports it like any
  driven uniform, which is what gives the panel row its play/stop button; it never enters
  `last_driven`. **Removal** is a live path today for nothing: `ScriptEngine.tick` returns
  at "no behavior" and leaves every slot as it was (measured: `pending_instances` survives
  the script file's deletion). It now clears both `pending_textures` and `pending_instances`
  on every pass of a document with no behavior; clearing an already empty dict is the only
  per-frame cost.

- **D5. An absent key reads the source; a refused one holds the last good frame.** Absent
  this tick: the slot has no entry and the source binds, the same as an absent `@instances`
  (draw fullscreen) and for the same reason -- the return value IS the frame's state. A
  script that wants a still picture returns the same array again, which costs one upload.
  Refused this tick (D2): the draw binds the pass's existing `script_textures` entry -- the
  last frame that validated -- or black if there is none, never the source. An entry lives
  only while the script keeps feeding that sampler: the pass's first draw after a tick whose
  slot lacks it releases it (in the app every tick is followed by that document's draw),
  so a field that stopped (dropped, stopped, or its script removed) cannot reappear as a
  later refusal's held frame. Falling back to
  the source would show a different picture because of a data mistake, which is the case
  `REFUSED_POPULATION` exists to prevent.

- **D6. One new `KeyFailReason`, `bad_texture`, which warns; a refused texture is driven AND
  skipped.** The warning goes through `_warn`, edge-triggered on the reason (102 D1), and the
  `KEY_FAIL_REASONS` gate forces its tier. It is not a third silent case, so 102 D1's revisit
  clause is not touched. `_warn` puts the pair in `skipped`, and that is kept: `skipped` feeds
  `ScriptStatus.soft_errors`, the only thing the script strip shows
  (`tabs/code.py::_script_errors_for`), so leaving it out would hide the warning. The pair is
  ALSO added to the texture-driven set (D3): the key is the script's, merely wrong this frame,
  and the play/stop button must not vanish while the held frame shows.

- **D7. The wiring stays the user's; the override is reported where a sampler is shown.**
  `Document.effective_wiring` is unchanged: it is the wiring the user made, and its
  consumers -- the planner, cycle refusal on a drop, layout, import planning, rename, entry
  points, the strip's order, the copilot's working set -- keep answering about that. A
  script's texture is a per-frame override decided in `Pass.render` (D4), so making the
  wiring depend on it would make a drop's cycle check, the layout and the import plan flip
  with script state. The producer a script-fed sampler is wired to keeps rendering; that is
  wasted GPU on an edge the author wired and then overrode, not a wrong picture. Every
  surface that shows ONE sampler's state reads the slot as well:
  - the graph: `node_ports` gains a `script_fed: Collection[str]` argument and a fifth
    `PortKind`, `"script"`, which wins over the other four while the slot has the name, and
    is decided before the self-read check, so a fed `u_prev` gets a port rather than none;
    `widgets/graph_state.py::ports_of` passes it from the pass's `pending_textures`; the pin
    is a hollow square (the shape a bound texture has, the fill an unfilled dot has). A
    `"script"` port carries `source=` the pass its wiring names, if any -- None when that is
    the pass itself, since a self-read is feedback and no builder draws an edge for it -- and a wire into it
    is still drawn, because it is still the user's: every place that decides "this port has
    an edge" tests `port.source is not None` rather than `kind == "wired"` -- the flat view's
    edge loop in `graph_canvas/adapter.py` and `pass_graph.py::group_boundary` test the kind
    today, the scoped builders in `widgets/graph_state.py` already test the source. So a wired
    script port inside a group stays an inner edge, and an unwired one is a box input the way
    a `"media"` port is.
  - the strip: `widgets/pass_list.py::_reads` builds its chips from the wiring row minus
    the script-fed samplers (it gains that set as a parameter), so a producer still read by
    another, unfed sampler of the same pass keeps its chip.
  - the shader editor's uniform note (`tabs/code.py::_note_uniform`) shows the script's
    texture instead of `document.input_texture(...)` for a script-fed sampler.
  - the uniforms panel row keeps its source combo (the source is still the user's, D4); while
    the script feeds it, the preview shows the script's texture and the caption reads
    `script · WxH · C x dtype`. Picking a source while the sampler PLAYS auto-stops the pair
    and then writes the pick, the 048 rule for a manual edit of a driven uniform. Two sites:
    a `none` or pass pick calls `set_sampler_source` directly and today stops nothing, so the
    stop is added at that call site; a `file...` pick goes through `new_value`, which the
    `not playing` gate drops today, leaking the opened media, so that gate goes for the
    texture branch. The generic auto-stop on `is_item_activated()` cannot cover either: the
    branch's last item is a caption, never activated.
  - the copilot: `copilot/backend.py::_sampler_reads` reports a script-fed sampler as
    `<- (script field WxH, C x dtype)` (or a REFUSED note) rather than the wired pass or `(nothing; reads BLACK)`, and
    `copilot/prompt.py`'s sampler-row vocabulary names that form.
  Two docstrings that claim the wiring IS what binds become narrower: `Document.sampler_source`
  ("the same resolution the renderer binds with") and `pass_list.py::_reads`; so does 069 D9's
  sentence in conventions, which now reads "the renderer binds the wiring unless a script
  texture overrides the sampler this frame".

- **D8. The copilot reads statistics, never the field, and texture pairs leave the value
  verdict.** `values_sink` carries no ndarray (103's ban). A `texture_stats_sink`, the twin of
  `population_stats_sink`, records per sample time and per `(pass, sampler)`:
  `(shape, dtype, min, max, mean, digest, verdict)`. `digest` is a CRC32 of a strided
  subsample of at most 64×64 texels, because min and max cannot see a field whose range is
  fixed while its content moves -- an advecting fluid in [0, 1] is exactly that. `verdict` is
  `""` when it validated or the refusal reason. `ScriptProbe` gains `texture_samples` and
  `ScriptWriteResult` (`copilot/capabilities.py`) a `texture_facts` field beside
  `population_facts`; `copilot/backend.py` formats it beside `_population_facts_lines`, with
  motion decided as 103's `_population_motion` decides it (the first sample against every
  later one), over the digest. Texture pairs are excluded from `samples` and from
  `_motion_verdict`, which otherwise reads a driven pair with no value as constant and calls
  a moving field STATIC (reproduced by the review). Excluding them alone lands a
  texture-only script in the verdict's `not probe.driven` branch, "Nothing animates"
  (reproduced too), so that branch gains the case populations already have: "drives 0
  SCALAR uniforms, but see the texture facts below".

- **D9. Completion offers `sampler2D` for an array the engine would accept.**
  `intel/script.py::glsl_type_of_value` returns `"sampler2D"` when `texture_problem(value)`
  is None, so `uniform sampler2D u_density;` is offered for a script that returns a field
  under a name no pass declares yet. The same review found the 1-D case already broken: a
  length-3 `f4` ndarray infers `vec3`, which `coerce_uniform_value` refuses (it accepts
  `list | tuple` only). Fixed in this wave by having the coercion accept a 1-D numeric
  ndarray, so a numpy user's vector works and the two rules agree again. Length 1 disagrees
  for lists too (`[1.0]` infers `float[1]`, a declaration whose `array_length` 1 sends the
  coercion down the scalar path, which refuses a sequence); `glsl_type_of_value` returns
  None for a one-element sequence, since no declaration it could offer would land, and
  `intel/script.py::_literal_type` -- whose rules it mirrors -- does the same for a
  one-element literal (`tests/test_intel_sources.py::test_an_empty_literal_offers_no_declaration`
  pins `float[1]` today and changes with it).

- **D10. A 109 buffer is just an ndarray, and its lifetime is already safe.** The parent
  maps a buffer once and the mapping lives as long as the array
  (`native_library.py::NativeLibrary.buffer`); a release unlinks the file and asks the host
  to unmap, but never closes the parent's mapping under a live array. So the slot may hold
  the view until the draw, across a reload. A script thread that writes the buffer during
  the upload tears that frame; the engine takes no lock, as with any shared state a script
  keeps.

- **D11. Export and the probe paths follow the existing split.** `tick_export` writes the
  live slot (as it writes `pending_instances`), so exports render script textures, and the
  export's last frame stays in the slot until the next live tick rebuilds it, as populations
  already do. `dry_run` and `returned_value_types` never write it (063).

## Files touched

- `shaderbox/script_texture.py` (new): `texture_problem`, the dtype map,
  `MAX_SCRIPT_TEXTURE_EXTENT`, `REFUSED_TEXTURE`.
- `shaderbox/scripting/keys.py`: `bad_texture` in `KeyFailReason`, its doc.
- `shaderbox/scripting/engine.py`: `is_scriptable` (the allowlist), `last_texture_driven`,
  `is_script_texture`, per-pass targeting in both loops of `_tick_script`, `script_driven_uniforms`
  returning the union, `pending_textures` on `ScriptPass`, the clear in `tick` when there is no
  behavior, `texture_stats_sink`, `ScriptProbe.texture_samples` and its docstring,
  `_binding_reject`'s message, the `_UPDATE_DOC` sentence on what a sampler takes.
- `shaderbox/uniform_coerce.py`: `VALUE_GL_TYPES` (D1); a 1-D numeric ndarray is a sequence (D9).
- `shaderbox/core.py`: `Pass.pending_textures`, `Pass.script_textures`, the binding order in
  `render`, release in `release`.
- `shaderbox/pass_graph.py`: `PortKind` `"script"`, `node_ports(script_fed=...)`,
  `group_boundary`.
- `shaderbox/widgets/graph_state.py::ports_of`, `shaderbox/graph_canvas/adapter.py`.
- `shaderbox/widgets/uniform.py`: the texture row's preview, caption and both pick sites.
- `shaderbox/widgets/pass_list.py::_reads`, `shaderbox/tabs/code.py::_note_uniform`,
  `shaderbox/document.py::sampler_source` (docstring): D7.
- `shaderbox/intel/script.py`: D9.
- `shaderbox/copilot/backend.py` (`_sampler_reads`, `_motion_verdict`, the texture facts),
  `shaderbox/copilot/capabilities.py` (`ScriptWriteResult.texture_facts`),
  `shaderbox/copilot/prompt.py` (sampler-row vocabulary).
- `shaderbox/scripting/api_doc.py`, `shaderbox/help_content.py`: what a script returns for a
  sampler, row 0 at the bottom, the two dtypes.
- `ai_docs/conventions.md`: the 072 sampler paragraph (a script texture is a per-frame
  override at draw time, never a source; its precedence), the 102 D1 list of non-landing keys.
- `ai_docs/features/109_native_libraries/01_spec.md`: the out-of-scope item points here.
- Tests whose premise changes: `tests/test_script_warn_path.py` (`_SCENARIOS["not_scriptable"]`
  moves to a block-shaped fake, a `bad_texture` scenario, the oscillation test's reason set),
  `tests/test_script_engine.py` (the two sampler-key tests become texture tests;
  `test_script_status_reflects_sentinel_and_soft_errors` and
  `test_a_bad_key_reaches_the_strip_and_not_the_console` re-pointed at a block or a bad texture),
  and the comment in
  `tests/test_script_dry_run.py::test_a_refused_key_and_an_undeclared_one_are_both_per_key_errors`.

## Verification

Each gate names the break it must catch; each break is tried once and the commit says which.
Fixtures follow `tests/test_instances_routing.py::_document` (render once before the tick, so
the pass is compiled and no key is `held_uncompiled`); every field differs from the colour
of anything else the sampler could bind, and the canvas is cleared to a third colour first.
"Wired" below means the sampler is `u_<pass>` with that pass present and drawing a known
non-black colour; a gate whose break is only visible against a non-black source says so.

1. **Round trip, exact, both key forms.** A pass writes
   `texelFetch(u_field, ivec2(gl_FragCoord.xy), 0)` to an `f4` target the size of the field;
   the script returns a 3-channel `f4` field with `H != W` and every texel distinct, once as
   a pass block and once as a bare key. The canvas equals the field both times. Break,
   separately: bind black; upload the transpose; flip rows; add the texture path to the
   block loop only.
2. **Mixed broadcast.** Pass A declares `u_x` as a `float`, pass B as a `sampler2D`; a bare
   `u_x` with a field drives B's texture and A's float refuses it as a coercion error on A
   alone. Break: today's "any pass binds, so skip the rest".
3. **Normalized `u1`.** 0, 128, 255 read 0, 128/255, 1.0. Break: map `uint8` to moderngl `"u1"`.
4. **Script outranks `inputs`.** `Pass.render(inputs={"u_field": tex_of_known_colour})` with a
   script field in the slot draws the field. Break: let `inputs` win.
5. **Absent returns the source.** Wired. The next tick omits the key: the draw shows the
   wired pass's colour through `Document.render`. Break: keep binding the last script texture.
6. **The source is never written.** UNWIRED sampler (`NoSource`), because `Pass.render`'s
   value write-back runs only for a sampler not in `inputs`. Run several ticks and renders,
   save: `document.json` and the document's asset directories are byte-identical to the save
   before the script ran. Break: let the write-back store the bound texture under the sampler.
7. **Stop binds the source.** Wired. Stopping the pair makes the next draw show the wired
   pass. Break: ignore `stopped` on the texture path.
8. **A behavior failure holds the slot.** A field on tick 1, `raise` on tick 2: the draw
   still shows the field. Break: rebuild the slot before the run.
9. **A behavior failure touches no source, live or probed.** Live: after the tick-2 raise of
   gate 8, the sampler's value is the same object as before the script ran. Dry-run: a script
   sending a field that raises at frame k leaves no texture pair in any of `probe.samples`.
   Some sample time is at or after k. Break: put texture pairs in `last_driven` and record
   `last_good` for them (then `_freeze` restores the array into the source live, and the
   probe's sink carries it).
10. **Removal clears.** Wired. Delete `script.py`, reload, tick, render: the source shows,
    an instanced pass whose script is deleted draws fullscreen, and
    `script_driven_uniforms` is empty. Break, separately: no slot clear; no reset of the
    texture set in `_drop_script`.
11. **Refusal holds the last frame.** Wired. A good field, a render, then an `f8` field: the
    draw still shows the good field. Break: clear the slot entry on refusal.
12. **Refusal warns, reaches the strip, stays driven.** The `f8` tick warns `bad_texture`
    naming `float64` and the fix, the pair is in `ScriptStatus.soft_errors`, and it is in
    `script_driven_uniforms`. Break: leave refused pairs out of the texture-driven set.
13. **First-frame refusal binds black.** Wired. An `f8` field on the first tick: the draw is
    black, not the source. Break: fall back to the source when there is no last good frame.
14. **Reallocation.** A 64×32 field then a 32×64 field, then an f4 1-channel field then a
    u1 4-channel one of equal bytes, on a 64×64 canvas compared over the field's region:
    each draw matches its own field. Break: keep the first texture; separately, reallocate on
    byte size only.
15. **Uncopied.** After a tick, `pending_textures["u_field"] is` the returned array, once for
    a plain ndarray (always runs) and once for a 109 buffer built by
    `tests/test_native.py::_build` (skipped without `cc`). Break: `.copy()` in the engine.
16. **Wrap.** The allocated texture has `repeat_x`/`repeat_y` False. Break: moderngl's default
    (`True`). The filter is not gated: moderngl's default is already linear, so no break of
    ours could show.
17. **Dry-run isolation.** After `dry_run`, every pass's `pending_textures` is the same object
    as before (identity). Break: write the slot unconditionally (103's bug for populations).
18. **Texture statistics.** A 32×32 field (small enough that the digest's stride is 1) that
    goes A, B, A over three samples, B a permutation of A's texels so min, max and mean are
    equal: the facts carry each sample's shape, dtype, min, max, mean, and the field reads as
    MOVING; a constant field reads static. Break: compare first against last only;
    separately, drop the digest.
19. **A texture-only script reads as driving a field.** A script driving only a moving
    field: the write-script feedback carries the texture motion line and neither "values
    UNCHANGED (STATIC)" nor "Nothing animates". Break, separately: leave texture pairs in
    `_motion_verdict`; omit the texture case from its `not probe.driven` branch.
20. **One predicate, samplers.** For every refusal case (each wrong dtype, big-endian `f4`,
    ndim, channel count, non-contiguous, zero extent, over-limit extent) and each accepted
    shape, `glsl_type_of_value` returns `"sampler2D"` exactly when an engine tick lands the
    value. Compared against the tick, not against `texture_problem`. Break: give completion
    its own dtype list missing `u1`.
21. **One predicate, 1-D arrays.** For a 1-D `f4` ndarray of length 1..4 and a list of the
    same numbers (length 1: no type is offered and none lands), `coerce_uniform_value` against the matching float/vec uniform succeeds
    exactly when `glsl_type_of_value` names that type. Break: revert the coercion change.
22. **Only a single `sampler2D` is drivable.** `sampler3D`, `samplerCube`, `isampler2D` and
    `sampler2D[2]`, each with a float and with a field: `not_scriptable`, nothing bound. And
    every `GL_*` constant in `OpenGL.GL` whose name contains `SAMPLER` or `IMAGE` is outside
    `VALUE_GL_TYPES`. The other direction: one uniform of every value type, doubles
    included, compiled in a `#version 460` shader, is scriptable. Break: today's denylist;
    separately, leave the double family out of the set.
23. **Every `PortKind` has a pin, and the graph sees the slot.** `set(get_args(PortKind)) <=
    set(_PIN_BY_KIND)`; `ports_of` reports `"script"` for a fed sampler and for a fed
    `u_prev`, which emits no edge in any of the three views (flat, root, group tab); in the flat view a
    wired script-fed port still has its edge. Break, separately: drop the adapter entry; do
    not pass `script_fed`; leave the flat edge loop testing `kind == "wired"`.
24. **Pick while playing, combo.** Picking `none` on a playing script-fed sampler whose source
    is a pass (so the pick is a change) stops the
    pair and writes `NoSource`. Break: no stop at the `set_sampler_source` call site.
25. **Pick while playing, file.** A `file...` pick on a playing sampler (with the file dialog
    monkeypatched) writes the media. Break: today's `not playing` gate.
26. **Export.** An exported frame of a document whose script sends a field equals the field
    (pattern of `tests/test_export_script_wiring.py`). Break: write the slot only on the live
    tick.
27. **Strip chips.** A pass with two samplers wired to one producer, one fed: the chip
    stays; feed both: it goes. Break: drop the producer when any sampler of it is fed.
28. **Headless driven set.** After `dry_run` on a document with no live tick, a fed sampler
    is in `script_driven_uniforms`. Break: stash only `last_driven`.
29. **Copilot cell.** `_sampler_reads` for a fed, wired `u_<pass>` reads `<- script (...)`.
    Break: today's wiring-only lookup.
30. **Release.** After a render then `Pass.release`, each texture `script_textures` held is
    released (`tests/test_gl_lifetime_guards.py::_released`). Break: drop the release loop.

## Open questions for the user

None. Settled in chat before the draft: the gap is closed in the engine (not by `@instances`
one quad per cell), the native library writes a buffer and the draw uploads it (the library
has no GL context), and the plain upload is used because it measured fastest.

## Review history

Round 1, two reviewers (design against code and conventions; verification and blast
radius), both PARTIAL, probes outside the repo. Accepted and applied:

- D7 reversed. The draft removed a script-fed sampler from `effective_wiring`; that function
  has many consumers, and a drop's cycle check, the layout and the import plan would have
  flipped with script state (a cycle accepted while a script feeds an edge). The wiring now
  stays the user's and the override is reported per sampler.
- Behavior-level failures: the slot is not rebuilt on the early returns, and `_freeze` would
  have written `last_good` (or the live source object) for texture pairs into the source and
  into the dry-run's `samples`. D3/D4 now state the hold and exclude texture pairs from
  `last_good` and `_freeze`.
- Script removal never cleared `pending_instances` either (measured); D4 clears both.
- Broadcast targeting skipped a sampler pass whenever any value pass bound the key; D1 is
  per pass.
- `is_scriptable` admits every non-2D sampler type (measured); D1 tightens it.
- The copilot would have called a moving field STATIC (reproduced) and `_sampler_reads` would
  have said BLACK; D7/D8. min/max cannot see advection; the digest added. 103's motion rule is
  first-against-every-later, not consecutive; the citation fixed.
- Refused = driven, not skipped (D6); `uint8` → moderngl `"f1"` named (D2); clamp spelled as
  moderngl spells it (D3); the sentinel typed as an array (D3); reallocation keyed on shape,
  channels and dtype (D3).
- The panel's texture branch drops a pick made while playing and leaks the media; D7.
- The 1-D ndarray completion/coercion disagreement, pre-existing, fixed here (D9).
- Gates: the "inputs win" break was unobservable through `Document.render` once D7 changed
  the wiring (now a direct `Pass.render` gate, and observable either way); the native gate
  broke the fixture rather than the product (now `is` identity, break is an engine copy);
  gates split so each fails for one reason; gates added for removal, behavior failure,
  mixed broadcast, statistics, STATIC, pins, export, pick-while-playing, the copilot cell,
  and the sampler-type refusal.

Rejected: none.

Round 2, the same two reviewers against the patched text, both PARTIAL, every round-1 item
closed but one. Accepted and applied:

- D6 contradicted `_warn`, which always records the pair as skipped, and skipped is the only
  thing the script strip reads: a refused texture is now driven AND skipped.
- `_freeze` had no way to tell a texture pair; D3 keeps them in their own set.
- The tightened `is_scriptable` had no stated domain; D1 makes it an allowlist.
- The reversed D7 left three surfaces describing the wiring as what binds: the flat graph
  view's edge loop (and `group_boundary`) keyed on `kind == "wired"`, the strip's chips, and
  the shader editor's uniform note. D7 lists each, and two docstrings and 069 D9's sentence.
- A texture-only script would read "Nothing animates" once excluded from the value verdict;
  D8 adds the branch populations already have.
- The panel's pick-while-playing is two sites, not one: the combo pick calls
  `set_sampler_source` directly.
- Gates: the source-never-written gate needs an unwired sampler (the write-back skips one in
  `inputs`); gate 8's second break was masked by the other guard and moved to its own gate
  with a dry-run case; statistics gate uses a permutation so the mean matches too; the 1-D
  coercion case got its own gate (the cited test compares scalars only); the filter half of
  the wrap gate was unbreakable and is dropped; the no-copy gate also runs without `cc`.

Round 3, the same two reviewers, both PARTIAL, every round-2 item closed. Accepted and
applied: the allowlist would have dropped the double types scripts drive today (measured,
every value type coerces and writes); the texture set was missing at `_drop_script` and the
dry-run stash; D4 still said the pair joins `driven`, which is gate 9's break; the strip's
chips are per producer, so removing a producer could remove a true chip; a fed `u_prev` got no
port because the self-read check ran first; gate 9's live break could not reach the code
without `last_good`; length 1 disagreed between completion and coercion for lists too. The
copilot render line's blindness to a field is the same as for a population and moved to Out
of scope with a trigger.

Post-implementation review, two reviewers (code correctness; architecture, conventions and
spec fidelity), three rounds to PASS. Found and fixed, each with a gate seen red under its
break:

- The warning-edge clear added with this feature re-armed any key not skipped, so a fault
  sent every other frame warned on each return; it re-arms only for a key that landed.
- `copilot/backend.py::_format_uniforms` marked every driven name before the sampler branch,
  so the field cell never reached the model; gate 29 had called the helper on a stub.
- `_motion_verdict`'s population flag was true for every probe (older than 110); facts read
  only the last sample.
- First-frame refusal: panel, editor note and copilot cell described a held frame or the
  wired pass while the draw was black.
- A good block field left the bare key's refusal on the strip.
- `script_textures` outlived the field that filled it, first entirely, then when a recompile
  left the program without the sampler; the prune runs at the top of every draw.

Departures from the text above, accepted: the copilot cell reads
`(script field WxH, C x dtype)`; the routing is a `_route` method beside `_write_one`, as
the engine structures every other per-key step.
