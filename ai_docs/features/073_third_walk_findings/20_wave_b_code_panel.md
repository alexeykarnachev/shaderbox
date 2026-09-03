# 073 W-B — Code panel: completion providers, auto-trigger, `K` lookup (#2 #3 host halves)

Parent: `01_spec.md § W-B`. Host-only. The completion half landed third; the `K` half with
W-A's re-vendor.

## What landed (completion)

- **`shaderbox/completion.py`**, creating no GL context: the provider table. A `CompletionProvider` is a
  name, the tab kinds it serves, an optional context regex anchored at the caret, two prefix
  floors (auto / explicit) and a candidate function. `offer` concatenates every eligible
  provider's matches in table order, without repeats, capped at 50, so after `uniform ` the
  builtin declarations come first and the glsl words (`sampler2D`, `vec3`) still follow; a
  caret inside a line comment (`//`, or `#` on the script) gets nothing.
  Providers: `builtin uniforms` (shader tabs, context `\buniform\s+\w*$`, floors 0 / 0,
  candidates `float u_time;`-style declarations from `ENGINE_UNIFORM_DOCS`); `glsl` (shader
  and lib tabs, no context, floors 2 / 1, `SB_*` functions + the pass's uniforms +
  `_GLSL_WORDS`, which moved here); `python` (script tabs, floors 2 / 1, `keyword.kwlist`).
- **`matches`**: a candidate extends the prefix, or, for a multi-word candidate, any of its
  words does, so `u_ti` after `uniform ` finds `float u_time;`. Accepting replaces only the
  library's identifier prefix, so `uniform u_ti` + accept gives `uniform float u_time;`.
- **The driver** (`tabs/code.py::_drive_completion`) has three ways in: Ctrl+N / Ctrl+P
  (explicit), an EDIT in insert mode while no popup is open (auto), and a re-filter while the
  popup is open and the prefix moved. An edit is a change of `(tab.path, revision)` since the
  last frame, so cursor motion never opens anything. The edit that CLOSED a popup (an accept,
  or the keystroke that emptied the prefix) does not re-offer.
- **An unasked popup highlights nothing** (`_offer_completion` calls `complete_select(-1)`
  after an auto batch; editor `469eec4`'s noselect state, W-A): Enter and Tab then act as with
  no popup and close it, so `in` + Enter is a newline and not `int`; Down / Ctrl+N pick row 0
  and Enter accepts. An explicit Ctrl+N batch keeps the library's row-0 highlight. The earlier
  host workaround (cancel before Enter unless navigated) is deleted with it.
- **`symbol_doc(word, lib_functions)`**: `K`'s lookup, ready for the popup: the `SB_*`
  signature + `///` doc from the lib index, else `uniform <type> <name>;` + the doc line from
  `ENGINE_UNIFORM_DOCS`, else None.

## Review history

Round 1 (opus, demonstrated by probes): PARTIAL. Three findings, all fixed: the `was_open`
guard was recorded before the offer and so never fired (an accept re-opened the popup one
frame later); the first-firing provider shadowed the glsl words after `uniform` (`uniform sam`
offered nothing); `// uniform ` fired inside a comment. Fixes: the flag is recorded after the
offer; providers concatenate; a line comment offers nothing. Block comments are not tracked.

Round 2 (same reviewer, patched tree): PASS. Each finding closed against its original probe;
the concatenation keeps declarations first, yields no duplicate rows, and the 50 cap applies
to the concatenated list. Two named non-defects: a `//` inside a string literal on the script
tab (a `#`, rather) suppresses the rest of that line, and a flood in one provider could starve
the next at the cap; neither reachable at the real vocabulary sizes.

## Pinned by tests

`tests/test_completion.py`: `uniform ` offers every builtin declaration and filters by type
or name; one letter opens only on Ctrl+N, two open unasked; a complete word is not its own
candidate; the script tab offers keywords only and ignores `uniform`; `symbol_doc` reads the
index then the builtin table; through a real `Editor` with host completion: the second letter
opens, a no-edit frame leaves it alone, `uniform ` opens the builtin list and Enter lands the
whole declaration, and Enter on an unasked popup inserts a newline until the user navigates.

## The `K` half

Landed with W-A on the seam the editor session chose: `K` stays unbound in the library and
the host catches it through `ed_key`'s false (`hotkeys.py::_is_lookup_key`), resolves the
word under the caret (`completion.word_at` + `symbol_doc`) and pins `anchored_note` one cell
below the caret; any key or click dismisses it. Details in `10_wave_a_editor.md`.

## Manual verification (the maintainer, in the app)

1. Type `SB` in a shader: the popup opens on the second letter without Ctrl+N; keep typing
   and it narrows; type `(` and it closes.
2. Type `in` then Enter: a newline, not `int`. Type `in`, Down, Enter: `int`.
3. Type `uniform `: the builtin declarations appear; Enter lands `uniform float u_time;`.
   Type `uniform u_re`: only `vec2 u_resolution;` remains.
4. Ctrl+N on one letter still opens; Ctrl+N with the popup open still advances.
5. In the script: `wh` opens `while`.
