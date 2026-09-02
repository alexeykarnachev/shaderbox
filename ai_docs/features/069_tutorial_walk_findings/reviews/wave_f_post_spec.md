# W-F post-implementation review: spec fidelity and architecture

Commit under review: `d2ade88` ("069 W-F: draw the editor chrome from the library").
Anchors: `40_wave_f_editor_chrome.md`, `01_spec.md § W-F` + D6, `00_findings.md` rows 11/12/14/15/16,
`conventions.md`, `50_wave_e_keyboard.md`, `067_custom_editor.md`.

## Verdict

| Dimension | Verdict |
|---|---|
| Wave-spec fidelity | **PARTIAL** — every decision landed; two landed deviations the spec does not record (F2, F3) |
| Parent fidelity (`01_spec.md § W-F`, D6) | **PASS** |
| Findings closure (#11 #12 #14 #15 #16) | **PARTIAL** — #14 closes for every glyph except column 0 (F1) |
| Vendored set | **PASS** |
| Architecture | **PASS** |
| Docs | **PARTIAL** — the roadmap banner it edited is stale on W-B (F4) |

`make gates` could not be judged in this shell: it exits 2 at `test`, on a `glfw.get_video_mode`
segfault in `tests/conftest.py:61`'s `app` fixture, which reproduces identically on the parent
commit `0ce84f8` in a clean worktree (both `EXIT=139`). Not a W-F regression. What was judged
instead: `make check` green (exit 0, 0 errors), `tests/test_editor_ffi.py` 53 passed, and the 641
tests in the modules that do not take the `app` fixture all pass.

## Coverage: design decisions

| # | Decision | Status | Evidence |
|---|---|---|---|
| 1 | Re-vendor `c5c6ae2`, two files change, seventh added | LANDED | `VERSION` = `c5c6ae230a51ece5…`; `git ls-files shaderbox/resources/editor/` returns seven; the four unchanged files md5-match upstream (below) |
| 2 | Bind full ABI + make the mirror rule enforceable | LANDED | `_SIG` holds 93 entries; `nm -D` on the vendored `.so` reports 93 `ed_*` T-symbols |
| 2a | `_SIG` 65 → 93 | LANDED | `ffi.py:306-370`, counted 93 |
| 2b | Names test against `nm -D` | LANDED | `tests/test_editor_ffi.py:860` `test_the_binding_mirrors_every_export_of_the_vendored_binary`, both directions, skips on missing `nm` |
| 2c | Argtypes test against vendored `abi_probe.py` | LANDED, **deviation** | `tests/test_editor_ffi.py:885`; `_upstream_sig` at `:839`. Deviation F2 |
| 2c | `.pre-commit-config.yaml` exclude | LANDED | `.pre-commit-config.yaml:1` |
| 2c | `[tool.pyright] exclude` | LANDED | `pyproject.toml:92-94`; `uv run pyright shaderbox/resources/editor/abi_probe.py` → 0 errors |
| 2c | `abi_probe.py` byte-identical to upstream | VERIFIED | `diff` against `git -C ~/src/editor show c5c6ae2:ffi/probe.py` is empty |
| 3 | 28 new signatures + `ed_add_marker` 8→12 floats | LANDED | `ffi.py:227` `+ [ctypes.c_float] * 12`; the argtypes gate passes, which is the proof for all 93 |
| 3 | `add_marker` gains `text`, `gutter_glyph` | LANDED | `ffi.py:533-551`; live probe returns `((0.98,0.29,0.20,1.0),'E')` from `get_marker_gutter(9)` |
| 3 | `Style` enum, `set_style`, `get_style` | LANDED | `ffi.py:120`, `:701`, `:707`; live round-trip 0→1→0 |
| 3 | `set_draw_chrome` | LANDED | `ffi.py:697` |
| 4 | `set_draw_chrome(True)` in `get_session` | LANDED | `app.py:1357`, on the fresh handle after `set_palette` / `set_host_completion` |
| 4 | `ed_layout` gets the whole widget, `origin` back to (0,0) | LANDED | `code.py:545` `editor.layout((float(size_px[0]), float(size_px[1])), px_per_em)` |
| 5 | Five host sites unchanged, `editor_visible_rows` fixed | LANDED | `code.py:510` `max(0, int(size_px[1] / cell_h) - 1)`, with the reason in the comment; `_handle_mouse`, hover tooltip, blit untouched in the diff |
| 5 | `render_state`'s `gutter_px` → `text_origin` | LANDED | `render.py:87`, `:114`; call site `code.py:559` |
| 6 | `_draw_gutter` + call site deleted | LANDED | `grep -rn "_draw_gutter" shaderbox/ tests/` returns nothing |
| 6 | `draw_chrome`'s vim half deleted | LANDED | `code.py:344-383`; the `session is not None` block is gone, the `tab.kind == "shader"` block intact |
| 6 | `_MODE_BADGES` deleted, `Mode` import kept | LANDED | `grep` returns nothing; `Mode` still imported at `code.py:8` and used twice |
| 7 | Marker call: 0.20 fill, `STATE_ERROR` gutter, `FG_PRIMARY` text, `E` glyph | LANDED | `code.py:168-179` |
| 7 | `E` only under `LINE_NUMBERS` | LANDED (library behaviour) | live probe: gutter cells 0 with numbers off |
| 7 | Red tab tint stays | LANDED | `_draw_tab_row` untouched in the diff |
| 7 | Parent's 2px bar dropped | LANDED as decided | commit body names it |
| 8 | No stale-marker fingerprint | LANDED as decided | `_apply_markers` fingerprint still `(errors, hover_line)`; anchoring verified live |
| 9 | Error strip stays below the editor | LANDED as decided | `_draw_error_strip` untouched |
| 10 | `render.py` needs no kind branch | LANDED | `build_vertices` untouched in the diff |
| 10 | `render_state` gains `command_message` | LANDED | `render.py:107` |
| 11 | `STATUS_BG` → `_P["bg_0"]`, no literal colour, every slot mapped | LANDED | `theme.py:547` |

## Coverage: tests

| # | Test | Status | Evidence |
|---|---|---|---|
| 1 | `test_the_binding_mirrors_every_export_of_the_vendored_binary` | LANDED | `:860`; independently reproduced: `nm -D` gives 93, `_SIG` gives 93 |
| 2 | `test_the_binding_mirrors_the_upstream_signature_table` | LANDED, weakened | `:885`; see F2 |
| 3 | `test_a_marker_follows_a_line_inserted_above_it` | LANDED | `:903`; live repro reproduces `marked == [10]`. Falsifier deviation F3 |
| 4 | `test_draw_chrome_adds_a_gutter_and_a_status_frame` | LANDED | `:916`; live buckets `{BACKGROUND:1, GLYPH:91, CARET:1, FRAME:1, POPUP_GLYPH:9}` |
| 5 | `test_text_origin_moves_right_by_the_gutter_under_chrome` | LANDED | `:936`; live `text_origin (40.0, 0.0)`, `gutter_cells 4`, `cell (10.0, 21.0)` |
| 6 | `test_render_state_reacts_to_every_editor_dimension` extended | LANDED | `:462-469`, the `:zzz<CR>` case with the `get_command_line() is None` assertion |

## Coverage: Files-touched, including the "not touched" rows

Every listed file appears in `git show --stat d2ade88`. The "not touched, and why" rows all hold:

- **`ai_docs/dev_flow.md`** — untouched. `:269-275` describes `editor/` by module role (`ffi.py`
  binding, `render.py` pass, `input.py` translation); `:325` is one clause, "inline GLSL editor —
  main-window LEFT split". Neither names the gutter, the badge or the bottom bar. Reason holds.
- **`conventions.md`'s inline-editor entry** (`:362-380`) — untouched. Its editor content is the
  tab-bar / session / dirty-tracking model plus a vendoring pointer; nothing chrome-specific.
- **`input.py`, `build.sh`, `ui.py`, `popups/settings.py`, `ui_models.py`** — none in the stat.
  `build.sh` keys on `libeditor.so` by name, not on a file count, so the seventh vendored file
  does not reach it.

## Manual steps

All seven are maintainer-facing and not runnable here (the app needs a display this shell does not
have). Four of them have a headless proxy that was run and passed: the gutter/filler picture (12
gutter rows emitted for a 3-line buffer at a 300px widget), the status row (`FRAME:1`,
`POPUP_GLYPH:9`), the marker following an `O` above (`[10]`), and visual `p` replacing
(`'alpha beta'` → `'alpha alpha'`, reverting on one `u`). The error-line readability step is the
one a proxy contradicts in part; see F1.

## Parent fidelity (`01_spec.md § W-F`, D6)

| Parent bullet | Satisfied by |
|---|---|
| Re-vendor `libeditor.so`, update `VERSION` + the conventions entry | `VERSION` = `c5c6ae2…`; `conventions.md:871`, `:875`, `:902`. Parent said `68def59`; the wave spec records the supersession to `c5c6ae2` in its opening paragraph |
| Issues to file on `alexeykarnachev/editor` | Void by maintainer instruction, recorded in the wave spec's opening and in the roadmap banner |
| Status line inside the editor rect (D6) | ABI fact: `ed_layout` emits it as one `Kind.FRAME` plus `Popup_Glyph` text inside the widget rect, verified live. Host reserves nothing — `code.py:545` passes the whole widget |
| Gutter (D6): relative numbers, `~` filler | ABI fact: `chrome_for(.Vim)` sets `RELATIVE_NUMBERS` and `filler_glyph='~'`; live probe emits gutter glyphs on 12 rows over a 3-line buffer |
| Error lines: gutter mark in `STATE_ERROR`, 2px bar | Gutter mark landed (`code.py:172-173`, `gutter_glyph="E"`); the 2px bar dropped as decision 7, reason recorded |
| Stale markers: dirty fingerprint | Premise retired at `c5c6ae2`; decision 8, verified live |
| Files: `editor/ffi.py` binds `ed_set_style`, `ed_style`, `ed_filler_glyph` | `ffi.py:365-368`; `ed_filler_glyph` at `:359` |
| Tests for a pure gutter label function | Superseded: the library owns the label, so there is no host function to test. The four chrome tests pin the library's answer instead |

D6's constraint ("inside the editor rect, never on the host's bottom bar") holds: `draw_chrome`
after the deletion (`code.py:344-383`) contains no mode, ruler or command-line drawing.

## Findings closure

| Row | Closed by | Would the complaint recur? |
|---|---|---|
| 11 (vim symbols on the bottom bar) | `code.py:344-383` — the `session is not None` block deleted; the bar is path + `(unsaved)`/`compiled` + Open dir | No |
| 12 (relative numbers, `~` filler) | The binary at `c5c6ae2` plus `app.py:1357`; host `_draw_gutter` gone | No |
| 14 (red keyword on a red line) | `code.py:168-175` — fill 0.20, `text=COLOR.FG_PRIMARY` | **Partly.** Column 0 keeps its syntax colour; see F1 |
| 15 (marker does not move with an insert above) | The binary; verified live, marker 9 → 10 after `O` + a char at line 6 | No |
| 16 (visual `p` does not replace) | The binary; verified live, one undo step | No |

### Row 14's contrast, computed independently

`COLOR.STATE_ERROR` = `COLOR.SYN_KEYWORD` = `_P["red_b"]` = `#fb4934` (`theme.py:167`, `:179`).
`COLOR.BG_SURFACE` = `slot.BACKGROUND` = `_P["bg_0h"]` = `#161819` (`theme.py:126`, `:535`).
The marker fill is `fade(COLOR.STATE_ERROR, 0.20)` (`code.py:168`), translucent by ABI, so the band
the glyphs sit on is that fill over `BG_SURFACE`: **`#44221e`**.

| Foreground on `#44221e` | WCAG ratio |
|---|---|
| `COLOR.BG_SURFACE` `#161819` (the vim white-on-red analogy's wrong turn) | 1.26 |
| `COLOR.SYN_KEYWORD` `#fb4934` (the defect) | 4.10 |
| **`COLOR.FG_PRIMARY` `#ebdbb2` (landed)** | **10.27** |
| `_P["fg_0"]` `#fbf1c7` | 12.42 |

Every figure matches the spec and the commit body to two decimals, including the 0.35-alpha
comparison (band `#662922`, `FG_PRIMARY` 8.02, `BG_SURFACE` 1.62). The colour decision is sound and
the drop from 0.35 to 0.20 does raise `FG_PRIMARY`'s ratio as claimed.

## Vendored set

`git ls-files shaderbox/resources/editor/` returns seven: `VERSION`, `abi_probe.py`, `atlas.json`,
`atlas.png`, `libeditor.so`, `standard_keymap.md`, `vim_coverage.md`.
`VERSION` = `c5c6ae230a51ece592114f349447b1d79d9563ef`; `git -C ~/src/editor log -1 c5c6ae2` is
"Draw the gutter and status line from ed_layout behind a switch".

| File | Tree md5 | Upstream `c5c6ae2` md5 | |
|---|---|---|---|
| `atlas.png` | `5d476903890dc4f478539ce99aa29603` | `assets/atlas.png` same | match |
| `atlas.json` | `6f7cb7b04298572a9537ff8b5d8822ea` | `assets/atlas.json` same | match |
| `vim_coverage.md` | `debd97b544cbae5a9fcfb8eea1ff261d` | `docs/vim_coverage.md` same | match |
| `standard_keymap.md` | `ba4bd01e98073181ca34ab27be321312` | `docs/standard_keymap.md` same | match |
| `abi_probe.py` | — | `diff` against `git show c5c6ae2:ffi/probe.py` empty | **byte-identical** |

One correction to the spec's own record, not a defect in the commit: decision 1 locates the atlas at
`resources/atlas.{png,json}` in the editor repo; at `c5c6ae2` those paths are empty and the real
files are `assets/atlas.{png,json}`. The bytes match either way.

## Architecture

**`Editor.get_marker_gutter` (`ffi.py:555-577`)** — right shape. It matches the module's existing
out-param readers exactly: `ctypes.c_float()` locals, `ctypes.byref` at the call, a tuple return,
same as `get_cell_size` (`:726`) and `get_text_origin` (`:732`). It adds a `None` return for the
absent case, which is `get_command_line` (`:740`) and `get_command_line_prompt` (`:745`) convention.
The `chr(codepoint) if codepoint else ""` mirrors `get_command_line_prompt`'s `None if c == 0`.
Verified live: returns `((0.98,0.29,0.20,1.0),'E')`. Consistent.

**`Style` / `set_style` / `get_style`** — home and naming match what W-E expects. `50_wave_e_keyboard.md:9-10`
says W-F "defines `class Style(IntEnum): VIM = 0; STANDARD = 1`, and exposes `Editor.set_style` /
`Editor.get_style`"; `:519` says W-E "imports `Style` from `shaderbox.editor.ffi`"; `:871` marks
`editor/ffi.py` a no-change row for W-E; `:1286-1291` records that W-E's earlier `EditorStyle` name
was withdrawn in W-F's favour. Landed as `ffi.py:120` `class Style(IntEnum)`, `:701`, `:707`. Names
match exactly. `set_style`'s docstring also carries the ordering constraint W-E must honour
("call it BEFORE any set_chrome_flag the host wants to keep"), which puts the trap at the symbol
W-E will import rather than only in a spec. `set_style` uses the raise-on-false shape of
`set_palette` / `set_chrome_flag`, as specced. Round-trip verified live: 0 → 1 → 0.

**`theme.py::editor_palette`** — `slot.STATUS_BG: _P["bg_0"]` at `:547`, a token not a literal. All
six furniture slots map (`GUTTER_TEXT`, `GUTTER_CURRENT`, `FILLER`, `STATUS_BG`, `STATUS_TEXT`,
`STATUS_ACCENT`), plus `BACKGROUND` and `POPUP_PANEL`. No call site names a colour. The band is one
step off the ground: `#1d2021` against `#161819`, a 1.087 luminance ratio — a faint band, which is
what decision 11 says the manual pass judges.

**`tabs/code.py` after the deletions** — no dead helper. Every one of the sixteen module-level
functions has a caller (checked in-file and across `shaderbox/` + `tests/`); `_MODE_BADGES` and
`_draw_gutter` leave no residue. Imports are all still live: `Mode` (2 uses), `SPACE` (2), and
`app.font_12` is still used by the error strip. The bottom bar's remaining content uses
`ui_primitives.draw_copyable_text` for the path; the raw `imgui.button("Open dir", …)` at `:378` is
pre-existing and byte-identical to `d2ade88^:code.py:430`, so it is outside this wave's blast radius.

**`render_state`** — `text_origin: tuple[float, float]` replaces the scalar `gutter_px`, and
`command_message` sits beside `command_line`, matching the docstring rule the function states.

## Docs

- `conventions.md:871-875` and `:902` both corrected from "three files" to seven, with the file
  list spelled out and `ffi/probe.py` → `abi_probe.py` named as a rebuild step. Reads as the now.
- `conventions.md:912-916` appends what this re-vendor deleted, as the accumulating record the
  entry is written to be.
- `conventions.md:863-867` — the ABI-mirror entry now names both gates and says which one matters
  and why. Reads as the now.
- `067_custom_editor.md:31-33` marks the gutter-font out-of-scope item CLOSED by 069 W-F;
  `:163-169` points decision 13 at the conventions entry as the current vendored set and records
  the re-vendor. Both correct.
- `dev_flow.md` untouched, correctly (verified above).
- `ai_docs/roadmap.md:29-43` — banner edited, and it is accurate about W-F. It is stale about W-B;
  see F4.

---

## Findings

### F1 — the marker's text-colour override skips column 0, so finding #14's exact repro still shows one red glyph

**Severity: medium.** The wave's own manual step says "every word on the line is legible, with no
red-on-red"; the glyph at column 0 is the exception, and the finding's own example word is `vec3`.

Evidence, driving the vendored `.so` directly with the landed binding:

```
Language.GLSL, text 'vec3 c = brokenfn(x);'
  no marker : [(-0.5, (0.78,0.57,0.92)), (10.0, (0.78,0.57,0.92)), (20.1, (0.78,0.57,0.92)), ...]
  with marker: [(-0.5, (0.78,0.57,0.92)), (10.0, (0.92,0.86,0.70)), (20.1, (0.92,0.86,0.70)), ...]
```

The first glyph keeps its syntax slot colour; every later glyph takes the override. It is
positional, not content-dependent: `'int x;'`, `'AAAA'` and `'zzzz'` all leave their column-0 glyph
unrecoloured, while `'  indented'` (no glyph at column 0) is recoloured throughout. With `SYN_KEYWORD`
in the marked line's first cell the contrast on the `#44221e` band is **4.10**, the exact figure the
wave's own table labels "today's unreadable case", against 10.27 for the rest of the line.

The bug is in the vendored library's marker emission, not in host code — `code.py:168-175` passes
`text=COLOR.FG_PRIMARY` correctly and the library applies it from column 1 on. So the fix is not
this repo's to write.

**Fix:** record it. Add a row to `conventions.md ## Known quirks`, in the re-vendoring entry that
already carries measured library behaviour, stating that at `c5c6ae2` a marker's `text` colour
overrides every glyph on the line except the one at column 0, with the three-buffer probe above as
the evidence, so the next re-vendor knows what to re-measure.

### F2 — the argtypes gate substitutes our `Prim` for upstream's, so a `Prim` layout change is invisible to it

**Severity: medium.** The gate's stated job is that "every restype and argtype" tracks upstream.
`_upstream_sig` (`tests/test_editor_ffi.py:852`) evaluates upstream's signature expressions in
`{"ctypes": ctypes, "Prim": Prim}` where `Prim` is imported from `shaderbox.editor.ffi:44` — so the
two `POINTER(Prim)` entries (`ed_primitive`, `ed_primitives`) compare our pointer type against
itself and can never disagree.

Measured: `ctypes.POINTER` memoizes per type object, so two differently-shaped `Prim` classes give
different pointer objects (`ctypes.POINTER(Prim) is ctypes.POINTER(Prim2)` → `False`, `sizeof` 52 vs
4). Upstream's own `Prim` at `abi_probe.py:22-25` is field-for-field identical to ours today, so the
gate is correct right now; what it cannot do is tell you when that stops being true. The exposure is
narrow (two entries, one struct) but it is the same silent class the gate exists for: a `Prim` field
added upstream is a stride mismatch across the whole primitive array.

The spec's decision 2c says "a namespace holding `ctypes` and the `Prim` structure", which does not
say whose. So this is a landed choice the spec neither authorised nor forbade, and it is not recorded.

**Fix:** in `_upstream_sig`, `exec` the upstream file's own `class Prim` definition into the
namespace (it is a self-contained `ctypes.Structure` at `abi_probe.py:22-25`, parseable out of the
same AST) instead of importing ours, and add one assertion that `ctypes.sizeof` of the two agrees —
then the gate covers the struct as well as the signatures.

### F3 — test 3's falsifier is not the one the spec records, and the spec was not updated

**Severity: low.** `40_wave_f_editor_chrome.md:806-807` gives test 3's falsifier as "the pre-`c5c6ae2`
binary answers 9 ... Measured both ways". That falsifier cannot be exercised against the landed tree:
`ensure_loaded` (`ffi.py:379-382`) `getattr`s every `_SIG` name at load, and `_SIG` now binds
`ed_set_draw_chrome` and `ed_draw_chrome`, which the old binary does not export.

Evidence — the two names in `_SIG` that the parent commit's binary lacks:

```
$ nm -D --defined-only <d2ade88^ libeditor.so> | awk '$2=="T" && $3~/^ed_/' | wc -l
91
missing from old (present in _SIG): ['ed_draw_chrome', 'ed_set_draw_chrome']
```

So swapping the old binary in raises `AttributeError` out of the binding loop before the test body
runs. The implementer reported this correctly; the spec still states the unavailable falsifier as
measured. The test itself is sound and its assertion is right (verified live: `marked == [10]`), and
the commit body records an alternative falsification ("the anchoring test red asserting the
pre-`c5c6ae2` answer of 9"), which is the falsifier that is actually reachable.

**Fix:** amend `40_wave_f_editor_chrome.md`'s test 3 falsifier to the one that is reachable —
assert the pre-`c5c6ae2` answer of 9 against the new binary and watch it go red — and add one clause
saying the old-binary swap is not available because `_SIG` now names two exports it lacks.

### F4 — the roadmap banner this wave edited says W-B is next, but W-B landed two commits earlier

**Severity: low.** The banner at `ai_docs/roadmap.md:29` reads "069 in progress: W-C, W-A and W-F
landed; W-B spec in review", and `:41` reads "**W-B next** (`30_wave_b_prose_diet.md`, pre-review
converging), then W-E, W-G, W-D, W-H".

W-B is not in review and is not next. `git log --oneline` on `dev`:

```
d2ade88 069 W-F: draw the editor chrome from the library
0ce84f8 069 W-B fixes: derive the gate's domain, pin the gear
ccd446b 069 W-B: cut UI prose to budget and gate it
```

Both W-B commits precede the one under review, so the wave that edited this block had them in its
own history. The W-F half of the edit is accurate (`c5c6ae2` vendored, 93 exports, two gates,
`set_draw_chrome` per session, host chrome deleted). The banner is the file `CLAUDE.md`'s cold-start
chain reads second, so a stale "next" is the highest-traffic wrong sentence in the repo.

**Fix:** rewrite the banner block in full so it reads "W-C, W-A, W-B and W-F landed" and names
**W-E next** (the wave whose spec is written and whose only blocker, `Style` in `editor/ffi.py`,
this commit removed), then W-G, W-D, W-H.

### F5 — `ruff check shaderbox/` is dirty on the vendored probe, though `make check` is green

**Severity: informational.** The pre-commit exclude (`.pre-commit-config.yaml:1`) and the pyright
exclude (`pyproject.toml:94`) are both correct and `make check` exits 0. But `[tool.ruff]`'s
`extend-exclude` (`pyproject.toml`) lists only `ai_docs/`, so a bare `uv run ruff check shaderbox/`
reports 6 errors on `abi_probe.py` (`SIM905` x3, `B905`, `RUF007`, `B007`) — the exact six decision
2c measured. Anyone invoking ruff directly rather than through `make check` sees a red tree.

Not a spec violation: decision 2c chose the pre-commit `exclude:` deliberately, naming the
`emoji-test.txt` precedent, and that precedent has the same property. Recording it so a future
session does not read those six as a regression.

**Fix (optional, and consistent with the file's own rationale):** add
`shaderbox/resources/editor/abi_probe.py` to `[tool.ruff] extend-exclude` beside `ai_docs/`, so the
"upstream's bytes, house rules do not apply" decision holds for every invocation rather than only
the gated one.

## False trails

- **The raw `imgui.button("Open dir")` at `code.py:378` bypassing `ui_primitives`.** Pre-existing and
  untouched — byte-identical to `d2ade88^:code.py:430`. Outside this wave.
- **`make gates` red.** The segfault is `glfw.get_video_mode` at `conftest.py:61`, reproduced on the
  parent commit `0ce84f8` in a clean worktree with the same exit 139. A display-less shell, not W-F.
- **`Editor.get_marker_gutter` having only a test caller.** That is the sanctioned shape for this
  module: `conventions.md:861-863` states the binding is a complete ABI mirror by design, and the
  method is the reader for an export the wave was required to bind.
- **`ed_marker_tooltip` still called from nowhere.** Decision 7 states this and keeps it: the error
  strip is where an error's text is read. Unchanged by design.
- **`_apply_markers`'s fingerprint not gaining a dirty bit.** Decision 8, deliberate, and the
  premise it drops was verified retired (marker 9 → 10 live).
- **The spec locating the editor atlas at `resources/atlas.*`.** The real path at `c5c6ae2` is
  `assets/atlas.*`; the vendored bytes match either way, so nothing shipped is wrong.

## Coverage statement

Walked all eleven numbered design decisions (including 2a/2b/2c), all six tests, every Files-touched
row including the five "not touched, and why" rows, and all seven manual steps. Read end-to-end:
the full diff of `shaderbox/tabs/code.py`, `shaderbox/editor/ffi.py`, `shaderbox/editor/render.py`,
`shaderbox/app.py`, `shaderbox/theme.py`, `pyproject.toml`, `.pre-commit-config.yaml`,
`tests/test_editor_ffi.py`, and the three edited doc files; plus `40_wave_f_editor_chrome.md` in
full, `01_spec.md § W-F` + D6, `00_findings.md` rows 11/12/14/15/16, the cited `50_wave_e_keyboard.md`
sections, `dev_flow.md`'s module map, and `conventions.md`'s three edited or cited entries.

Independently executed rather than taken from the spec: the seven-file `git ls-files`; md5 of the
four unchanged files against `~/src/editor` at `c5c6ae2`; `diff` of `abi_probe.py` against
`git show c5c6ae2:ffi/probe.py`; `nm -D` export counts on both the new (93) and old (91) binaries;
the `_SIG` entry count (93); the WCAG table at both alphas from the `theme.py` hex values; a live
`ctypes` session exercising `set_draw_chrome`, `get_style` round-trip, `get_marker_gutter`, marker
anchoring, the text-colour override across four buffers, filler-row emission, and visual `p`;
`make check` (exit 0); `tests/test_editor_ffi.py` (53 passed); the 641 non-`app`-fixture tests
(passed); and the parent-commit worktree reproduction of the `make gates` segfault.

Not verified: the seven manual steps as the maintainer sees them in the running app (no display in
this shell) — four have headless proxies that passed, and the error-line step is where F1 was found.
