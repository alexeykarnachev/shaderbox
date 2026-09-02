# 069 W-F: Editor chrome from the library

Implementation spec for wave W-F of feature 069. The parent spec (`01_spec.md § W-F`) fixes the
shape; this file fixes the code. Locked decision D6 applies and is not re-opened: the vim
furniture is drawn INSIDE the editor rect, in the editor's own visual language, never on the
host's bottom bar.

**The parent's W-F bullets are superseded by the editor's own delivery.** The parent was written
against upstream `68def59`, where `ed_layout` drew no furniture and the ABI was a query surface, so
its bullets read as "host re-derives the picture" plus "file four issues upstream". The editor
session then landed `c5c6ae2` ("Draw the gutter and status line from ed_layout behind a switch"),
which emits the whole furniture as primitives, anchors markers like nvim extmarks, and gives a
marker a text colour. All four findings are fixed there. **The maintainer wants no GitHub issues
filed; the parent's "issues to file" bullet is void.** What survives from the parent is D6's
constraint and the division of labour with W-E: W-F binds `ed_set_style` / `ed_style`, W-E owns
the keymap setting that calls them.

W-C and W-A land before this wave. Neither touches `tabs/code.py`, `editor/`, or `theme.py`'s
editor palette, so no citation here is at risk from them. Every citation names a symbol.

## Goal

The code panel stops being an imgui caption row bolted to a text widget. The gutter shows nvim's
`number relativenumber` picture, the `~` filler runs past the end of the buffer, and the status row
sits inside the editor rect with the mode badge, the `line,col` ruler, and the `:` / `/` / `?` line
with its own block caret while one is open. None of it is re-derived: the library emits every piece
as a primitive in the same array as the text, in the same font, on the same cell grid, from the
emitters `behavior_test.odin` pins against nvim. An error line becomes readable, because the marker
now carries the foreground colour and the glyphs on that line stop being red-on-red. A marker
follows its line through edits the way an extmark does, so the red band is where the broken code is
now rather than where it was at the last compile. The host's bottom bar keeps only host things:
file path, compiled or unsaved, Open dir, Copilot.

## Findings folded

Five, quoted verbatim from `00_findings.md`:

- **#11** (UX feature request, Code panel, bottom bar): "vim symbols ('NORMAL', line number,
  command line input) are on the same line as the file path, compile status, 'Open dir' button.
  Mess. Make the vim symbols part of the editor itself, first-class editor elements — we have the
  custom editor exactly for this."
- **#12** (UX feature request, Code panel, gutter): "the left strip with line numbers is wrong. We
  want relative line numbers, with `~` for the empty rows past the end, and so on. The editor should
  replicate real vim as much as possible. The editor lib must provide this — if it doesn't, file the
  issue there when doing this."
- **#14** (UX + editor-lib gap, Code panel, error line): "when I trigger an error, the line gets
  highlighted red but the text keeps its syntax colours — red keyword on a red line is unreadable.
  Real vim flips the colours (error highlight group overrides the text colour)."
- **#15** (ENGINE / UX, Code panel, error line): "the red line doesn't move when I insert a line
  above the error: error on 9, insert at 7, the code that caused it is now on 10 but the red stays
  on 9. Resets on save, but before that the layout is confused. Handle it properly — probably reset
  all errors when the file changes."
- **#16** (BUG editor lib, Code panel, visual-mode paste): "with something in the register, select
  a span and paste: the selection is not replaced, the paste lands on top of it, making a mess."

#12's "the editor lib must provide this — if it doesn't, file the issue there" is the clause this
wave answers by re-vendoring rather than by filing: the lib does provide it as of `c5c6ae2`.

#16 needs no ShaderBox code at all. It is closed by the binary swap, and the vendored
`vim_coverage.md` at `c5c6ae2` records it (`:293`, "`p` `P` over a visual selection — the selection
is replaced, as one" undo step; `:313`, "A visual `p` writes the register with what it replaced,
and `P` does" not). The manual pass verifies it.

## Out of scope

- **The keymap SETTING** (`EditorSettings.keymap`, the Settings combo, the call in
  `_apply_editor_settings_to`): **W-E** (D5). W-F binds `ed_set_style` / `ed_style` and exposes them
  as `Editor.set_style` / `Editor.get_style` so W-E has something to call. No W-F code calls either;
  every session stays on the vim style, which is what a fresh handle already is (verified:
  `ed_style` returns 0 on a fresh handle).
  **One ordering constraint W-E must honour, recorded here because W-F is where `set_style` is
  written.** `ed_set_style` calls `editor_set_keymap` and then replaces the WHOLE `Chrome` from
  `chrome_for(style)` (`ffi.odin:940`), so it resets `LINE_NUMBERS`, `RELATIVE_NUMBERS` and the
  status flags to that style's defaults. Measured: with `LINE_NUMBERS` set False by the host,
  `ed_set_style(h, 0)` leaves it True. `ed_draw_chrome` is NOT part of `Chrome` and survives the
  switch (measured: still True after `ed_set_style(h, 1)`), so this wave's own switch is safe. W-E
  must therefore call `set_style` BEFORE
  `set_chrome_flag(ChromeFlag.LINE_NUMBERS, settings.show_line_numbers)` in
  `_apply_editor_settings_to`, or the user's line-numbers setting is silently discarded on every
  settings apply. The same constraint is filed in `02_keybindings.md` and in
  `50_wave_e_keyboard.md`, whose apply block already places `set_style` first and now says why.
- **The chord ownership audit** and the per-keymap reserved sets: **W-E**. It reads
  `resources/editor/vim_coverage.md` and `standard_keymap.md`, which are ALREADY vendored and
  byte-identical at `c5c6ae2` (see § Verified / corrected premises), so W-E does not wait on this
  wave for its inputs, only for the `.so` that makes the standard style reachable.
- **A standard-keymap gutter/status design of its own.** The parent's Out-of-scope item stands
  unchanged: under the standard style the wave relies on what the library draws, which is
  `chrome_for(.Standard)`: no filler column, no modal badge, a caret readout in the same row.
  Verified by layout: the standard style still emits a status `Frame` and its `Popup_Glyph` text
  (three glyphs against vim's nine on the same buffer). Trigger unchanged: the maintainer uses the
  standard keymap daily and wants more.
- **Windows `libeditor.dll`.** The re-vendor is the Linux `.so` only. Trigger: next `/ship` on a
  Windows host.
- **The `_MODE_BADGES` colour scheme as a host concept.** It is deleted, not ported: the badge is
  the library's, coloured from `Status_Accent`. A per-mode host colour would need a slot the ABI
  does not have.
- **The 067 out-of-scope item "imgui-side font of the gutter matching the atlas font"** is not
  deferred further; it is CLOSED by this wave, because the gutter is now atlas-drawn like the text.
  Said here so a reader does not go looking for a deferral that no longer has a subject.

## Design decisions

### 1. Re-vendor `c5c6ae2`: two files change, not five

The parked build is at
`/tmp/claude-1000/-home-akarnachev-src-shaderbox/6d39c1c7-0520-4c0a-8808-186ecbf60c39/scratchpad/editor_c5c6ae2/`.
Six files are vendored today: `libeditor.so`, `atlas.png`, `atlas.json`, `VERSION`,
`vim_coverage.md`, `standard_keymap.md` (`git ls-files shaderbox/resources/editor/`). **Two of them
change.** Copy `libeditor.so` and `VERSION` into `shaderbox/resources/editor/`, and do NOT copy the
other four: `atlas.png` is byte-identical (md5 `5d476903890dc4f478539ce99aa29603` both sides),
`atlas.json`, `vim_coverage.md` and `standard_keymap.md` are byte-identical between the tree and the
parked directory (`diff` clean on all three), and the charset was not re-baked. A copy of an
identical file is a no-op in the diff, so the wave's `git status` shows exactly the files that
actually changed, and a reviewer reading the diff sees the binary swap and nothing else. The wave
ADDS a seventh, `abi_probe.py` (item 2c), which is a new vendored file rather than a changed one.

`build.sh` needs no change. Its editor handling is a Windows-stage strip of `libeditor.so` by name
(`build.sh:48-51`) plus a `verify_clean` `find` for the same name (`:72`); neither enumerates the
directory, so a file count that did not change was never the thing it keyed on.

`conventions.md ## Known quirks` owns the vendored sha in the entry beginning "**The vendored editor
binary (`shaderbox/resources/editor/`) rebuilds from a COMMITTED editor-repo sha**". That entry
under-counts the vendored set in TWO sentences: its shipped-files list says "`libeditor.so` +
`atlas.{png,json}` + `VERSION` (the sha)", and its rebuild step says "copy the three files"
(`conventions.md:857`). Both predate the two keymap docs, which are also vendored and which W-E's
audit test reads. Correct both to name all **seven** files after this wave: `libeditor.so`,
`atlas.png`, `atlas.json`, `VERSION`, `vim_coverage.md`, `standard_keymap.md`, `abi_probe.py`. The
sibling re-vendoring entry repeats "copy the three files" at `:884` and gets the same correction. A
doc that under-counts the vendored set is exactly what lets a re-vendor forget one, which is why
this is a correction and not a tidy-up.

The sibling entry "**Re-vendoring the editor: rebuild, copy, then delete the mitigations the new sha
makes dead**" is the one this wave is executing. Its measured examples (the `bf0f8d5` Ctrl+N
intercept, the visual-scroll consume-noop) stay as the record. Append what THIS re-vendor deletes:
the host `_draw_gutter`, the bottom bar's vim half, and the marker-fill-only error mark. That is the
entry's whole purpose and it is written as an accumulating record, not a snapshot.

### 2. Bind the full ABI, and make the mirror rule enforceable

`conventions.md ## Known quirks` states that `editor/ffi.py` is "a ctypes MIRROR of the vendored C
ABI. Completeness against the ABI is the point; deleting the unused half would make the binding lie
about the library's surface." **That claim is currently false and nothing checks it.** Measured:
`nm -D` on the vendored `65264dc` `.so` lists 91 `ed_*` exports; `_SIG` in `editor/ffi.py` binds 65.
Twenty-six exports are unbound, and they have been unbound since before this feature. `c5c6ae2`
takes the export count to 93.

So this wave does two things in one commit, per the maintainer's standing rule that a sweep ships
with the check that prevents its recurrence:

**(a) Bind every export.** `_SIG` grows from 65 to 93 entries. The ones this wave's own behaviour
needs are listed in item 3; the rest are bound with argtypes and restype and no `Editor` method,
which is exactly the shape the convention entry describes and defends. `ed_new` and
`ed_language_for_path` take no handle; everything else takes `c_void_p` first.

**(b) A completeness test.** `tests/test_editor_ffi.py` gains a test that reads the export set out
of the vendored binary and asserts `_SIG` covers it exactly:

```python
def test_the_binding_mirrors_every_export_of_the_vendored_binary() -> None:
    out = subprocess.run(
        ["nm", "-D", "--defined-only", str(EDITOR_RESOURCES_DIR / "libeditor.so")],
        capture_output=True, text=True, check=True,
    ).stdout
    exported = {
        parts[2] for line in out.splitlines()
        if len(parts := line.split()) == 3 and parts[1] == "T"
        and parts[2].startswith("ed_")
    }
    assert exported == set(_SIG), (
        f"unbound: {sorted(exported - set(_SIG))}; "
        f"bound but absent: {sorted(set(_SIG) - exported)}"
    )
```

Both directions matter. Unbound-export is the mirror rule. Bound-but-absent is the re-vendor
regression, and it is worth having for a narrower reason than "it would otherwise crash":
`ensure_loaded` (`ffi.py:307`) already `getattr`s every `_SIG` name on the first `Editor`
construction, so a dropped export already fails the suite loudly, at every test that opens an
editor. What the assertion adds is the NAME of the missing export in one message, instead of an
`AttributeError` thrown out of a binding loop.

`nm` is a binutils tool, present wherever the `.so` is built, and the test skips (not passes) if the
binary is absent, the same posture `make gates` takes for a display-less smoke. The alternative,
retyping the export list into the test, is the thing the parent spec already rejected for W-E's
keymap lists: a list retyped from an artifact stops tracking it.

**(c) An ARGTYPES gate, because (b) checks names and names are the easy half.** A missing or wrong
argument type is the silent class: ctypes pushes what the binding declares, the callee reads what
the Odin proc declares, and a short call reads its trailing parameters off whatever the stack
happens to hold. No exception, no crash, a wrong answer. Five of the twenty-eight signatures in item
3 were wrong exactly this way when this spec was first drafted, and the name-only test passed on
every one of them. So the class gets a gate, not a warning.

**The upstream artifact exists, and it is authoritative.** `ffi/probe.py` at `c5c6ae2` is the editor
repo's own reference ctypes binding ("Exercises the whole C ABI from Python, the way a host would
... this checks the BOUNDARY: that every exported function is callable through ctypes"), and it
carries a module-level `_SIG` dict of exactly the same shape as ours: `{name: (restype, argtypes)}`.
Verified against the parked build: it holds **93 entries, unique, and its key set is byte-equal to
the 93 `ed_*` symbols `nm -D` reports** (both set differences empty). It is maintained by the side
that owns the ABI, it is exercised by the editor's own `make ffi && python3 ffi/probe.py`, and it
independently agrees with `ffi.odin` on all five signatures this spec had wrong. So no generated
`abi.json` and no `scripts/` Odin parser is needed; the artifact to vendor already exists and is
already the editor repo's own gate on itself.

**Vendor it, and exclude it from the formatters.** `ffi/probe.py` at the vendored sha is copied to
`shaderbox/resources/editor/abi_probe.py` beside `VERSION`, as a seventh vendored file. It is
vendored DATA, not a module the app imports: nothing in `shaderbox/` imports it, the test reads it
as a file, and it is renamed on the way in so no one mistakes it for a repo-authored probe.

**The gate only works while the file stays byte-identical to upstream, and the repo's own hooks
would rewrite it.** Measured: `ruff` reports six lint errors on the file (`SIM905` x3, `B905`,
`RUF007`, `B007`), and `.pre-commit-config.yaml` runs `ruff` with `--fix` plus `ruff-format`, so
committing it unguarded would silently reformat and partially auto-fix a file whose whole value is
being upstream's bytes. So the wave adds it to the config's existing top-level `exclude:` regex,
which already carries exactly this precedent for a vendored data file
(`shaderbox/resources/emoji/emoji-test.txt`):

```yaml
exclude: ^(projects/|shaderbox/resources/emoji/emoji-test\.txt$|shaderbox/resources/editor/abi_probe\.py$)
```

That covers `ruff`, `ruff-format`, `trailing-whitespace` and `end-of-file-fixer` in one place.

**`pyright` needs its own exclusion, and this is a decided step rather than a check.** Its hook is
`pass_filenames: false` with `entry: uv run pyright shaderbox`, and `[tool.pyright]` has
`include = ["shaderbox"]` with **no `exclude` key at all**, so the vendored file is reached.
Measured on it: **three errors** (`reportOptionalIterable` at `:382`, and a
`reportAssignmentType` tuple-size mismatch at `:1391` where a `_fields_` entry may carry a
bitfield width). So `pyproject.toml` gains the key:

```toml
[tool.pyright]
exclude = ["shaderbox/resources/editor/abi_probe.py"]
```

Adding `exclude` does not narrow anything else: pyright's default `exclude` list applies only when
the key is absent, and the defaults it would drop (`**/node_modules`, `**/__pycache__`, `**/.*`) are
directories this repo does not type-check anyway. The implementer confirms `make check` is green
after the copy rather than assuming it.

This is not a sidestepped convention, and the `## Hard rules` ban on `# type: ignore` is not in
tension with it: the file is not this repo's source, no `# noqa` or `# pyright: ignore` is added to
it (which is what the ban forbids), and the rule it would otherwise be held to is a house style
upstream never agreed to. Not excluding it is what would be wrong, because a formatter or fixer run
would break the byte-identity the comparison exists to make.

**W-F owns `Editor.set_style`, `Editor.get_style` and the style enum; W-E only calls them.** They
are named once, here, in `editor/ffi.py` (item 3 gives the bodies and the `Style` enum). W-E's spec
places the call in `_apply_editor_settings_to` and takes the names as given. If the two specs ever
disagree on the enum's name, W-F's is the definition, because the binding is W-F's file and the
mirror rule is W-F's gate.

A verification the implementer owes, because it is what the whole gate rests on: after copying,
`diff` the vendored file against `git -C ~/src/editor show <VERSION>:ffi/probe.py` and confirm it is
byte-identical. If a hook has touched it, the diff is the thing that says so.

```python
def test_the_binding_mirrors_the_upstream_signature_table() -> None:
    # The editor repo's own reference binding, vendored at the same sha as the .so.
    # Parsed, never imported: it runs a probe session at import time.
    upstream = _parse_upstream_sig(EDITOR_RESOURCES_DIR / "abi_probe.py")
    ours = {name: (restype, list(argtypes)) for name, (restype, argtypes) in _SIG.items()}
    assert ours == upstream
```

`_parse_upstream_sig` is a test-local helper that `ast.parse`s the file, finds the `_SIG` assignment,
and evaluates each value in a namespace holding `ctypes` and the `Prim` structure, the same two
names the upstream file's signatures reference. Comparison is on the ctypes type OBJECTS, which
compare by identity for the scalar types and by cached identity for `POINTER(T)` (ctypes memoizes
pointer types, so `POINTER(c_float) is POINTER(c_float)`), so `_P(ctypes.c_float)` on our side
equals `ctypes.POINTER(ctypes.c_float)` on theirs.

*Falsifier:* change any one argtype or restype in `_SIG` and the assertion names the entry.
Measured end to end before writing this: the helper parses both tables (93 upstream entries, 65
ours), `ctypes.POINTER(c_float) is ctypes.POINTER(c_float)` is True so the pointer comparison is
sound, and across the 65 entries the two tables already share, the ONLY mismatch is
`ed_add_marker`, which is precisely the signature this wave changes. So the gate is red on today's
tree for exactly the right reason and goes green when the wave lands. Verified further against the
five defects this spec shipped in its first draft: with `"ed_find": (ctypes.c_int32,
[c_void_p, c_char_p, c_bool])` the test reports `ed_find` and passes once corrected to
`(ctypes.c_bool, [c_void_p, c_char_p, c_bool, c_bool])`. This is the gate that catches a WRONG
signature; test (b) catches a MISSING one; and after a future re-vendor, a signature the editor
changed goes red on the next `make gates` rather than corrupting a stack in the frame loop.

The `conventions.md` re-vendoring entry gains the one step this adds: copy `ffi/probe.py` at the new
sha to `resources/editor/abi_probe.py` alongside the `.so`, and let the two tests say whether the
binding still mirrors it.

### 3. The ctypes signatures for every added or changed export

Added to `_SIG`, with `_P = ctypes.POINTER` as the module already defines. The full 28 new entries
follow the same shape; these are the ones with behaviour attached to them in this wave:

```python
    "ed_set_draw_chrome": (None, [ctypes.c_void_p, ctypes.c_bool]),
    "ed_draw_chrome": (ctypes.c_bool, [ctypes.c_void_p]),
    "ed_set_style": (ctypes.c_bool, [ctypes.c_void_p, ctypes.c_int32]),
    "ed_style": (ctypes.c_int32, [ctypes.c_void_p]),
    "ed_set_chrome_style": (ctypes.c_bool, [ctypes.c_void_p, ctypes.c_int32]),
    "ed_chrome_flag": (ctypes.c_bool, [ctypes.c_void_p, ctypes.c_int32, _P(ctypes.c_bool)]),
    "ed_filler_glyph": (ctypes.c_int32, [ctypes.c_void_p]),
    "ed_set_filler_glyph": (None, [ctypes.c_void_p, ctypes.c_int32]),
    "ed_set_number_width": (None, [ctypes.c_void_p, ctypes.c_int32]),
    "ed_marker_count": (ctypes.c_int32, [ctypes.c_void_p]),
    "ed_marker_gutter": (
        ctypes.c_bool,
        [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32]
        + [_P(ctypes.c_float)] * 4
        + [_P(ctypes.c_int32)],
    ),
    "ed_color": (
        ctypes.c_bool,
        [ctypes.c_void_p, ctypes.c_int32] + [_P(ctypes.c_float)] * 4,
    ),
    "ed_reset_theme": (None, [ctypes.c_void_p]),
    "ed_primitive": (ctypes.c_bool, [ctypes.c_void_p, ctypes.c_int32, _P(Prim)]),
    "ed_tab_width": (ctypes.c_int32, [ctypes.c_void_p]),
    "ed_language": (ctypes.c_int32, [ctypes.c_void_p]),
    "ed_view_flag": (ctypes.c_bool, [ctypes.c_void_p, ctypes.c_int32, _P(ctypes.c_bool)]),
    "ed_line_spacing": (ctypes.c_float, [ctypes.c_void_p]),
    "ed_host_completion": (ctypes.c_bool, [ctypes.c_void_p]),
    "ed_class_at": (ctypes.c_int32, [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32]),
    "ed_insert_at": (None, [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_char_p]),
    "ed_replace_at_cursor": (
        ctypes.c_bool,
        [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_bool],
    ),
    "ed_replace_all": (
        ctypes.c_int32,
        [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_bool],
    ),
    "ed_paste": (ctypes.c_bool, [ctypes.c_void_p, ctypes.c_bool, ctypes.c_int32]),
    "ed_set_line_selection": (None, [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32]),
    "ed_find": (
        ctypes.c_bool,
        [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_bool, ctypes.c_bool],
    ),
    "ed_find_count": (ctypes.c_int32, [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_bool]),
    "ed_find_next": (ctypes.c_bool, [ctypes.c_void_p, ctypes.c_bool]),
```

The implementer diffs every one of the 28 against `ffi.odin` at `c5c6ae2` before running the suite,
and item 2c's argtypes test is what makes that non-optional: the name-only test in 2b passes on a
signature that is short an argument, and five of the rows below were wrong exactly that way in this
spec's first draft. Where this list and `ffi.odin` disagree, `ffi.odin` wins. Odin's
`bool` is one byte and maps to `ctypes.c_bool`; `[^]u8` out-buffers map to `_P(ctypes.c_ubyte)` with
an `i32` capacity, as every existing text getter already does.

**One CHANGED signature.** `ed_add_marker` gains four floats between the gutter colour and the
glyph:

```python
    "ed_add_marker": (
        None,
        [ctypes.c_void_p, ctypes.c_int32]
        + [ctypes.c_float] * 12
        + [ctypes.c_int32, ctypes.c_char_p],
    ),
```

This is the one entry where getting it wrong is silent rather than loud: ctypes would happily push
eight floats at a twelve-float callee and the marker would read the glyph and tooltip out of
garbage. `Editor.add_marker` changes with it:

```python
    def add_marker(
        self,
        line: int,
        fill: tuple[float, float, float, float],
        gutter: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        text: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        gutter_glyph: str = "",
        tooltip: str = "",
    ) -> None:
        self._lib.ed_add_marker(
            self._h,
            line,
            *fill,
            *gutter,
            *text,
            ord(gutter_glyph) if gutter_glyph else 0,
            tooltip.encode() if tooltip else None,
        )
```

`text` defaults to alpha 0, which the ABI defines as "leave the syntax colours alone", so the hover
mark keeps its current behaviour by taking the default. `gutter_glyph` becomes a real parameter
because the library now DRAWS it (item 5). The old binding hard-coded `0` at the call, which is why
the gutter mark has never been visible.

Two more `Editor` methods, for W-E:

```python
    def set_style(self, style: Style) -> None:
        if not self._lib.ed_set_style(self._h, int(style)):
            raise ValueError(f"unknown keymap style: {style}")

    def get_style(self) -> Style:
        return Style(self._lib.ed_style(self._h))
```

with `class Style(IntEnum): VIM = 0; STANDARD = 1` beside the existing `Mode` / `Language` /
`ChromeFlag` enums. The raise-on-false shape matches `set_palette` and `set_chrome_flag`, which is
the module's convention for a `bool`-returning setter that means "I did not know that value".

And the switch this wave actually flips:

```python
    def set_draw_chrome(self, on: bool) -> None:
        self._lib.ed_set_draw_chrome(self._h, on)
```

### 4. Turn chrome on at session creation, and pass the WHOLE widget to `ed_layout`

`App.get_session` (`app.py:1353` applies `editor_palette()`, `:1361` applies the settings) gains one
call on the fresh handle: `editor.set_draw_chrome(True)`. It belongs there and not in
`_apply_editor_settings_to`, because it is not a setting: there is no UI for it and nothing toggles
it. `ed_draw_chrome` is off on a fresh handle (verified), so every session must be told once.

`tabs/code.py::draw` then changes what it passes to `ed_layout`. Today (`code.py:562-565, 601-603`):

```python
    cell_w, cell_h = editor.get_cell_size()
    gutter_px = (
        editor.get_gutter_cells() * cell_w if settings.show_line_numbers else 0.0
    )
    ...
    editor.layout(
        (float(size_px[0]), float(size_px[1])), px_per_em, origin=(gutter_px, 0.0)
    )
```

The host computed the gutter offset itself because `ed_layout` reserved nothing. With chrome on the
rect is the whole widget, so `origin` goes back to `(0, 0)` and `gutter_px` as a host-computed
quantity disappears:

```python
    editor.layout((float(size_px[0]), float(size_px[1])), px_per_em)
```

`Editor.layout`'s `origin` parameter stays on the method (the ABI still takes it, and the binding
mirrors the ABI), with its docstring corrected: it is the widget's origin in widget space, which for
a host that composites at `(0,0)` is `(0,0)`, and it is no longer where the TEXT starts.

`settings.show_line_numbers` keeps working, through the flag it already drives:
`_apply_editor_settings_to` calls `editor.set_chrome_flag(ChromeFlag.LINE_NUMBERS,
settings.show_line_numbers)` (`app.py:1386`), and with numbers off `ed_gutter_cells` is 0, so the
library reserves no gutter and the text starts at the widget's left edge. That path is unchanged and
untouched; what goes away is the host multiplying that cell count by a cell width.

### 5. What `ed_text_origin` changes, site by site

`ed_text_origin` already exists and is already bound; what changes is that it stops being zero.
Measured on the parked build at a 600x300 widget, 16 px/em, a 20-line buffer: chrome off reports
`(0.0, 0.0)`; chrome on reports `(40.0, 0.0)` with a cell of `10.0 x 21.0`, i.e. four cells of
gutter. So every host site that assumed text-starts-at-rect-origin has to be re-read. There are
exactly six, all in `tabs/code.py`, and the answer is the same for five of them: **nothing changes,
because the library's own hit tests answer against the offset text.** The ABI states it ("Hit tests
and scroll queries answer against the offset text, since they read the same layout") and the
mechanism is that `ed_pixel_to_cursor`, `ed_pixel_over_glyph` and `ed_word_at_pixel` all read
`prev_layout`, which is the layout built at the offset origin.

- **`_handle_mouse` press and drag** (`code.py:451, 453, 465`): `rel = (mouse.x - origin.x, mouse.y
  - origin.y)` where `origin` is `editor_pos`, the widget's top-left. Unchanged, and now
  CORRECT for a click in the gutter too: a click at widget-x 20 is left of the text, and
  `ed_pixel_to_cursor` puts the caret at column 0 of that row rather than at a column derived from a
  negative offset. Before this wave the same click landed on the host's own gutter drawing, outside
  the editor image entirely.
- **The uniform-hover tooltip** (`code.py:671-673`): same `rel`, same conclusion.
  `is_mouse_pos_over_glyph` now answers false over the gutter and over the status row, which is what
  a reader wants, since hovering a line number is not hovering a uniform.
- **The image blit** (`code.py:637-644`): draws the panel texture at `editor_pos` at full widget
  size. Unchanged: the furniture is INSIDE the texture now, so the blit that already covered the
  whole widget covers it.
- **`app.editor_visible_rows`** (`code.py:566`): `int(size_px[1] / cell_h)`. This one is now WRONG
  by one row, because the status row takes a row off the bottom and the value is used to decide
  whether the cursor is off-screen (`code.py:592-599`). Left uncorrected, a cursor on the last text
  row reads as visible while it sits behind the status bar. Fix: subtract the status row when chrome
  is on. The robust form asks the library rather than assuming one row:
  `rows = int((size_px[1] - text_y_bottom_margin) / cell_h)` needs a number the ABI does not export,
  so the wave uses the one it does: the status row is one cell tall by construction (it is a `Frame`
  one `cell.y` high on the widget's bottom edge, `chrome_emit_status`), so
  `app.editor_visible_rows = max(0, int(size_px[1] / cell_h) - 1)` under chrome. Written as a
  subtraction with the reason in the code, since "minus one row for the status line" is not
  recoverable from the expression.
- **`_draw_gutter`** (`code.py:342-373`): deleted whole (item 6). It is the only site that read
  `get_text_origin` and the only one that needed it.

The **completion popup** needs no host change and is worth stating because it is the site a reader
expects to break. The popup is not host furniture: `ed_layout` emits it itself
(`layout_emit_popup`), anchored to a buffer position and flipped above the caret near the bottom
edge, from geometry the host does not have. It has been emitted in the same array as the text since
before this wave, arriving as `Popup_Panel` (kind 5) / `Popup_Glyph` (kind 6). With chrome on it is
computed against the offset text origin like everything else, so it follows the text right by the
gutter width with no host arithmetic. The host's only involvement stays what it is: pushing the
vocabulary in `_drive_completion` before the layout call (`code.py:600`).

`render_state`'s `gutter_px` member becomes the argument that no longer exists. Replace it with
`editor.get_text_origin()`, which moves for the same reasons `gutter_px` did (the line count crosses a
power of ten, the numbers setting toggles) and it is now the library's answer rather than the host's
guess. `tests/test_editor_ffi.py::test_render_state_reacts_to_every_editor_dimension` walks the
domain by keyword and gets the rename with it.

### 6. Deletions

**`tabs/code.py::_draw_gutter`, lines 342-373 (32 lines), and its call site, lines 645-648.** The
call site is the `if settings.show_line_numbers:` block that pushes `app.font_12`, calls
`_draw_gutter(editor, editor_pos, float(size_px[1]))`, and pops. Both go. The setting keeps its
meaning through `ChromeFlag.LINE_NUMBERS` (item 4), so nothing about the checkbox changes. This
deletion is what closes 067's out-of-scope item "imgui-side font of the gutter matching the atlas
font": the numbers are atlas glyphs on the same cell grid as the text, so there is no second font to
mismatch and no `push_font` around them.

**`tabs/code.py::draw_chrome`'s vim half, lines 386-406.** The `session is not None` block: the mode
badge (`_MODE_BADGES` lookup, `text_colored`, `same_line`), the `line:col` readout, and the `:` /
`/` / `?` command line with its message fallback. Everything from `session = app.editor_sessions.get(tab.path)`
through the `imgui.same_line(spacing=float(SPACE.MD))` that closes the message branch. What stays is
the whole `if tab.kind == "shader":` block from line 407 down (file path, `(unsaved)` / `compiled`,
Open dir) and the `else` branch's tab label. The function keeps its name and its two early returns
("No file open", "No document selected"), and `ui.py`'s `_draw_copilot_bar` keeps hosting it with
the Copilot toggle on its right; the bar is not restructured, it just carries less.

**`tabs/code.py::_MODE_BADGES`, lines 24-29,** and the `Mode` import it needs on line 8. With the
badge gone, `Mode` is still used by `_handle_mouse`'s double-click branch (`code.py:455`), so the
import stays and only the dict goes.

The three deletions are what the `conventions.md` re-vendoring entry means by "delete the
mitigations the new sha makes dead". Each was written because the ABI could not emit the thing;
each is now a second derivation of something the library emits.

### 7. The marker call: twelve floats, and which one carries `STATE_ERROR`

`_apply_markers` (`code.py:146-179`) rebuilds the marker list whenever its fingerprint changes. The
error branch today is:

```python
    err_fill = fade(COLOR.STATE_ERROR, 0.35)
    for line, message in fingerprint[0]:
        editor.add_marker(line, err_fill, err_fill, message)
```

That call is CORRECT today, and a reader who counts the positionals wrong will think otherwise:
`add_marker(self, line, fill, gutter, tooltip)` binds `self`, so `message` is the fourth positional
and lands in `tooltip` where it belongs. What has never been visible is the RESULT.
`ed_marker_tooltip` is bound at `ffi.py:225` and called from nowhere in the repo (`grep -rn
"marker_tooltip" shaderbox/ tests/` returns only the `_SIG` entry), and the binding hard-codes the
gutter glyph to `0` at `ffi.py:473`, so no mark was ever drawn to hover in the first place.
`c5c6ae2` draws the glyph itself once chrome is on, which closes the mark half; the tooltip stays
unread by the host, and the error strip remains where an error's text is read.

The new call:

```python
    err_fill = fade(COLOR.STATE_ERROR, 0.20)
    for line, message in fingerprint[0]:
        editor.add_marker(
            line,
            fill=err_fill,
            gutter=COLOR.STATE_ERROR,
            text=COLOR.FG_PRIMARY,
            gutter_glyph="E",
            tooltip=message,
        )
    if hover_line is not None:
        accent = fade(COLOR.ACCENT_PRIMARY, 0.15)
        editor.add_marker(hover_line, fill=accent, gutter=accent)
```

**The text colour is `FG_PRIMARY`, and the number that picks it is a contrast measurement, not an
analogy.** The finding is "red keyword on a red line is unreadable", and the palette measurement
behind it is sharp: `COLOR.STATE_ERROR` and `COLOR.SYN_KEYWORD` are the SAME entry, both
`_P["red_b"]` (`theme.py:167` and `:179`). So a keyword on an error line is drawn in the fill's own
hue, which is the exact unreadable case.

The trap is reasoning from vim's `hi Error` being white ON RED and concluding that the dark ground
colour is the right foreground here. **The marked line is not a red band.** The fill is translucent
by ABI ("The fill is translucent by necessity; it draws behind the text as a `Background`
primitive, and an opaque one hides the code it marks", README § Line markers), so the band the
glyphs actually sit on is the fill composited over `slot.BACKGROUND`, which is `COLOR.BG_SURFACE`
`#161819`. At 0.20 alpha that composite is `#44221e`, a dark brown. Measured WCAG contrast on it:

| foreground | ratio on `#44221e` |
|---|---|
| `COLOR.BG_SURFACE` `#161819` | **1.26** |
| `COLOR.SYN_KEYWORD` `#fb4934` (today's unreadable case) | 4.10 |
| `COLOR.FG_PRIMARY` `#ebdbb2` | **10.27** |
| `_P["fg_0"]` `#fbf1c7` | 12.42 |

The dark ground colour is three times WORSE than the defect it would be fixing, and at the old 0.35
alpha (band `#662922`) it still only reaches 1.62. The transferable half of vim's white-on-red is
"a LIGHT foreground against a dark red ground", and this palette's light foreground is
`FG_PRIMARY`. It is also the colour the rest of the app already means by "readable text"
(`theme.py:140`), so the error line reads as emphasised rather than as a second theme.

The fill drops from 0.35 to 0.20 alpha for a related reason: with the foreground now replaced, the
fill's job is to say "this line" rather than to be the only signal, and a lighter band raises
`FG_PRIMARY`'s contrast (10.27 against 8.02 at 0.35). The gutter mark is `STATE_ERROR` at full
alpha with an `E` glyph, which is what the parent's "the marker's gutter mark (already passed,
never drawn) draws in `STATE_ERROR` in the gutter" asked for, and it is now actually drawn, in the
gutter's separator cell, because the library draws the gutter.

**The `E` is drawn only while `ChromeFlag.LINE_NUMBERS` is on.** `ed_layout` draws a marker's
gutter mark only in a gutter it reserved (README § Line markers: "`ed_layout` draws the gutter mark
only when it draws the gutter"), and with line numbers off `ed_gutter_cells` is 0 and there is no
gutter. So under `show_line_numbers = False` an error line keeps its fill and its replaced text
colour and shows no gutter glyph. That is the intended degradation, not a gap: the two signals that
survive are the two that live on the line itself.

**The red tab tint stays.** `_draw_tab_row` (`code.py:84-88, 100-101`) pushes `COLOR.STATE_ERROR`
into `Col_.tab` / `tab_hovered` / `tab_selected` for a script tab with an error, and that is a
different surface answering a different question ("which of my open tabs is broken"), visible when
the broken tab is not the one on screen. Untouched.

**The parent's "2px left bar on the row" is dropped.** It was the host-side fallback for a library
that could not override the text colour ("Host-side fallback until then", `00_findings.md` #14).
With the override landed, a third error signal on the same row (fill, gutter glyph, and a bar) is
one more than the picture needs, and the bar has no ABI to draw it in now that the host draws
nothing over the editor image.

### 8. The stale-marker dirty fingerprint is not built, because markers are no longer stale

The parent's W-F bullet: "the marker fingerprint includes `is_tab_dirty`; while dirty, no line
markers; the error strip stays with a dim '(stale until save)' suffix". **Do not build it.** Its
whole premise was #15's mechanism: "the lib's marker is a plain buffer-line index ... it does not
shift markers with edits. So between an edit and the next save, marker lines are stale by
construction."

That premise is retired at `c5c6ae2`. Markers anchor to the line's start and move like nvim
extmarks: a line inserted above moves the mark down, a line deleted above moves it up, deleting the
marked line lands it on the line that takes its place, and `ed_set_text` keeps each marker at its
last line. Verified through the parked `.so`: a marker added on line 9, then `O` on line 6 and a
typed character, reads back at line 10.

So the finding's own repro, "error on 9, insert at 7, the code that caused it is now on 10 but the
red stays on 9", now moves the red to 10 with the code. Blanking the markers while dirty would
DELETE a signal that is correct: the user would lose the error highlight for the whole editing
session, which is precisely the interval in which the highlight is useful. And "(stale until save)"
would be a false statement in the UI.

`_apply_markers`'s fingerprint stays `(errors, hover_line)`. It is not a staleness check; it is a
"have my diagnostics changed" check that keeps the wave from re-pushing an identical list every
frame, and that job is unaffected.

One consequence worth naming: the LINE the host holds in `ShaderError` and the line the marker sits
on now diverge as the user edits, and that is correct rather than a bug. The error strip's
click-to-jump (`code.py:217-218`, `JumpRequest(err.path, err.line, 0)`) jumps to the compile-time
line, which after edits may not be the marked one. That is the same behaviour every editor has
between a compile and the next one, and fixing it would mean reading the marker's current line back
through `ed_marker_gutter`, an ABI walk per error per click, to make a jump slightly less wrong
until the next save re-compiles anyway. Not done; stated so it is a decision rather than an
oversight.

### 9. The error strip stays below the editor

The question is whether the strip folds into the status row's message slot now that the row exists.
It stays, and the reason is that they answer different questions with different affordances.

The status row's message slot is the EDITOR's: `ed_command_message` holds what the last ex command
left ("not an editor command", "pattern not found"), it holds one line, it is written by the
library, and there is no ABI to write it from the host. The strip is the HOST's compile
diagnostics: it holds N of them (with an expand toggle past three, `code.py:219-227`), each row is
clickable and jumps the caret to its line (`:215-218`), and it renders script-engine soft errors
adapted to the same shape for a script tab (`_script_errors_for`, 045 decision 7). Folding N
clickable rows into one library-owned string would lose the count, the jump, and the expand, three
affordances the maintainer's walk did not complain about, to save a strip nobody complained about
either. #11 named the mess as "vim symbols on the same line as the file path", and the strip is
neither.

They also never collide visually: the strip is a separate imgui child BELOW the editor image
(`code.py:553` reserves `strip_height` off the editor's height), and the status row is inside the
image. So the panel bottom-to-top reads: error strip, status row, text. That is the same stacking
vim-with-a-quickfix-window has.

### 10. Render dispatch: `render.py` needs no new kind branch, but the redraw gate needs one member

`build_vertices` (`render.py:120-152`) is kind-agnostic by construction: every primitive expands to
the same two triangles, and `kind` is read at exactly one place, to set the `textured` vertex flag
for `GLYPH` and `POPUP_GLYPH` (`:147-150`). `Frame` (kind 4) is solid geometry and draws correctly
as-is; `Popup_Glyph` (kind 6) is already in the textured set because the completion popup uses it.
The array arrives pre-sorted into draw order from `ed_layout`, and `EditorPanel.render` walks it in
order. So the status bar's `Frame` and its `Popup_Glyph` text render with no dispatch change at all.
Verified by layout on the parked build: chrome off emits kinds `{Background: 1, Glyph: 99, Caret: 1}`;
chrome on emits `{Background: 1, Glyph: 114, Caret: 1, Frame: 1, Popup_Glyph: 9}` on the same
buffer. The +15 glyphs are the gutter numbers and filler, the `Frame` is the status bar, the 9
`Popup_Glyph` are its text.

**The redraw gate does need one member.** `render_state` (`render.py:80-117`) lists everything a
drawn frame depends on, and its docstring states the rule: "A member added to the layout's inputs
MUST be added here". The command MESSAGE is now such an input, because it is drawn into the status
row, and no current member moves when it changes. Measured on the parked build: typing `:zzz<CR>`
takes the primitive count's `Popup_Glyph` bucket from 9 to 18 while `get_command_line()` returns to
`None` and the revision, cursor, mode and selection are all unchanged, so the frame that shows
"not an editor command" would never be painted. Add `editor.get_command_message()` beside the
existing `editor.get_command_line()`, and a case to
`test_render_state_reacts_to_every_editor_dimension`.

Two precisions. `Editor.get_command_message` already exists (`ffi.py:632`) and `ed_command_message`
is already bound (`ffi.py:291`), so the change is one member in the `render_state` tuple, not a new
binding. And this is a gap the wave CREATES rather than one it discovers: today the message is drawn
by `draw_chrome` (`code.py:400-405`) as ungated imgui text, repainted every frame, so it is visible
now and becomes invisible the moment decision 6 deletes that branch and the library draws it into
the redraw-gated texture instead. That is what makes the member non-optional rather than a nicety.

The mode is already a member (`editor.get_mode()`), so the badge repaints on a mode switch; the
cursor is already a member, so the ruler repaints on a motion.

### 11. The theme slot mapping is already complete, and one value changes

`theme.py::editor_palette` (`:524-556`) already maps all six slots the furniture draws in. No new
token, no new slot; one existing value is corrected below:

| ABI slot | `Slot` member | `theme.py` token | Drawn as | What it colours |
|---|---|---|---|---|
| 5 `Gutter_Text` | `GUTTER_TEXT` | `COLOR.FG_DIM` | `Glyph` | relative distances on non-cursor rows |
| 6 `Gutter_Current` | `GUTTER_CURRENT` | `COLOR.FG_SECONDARY` | `Glyph` | the cursor row's absolute number, flush left |
| 7 `Filler` | `FILLER` | `COLOR.FG_DIM` | `Glyph` | the `~` rows past the end of the buffer |
| 8 `Status_Bg` | `STATUS_BG` | `COLOR.BG_SURFACE` | `Frame` | the status row's band |
| 9 `Status_Text` | `STATUS_TEXT` | `COLOR.FG_SECONDARY` | `Popup_Glyph` | the ruler and the command message |
| 10 `Status_Accent` | `STATUS_ACCENT` | `COLOR.ACCENT_PRIMARY` | `Popup_Glyph` | the mode badge, the `:` prompt, the block caret |

They have been applied since 067 (`app.py:1353`, `editor.set_palette(editor_palette())` on a fresh
session) against a library that emitted none of them. This wave is what makes them visible; it does
not add them.

**One value changes.** `STATUS_BG` maps to `COLOR.BG_SURFACE` today, which is also
`slot.BACKGROUND` (`theme.py:531`) and the panel clear colour (`code.py:631`), so the status band
would be exactly the same colour as the editor ground and the row would read as text on an
unbroken field rather than as a bar. Vim's statusline is a distinct band, and the maintainer asked
for nvim's look. So `slot.STATUS_BG` becomes `_P["bg_0"]` (`#1d2021`), one step up from `bg_0h`
(`#161819`). The change is one token in `editor_palette`, which is where a colour decision belongs;
no call site names a colour. The manual pass judges whether one step is enough.

## Files touched

- `shaderbox/resources/editor/libeditor.so`: the `c5c6ae2` build, copied from the parked
  directory. Binary; the diff shows a mode-preserving replace.
- `shaderbox/resources/editor/VERSION`: `65264dc4930838483b6ac3ebbcc5774a5f5ddfef` becomes
  `c5c6ae230a51ece592114f349447b1d79d9563ef`.
- `shaderbox/resources/editor/abi_probe.py`: NEW, the editor repo's `ffi/probe.py` at `c5c6ae2`,
  vendored as the argtypes source (item 2c). Data, not an imported module.
- `.pre-commit-config.yaml`: `abi_probe.py` added to the top-level `exclude:` regex, so `ruff --fix`
  and `ruff-format` cannot rewrite a file whose value is being byte-identical to upstream (item 2c).
- `pyproject.toml`: `[tool.pyright]` gains `exclude = ["shaderbox/resources/editor/abi_probe.py"]`.
  The section has no `exclude` key today and `include = ["shaderbox"]` reaches the vendored file,
  which fails pyright with three errors (item 2c).
- `shaderbox/editor/ffi.py`: `_SIG` 65 entries to 93; `ed_add_marker`'s argtypes; `Editor.add_marker`
  gains `text` and `gutter_glyph`; `set_draw_chrome`, `set_style`, `get_style`; the `Style` enum;
  `layout`'s `origin` docstring.
- `shaderbox/editor/render.py`: `render_state`'s `gutter_px` parameter becomes `text_origin`;
  a `command_message` member.
- `shaderbox/tabs/code.py`: `_draw_gutter` and its call deleted; `_MODE_BADGES` deleted;
  `draw_chrome`'s vim half deleted; `_apply_markers`' error call; the `ed_layout` origin; the
  `editor_visible_rows` row count; the `render_state` call's arguments.
- `shaderbox/app.py`: `set_draw_chrome(True)` on a fresh session in `get_session`.
- `shaderbox/theme.py`: `editor_palette`'s `slot.STATUS_BG` becomes `_P["bg_0"]` (item 11). One
  value; W-B does not enter this function, so the two in-flight waves do not collide.
- `tests/test_editor_ffi.py`: the ABI completeness test; the marker-anchoring test; the
  draw-chrome test; the text-origin test; `command_message` in the redraw-domain walk.
- `ai_docs/conventions.md`: the vendored-binary entry's file list corrected to five; the
  re-vendoring entry gains what this re-vendor deleted.
- `ai_docs/features/067_custom_editor.md`: a note where § Out of scope defers the gutter font, and
  where design decision 13 names the vendored sha, saying 069 W-F re-vendored and the host gutter is
  gone.
- `ai_docs/roadmap.md`: the 069 row and the Active-context banner, per the wave-closing habit.

Not touched, and each for a reason a reader might otherwise doubt. **`ai_docs/dev_flow.md`**: the
module map's `editor/` entry (`:269-275`) and its `tabs/code.py` line (`:325`) describe the modules'
ROLES, not their chrome; neither mentions the host gutter, the mode badge or the bottom bar, so
nothing in them goes stale. Named because the docs checklist asks for the module map and a reader
following it would otherwise go looking. **`conventions.md`'s inline-editor entry** (`:362-380`):
its only chrome-adjacent sentence is the vendoring pointer, and its "vim-modal library" phrasing is
assigned to W-E by the parent spec (`01_spec.md § W-E`), so the two waves do not collide in it.
Also untouched: `shaderbox/editor/input.py` (key
translation is unchanged by chrome, since the same `ed_key` codes reach the same keymap), `build.sh` (item 1), `ui.py` (the bottom bar's host is unchanged;
only what `draw_chrome` emits into it shrinks), `popups/settings.py` and `ui_models.py` (the keymap
setting is W-E's).

## Tests

Each states what makes it go red. All six live in `tests/test_editor_ffi.py`, which already loads
the real vendored `.so`, and there is no mock of this ABI anywhere in the repo, by design. The first
two are the mirror-rule gate: one for names, one for types.

1. **`test_the_binding_mirrors_every_export_of_the_vendored_binary`** (item 2b). Reads `nm -D` on
   the vendored binary, compares to `set(_SIG)` in both directions.
   *Falsifier:* delete any one entry from `_SIG` and it names that export in the failure message.
   Verified to be currently red: against the tree's `65264dc` binary it reports 26 unbound exports,
   which is the mirror-rule violation this wave closes. Against the re-vendored `c5c6ae2` binary
   with the full `_SIG` it passes.

2. **`test_the_binding_mirrors_the_upstream_signature_table`** (item 2c). Parses the `_SIG` dict out
   of the vendored `resources/editor/abi_probe.py` by AST and compares every entry's restype and
   argtypes against `editor/ffi.py`'s `_SIG`.
   *Falsifier:* change any one argtype or restype and the assertion names the entry. Verified by
   running the comparison against the tree today: across the 65 entries the two tables already
   share, the only mismatch is `ed_add_marker` (the twelve-float change this wave makes), so the
   gate is red now for exactly the right reason and green once the wave lands. Verified further
   against this spec's own first-draft errors: `"ed_find": (c_int32, [void_p, char_p, c_bool])` is
   reported and passes once corrected to `(c_bool, [void_p, char_p, c_bool, c_bool])`. This is the
   test that catches a WRONG signature, which test 1 structurally cannot: it compares names.

3. **`test_a_marker_follows_a_line_inserted_above_it`** (item 8). Through the real `.so`: build a
   20-line buffer, `add_marker` on line 9, then `set_cursor(6, 0)`, `feed("O")`, `feed("x")` to open
   a line above it, then walk `ed_marker_gutter(line, 0)` from 0 upward and assert the first line
   that answers is 10.
   *Falsifier:* the pre-`c5c6ae2` binary answers 9, which is finding #15's repro verbatim. So this
   test fails on the old binary and passes on the new one, which makes it the pin on the re-vendor
   rather than only on the host code. Measured both ways.

4. **`test_draw_chrome_adds_a_gutter_and_a_status_frame`** (item 10). Lay a buffer out at a fixed
   size with chrome OFF and bucket `prims_list()` by `kind`: no `Kind.FRAME`, and
   `get_text_origin()[0]` is 0.0. Turn `set_draw_chrome(True)` on, lay out again at the same size:
   exactly one `Kind.FRAME`, a non-zero `Kind.POPUP_GLYPH` count, and `get_text_origin()[0]` equal
   to `get_gutter_cells() * cell_w` and strictly positive.
   *Falsifier:* an implementer who binds `ed_set_draw_chrome` but never calls it, or calls it on the
   wrong handle, gets identical buckets and both the `FRAME` and the origin assertions fail.
   **A GLYPH-count comparison is NOT a valid falsifier and is deliberately absent:** the gutter takes
   four cells off the left, so the text viewport narrows and glyphs past the new right edge stop
   being emitted. On a wide buffer chrome-on emits FEWER glyphs, measured at the same 600x300 /
   16 px-per-em geometry: twenty 60-character lines go from 840 glyphs off to 757 on, while twenty
   lines of `abc` go from 45 to 58. A test asserting "more glyphs under chrome" passes on a short
   buffer and fails on a realistic shader line width. The `FRAME`, `POPUP_GLYPH` and origin
   assertions are the ones that actually pin the switch, and none of them is buffer-dependent.

5. **`test_text_origin_moves_right_by_the_gutter_under_chrome`** (item 5). Chrome off, after a
   layout, `get_text_origin()` is `(0.0, 0.0)`. Chrome on, after a layout on the same buffer,
   `x == editor.get_gutter_cells() * cell_w` and is strictly positive.
   *Falsifier:* a host that keeps passing `origin=(gutter_px, 0)` into `ed_layout` double-offsets,
   and the identity against `get_gutter_cells()` breaks. Measured: 40.0 at 4 gutter cells of 10.0px.

6. **`test_render_state_reacts_to_every_editor_dimension`** (extended, item 10). The existing domain
   walk gains a `command_message` case: run a failing ex command (`:zzz` then Enter), assert the
   tuple moved.
   *Falsifier:* omit the `get_command_message()` member from `render_state` and the case fails.
   Measured that it is currently a real gap: the failing command changes the drawn primitives (9
   `Popup_Glyph` to 18) while every existing member holds still.

The existing suite covers the rest of the surface unchanged: nothing in it asserts a host gutter or
a bottom-bar badge, so no test is deleted with the code. `make gates` is the exit-code judge, per
the hard rule, captured unpiped.

## Manual verification

The parent's W-F line, rewritten for lib-drawn furniture. The maintainer, in the app:

- **The gutter is nvim's `number relativenumber` picture.** Open a shader tab, put the caret
  mid-file: the cursor row shows its ABSOLUTE number, flush LEFT in the gutter; every other visible
  row shows its DISTANCE from the caret, right-aligned; move `j` and every number changes but the
  cursor row's. Past the end of a short buffer the rows show `~`. The numbers are in the editor's
  font on the editor's grid, not the app UI font.
- **The status row is inside the editor rect.** At the bottom of the editor image, above the error
  strip: the mode badge on the left, the `line,col` ruler on the right. Type `:` and the badge and
  ruler give way to the command line with a block caret. Type garbage and Enter, and the row shows "not
  an editor command" and repaints when it does. `/foo` shows the `/` prompt and the pattern.
- **The bottom bar carries host things only.** File path, `(unsaved)` or `compiled`, Open dir,
  Copilot. No NORMAL, no `1:1`, no `:` line.
- **An error line is readable.** Break a shader with a keyword on the failing line (`vec3` misspelt,
  say). The line takes a light red wash, the glyphs on it turn to the dark ground colour, and every
  word on the line is legible, with no red-on-red. **With line numbers on**, an `E` shows in the
  gutter's separator cell on that row; with the setting off there is no gutter and no `E`, and the
  fill plus the replaced text colour are the whole signal. The tab tints red as it always has, and the error strip lists the error and still jumps on a
  click.
- **The mark follows an `O` above it.** With an error on some line, put the caret a few lines above
  it and press `O`, type something, leave insert. The red band moves DOWN with the code, in the same
  frame, without saving. Delete a line above it and the band moves back up. This is the finding-#15
  repro and it should now come out the other way.
- **Visual `p` replaces.** Yank a word (`yiw`), select a different span (`viw`), press `p`: the
  selection is REPLACED by the register, not pasted on top of it, in one undo step. Press `u` once
  and the whole thing reverts.
- **Standard style, briefly** (W-E ships the setting; here it is a spot check that the wave did not
  break the other style): nothing to do from the UI yet, so this one is covered by the test suite
  only, and the maintainer sees it at W-E.

## Verified / corrected premises

Every claim the task statement carried, checked against `ffi.odin` at `c5c6ae2`, `nm -D` on the
parked `.so`, and a ctypes probe of the parked build; rows 36-44 were added by the round-1 review
and its re-verification, rows 45 and 46 while folding it and at round-2 closure. **22 of 46 are
corrected, refuted or superseded, eight of them this spec's own claims (marked "self-correction");
the other 24 are confirmed.**

| # | Premise | Verdict | Evidence |
|---|---|---|---|
| 1 | `ed_set_style(h, int32) -> bool` and `ed_style(h) -> int32`, 0 Vim / 1 Standard, are ADDED since the binding was written | **CORRECTED**. They exist, but they are not new. Both are present at the currently vendored `65264dc` (`ffi.odin:894, 906`) and in its `.so`. What is true is that they are UNBOUND in `editor/ffi.py`. The semantics are as stated: probe returns 0 on a fresh handle, `ed_set_style(h,1)` returns true and `ed_style` then reads 1 | `git show 65264dc:ffi/ffi.odin`; `nm -D` diff of the two binaries |
| 2 | `ed_set_draw_chrome(h, bool)` and `ed_draw_chrome(h) -> bool` are ADDED, off on a fresh handle | **CONFIRMED**, and these two are the ONLY exports new in `c5c6ae2`. 91 exports at `65264dc`, 93 at `c5c6ae2`, none removed. Probe: `ed_draw_chrome` is false on a fresh handle | `nm -D` both binaries, `comm`; probe |
| 3 | With chrome on, the rect passed to `ed_layout` is the WHOLE widget; gutter off the left, status row off the bottom; `ed_text_origin` reports where the text starts | **CONFIRMED** | `ffi.odin:379-387` (`chrome_text_area`); probe: `text_origin` 0.0 off, 40.0 on, at a 10.0px cell and 4 gutter cells |
| 4 | Gutter (relative numbers, cursor line absolute flush-left, `~` filler, marker glyphs in the separator cell) and status row arrive in the same array as the text, in draw order | **CONFIRMED** | `ffi.odin:441-450` (`chrome_emit_gutter` / `chrome_emit_status` under `if s.draw_chrome`), `:452-467` (the stable sort into draw order); README § Chrome |
| 5 | Numbers/filler/marks are `Glyph`, the status bar is a `Frame`, its text is `Popup_Glyph` | **CONFIRMED** | README § Chrome; probe bucket `{0:1, 2:114, 3:1, 4:1, 6:9}` chrome-on against `{0:1, 2:99, 3:1}` chrome-off |
| 6 | Colours come from `Gutter_Text`, `Gutter_Current`, `Filler`, `Status_Bg`, `Status_Text`, `Status_Accent` | **CONFIRMED**, and additionally **all six are already mapped** in `theme.py::editor_palette` and already applied per session at `app.py:1353`. One value is corrected (`STATUS_BG`, item 11); no slot is added | README § Theme (slots 5-10); `theme.py:538-543` |
| 7 | Hit tests answer against the offset text | **CONFIRMED** | README § Chrome. `ffi.odin`: `ed_pixel_to_cursor` / `ed_pixel_over_glyph` / `ed_word_at_pixel` all read `prev_layout`, which is built at the offset origin |
| 8 | `ed_add_marker` gains four floats between gutter colour and glyph: `(h, line, fill rgba, gutter rgba, TEXT rgba, gutter_glyph, tooltip)` | **CONFIRMED**, and it is the only changed signature in the delta | `ffi.odin:601-624` at `c5c6ae2` vs the same proc at `65264dc` (8 floats) |
| 9 | Text alpha 0 keeps syntax colours | **CONFIRMED** | README § Line markers: "The text colour, when its alpha is non-zero, replaces the syntax colour of every glyph on the line ... pass alpha 0 to leave the syntax colours alone" |
| 10 | Markers anchor to the line's start and move like nvim extmarks; `ed_set_text` keeps each marker at its last line | **CONFIRMED**, empirically as well as in prose | README § Line markers; probe: marker on line 9, `O` above at line 6 plus a typed char, reads back at line 10. `ffi.odin:270, 272`: `marker_store_detach` / `marker_store_attach` around the rebuild in `ed_set_text` |
| 11 | So the parent's "no markers while dirty / stale until save" mechanism is not needed | **CONFIRMED**, and this wave drops it (§ Design decisions item 8) | The probe above is the finding's own repro coming out right |
| 12 | `ed_set_cursor` no longer clamps while a visual selection is live | **CONFIRMED** | `ffi.odin:320-326`: the `Visual` / `Visual_Line` branch assigns `p` unclamped |
| 13 | `ed_clear_selection` under the standard style ends the selection and closes the popup | **CONFIRMED** | `ffi.odin` `ed_clear_selection`: under `.Standard` it calls `ffi_host_edit(e)` and returns, instead of feeding Escape |
| 14 | `ed_set_text` re-establishes the carried style | **CONFIRMED** | `ffi.odin:275-286`: `editor_carry_put` then the `.Standard` re-clamp |
| 15 | The parked build is `c5c6ae2` and matches its source | **CONFIRMED** | `VERSION` in the parked dir reads `c5c6ae230a51ece592114f349447b1d79d9563ef`; the 93 `ed_*` symbols from `nm -D` are identical to the 93 `@(export) ed_*` procs in `ffi.odin` at that sha (`diff` clean) |
| 16 | The atlas is unchanged | **CONFIRMED** | `md5sum` identical for `atlas.png` on both sides |
| 17 | The wave must copy the parked files and update `VERSION` | **CORRECTED**. SIX files are vendored today, not five, and only two of them change: `atlas.png`, `atlas.json`, `vim_coverage.md` and `standard_keymap.md` are already vendored and byte-identical. The wave adds a seventh, `abi_probe.py` | `git ls-files shaderbox/resources/editor/` returns six; `diff` clean on `atlas.json` and both `.md` files; `md5sum` on the PNG |
| 18 | W-E's audit reads `resources/editor/vim_coverage.md` and `standard_keymap.md`, which W-F copies under `resources/editor/` (parent `01_spec.md § W-E`: "they are not in this repo until W-F copies them") | **REFUTED**. Both files are already tracked in the repo and already current at `c5c6ae2`. W-E's inputs do not depend on this wave's copy step | `git ls-files`; `diff` against the parked copies |
| 19 | `editor/ffi.py` MIRRORS the ABI completely (`conventions.md ## Known quirks`) | **REFUTED as written**. The rule is stated but not held and not checked: 65 of 91 exports bound at the vendored sha, 26 unbound. The wave closes the gap and ships the check (§ Design decisions item 2) | `nm -D` vs the `_SIG` keys |
| 20 | A test or script pins the ABI mirror | **REFUTED**. None exists. `tests/test_editor_ffi.py` has 60+ tests, none of them about export coverage; `scripts/` has no ABI check; nothing greps `nm` | grep over `tests/` and `scripts/` |
| 21 | `render.py` must learn to dispatch `Frame` and `Popup_Glyph` | **CORRECTED**. `build_vertices` is kind-agnostic; `kind` is read at one place, to set the texture flag, and `POPUP_GLYPH` is already in that set. No dispatch change | `render.py:120-152` |
| 22 | The parent's W-F bullet "bind `ed_set_style`, `ed_style`, `ed_filler_glyph`" names the whole binding job | **CORRECTED**. Those three are a subset. Under the mirror rule the wave binds all 28 unbound-or-new exports | `nm -D` vs `_SIG` |
| 23 | `_draw_gutter` is the host's line-number drawing to delete | **CONFIRMED**, at `tabs/code.py:342-373`, called at `:645-648` under `if settings.show_line_numbers:` with a `push_font(app.font_12)` around it | read |
| 24 | The bottom bar's mode badge is deleted, keeping path / compiled / Open dir / Copilot | **CONFIRMED** as the shape; the badge, ruler and command line are `tabs/code.py:386-406` inside `draw_chrome`, and `_MODE_BADGES` at `:24-29` goes with them. Copilot is `ui.py`'s, not this function's, and is untouched | read |
| 25 | The parent's error-line fix is "gutter mark in `STATE_ERROR` + a 2px left bar, no whole-line fill" | **SUPERSEDED**. With the text-colour override landed, the fill stays (lighter) and the bar is dropped (§ Design decisions item 7). The parent wrote it as an explicit fallback "until then" | `00_findings.md` #14 |
| 26 | `STATE_ERROR` vs syntax colours is the readability problem | **CONFIRMED, and sharper than the finding states**. `COLOR.STATE_ERROR` and `COLOR.SYN_KEYWORD` are the SAME palette entry, `_P["red_b"]`. So a keyword on an error line is exactly invisible | `theme.py:167, 179` |
| 27 | Finding #16 (visual `p`) is fixed at `c5c6ae2` and needs no ShaderBox code | **CONFIRMED** | the vendored `vim_coverage.md:293, 313` documents visual `p`/`P` replacing the selection, with `p` taking the replaced text into the register |
| 28 | The parent's Out-of-scope item "a standard-keymap gutter/status design of its own" stands | **CONFIRMED**, and the standard style does emit its own furniture under the same switch: probe on the same buffer gives `{0:1, 2:121, 3:1, 4:1, 6:3}`, a status `Frame` with a three-glyph readout | probe after `ed_set_style(h, 1)` |
| 29 | The completion popup needs host repositioning under the new origin | **REFUTED**. `ed_layout` emits the popup itself (`layout_emit_popup`, `ffi.odin:424-431`), anchored to a buffer position and flipped near the bottom edge, from the same layout the offset produced | `ffi.odin`; README § Drawing |
| 30 | `render_state` covers every drawn input | **CORRECTED**. It does not cover the command MESSAGE, which the lib-drawn status row now draws. Measured: `:zzz<CR>` takes the `Popup_Glyph` bucket 9 to 18 with every existing member unchanged | probe; `render.py:80-117` |
| 31 | `app.editor_visible_rows` stays correct under chrome | **CORRECTED**. `int(size_px[1] / cell_h)` (`code.py:566`) over-counts by the status row, and the value gates the cursor-follow at `:592-599` | read + the one-cell status row in `chrome_emit_status` |
| 32 | `build.sh`'s vendored-file allowlist needs updating | **REFUTED**. There is no allowlist; `build.sh` handles the editor by name (`libeditor.so`) at `:48-51` and `:72`, and a changed file COUNT was never its input | read |
| 33 | The current `add_marker` call mis-positions the tooltip (this spec's own first-draft claim) | **REFUTED, self-correction.** `add_marker(self, line, fill, gutter, tooltip)` binds `self`, so `code.py:175`'s `message` is the FOURTH positional and lands in `tooltip`. The call has always been correct; the first draft miscounted by one. The real defect nearby: `ed_marker_tooltip` is bound (`ffi.py:225`) and called from nowhere in the repo, and the binding hard-codes the gutter glyph to `0` (`ffi.py:473`), so no mark was ever drawn to hover | AST-extracted params of `ffi.py:465-474`; `grep -rn "marker_tooltip" shaderbox/ tests/` returns only the `_SIG` entry |
| 34 | 067 § 13 is the vendoring procedure to follow | **CONFIRMED** for the shape (build from a committed sha, copy, update `VERSION`), and it defers the detail to `conventions.md ## Known quirks`, which is the entry this wave edits | `067_custom_editor.md:157-165` |
| 35 | 067's out-of-scope item on the gutter font is still open | **CONFIRMED as open, and closed by this wave**. "imgui-side font of the gutter matching the atlas font: gutter numbers render in the app UI font. Revisit if misalignment reads badly" (`067_custom_editor.md:31-32`). The lib-drawn gutter is atlas-drawn | read |
| 36 | The five signatures `ed_replace_at_cursor` / `ed_replace_all` / `ed_find` / `ed_find_count` / `ed_paste` as this spec first listed them | **CORRECTED, self-correction (review finding 2).** All five were short an argument or wrong in restype against `ffi.odin` at `c5c6ae2`: `ed_paste` is `(h, before: bool, count: i32) -> bool`, `ed_find` is `(h, pattern, backward, ignore_case) -> bool`, `ed_find_count` is `(h, pattern, ignore_case) -> i32`, and both replace procs take `(h, pattern, replacement, ignore_case)`. Corrected in item 3, and the class is now gated by item 2c | `git show c5c6ae2:ffi/ffi.odin` at `:1218, :1238, :1263, :1279, :1300`; upstream `ffi/probe.py` agrees on all five |
| 37 | A machine-parseable per-export signature source exists upstream | **CONFIRMED, and it is `ffi/probe.py`**, the editor repo's own reference ctypes binding. Its `_SIG` holds **93 entries whose key set is byte-equal to `nm -D`'s 93 `ed_*` symbols** (both set differences empty), in the same `{name: (restype, argtypes)}` shape as ours. No C header exists in the repo and no `.h` is generated, so this is the artifact; no `scripts/` Odin parser is needed | `git ls-tree -r c5c6ae2 ffi/` lists exactly `README.md`, `ffi.odin`, `probe.py`; AST parse of `probe.py`'s `_SIG` against the `nm` set |
| 38 | The argtypes gate compares cleanly through ctypes | **CONFIRMED by running it.** `ctypes.POINTER(c_float) is ctypes.POINTER(c_float)` is True (ctypes memoizes pointer types), both tables parse by AST (93 upstream, 65 ours), and across the 65 entries they share the ONLY mismatch is `ed_add_marker` -- exactly the signature this wave changes. So the gate is red today for the right reason and green when the wave lands | executed against the tree and the parked build |
| 39 | Test 3's "more glyphs under chrome" is a valid falsifier (this spec's own first-draft claim) | **REFUTED, self-correction (review finding 4).** The gutter narrows the text viewport, so a wide buffer emits FEWER glyphs under chrome: twenty 60-character lines go 840 to 757 at 600x300 / 16 px-per-em, while twenty lines of `abc` go 45 to 58. The assertion is dropped for `FRAME` / `POPUP_GLYPH` / text-origin, none of which is buffer-dependent | measured on the parked build at the spec's own geometry |
| 40 | `BG_SURFACE` is the right marker text colour (this spec's own first-draft claim) | **REFUTED, self-correction (review finding 1).** The marker fill is translucent by ABI, so the band is the fill composited over `BG_SURFACE`, which at 0.20 alpha is `#44221e`. `BG_SURFACE` on that band measures **1.26:1**, three times WORSE than the 4.10:1 red-on-red it was meant to fix; `FG_PRIMARY` measures 10.27:1. The choice is now `FG_PRIMARY` | WCAG contrast computed over the composite; `theme.py:59-73, 126, 140, 167, 179` |
| 41 | A dropped export "crashes in the UI rather than a red gate" (this spec's own first-draft claim) | **CORRECTED, self-correction (review finding 10).** `ensure_loaded` (`ffi.py:307`) `getattr`s every `_SIG` name on the first `Editor` construction, and every test reaches it, so a dropped export already fails the suite loudly. The bound-but-absent assertion adds the NAME in one message, not the failure itself | `ffi.py:307-317`; the test suite's load path |
| 42 | `ed_set_style` preserves the host's chrome flags | **REFUTED (review finding 8).** It replaces the whole `Chrome` from `chrome_for(style)`, so `LINE_NUMBERS`, `RELATIVE_NUMBERS` and the status flags reset to the style's defaults; measured, with `LINE_NUMBERS` False, `ed_set_style(h, 0)` leaves it True. `ed_draw_chrome` is not part of `Chrome` and survives. Recorded as an ordering constraint for W-E | `ffi.odin:940`; probe |
| 43 | The marker's `E` glyph is drawn regardless of the line-numbers setting | **REFUTED (review finding 7).** `ed_layout` draws a marker's gutter mark only in a gutter it reserved, so with `LINE_NUMBERS` off `ed_gutter_cells` is 0 and no mark is drawn; measured, the marker adds 17 glyphs with numbers on and 0 with them off | README § Line markers; probe |
| 44 | `dev_flow.md`'s module map goes stale with this wave | **REFUTED (review finding 9).** The `editor/` entry (`:269-275`) and the `tabs/code.py` line (`:325`) describe module roles and mention neither the host gutter, the mode badge nor the bottom bar. No edit | read |
| 45 | Vendoring `ffi/probe.py` as a `.py` under `shaderbox/resources/` is inert | **REFUTED, found while folding review finding 2.** `ruff` reports six errors on it, and `.pre-commit-config.yaml` runs `ruff` with `--fix` plus `ruff-format`, so `make check` would silently rewrite the file the argtypes gate compares against. The wave adds it to the config's existing top-level `exclude:`, which already carries the same precedent for `resources/emoji/emoji-test.txt` | `uv run ruff check` on the file; `.pre-commit-config.yaml:1, 11-17` |
| 46 | `pyright` needs no exclusion for the vendored probe (this spec's own round-1 wording, which left it conditional) | **CORRECTED, self-correction (round-2 residual).** `[tool.pyright]` has `include = ["shaderbox"]` and **no `exclude` key**, so the file is reached, and pyright reports **three** errors on it (`reportOptionalIterable` at `:382`, `reportAssignmentType` tuple-size mismatch at `:1391`). The conditional becomes a decided step: `pyproject.toml` gains `exclude = ["shaderbox/resources/editor/abi_probe.py"]`. Verified that adding the key leaves pyright clean on the real tree | `uv run pyright` on the file staged at its vendored path; `pyproject.toml:87-92`; a full `uv run pyright` run with the key added, then reverted |

## Open questions

Each carries a robust default, marked. None blocks implementation.

1. **CLOSED: the status band gets its own grey.** `STATUS_BG` maps to `COLOR.BG_SURFACE`, which is
   also the editor's `BACKGROUND` slot (`theme.py:531, 541`) and the panel clear colour
   (`code.py:631`), so the band is provably invisible before anyone looks at it. That is a known
   defect, not something to discover at the manual pass. **Ruling:** `editor_palette`'s
   `slot.STATUS_BG` becomes `_P["bg_0"]` (`#1d2021`), one step up from `bg_0h` (`#161819`), and the
   manual pass corrects it if the step is too small. The maintainer asked for nvim's look, and
   nvim's statusline is a distinct band. This is the wave's one `theme.py` edit.

2. **Should `set_draw_chrome` become a Setting?** A user might want the editor bare. **Default
   (implement this):** no, it is on unconditionally at session creation. D6 says the furniture is
   the editor's, and `show_line_numbers` already exists for the one half anyone asks to turn off.
   Trigger to revisit: a request to hide the status row specifically.

3. **Does the `E` gutter glyph belong to a warning tier later?** The ABI takes one codepoint per
   marker and the host has exactly one diagnostic class today. **Default (implement this):** `E` for
   every compile error, no glyph on the hover mark. A second class (a warning) would pick its own
   glyph at the site that creates it, with no ABI question to answer.

4. **Should the error strip's jump follow the marker rather than the compile line?** After edits the
   two diverge (§ Design decisions item 8). **Default (implement this):** jump to the compile line,
   as today. Reading the live marker line back costs an ABI walk per error per click to make a jump
   slightly less wrong until the next save re-compiles anyway.

## Review history

**Round 1, pre-implementation reviewer** (`reviews/wave_f_pre.md`), one run, on opus: judgement
against the ABI and the code was the deliverable. Verdicts: parent coverage PASS, host-origin
coverage PASS, ABI table accuracy **FAIL**, test falsifiability PARTIAL, docs PARTIAL. Ten findings,
**all ten accepted and folded**; the reviewer anchored to artifacts this spec did not author (the
editor repo at `c5c6ae2`, `nm -D`, six ctypes probes, a WCAG computation) and re-derived every
number rather than restating the spec's.

Two were correctness defects that would have shipped a visibly wrong result, and both were the
spec's own reasoning rather than a missing citation:

- **Finding 1, the marker text colour.** The spec chose `BG_SURFACE` by analogy to vim's
  white-on-red. The marker fill is translucent by ABI, so the band is a dark brown composite and
  `BG_SURFACE` on it measures 1.26:1, three times worse than the 4.10:1 red-on-red it was fixing.
  Ruling: `COLOR.FG_PRIMARY` (10.27:1), with the measured composite and the four-way contrast table
  as the justification in decision 7. Re-derived independently before folding; the numbers
  reproduce.
- **Finding 2, five wrong ctypes signatures.** `ed_replace_at_cursor`, `ed_replace_all`, `ed_find`,
  `ed_find_count` and `ed_paste` were each short an argument or wrong in restype, which is the
  silent-corruption class rather than a `TypeError`. All five corrected in item 3.

**The ruling on finding 2 closed the class, not the instance.** The name-only completeness test
passed on every one of the five, so the wave now also gates argtypes. What `~/src/editor` ships
under `ffi/` at `c5c6ae2` is exactly three files -- `README.md`, `ffi.odin`, `probe.py` -- and no C
header anywhere in the repo. `ffi/probe.py` is the editor's own reference ctypes binding and carries
a module-level `_SIG` in the same `{name: (restype, argtypes)}` shape as ours, with 93 entries whose
key set is byte-equal to `nm -D`'s 93 symbols. So the wave vendors it as
`resources/editor/abi_probe.py` and the test parses it (item 2c); no generated `abi.json` and no
`scripts/` Odin parser is needed, because the authoritative table already exists and the side that
owns the ABI maintains it. Verified by running the comparison: the only mismatch across the 65
entries both tables share is `ed_add_marker`, the one signature this wave changes.

**Finding 3 refuted a bug this spec had invented.** The first draft claimed `code.py:175`
mis-positioned the tooltip; it counts four positionals after `self` binds and has always been
correct. The real defect in the same neighbourhood replaced it: `ed_marker_tooltip` is bound and
called from nowhere, and the binding hard-codes the gutter glyph to `0`. Premise 33 now records the
self-correction.

The remaining seven landed as written: test 3's buffer-dependent GLYPH assertion dropped for
falsifiers that are not buffer-dependent (4); the `render_state` gap restated as one the wave
CREATES by deleting the ungated imgui draw (5); six vendored files, not five, and `conventions.md`
under-counts in two sentences (6); the `E` glyph's dependence on `LINE_NUMBERS`, with the manual
bullet qualified (7); `ed_set_style` clobbering the whole `Chrome`, recorded here and in
`02_keybindings.md` and in `50_wave_e_keyboard.md`, whose apply block already orders it correctly
and now carries the reason (8); `dev_flow.md`'s module map needs
no edit and the spec now says so rather than being silent (9); and `ensure_loaded` already failing
loudly, so the bound-but-absent assertion adds a name and not the failure (10).

**Open question 1 closed by ruling** rather than deferred to the manual pass: `STATUS_BG` and
`BACKGROUND` are both `COLOR.BG_SURFACE`, so the band is provably invisible before anyone looks.
`slot.STATUS_BG` becomes `_P["bg_0"]`, which makes `theme.py` a touched file. The reviewer agreed
with the defaults on the other three.

**One defect found while folding, not by the reviewer.** Vendoring `ffi/probe.py` as a `.py` under
`shaderbox/resources/` is not inert: `ruff` reports six errors on it and the pre-commit config runs
`ruff --fix` plus `ruff-format`, so `make check` would silently rewrite the very file the argtypes
gate compares against. The wave adds it to the config's existing top-level `exclude:`, which already
carries the same precedent for a vendored data file. Recorded as premise 45.

Nothing was rejected. No finding escalated to "should not land".

**Round 2, closure: PASS, all ten closed.** One residual, on the round-1 fix rather than on the
original spec: decision 2c had left pyright as a conditional ("confirms pyright is clean on it or
adds it to pyright's `exclude`"), which is the shape that reaches an implementer as a question
rather than a step. Measured: `[tool.pyright]` carries `include = ["shaderbox"]` and no `exclude`
key at all, and pyright reports three errors on the vendored file. So the conditional is now a
decided step, `pyproject.toml` joins § Files touched, and premise 46 records the self-correction.
Verified before folding that adding the key leaves a full `pyright` run clean on the tree, so the
new key costs nothing elsewhere. Also stated in 2c: **W-F owns `Editor.set_style` / `get_style` and
the style enum**, named once in `editor/ffi.py`; W-E only calls them, and W-F's naming is the
definition if the two specs ever drift.
