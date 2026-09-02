# 067 — Custom editor integration (libeditor)

Replace the imgui `TextEditor` (imgui_color_text_edit) with the maintainer's own editor
(`~/src/editor` — headless Odin core, C ABI, MTSDF-atlas primitive renderer). This is the
shaderbox half of the editor repo's feature 004 (two-repo feature; findings flow to that
session, fixes land there and get re-vendored here).

## Goal

The code panel (shader / script / lib tabs) is driven by `libeditor.so` instead of
`TextEditor`: vim-modal keymap, host-rendered MTSDF text, all existing editor-adjacent
features preserved (tabs, dirty dots, error markers + strip, jump-to-error, uniform hover
tooltips, copilot read-only lock, hot-reload sync, lib-picker insert-at-caret, settings).

**Product decision (locked with the maintainer): the editor integrates AS-IS, vim-modal —
the deliberate worst case, outlining the integration's upper bound. A non-modal keymap is
deferred editor-side work; a finding asking for one is out of scope.** The one hard
requirement that follows: modal bindings must not collide with the global hotkey set.

## Out of scope

- **Non-modal / classical keymap** — deferred. Trigger: maintainer decides post-dogfood.
- **Windows `libeditor.dll`** — the vendored binary is linux-x86_64 only for now.
  Trigger: next `/ship`. NOT deferred: `build.sh` must already exclude the `.so` from the
  Windows stage (see Files touched) — a linux ELF silently riding the Windows zip is a
  build defect from the day the binary is committed.
- *(resolved during the wave)* Vim registers were an editor-side product question; the
  maintainer decided ONE unnamed register (editor 4befeaf) and D8 now unifies it with the
  OS clipboard. Named registers / the numbered ring stay editor-side out of scope.
- **Caret blink** — steady caret; revisit only if it reads badly in dogfood.
- **imgui-side font of the gutter matching the atlas font** — gutter numbers render in the
  app UI font. CLOSED by 069 W-F: the gutter is drawn by `ed_layout` itself, in atlas glyphs
  on the text's own cell grid, so there is no second font to mismatch.

## Design decisions

1. **Binding: a repo-owned ctypes module, `shaderbox/editor/ffi.py`.** Start from the editor
   repo's `examples/python/editor_widget.py` (written to mirror `EditorSession`'s needs
   method-for-method), adapted to repo conventions: no `from __future__`, full annotations,
   free function instead of the `@staticmethod language_for_path`. Two inherited choices
   made explicit: the loaded library lives in a module-global (`_LIB`), populated by an
   idempotent `ensure_loaded()` called from `Editor.__init__` (lazy — no import-time side
   effect); the 1 MiB text scratch buffer is ONE module-level buffer shared by all
   instances (single-threaded frame loop), not per-Editor. Method names keep the
   TextEditor-mirroring shape (`get_text`, `set_text`, `get_undo_index` = `ed_revision`,
   `replace_text_in_current_cursor` = `ed_insert_at_cursor`-over-selection, `add_marker`,
   `scroll_to_line`, `is_mouse_pos_over_glyph`, `get_word_at_mouse_pos`) so `app.py` /
   `watch.py` / `widgets/uniform.py` call sites survive nearly unchanged.
2. **One `Editor` handle per `EditorSession`, keyed by path as today.** The handle owns
   buffer + mode + undo, so tab switches preserve mode and history. `EditorSession.editor`
   changes type; `saved_undo` stays "store at save, compare by equality". BEHAVIOR FLIP to
   respect: `ed_revision` RISES across `ed_set_text` (ABI contract; `TextEditor`'s undo
   index did not), so every `set_text` + re-baseline pair must read the revision AFTER the
   set — `watch.py` and `app.py::sync_editor_from_disk`/`get_session` already do; do not
   "tidy" the order. `tests/test_pass_editor_wiring.py::_edit`'s `saved_undo = idx - 1`
   fudge inverts under this and gets rewritten (set_text alone now reads dirty).
3. **Rendering: moderngl offscreen pass + `imgui.image`.** New `shaderbox/editor/render.py`:
   one shared atlas GL texture + MTSDF shader program (median(r,g,b) distance decode,
   screen-space AA from `ed_atlas_distance_range`), one FBO per panel resized to the
   content region. Per frame: `ed_layout(0, 0, w, h, px_per_em=font_size, wrap=False)`;
   primitives read in ONE call via `ed_primitives(h, out, cap)` (bulk getter, editor
   commit e55b997) into a preallocated buffer, then batched into a single interleaved
   VBO (solid quads carry a zero-area UV flag) drawn in array order — the array arrives
   draw-ordered. Present via
   `imgui.image(ImTextureRef(tex.glo))` inside the existing editor child window.
   **Redraw gate, structured for the domain test:** `render_state(...) -> tuple` (the
   session's PATH — tabs share one panel, so identity must move the tuple — plus
   revision, scroll, cursor, mode, selection, command-line text, size, px_per_em,
   gutter_px, marker fingerprint, settings fingerprint, focus) and
   `should_redraw(prev, cur) -> bool` are free
   functions; the render path re-reads primitives + re-renders ONLY when the tuple
   changed, and increments `App.editor_redraw_count` PAST the gate (counts decisions, not
   calls). `ed_layout` itself runs every visible frame so hit-testing and scroll clamping
   stay current.
4. **Input: glfw callback queue, drained in-frame before command dispatch.** `App` chains
   glfw key + char callbacks (precedent: the Esc filter) into a per-frame event queue.
   `shaderbox/editor/input.py` translates: char event → `ed_key(CHAR, mods, codepoint)`;
   key press/repeat of specials → the ABI code (Esc, Enter, Tab, Backspace, Delete,
   arrows, Home, End, PgUp/PgDn); key press of a printable WITH Ctrl/Alt/Super held →
   synthesized `ed_key(CHAR, mods, lowercased char)` (the platform emits no char event for
   those; Ctrl+R crosses as `text='r'` + mod bit, verified against `keymap_normal.odin`).
   The drain runs at the top of `dispatch_commands`, ONLY while `app.editor_focused` — an
   unfocused editor receives no events (the bare-`d`-is-an-edit hazard). **Gate staleness,
   stated:** `editor_focused` at drain time is last frame's value (written after the
   editor draw). Newly-focused-for-one-frame-deaf is the safe direction; the focus-
   transition frames toward the chat are a known ~1-2 frame keystroke misroute
   (sweep-verified, accepted — the character visibly fails to appear in the chat).
5. **Hotkey arbitration: editor first, registry skips consumed chords.** The drain records
   each consumed keypress that carries Ctrl/Alt/Super AS AN IMGUI KEYCHORD INT
   (translated at the record site: glfw key+mods → `int(imgui.Key.x) | mod bits` — the
   registry's comparison space) into `App.editor_consumed_chords: set[int]`, cleared at
   the top of each drain; `_dispatch_registry` skips a spec whose effective chord is in
   the set. **This skip is the ONLY collision guard: `OPEN_SCRIPT` (Ctrl+R) and
   `NEW_DOCUMENT` (Ctrl+N) are `CommandScope.GLOBAL` — without the skip, a focused-editor
   Ctrl+R would BOTH redo and open the script tab on one press.** Editor-side the chord
   domain is EIGHT as of editor 4b110f0 (Ctrl+R redo + the six scroll motions
   Ctrl+D/U/F/B/E/Y in normal/visual; Ctrl+N in insert — pinned by `src/chord_test.odin`
   and by our `test_the_consumed_ctrl_chord_domain_is_eight`); everything else returns
   false from `ed_key` and falls through to the registry unchanged. Resulting UX: Ctrl+R
   = redo while focused, opens the script tab while unfocused; Ctrl+N = completion only
   mid-insert. No rebinding of defaults.
6. **Esc belongs to the editor UNCONDITIONALLY while focused** (maintainer decision at
   the manual pass — vim users hit Esc reflexively in every mode). The guard has ONE
   reader: the glfw `_install_escape_filter` swallows a focused-editor Esc before imgui
   ever sees it (imgui's nav-cancel would otherwise walk focus out of the child), while
   the drain still queues it for the editor. `_handle_escape` therefore has NO editor
   branch — on focused-editor frames its `is_key_pressed(escape)` is simply False.
   Ctrl+[ translates to the same Esc event (vim's second Escape). Keyboard defocus
   lives on CYCLE_REGION (Ctrl+`) and the mouse.
7. **Focus model unchanged.** The editor child window + an `invisible_button` over the
   image carry click-to-focus; `app.editor_focused` still reads
   `is_window_focused(child_windows)` after the draw; `editor_focus_requested` /
   `editor_defocus_requested` / `editor_was_ever_focused` keep their meanings. The child
   gets `no_nav_inputs | no_scrollbar | no_scroll_with_mouse`. The TextEditor
   first-render-focus-grab quirk disappears.
8. **Clipboard: Ctrl+C/X/V host-wired, UNIFIED with the editor's unnamed register**
   (editor commit 4befeaf gave the keymap one register — dd/yy/x/cw write it, p/P paste
   it). One slot, never two clipboards that disagree: the drain syncs OS→register before
   the frame's keys and register→OS after them, comparing against the ACTIVE session's
   own register (registers are per-handle — the sweep killed an app-global "last seen"
   that left a switched-to tab unseeded, silently breaking cross-tab yy/p); Ctrl+C/X
   also write the register immediately. Linewise is inferred from a trailing newline.
   At most one OS round-trip per keyed frame. Ctrl+V stays the host insert path (works
   in insert mode; vim-paste semantics belong to p/P).
9. **Chrome is host-drawn imgui, on the ABI's furniture queries.** Gutter: the HOST
   reserves the space — `ed_layout` reserves nothing, so the layout origin's x is the
   gutter width (`ed_gutter_cells` × the cell width, converging one frame behind a
   line-count change), and line numbers draw left of `ed_text_origin` at rows placed
   by `ed_cell_size` (all answer against the last `ed_layout`; the reference UI
   offsets its origin the same way). Status row: extend
   `draw_chrome` with a mode badge (NORMAL/INSERT/VISUAL/V-LINE, `ed_mode`) + `line:col`;
   while the command line is open (`ed_command_line` ≥ 0) the row shows
   `ed_command_line_prompt` + the typed text + `ed_command_message` — search/substitute
   typing is fully visible. (These six calls exist and are exported; they are missing
   from `ffi/README.md` — filed with the editor session.)
10. **Mouse: host-owned.** Click → `ed_pixel_to_cursor` + `ed_set_cursor` (+ focus);
    drag → `ed_set_selection` (anchor at press, head follows); wheel →
    `ed_set_scroll(scroll − wheel×3)`; Ctrl+wheel keeps the font-size zoom (consumed
    before the editor sees it, as today). Double-click word-select via
    `ed_word_at_pixel` + `ed_set_selection`. The splitter-drag / copilot-hover mouse
    suppression becomes trivial: our handler simply doesn't act when
    `app.splitter_dragging or app.copilot_hovered` (no more `io.mouse_down` forcing).
11. **Theme + settings: five map through the funnel, font size does not.** A
    `PALETTE: dict[Slot, rgba]` built from `theme.py` COLOR tokens (gruvbox), applied at
    session creation. `_apply_editor_settings_to` maps five settings:
    `show_whitespace → ed_set_show_whitespace`, `show_line_numbers → chrome flag
    Line_Numbers (drives ed_gutter_cells) + host gutter`, `show_matching_brackets → view
    flag`, `tab_size → ed_set_tab_width`, `line_spacing → ed_set_line_spacing`.
    **`font_size` is NOT a mapping:** it reaches the editor only as `ed_layout`'s
    `px_per_em` through the render path, and takes effect via the redraw-gate tuple —
    the settings funnel is no longer its application point. Language:
    `ed_language_for_path`, falling back to GLSL when unknown (host policy per the
    editor's feature 004).
12. **Copilot lock: `ed_set_read_only` per frame** from `copilot_turn_active`.
    Host writes (`set_text`, `insert_at_caret`) are unaffected by read-only, which is
    exactly the lock semantics feature 020 wants. The editor now DRAWS behind modals
    (no more TextEditor FPE gate) — input is already focus-gated.
13. **Vendoring: `shaderbox/resources/editor/{libeditor.so, atlas.png, atlas.json}`**,
    committed, plus a one-line `VERSION` file with the editor-repo commit sha of the
    build. The prerequisite is met: the editor repo committed the D6/D9 exports + the
    bulk getter as `e55b9979a122f2630191dceca51d5014c888f414`; that sha is what gets
    vendored (rebuild from it, never from a dirty tree). Loaded lazily on
    first `Editor` construction (a `ctypes.CDLL` load is milliseconds; no GL needed).
    Rebuild procedure documented in `conventions.md ## Known quirks` (build in the editor
    repo: `odin build ffi -build-mode:shared -no-entry-point -out:libeditor.so`, copy the
    files, update `VERSION`) — that entry, not this one, carries the current vendored set,
    which has since grown to seven files. 069 W-F re-vendored `c5c6ae230a51ece5` and, with
    `ed_set_draw_chrome` on, the library draws the gutter and the status row itself: the host
    `_draw_gutter` and the bottom bar's vim half are gone.
14. **Completion: host-driven on the deliberate-offer rule (editor commit f57e8d0).**
    First landed against a misread of `ed_complete_prefix` (it reports the buffer
    prefix whether or not the popup is open — feeding on it armed an invisible popup
    whose Enter silently accepted; and `ed_layout` didn't emit the popup at all). Both
    were fixed editor-side same-day, plus `ed_set_host_completion` (suppresses the
    built-in buffer-word source so Ctrl+N reports consumed but opens NOTHING — no
    one-frame flash of the wrong list). The host wiring: sessions set
    `host_completion`; the drain marks a consumed insert-mode Ctrl+N; `code.draw`
    answers by pushing the filtered vocabulary (pushing IS opening — `SB_*` lib names
    + the pass's uniforms + GLSL words for shader/lib tabs, Python keywords for script
    tabs, capped at 50), re-filters while the popup is open and the prefix moves, and
    `ed_complete_cancel`s when nothing matches (stays in INSERT). `complete_open` /
    `selected` / `count` are redraw-tuple members, so navigation repaints.
15. **Vim-reserved Ctrl chords while focused: the vim thing or nothing, never an app
    action** (maintainer requirement at the manual pass: "hotkeys must not mess with
    vim's bindings"). As of editor 4b110f0 the six scroll chords are REAL keymap
    motions (nvim-measured: Ctrl+D/U/F/B move the CURSOR; Ctrl+E/Y and zz/zt/zb are
    view-only and surface via `ed_take_scroll_request` — measured contract: reading
    consumes, returns the absolute target row, and applies it; `code.draw` takes once
    per frame). The consumed chords suppress DELETE_DOCUMENT / OPEN_SHADER etc. via
    the D5 skip. The host adds what `ed_layout` deliberately doesn't: FOLLOW-THE-CURSOR
    scrolling (on a cursor CHANGE landing outside the view → `scroll_to_line`; an idle
    caret never snaps a wheel-scrolled view back). Host fallbacks that remain (run
    AFTER `ed_key`, so future keymap growth shadows them automatically): insert-mode
    protections — Ctrl+W deletes the word back (must NOT close the tab; CLOSE_CODE_TAB
    keeps normal-mode Ctrl+W), Ctrl+U deletes to line start, Ctrl+P
    navigates/triggers completion (with the popup open, Ctrl+N/P navigate — the
    host intercepts Ctrl+N before `ed_key`, which would otherwise CLOSE the popup,
    measured), Ctrl+H backspaces and Ctrl+J breaks the line (their vim meanings),
    the remaining reserved chords consume-noop mid-insert. Outside insert, the
    unconsumed reserved chords (visual-mode scrolls — an editor-side gap — plus
    Ctrl+N/P/J/H as line/left motions, Ctrl+R/O) do the vim motion or nothing.
    The FULL reserved set is `_VIM_RESERVED_CHORDS` = d u f b e y r o w n p h j,
    with ONE carve-out: NORMAL-mode Ctrl+W falls through to CLOSE_CODE_TAB (vim's
    own normal Ctrl+W is only a window prefix and no windows exist). Ex commands
    whose object is the FILE arrive via `ed_take_host_command` (editor c5fabc8;
    parser-validated, force+arg included): `:w` saves through `App.save` — the one
    funnel with the copilot busy gate and a real DISK write (a memory-only flush
    made `:w` then `:q!` revert past the save) — `:q` refuses a dirty buffer unless
    forced (`:q!` reloads from disk and closes), `:wq`/`:x` save+close; a path
    argument is refused with a notice. Known deviations, judged acceptable: Ctrl+V stays host
    paste (no visual-block mode exists), Ctrl+A/X increment/decrement don't exist.
    The app halves of every suppressed chord stay reachable while unfocused.
16. **TextEditor is deleted outright — no fallback path.** `imgui_color_text_edit`
    imports go from `editor_types.py`, `app.py`, `tabs/code.py`; the palette call dies.
    Doc fallout is enumerated in Files touched (conventions bullet, skill §8 lines,
    dev_flow module map). Per the NO-backward-compat rule there is no dual-editor mode.

## Files touched

- **New:** `shaderbox/editor/__init__.py`, `editor/ffi.py` (ctypes binding, leaf, no
  imgui/moderngl), `editor/render.py` (moderngl MTSDF pass; imports `ffi`),
  `editor/input.py` (glfw→ed_key translation + drain; imports `ffi`, `commands`),
  `shaderbox/resources/editor/{libeditor.so, atlas.png, atlas.json, VERSION}`.
- **Changed:** `editor_types.py` (EditorSession.editor type), `app.py` (session
  creation/settings/palette, glfw callback chaining, drain state, consumed-chords set,
  redraw counter), `tabs/code.py` (render path, chrome incl. command line, mouse,
  markers), `hotkeys.py` (drain call + consumed-chord gate + Esc branch), `watch.py` +
  `widgets/uniform.py` (method names survive; type touch only), `theme.py` (slot
  palette), `popups/settings.py` (comment touch: the FPE-quirk rationale for
  apply-on-close is gone), **`build.sh`** (per-platform resource filter: the Windows
  stage excludes `libeditor.so`; `verify_clean` gains the rule so the leak aborts the
  build), `scripts/smoke.py` (editor render assertion — see Manual verification 11),
  `tests/test_pass_editor_wiring.py` (the `_edit` dirty-marking helper — D2). Other
  editor-touching test files verified type-clean by pre-impl review (they touch
  `editor_tabs`/settings/stubs only): no changes expected.
- **Docs:** `conventions.md` (the "one `TextEditor` per opened FILE" design bullet
  rewritten for the new editor; vendoring quirk added; Known-quirks TextEditor entries
  marked superseded), `roadmap.md` row + banner, `/imgui-ui` SKILL.md (§8 TextEditor
  quirks + the `imgui_color_text_edit` pyright-suppression mention + §9 focus-flag note),
  `dev_flow.md` module map (new `editor/` package; `editor_types` line).

## Safety-guard wiring (reader + exerciser per guard)

| Guard | Reader (consumer call site) | Exerciser |
|---|---|---|
| Unfocused drain gate (D4) | `hotkeys.dispatch_commands` → drain's `editor_focused` check | MV 2 + unit test: drain with `editor_focused=False` forwards nothing |
| Consumed-chord skip (D5) | `_dispatch_registry`'s membership test on `app.editor_consumed_chords` | MV 3 + unit test: seed set with Ctrl+R chord int → `OPEN_SCRIPT` callback not invoked; empty set → invoked |
| Esc editor-ownership (D6) | the glfw `_install_escape_filter`'s focused-editor swallow | unit test: Esc forwards in every mode; maintainer: Esc never defocuses |
| Vim-reserved chords (D15) | the drain's `_handle_vim_chord` + the consumed-set skip | unit tests: motions consumed + DELETE_DOCUMENT suppressed; insert Ctrl+R/W reserved; visual-mode fall-through reserved |
| Copilot read-only lock (D12) | `tabs/code.py`'s per-frame `ed_set_read_only` | MV 6: typing refused mid-turn, host edit lands |
| Redraw gate (D3) | the render path's `should_redraw` branch | MV 10 (counted) + per-field domain unit test |

## Manual verification

Each step fails for one reason and names its falsifier; the consumer is verified.

1. **Global-collision canary (insert mode):** focus editor, enter insert, type
   `dnwspqre` with and without Ctrl held on the `n`. Buffer receives the plain chars;
   document count, tab count, popup state unchanged (Ctrl+N mid-insert = completion, not
   New document). Falsifier: remove the consumed-chord skip — Ctrl+N opens a new
   document mid-typing. (Vim verb correctness itself is the editor repo's test corpus,
   not re-verified here.)
2. **Unfocused editor is deaf:** click the uniform panel, type `dw` — buffer unchanged
   (dirty dot stays absent). Pinned by the drain-gate unit test.
3. **Chord arbitration (the one hard requirement):** editor focused, delete a line with
   `dd`, press Ctrl+R once — the line comes back AND `len(app.editor_tabs)` is
   unchanged. Falsifier: without the skip, the same press also appends a script tab.
   Unfocused: Ctrl+R opens the script tab (baseline `GLOBAL` behavior, unchanged).
4. **Esc never leaves the editor:** (a) insert + Esc → normal mode, still focused;
   (b) normal + pending `3d` + Esc → pending cancelled, still focused; (c) normal idle
   + Esc → STILL focused (falsifier: any defocus on Esc is the regression); Ctrl+`
   cycles the region out, a click elsewhere defocuses.
4b. **Vim chords while focused:** Ctrl+D/U half-page scroll (no document deleted — the
   falsifier), Ctrl+F/B page, Ctrl+E/Y line; in insert: Ctrl+W deletes the word back
   (tab stays open — the falsifier), Ctrl+U clears to line start, Ctrl+P completes;
   Ctrl+O does nothing (no project dialog). Unfocused: Ctrl+D/E/O/W regain their app
   meanings.
5. **Error jump verifies the scroll consumer:** break a shader with the error line BELOW
   the fold. Strip click / F8 → caret on the line, line V-LINE-selected, view scrolled
   to center it. Falsifier: an unwired `ed_scroll_to_line` leaves the error off-screen
   while the caret silently moved.
6. **Copilot lock:** start a copilot turn → typing refused (insert entry refused),
   copilot's `edit_shader` still lands, dirty/undo intact after the turn.
7. **Hot reload against a dirty buffer:** type one char (dot appears), edit the same
   file externally → buffer shows disk text, dot clears. Falsifier: re-baselining
   `saved_undo` BEFORE `set_text` leaves the dot stuck (revision rises across
   `set_text`).
8. **Clipboard:** visual-select, Ctrl+C, move, Ctrl+V; Ctrl+X removes and carries.
9. **Settings, individually:** each of the five funnel settings toggled alone changes
   the render next frame (each bumps `editor_redraw_count` ≥ 1 — a silently-dropped
   mapping is the failure mode); font size via slider AND Ctrl+wheel re-renders (its
   path is the render tuple, not the funnel).
10. **Redraw gate, counted not felt:** static: focus the editor, idle 120 frames →
    `editor_redraw_count` unchanged (falsifier: gate deleted → +~120). Live: one `j` →
    counter +1 exactly (falsifier: cursor dropped from the tuple → counter still, caret
    frozen). The per-field domain unit test walks every tuple member.
11. **Command line visible:** type `/foo` — the status row shows the prompt + pattern as
    typed; Enter jumps; `n` repeats. Falsifier: an unwired `ed_command_line` leaves the
    row blank while the buffer jumps on Enter.
12. **Gates:** `make check` / `make test` green; `make smoke` gains: after the seeded
    frames, the active session's last layout produced > 0 primitives INCLUDING ≥ 1
    Glyph-kind (falsifier: a missing atlas load yields only Missing_Glyph kinds), and
    `editor_redraw_count ≥ 1`. FFI unit tests run headless (no GL).

## Review history

- **Pre-impl round (2 × opus, 2026-09-01): both PARTIAL → all findings accepted, folded.**
  Design reviewer: Esc gate moved from `ed_mode`-only to `ed_pending` (the ABI's dedicated
  query, uncommitted editor-side at review time); `ed_cell_size`/`ed_text_origin`/
  `ed_command_line`+prompt+message discovered already-exported → chrome workarounds and
  the "search is blind" open question deleted; vendoring gated on the editor repo
  committing its tree; consumed-chord set specified in imgui-chord-int space; drain-gate
  staleness stated + same-frame post-Esc queue drop added; `_LIB` global + shared scratch
  buffer declared; font_size's non-funnel path stated; `test_pass_editor_wiring.py`
  helper inversion caught. Verification reviewer: OPEN_SCRIPT/`SAVE` are GLOBAL-scoped
  (category ≠ scope) → the skip is the ONLY Ctrl+R guard and MV 3 was rewritten
  falsifiably; redraw gate restructured to counted free functions + domain test (fps
  sleep made "no visible lag" theater); guard/reader/exerciser table added; `build.sh`
  Windows-stage leak added to scope; hot-reload step now starts dirty; smoke assertion
  made decidable (Glyph-kind count). No findings rejected.

- **Post-impl round (3 × opus, 2026-09-01): correctness FAIL, architecture PARTIAL,
  spec-fidelity PARTIAL → all findings fixed in the same wave.** Correctness (all
  probe-demonstrated): render_state carried no editor identity, so two fresh tabs
  sharing the one panel compared equal and a tab switch showed stale text → identity
  (the session path) + gutter_px joined the tuple, with a two-editor falsifier test;
  the completion feature was cut entirely (D14 above — both root causes are editor-side
  ABI gaps); `ed_free` was never called (~1.5 MB native per opened file) → sessions
  close at every eviction site and in `App.release`; `EditorPanel` leaked FBO+texture
  per resize → released before reassign + a `release()`; Ctrl+X/V bypassed the copilot
  read-only lock through the host write path → gated on `copilot_turn_active`; the key
  queue flooded after degenerate frames → cleared on that path; `get_text` silently
  truncated at 1 MiB → grow-and-retry; the focus-request latch could stick with no tab
  open → cleared on the early return; marker-state entries leaked on renames → popped.
  Architecture: the commit's "make check clean" claim was FALSE — the gate was judged
  through a pipe (`| tail` eats the exit code; the exact 064 failure mode) and the tree
  carried an unused `noqa` + four unformatted files; seven stale TextEditor comments
  (two describing behavior the code no longer has) rewritten; two dead params dropped;
  a dangling test-file reference fixed. Spec-fidelity: the D9 gutter was dead code
  (origin never offset — `ed_text_origin` stayed 0 and the draw early-returned; fixed
  per D9's rewritten mechanism); the D5 registry-skip falsifier existed only as the
  recording half → `spec_eligible` extracted + tested; D7's missing child flags added.
  No findings rejected.

- **Review-sweep round (workflow: 9 dimensions + adversarial verify, 20 agents,
  2026-09-01): 66 confirmed findings — every BLOCKER/MAJOR fixed same-wave.** The
  heavy ones: mouse/wheel mutated editor state AFTER layout, so the redraw gate
  recorded a state the texture didn't show (draw reordered: interactions →
  reconciliation → layout → gate); visual-mode scroll chords fell through to the
  registry (Ctrl+D deleted a document from visual mode) and insert-mode Ctrl+R
  opened the script tab — the reserved-chord set generalized to d u f b e y r o w
  n p h j across all modes with vim motions where cheap (Ctrl+N/P/J/H) and one
  carve-out (normal Ctrl+W = close tab); the clipboard sync's app-global "seen"
  broke cross-tab yy/p (now compares the active session's own register); `:w` only
  flushed memory (now `App.save`: disk + busy gate); jump-to-error left V-LINE
  mode so the next key operated on a selection (select_line dropped); repeated
  Ctrl+N reset completion to candidate 0 (editor-side Ctrl+N CLOSES an open popup
  — measured — so the host navigates instead of forwarding); a jump into a
  never-laid-out session was never followed (rows==0 frames no longer record the
  cursor); pass deletion never tore down its editor session/tab (new
  `App.close_editor_for_path` funnel); GL blend state leaked out of the panel
  draw; project switch leaked tabs/cursor state (release() clears); embedded NUL
  truncated every text crossing (one _encode funnel strips); Ctrl+[ now = Esc.
  Accepted with rationale: the 1-2-frame focus-transition keystroke misroute
  (inherent to the draw order, visibly lost not silently misapplied), gutter
  metrics one frame behind (self-correcting), theme.py's ffi import (tokens stay
  in the one bag). Editor-side items filed to feature 004: dot-repeat missing,
  c-family double undo step, di{/ci{ linewise collapse, visual-mode scroll
  motions unbound, sticky ed_command_message, insert-Ctrl+R register paste.
  Test debt paid: 48 tests in the editor file (isolated redraw-tuple pairs, the
  spec_eligible gates, cross-tab clipboard, reserved-chord matrix, _pass_for_tab
  falsifier in the wiring file); smoke's redraw assert is now two-sided (a
  stuck-open gate fails it, not just a dead one).

## Open questions for the user

1. **Windows story at ship time:** the vendored binary is linux-only. Proposal: land on
   `dev` now with `build.sh` excluding the `.so` from the Windows stage; `/ship` gains a
   blocking note ("needs libeditor.dll + a Windows verify") until an Odin Windows build
   exists. OK? (Robust default applied: proceeding with this.)
2. **Ctrl+R while focused = redo (OPEN_SCRIPT only when unfocused)** — acceptable, or
   rebind OPEN_SCRIPT's default now? (Default applied: keep bindings, editor wins while
   focused.)
