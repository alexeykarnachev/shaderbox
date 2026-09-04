# 073 — the third walk: findings into a plan

The maintainer tested the working tree (v0.28.0 plus the uncommitted 070 + 072 diff) in the
running app and filed 8 findings and 5 decisions (`00_findings.md`, each finding verified
against the code before filing). This spec groups them into waves, fixes the order, and carries
the decisions as constraints. Where a finding needs a redesign, the redesign is the fix.

Size: **medium-blast-radius** — it changes the editor library and re-vendors it, adds three
commands, a viewer mode, and reshapes the pass strip's dormant/live marking. Each wave is its
own commit series with its own pre/post review per `dev_flow.md`.

Two repos, two sessions, in parallel: W-A lands in the editor repo (`~/src/editor`,
`alexeykarnachev/editor`) under that repo's own flow, done by the editor session from ONE batch
request carrying the ledger items verbatim; it pings this session back when its commit is ready,
and this session re-vendors per `conventions.md ## Known quirks`. Every other wave is host-only
and lands here while the editor session works.

---

## Goal

After this feature: a closing bracket lands where vim puts it; the caret and the highlighted
line read as vim's reverse video; completion opens as you type and knows that `uniform ` wants a
builtin; `K` over an `SB_*` name shows its signature and doc; the viewer can show alpha on
demand without changing its default look; a live pass tile is marked by a quiet fill instead of
a dormant one being darkened; `Alt+Left` / `Alt+Right` walk the passes without moving the
focus; and Reset sits with the other document-level verbs.

## Out of scope (each with a trigger)

- **Docs for GLSL builtins and engine uniforms under `K`.** W-B's lookup table is the `SB_*`
  index plus `ENGINE_UNIFORM_DOCS`; `mix` / `smoothstep` have no doc source in the repo.
  Trigger: the maintainer presses `K` on a builtin and wants an answer.
- **Hover-triggered popups (mouse dwell over a symbol).** `K` is the trigger this feature ships;
  the same lookup serves a dwell later. Trigger: the maintainer asks for it after using `K`.
- **A completion detail column (type / doc beside each candidate).** Needs a `Completion` field
  and a chrome change in the library. The editor session decides whether it comes with W-A; if
  not, trigger: the maintainer wants to tell two `u_*` candidates apart in the popup.
- **A checker-tile size or color setting.** The light checker of the Color+Alpha view is a
  theme token; the Color view's quiet checker is unchanged. Trigger: a complaint about either.
- **Windows `libeditor.dll`.** Still owed from 067; W-A's re-vendor is the Linux `.so` only.
  Trigger: next `/ship` on a Windows host.

## Locked decisions (from the walk — constraints, not options)

- **D1. Closing-bracket snap is the matching-opener rule.** A closer typed as the first
  non-blank of a line takes the indent of the line holding its matching opener, at any depth,
  for every bracket kind the pair table knows. "Column 0" and "current minus one step" are out.
- **D2. The viewer gets a three-state channel view, default unchanged.** Color (today's frame,
  byte-identical), Color+Alpha (over a light checker), Alpha (alpha as grayscale). A registered
  command with a chord and a visible toggle on the viewer. The Alpha view must not disturb the
  texture feedback reads and exports sample.
- **D3. The dormant tint goes; a live tile gets a quiet fill.** `STALE_TINT` is deleted, a
  dormant image is blitted untinted; every pass in the output's chain gets an accent-derived
  fill as the cell ground; dormant tiles keep the plain ground and the corner tick; the output's
  accent border and error red are unchanged. Fill color tuned by eye. Supersedes 071 D2.
  *(Landed: after two fills were rejected by eye, no fill at all; a live name is bold
  `FG_TITLE`, a dormant name `FG_DORMANT`. `40_wave_d_strip.md`.)*
- **D4. `Alt+Right` / `Alt+Left` are next / previous pass, and switching keeps the focus where
  it was.** A switch is what a tile click is (output pass + shader tab), in the strip's drawn
  order, wrapping. Both are registered commands.
- **D5. Reset moves to the `New document` / `Render all` row.** `F6` and the tooltip stay.
  *(Superseded by 078 D4: Reset sits on the Document tab's first row.)*
- **Standing (from 069/071, unchanged):** the editor lib is the vendored vim; a vim behavior
  the maintainer expects is filed on the library, never approximated in the host. Fixes land at
  the class, not the instance; no compat code; `make gates` green before "done".

## Workstreams

### W-A — Editor library: closer snap, reverse-video caret, cursor line, and the completion / lookup seams (#1 #6 + D1; #2 #3 library halves)

Two repos. The library side is the editor session's, from one batch message carrying the four
ledger rows verbatim with their headless reproductions:

1. **#1 + D1, fixed ask.** A `Behavior` flag applied in the insert path: when the typed rune is
   a closer and everything before the cursor on its line is whitespace, replace that leading
   whitespace with the indent of the line holding the matching opener (`bracket_match`). One
   undo step with the character. Corpus: `A<CR>x;<CR>}` on `void f() {` gives `}` at column 0;
   the nested `if` case from the ledger gives the inner `}` at 4 and the outer at 0;
   `vim_coverage.md` line.
2. **#6, fixed ask.** Reverse-video caret in normal / visual mode: the glyph under the caret is
   emitted in the background color and the caret quad is opaque (or a `CARET_TEXT` slot the
   host sets). Plus a cursor-line band: whichever is cheaper in the library, a chrome primitive
   with its own slot, or nothing (the host then marks the cursor line per frame through
   `ed_add_marker` and sets its `text` color). The editor session says which.
3. **#2, open.** Whether `K` is claimed by the editor (a `vim_coverage.md` entry plus a host
   callback like the ex-command seam) or stays an unbound key the host catches through
   `ed_key`'s false. And whether the popup is a library chrome primitive or host imgui.
4. **#3, open.** Whether a `Completion` gains a detail string (type / doc) shown beside the
   candidate. Nothing else is needed from the library for #3: the host offers unprompted and
   reads its own context.

Shaderbox half, after the ping: rebuild from the committed sha, copy the vendored set, bind any
new slot / primitive / flag in `theme.py` and `editor/ffi.py`, delete the mitigations the new
sha makes dead, `abi_probe.py` green, the vendored `vim_coverage.md` refreshed. Wave file
`10_wave_a_editor.md` records what the editor session reported and what the re-vendor changed.

### W-B — Code panel: completion providers, auto-trigger, `K` lookup (#2 #3 host halves)

- Replace `_completion_vocabulary` with a provider table: each provider is a predicate on
  (the line before the caret, the tab kind) and a candidate list. Providers, in order:
  `uniform\s+$` on a shader tab -> builtin declarations from `ENGINE_UNIFORM_DOCS` (`float
  u_time;` form); identifier prefix on a shader tab -> `SB_*` functions + the pass's uniforms +
  `_GLSL_WORDS`; identifier prefix on a script tab -> Python keywords. The first provider that
  fires is offered. *(Landed shape, after review round 1: every eligible provider's matches
  concatenate in table order; a line comment offers nothing. `20_wave_b_code_panel.md`.)*
- `_drive_completion` offers on every insert-mode frame where the revision or cursor changed and
  a provider fires (auto-trigger, with a minimum prefix length for the identifier providers so
  one letter does not open a list), and on Ctrl+N / Ctrl+P as today. An offer that yields no
  candidates cancels, as today.
- `K` in normal mode over a word: look it up in `app.shader_lib_index.functions` (signature +
  doc) or `ENGINE_UNIFORM_DOCS` (type + doc); draw an imgui tooltip-style popup anchored at the
  caret cell, dismissed by any key or click. Where the editor session's W-A answer changes the
  seam (a library callback, a library-drawn popup), this bullet follows it; the host lookup is
  the same either way.
- Tests: the provider table over the three contexts; auto-trigger opens on the second typed
  letter and not the first; `uniform ` opens with the builtin set; `K` over `SB_hash` yields its
  signature; `K` over nothing yields nothing.

### W-C — Main view: channel view and Reset placement (#4 #8 + D2 D5)

- `CommandId.CYCLE_CHANNEL_VIEW`, chord to pick (`Alt+V` if free), scope DOCUMENT; app state
  `channel_view: ChannelView` (`COLOR` / `COLOR_ALPHA` / `ALPHA`), persisted in `AppState`.
- Color: exactly today's draw. Color+Alpha: `_draw_canvas_backdrop` with a second checker
  texture from two light tokens (`CHECKER_LIGHT_LOUD` / `CHECKER_DARK_LOUD`). Alpha: a view
  texture the size of the output, filled by a one-quad blit shader that writes `vec4(a, a, a,
  1)`, drawn in place of the output; the output texture itself is never swizzled.
- A small three-state control on the viewer beside the FPS chip, showing the current state.
- The `Reset` ghost button leaves `_draw_document_image` and is drawn in
  `widgets/document_grid.py` after `Render all`, same handler, same tooltip and hint.
- Tests: the channel state cycles and persists; the Alpha blit of a known RGBA texture yields
  the expected grayscale; the Reset handler is reachable from the grid draw.

### W-D — Pass strip: live fill, next / previous pass (#5 #7 + D3 D4)

- `theme.py`: delete `STALE_TINT`; add `PASS_LIVE_FILL` (accent at low alpha over the panel
  ground, tuned at the display). `preview_cell` drops the `tint_col` branch; `_draw_pass_tile`
  passes `bg_color=COLOR.PASS_LIVE_FILL` for a live tile and nothing for a dormant one; the
  corner tick and dim footer stay on dormant tiles.
- `CommandId.NEXT_PASS` / `PREV_PASS` on `Alt+Right` / `Alt+Left`, scope DOCUMENT. Handler:
  index of the output in `_strip_order(document.passes, wiring)`, step with wrap, then exactly
  what a tile click does (`ensure_shader_tab(document_id, name, focus_editor=app.editor_focused)`
  and `session.set_output_pass`). `_strip_order` moves to where both the widget and the command
  can reach it without the widget importing the app's command layer.
- Tests: the order and wrap; the focus argument follows the frame's focus state; a dormant tile
  is blitted white-tinted and a live one carries the fill.

### W-E — Sanitize

`/sanitize` after the last wave: `todo.md` triggers, `conventions.md` (a note that the viewer
has a channel view and the strip marks live, not dormant), roadmap row + banner, cold-context
check, the tutorial's regenerated cards if any strip picture changed.

## Order

1. **W-C** and **W-D** first — host-only, independent of the editor session, and the ones the
   maintainer will see on the next launch.
2. **W-B's completion half** — host-only; its `K` half waits for W-A's answer on the seam.
3. **W-A shaderbox half** when the editor session pings: re-vendor, bind, verify.
4. **W-B's `K` half** on the seam W-A settled.
5. **W-E.**

Each wave: pre-impl review of its wave file where it changes a design (W-B, W-C), implement,
`make gates`, post-impl review, commit on `dev`.

## Files touched

`shaderbox/tabs/code.py`, `shaderbox/commands.py`, `shaderbox/app.py`, `shaderbox/app_state.py`,
`shaderbox/ui.py`, `shaderbox/widgets/document_grid.py`, `shaderbox/widgets/pass_list.py`,
`shaderbox/ui_primitives.py`, `shaderbox/theme.py`, `shaderbox/help_content.py` (the shared
builtin table), `shaderbox/editor/ffi.py` + `shaderbox/resources/editor/*` (W-A), tests beside
each.

## Manual verification (the maintainer, in the app)

1. Type `void f() {`, Enter, `if (a) {`, Enter, `x;`, Enter, `}`, Enter, `}`: the inner brace
   sits at 4, the outer at 0, nothing typed after them drifts.
2. In normal mode the glyph under the caret reads as reverse video; the cursor line is visible
   and its text readable; an error line still flips to the primary text color.
3. Type `u_` in a shader: the popup opens by itself on the second letter. Type `uniform ` and
   the builtin declarations appear; accept one and a whole `uniform float u_time;` lands.
4. `K` over `SB_hash` shows its signature and doc; `K` over `u_time` shows its type and doc;
   any key dismisses it.
5. The viewer's channel control cycles Color / Color+Alpha / Alpha with the chord; the default
   frame is unchanged; on the cascades example the Alpha view shows where the output is
   transparent.
6. Dormant tiles show their picture untouched with the corner tick; live tiles carry the fill;
   the output keeps its accent border. Judge the fill color and say if it should change.
7. `Alt+Right` / `Alt+Left` from the editor keep the editor focused; from the panel they do not
   focus it; the order matches the strip and wraps.
8. Reset sits after `Render all`; `F6` still resets.

## Review history

- **Host waves W-C, W-D, W-B (completion half), post-implementation, one opus reviewer anchored
  on the ledger's verbatim words and D2-D5, every claim demonstrated by a probe.** Round 1:
  PARTIAL -- W-C and W-D PASS on every item; W-B carried three defects (an inert accept guard,
  a first-wins provider that shadowed `sampler2D` after `uniform`, completion inside a line
  comment). Fixed at the root (`20_wave_b_code_panel.md § Review history`). Round 2: PASS.
