# 011 — UI library consolidation + share-stack de-leak

Status: **spec / plan-locked (2026-05-24); NOT implemented — pre-implementation triage only.**
Shape: refactor (no behaviour change) — moves/renames/deletes + one leak fix + tests. Touches the UI
primitive sub-library (`theme.py`, `ui_utils.py`), the exporter seam (`exporters/base.py`), the
Telegram exporter, and the share-tab state. **The fragile imgui draw code (carousel/grid/cell
overlays, the cursor/layout math) is OUT of scope — see Decision 1.**

---

## Goal

After the long 010 UI-polish session, consolidate what we learned into a **solid, reusable UI
sub-library** and **stop Telegram/sharing concepts leaking up into generic layers** — WITHOUT
churning the freshly-stabilized draw code (where this session's bugs lived: the jitter, the
SetCursorPos assert, the scrollbar offset, the in-flight shift). This spec is the output of a 5-agent
review swarm (2026-05-24); it triages findings into **do-now** (low-risk, clear payoff) and
**defer-with-trigger** (premature at N=1).

The companion artifact is the **`imgui-ui` skill** (`.claude/skills/imgui-ui/`) — the generic
principles. This spec is the repo-specific cleanup.

## The layering contract being enforced

1. **Generic UI primitives** — `theme.py` (tokens) + the button/widget helpers. Know NOTHING of any
   feature.
2. **Generic sharing** — `exporters/base.py` (`Exporter` ABC, `RenderControl`, `RenderedArtifact`),
   `registry.py`, `render_preset.py`. Know NOTHING of Telegram.
3. **Telegram impl** — `exporters/telegram.py`, `stubs.py`.
4. **Share orchestration** — `tabs/share.py`, `tabs/share_state.py`.
5. **App / high-level** — `app.py`, `ui.py`, `popups/emoji_picker.py`.

Import direction is currently **clean** (verified: no lower layer imports a higher one, no cycle).
The failures are *vocabulary* leaking up, not import-direction violations.

## What the swarm validated as CLEAN (do not touch)

- Import direction + no cycles; the registry decoupling; "exporter owns its panel"; the GL-free
  `RenderedArtifact`/`RenderPreset` value types; the worker/render thread split in `telegram.py`
  (clean banners, queue-only communication); `render_preset.py` vocabulary (zero telegram terms);
  `tabs/share.py` is exporter-agnostic; `apply_theme` runtime re-skin works; the 4-tier button system
  + token bags are genuinely reusable in shape.
- **No accreted reverted-helper mess** in `telegram.py` (the central worry — confirmed absent; every
  `def _` has ≥1 caller, no `@staticmethod`, no unused imports).

---

## DO NOW (low risk, real payoff)

Ordered by leverage. All are move/rename/delete/test — none touch the fragile draw geometry.

### Decision A — De-leak Telegram tokens from the generic `SIZE` bag
**Consensus #1 leak (3 of 5 agents).** Move feature-specific tokens out of `theme.SIZE` into a
telegram-local constants block (`telegram.py` already has `_TG_*` + `_PREVIEW_THUMB_HEIGHT`):
`STICKER_PREVIEW_W/H`, `CAROUSEL_ARROW_W`, `PACK_COMBO_W`. **Delete dead `EMOJI_BTN`** (0 uses — even
telegram uses `ROW_HEIGHT`). **Resolve the drift:** `theme.TG_THUMB_H=90` and `TG_GRID_COLS` are dead
AND shadowed by `telegram._PREVIEW_THUMB_HEIGHT=110`/`_GRID_COLUMNS=4` — delete the theme copies (one
source of truth). Keep in `SIZE` only generic tokens (`ROW_HEIGHT`, `BTN_*`, `LABEL_W`, `CHIP_*`,
`TAB_MIN_W`, `NAME_INPUT_W`, `RES_COMBO_W`, scrollbar/grab). Risk: low (rename, no draw change).

### Decision B — Delete dead tokens + the dead `replace` job path
- `SIZE` dead tokens (0 non-def uses): `ROW_COMPACT`, `TOPBAR`, `STATUSBAR`, `BTN_MD_W`, `THUMB_MD`,
  `PANEL_RIGHT_W`, `RENDER_MAX_H`, `RENDER_MIN_H`, `GUTTER_W`, `NODE_CREATOR_COLS`. Plus `_FONT_DIR`.
  Verify each with grep at fix time (counts drift); keep `PREVIEW_W` (1 use in `ui.py`).
- **The entire `replace` path in `telegram.py` is unreachable** — no `_Job(kind="replace")` is ever
  enqueued (the Replace button was removed in 010 in favour of delete+add). Delete the dispatch
  branch + `_handle_replace` + `_do_replace` + the `verb="Replaced…"` `_notify_user` path (~55 lines).
  Risk: low (dead code).
- Stale comment at the `_UPLOAD_KINDS` site says "Add as new sticker button" — the button is "Add to
  pack". Reword/drop (conventions: comments don't narrate renamed UI).
- Drop the unused `artifact` param on `_draw_preview_box` (`_ = artifact` vestige).

### Decision C — Fix the preview/artifact GL leak on project switch
**The only correctness bug found.** `App._init` reuses `share_tab_state` (only reassigns
`scratch_dir`); `App.release` never touches it. So per-outlet `OutletRenderState.preview` GL textures
+ stale artifacts from the old project leak on project switch. Add `TabState.release()` (release every
outlet's preview, clear outlets) and call it on project switch + in `App.release`. Risk: low; covered
by smoke.

### Decision D — De-leak the exporter seam (`RenderControl` + ABC pack methods)
- **`RenderControl` carries sticker/emoji concepts** into the generic exporter contract: `emoji`,
  `set_emoji`, `emoji_button`, `open_emoji_picker`. A YouTube exporter wants none of these. Shrink
  `RenderControl` to pure render plumbing (`duration`, `artifact`, `artifact_is_fresh`, `set_duration`,
  `render`, `preview_texture_glo`, `preview_size`); move the emoji bundle into a telegram-owned
  sub-struct or closure set passed separately. (This also deletes the `_emoji_button` closure smell in
  `share.py`.)
- **`Exporter.set_default_pack`/`current_default_pack` on the generic ABC** — "pack" is a sticker-set
  concept forced onto YouTube/X stubs. Rename to a neutral notion OR drop from the ABC and reach it via
  `isinstance` at the one `app.py` call site. Decide which at implementation.
- Risk: medium-contained (dataclass + ABC reshape; mechanical, but touches `base.py`/`telegram.py`/
  `share.py`/`stubs.py`/`app.py` together — do it as one atomic change with `make check`).

### Decision E — Split `ui_utils.py`; land the promised tests
- `ui_utils.py` mixes UI draws with non-UI utils (`get_uniform_hash`, `unicode_to_str`, `adjust_size`,
  `select_next_value`, …). Split: `ui/primitives.py` (imgui+theme only: the button tiers,
  `caption_text`, `close_cross_button`, `draw_copyable_text`) and `util.py` (the non-UI helpers).
  Export the private `fade(color, a)` from `theme.py` (3 inline `COLOR.X[:3] + (a,)` sites).
  Risk: low (pure move; update imports).
- **Land the tests 010's build-step-1 promised but skipped:** `resolve_dims` unit cases
  (square/landscape/portrait + alignment), `render_media(preset=None)` byte-identical, `render_for`
  mint→exists→cleanup. No `tests/` dir exists yet — add one or extend `smoke.py`. Risk: none (additive).

### Decision F — `duration_slider` is telegram-flavored in a generic module
It hardcodes `0.5` min + `"%.1f s"` + `SIZE.LABEL_W`. Either parametrize (`v_min`, `fmt`, `label_w`)
to make it generic, or move it telegram-local. Low priority; bundle with E.

---

## DEFER (premature at N=1 — trigger required)

The button system + every primitive has **exactly one consumer** (`telegram.py`; `code.py` uses only
`draw_copyable_text`). Extracting abstractions from one example misfits the second. Each below waits
for a real second consumer:

| Deferred | Trigger |
|---|---|
| Extract a generic `GridCell` / thumbnail-grid component from the carousel | a 2nd grid with different cell content lands (e.g. a YouTube thumbnail grid) — **HIGHEST regression risk; this is the code that earned the per-cell-child + fixed-Y fixes** |
| `Widget` / `Component` protocol | a polymorphic `list[Widget]` dispatcher actually materializes (already deferred in `conventions.md`) |
| Generalize `RenderControl` / the outlet panel shape for reuse | the 2nd real outlet's control set is known — extract from two, not one |
| Extract `confirm_inline` / `thumbnail_box` / `label_row` primitives | a 2nd non-telegram caller needs them (today they're 1-2 sites each) |
| Trim/exercise `RenderPreset` `FIXED_DIMS`/`FIXED_ASPECT`/`aspect`/`target_*` + `LETTERBOX`/`CROP` | YouTube Shorts (9:16) lands — keep them dormant (latent enums, cheap), per 010 Decision 6 + Out-of-scope; do NOT rip out now (reversible churn for nothing) |

## Decision 1 — The fragile-draw exclusion (the load-bearing scope guard)

`_draw_sticker_grid_full` / `_draw_grid_cell` / `_draw_cell_contents` / `_draw_cell_overlays` /
`_draw_cell_delete_confirm` and the `_draw_sticker_section` cursor math are **explicitly out of scope**.
This is the exact code that earned: the per-cell `begin_child` (jitter + SetCursorPos assert fix), the
fixed-Y carousel anchor (in-flight shift fix), the content-region preview fit (scrollbar fix). Touching
it to satisfy a *hypothetical* second grid is the worst risk/reward in the whole refactor. Revisit only
under the GridCell trigger above, with the `imgui-ui` skill's §3/§4 open.

## Regression-risk ranking (for whoever implements)
1. HIGHEST — any reshape of the carousel/grid/cell-overlay draw code → DEFERRED (Decision 1).
2. HIGH — re-laying-out the accordion/preview cursor math → out of scope.
3. MEDIUM — reshaping `RenderControl` (Decision D) → contained; do atomically + `make check` + headless drive.
4. LOW — token moves/renames/deletes, dead-`replace` deletion, `ui_utils` split, tests, leak fix.

## Out of scope
- Behaviour changes / new UI. This is consolidation only.
- The fragile draw geometry (Decision 1).
- Generalizing for the 2nd outlet/tab/grid before it exists (Defer table).

## Build order (when implemented — NOT this session)
1. Decision C (leak fix) — isolated, lands first, smoke-verified.
2. Decision B (dead code/tokens/`replace`) — pure deletes.
3. Decision A (token de-leak) — moves/renames.
4. Decision E (module split + tests).
5. Decision D (seam de-leak) — atomic cross-file change; headless-drive the share panel after.
6. Decision F.
7. `/sanitize`: roadmap row/banner, conventions (token rule reaffirmed; button-tier rule already added).

## Verification
Every step: `make check` (0 errors) + `make smoke`. Decisions C/D: headless-drive the connected
Telegram panel (the `imgui-ui` skill §0 pattern) — render/add/delete/emoji-change/carousel + project
switch — to confirm no regression of the 010 fixes. No visual change expected (refactor), so minimal
screenshot need; the maintainer eyeballs once after D.
