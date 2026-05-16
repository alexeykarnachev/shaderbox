# shaderbox · design deliverables · v1

Everything in this folder is what the design pass produced. Read in this order:

1. **`prototype.html`** (in this folder) — canonical visual + interaction source-of-truth. Open it in a browser; toggle the Tweaks panel (⚙ top-right) to preview accent / density / rounding / editor-side variants and the compile-error demo state. Every pixel value and color in here is intentional — grep CSS variables for the exact tokens.

2. **[`SPEC.md`](./SPEC.md)** — the bridge document. Per-panel layout intent + imgui call sequences + the new custom-draw effects + a suggested 8-PR adoption sequence.

3. **[`tokens.json`](./tokens.json)** — machine-readable token surface (colors, sizes, spacing, rounding, fonts). Source-of-truth for both `theme.py` and `prototype.html` — keep them in sync.

4. **[`theme.py`](./theme.py)** — drop-in Python for `imgui-bundle 1.92.x`. Exposes `apply_theme()`, `COLOR`, `SIZE`, `SPACE` and `load_fonts()`. Re-callable at runtime to swap accent / density / rounding from a Tweaks panel.

## Quick-start (literal Day-1 integration)

```python
# in shaderbox/app.py, right after imgui context creation
from shaderbox.theme import apply_theme, load_fonts

apply_theme(imgui.get_style(), accent="yellow", density="tight", rounding="subtle")

# Font: pick one of the two below at integration time.
#   (a) Keep using the existing Anonymous Pro file you already ship:
self.font_14, self.font_18 = load_fonts(
    io=imgui.get_io(),
    ttf_path="resources/fonts/AnonymousPro-Regular.ttf",  # adjust to your path
)
#   (b) Or ship JetBrains Mono — better pairing with gruvbox visually,
#       but you'll need to add the TTF to the repo:
# self.font_14, self.font_18 = load_fonts(
#     io=imgui.get_io(),
#     ttf_path="resources/fonts/JetBrainsMono-Regular.ttf",
# )
```

That's it — the app is now gruvbox-skinned. Everything else (the wide-screen layout, the embedded editor, the status bar) is incremental work tracked in `SPEC.md` §15.

## Scope reminder

This package covers what was asked for in the original handoff + the `layout-intent.md` scope expansion:

- ✅ Gruvbox palette mapped to all ImGuiCol_* roles
- ✅ ImGuiStyle numeric values
- ✅ Wide-screen layout (50/50 editor | right pane)
- ✅ Embedded GLSL editor (mocked in prototype; spec'd against `imgui_color_text_edit` in SPEC.md §3.1)
- ✅ Status bar (replaces notification overlay + buries FPS readout less awkwardly)
- ✅ Token-driven sizing (tokens.json §sizes consumed by both theme.py and the CSS)
- ✅ Tweaks panel for runtime accent / density / rounding / editor-side swaps

Out of scope (intentionally — see SPEC.md §16):

- Multi-file editor (vertex shader, node.json) — single-file for now
- Node-graph editor (`imgui_node_editor`) — long-term direction, not v1
- Drag-and-drop import — wishlist item, deferred
- Camera / uniform preset system — wishlist item, deferred

## Versioning

This is `v1`. If the developer comes back with revision notes, the next round will be `v2` — a new `prototype.html` (kept side-by-side with v1 for diff'ing), a regenerated `theme.py` / `tokens.json` / `SPEC.md`. Don't merge v1 → v2 in-place; we want the round-trip history visible.

## Loose ends in the prototype to keep in mind

- **Syntax highlighting** in `prototype.html`'s editor is a hand-rolled regex pass — the real `imgui_color_text_edit` has a proper GLSL lexer; expect minor differences in token classification.
- **Render preview** in the prototype is a static SVG approximation of `shader-uv-mango.frag`'s output, not a live shader. The real app obviously runs the actual GLSL.
- **Node thumbnails** are SVG placeholders. In the app these are live `moderngl.Texture`s drawn via `add_image` (see SPEC.md §6.1).
- **Click handlers** in the prototype are intentionally narrow: tab switch, modal open/close, Tweaks panel swap, and the demo error-state toggle. Drag-resize, drag-reorder, etc are not wired — the prototype is for visual and structural review, not behavior.

## Contact

If something in here is ambiguous, the designer would rather you ask than guess. Drop questions inline in `SPEC.md` review comments or in chat.
