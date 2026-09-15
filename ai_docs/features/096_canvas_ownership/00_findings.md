# 096 — Canvas ownership: the raw ledger

Three canvas bugs landed in one week, each invisible at runtime — no error, no crash, only a
setting that appeared not to work. This feature treats them as one class rather than three
incidents, and this file is the evidence behind it.

**The shared root:** a canvas's configuration (size, dtype, filter, wrap) is decided in several
places, and each place knows about some canvases but not all.

Every row is verified in this session before it is filed. A finding relayed by an agent and not
re-checked against the source does not belong here; two relayed findings were wrong earlier in
the week, which is why the rule exists.

## The three that are already fixed

They are listed for the pattern, not as open work.

| # | Symptom the maintainer saw | Root | Landed |
|---|---|---|---|
| 1 | Toggling `smooth` on `paint` did nothing | `Pass.set_target` reallocated the LIVE canvas; `begin_frame` swapped the old-config history into the live slot the next frame | `0753b03` |
| 2 | Still nothing after 1 | The viewer was in RGB view; `ChannelBlit` allocated its canvas with no filter, and the viewer magnifies the BLIT's texture | `dc742d3` |
| 3 | Changing the resolution left the other pass at the old size until clicked away and back | `set_canvas_size` resampled only the output + histories; `render` fixes others lazily but exempts the output at draw time | `07dbfce` |

Each is the same shape: a canvas that some code path owns and another path forgot.

## Open findings

### F1 — the export's `RENDER_AT_TARGET` branch drops the output pass's whole target config

`document.py` allocates the export target in two sibling branches twelve lines apart. The first
copies `dtype`, `filter` and `wrap` from the output pass's canvas; the second passes none of
them, so the canvas takes the `Canvas` defaults.

`Document.render` hands that canvas straight to the output pass as its draw target, so the
canvas's format IS the pass's format for the whole export. The loss is at WRITE time, inside
the pass's draw — not at readback — so inspecting the output file finds nothing: the readback
tonemaps any float target to 8-bit either way.

Verified here, independently of the agent that reported it. A shader writing `3.0`:

    dest dtype=f1: max R = 255      (clamped to 1.0)
    dest dtype=f2: max R = 3.0      (headroom kept)

So a document whose output pass is `f2` exports with its HDR headroom clamped away through one
branch and intact through the other — the same document, two pictures, depending on which
button was pressed.

Which path takes the lossy branch, verified by search: `FitPolicy.RENDER_AT_TARGET` is used by
`render_shape.py` (the shared shapes) and `exporters/telegram.py`. The Render tab's own export
passes `preset=None` and takes the correct branch. So the branch that ships to users is the
lossy one.

Nothing asserts the export canvas's dtype, filter or wrap, so the gap is unguarded.

**The two defaults disagree, which is what makes this bite the ordinary case rather than an
exotic one.** `TargetConfig.dtype` defaults to `f2`, while `Canvas`'s own `dtype` parameter
defaults to `f1` — verified by reading both. So a pass created through the normal path is `f2`,
and the bare `Canvas(gl, size=...)` at the lossy branch is `f1`: the mismatch is not something a
user has to opt into by choosing an unusual format, it is what every default document gets.

This is the already-solved-twin shape the maintainer's debugging rules name: one of two
branches got the fix, and the missing half is the strongest evidence the class is real.

### F2 — promoting a scaled pass to output strands it at the scaled size

The mirror of bug 3, still open. `render` sizes a non-output pass to
`entry.target.target_size(canvas_size)` and exempts whichever pass is the output. So a pass
with `scale < 1.0` is correctly small while it is off-output, and when the user clicks its tile
to make it the output, nothing resizes it back to full — the exemption now protects the wrong
value, and no later frame corrects it.

Verified here with a two-pass document at 256x256, `helper` at `scale=0.5`:

    helper as scaled non-output: (128, 128)   correct
    helper AFTER promotion:      (128, 128)   document is 256x256
    what the viewer/export reads:(128, 128)

The viewer and every export then read a half-size canvas for a full-size document, and it never
self-corrects. Reachable with one click on a pass tile.

Bug 3 fixed the resize direction; this is the promotion direction. Two directions of one rule,
fixed one at a time, is the signature of a rule with no single home.

### F3 — the sizing rule is spelled out four times, in three different syntactic shapes

`document.py` re-derives "the output keeps full size, everyone else scales" at four sites: the
non-output loop and the feedback loop in `set_canvas_size`, `_seed_feedback`, and `render`'s
lazy fix-up — variously as a `continue`, a ternary and an `if name != output` guard. Bug 3 was
precisely this: the rule existed at one site and was missing at another.

A fifth copy sits in `popups/pass_settings.py`, and it is WRONG: it computes the displayed size
as `canvas * scale` with no output exemption, so an output pass carrying a stored scale displays
a size it does not have. Verified by reading the source: `is_output` reaches the slider's
`begin_disabled` but not the label above it.

### F4 — two sizing verbs with opposite content semantics

`resample_canvas` allocates, blits and releases, so content survives. `Canvas.set_size` releases
then allocates, so content is blanked. `set_canvas_size` uses the first and `render`'s fix-up
uses the second, for what is logically the same operation. Which one a canvas gets depends on
which path reached it first.

### F5 — nothing reconciles a canvas to its graph entry

`PassGraph.with_target` writes the model; `Pass.set_target` writes the live side. The production
funnel calls both, so they agree today by convention rather than by structure. And
`resample_canvas` copies the OLD canvas's dtype/filter/wrap forward, so a resize propagates a
stale format instead of correcting it: there is no path anywhere that re-derives a canvas from
its `TargetConfig`.

## What is NOT a defect

Recorded so a later wave does not re-raise it.

- **The output pass ignoring its own `scale` is intended.** The settings modal disables that
  slider for the output and says "output always full". F2 above is not a case for honouring the
  scale — it is a case for the canvas being resized to FULL on promotion.
- **`ChannelBlit` not carrying dtype or wrap.** The blit canvas is only ever sampled by imgui for
  display. Wrap cannot show at uv 0..1, and the blit's own shader writes an in-range opaque
  value, so dtype cannot show either. The filter was the one observable field and it is fixed.
- **The Share/Telegram previews showing the decoded artifact rather than the pass texture.** That
  file is what gets uploaded; showing anything else would misrepresent it.
- **Tiles that minify.** A `filter` pair covers min and mag both and the tiles hand over the
  pass's own texture, so they are faithful whether or not magnification happens.

### F6 — a self-reading OUTPUT pass exports a frozen picture at every iteration count

The most serious finding, and it ships: exporting a document whose output pass reads its own
previous frame produces a video where the feedback chain never advances. Every exported frame
holds the same accumulated value while the live viewer shows it evolving correctly.

The mechanism is documented in a comment at the iteration loop, which explains that only the
LAST iteration draws into the caller's canvas so the chain can advance by swapping the pass's
own. What the comment does not notice: when the last iteration goes to the external canvas, the
pass's OWN canvas is left unwritten, so there is nothing for the swap to advance.

Verified here with an accumulator shader (`prev + 0.1`), reading the live canvas and the history
at each frame:

    iterations=1, EXPORT: ext=0.1 live=0.0 hist=0.0   -- every frame identical
    iterations=2, EXPORT: ext=0.2 live=0.0 hist=0.1   -- every frame identical
    iterations=1, LIVE  : 0.1, 0.2, 0.2998, 0.3997    -- advances correctly

At N=1 the single draw goes external and `live` is never written. At N=2 iteration 1 fills
`live`, the swap moves it to `hist`, and iteration 2 (the last) goes external — so `live` is
never refilled. The chain is frozen at EVERY N, not only at N=1.

Scope correction, recorded because the first report had it narrower: a swarm agent reported this
as an N=1 defect that resolves at N>=2, having read motion in an N=2 export. That motion comes
from `u_time`, not from feedback. The trace above shows `live` pinned at 0.0 for both counts.

This is invisible in the app: the viewer renders without an external canvas and is correct, so
the defect appears only in the exported file.
