# The X11 input method and the after-release motion

Measured 2026-09-11 on the dev box (X11 display `:1`, GNOME session, `ibus-daemon --xim`
running, `XMODIFIERS=@im=ibus`, glfw 3.4 via pyGLFW 2.10, X auto-repeat 500 ms / 33 per s).
Probe: `../probes/im_lag.py`.

## Method

A bare glfw window with key and char callbacks recording `glfw.get_time()`. XTest fakes a
`j` press, the script polls glfw every `FRAME_S` seconds for one second, XTest fakes the
release, polling continues for another second. Counted: char events whose arrival time is
after the release, and the arrival time of the last one.

## Results

| XMODIFIERS | poll period | chars total | chars after release | last char after release |
|---|---|---|---|---|
| `@im=ibus` | 16 ms | 18 | 1 | 0 ms |
| `@im=ibus` | 40 ms | 18 | 6 | 202 ms |
| `@im=ibus` | 60 ms | 18 | 10 | 543 ms |
| `@im=ibus` | 100 ms | 15 | 10 | 903 ms |
| `@im=none` | 16 ms | 18 | 0 | – |
| `@im=none` | 100 ms | 18 | 3 | 0 ms |

The glfw RELEASE key event arrived at +0 ms in every run: the release takes the plain key
path and is on time even while the chars trail.

Setting `os.environ["XMODIFIERS"] = "@im=none"` inside the process before `glfw.init()`
reproduces the `@im=none` rows (18 chars, last at 0 ms after release at a 100 ms poll).

## Interpretation

With the input method, each key press is forwarded to ibus and comes back as a re-injected
event that the client only sees at its next pump, so throughput is one key per pump. Once
the pump period exceeds the 30 ms repeat interval the backlog builds on the ibus side and
drains one per frame after the key is up. Without the input method a slow pump applies the
whole backlog at the next poll and nothing arrives after the release.

This is a separate mechanism from the frame-time coupling 090 addresses. A UI thread that
holds 60 fps never lets the backlog build, but the protocol is one-per-pump by construction,
and the app cannot tell a late repeat from a fresh keypress, so the spec should still decide
whether the app opts out of XIM. Cost of opting out: no compose / dead keys and no CJK
engines inside the app; XKB layouts (Cyrillic) still work because glfw translates keysyms
itself.

## False trails

- The editor pump (`hotkeys.py::_drain_editor_input`) feeds every queued event each frame;
  it paces nothing.
- The cursor follow (`tabs/code.py::layout_following_cursor`) scrolls immediately, no
  animation.
- imgui's `key_repeat_rate` (`app.py`) only affects `is_key_pressed(repeat=True)` users; the
  editor takes glfw events directly.
- `BreakBinaryOperations`-style pacing in the editor library: the ABI has no time-based call
  (`editor/ffi.py` exposes none).
