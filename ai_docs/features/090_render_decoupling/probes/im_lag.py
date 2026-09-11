# Run: XMODIFIERS=@im=ibus FRAME_S=0.100 uv run --with python-xlib python ai_docs/features/090_render_decoupling/probes/im_lag.py
# Holds a synthetic `j` via XTest for 1 s on a bare glfw window polling every FRAME_S seconds,
# releases it, and counts the char events that arrive after the release.
import os, sys, time
import glfw
from Xlib import display, X
from Xlib.ext import xtest

if not glfw.init():
    sys.exit("glfw init failed")
glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
win = glfw.create_window(320, 200, "im_lag", None, None)
glfw.make_context_current(win)
events = []
def key_cb(w, key, sc, action, mods):
    events.append((glfw.get_time(), "key", action))
def char_cb(w, cp):
    events.append((glfw.get_time(), "char", chr(cp)))
glfw.set_key_callback(win, key_cb)
glfw.set_char_callback(win, char_cb)

d = display.Display()
xid = glfw.get_x11_window(win)
for _ in range(30):
    glfw.poll_events(); time.sleep(0.02)
xw = d.create_resource_object("window", xid)
xw.set_input_focus(X.RevertToParent, X.CurrentTime); d.sync()
for _ in range(20):
    glfw.poll_events(); time.sleep(0.02)
kc = d.keysym_to_keycode(ord("j"))
frame = float(os.environ.get("FRAME_S", "0.016"))

xtest.fake_input(d, X.KeyPress, kc); d.sync()
t_press = glfw.get_time()
end = t_press + 1.0
while glfw.get_time() < end:
    glfw.poll_events(); time.sleep(frame)
xtest.fake_input(d, X.KeyRelease, kc); d.sync()
t_release = glfw.get_time()
end = t_release + 1.0
while glfw.get_time() < end:
    glfw.poll_events(); time.sleep(frame)

chars = [t for t, k, v in events if k == "char"]
late = [t - t_release for t in chars if t > t_release]
releases = [t - t_release for t, k, v in events if k == "key" and v == glfw.RELEASE]
print(f"XMODIFIERS={os.environ.get('XMODIFIERS')!r} frame={frame*1000:.0f}ms: "
      f"{len(chars)} chars total, {len(late)} after release, "
      f"last char {max(late)*1000:.0f} ms after release" if late else
      f"XMODIFIERS={os.environ.get('XMODIFIERS')!r} frame={frame*1000:.0f}ms: {len(chars)} chars total, 0 after release")
print("  glfw RELEASE seen at +%s ms" % ", ".join(f"{r*1000:.0f}" for r in releases))
glfw.terminate()
