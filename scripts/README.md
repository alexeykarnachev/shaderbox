# ShaderBox

A real-time GLSL fragment-shader playground. Write a shader in the built-in editor;
ShaderBox reads its uniforms and auto-builds the sliders, color pickers, and inputs to
drive them. It hot-reloads as you type, an optional AI copilot edits your shaders in
chat, and it exports to images, video, Telegram sticker sets, or YouTube.

## Running it

### Before you start — what the first run does

The **first** launch sets up a private Python environment for ShaderBox:

- It installs `uv` (a Python tool, from astral.sh) if you don't already have it.
- It downloads about **500 MB** of dependencies, including a Python runtime.
- This needs an **internet connection** and roughly **1 GB of free disk**.
- It happens **once**. Later launches start in a couple of seconds.

Nothing is installed system-wide — everything lands in a local `.venv` folder next to
this file. Delete the folder and it's gone.

### Windows

Double-click **run.bat**.

The first time, Windows may show a blue **"Windows protected your PC"** screen — this is
SmartScreen reacting to an unsigned script, not a virus warning. Click **More info**, then
**Run anyway**.

### Linux

Open a terminal in this folder and run:

```
./run.sh
```

(Double-clicking from a file manager often won't work — it may open the script in a text
editor or refuse for lack of the execute bit. If `./run.sh` says "Permission denied", run
`bash run.sh` instead.)

ShaderBox needs the system OpenGL + GLFW libraries. If launch fails with a GL error,
install them:

- Debian/Ubuntu: `sudo apt install libgl1 libglfw3`
- Fedora: `sudo dnf install mesa-libGL glfw`
- Arch: `sudo pacman -S libglvnd glfw`

### If something goes wrong

The launcher pauses on errors so you can read the message. The most common first-run
failures are **no internet** or **not enough free disk**. A full log is written to:

- Windows: `%LOCALAPPDATA%\shaderbox\logs`
- Linux: `~/.local/share/shaderbox/logs`

## Your first shader in 60 seconds

ShaderBox opens on a starter shader ("UV Mango") rendering live.

1. The shader's source is in the **left editor pane**. Change a line.
2. Press **Ctrl+S**. The render updates instantly — that's the hot-reload.
3. The controls on the right are generated automatically from your shader's `uniform`
   declarations. Drag a slider and watch the image respond.
4. Press **Ctrl+N** to start a new shader from a template.
5. Click **Open dir** to open the shader's folder in your file manager — edit the
   `.frag.glsl` in your own editor if you prefer; it hot-reloads on save.

## The AI copilot (optional)

Press **Ctrl+J** to open the copilot chat. It's an LLM agent that reads and edits your
shaders, sees compile errors the moment they happen, creates shaders, tweaks uniforms,
renders previews, and can publish for you — every risky action (delete, render, publish)
waits for your confirmation, and any turn can be reverted with one click.

It needs an [OpenRouter](https://openrouter.ai) API key: open **Settings** (Alt+S), paste
the key, pick a tool-calling-capable model (the default works). You pay OpenRouter
directly for what you use; without a key the rest of ShaderBox works as usual.

## Keyboard shortcuts

All of these are rebindable in Settings; press **Ctrl+/** in the app for the full live
cheatsheet.

| Key | Action |
|-----|--------|
| Ctrl+S | Save + hot-reload the current shader |
| Ctrl+N | New shader (pick a template) |
| Ctrl+O | Open a project |
| Ctrl+D | Delete the current shader |
| Ctrl+P | Shader library picker |
| Ctrl+Shift+P | Command palette |
| Ctrl+J | Toggle the AI copilot chat |
| Ctrl+/ | Keyboard cheatsheet |
| Ctrl+Tab | Cycle focus region (editor / grid / panel) |
| Ctrl+1 / 2 / 3 | Node / Render / Share tab |
| F8 | Jump to next compile error |
| Ctrl+Q | Quit |
| Alt+S | Settings |
| Esc | Unfocus the editor / close a popup |
| Ctrl+scroll | Resize the editor font (hover the editor) |

## Manual install

If the launcher doesn't work, set it up yourself:

- Install uv: https://docs.astral.sh/uv/
- `uv sync`
- `uv run python ./shaderbox/ui.py`
