# ShaderBox

A real-time GLSL fragment-shader playground. Write a shader in the built-in editor;
ShaderBox reads its uniforms and auto-builds the sliders, color pickers, and inputs to
drive them. Chain shaders into multi-pass graphs, drive them from Python (or your own
C/Odin library), let an optional AI copilot edit them in chat, and export to images,
video, Telegram sticker sets, or YouTube.

## Install and run

1. **Unzip the download.** Windows: right-click the zip, then **Extract All**. Linux: use your
   file manager's **Extract**, or `unzip shaderbox-linux.zip` in a terminal (no `unzip`?
   `python3 -m zipfile -e shaderbox-linux.zip .` does the same).
2. **Start it** from the unzipped folder:
   - Windows: double-click **run.bat**.
   - Linux: open a terminal in the folder and run `./run.sh`.

That is all. The first start downloads what ShaderBox needs (about 500 MB, a few minutes, once)
and then opens the app. It needs an internet connection and about 1 GB of free disk. Later
starts take a few seconds.

A console window shows the progress and stays open while ShaderBox runs: that is normal, and
closing it closes the app.

**Windows** may show a blue **"Windows protected your PC"** screen the first time. That is
SmartScreen reacting to an unsigned script; click **More info**, then **Run anyway**.

**Linux** needs the system OpenGL library, which desktops normally have. If the launcher says
it is missing: `sudo apt install libgl1` (Debian/Ubuntu), `sudo dnf install mesa-libGL`
(Fedora) or `sudo pacman -S libglvnd` (Arch). If `./run.sh` says "Permission denied", run
`bash run.sh`.

### What gets installed where

- ShaderBox's libraries: a `.venv` folder inside the unzipped folder.
- `uv`, the tool that installs them: `~/.local/bin` (Windows: `%USERPROFILE%\.local\bin`),
  only if you don't have it already. It also keeps a Python runtime in its own data folder.

To uninstall, delete the unzipped folder (and `uv`, if you don't use it otherwise).

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
4. Press **Ctrl+Shift+N** to start a new document from a template.
   Press **Alt+E** to browse the built-in examples and open an editable copy of one.
5. Click **Open folder** to open the document's folder in your file manager — edit the
   `.frag.glsl` in your own editor if you prefer; it hot-reloads on save.
6. Press **Alt+G** on a multi-pass example (Radiance Cascades) to see its passes as a graph.

## The AI copilot (optional)

Press **Alt+J** to open the copilot chat. It's an LLM agent that reads and edits your
shaders, sees compile errors the moment they happen, creates shaders, tweaks uniforms,
renders previews, and can publish for you — every risky action (delete, render, publish)
waits for your confirmation, and any turn can be reverted with one click.

It needs an [OpenRouter](https://openrouter.ai) API key: open **Settings** (Alt+S), paste
the key, pick a tool-calling-capable model (the default works). You pay OpenRouter
directly for what you use; without a key the rest of ShaderBox works as usual.

## Keyboard shortcuts

All of these are rebindable in Settings; press **F1** for the in-app help, or **Alt+/** for the live
cheatsheet.

| Key | Action |
|-----|--------|
| Ctrl+S | Save + hot-reload the current shader |
| Ctrl+Shift+N | New document |
| Alt+E | Examples browser |
| Alt+O | Projects (new / duplicate / switch / delete) |
| Alt+S | Settings |
| Alt+G | Open the document's pass graph |
| Alt+R | Open the document's Python script |
| F5 | Play / stop the script |
| Alt+A | Add a pass |
| Alt+L | Shader library |
| Ctrl+Shift+P | Command palette |
| Ctrl+1 / 2 / 3 / 4 | Document / Uniforms / Render / Share panel |
| Alt+J | Toggle the AI copilot chat |
| F8 | Jump to the next compile error |
| Ctrl+Tab | Next code tab |
| Alt+D | Delete the current document |
| F1 | Help |
| Alt+/ | Keyboard cheatsheet |
| Ctrl+Q | Quit |
| Ctrl+scroll | Resize the editor font (hover the editor) |

## Manual install

If the launcher doesn't work, set it up yourself:

- Install uv: https://docs.astral.sh/uv/
- `uv sync --no-dev`
- `uv run --no-dev python ./shaderbox/ui.py`

On Linux, `run.sh` also sets a few graphics variables before that last command; copy them
from it if you start ShaderBox yourself.
