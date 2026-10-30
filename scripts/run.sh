#!/bin/bash
set -e

# Run from the script's own directory, so double-clicking from a file manager
# (cwd = $HOME) still finds ./shaderbox and the project.
cd "$(dirname "$(readlink -f "$0")")"

echo "ShaderBox Launcher"
echo

fail() {
    echo
    echo "ERROR: $1"
    echo
    read -rp "Press Enter to close..."
    exit 1
}

# The only system library ShaderBox needs is OpenGL; the glfw wheel carries its own libglfw.
if ! ldconfig -p 2>/dev/null | grep -q "libGL.so.1"; then
    echo "WARNING: the system OpenGL library (libGL.so.1) was not found. Install it:"
    echo "  Debian/Ubuntu: sudo apt install libgl1"
    echo "  Fedora:        sudo dnf install mesa-libGL"
    echo "  Arch:          sudo pacman -S libglvnd"
    echo
fi
# glfw and PyOpenGL must agree on one window system, and they decide differently: PyOpenGL
# picks EGL whenever WAYLAND_DISPLAY is set, glfw by XDG_SESSION_TYPE. Both are pinned to
# X11, which a Wayland desktop serves through XWayland.
if [ -n "${DISPLAY:-}" ]; then
    export PYGLFW_LIBRARY_VARIANT="${PYGLFW_LIBRARY_VARIANT:-x11}"
    export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-glx}"
fi
# The shaders are GLSL 4.60. Mesa's software renderer (VMs, WSL, GPUs without a driver)
# reports 4.5 and refuses them; these overrides let it compile them. A driver that
# already offers 4.6 is unaffected, and non-Mesa drivers ignore them.
export MESA_GL_VERSION_OVERRIDE="${MESA_GL_VERSION_OVERRIDE:-4.6}"
export MESA_GLSL_VERSION_OVERRIDE="${MESA_GLSL_VERSION_OVERRIDE:-460}"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    echo

    curl -LsSf https://astral.sh/uv/install.sh | sh || fail "uv install failed. Install manually: https://docs.astral.sh/uv/"

    # Installer puts uv in ~/.local/bin
    [ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"
    export PATH="$HOME/.local/bin:$PATH"

    command -v uv &> /dev/null || fail "uv installed but not on PATH. Restart your terminal, then run ./run.sh again."

    echo "uv installed successfully!"
    echo
fi

# Check if dependencies are installed
if [ ! -d ".venv" ]; then
    echo "First run: downloading Python dependencies (~500 MB, incl. a Python runtime)."
    echo "Needs internet + ~1 GB free disk. Happens once; may take a few minutes — please wait..."
    echo
    uv sync --no-dev || fail "Dependency install failed. See the messages above; check your internet connection and disk space."

    echo "Dependencies installed successfully!"
    echo
fi

echo "Starting ShaderBox..."
uv run --no-dev python ./shaderbox/ui.py || fail "ShaderBox exited with an error. See the traceback above."
