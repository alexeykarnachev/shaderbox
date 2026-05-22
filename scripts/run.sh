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

# ShaderBox needs system OpenGL + GLFW libs (the pip glfw wheel does not bundle them).
libs="$(ldconfig -p 2>/dev/null || true)"
if ! grep -q "libGL.so" <<<"$libs" || ! grep -q "libglfw" <<<"$libs"; then
    echo "WARNING: system OpenGL / GLFW libraries (libGL / libglfw) not found."
    echo "  If launch fails with a GL error, install them:"
    echo "  Debian/Ubuntu: sudo apt install libgl1 libglfw3"
    echo "  Fedora:        sudo dnf install mesa-libGL glfw"
    echo "  Arch:          sudo pacman -S libglvnd glfw"
    echo
fi

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
    uv sync || fail "Dependency install failed. See the messages above; check your internet connection and disk space."

    echo "Dependencies installed successfully!"
    echo
fi

echo "Starting ShaderBox..."
uv run python ./shaderbox/ui.py || fail "ShaderBox exited with an error. See the traceback above."
