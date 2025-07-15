#!/bin/bash
set -e

echo "ShaderBox Launcher"
echo

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    echo

    # Download and install uv
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add uv to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"

    # Check if uv installation was successful
    if ! command -v uv &> /dev/null; then
        echo "Failed to install uv. Please install it manually from https://docs.astral.sh/uv/"
        echo "You may need to restart your terminal or run: source ~/.bashrc"
        exit 1
    fi

    echo "uv installed successfully!"
    echo
fi

# Check if dependencies are installed
if [ ! -d ".venv" ]; then
    echo "Installing Python dependencies..."
    uv sync

    echo "Dependencies installed successfully!"
    echo
fi

echo "Starting ShaderBox..."
uv run python ./shaderbox/ui.py
