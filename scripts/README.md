# ShaderBox - Installation and Usage Guide

## Quick Start

### Windows

Simply double-click **run.bat** to start ShaderBox. The script will automatically:
- Install uv (Python package manager) if needed
- Install all required dependencies on first run
- Launch the application

### Linux

Run **./run.sh** to start ShaderBox. The script will automatically:
- Install uv (Python package manager) if needed
- Install all required dependencies on first run
- Launch the application

## Manual Installation

If the automated script doesn't work, you can install manually:

- Install uv: https://docs.astral.sh/uv/
- Run: `uv sync`
- Run: `uv run python ./shaderbox/ui.py`
