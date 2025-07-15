# ShaderBox - Installation and Usage Guide

Welcome to ShaderBox! This guide will help you run the application.

## Quick Start

### Windows Users

Simply double-click **run.bat** to start ShaderBox. The script will automatically:
- Install uv (Python package manager) if needed
- Install all required dependencies on first run
- Launch the application

### Linux Users

Run **./run.sh** to start ShaderBox. The script will automatically:
- Install uv (Python package manager) if needed
- Install all required dependencies on first run
- Launch the application

## What's Included

- **shaderbox/** - Main application source code
- **pyproject.toml** - Project configuration
- **uv.lock** - Dependency lock file
- **run.bat** (Windows) / **run.sh** (Linux) - Application launcher
- **README.md** - This guide

## Requirements

- **Python 3.12+** (will be installed automatically via uv)
- **Internet connection** (for initial setup only)

## How It Works

The run script will:
1. Check if uv is already installed
2. If not, download and install uv from https://astral.sh/uv/
3. Check if dependencies are installed (looks for .venv directory)
4. If not, run `uv sync` to install Python 3.12+ and all required dependencies
5. Launch ShaderBox with `uv run python ./shaderbox/ui.py`

## Troubleshooting

### Windows

- If you get a security warning, click "More info" → "Run anyway"
- If installation fails, try running as administrator
- Make sure your antivirus isn't blocking the installation

### Linux

- If you get permission errors, make sure the script is executable:
  ```bash
  chmod +x run.sh
  ```
- If uv installation fails, you may need to restart your terminal or run:
  ```bash
  source ~/.bashrc
  ```

### Common Issues

- **"uv: command not found"** - Restart your terminal and try again
- **Network errors** - Check your internet connection
- **Permission errors** - Make sure you have write permissions in the directory

## Manual Installation

If the automated script doesn't work, you can install manually:

1. Install uv: https://docs.astral.sh/uv/
2. Run: `uv sync`
3. Run: `uv run python ./shaderbox/ui.py`

## Support

If you encounter any issues, please check:
- Your internet connection
- Antivirus/firewall settings
- Available disk space

For more help, visit the project repository or contact the developer.

---

*This distribution was created with the automated build system. The application will run in a Python virtual environment managed by uv.*
