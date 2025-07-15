@echo off
echo ShaderBox Launcher
echo.

REM Check if uv is installed
where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo uv is not installed. Installing uv...
    echo.

    REM Download and install uv
    powershell -Command "irm https://astral.sh/uv/install.ps1 | iex"

    REM Add uv to PATH for current session
    set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"

    REM Check if uv installation was successful
    where uv >nul 2>&1
    if %errorlevel% neq 0 (
        echo Failed to install uv. Please install it manually from https://docs.astral.sh/uv/
        pause
        exit /b 1
    )

    echo uv installed successfully!
    echo.
)

REM Check if dependencies are installed
if not exist ".venv" (
    echo Installing Python dependencies...
    uv sync

    if %errorlevel% neq 0 (
        echo Failed to install dependencies.
        pause
        exit /b 1
    )

    echo Dependencies installed successfully!
    echo.
)

echo Starting ShaderBox...
uv run python ./shaderbox/ui.py
