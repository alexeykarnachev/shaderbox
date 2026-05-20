@echo off
setlocal
echo ShaderBox Launcher
echo.

REM Check if uv is installed
where uv >nul 2>&1
if %errorlevel% neq 0 (
    echo uv is not installed. Installing uv...
    echo.

    powershell -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if %errorlevel% neq 0 goto :uv_install_failed

    REM Installer puts uv in %USERPROFILE%\.local\bin
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"

    where uv >nul 2>&1
    if %errorlevel% neq 0 goto :uv_path_failed

    echo uv installed successfully!
    echo.
)

REM Check if dependencies are installed
if not exist ".venv" (
    echo First run: downloading Python dependencies ^(several hundred MB^).
    echo This happens once and may take a few minutes -- please wait...
    echo.
    uv sync
    if %errorlevel% neq 0 goto :sync_failed

    echo Dependencies installed successfully!
    echo.
)

echo Starting ShaderBox...
uv run python ./shaderbox/ui.py
if %errorlevel% neq 0 goto :app_failed

endlocal
exit /b 0

:uv_install_failed
echo.
echo ERROR: uv install failed. Install manually from https://docs.astral.sh/uv/
echo.
pause
exit /b 1

:uv_path_failed
echo.
echo ERROR: uv installed but not on PATH. Close this window, reopen it, and run run.bat again.
echo.
pause
exit /b 1

:sync_failed
echo.
echo ERROR: dependency install failed. See the messages above; check your internet connection and disk space.
echo.
pause
exit /b 1

:app_failed
echo.
echo ERROR: ShaderBox exited with an error. See the traceback above.
echo.
pause
exit /b 1
