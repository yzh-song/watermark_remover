@echo off
setlocal enabledelayedexpansion
:: ============================================================
::  Watermark Remover v9.0 - Debug Mode
::  Aggressive __pycache__ cleanup + foreground run
::  Fixes: U2-Net detection, LaMa inpainting, video preview, frame capture
:: ============================================================
title Watermark Remover v9.0 - Debug
cd /d "%~dp0"

:: ============================================================
:: 0. Clean all __pycache__ dirs (ensure new code runs)
:: ============================================================
echo [0] Cleaning all __pycache__ directories...
for /d /r "%~dp0" %%d in (__pycache__) do (
    if exist "%%d" (
        echo   Removing: %%d
        rmdir /s /q "%%d" 2>nul
    )
)
echo   Done.
echo.

:: ============================================================
:: 1. Locate Python
:: ============================================================
set "PYTHON_EXE="

if exist "C:\Users\qsong\AppData\Roaming\aipy-pro\venv\Scripts\python.exe" (
    set "PYTHON_EXE=C:\Users\qsong\AppData\Roaming\aipy-pro\venv\Scripts\python.exe"
    goto :found
)
if exist "%~dp0venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
    goto :found
)
for %%p in (python python3 py) do (
    where %%p >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_EXE=%%p"
        goto :found
    )
)

echo ============================================================
echo [ERROR] Python not found!
echo Please install Python 3.8+ and try again.
echo ============================================================
pause
exit /b 1

:found
echo [1] Python: %PYTHON_EXE%

:: ============================================================
:: 2. Detailed dependency check
:: ============================================================
echo.
echo [2] Checking dependencies...
echo   --- torch ---
"%PYTHON_EXE%" -c "import torch; print('  torch:', torch.__version__); print('  cuda:', torch.cuda.is_available())" 2>&1
echo   --- flask ---
"%PYTHON_EXE%" -c "import flask; print('  flask:', flask.__version__)" 2>&1
echo   --- flask_cors ---
"%PYTHON_EXE%" -c "import flask_cors; print('  flask_cors: OK')" 2>&1
echo   --- opencv ---
"%PYTHON_EXE%" -c "import cv2; print('  cv2:', cv2.__version__)" 2>&1
echo   --- simple_lama ---
"%PYTHON_EXE%" -c "from simple_lama_inpainting import SimpleLama; print('  simple_lama: OK')" 2>&1
echo   --- numpy ---
"%PYTHON_EXE%" -c "import numpy; print('  numpy:', numpy.__version__)" 2>&1
echo   --- Pillow ---
"%PYTHON_EXE%" -c "import PIL; print('  Pillow:', PIL.__version__)" 2>&1
echo   Done.
echo.

:: ============================================================
:: 3. Clean port 5000
:: ============================================================
echo [3] Cleaning port 5000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5000" ^| findstr "LISTENING"') do (
    echo   Killing old server PID=%%a
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 1 /nobreak >nul
echo   Done.
echo.

:: ============================================================
:: 4. Set env and start server (foreground, all output visible)
:: ============================================================
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set PYTHONDONTWRITEBYTECODE=1

echo ============================================================
echo   Watermark Remover v9.0 - Debug Mode
echo ============================================================
echo.
echo   Changes in v9.0:
echo     - U2-Net AI watermark detection (primary)
echo     - Multi-strategy CV fallback (4 strategies)
echo     - Enhanced LaMa inpainting quality
echo     - Natural edge feathering blend
echo     - Full video preview (original + result)
echo     - Video frame capture + canvas selection
echo     - Smart mask dilation + feathering
echo.
echo   Starting AI engine (10-30 seconds on first run)...
echo   Browser will open automatically when ready.
echo   Logs: logs\api_server.log, logs\engine.log
echo.
echo   Press Ctrl+C or close this window to stop the server.
echo ============================================================
echo.

"%PYTHON_EXE%" -B -m core.api_server

:: ============================================================
:: 5. Server exit - show status
:: ============================================================
echo.
echo ============================================================
echo Server stopped (exit code: %errorlevel%)
echo.
echo Log files:
echo   api_server.log: logs\api_server.log
echo   engine.log:     logs\engine.log
echo ============================================================
echo.
pause