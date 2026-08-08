@echo off
setlocal enabledelayedexpansion
:: ============================================================
::  Watermark Remover - Auto Install Dependencies
::  Auto-detect and install all required Python packages
:: ============================================================
title Watermark Remover - Install Dependencies
cd /d "%~dp0"

echo ============================================================
echo   Watermark Remover - Auto Install Dependencies
echo ============================================================
echo.

:: ============================================================
:: 1. Locate Python
:: ============================================================
set "PYTHON_EXE="
set "PIP_EXE="

if exist "C:\Users\qsong\AppData\Roaming\aipy-pro\venv\Scripts\python.exe" (
    set "PYTHON_EXE=C:\Users\qsong\AppData\Roaming\aipy-pro\venv\Scripts\python.exe"
    set "PIP_EXE=C:\Users\qsong\AppData\Roaming\aipy-pro\venv\Scripts\pip.exe"
    goto :found
)
if exist "%~dp0venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
    set "PIP_EXE=%~dp0venv\Scripts\pip.exe"
    goto :found
)
for %%p in (python python3 py) do (
    where %%p >nul 2>&1
    if !errorlevel! equ 0 (
        for /f "delims=" %%f in ('where %%p 2^>nul') do (
            set "PYTHON_EXE=%%f"
            set "PIP_EXE=%%~dpfScripts\pip.exe"
            if not exist "!PIP_EXE!" set "PIP_EXE=%%~dpfpip3.exe"
            if not exist "!PIP_EXE!" set "PIP_EXE=%%~dpfpip.exe"
            goto :found
        )
    )
)

echo [ERROR] Python not found!
echo Please install Python 3.8+ from https://www.python.org/
echo.
pause
exit /b 1

:found
echo [1] Python: %PYTHON_EXE%
echo [1] Pip:    %PIP_EXE%
echo.

:: Verify pip is available
"%PYTHON_EXE%" -m pip --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pip is not available!
    echo Installing pip...
    "%PYTHON_EXE%" -m ensurepip --upgrade
    if errorlevel 1 (
        echo [FAIL] Cannot install pip. Please install manually.
        pause
        exit /b 1
    )
)

:: ============================================================
:: 2. Upgrade pip
:: ============================================================
echo [2] Upgrading pip...
"%PYTHON_EXE%" -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
echo.

:: ============================================================
:: 3. Install packages
::    IMPORTANT: simple-lama-inpainting wants numpy<2.0 pillow<10
::    but we need numpy>=2.0 pillow>=12.2 for other packages.
::    Strategy: install lama FIRST (gets its deps), then
::    force-upgrade numpy/pillow last (they are backward-compatible).
:: ============================================================
echo [3] Installing dependencies...
echo.

:: --- simple-lama-inpainting (install FIRST, it pulls numpy 1.x + pillow 9.x) ---
echo [3.1] Installing simple-lama-inpainting (base deps)...
"%PYTHON_EXE%" -m pip install --upgrade simple-lama-inpainting
if errorlevel 1 (
    echo [WARN] Default source failed, trying Tsinghua mirror...
    "%PYTHON_EXE%" -m pip install --upgrade simple-lama-inpainting -i https://pypi.tuna.tsinghua.edu.cn/simple
)
echo.

:: --- opencv-python ---
echo [3.2] Installing opencv-python...
"%PYTHON_EXE%" -m pip install --upgrade opencv-python
if errorlevel 1 (
    "%PYTHON_EXE%" -m pip install --upgrade opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
)
echo.

:: --- flask ---
echo [3.3] Installing flask...
"%PYTHON_EXE%" -m pip install --upgrade flask
if errorlevel 1 (
    "%PYTHON_EXE%" -m pip install --upgrade flask -i https://pypi.tuna.tsinghua.edu.cn/simple
)
echo.

:: --- flask-cors ---
echo [3.4] Installing flask-cors...
"%PYTHON_EXE%" -m pip install --upgrade flask-cors
if errorlevel 1 (
    "%PYTHON_EXE%" -m pip install --upgrade flask-cors -i https://pypi.tuna.tsinghua.edu.cn/simple
)
echo.

:: --- torch (large, takes time) ---
echo [3.5] Installing PyTorch (this may take a while)...
echo   Checking if PyTorch is already installed...
"%PYTHON_EXE%" -c "import torch; print('torch:', torch.__version__)" 2>nul
if errorlevel 1 (
    echo   Installing PyTorch CPU version...
    "%PYTHON_EXE%" -m pip install torch torchvision torchaudio
    if errorlevel 1 (
        echo   [WARN] Default source failed, trying Tsinghua mirror...
        "%PYTHON_EXE%" -m pip install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
    )
    if errorlevel 1 (
        echo   [WARN] Full torch failed, trying CPU-only version...
        "%PYTHON_EXE%" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    )
) else (
    echo   PyTorch already installed, skipping.
)
echo.

:: --- numpy (force-upgrade LAST to override lama's old version) ---
::    Must be >=2.0.0 for scipy/tifffile compatibility
echo [3.6] Force-upgrading numpy to >=2.0.0 (overrides lama constraint)...
"%PYTHON_EXE%" -m pip install --upgrade "numpy>=2.0.0"
if errorlevel 1 (
    echo [WARN] Default source failed, trying Tsinghua mirror...
    "%PYTHON_EXE%" -m pip install --upgrade "numpy>=2.0.0" -i https://pypi.tuna.tsinghua.edu.cn/simple
)
echo.

:: --- Pillow (force-upgrade LAST to override lama's old version) ---
::    Must be >=12.2.0 for pdfplumber/scikit-image compatibility
echo [3.7] Force-upgrading Pillow to >=12.2.0 (overrides lama constraint)...
"%PYTHON_EXE%" -m pip install --upgrade "Pillow>=12.2.0"
if errorlevel 1 (
    echo [WARN] Default source failed, trying Tsinghua mirror...
    "%PYTHON_EXE%" -m pip install --upgrade "Pillow>=12.2.0" -i https://pypi.tuna.tsinghua.edu.cn/simple
)
echo.

:: ============================================================
:: 4. Verify installations
:: ============================================================
echo ============================================================
echo [4] Verifying installations...
echo ============================================================
echo.

set "ALL_OK=1"

echo --- torch ---
"%PYTHON_EXE%" -c "import torch; print('  torch:', torch.__version__); print('  cuda:', torch.cuda.is_available())" 2>&1
if errorlevel 1 set "ALL_OK=0"

echo --- flask ---
"%PYTHON_EXE%" -c "import flask; print('  flask:', flask.__version__)" 2>&1
if errorlevel 1 set "ALL_OK=0"

echo --- flask_cors ---
"%PYTHON_EXE%" -c "import flask_cors; print('  flask_cors: OK')" 2>&1
if errorlevel 1 set "ALL_OK=0"

echo --- opencv ---
"%PYTHON_EXE%" -c "import cv2; print('  cv2:', cv2.__version__)" 2>&1
if errorlevel 1 set "ALL_OK=0"

echo --- numpy ---
"%PYTHON_EXE%" -c "import numpy; print('  numpy:', numpy.__version__)" 2>&1
if errorlevel 1 set "ALL_OK=0"

echo --- Pillow ---
"%PYTHON_EXE%" -c "import PIL; print('  Pillow:', PIL.__version__)" 2>&1
if errorlevel 1 set "ALL_OK=0"

echo --- simple_lama ---
"%PYTHON_EXE%" -c "from simple_lama_inpainting import SimpleLama; print('  simple_lama: OK')" 2>&1
if errorlevel 1 set "ALL_OK=0"

echo.
echo ============================================================
if "%ALL_OK%"=="1" (
    echo   [OK] All dependencies installed successfully!
    echo   You can now run the debug mode .bat to start the server.
) else (
    echo   [WARN] Some dependencies failed to install.
    echo   Please check the output above for error details.
    echo   You may need to install them manually:
    echo     pip install numpy Pillow opencv-python flask flask-cors
    echo     pip install torch torchvision torchaudio
    echo     pip install simple-lama-inpainting
)
echo ============================================================
echo.
pause