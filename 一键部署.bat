@echo off
setlocal enabledelayedexpansion
:: ============================================================
::  Watermark Remover v11.0 - One-click Deploy
::  Copy all files to target dir and clean cache
:: ============================================================
title Watermark Remover - Deploy v11.0
cd /d "%~dp0"

set "SOURCE_DIR=%~dp0"
set "TARGET_DIR=D:\AI\watermark_remover"

echo ============================================================
echo   Watermark Remover v11.0 - Deploy
echo ============================================================
echo.
echo   Source: %SOURCE_DIR%
echo   Target: %TARGET_DIR%
echo.
echo ============================================================

:: ============================================================
:: 1. Check target directory exists
:: ============================================================
if not exist "%TARGET_DIR%" (
    echo [ERROR] Target directory not found: %TARGET_DIR%
    echo Please make sure the project is installed at D:\AI\watermark_remover
    pause
    exit /b 1
)

:: ============================================================
:: 2. Copy core files
:: ============================================================
echo [1] Copying core files...

echo   api_server.py v11.0 ...
copy /Y "%SOURCE_DIR%core\api_server.py" "%TARGET_DIR%\core\api_server.py" >nul 2>&1
if errorlevel 1 (
    echo   [FAIL] Failed to copy api_server.py
) else (
    echo   [OK] api_server.py copied
)

echo   engine.py v11.0 ...
copy /Y "%SOURCE_DIR%core\engine.py" "%TARGET_DIR%\core\engine.py" >nul 2>&1
if errorlevel 1 (
    echo   [FAIL] Failed to copy engine.py
) else (
    echo   [OK] engine.py copied
)

echo   detector.py v11.0 (YOLO+U2Net+CV) ...
copy /Y "%SOURCE_DIR%core\detector.py" "%TARGET_DIR%\core\detector.py" >nul 2>&1
if errorlevel 1 (
    echo   [FAIL] Failed to copy detector.py
) else (
    echo   [OK] detector.py copied
)

echo   inpainter.py v11.0 ...
copy /Y "%SOURCE_DIR%core\inpainter.py" "%TARGET_DIR%\core\inpainter.py" >nul 2>&1
if errorlevel 1 (
    echo   [FAIL] Failed to copy inpainter.py
) else (
    echo   [OK] inpainter.py copied
)

echo   segmenter.py v11.0 ...
copy /Y "%SOURCE_DIR%core\segmenter.py" "%TARGET_DIR%\core\segmenter.py" >nul 2>&1
if errorlevel 1 (
    echo   [FAIL] Failed to copy segmenter.py
) else (
    echo   [OK] segmenter.py copied
)

echo   __init__.py ...
if not exist "%TARGET_DIR%\core\__init__.py" (
    echo. > "%TARGET_DIR%\core\__init__.py"
    echo   [OK] __init__.py created
) else (
    echo   [OK] __init__.py already exists
)

echo.
echo [2] Copying training scripts...

echo   train_yolo.py ...
copy /Y "%SOURCE_DIR%train_yolo.py" "%TARGET_DIR%\train_yolo.py" >nul 2>&1
if errorlevel 1 (
    echo   [FAIL] Failed to copy train_yolo.py
) else (
    echo   [OK] train_yolo.py copied
)

echo.

:: ============================================================
:: 3. Copy all .bat scripts from source to target
:: ============================================================
echo [3] Copying scripts...
for %%f in ("%SOURCE_DIR%*.bat") do (
    echo   %%~nxf ...
    copy /Y "%%f" "%TARGET_DIR%\%%~nxf" >nul 2>&1
    if errorlevel 1 (
        echo   [FAIL] Failed to copy %%~nxf
    ) else (
        echo   [OK] %%~nxf copied
    )
)

echo.

:: ============================================================
:: 4. Clean all __pycache__ directories
:: ============================================================
echo [4] Cleaning __pycache__ directories...
for /d /r "%TARGET_DIR%" %%d in (__pycache__) do (
    if exist "%%d" (
        echo   Removing: %%d
        rmdir /s /q "%%d" 2>nul
    )
)
echo   [OK] Cache cleaned
echo.

:: ============================================================
:: 5. Ensure required directories exist
:: ============================================================
echo [5] Ensuring directories exist...
for %%d in (logs output uploads cache models) do (
    if not exist "%TARGET_DIR%\%%d" (
        mkdir "%TARGET_DIR%\%%d" 2>nul
        echo   Created: %%d
    )
)
echo   [OK] Directories ready
echo.

:: ============================================================
:: 6. Done
:: ============================================================
echo ============================================================
echo   [OK] Deployment complete!
echo ============================================================
echo.
echo   Core modules deployed:
echo     - core\api_server.py (v11.0)
echo     - core\engine.py (v11.0) - orchestration + KCF tracking
echo     - core\detector.py (v11.0) - YOLO + U2-Net + CV（默认关闭）
echo     - core\inpainter.py (v11.0) - LaMa + Poisson blending
echo     - core\segmenter.py (v11.0) - SAM2 + GrabCut + 边缘感知羽化
echo.
echo   Training scripts:
echo     - train_yolo.py - YOLO training + synthetic data gen
echo.
echo   Next steps:
echo     1. Run install-deps .bat to install missing dependencies
echo     2. Run debug-mode .bat to start the server
echo     3. (Optional) Run: python train_yolo.py to train YOLO
echo ============================================================
echo.
pause