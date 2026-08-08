@echo off
setlocal enabledelayedexpansion
:: ============================================================
::  Watermark Remover v5.1 - 一键启动
::  极简设计：前台运行，所有输出可见，无复杂重定向
::  修复：torch 模块级导入、__init__.py 缺失、缓存清理
:: ============================================================
title Watermark Remover v5.1
cd /d "%~dp0"

:: ============================================================
:: 0. 清理所有 __pycache__ 缓存（确保新代码生效）
:: ============================================================
for /d /r "%~dp0" %%d in (__pycache__) do (
    if exist "%%d" rmdir /s /q "%%d" 2>nul
)

:: ============================================================
:: 1. 定位 Python
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
echo Python: %PYTHON_EXE%

:: 验证 Python 可执行
"%PYTHON_EXE%" --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python failed to run (exit code: %errorlevel%)
    echo This may indicate a corrupted Python installation.
    pause
    exit /b 1
)

:: ============================================================
:: 2. 快速检查关键依赖
:: ============================================================
echo Checking dependencies...
"%PYTHON_EXE%" -c "import flask; import flask_cors" >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] Missing dependencies! Run:
    echo   pip install flask flask-cors
    echo.
    echo Error details:
    "%PYTHON_EXE%" -c "import flask; import flask_cors" 2>&1
    echo.
    pause
    exit /b 1
)

:: ============================================================
:: 3. 清理端口 5000
:: ============================================================
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5000" ^| findstr "LISTENING"') do (
    echo Killing old server PID=%%a
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 1 /nobreak >nul

:: ============================================================
:: 4. 设置环境并启动服务器（前台运行，所有输出直接可见）
:: ============================================================
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set PYTHONDONTWRITEBYTECODE=1

cls
echo.
echo ============================================================
echo   Watermark Remover v5.1
echo ============================================================
echo.
echo   Starting AI engine (10-30 seconds on first run)...
echo   Browser will open automatically when ready.
echo   Logs: D:\AI\watermark_remover\logs\
echo.
echo   Press Ctrl+C or close this window to stop the server.
echo ============================================================
echo.

"%PYTHON_EXE%" -B -m core.api_server

:: ============================================================
:: 5. 服务器退出 - 显示状态
:: ============================================================
echo.
echo ============================================================
echo Server stopped (exit code: %errorlevel%)
echo.
echo Log files:
echo   api_server.log: D:\AI\watermark_remover\logs\api_server.log
echo   engine.log:     D:\AI\watermark_remover\logs\engine.log
echo ============================================================
echo.
pause