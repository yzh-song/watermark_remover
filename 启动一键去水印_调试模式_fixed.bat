@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
:: ============================================================
::  调试模式 v3.4 - 捕获所有错误信息
::  如果正常启动闪退，请用此脚本查看错误
:: ============================================================
cd /d "%~dp0"

:: 设置环境变量
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set PYTHONDONTWRITEBYTECODE=1

:: 日志文件
set "LOG_FILE=%~dp0_debug_launch.log"
:: 清空旧日志
break > "%LOG_FILE%"

echo ============================================================ >> "%LOG_FILE%"
echo Debug Launch Log - %date% %time% >> "%LOG_FILE%"
echo ============================================================ >> "%LOG_FILE%"

:: 1. 定位 Python
echo. >> "%LOG_FILE%"
echo [1] Checking Python... >> "%LOG_FILE%"
set "PYTHON_EXE="
if exist "C:\Users\qsong\AppData\Roaming\aipy-pro\venv\Scripts\python.exe" (
    set "PYTHON_EXE=C:\Users\qsong\AppData\Roaming\aipy-pro\venv\Scripts\python.exe"
    echo Found: AiPy venv Python >> "%LOG_FILE%"
) else if exist "%~dp0venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
    echo Found: Local venv Python >> "%LOG_FILE%"
) else (
    where python >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_EXE=python"
        echo Found: System Python >> "%LOG_FILE%"
    ) else (
        echo [ERROR] Python not found! >> "%LOG_FILE%"
        echo.
        echo [错误] 找不到 Python！
        echo 请安装 Python 3.8 或更高版本
        echo.
        pause
        exit /b 1
    )
)
echo Python: >> "%LOG_FILE%"
"%PYTHON_EXE%" --version >> "%LOG_FILE%" 2>&1

:: 2. 检查依赖（包括 flask_cors）
echo. >> "%LOG_FILE%"
echo [2] Checking dependencies... >> "%LOG_FILE%"
"%PYTHON_EXE%" -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available())" >> "%LOG_FILE%" 2>&1
"%PYTHON_EXE%" -c "import flask; print('flask:', flask.__version__)" >> "%LOG_FILE%" 2>&1
"%PYTHON_EXE%" -c "import flask_cors; print('flask_cors: OK')" >> "%LOG_FILE%" 2>&1
"%PYTHON_EXE%" -c "import cv2; print('cv2:', cv2.__version__)" >> "%LOG_FILE%" 2>&1
"%PYTHON_EXE%" -c "from simple_lama_inpainting import SimpleLama; print('simple_lama: OK')" >> "%LOG_FILE%" 2>&1
"%PYTHON_EXE%" -c "import numpy; print('numpy:', numpy.__version__)" >> "%LOG_FILE%" 2>&1
"%PYTHON_EXE%" -c "import PIL; print('Pillow:', PIL.__version__)" >> "%LOG_FILE%" 2>&1

:: 3. 清理端口占用
echo. >> "%LOG_FILE%"
echo [3] Cleaning port 5000... >> "%LOG_FILE%"
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5000" ^| findstr "LISTENING"') do (
    echo Killing old server PID=%%a >> "%LOG_FILE%"
    taskkill /F /PID %%a >> "%LOG_FILE%" 2>&1
)
timeout /t 1 /nobreak >nul

:: 4. 启动服务器
echo. >> "%LOG_FILE%"
echo [4] Starting server... >> "%LOG_FILE%"
echo. >> "%LOG_FILE%"

cls
echo.
echo  ============================================================
echo     水印清除工具 v3.4 - 调试模式
echo  ============================================================
echo.
echo  正在加载 AI 引擎，请稍候...
echo.
echo  启动后浏览器会自动打开： http://localhost:5000
echo  关闭此窗口即停止服务器
echo.
echo  调试日志保存到： %LOG_FILE%
echo  ============================================================
echo.

:: 启动服务器（控制台输出 + 同时写入日志）
"%PYTHON_EXE%" -B -m core.api_server 2>&1

:: 记录退出
set EXIT_CODE=%errorlevel%
echo. >> "%LOG_FILE%"
echo [5] Server exited with code: %EXIT_CODE% >> "%LOG_FILE%"
echo ============================================================ >> "%LOG_FILE%"
echo End of log >> "%LOG_FILE%"

echo.
echo 服务器已停止（退出码: %EXIT_CODE%）。
echo 调试日志： %LOG_FILE%
echo.
pause