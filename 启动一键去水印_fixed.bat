@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
:: ============================================================
::  Watermark Remover v3.5 - 一键启动
::  修复：去掉 /b 解决进程检测Bug（/b 不创建窗口，tasklist检测不到）
:: ============================================================
title 一键去水印 - 启动中...
cd /d "%~dp0"

:: 设置环境变量
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
set PYTHONDONTWRITEBYTECODE=1
set "SERVER_LOG=%TEMP%\watermark_server.log"

cls
echo.
echo  ============================================================
echo     水印清除工具 v3.5
echo  ============================================================
echo.
echo  正在启动，请稍候...
echo.

:: ============================================================
:: Step 1: 定位 Python
:: ============================================================
set "PYTHON_EXE="
:: 优先使用 AiPy 内置 venv
if exist "C:\Users\qsong\AppData\Roaming\aipy-pro\venv\Scripts\python.exe" (
    set "PYTHON_EXE=C:\Users\qsong\AppData\Roaming\aipy-pro\venv\Scripts\python.exe"
    goto :python_found
)
:: 回退：项目本地 venv
if exist "%~dp0venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
    goto :python_found
)
:: 回退：系统 Python
for %%p in (python python3 py) do (
    where %%p >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_EXE=%%p"
        goto :python_found
    )
)
:: 找不到 Python
echo [错误] 找不到 Python！
pause
exit /b 1

:python_found
"%PYTHON_EXE%" --version >nul 2>&1
if errorlevel 1 (
    echo [错误] Python 无法运行！
    pause
    exit /b 1
)

:: ============================================================
:: Step 2: 快速检查关键依赖
:: ============================================================
echo 正在检查依赖...
"%PYTHON_EXE%" -c "import flask; import flask_cors" >nul 2>&1
if errorlevel 1 (
    echo.
    echo [错误] 缺少关键依赖！请先安装：
    echo   pip install flask flask-cors
    echo.
    echo 查看详细错误：
    "%PYTHON_EXE%" -c "import flask; import flask_cors" 2>&1
    echo.
    pause
    exit /b 1
)

:: ============================================================
:: Step 3: 清理可能残留的旧服务器进程（占用5000端口）
:: ============================================================
echo 正在检查端口 5000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5000" ^| findstr "LISTENING"') do (
    echo 发现旧进程 PID=%%a，正在关闭...
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 1 /nobreak >nul

:: ============================================================
:: Step 4: 后台启动服务器（创建独立窗口，标题=WatermarkServer 供tasklist检测）
:: ============================================================
cls
echo.
echo  ============================================================
echo     水印清除工具 v3.5
echo  ============================================================
echo.
echo  正在加载 AI 引擎，请稍候...
echo  首次启动可能需要 10-30 秒（下载模型）
echo  请等待浏览器自动打开...
echo  ============================================================
echo.

:: 清空旧日志
break > "%SERVER_LOG%" 2>nul

:: 启动服务器（去掉 /b，创建独立窗口，标题 WatermarkServer 供 tasklist 检测）
start "WatermarkServer" cmd /c ""%PYTHON_EXE%" -B -m core.api_server > "%SERVER_LOG%" 2>&1"

:: 等待窗口创建
timeout /t 2 /nobreak >nul

:: ============================================================
:: Step 5: 等待服务器就绪（最多等120秒，检测进程存活）
:: ============================================================
echo.
echo  正在等待服务器启动...
set /a wait_count=0

:wait_loop
timeout /t 2 /nobreak >nul
set /a wait_count+=1

:: 检查服务器进程是否还活着
tasklist /fi "WINDOWTITLE eq WatermarkServer" 2>nul | findstr /i "cmd" >nul
if errorlevel 1 (
    :: 进程已退出，检查日志
    if !wait_count! leq 10 (
        echo.
        echo [错误] 服务器进程异常退出！请查看错误信息：
        echo ============================================================
        type "%SERVER_LOG%" 2>nul
        echo ============================================================
        echo.
        echo 日志文件：%SERVER_LOG%
        pause
        exit /b 1
    )
)

:: 检查服务器是否就绪
powershell -Command "try { $r = Invoke-WebRequest -Uri 'http://localhost:5000/api/health' -TimeoutSec 2 -UseBasicParsing; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
if !errorlevel! equ 0 (
    echo.
    echo  [OK] 服务器已就绪！正在打开浏览器...
    goto :open_browser
)

if !wait_count! geq 60 (
    echo.
    echo  [警告] 服务器启动超时（已等待120秒）
    echo.
    echo  服务器日志（最后 30 行）：
    echo ============================================================
    powershell -Command "Get-Content '%SERVER_LOG%' -Tail 30 -Encoding UTF8" 2>nul
    if errorlevel 1 (
        type "%SERVER_LOG%" 2>nul
    )
    echo ============================================================
    echo.
    echo  完整日志文件：%SERVER_LOG%
    echo.
    pause
    exit /b 1
)

:: 每10秒打印一次进度
set /a mod=!wait_count! %% 5
if !mod! equ 0 (
    echo   ...等待中 (!wait_count!0秒)...
)
goto :wait_loop

:open_browser
:: 打开浏览器
start "" http://localhost:5000
echo.
echo  ============================================================
echo  服务器已启动： http://localhost:5000
echo  关闭此窗口前请先关闭浏览器标签页
echo  按任意键停止服务器...
echo  ============================================================
echo.
:: 保持窗口打开，等待用户按任意键停止
pause >nul
:: 停止服务器
echo 正在停止服务器...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5000" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)
:: 关闭服务器窗口
taskkill /FI "WINDOWTITLE eq WatermarkServer" /F >nul 2>&1
echo 服务器已停止。
timeout /t 2 /nobreak >nul
exit /b 0