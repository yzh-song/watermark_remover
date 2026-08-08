@echo off
chcp 65001 >nul
title 安装CUDA版PyTorch
echo.
echo ╔══════════════════════════════════════════╗
echo ║     安装CUDA版PyTorch (GPU加速)          ║
echo ╚══════════════════════════════════════════╝
echo.
echo 此脚本将下载约2.5GB的CUDA版PyTorch
echo 请确保网络畅通，可能需要5-15分钟
echo.
cd /d "%~dp0"
echo [1/2] 卸载CPU版PyTorch...
python -m pip uninstall torch torchvision torchaudio -y
echo.
echo [2/2] 安装CUDA 12.1版PyTorch...
echo 正在从PyTorch官方源下载，请耐心等待...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if %errorlevel% equ 0 (
    echo.
    echo ✅ CUDA版PyTorch安装成功！
    python -c "import torch; print('GPU: ' + torch.cuda.get_device_name(0)); print('显存: ' + str(round(torch.cuda.get_device_properties(0).total_mem/1024**3, 1)) + ' GB')"
) else (
    echo.
    echo ❌ 安装失败，请检查网络连接
    echo 可以尝试使用清华镜像：
    echo python -m pip install torch torchvision torchaudio -f https://mirrors.tuna.tsinghua.edu.cn/pytorch/whl/cu121/torch_stable.html
)
pause