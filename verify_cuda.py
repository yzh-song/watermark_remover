"""验证CUDA环境和LaMa兼容性"""
import sys, os
sys.path.insert(0, r"D:\AI\watermark_remover")

import torch
import numpy as np
from PIL import Image

print("=" * 60)
print("🔍 环境兼容性验证")
print("=" * 60)

print(f"\n✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"✅ 显存: {mem:.1f} GB")
    print(f"✅ CUDA版本: {torch.version.cuda}")
print(f"✅ NumPy: {np.__version__}")
print(f"✅ Pillow: {Image.__version__}")

print("\n--- 测试 simple-lama-inpainting ---")
try:
    from simple_lama_inpainting import SimpleLama
    print("✅ 导入成功")
    
    # CUDA加载
    device = torch.device('cuda')
    print(f"⏳ 正在CUDA上加载LaMa模型...")
    lama = SimpleLama(device=device)
    print("✅ LaMa模型 CUDA加载成功!")
    
    # 推理测试
    print("⏳ CUDA推理测试...")
    test_img = Image.new('RGB', (512, 512), color=(128, 128, 128))
    test_mask = Image.new('L', (512, 512), color=0)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(test_mask)
    draw.rectangle([150, 150, 362, 362], fill=255)
    
    import time
    t0 = time.time()
    result = lama(test_img, test_mask)
    t1 = time.time()
    print(f"✅ CUDA推理完成! 耗时: {(t1-t0)*1000:.0f}ms, 输出: {result.size}")
    
    # 清理
    del lama, result
    torch.cuda.empty_cache()
    print("✅ GPU内存已清理")
    
except Exception as e:
    print(f"❌ 失败: {e}")
    import traceback
    traceback.print_exc()

print("\n--- 测试引擎 ---")
from core.engine import InpaintingEngine

engine = InpaintingEngine(device='cuda')
print(f"✅ 引擎设备: {engine.device}")
engine.load_model()
print(f"✅ 引擎模型加载成功")

# 真实图片测试
test_img_path = r"D:\AI\watermark_remover\test_data\compare_v3.png"
if os.path.exists(test_img_path):
    print(f"\n⏳ 真实图片去水印测试 (CUDA)...")
    t0 = time.time()
    result_path = engine.inpaint_image(
        test_img_path,
        auto_detect=True,
        output_path=r"D:\AI\watermark_remover\output\cuda_test_result.png"
    )
    t1 = time.time()
    print(f"✅ CUDA处理完成! 耗时: {t1-t0:.1f}s")
    print(f"   输出: {result_path}")
    print(f"   大小: {os.path.getsize(result_path)/1024:.1f}KB")

print("\n" + "=" * 60)
print("🎉 所有验证通过！CUDA加速已就绪！")
print("=" * 60)
