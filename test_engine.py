import sys, os
sys.path.insert(0, r"D:\AI\watermark_remover")

print("=== 测试引擎初始化 ===")
from core.engine import InpaintingEngine

engine = InpaintingEngine()
print(f"设备: {engine.device}")

print("\n=== 测试模型加载 ===")
try:
    success = engine.load_model()
    print(f"模型加载: {'成功' if success else '失败'}")
except Exception as e:
    print(f"模型加载异常: {e}")
    import traceback
    traceback.print_exc()

print("\n=== 测试图片处理 ===")
test_img = r"D:\AI\watermark_remover\test_data\compare_v3.png"
if os.path.exists(test_img):
    try:
        result = engine.inpaint_image(
            test_img,
            auto_detect=True,
            output_path=r"D:\AI\watermark_remover\output\test_result.png"
        )
        print(f"处理完成: {result}")
    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"测试图片不存在: {test_img}")
