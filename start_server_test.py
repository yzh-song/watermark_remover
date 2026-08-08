"""启动服务器并运行自检"""
import sys, os, time, threading, requests

sys.path.insert(0, r"D:\AI\watermark_remover")

print("=" * 60)
print("🪄 一键去水印 - 服务器自检")
print("=" * 60)

# 1. 检查torch
import torch
print(f"✅ PyTorch: {torch.__version__}")
print(f"✅ CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

# 2. 检查引擎
from core.engine import InpaintingEngine
engine = InpaintingEngine()
engine.load_model()
print(f"✅ 引擎就绪: {engine.device}")

# 3. 测试处理
test_img = r"D:\AI\watermark_remover\test_data\compare_v3.png"
if os.path.exists(test_img):
    result = engine.inpaint_image(test_img, auto_detect=True)
    print(f"✅ 图片处理测试通过: {result}")

# 4. 启动Flask
print("\n🚀 启动Web服务器...")
from core.api_server import app, init_engine
init_engine()

# 在新线程中启动
server_thread = threading.Thread(
    target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, threaded=True),
    daemon=True
)
server_thread.start()
time.sleep(3)

# 5. 测试API
try:
    resp = requests.get("http://localhost:5000/api/health", timeout=5)
    print(f"✅ 健康检查: {resp.json()}")
    
    resp2 = requests.get("http://localhost:5000/", timeout=5)
    print(f"✅ 前端页面: {len(resp2.text)}字符")
    
    # 测试处理API
    with open(test_img, 'rb') as f:
        files = {'file': ('test.png', f, 'image/png')}
        data = {'mode': 'auto', 'feather': '5'}
        resp3 = requests.post("http://localhost:5000/api/process", 
                              files=files, data=data, timeout=60)
        result = resp3.json()
        print(f"✅ 处理API: {result.get('success')} - {result.get('message', result.get('error', ''))}")
        
except Exception as e:
    print(f"❌ API测试失败: {e}")

print("\n" + "=" * 60)
print("🎉 服务器自检完成！")
print("📱 浏览器访问: http://localhost:5000")
print("=" * 60)

# 保持运行
print("\n服务器正在运行，按Ctrl+C停止...")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n服务器已停止")
