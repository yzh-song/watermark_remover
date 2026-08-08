"""端到端测试 - 模拟.bat启动流程"""
import sys, os, time, threading

sys.path.insert(0, r"D:\AI\watermark_remover")
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'

print("=" * 60)
print("E2E Test - Simulating .bat startup")
print("=" * 60)

# Step 1: Check imports
print("\n[1] Checking imports...")
try:
    import torch
    import flask
    import cv2
    from simple_lama_inpainting import SimpleLama
    print("[OK] All imports successful")
except Exception as e:
    print(f"[FAIL] Import error: {e}")
    sys.exit(1)

# Step 2: Check CUDA
print("\n[2] Checking GPU...")
if torch.cuda.is_available():
    print(f"[GPU] {torch.cuda.get_device_name(0)}")
else:
    print("[CPU] Running in CPU mode")

# Step 3: Init engine
print("\n[3] Initializing engine...")
from core.engine import InpaintingEngine
engine = InpaintingEngine()
engine.load_model()
print(f"[OK] Engine ready: {engine.device}")

# Step 4: Test processing
print("\n[4] Testing image processing...")
test_img = r"D:\AI\watermark_remover\test_data\compare_v3.png"
if os.path.exists(test_img):
    t0 = time.time()
    result = engine.inpaint_image(
        test_img,
        auto_detect=True,
        output_path=r"D:\AI\watermark_remover\output\e2e_test.png"
    )
    t1 = time.time()
    print(f"[OK] Done in {t1-t0:.1f}s -> {result}")
else:
    print(f"[SKIP] Test image not found")

# Step 5: Start Flask server
print("\n[5] Starting Flask server...")
from core.api_server import app, init_engine
init_engine()

def run_server():
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()
time.sleep(3)

# Step 6: Test API
print("\n[6] Testing API endpoints...")
import requests

try:
    # Health check
    resp = requests.get("http://localhost:5000/api/health", timeout=5)
    data = resp.json()
    print(f"[OK] Health: {data['status']}, device: {data.get('device', 'N/A')}")
    
    # Frontend page
    resp2 = requests.get("http://localhost:5000/", timeout=5)
    print(f"[OK] Frontend: {resp2.status_code} ({len(resp2.text)} chars)")
    
    # Process API
    with open(test_img, 'rb') as f:
        files = {'file': ('test.png', f, 'image/png')}
        data = {'mode': 'auto', 'feather': '5'}
        resp3 = requests.post("http://localhost:5000/api/process",
                              files=files, data=data, timeout=60)
        result = resp3.json()
        if result.get('success'):
            print(f"[OK] Process API: {result['message']}")
        else:
            print(f"[FAIL] Process API: {result.get('error')}")
    
    # Download API
    if result.get('success'):
        resp4 = requests.get(f"http://localhost:5000{result['output_url']}", timeout=5)
        print(f"[OK] Download: {resp4.status_code} ({len(resp4.content)} bytes)")

except Exception as e:
    print(f"[FAIL] API test error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("E2E Test Complete!")
print("=" * 60)
