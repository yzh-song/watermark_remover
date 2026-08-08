"""
scripts/train.py - One-click watermark detection model training.
Version 12.0 - Wraps auto_train.py with proper path handling.

Usage:
    python scripts/train.py --mode text --text "AI Generated" --epochs 100 --num 2000
    python scripts/train.py --mode logo --logo_dir "./WatermarkDataset/logos" --epochs 100
    python scripts/train.py --mode both --text "AI Generated" --logo_dir "./WatermarkDataset/logos"
"""

import argparse
import subprocess
import sys
import shutil
import time
from pathlib import Path

PROJECT_ROOT = Path(r"D:\AI\watermark_remover")
MODEL_DIR = PROJECT_ROOT / "models"
DATASET_DIR = PROJECT_ROOT / "WatermarkDataset"
AUTO_TRAIN_SCRIPT = PROJECT_ROOT / "auto_train.py"
PYTHON_EXE = r"C:\Users\qsong\AppData\Roaming\aipy-pro\venv\Scripts\python.exe"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd, cwd=None):
    """Run a command and stream output in real time."""
    print(f"\n[CMD] {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end='')
    proc.wait()
    if proc.returncode != 0:
        print(f"\n[ERROR] Command failed with code {proc.returncode}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Watermark Detection Model Training")
    parser.add_argument("--mode", type=str, required=True, choices=["text", "logo", "both"],
                        help="Training mode: text, logo, or both")
    parser.add_argument("--text", type=str, default="AI Generated",
                        help="Watermark text content")
    parser.add_argument("--logo_dir", type=str, default=str(DATASET_DIR / "logos"),
                        help="Directory containing logo PNG images")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--bg_dir", type=str, default=str(DATASET_DIR / "backgrounds"),
                        help="Background images directory")
    parser.add_argument("--num", type=int, default=2000, help="Number of synthetic images")
    parser.add_argument("--model_size", type=str, default="n", help="YOLOv8 model size (n/s/m)")
    parser.add_argument("--img_size", type=int, default=640, help="Input image size")
    parser.add_argument("--device", type=str, default="0", help="GPU device (0 or cpu)")
    args = parser.parse_args()

    if not AUTO_TRAIN_SCRIPT.exists():
        print(f"[ERROR] auto_train.py not found at {AUTO_TRAIN_SCRIPT}")
        sys.exit(1)

    print("=" * 60)
    print(f"  Watermark Model Training - Mode: {args.mode}")
    print("=" * 60)
    print(f"  Epochs:      {args.epochs}")
    print(f"  Model size:  yolov8{args.model_size}")
    print(f"  Image size:  {args.img_size}")
    print(f"  Device:      {args.device}")
    print(f"  Synthetic:   {args.num} images")
    print("=" * 60)

    cmd = [
        PYTHON_EXE,
        str(AUTO_TRAIN_SCRIPT),
        "--mode", args.mode,
        "--text", args.text,
        "--logo_dir", args.logo_dir,
        "--epochs", str(args.epochs),
        "--num", str(args.num),
        "--bg_dir", args.bg_dir,
        "--model_size", args.model_size,
        "--img_size", str(args.img_size),
        "--device", args.device,
        "--no_reload",
    ]
    run_cmd(cmd, str(PROJECT_ROOT))

    # Verify model was created
    target = MODEL_DIR / "watermark_yolo.pt"
    if target.exists():
        print(f"\n[OK] Model ready: {target}")
    else:
        print(f"\n[WARN] Model not found at {target}. Check training logs.")

    print("\n" + "=" * 60)
    print("  [SUCCESS] Training pipeline completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()