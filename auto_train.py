#!/usr/bin/env python
"""
auto_train.py - Automatic watermark detection model training and deployment
Version 11.0

Usage:
    python auto_train.py --mode text --text "AI Generated" --epochs 100 --num 2000
    python auto_train.py --mode logo  --logo_dir "./WatermarkDataset/logos" --epochs 100
    python auto_train.py --mode both  --text "AI Generated" --logo_dir "./WatermarkDataset/logos"
"""

import argparse
import subprocess
import sys
import shutil
import time
from pathlib import Path

# ---------- Project root ----------
PROJECT_ROOT = Path(r"D:\AI\watermark_remover")
MODEL_DIR = PROJECT_ROOT / "models"
DATASET_DIR = PROJECT_ROOT / "WatermarkDataset"
TRAIN_SCRIPT = PROJECT_ROOT / "train_yolo.py"
GEN_SCRIPT = PROJECT_ROOT / "generate_watermark_dataset.py"
# Use the project's dedicated Python environment for training
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


def copy_best_model(save_dir):
    """Copy best.pt from training output to models/watermark_yolo.pt."""
    best_path = Path(save_dir) / "weights" / "best.pt"
    if best_path.exists():
        target = MODEL_DIR / "watermark_yolo.pt"
        # Backup old model
        if target.exists():
            backup = MODEL_DIR / f"watermark_yolo_backup_{int(time.time())}.pt"
            shutil.copy(str(target), str(backup))
            print(f"[INFO] Old model backed up: {backup}")
        shutil.copy(str(best_path), str(target))
        print(f"\n[OK] Model copied to {target}")
        return True
    else:
        print(f"\n[ERROR] best.pt not found at {best_path}")
        return False


def notify_flask_reload():
    """Notify Flask server to reload the detector model."""
    try:
        import requests
        resp = requests.post("http://localhost:5000/api/reload_model", timeout=5)
        if resp.status_code == 200:
            print("[OK] Flask server notified to reload model.")
        else:
            print(f"[WARN] Reload API returned {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"[WARN] Could not notify Flask: {e}")


def organize_logo_files(logo_dir):
    """
    Ensure logo files are in the expected subdirectory structure:
    WatermarkDataset/logos/combined/
    If the user's logo_dir has PNG files directly, copy them to combined/.
    """
    combined_dir = DATASET_DIR / "logos" / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    logo_path = Path(logo_dir)
    if not logo_path.exists():
        print(f"[ERROR] Logo directory not found: {logo_dir}")
        return False

    png_files = list(logo_path.glob("*.png")) + list(logo_path.glob("*.jpg")) + list(logo_path.glob("*.jpeg"))
    if not png_files:
        print(f"[ERROR] No PNG/JPG files found in {logo_dir}")
        return False

    copied = 0
    for f in png_files:
        target = combined_dir / f.name
        if not target.exists():
            shutil.copy2(str(f), str(target))
            copied += 1

    print(f"[INFO] Organized {copied} logo files into {combined_dir}")
    return True


def generate_text_dataset(text, num_images, bg_dir=None, img_size=640):
    """Call generate_watermark_dataset.py to create a text watermark dataset."""
    out_dir = DATASET_DIR / "yolo_text"
    cmd = [
        PYTHON_EXE, str(GEN_SCRIPT),
        "--num", str(num_images),
        "--output_dir", str(out_dir),
        "--text", text,
        "--img_size", str(img_size),
    ]
    if bg_dir:
        cmd.extend(["--bg_dir", str(bg_dir)])
    run_cmd(cmd)
    yaml_path = out_dir / "dataset.yaml"
    if not yaml_path.exists():
        print(f"[ERROR] Dataset YAML not generated at {yaml_path}")
        sys.exit(1)
    return yaml_path


def find_best_model_dir(name="watermark_custom"):
    """Find the training output directory containing best.pt."""
    # Try the custom name first
    candidates = [
        PROJECT_ROOT / "runs" / "detect" / name,
        PROJECT_ROOT / "runs" / "detect" / "watermark_detect",
    ]
    for cand in candidates:
        best = cand / "weights" / "best.pt"
        if best.exists():
            return cand
    # Search all runs/detect/ subdirectories sorted by modification time
    runs_detect = PROJECT_ROOT / "runs" / "detect"
    if runs_detect.exists():
        subdirs = sorted(
            [d for d in runs_detect.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
            reverse=True
        )
        for d in subdirs:
            best = d / "weights" / "best.pt"
            if best.exists():
                return d
    return None


def main():
    parser = argparse.ArgumentParser(description="Watermark Detection Model Auto Trainer")
    parser.add_argument("--mode", type=str, required=True, choices=["text", "logo", "both"],
                        help="Training mode: text, logo, or both")
    parser.add_argument("--text", type=str, default="AI Generated",
                        help="Watermark text content (for text/both mode)")
    parser.add_argument("--logo_dir", type=str, default=str(DATASET_DIR / "logos"),
                        help="Directory containing logo PNG images (for logo/both mode)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--bg_dir", type=str, default=str(DATASET_DIR / "backgrounds"),
                        help="Directory containing background images for dataset generation")
    parser.add_argument("--num", type=int, default=2000, help="Number of synthetic images")
    parser.add_argument("--model_size", type=str, default="n", help="YOLOv8 model size (n/s/m)")
    parser.add_argument("--img_size", type=int, default=640, help="Input image size")
    parser.add_argument("--device", type=str, default="0", help="GPU device (0 or cpu)")
    parser.add_argument("--no_reload", action="store_true", help="Skip Flask model reload notification")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  Watermark Model Training - Mode: {args.mode}")
    print("=" * 60)
    print(f"  Epochs:      {args.epochs}")
    print(f"  Model size:  yolov8{args.model_size}")
    print(f"  Image size:  {args.img_size}")
    print(f"  Device:      {args.device}")
    print(f"  Synthetic:   {args.num} images")
    print("=" * 60)

    # --- Step 1: Generate datasets ---
    dataset_yamls = []

    if args.mode in ("text", "both"):
        print(f"\n[Step 1] Generating text watermark dataset: '{args.text}'")
        yaml_path = generate_text_dataset(args.text, args.num, args.bg_dir, args.img_size)
        dataset_yamls.append(yaml_path)

    if args.mode in ("logo", "both"):
        logo_dir = Path(args.logo_dir)
        if not organize_logo_files(logo_dir):
            sys.exit(1)
        print(f"\n[Step 1] Logo files organized in WatermarkDataset/logos/combined/")

    # --- Step 2: Train ---
    print(f"\n[Step 2] Starting YOLOv8 training...")

    if args.mode == "logo":
        # Logo mode: use train_yolo.py's built-in synthetic data generation
        cmd = [
            PYTHON_EXE, str(TRAIN_SCRIPT),
            "--epochs", str(args.epochs),
            "--model-size", args.model_size,
            "--img-size", str(args.img_size),
            "--device", args.device,
            "--name", "watermark_custom",
            "--num-synthetic", str(args.num),
        ]
        run_cmd(cmd)
    elif args.mode == "text":
        # Text mode: train with pre-generated dataset YAML
        cmd = [
            PYTHON_EXE, str(TRAIN_SCRIPT),
            "--data", str(dataset_yamls[0]),
            "--epochs", str(args.epochs),
            "--model-size", args.model_size,
            "--img-size", str(args.img_size),
            "--device", args.device,
            "--name", "watermark_custom",
            "--skip-generate",
        ]
        run_cmd(cmd)
    elif args.mode == "both":
        # Both mode: first train with logo synthetic data, then with text dataset
        # Strategy: train logo synthetic first, then fine-tune with text dataset
        print("\n[Step 2a] Training with logo synthetic data...")
        cmd1 = [
            PYTHON_EXE, str(TRAIN_SCRIPT),
            "--epochs", str(args.epochs // 2),
            "--model-size", args.model_size,
            "--img-size", str(args.img_size),
            "--device", args.device,
            "--name", "watermark_logo",
            "--num-synthetic", str(args.num // 2),
        ]
        run_cmd(cmd1)

        # Find the logo-trained model
        logo_model_dir = find_best_model_dir("watermark_logo")
        if logo_model_dir:
            logo_model_path = logo_model_dir / "weights" / "best.pt"
            print(f"\n[Step 2b] Fine-tuning with text dataset using {logo_model_path}")
            # Use the logo-trained model as starting point for text fine-tuning
            # We need to modify train_yolo.py to support --weights, or we can
            # copy the logo model as the starting model and train with text data
            cmd2 = [
                PYTHON_EXE, str(TRAIN_SCRIPT),
                "--data", str(dataset_yamls[0]),
                "--epochs", str(args.epochs),
                "--model-size", args.model_size,
                "--img-size", str(args.img_size),
                "--device", args.device,
                "--name", "watermark_custom",
                "--skip-generate",
                "--weights", str(logo_model_path),
            ]
            run_cmd(cmd2)
        else:
            print("[WARN] Logo model not found, training with text data only")
            cmd2 = [
                PYTHON_EXE, str(TRAIN_SCRIPT),
                "--data", str(dataset_yamls[0]),
                "--epochs", str(args.epochs),
                "--model-size", args.model_size,
                "--img-size", str(args.img_size),
                "--device", args.device,
                "--name", "watermark_custom",
                "--skip-generate",
            ]
            run_cmd(cmd2)

    # --- Step 3: Copy best model ---
    print(f"\n[Step 3] Installing trained model...")
    # train_yolo.py already copies best.pt to models/watermark_yolo.pt internally
    target_model = MODEL_DIR / "watermark_yolo.pt"
    if target_model.exists():
        print(f"[OK] Model already installed at {target_model}")
    else:
        # Fallback: search for best.pt in runs/detect/ and copy manually
        save_dir = find_best_model_dir("watermark_custom")
        if save_dir is None:
            save_dir = find_best_model_dir("watermark_detect")
        if save_dir is None:
            # Also search CWD's runs/detect (YOLO may write to CWD)
            cwd_runs = Path.cwd() / "runs" / "detect"
            if cwd_runs.exists():
                subdirs = sorted(
                    [d for d in cwd_runs.iterdir() if d.is_dir()],
                    key=lambda d: d.stat().st_mtime, reverse=True
                )
                for d in subdirs:
                    best = d / "weights" / "best.pt"
                    if best.exists():
                        save_dir = d
                        break
        if save_dir is None:
            print("[ERROR] Cannot find best.pt in any training output directory")
            print("Check runs/detect/ for training outputs")
            sys.exit(1)
        if not copy_best_model(save_dir):
            sys.exit(1)
        print("\n[OK] Model installed to models/watermark_yolo.pt")

    # --- Step 4: Notify Flask ---
    if not args.no_reload:
        notify_flask_reload()

    print("\n" + "=" * 60)
    print("  [SUCCESS] Training pipeline completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()