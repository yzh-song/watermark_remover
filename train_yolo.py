"""
YOLO Watermark Detection Training Script
Version 10.0 - Synthetic data generation + YOLOv8 training

Usage:
    python train_yolo.py              # Train with default settings
    python train_yolo.py --epochs 100  # Custom epochs
    python train_yolo.py --export-only # Export existing model only
"""
import os
import sys
import argparse
import logging
import random
import shutil
from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(r"D:\AI\watermark_remover")
DATASET_DIR = BASE_DIR / "WatermarkDataset"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"
LOGOS_DIR = DATASET_DIR / "logos"
SAMPLE_DIR = DATASET_DIR / "sample images 100"
TRAIN_DIR = DATASET_DIR / "yolo_dataset" / "train"
VAL_DIR = DATASET_DIR / "yolo_dataset" / "val"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def generate_synthetic_data(num_images: int = 500,
                            val_split: float = 0.2,
                            img_size: int = 640) -> Tuple[int, int]:
    """
    Generate synthetic training data by compositing watermark logos
    onto sample images. Creates YOLO-format labels automatically.

    Returns:
        (num_train, num_val) counts
    """
    logger.info("=" * 60)
    logger.info("Generating synthetic training data...")
    logger.info("=" * 60)

    # --- Collect source images ---
    sample_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        sample_images.extend(list(SAMPLE_DIR.glob(ext)))
    if not sample_images:
        logger.error(f"No sample images found in {SAMPLE_DIR}")
        return 0, 0
    logger.info(f"Found {len(sample_images)} sample images")

    # --- Collect logos ---
    logo_files = []
    for logo_dir in [LOGOS_DIR / "combined", LOGOS_DIR / "independent", LOGOS_DIR]:
        if logo_dir.exists():
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                logo_files.extend(list(logo_dir.glob(ext)))
    if not logo_files:
        logger.error(f"No logo images found in {LOGOS_DIR}")
        return 0, 0
    logger.info(f"Found {len(logo_files)} logo images")

    # --- Create output directories ---
    for subdir in ['images', 'labels']:
        (TRAIN_DIR / subdir).mkdir(parents=True, exist_ok=True)
        (VAL_DIR / subdir).mkdir(parents=True, exist_ok=True)

    num_val = int(num_images * val_split)
    num_train = num_images - num_val

    logger.info(f"Generating {num_train} train + {num_val} val images...")

    for i in range(num_images):
        is_val = i < num_val
        out_dir = VAL_DIR if is_val else TRAIN_DIR
        idx = i if is_val else i - num_val

        # Pick random background and logo
        bg_path = random.choice(sample_images)
        logo_path = random.choice(logo_files)

        try:
            # Load background
            bg = cv2.imread(str(bg_path))
            if bg is None:
                continue
            bg = cv2.resize(bg, (img_size, img_size))
            h, w = bg.shape[:2]

            # Load logo with alpha
            logo = cv2.imread(str(logo_path), cv2.IMREAD_UNCHANGED)
            if logo is None:
                continue

            # Resize logo to random size (10-40% of image)
            scale = random.uniform(0.10, 0.35)
            logo_h = int(h * scale)
            logo_w = int(logo_h * logo.shape[1] / max(logo.shape[0], 1))
            if logo_w < 20 or logo_h < 10:
                logo_w = max(logo_w, 30)
                logo_h = max(logo_h, 15)
            logo = cv2.resize(logo, (logo_w, logo_h))

            # Random position (biased toward corners)
            corner = random.choice([
                'tl', 'tr', 'bl', 'br', 'center'
            ])
            margin = 20
            if corner == 'tl':
                x = random.randint(margin, max(margin, w // 5))
                y = random.randint(margin, max(margin, h // 5))
            elif corner == 'tr':
                x = random.randint(w - logo_w - w // 5, max(logo_w, w - logo_w - margin))
                y = random.randint(margin, max(margin, h // 5))
            elif corner == 'bl':
                x = random.randint(margin, max(margin, w // 5))
                y = random.randint(h - logo_h - h // 5, max(logo_h, h - logo_h - margin))
            elif corner == 'br':
                x = random.randint(w - logo_w - w // 5, max(logo_w, w - logo_w - margin))
                y = random.randint(h - logo_h - h // 5, max(logo_h, h - logo_h - margin))
            else:
                x = random.randint(margin, max(margin, w - logo_w - margin))
                y = random.randint(margin, max(margin, h - logo_h - margin))

            x = max(0, min(x, w - logo_w))
            y = max(0, min(y, h - logo_h))

            # Composite logo onto background
            if logo.shape[2] == 4:
                alpha = logo[:, :, 3].astype(np.float32) / 255.0
                alpha = alpha * random.uniform(0.5, 0.95)  # Random opacity
                for c in range(3):
                    bg[y:y+logo_h, x:x+logo_w, c] = (
                        alpha * logo[:, :, c] + (1 - alpha) * bg[y:y+logo_h, x:x+logo_w, c]
                    ).astype(np.uint8)
            else:
                alpha = random.uniform(0.3, 0.7)
                bg[y:y+logo_h, x:x+logo_w] = cv2.addWeighted(
                    logo, alpha, bg[y:y+logo_h, x:x+logo_w], 1 - alpha, 0
                )

            # Save image
            img_name = f"synth_{idx:05d}.jpg"
            cv2.imwrite(str(out_dir / "images" / img_name), bg, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # YOLO format label: class x_center y_center width height (normalized)
            x_center = (x + logo_w / 2) / w
            y_center = (y + logo_h / 2) / h
            width_norm = logo_w / w
            height_norm = logo_h / h

            label_name = f"synth_{idx:05d}.txt"
            with open(out_dir / "labels" / label_name, 'w') as f:
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")

        except Exception as e:
            logger.warning(f"Failed to generate image {i}: {e}")
            continue

        if (i + 1) % 100 == 0:
            logger.info(f"  Generated {i + 1}/{num_images} images...")

    logger.info(f"[OK] Synthetic data generated: {num_train} train + {num_val} val")
    return num_train, num_val


def create_dataset_yaml() -> Path:
    """Create dataset YAML config for YOLO training."""
    yaml_path = DATASET_DIR / "yolo_dataset" / "dataset.yaml"
    yaml_content = f"""# YOLO Watermark Detection Dataset
path: {DATASET_DIR / 'yolo_dataset'}
train: train/images
val: val/images

# Classes
nc: 1
names: ['watermark']
"""
    yaml_path.write_text(yaml_content)
    logger.info(f"Dataset YAML created: {yaml_path}")
    return yaml_path


def train_yolo(yaml_path: Path, epochs: int = 100,
               model_size: str = 'n', img_size: int = 640,
               batch: int = 16, device: str = '0',
               run_name: str = 'watermark_detect',
               weights_path: str = None) -> str:
    """
    Train YOLOv8 model on the watermark dataset.

    Args:
        yaml_path: Path to dataset YAML
        epochs: Number of training epochs
        model_size: 'n' (nano), 's' (small), 'm' (medium)
        img_size: Input image size
        batch: Batch size
        device: CUDA device or 'cpu'
        run_name: Name for the training run directory
        weights_path: Path to initial weights (overrides default yolov8.pt)

    Returns:
        Path to the best model weights
    """
    logger.info("=" * 60)
    logger.info("Training YOLOv8 model...")
    logger.info("=" * 60)

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        return ""

    model_name = weights_path if weights_path else f"yolov8{model_size}.pt"
    logger.info(f"Using model: {model_name}")

    model = YOLO(model_name)

    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        device=device,
        name=run_name,
        exist_ok=True,
        patience=20,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        verbose=True,
    )

    # Copy best model to models directory
    # Use results.save_dir for dynamic path (avoids hardcoded path issues)
    save_dir = Path(results.save_dir)
    best_path = save_dir / 'weights' / 'best.pt'
    if best_path.exists():
        target_path = MODEL_DIR / "watermark_yolo.pt"
        shutil.copy(str(best_path), str(target_path))
        logger.info(f"Best model saved to: {target_path}")
        return str(target_path)
    else:
        # Fallback: try legacy path
        legacy_path = Path(f"runs/detect/{run_name}/weights/best.pt")
        if legacy_path.exists():
            target_path = MODEL_DIR / "watermark_yolo.pt"
            shutil.copy(str(legacy_path), str(target_path))
            logger.info(f"Best model saved to: {target_path} (from legacy path)")
            return str(target_path)
        logger.error(f"Best model not found at {best_path} or {legacy_path}")
        return ""


def export_model(model_path: str, export_format: str = 'onnx') -> str:
    """Export trained model to ONNX or other format."""
    logger.info(f"Exporting model to {export_format}...")

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed")
        return ""

    if not os.path.exists(model_path):
        logger.error(f"Model not found: {model_path}")
        return ""

    model = YOLO(model_path)
    export_path = model.export(format=export_format, imgsz=640, simplify=True)
    logger.info(f"Model exported: {export_path}")
    return str(export_path)


def main():
    parser = argparse.ArgumentParser(description='YOLO Watermark Detection Training')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--model-size', type=str, default='n', choices=['n', 's', 'm'],
                        help='YOLOv8 model size (n=nano, s=small, m=medium)')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--num-synthetic', type=int, default=500,
                        help='Number of synthetic images to generate')
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA device (0, 1, ...) or cpu')
    parser.add_argument('--export-only', action='store_true',
                        help='Only export existing model, skip training')
    parser.add_argument('--skip-generate', action='store_true',
                        help='Skip synthetic data generation')
    parser.add_argument('--export-format', type=str, default='onnx',
                        choices=['onnx', 'torchscript', 'openvino', 'tflite'])
    parser.add_argument('--data', type=str, default=None,
                        help='Path to dataset YAML (overrides default, skips data generation)')
    parser.add_argument('--name', type=str, default='watermark_detect',
                        help='Run name for training output directory')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to initial weights (overrides default yolov8.pt)')

    args = parser.parse_args()

    print("=" * 60)
    print("  YOLO Watermark Detection Training v10.0")
    print("=" * 60)
    print(f"  Epochs:      {args.epochs}")
    print(f"  Model:       yolov8{args.model_size}")
    print(f"  Image size:  {args.img_size}")
    print(f"  Batch:       {args.batch}")
    print(f"  Device:      {args.device}")
    print(f"  Synthetic:   {args.num_synthetic} images")
    print("=" * 60)
    print()

    # --- Export only ---
    model_path = str(MODEL_DIR / "watermark_yolo.pt")
    if args.export_only:
        if not os.path.exists(model_path):
            logger.error(f"No trained model found at {model_path}")
            sys.exit(1)
        export_model(model_path, args.export_format)
        return

    # --- Generate synthetic data ---
    if args.data:
        logger.info(f"Using provided dataset YAML: {args.data}")
    elif not args.skip_generate:
        n_train, n_val = generate_synthetic_data(args.num_synthetic)
        if n_train == 0:
            logger.error("Data generation failed. Check dataset paths.")
            sys.exit(1)
    else:
        logger.info("Skipping data generation (--skip-generate)")

    # --- Create or use dataset YAML ---
    if args.data:
        yaml_path = Path(args.data)
        if not yaml_path.exists():
            logger.error(f"Dataset YAML not found: {args.data}")
            sys.exit(1)
    else:
        yaml_path = create_dataset_yaml()

    # --- Train ---
    best_path = train_yolo(
        yaml_path, args.epochs, args.model_size,
        args.img_size, args.batch, args.device,
        run_name=args.name, weights_path=args.weights
    )

    if not best_path:
        logger.error("Training failed.")
        sys.exit(1)

    # --- Export ---
    export_model(best_path, args.export_format)

    print()
    print("=" * 60)
    print("  [OK] Training complete!")
    print(f"  Model:     {best_path}")
    print(f"  Export:    {args.export_format}")
    print("  Use:       detector.load_yolo()")
    print("=" * 60)


if __name__ == '__main__':
    main()