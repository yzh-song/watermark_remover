"""
generate_watermark_dataset.py
Generate YOLO training dataset with "AI Generated" watermark text.
Version 11.0 - Chinese text watermark synthesis for YOLO training

Usage:
    python generate_watermark_dataset.py --num 2000 --output_dir ./WatermarkDataset/yolo_ai_gen
    python generate_watermark_dataset.py --num 3000 --bg_dir "D:/my_photos" --text "AI Generated"

Dependencies: opencv-python, numpy, Pillow
"""

import os
import random
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Try to find a system Chinese font
_CHINESE_FONT_CANDIDATES = [
    "simhei.ttf",
    "simsun.ttc",
    "msyh.ttc",
    "msyhbd.ttc",
    "NotoSansCJK-Regular.ttc",
    "NotoSansSC-Regular.otf",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
]

_FONT_PATH = None
for _fp in _CHINESE_FONT_CANDIDATES:
    if os.path.exists(_fp):
        _FONT_PATH = _fp
        break


def generate_text_watermark(text: str, size: int, font_path: str = None) -> np.ndarray:
    """
    Generate transparent PNG with watermark text.
    Args:
        text: Watermark text (e.g. "AI Generated")
        size: Font size in pixels
        font_path: Path to .ttf/.ttc font file (uses system Chinese font if None)
    Returns:
        RGBA numpy array (H, W, 4)
    """
    fp = font_path or _FONT_PATH
    try:
        font = ImageFont.truetype(fp, size) if fp else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # Estimate text dimensions
    temp_img = Image.new('RGBA', (size * len(text) * 2, size * 2), (0, 0, 0, 0))
    temp_draw = ImageDraw.Draw(temp_img)
    bbox = temp_draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if w <= 0 or h <= 0:
        w, h = size * len(text), size

    # Create watermark image with padding
    pad = max(4, size // 8)
    img = Image.new('RGBA', (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((pad, pad), text, font=font, fill=(255, 255, 255, 255))
    return np.array(img)


def composite_watermark(background: np.ndarray, watermark: np.ndarray,
                        pos: tuple, opacity: float) -> np.ndarray:
    """
    Composite watermark onto background with alpha blending.
    Args:
        background: BGR numpy array (H, W, 3)
        watermark: RGBA numpy array (H, W, 4)
        pos: (x, y) top-left position
        opacity: 0.0-1.0 transparency
    Returns:
        Modified background (in-place)
    """
    h, w = watermark.shape[:2]
    x, y = pos
    # Clamp position to valid range
    x = max(0, min(x, background.shape[1] - 1))
    y = max(0, min(y, background.shape[0] - 1))
    # Crop watermark to fit within background bounds
    avail_w = min(w, background.shape[1] - x)
    avail_h = min(h, background.shape[0] - y)
    if avail_w <= 0 or avail_h <= 0:
        return background
    wm_cropped = watermark[:avail_h, :avail_w]
    alpha = wm_cropped[:, :, 3].astype(np.float32) / 255.0 * opacity
    alpha_3ch = np.stack([alpha] * 3, axis=2)
    roi = background[y:y + avail_h, x:x + avail_w].astype(np.float32)
    wm_rgb = wm_cropped[:, :, :3].astype(np.float32)
    blended = wm_rgb * alpha_3ch + roi * (1 - alpha_3ch)
    background[y:y + avail_h, x:x + avail_w] = blended.astype(np.uint8)
    return background


def main():
    parser = argparse.ArgumentParser(
        description='Generate YOLO training dataset with watermark text')
    parser.add_argument('--num', type=int, default=2000,
                        help='Number of images to generate')
    parser.add_argument('--output_dir', type=str,
                        default='./WatermarkDataset/yolo_ai_gen',
                        help='Output directory for dataset')
    parser.add_argument('--bg_dir', type=str,
                        default='./WatermarkDataset/backgrounds',
                        help='Directory containing background images')
    parser.add_argument('--img_size', type=int, default=640,
                        help='Output image size')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--text', type=str, nargs='+',
                        default=['AI Generated', 'AI Watermark'],
                        help='Watermark text(s) to use')
    parser.add_argument('--font', type=str, default=None,
                        help='Path to font file (.ttf/.ttc)')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    train_img = out_dir / 'train' / 'images'
    train_lbl = out_dir / 'train' / 'labels'
    val_img = out_dir / 'val' / 'images'
    val_lbl = out_dir / 'val' / 'labels'
    for p in [train_img, train_lbl, val_img, val_lbl]:
        p.mkdir(parents=True, exist_ok=True)

    # Collect background images
    bg_dir = Path(args.bg_dir)
    bg_files = []
    if bg_dir.exists():
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            bg_files.extend(bg_dir.glob(ext))
    if not bg_files:
        # Fallback: use solid color backgrounds
        print(f"[WARN] No background images found in {bg_dir}. Using solid color backgrounds.")
        bg_files = None

    watermark_texts = args.text
    font_path = args.font or _FONT_PATH
    if font_path:
        print(f"Using font: {font_path}")
    else:
        print("[WARN] No Chinese font found. Watermark text may not render correctly.")
        print("  Download a Chinese font (e.g. simhei.ttf) or use --text with English text.")

    num_val = int(args.num * args.val_ratio)
    num_train = args.num - num_val
    print(f"Generating {num_train} train + {num_val} val images...")
    print(f"Watermark texts: {watermark_texts}")
    print(f"Image size: {args.img_size}x{args.img_size}")

    generated = 0
    for i in range(args.num):
        is_val = i < num_val
        out_img = (val_img if is_val else train_img) / f'img_{i:06d}.jpg'
        out_lbl = (val_lbl if is_val else train_lbl) / f'img_{i:06d}.txt'

        # Background
        if bg_files:
            bg_path = random.choice(bg_files)
            bg = cv2.imread(str(bg_path))
            if bg is None:
                continue
            bg = cv2.resize(bg, (args.img_size, args.img_size))
        else:
            # Solid random color background
            color = [random.randint(30, 225) for _ in range(3)]
            bg = np.full((args.img_size, args.img_size, 3), color, dtype=np.uint8)

        h, w = bg.shape[:2]

        # Random watermark parameters
        text = random.choice(watermark_texts)
        font_size = random.randint(int(args.img_size * 0.05), int(args.img_size * 0.20))
        opacity = random.uniform(0.4, 0.9)

        # Generate watermark
        wm = generate_text_watermark(text, font_size, font_path)
        wm_h, wm_w = wm.shape[:2]
        if wm_w < 5 or wm_h < 5:
            continue

        # Random position (biased toward corners)
        corner = random.choice(['tl', 'tr', 'bl', 'br'])
        margin = 20
        if corner == 'tl':
            x = random.randint(margin, max(margin, w // 4 - wm_w))
            y = random.randint(margin, max(margin, h // 4 - wm_h))
        elif corner == 'tr':
            x = random.randint(w - wm_w - w // 4, w - wm_w - margin)
            y = random.randint(margin, max(margin, h // 4 - wm_h))
        elif corner == 'bl':
            x = random.randint(margin, max(margin, w // 4 - wm_w))
            y = random.randint(h - wm_h - h // 4, h - wm_h - margin)
        else:  # br
            x = random.randint(w - wm_w - w // 4, w - wm_w - margin)
            y = random.randint(h - wm_h - h // 4, h - wm_h - margin)

        x = max(0, min(x, w - wm_w))
        y = max(0, min(y, h - wm_h))

        # Composite
        bg = composite_watermark(bg, wm, (x, y), opacity)
        cv2.imwrite(str(out_img), bg, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # YOLO label: use actual rendered dimensions (watermark may be cropped to fit)
        actual_w = min(wm_w, w - x)
        actual_h = min(wm_h, h - y)
        x_center = (x + actual_w / 2) / w
        y_center = (y + actual_h / 2) / h
        width_norm = actual_w / w
        height_norm = actual_h / h
        with open(out_lbl, 'w') as f:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")

        generated += 1
        if (i + 1) % 200 == 0:
            print(f"  Generated {i + 1}/{args.num} images...")

    # Generate dataset.yaml
    yaml_path = out_dir / 'dataset.yaml'
    yaml_content = f"""# YOLO Watermark Detection Dataset
# Generated by generate_watermark_dataset.py
path: {out_dir.resolve()}
train: train/images
val: val/images

# Classes
nc: 1
names: ['watermark']
"""
    yaml_path.write_text(yaml_content, encoding='utf-8')

    print(f"\n[OK] Dataset generated: {generated} images")
    print(f"  Train: {train_img} ({max(0, generated - num_val)} images)")
    print(f"  Val:   {val_img} ({min(generated, num_val)} images)")
    print(f"  YAML:  {yaml_path}")
    print()
    print("Next steps:")
    print(f"  1. Train YOLO: yolo train data={yaml_path} model=yolov8n.pt epochs=100 imgsz=640")
    print(f"  2. Copy model:  cp runs/detect/train/weights/best.pt ./models/watermark_yolo.pt")


if __name__ == '__main__':
    main()