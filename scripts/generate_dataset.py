"""
scripts/generate_dataset.py - Generate YOLO training dataset with watermark text/logo.
Version 12.0 - Chinese text watermark synthesis for YOLO training.

Usage:
    python scripts/generate_dataset.py --num 2000 --output_dir ./WatermarkDataset/yolo_text --text "AI Generated"
    python scripts/generate_dataset.py --num 3000 --bg_dir "D:/my_photos" --text "Watermark"
"""

import os
import random
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Chinese font candidates
_CHINESE_FONT_CANDIDATES = [
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "simhei.ttf", "simsun.ttc", "msyh.ttc",
]

_FONT_PATH = None
for _fp in _CHINESE_FONT_CANDIDATES:
    if os.path.exists(_fp):
        _FONT_PATH = _fp
        break


def generate_text_watermark(text: str, size: int, font_path: str = None) -> np.ndarray:
    """Generate transparent PNG with watermark text. Returns RGBA numpy array."""
    fp = font_path or _FONT_PATH
    try:
        font = ImageFont.truetype(fp, size) if fp else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    temp_img = Image.new('RGBA', (size * len(text) * 2, size * 2), (0, 0, 0, 0))
    temp_draw = ImageDraw.Draw(temp_img)
    bbox = temp_draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if w <= 0 or h <= 0:
        w, h = size * len(text), size

    pad = max(4, size // 8)
    img = Image.new('RGBA', (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.text((pad, pad), text, font=font, fill=(255, 255, 255, 255))
    return np.array(img)


def composite_watermark(background: np.ndarray, watermark: np.ndarray,
                        pos: tuple, opacity: float) -> np.ndarray:
    """Composite watermark onto background with alpha blending."""
    h, w = watermark.shape[:2]
    x, y = pos
    x = max(0, min(x, background.shape[1] - 1))
    y = max(0, min(y, background.shape[0] - 1))
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
    parser = argparse.ArgumentParser(description='Generate YOLO training dataset with watermark text')
    parser.add_argument('--num', type=int, default=2000, help='Number of images')
    parser.add_argument('--output_dir', type=str, default='./WatermarkDataset/yolo_text',
                        help='Output directory')
    parser.add_argument('--bg_dir', type=str, default='./WatermarkDataset/backgrounds',
                        help='Background images directory')
    parser.add_argument('--img_size', type=int, default=640, help='Output image size')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--text', type=str, nargs='+', default=['AI Generated'],
                        help='Watermark text(s)')
    parser.add_argument('--font', type=str, default=None, help='Path to font file')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    train_img = out_dir / 'train' / 'images'
    train_lbl = out_dir / 'train' / 'labels'
    val_img = out_dir / 'val' / 'images'
    val_lbl = out_dir / 'val' / 'labels'
    for p in [train_img, train_lbl, val_img, val_lbl]:
        p.mkdir(parents=True, exist_ok=True)

    bg_dir = Path(args.bg_dir)
    bg_files = []
    if bg_dir.exists():
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            bg_files.extend(bg_dir.glob(ext))
    if not bg_files:
        print(f"[WARN] No background images found in {bg_dir}. Using solid color backgrounds.")
        bg_files = None

    watermark_texts = args.text
    font_path = args.font or _FONT_PATH
    if font_path:
        print(f"Using font: {font_path}")
    else:
        print("[WARN] No Chinese font found. Watermark text may not render correctly.")

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

        if bg_files:
            bg_path = random.choice(bg_files)
            bg = cv2.imread(str(bg_path))
            if bg is None:
                continue
            bg = cv2.resize(bg, (args.img_size, args.img_size))
        else:
            color = [random.randint(30, 225) for _ in range(3)]
            bg = np.full((args.img_size, args.img_size, 3), color, dtype=np.uint8)

        h, w = bg.shape[:2]

        text = random.choice(watermark_texts)
        font_size = random.randint(int(args.img_size * 0.05), int(args.img_size * 0.20))
        opacity = random.uniform(0.4, 0.9)

        wm = generate_text_watermark(text, font_size, font_path)
        wm_h, wm_w = wm.shape[:2]
        if wm_w < 5 or wm_h < 5:
            continue

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
        else:
            x = random.randint(w - wm_w - w // 4, w - wm_w - margin)
            y = random.randint(h - wm_h - h // 4, h - wm_h - margin)

        x = max(0, min(x, w - wm_w))
        y = max(0, min(y, h - wm_h))

        bg = composite_watermark(bg, wm, (x, y), opacity)
        cv2.imwrite(str(out_img), bg, [cv2.IMWRITE_JPEG_QUALITY, 95])

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

    yaml_path = out_dir / 'dataset.yaml'
    yaml_content = f"""# YOLO Watermark Detection Dataset
path: {out_dir.resolve()}
train: train/images
val: val/images

nc: 1
names: ['watermark']
"""
    yaml_path.write_text(yaml_content, encoding='utf-8')

    print(f"\n[OK] Dataset generated: {generated} images")
    print(f"  Train: {train_img}")
    print(f"  Val:   {val_img}")
    print(f"  YAML:  {yaml_path}")


if __name__ == '__main__':
    main()