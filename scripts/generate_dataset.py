"""
scripts/generate_dataset.py - Generate YOLO training dataset with watermark text/logo.
Version 12.0 - Multi-background dirs, negative samples, distractors, watermark rotation, Pexels API.

Usage:
    python scripts/generate_dataset.py --num 2000 --output_dir ./WatermarkDataset/yolo_text --text "AI Generated"
    python scripts/generate_dataset.py --num 3000 --backgrounds data/backgrounds/indoor data/backgrounds/outdoor --text "Watermark"
    python scripts/generate_dataset.py --num 5000 --backgrounds data/backgrounds/indoor data/backgrounds/outdoor data/backgrounds/landscape data/backgrounds/text_background --negative_ratio 0.1 --distractors --pexels_api_key YOUR_KEY
"""

import os
import sys
import random
import argparse
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Chinese font candidates (Windows)
_CHINESE_FONT_CANDIDATES = [
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simkai.ttf",
    "C:/Windows/Fonts/STKAITI.TTF",
    "simhei.ttf", "simsun.ttc", "msyh.ttc",
]

_FONT_PATH = None
for _fp in _CHINESE_FONT_CANDIDATES:
    if os.path.exists(_fp):
        _FONT_PATH = _fp
        break

_FONT_FALLBACK = _FONT_PATH is None

# Pexels API URL
PEXELS_API_URL = "https://api.pexels.com/v1/search"

# Pexels scene queries for background download
SCENE_QUERIES = {
    "indoor": [
        "living room", "office desk", "kitchen", "bedroom interior",
        "coffee shop interior", "library interior"
    ],
    "outdoor": [
        "city street", "park nature", "beach sunset", "forest path",
        "mountain landscape", "lake reflection"
    ],
    "landscape": [
        "landscape nature", "countryside field", "desert dunes",
        "river valley", "autumn forest", "winter snow"
    ],
    "text_background": [
        "newspaper background", "book page", "magazine cover",
        "poster with text", "sign board", "texture with words"
    ]
}


def download_pexels_backgrounds(api_key: str, output_dir: str,
                                 num_per_scene: int = 80,
                                 min_width: int = 800,
                                 min_height: int = 600) -> int:
    """
    Download background images from Pexels for training data augmentation.

    Args:
        api_key: Pexels API key
        output_dir: Output root directory
        num_per_scene: Number of images per scene sub-query
        min_width: Minimum image width
        min_height: Minimum image height

    Returns:
        Total number of downloaded images
    """
    try:
        import requests
    except ImportError:
        print("[ERROR] requests library not installed. Run: pip install requests")
        return 0

    headers = {"Authorization": api_key}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_downloaded = 0
    for scene, queries in SCENE_QUERIES.items():
        scene_dir = output_path / scene
        scene_dir.mkdir(exist_ok=True)
        downloaded = len(list(scene_dir.glob("*.jpg")))
        print(f"[Pexels] [{scene}] existing: {downloaded}, target per query: {num_per_scene}")

        for query in queries:
            query_downloaded = 0
            page = 1
            while query_downloaded < num_per_scene:
                params = {
                    "query": query,
                    "per_page": 20,
                    "page": page,
                    "orientation": "landscape",
                    "size": "medium"
                }
                try:
                    response = requests.get(PEXELS_API_URL, headers=headers, params=params, timeout=15)
                except Exception as e:
                    print(f"  [ERROR] Request failed: {e}")
                    break

                if response.status_code != 200:
                    print(f"  [ERROR] HTTP {response.status_code}: {response.text[:200]}")
                    break

                data = response.json()
                photos = data.get("photos", [])
                if not photos:
                    break

                for photo in photos:
                    src = photo["src"].get("large2x") or photo["src"].get("large") or photo["src"]["original"]
                    if not src:
                        continue
                    width = photo.get("width", 0)
                    height = photo.get("height", 0)
                    if width < min_width or height < min_height:
                        continue

                    filename = f"{query.replace(' ', '_')}_{photo['id']}.jpg"
                    filepath = scene_dir / filename
                    if filepath.exists():
                        query_downloaded += 1
                        continue

                    try:
                        img_data = requests.get(src, timeout=15).content
                        with open(filepath, "wb") as f:
                            f.write(img_data)
                        query_downloaded += 1
                        total_downloaded += 1
                        print(f"  [OK] [{scene}/{query}] {query_downloaded}/{num_per_scene}: {filename}")
                        time.sleep(0.2)
                    except Exception as e:
                        print(f"  [FAIL] {src[:60]}... error: {e}")

                page += 1
                time.sleep(0.5)

    print(f"[Pexels] Total downloaded: {total_downloaded} images -> {output_dir}")
    return total_downloaded


def load_background_paths(background_dirs: list) -> list:
    """
    Load all image paths from multiple background directories.

    Args:
        background_dirs: List of directory paths to scan

    Returns:
        List of Path objects for all found images
    """
    paths = []
    for d in background_dirs:
        bg_dir = Path(d)
        if not bg_dir.exists():
            print(f"[WARN] Background directory not found: {bg_dir}")
            continue
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            found = list(bg_dir.rglob(ext))
            paths.extend(found)
        print(f"  Loaded {len([p for p in paths if p.parent == bg_dir or bg_dir in p.parents])} "
              f"images from {bg_dir}")
    return paths


def generate_text_watermark(text: str, size: int, font_path: str = None) -> np.ndarray:
    """Generate transparent PNG with watermark text. Returns RGBA numpy array."""
    fp = font_path or _FONT_PATH
    font = None

    if fp:
        try:
            font = ImageFont.truetype(fp, size)
        except Exception:
            font = None

    if font is None:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    if font is not None:
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

    return generate_text_watermark_cv2(text, size)


def generate_text_watermark_cv2(text: str, size: int) -> np.ndarray:
    """
    OpenCV putText fallback for watermark text generation.
    Only supports ASCII characters; non-ASCII will be replaced with 'WATERMARK'.
    """
    try:
        text.encode('ascii')
    except UnicodeEncodeError:
        print(f"[WARN] Chinese text '{text}' cannot be rendered without Chinese fonts.")
        print("  Download Chinese fonts: https://github.com/adobe-fonts/source-han-sans/releases")
        print("  Or place simhei.ttf in C:/Windows/Fonts/")
        print("  Using 'WATERMARK' as placeholder text.")
        text = "WATERMARK"

    font_scale = size / 30.0
    thickness = max(2, int(font_scale * 2))
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

    pad = max(4, size // 8)
    img_w = text_w + pad * 2
    img_h = text_h + baseline + pad * 2

    rgba = np.zeros((img_h, img_w, 4), dtype=np.uint8)
    cv2.putText(rgba, text, (pad, pad + text_h),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255, 255), thickness)

    gray = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2GRAY)
    rgba[:, :, 3] = gray
    return rgba


def rotate_watermark(watermark: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate watermark image by the given angle (degrees).

    Args:
        watermark: RGBA numpy array
        angle: Rotation angle in degrees (-15 to 15 for subtle rotation)

    Returns:
        Rotated RGBA numpy array
    """
    if abs(angle) < 0.5:
        return watermark

    h, w = watermark.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Compute new bounds
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    matrix[0, 2] += new_w / 2 - center[0]
    matrix[1, 2] += new_h / 2 - center[1]

    rotated = cv2.warpAffine(
        watermark, matrix, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )
    return rotated


def add_distractors(background: np.ndarray, num_distractors: int = 3) -> np.ndarray:
    """
    Add random distracting text/shapes to the background to simulate real-world complexity.

    Args:
        background: BGR numpy array
        num_distractors: Number of distracting elements to add

    Returns:
        Background with distractors added
    """
    h, w = background.shape[:2]
    result = background.copy()

    for _ in range(num_distractors):
        distractor_type = random.choice(['text', 'line', 'circle', 'rect'])

        if distractor_type == 'text':
            # Random small text
            fake_text = random.choice(['Note', 'Draft', 'Sample', 'Ref', 'Copy', 'V1', 'X'])
            font_scale = random.uniform(0.3, 0.7)
            thickness = random.randint(1, 2)
            color = tuple(random.randint(30, 180) for _ in range(3))
            x = random.randint(10, w - 100)
            y = random.randint(10, h - 10)
            cv2.putText(result, fake_text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

        elif distractor_type == 'line':
            color = tuple(random.randint(30, 180) for _ in range(3))
            thickness = random.randint(1, 3)
            x1 = random.randint(0, w)
            y1 = random.randint(0, h)
            x2 = random.randint(0, w)
            y2 = random.randint(0, h)
            cv2.line(result, (x1, y1), (x2, y2), color, thickness)

        elif distractor_type == 'circle':
            color = tuple(random.randint(30, 180) for _ in range(3))
            radius = random.randint(5, 30)
            x = random.randint(radius, w - radius)
            y = random.randint(radius, h - radius)
            cv2.circle(result, (x, y), radius, color, -1)

        elif distractor_type == 'rect':
            color = tuple(random.randint(30, 180) for _ in range(3))
            x1 = random.randint(0, w - 50)
            y1 = random.randint(0, h - 30)
            x2 = x1 + random.randint(20, 80)
            y2 = y1 + random.randint(10, 40)
            cv2.rectangle(result, (x1, y1), (x2, y2), color, -1)

    return result


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


def generate_yolo_label(box: tuple, img_w: int, img_h: int) -> str:
    """
    Generate YOLO format label string.

    Args:
        box: (x1, y1, x2, y2) bounding box
        img_w: Image width
        img_h: Image height

    Returns:
        YOLO label line: "0 x_center y_center width height"
    """
    x1, y1, x2, y2 = box
    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"


def main():
    parser = argparse.ArgumentParser(
        description='Generate YOLO training dataset with watermark text/logo. v12.0'
    )
    parser.add_argument('--num', type=int, default=2000, help='Total number of images to generate')
    parser.add_argument('--output_dir', type=str, default='./WatermarkDataset/yolo_text',
                        help='Output directory')
    parser.add_argument('--bg_dir', type=str, default='./WatermarkDataset/backgrounds',
                        help='Background images directory (single, legacy)')
    parser.add_argument('--backgrounds', nargs='+', default=None,
                        help='Multiple background directories (e.g. data/backgrounds/indoor data/backgrounds/outdoor)')
    parser.add_argument('--img_size', type=int, default=640, help='Output image size')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--text', type=str, nargs='+', default=['AI Generated'],
                        help='Watermark text(s)')
    parser.add_argument('--font', type=str, default=None, help='Path to font file')
    parser.add_argument('--negative_ratio', type=float, default=0.0,
                        help='Ratio of negative samples (no watermark, 0.0-1.0)')
    parser.add_argument('--distractors', action='store_true', default=False,
                        help='Add random distracting text/shapes to backgrounds')
    parser.add_argument('--max_rotation_angle', type=float, default=15.0,
                        help='Maximum watermark rotation angle in degrees')
    parser.add_argument('--pexels_api_key', type=str, default=None,
                        help='Pexels API key for downloading background images')
    parser.add_argument('--pexels_output_dir', type=str, default='data/backgrounds',
                        help='Output directory for Pexels downloaded backgrounds')
    parser.add_argument('--pexels_num_per_scene', type=int, default=80,
                        help='Number of images per Pexels scene sub-query')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    train_img = out_dir / 'train' / 'images'
    train_lbl = out_dir / 'train' / 'labels'
    val_img = out_dir / 'val' / 'images'
    val_lbl = out_dir / 'val' / 'labels'
    for p in [train_img, train_lbl, val_img, val_lbl]:
        p.mkdir(parents=True, exist_ok=True)

    # --- Step 0: Download Pexels backgrounds if API key is provided ---
    if args.pexels_api_key and args.pexels_api_key != "YOUR_API_KEY":
        print("=" * 60)
        print("[Pexels] Downloading background images from Pexels...")
        print("=" * 60)
        downloaded = download_pexels_backgrounds(
            api_key=args.pexels_api_key,
            output_dir=args.pexels_output_dir,
            num_per_scene=args.pexels_num_per_scene
        )
        if downloaded > 0:
            if args.backgrounds is None:
                args.backgrounds = []
            # Add Pexels scene subdirs to backgrounds
            pexels_dir = Path(args.pexels_output_dir)
            for scene in SCENE_QUERIES.keys():
                scene_path = pexels_dir / scene
                if scene_path.exists():
                    args.backgrounds.append(str(scene_path))
        print()

    # --- Step 1: Load background images ---
    bg_files = []
    if args.backgrounds:
        print(f"Loading backgrounds from {len(args.backgrounds)} directories...")
        bg_files = load_background_paths(args.backgrounds)
    elif args.bg_dir:
        bg_dir = Path(args.bg_dir)
        if bg_dir.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                bg_files.extend(bg_dir.glob(ext))

    if not bg_files:
        print(f"[WARN] No background images found. Using solid color backgrounds.")
        bg_files = None
    else:
        print(f"Total background images: {len(bg_files)}")

    watermark_texts = args.text
    font_path = args.font or _FONT_PATH
    if font_path:
        print(f"Using font: {font_path}")
    elif _FONT_FALLBACK:
        print("[WARN] No Chinese font found. Will use OpenCV putText fallback with ASCII text.")
        print("  To use Chinese text, download a font (e.g. simhei.ttf) and place it in C:/Windows/Fonts/")

    num_val = int(args.num * args.val_ratio)
    num_train = args.num - num_val
    num_negative = int(args.num * args.negative_ratio) if args.negative_ratio > 0 else 0
    num_watermarked = args.num - num_negative

    print(f"Generating {num_train} train + {num_val} val (total: {args.num})")
    print(f"  Watermarked: {num_watermarked} | Negative (no watermark): {num_negative}")
    print(f"  Watermark texts: {watermark_texts}")
    print(f"  Image size: {args.img_size}x{args.img_size}")
    print(f"  Distractors: {args.distractors}")
    print(f"  Max rotation angle: {args.max_rotation_angle}")

    generated = 0
    negative_count = 0

    for i in range(args.num):
        is_val = i < num_val
        is_negative = (args.negative_ratio > 0 and negative_count < num_negative)
        out_img = (val_img if is_val else train_img) / f'img_{i:06d}.jpg'
        out_lbl = (val_lbl if is_val else train_lbl) / f'img_{i:06d}.txt'

        # Load or generate background
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

        if is_negative:
            # Negative sample: no watermark, just background (with optional distractors)
            if args.distractors:
                bg = add_distractors(bg, random.randint(1, 4))
            cv2.imwrite(str(out_img), bg, [cv2.IMWRITE_JPEG_QUALITY, 95])
            # Empty label file (no objects)
            with open(out_lbl, 'w') as f:
                f.write('')
            negative_count += 1
            generated += 1
            if (i + 1) % 200 == 0:
                print(f"  Generated {i + 1}/{args.num} images...")
            continue

        # Add distractors to background before watermark
        if args.distractors:
            bg = add_distractors(bg, random.randint(1, 4))

        text = random.choice(watermark_texts)
        font_size = random.randint(int(args.img_size * 0.05), int(args.img_size * 0.20))
        opacity = random.uniform(0.4, 0.9)

        wm = generate_text_watermark(text, font_size, font_path)

        # Apply rotation if enabled
        if args.max_rotation_angle > 0:
            rotation_angle = random.uniform(-args.max_rotation_angle, args.max_rotation_angle)
            wm = rotate_watermark(wm, rotation_angle)

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
        with open(out_lbl, 'w') as f:
            f.write(generate_yolo_label((x, y, x + actual_w, y + actual_h), w, h))

        generated += 1
        if (i + 1) % 200 == 0:
            print(f"  Generated {i + 1}/{args.num} images...")

    # Write dataset.yaml
    yaml_path = out_dir / 'dataset.yaml'
    yaml_content = f"""# YOLO Watermark Detection Dataset (v12.0)
path: {out_dir.resolve()}
train: train/images
val: val/images

nc: 1
names: ['watermark']

# Dataset stats
# Total images: {generated}
# Negative samples: {negative_count}
# Distractors: {args.distractors}
# Rotation: +/-{args.max_rotation_angle} deg
"""
    yaml_path.write_text(yaml_content, encoding='utf-8')

    print(f"\n[OK] Dataset generated: {generated} images ({negative_count} negative)")
    print(f"  Train: {train_img}")
    print(f"  Val:   {val_img}")
    print(f"  YAML:  {yaml_path}")


if __name__ == '__main__':
    main()