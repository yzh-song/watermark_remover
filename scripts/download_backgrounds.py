#!/usr/bin/env python3
"""
scripts/download_backgrounds.py - Download background images from Pexels free image library.
Used for training data augmentation in watermark detection model training.

Integrates seamlessly with generate_dataset.py via --backgrounds parameter.

Usage:
    python scripts/download_backgrounds.py --api_key YOUR_PEXELS_API_KEY
    python scripts/download_backgrounds.py --api_key YOUR_KEY --output_dir data/backgrounds --num_per_scene 80

API Key: Register for free at https://www.pexels.com/api/
License: https://www.pexels.com/license/ (free for commercial & non-commercial use)
"""

import os
import time
import requests
import argparse
from pathlib import Path

PEXELS_API_URL = "https://api.pexels.com/v1/search"

# Scene categories and their search queries
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


def download_images(api_key: str, output_dir: str,
                    num_per_scene: int = 80,
                    min_width: int = 800,
                    min_height: int = 600) -> int:
    """
    Download background images from Pexels for all scene categories.

    Args:
        api_key: Pexels API key
        output_dir: Output root directory (subdirs: indoor/, outdoor/, landscape/, text_background/)
        num_per_scene: Number of images per sub-query (total ~ num_per_scene * 6 * 4 = 1920)
        min_width: Minimum image width to accept
        min_height: Minimum image height to accept

    Returns:
        Total number of images downloaded
    """
    headers = {"Authorization": api_key}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total_downloaded = 0

    for scene, queries in SCENE_QUERIES.items():
        scene_dir = output_path / scene
        scene_dir.mkdir(exist_ok=True)
        downloaded = len(list(scene_dir.glob("*.jpg")))
        print(f"[{scene}] Existing: {downloaded}, target per query: {num_per_scene}")

        for query in queries:
            query_downloaded = 0
            page = 1

            while query_downloaded < num_per_scene:
                params = {
                    "query": query,
                    "per_page": 20,       # Max 80 per page, using 20 to spread queries
                    "page": page,
                    "orientation": "landscape",
                    "size": "medium"
                }

                try:
                    response = requests.get(PEXELS_API_URL, headers=headers,
                                           params=params, timeout=15)
                except requests.exceptions.RequestException as e:
                    print(f"  [ERROR] Request failed: {e}")
                    break

                if response.status_code != 200:
                    print(f"  [ERROR] HTTP {response.status_code}: {response.text[:200]}")
                    break

                data = response.json()
                photos = data.get("photos", [])
                if not photos:
                    print(f"  No more results for '{query}' (page {page})")
                    break

                for photo in photos:
                    # Pick best available resolution
                    src = (photo["src"].get("large2x") or
                           photo["src"].get("large") or
                           photo["src"]["original"])
                    if not src:
                        continue

                    # Skip images that are too small
                    width = photo.get("width", 0)
                    height = photo.get("height", 0)
                    if width < min_width or height < min_height:
                        continue

                    # Use photo ID for deduplication
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
                        print(f"  [OK] [{scene}/{query}] "
                              f"{query_downloaded}/{num_per_scene}: {filename}")
                        # Rate limiting: free tier = 200 req/hour
                        time.sleep(0.2)
                    except Exception as e:
                        print(f"  [FAIL] Download error: {src[:60]}... -> {e}")

                page += 1
                # Additional delay between pages
                time.sleep(0.5)

    print(f"\n[Total] Downloaded {total_downloaded} images to {output_dir}")
    print(f"  Indoor:         {len(list((output_path / 'indoor').glob('*.jpg')))} images")
    print(f"  Outdoor:        {len(list((output_path / 'outdoor').glob('*.jpg')))} images")
    print(f"  Landscape:      {len(list((output_path / 'landscape').glob('*.jpg')))} images")
    print(f"  Text Background: {len(list((output_path / 'text_background').glob('*.jpg')))} images")

    return total_downloaded


def main():
    parser = argparse.ArgumentParser(
        description="Download training background images from Pexels free image library. "
                    "Integrates with generate_dataset.py for watermark detection model training."
    )
    parser.add_argument(
        "--api_key", required=True,
        help="Pexels API key. Register for free at https://www.pexels.com/api/"
    )
    parser.add_argument(
        "--output_dir", default="data/backgrounds",
        help="Output directory for downloaded backgrounds (default: data/backgrounds)"
    )
    parser.add_argument(
        "--num_per_scene", type=int, default=80,
        help="Images per scene sub-query (default: 80). "
             "Total approx: num_per_scene * 6 queries * 4 scenes = 1920 images"
    )
    parser.add_argument(
        "--min_width", type=int, default=800,
        help="Minimum image width to accept (default: 800)"
    )
    parser.add_argument(
        "--min_height", type=int, default=600,
        help="Minimum image height to accept (default: 600)"
    )
    args = parser.parse_args()

    # Validate API key
    if not args.api_key or args.api_key == "YOUR_API_KEY":
        print("=" * 60)
        print("[ERROR] Please provide a valid Pexels API key.")
        print("  Register for free at: https://www.pexels.com/api/")
        print("  Then run: python scripts/download_backgrounds.py --api_key YOUR_KEY")
        print("=" * 60)
        return

    print("=" * 60)
    print("Pexels Background Image Downloader")
    print("=" * 60)
    print(f"Output dir:     {args.output_dir}")
    print(f"Per sub-query:  {args.num_per_scene}")
    print(f"Min resolution: {args.min_width}x{args.min_height}")
    print(f"Estimated total: ~{args.num_per_scene * 6 * 4} images")
    print("=" * 60)
    print()

    download_images(
        api_key=args.api_key,
        output_dir=args.output_dir,
        num_per_scene=args.num_per_scene,
        min_width=args.min_width,
        min_height=args.min_height
    )

    print()
    print("=" * 60)
    print("[Next Step] Use the downloaded backgrounds for training:")
    print(f"  python scripts/generate_dataset.py \\")
    print(f"    --backgrounds {args.output_dir}/indoor {args.output_dir}/outdoor \\")
    print(f"                 {args.output_dir}/landscape {args.output_dir}/text_background \\")
    print(f"    --num 5000 --negative_ratio 0.1 --distractors")
    print("=" * 60)


if __name__ == "__main__":
    main()