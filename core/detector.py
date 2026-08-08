"""
Watermark Detector - YOLO-first with saliency fallback and texture filtering.
Version 12.0 - Strict YOLO (no pretrained fallback), rembg/U2Net saliency, texture filter.
"""
import os
import logging
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import torch
import yaml

logger = logging.getLogger(__name__)

# Load config
def _load_config():
    cfg_path = Path(r"D:\AI\watermark_remover\config.yaml")
    if cfg_path.exists():
        with open(cfg_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}

CONFIG = _load_config()
MODEL_DIR = Path(CONFIG.get("paths", {}).get("models_dir", r"D:\AI\watermark_remover\models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DETECT_CFG = CONFIG.get("detect", {})
DEFAULT_CONF = DETECT_CFG.get("conf", 0.25)
TEXTURE_LAP_VAR_MAX = DETECT_CFG.get("texture_lap_var_max", 500.0)
FALLBACK_SALIENCY = DETECT_CFG.get("fallback_saliency", True)


class WatermarkNotFoundError(Exception):
    """Raised when no watermark is detected by any strategy.
    The message should guide the user to use manual selection mode."""
    pass


class WatermarkDetector:
    """
    Multi-strategy watermark detector.
    Priority: YOLO (trained) > U2-Net saliency (rembg) > Error.
    Strict mode: no pretrained YOLO fallback, no CV corner detection.
    """

    def __init__(self, device: str = 'auto'):
        self.device = self._get_device(device)
        self.yolo_model = None
        self.yolo_loaded = False
        self.u2net_model = None
        self.u2net_loaded = False
        self.detection_stats = {'yolo': 0, 'u2net': 0}

    def _get_device(self, device: str) -> torch.device:
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)

    # ================================================================
    # YOLO Detection (Primary)
    # ================================================================

    def load_yolo(self) -> bool:
        """Load trained YOLO watermark detection model. Strict: no pretrained fallback."""
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            return False

        custom_path = MODEL_DIR / "watermark_yolo.pt"
        if not custom_path.exists():
            logger.error(f"[FAIL] YOLO watermark model not found: {custom_path}")
            logger.error("  Run: python scripts/train.py --mode text --text 'AI Generated'")
            return False

        try:
            self.yolo_model = YOLO(str(custom_path))
            self.yolo_loaded = True
            logger.info(f"[OK] YOLO watermark model loaded: {custom_path.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            return False

    def detect_with_yolo(self, image: np.ndarray,
                         conf: float = DEFAULT_CONF) -> List[Tuple[int, int, int, int]]:
        """Run YOLO detection. Returns list of (x1, y1, x2, y2) bounding boxes."""
        if not self.yolo_loaded:
            return []

        try:
            results = self.yolo_model(image, conf=conf, verbose=False)
            if not results or len(results) == 0:
                return []

            h, w = image.shape[:2]
            boxes = []

            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1 = max(0, int(x1)); y1 = max(0, int(y1))
                    x2 = min(w, int(x2)); y2 = min(h, int(y2))
                    if x2 - x1 > 5 and y2 - y1 > 3:
                        pad_w = int((x2 - x1) * 0.15) + 8
                        pad_h = int((y2 - y1) * 0.15) + 8
                        boxes.append((
                            max(0, x1 - pad_w), max(0, y1 - pad_h),
                            min(w, x2 + pad_w), min(h, y2 + pad_h)
                        ))

            if boxes:
                self.detection_stats['yolo'] += 1
                logger.info(f"YOLO detected {len(boxes)} regions")
            return boxes

        except Exception as e:
            logger.warning(f"YOLO detection failed: {e}")
            return []

    # ================================================================
    # U2-Net / rembg Saliency Detection (Secondary)
    # ================================================================

    def load_u2net(self) -> bool:
        """Load U2-Net saliency model via rembg."""
        try:
            from rembg import new_session
            self.u2net_model = new_session('u2net')
            self.u2net_loaded = True
            logger.info("[OK] U2-Net saliency model loaded via rembg")
            return True
        except ImportError:
            logger.warning("rembg not installed. Run: pip install rembg")
            return False
        except Exception as e:
            logger.error(f"U2-Net load failed: {e}")
            return False

    def detect_with_u2net(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Use U2-Net to detect salient regions. Returns saliency mask (0-255) or None."""
        if not self.u2net_loaded:
            if not self.load_u2net():
                return None

        try:
            from rembg import remove
            pil_img = Image.fromarray(image.astype(np.uint8))
            result = remove(pil_img, session=self.u2net_model, only_mask=True)
            if result is None:
                return None
            mask = np.array(result)
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            return mask
        except Exception as e:
            logger.error(f"U2-Net detection failed: {e}")
            return None

    # ================================================================
    # Region Filtering
    # ================================================================

    @staticmethod
    def _filter_texture_regions(image: np.ndarray,
                                 boxes: List[Tuple[int, int, int, int]],
                                 max_lap_var: float = TEXTURE_LAP_VAR_MAX) -> List[Tuple[int, int, int, int]]:
        """Filter out high-frequency natural texture regions using Laplacian variance."""
        if not boxes:
            return boxes
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        filtered = []
        for x1, y1, x2, y2 in boxes:
            roi = gray[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            lap = cv2.Laplacian(roi, cv2.CV_64F)
            lap_var = lap.var()
            if lap_var < max_lap_var:
                filtered.append((x1, y1, x2, y2))
            else:
                logger.debug(f"Filtered texture region ({x1},{y1}-{x2},{y2}) lap_var={lap_var:.0f}")
        if len(filtered) < len(boxes):
            logger.info(f"Texture filter: {len(boxes)} -> {len(filtered)} regions "
                        f"(removed {len(boxes) - len(filtered)} false positives)")
        return filtered

    def _filter_bad_regions(self, boxes: List[Tuple[int, int, int, int]],
                           w: int, h: int) -> List[Tuple[int, int, int, int]]:
        """Filter out obviously non-watermark regions by area and aspect ratio."""
        img_area = w * h
        min_area_ratio = 0.001
        max_area_ratio = 0.12
        good = []
        for (x1, y1, x2, y2) in boxes:
            bw = x2 - x1
            bh = y2 - y1
            area = bw * bh
            ratio = area / img_area
            if ratio < min_area_ratio or ratio > max_area_ratio:
                logger.debug(f"Filtered by area: ({x1},{y1}-{x2},{y2}) ratio={ratio:.4f}")
                continue
            aspect = bw / max(bh, 1)
            if aspect < 0.3 or aspect > 10:
                logger.debug(f"Filtered by aspect: ({x1},{y1}-{x2},{y2}) aspect={aspect:.2f}")
                continue
            good.append((x1, y1, x2, y2))
        if len(good) < len(boxes):
            logger.info(f"Region filter: {len(boxes)} -> {len(good)} "
                        f"(removed {len(boxes) - len(good)} false positives)")
        return good

    def _merge_regions(self, regions: List[Tuple[int, int, int, int]],
                       w: int, h: int) -> List[Tuple[int, int, int, int]]:
        """Merge overlapping/nearby bounding boxes."""
        if not regions:
            return []
        regions = sorted(regions, key=lambda r: (r[0], r[1]))
        merged = []
        for r in regions:
            if not merged:
                merged.append(list(r))
                continue
            last = merged[-1]
            x1, y1, x2, y2 = r
            lx1, ly1, lx2, ly2 = last
            gap = 20
            near = not (x2 + gap < lx1 or x1 > lx2 + gap or y2 + gap < ly1 or y1 > ly2 + gap)
            if near:
                last[0] = min(lx1, x1); last[1] = min(ly1, y1)
                last[2] = max(lx2, x2); last[3] = max(ly2, y2)
            else:
                merged.append(list(r))
        return [tuple(r) for r in merged]

    # ================================================================
    # Main Detection Interface
    # ================================================================

    def detect_watermarks(self,
                          image: np.ndarray,
                          use_yolo: bool = True,
                          use_saliency: bool = True) -> List[Tuple[int, int, int, int]]:
        """
        Main watermark detection.
        YOLO-first with saliency fallback. No CV fallback.

        Returns:
            List of (x1, y1, x2, y2) bounding boxes, max 5 regions.

        Raises:
            WatermarkNotFoundError: If no watermark is detected by any strategy.
        """
        h, w = image.shape[:2]

        # Strategy 1: YOLO (primary)
        if use_yolo and self.yolo_loaded:
            yolo_boxes = self.detect_with_yolo(image, conf=DEFAULT_CONF)
            if yolo_boxes:
                yolo_boxes = self._filter_bad_regions(yolo_boxes, w, h)
                if yolo_boxes:
                    yolo_boxes = self._filter_texture_regions(image, yolo_boxes)
                    if yolo_boxes:
                        self.detection_stats['yolo'] += 1
                        regions = self._merge_regions(yolo_boxes, w, h)
                        logger.info(f"YOLO final: {len(regions)} regions")
                        return regions[:5]

        # Strategy 2: U2-Net saliency (secondary)
        if use_saliency and FALLBACK_SALIENCY and self.u2net_loaded:
            logger.info("YOLO found nothing, trying saliency detection...")
            saliency = self.detect_with_u2net(image)
            if saliency is not None and saliency.max() > 0:
                _, binary = cv2.threshold(saliency, 30, 255, cv2.THRESH_BINARY)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
                dilated = cv2.dilate(binary, kernel, iterations=2)
                contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                all_boxes = []
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > 500:
                        x, y, bw, bh = cv2.boundingRect(cnt)
                        pad_w = int(bw * 0.15) + 10
                        pad_h = int(bh * 0.15) + 10
                        x1 = max(0, x - pad_w)
                        y1 = max(0, y - pad_h)
                        x2 = min(w, x + bw + pad_w)
                        y2 = min(h, y + bh + pad_h)
                        all_boxes.append((x1, y1, x2, y2))
                if all_boxes:
                    all_boxes = self._filter_bad_regions(all_boxes, w, h)
                    all_boxes = self._filter_texture_regions(image, all_boxes)
                    if all_boxes:
                        self.detection_stats['u2net'] += 1
                        regions = self._merge_regions(all_boxes, w, h)
                        logger.info(f"Saliency detected {len(regions)} regions")
                        return regions[:5]

        # No watermark detected by any strategy -> raise error
        raise WatermarkNotFoundError(
            "No watermark detected. Please use Manual Selection mode "
            "to draw the watermark region on the canvas."
        )

    def load_all(self) -> dict:
        """Load all available models. Returns status dict."""
        status = {}
        status['yolo'] = self.load_yolo()
        status['u2net'] = self.load_u2net()
        logger.info(f"Detector status: {status}")
        return status