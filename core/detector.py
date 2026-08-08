"""
Watermark Detector - Multi-strategy: YOLO > U2-Net > CV analysis
Version 11.0 - CV fallback disabled by default, texture filtering, YOLO conf=0.4
"""
import os
import logging
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import torch

logger = logging.getLogger(__name__)
MODEL_DIR = Path(r"D:\AI\watermark_remover\models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class WatermarkDetector:
    """
    Multi-strategy watermark detector.
    Priority: YOLO (trained) > U2-Net (saliency) > CV (traditional)
    """

    def __init__(self, device: str = 'auto'):
        self.device = self._get_device(device)
        # YOLO
        self.yolo_model = None
        self.yolo_loaded = False
        # U2-Net
        self.u2net_model = None
        self.u2net_loaded = False
        # Stats
        self.detection_stats = {'yolo': 0, 'u2net': 0, 'cv': 0}

    def _get_device(self, device: str) -> torch.device:
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)

    # ================================================================
    # YOLO Detection (Primary)
    # ================================================================

    def load_yolo(self) -> bool:
        """
        Load trained YOLO watermark detection model.
        Tries: custom model > pretrained model > fails gracefully.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.warning("ultralytics not installed, YOLO unavailable")
            logger.warning("  Run: pip install ultralytics")
            return False

        # Try custom trained model first
        custom_paths = [
            MODEL_DIR / "watermark_yolo.pt",
            MODEL_DIR / "yolo_watermark.pt",
            MODEL_DIR / "best.pt",
        ]
        for model_path in custom_paths:
            if model_path.exists():
                try:
                    self.yolo_model = YOLO(str(model_path))
                    self.yolo_loaded = True
                    logger.info(f"[OK] YOLO watermark model loaded: {model_path.name}")
                    return True
                except Exception as e:
                    logger.warning(f"Failed to load {model_path.name}: {e}")

        # Try pretrained YOLO as fallback (general object detection)
        try:
            logger.info("No custom model found, loading YOLOv8n pretrained...")
            self.yolo_model = YOLO("yolov8n.pt")
            self.yolo_loaded = True
            logger.info("[OK] YOLOv8n pretrained loaded (general detection)")
            return True
        except Exception as e:
            logger.warning(f"YOLOv8n load failed: {e}")
            return False

    def detect_with_yolo(self, image: np.ndarray,
                         conf: float = 0.25) -> List[Tuple[int, int, int, int]]:
        """
        Run YOLO detection on image.
        Uses confidence 0.25: balanced recall vs false positives.
        Returns list of (x1, y1, x2, y2) bounding boxes.
        """
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
                    # Clamp to image bounds
                    x1 = max(0, int(x1)); y1 = max(0, int(y1))
                    x2 = min(w, int(x2)); y2 = min(h, int(y2))
                    if x2 - x1 > 5 and y2 - y1 > 3:
                        # Add padding
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
    # U2-Net Detection (Secondary)
    # ================================================================

    def load_u2net(self) -> bool:
        """Load U2-Net saliency detection model."""
        try:
            from rembg import new_session
            self.u2net_model = new_session('u2net')
            self.u2net_loaded = True
            logger.info("U2-Net model loaded via rembg")
            return True
        except ImportError:
            logger.warning("rembg not installed, trying direct ONNX...")

        try:
            import onnxruntime as ort
            model_path = MODEL_DIR / "u2net.onnx"
            if not model_path.exists():
                logger.warning(f"U2-Net model not found: {model_path}")
                return False
            providers = (
                ['CUDAExecutionProvider', 'CPUExecutionProvider']
                if self.device.type == 'cuda' else ['CPUExecutionProvider']
            )
            self.u2net_model = ort.InferenceSession(str(model_path), providers=providers)
            self.u2net_loaded = True
            logger.info("U2-Net ONNX model loaded")
            return True
        except Exception as e:
            logger.error(f"U2-Net load failed: {e}")
            return False

    def detect_with_u2net(self, image: np.ndarray) -> np.ndarray:
        """
        Use U2-Net to detect salient regions (watermark candidates).
        Returns saliency mask (0-255).
        """
        if not self.u2net_loaded:
            if not self.load_u2net():
                return self._detect_fallback(image)

        try:
            from rembg import remove
            pil_img = Image.fromarray(image.astype(np.uint8))
            result = remove(pil_img, session=self.u2net_model, only_mask=True)
            if result is None:
                return self._detect_fallback(image)
            mask = np.array(result)
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
            return mask
        except Exception as e:
            logger.error(f"U2-Net detection failed: {e}")
            return self._detect_fallback(image)

    def _detect_fallback(self, image: np.ndarray) -> np.ndarray:
        """Fallback: traditional CV-based detection with adaptive thresholding."""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mask = np.zeros((h, w), dtype=np.uint8)

        # Bottom region text detection with adaptive thresholds
        bottom_h = int(h * 0.25)  # Expanded from 0.2
        bottom = gray[h - bottom_h:, :]
        for thresh_mode in [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV]:
            _, binary = cv2.threshold(bottom, 0, 255, thresh_mode + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 8))
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 100:  # Lowered from 200
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    pad_w = int(bw * 0.35) + 18
                    pad_h = int(bh * 0.65) + 10
                    x1 = max(0, x - pad_w)
                    y1 = max(0, h - bottom_h + y - pad_h)
                    x2 = min(w, x + bw + pad_w)
                    y2 = min(h, h - bottom_h + y + bh + pad_h)
                    mask[y1:y2, x1:x2] = 255

        # Four corners detection - expanded regions
        corners = [
            (0, 0, int(w * 0.30), int(h * 0.30)),          # top-left
            (int(w * 0.70), 0, w, int(h * 0.30)),          # top-right
            (0, int(h * 0.70), int(w * 0.30), h),          # bottom-left
            (int(w * 0.70), int(h * 0.70), w, h),          # bottom-right
        ]
        for cx1, cy1, cx2, cy2 in corners:
            corner = gray[cy1:cy2, cx1:cx2]
            if corner.size > 0:
                std = np.std(corner)
                if std > 30:  # Lowered from 40
                    mask[cy1:cy2, cx1:cx2] = 255

        return mask

    # ================================================================
    # Main Detection Interface
    # ================================================================

    def detect_watermarks(self,
                          image: np.ndarray,
                          use_ai: bool = True,
                          use_yolo: bool = True,
                          use_u2net: bool = True,
                          use_cv_fallback: bool = False) -> List[Tuple[int, int, int, int]]:
        """
        Main watermark detection. YOLO-first with low confidence (0.15).
        CV fallback is OFF by default (CV is ineffective for semi-transparent watermarks).

        Args:
            image: RGB uint8 numpy array (H, W, 3)
            use_ai: Enable AI detection (YOLO and/or U2-Net)
            use_yolo: Enable YOLO detection (primary)
            use_u2net: Enable U2-Net detection (secondary)
            use_cv_fallback: Enable CV fallback (OFF by default)

        Returns:
            List of (x1, y1, x2, y2) bounding boxes, max 5 regions
        """
        h, w = image.shape[:2]

        # ===== Strategy 1: YOLO (primary, balanced confidence) =====
        if self.yolo_loaded:
            yolo_boxes = self.detect_with_yolo(image, conf=0.25)
            if yolo_boxes:
                self.detection_stats['yolo'] += 1
                logger.info(f"YOLO detected {len(yolo_boxes)} regions "
                            f"(YOLO:{self.detection_stats['yolo']} "
                            f"U2Net:{self.detection_stats['u2net']} "
                            f"CV:{self.detection_stats['cv']})")
                # Filter bad regions (area/aspect ratio)
                yolo_boxes = self._filter_bad_regions(yolo_boxes, w, h)
                if yolo_boxes:
                    regions = self._merge_regions(yolo_boxes, w, h)
                    return regions[:5]

        # ===== Strategy 2: U2-Net saliency (secondary) =====
        if use_ai and use_u2net and self.u2net_loaded:
            saliency = self.detect_with_u2net(image)
            if saliency is not None:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
                dilated = cv2.dilate(saliency, kernel, iterations=2)
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
                    self.detection_stats['u2net'] += 1
                    regions = self._merge_regions(all_boxes, w, h)
                    logger.info(f"U2Net detected {len(regions)} regions")
                    return regions[:5]

        # ===== Strategy 3: CV fallback (only if explicitly enabled) =====
        if use_cv_fallback:
            cv_boxes = self._detect_watermark_cv(image)
            if cv_boxes:
                cv_boxes = self._filter_texture_regions(image, cv_boxes)
                self.detection_stats['cv'] += 1
                regions = self._merge_regions(cv_boxes, w, h)
                logger.info(f"CV fallback detected {len(regions)} regions")
                return regions[:5]

        logger.info("YOLO detected nothing - watermark may be too faint. Try manual mode.")
        return []

    def _detect_watermark_cv(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Traditional CV-based multi-strategy detection with adaptive thresholds."""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        boxes = []

        # --- Corner text detection with adaptive (OTSU) thresholding ---
        # Expanded search regions for better corner coverage
        corner_cfgs = [
            # (name, x_start, y_start, x_end, y_end, min_area, min_w, min_h, kernel_w, kernel_h)
            ("top-left",     0.0,  0.0,  0.35, 0.20,  60, 12,  4,  8,  3),
            ("top-right",    0.65, 0.0,  1.0,  0.20,  60, 12,  4,  8,  3),
            ("bottom-left",  0.0,  0.80, 0.35, 1.0,   60, 12,  4,  8,  3),
            ("bottom-right", 0.45, 0.80, 1.0,  1.0,   80, 15,  4, 15,  5),
            # Additional: mid-left and mid-right for side watermarks
            ("mid-left",     0.0,  0.20, 0.20, 0.80,  50, 10,  4,  6,  3),
            ("mid-right",    0.80, 0.20, 1.0,  0.80,  50, 10,  4,  6,  3),
            # Additional: top-center and bottom-center for centered watermarks
            ("top-center",   0.30, 0.0,  0.70, 0.15,  60, 15,  4, 10,  3),
            ("bottom-center",0.30, 0.85, 0.70, 1.0,   60, 15,  4, 10,  3),
        ]
        for name, xs, ys, xe, ye, min_a, min_w, min_h, kw, kh in corner_cfgs:
            x1_r = int(w * xs); y1_r = int(h * ys)
            x2_r = int(w * xe); y2_r = int(h * ye)
            if x2_r <= x1_r + 10 or y2_r <= y1_r + 10:
                continue
            region_gray = gray[y1_r:y2_r, x1_r:x2_r]
            if region_gray.size == 0:
                continue
            mean_val = np.mean(region_gray)
            if mean_val < 15 or mean_val > 240:
                continue

            # Adaptive thresholding: try OTSU first, then fixed thresholds
            for thresh_method in ['otsu', 'fixed_light', 'fixed_dark']:
                if thresh_method == 'otsu':
                    try:
                        _, binary = cv2.threshold(region_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    except Exception:
                        continue
                elif thresh_method == 'fixed_light':
                    _, binary = cv2.threshold(region_gray, 110, 255, cv2.THRESH_BINARY)
                else:  # fixed_dark
                    _, binary = cv2.threshold(region_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
                closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    rx, ry, rbw, rbh = cv2.boundingRect(cnt)
                    if area < min_a or rbw < min_w or rbh < min_h:
                        continue
                    aspect = rbw / max(rbh, 1)
                    if aspect < 0.6:  # Relaxed: allow slightly narrower text
                        continue
                    region_area = (x2_r - x1_r) * (y2_r - y1_r)
                    if area > region_area * 0.6:  # Relaxed from 0.5
                        continue
                    pad_w = int(rbw * 0.3) + 15
                    pad_h = int(rbh * 0.45) + 12
                    bx1 = max(0, x1_r + rx - pad_w)
                    by1 = max(0, y1_r + ry - pad_h)
                    bx2 = min(w, x1_r + rx + rbw + pad_w)
                    by2 = min(h, y1_r + ry + rbh + pad_h)
                    boxes.append((bx1, by1, bx2, by2))
                break  # Use first successful threshold method only

        # --- Edge density (logo/icon detection) ---
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_abs = np.abs(lap)
        lap_blur = cv2.GaussianBlur(lap_abs, (21, 21), 0)
        lap_norm = cv2.normalize(lap_blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, tex_binary = cv2.threshold(lap_norm, 30, 255, cv2.THRESH_BINARY)
        tex_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        tex_closed = cv2.morphologyEx(tex_binary, cv2.MORPH_CLOSE, tex_kernel)
        tex_contours, _ = cv2.findContours(tex_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in tex_contours:
            area = cv2.contourArea(cnt)
            if area < 150 or area > w * h * 0.06:
                continue
            rx, ry, rbw, rbh = cv2.boundingRect(cnt)
            cx = rx + rbw // 2; cy = ry + rbh // 2
            dist_to_edge = min(cx / max(w, 1), cy / max(h, 1),
                              (w - cx) / max(w, 1), (h - cy) / max(h, 1))
            if dist_to_edge > 0.35:
                continue
            pad_w2 = int(rbw * 0.3) + 12
            pad_h2 = int(rbh * 0.3) + 12
            bx1 = max(0, rx - pad_w2); by1 = max(0, ry - pad_h2)
            bx2 = min(w, rx + rbw + pad_w2); by2 = min(h, ry + rbh + pad_h2)
            boxes.append((bx1, by1, bx2, by2))

        # --- Brightness anomaly ---
        block_h = max(h // 8, 50); block_w = max(w // 8, 50)
        for by in range(0, h, block_h):
            for bx in range(0, w, block_w):
                ex = min(bx + block_w, w); ey = min(by + block_h, h)
                block = gray[by:ey, bx:ex]
                if block.size == 0:
                    continue
                block_std = np.std(block)
                if block_std < 12:
                    continue
                _, otsu = cv2.threshold(block, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                bright_pct = np.sum(otsu > 0) / block.size
                if 0.005 < bright_pct < 0.25:
                    otsu_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 4))
                    otsu_closed = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, otsu_kernel)
                    otsu_contours, _ = cv2.findContours(otsu_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in otsu_contours:
                        area = cv2.contourArea(cnt)
                        if area < 100 or area > w * h * 0.04:
                            continue
                        rx, ry, rbw, rbh = cv2.boundingRect(cnt)
                        pad = 14
                        bx1 = max(0, bx + rx - pad); by1 = max(0, by + ry - pad)
                        bx2 = min(w, bx + rx + rbw + pad); by2 = min(h, by + ry + rbh + pad)
                        boxes.append((bx1, by1, bx2, by2))

        return boxes

    @staticmethod
    def _filter_texture_regions(image: np.ndarray,
                                 boxes: List[Tuple[int, int, int, int]],
                                 max_lap_var: float = 800.0) -> List[Tuple[int, int, int, int]]:
        """
        Filter out high-frequency natural texture regions (grass, petals, brick).
        Uses Laplacian variance as texture complexity measure.
        High var = complex texture (natural), Low var = flat/smooth (watermark).
        """
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
        """
        Filter out obviously non-watermark regions by area and aspect ratio.
        Watermarks are typically 0.1%-12% of image area, with aspect ratio 0.3-10.
        """
        img_area = w * h
        min_area_ratio = 0.001  # 0.1%
        max_area_ratio = 0.12   # 12%
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
            logger.info(f"Region filter: {len(boxes)} -> {len(good)} (removed {len(boxes) - len(good)} false positives)")
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
    # Convenience: load all
    # ================================================================

    def load_all(self) -> dict:
        """Load all available models. Returns status dict."""
        status = {}
        status['yolo'] = self.load_yolo()
        status['u2net'] = self.load_u2net()
        logger.info(f"Detector status: {status}")
        return status


if __name__ == '__main__':
    detector = WatermarkDetector()
    status = detector.load_all()
    print(f"Load status: {status}")

    test_img = r"D:\AI\watermark_remover\test_data\compare_v3.png"
    if os.path.exists(test_img):
        img = cv2.imread(test_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        regions = detector.detect_watermarks(img)
        print(f"Detected regions: {regions}")