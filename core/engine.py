"""
Core Inpainting Engine - Orchestration layer for modular pipeline
Version 11.0 - Poisson blending, KCF tracking, EMA temporal smoothing
"""
import os
import sys
import time
import logging
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple, List, Union
import torch
import cv2

# KCF tracker compatibility: OpenCV 4.5+ moved tracking to cv2.legacy
def _create_kcf_tracker():
    """Create KCF tracker with OpenCV version compatibility."""
    for factory in [
        lambda: cv2.legacy.TrackerKCF_create(),
        lambda: cv2.TrackerKCF_create(),
    ]:
        try:
            return factory()
        except (AttributeError, cv2.error):
            continue
    return None

LOG_DIR = Path(r"D:\AI\watermark_remover\logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger('engine')
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = logging.FileHandler(LOG_DIR / 'engine.log', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

MODEL_DIR = Path(r"D:\AI\watermark_remover\models")
CACHE_DIR = Path(r"D:\AI\watermark_remover\cache")
OUTPUT_DIR = Path(r"D:\AI\watermark_remover\output")
for d in [MODEL_DIR, CACHE_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


class InpaintingEngine:
    """
    Image/video watermark removal engine.
    v11.0: Modular architecture - uses Inpainter, Segmenter, Detector sub-modules.
    """

    def __init__(self, device: str = 'auto'):
        self.device = self._get_device(device)
        # Sub-modules (lazy-loaded)
        self.inpainter = None
        self.segmenter = None
        self.detector = None
        # Module status
        self.inpainter_loaded = False
        self.segmenter_loaded = False
        self.detector_loaded = False
        self.u2net_available = False
        self.yolo_available = False
        logger.info(f"Engine v11.0 initialized on device: {self.device}")

    def _get_device(self, device: str) -> torch.device:
        if device == 'auto':
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                try:
                    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                except AttributeError:
                    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
                logger.info(f"[GPU] {gpu_name} ({gpu_mem:.1f}GB) - CUDA enabled")
                return torch.device('cuda')
            else:
                logger.info("[CPU] CUDA not available, using CPU mode")
                return torch.device('cpu')
        return torch.device(device)

    # ================================================================
    # Module Loading (lazy)
    # ================================================================

    def load_inpainter(self) -> bool:
        """Load Inpainter module (LaMa + optional SDXL)."""
        try:
            from core.inpainter import Inpainter
            self.inpainter = Inpainter(device=self.device)
            ok = self.inpainter.load_lama()
            if ok:
                self.inpainter_loaded = True
                logger.info("[OK] Inpainter module loaded (LaMa)")
            return ok
        except Exception as e:
            logger.error(f"[FAIL] Inpainter load failed: {e}")
            return False

    def load_segmenter(self) -> bool:
        """Load Segmenter module (SAM2/MobileSAM + GrabCut)."""
        try:
            from core.segmenter import Segmenter
            self.segmenter = Segmenter(device=self.device)
            self.segmenter_loaded = True
            logger.info("[OK] Segmenter module loaded")
            return True
        except Exception as e:
            logger.warning(f"[WARN] Segmenter load failed: {e}")
            return False

    def load_detector(self) -> bool:
        """Load Detector module (YOLO + U2-Net + CV)."""
        try:
            from core.detector import WatermarkDetector
            self.detector = WatermarkDetector(device=str(self.device))
            self.detector_loaded = True

            # Try YOLO first (primary)
            yolo_ok = self.detector.load_yolo()
            self.yolo_available = yolo_ok

            # Try U2-Net (secondary)
            u2_ok = self.detector.load_u2net()
            self.u2net_available = u2_ok

            if yolo_ok and u2_ok:
                logger.info("[OK] Detector: YOLO + U2-Net + CV")
            elif yolo_ok:
                logger.info("[OK] Detector: YOLO + CV")
            elif u2_ok:
                logger.info("[OK] Detector: U2-Net + CV")
            else:
                logger.info("[OK] Detector: CV-only (install ultralytics/rembg for AI)")
            return True
        except Exception as e:
            logger.warning(f"[WARN] Detector import failed: {e}")
            self.detector = None
            return False

    def load_model(self) -> bool:
        """Load all modules. Returns True if at least Inpainter is loaded."""
        ok = self.load_inpainter()
        if not ok:
            return False
        self.load_segmenter()
        self.load_detector()
        return True

    def _ensure_modules(self):
        """Ensure required modules are loaded. Raises RuntimeError if not."""
        if not self.inpainter_loaded:
            if not self.load_inpainter():
                raise RuntimeError(
                    "CRITICAL: Cannot load LaMa inpainting model. "
                    "Run: pip install simple-lama-inpainting"
                )
        if not self.segmenter_loaded:
            self.load_segmenter()
        if not self.detector_loaded:
            self.load_detector()

    # ================================================================
    # Image I/O
    # ================================================================

    def _load_image(self, image_path: str) -> np.ndarray:
        try:
            img = Image.open(image_path).convert('RGB')
            return np.array(img)
        except Exception:
            with open(image_path, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Cannot read image: {image_path}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _save_image(self, image: np.ndarray, save_path: str):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        image = np.clip(image, 0, 255).astype(np.uint8)
        ext = Path(save_path).suffix.lower()
        if ext in ['.jpg', '.jpeg']:
            cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # ================================================================
    # Mask Generation (delegates to Segmenter)
    # ================================================================

    def _create_mask_from_bboxes(self, image_shape: Tuple[int, int],
                                  bboxes: List[Tuple[int, int, int, int]],
                                  feather: int = 12,
                                  image: np.ndarray = None) -> np.ndarray:
        """
        Guaranteed mask generation from bounding boxes. Never returns empty mask
        when valid bboxes are provided. No dependency on external modules.
        """
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w, int(x2))
            y2 = min(h, int(y2))
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 255

        if np.max(mask) == 0:
            return mask

        # Simple dilation + Gaussian feather (always works, no external deps)
        ksize = max(6, feather)
        if ksize % 2 == 0:
            ksize += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        mask = cv2.dilate(mask, kernel, iterations=1)
        gauss_ksize = feather * 2 + 1
        if gauss_ksize % 2 == 0:
            gauss_ksize += 1
        mask = cv2.GaussianBlur(mask, (gauss_ksize, gauss_ksize), feather // 2)
        return mask

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
    # Watermark Detection (delegates to Detector)
    # ================================================================

    def _detect_watermark_auto(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Multi-strategy watermark detection.
        YOLO-first (conf=0.25). Falls back to CV emergency detection when YOLO fails,
        ensuring video auto mode almost never returns empty.
        """
        if self.detector is not None and self.detector_loaded:
            try:
                logger.info("Running YOLO watermark detection...")
                regions = self.detector.detect_watermarks(
                    image, use_ai=True, use_cv_fallback=False
                )
                if regions:
                    logger.info(f"Detector found {len(regions)} watermark regions")
                    return regions
            except Exception as e:
                logger.warning(f"Detector failed: {e}")

        # YOLO found nothing - try emergency CV detection with strict texture filtering
        if self.detector is not None and self.detector_loaded:
            logger.warning("YOLO found nothing, trying emergency CV detection...")
            try:
                cv_regions = self.detector.detect_watermarks(
                    image, use_ai=False, use_cv_fallback=True
                )
                if cv_regions:
                    from core.detector import WatermarkDetector
                    cv_regions = WatermarkDetector._filter_texture_regions(
                        image, cv_regions, max_lap_var=500
                    )
                    if cv_regions:
                        logger.info(f"Emergency CV found {len(cv_regions)} regions")
                        return cv_regions[:3]
            except Exception as e:
                logger.warning(f"Emergency CV detection failed: {e}")

        logger.warning("All detection strategies failed. "
                       "Try manual selection mode for best results.")
        return []

    def _detect_watermark_cv_fallback(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Inline CV fallback detection (used when Detector is unavailable)."""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        all_boxes = []

        corner_cfgs = [
            ("top-left",     0.0,  0.0,  0.30, 0.15, 110, 80,  15, 5,  6,  3),
            ("top-right",    0.70, 0.0,  1.0,  0.15, 110, 80,  15, 5,  6,  3),
            ("bottom-left",  0.0,  0.85, 0.30, 1.0,  110, 80,  15, 5,  6,  3),
            ("bottom-right", 0.48, 0.83, 1.0,  1.0,  125, 200, 30, 5, 20, 6),
        ]
        for name, xs, ys, xe, ye, thresh, min_a, min_w, min_h, kw, kh in corner_cfgs:
            x1_r = int(w * xs); y1_r = int(h * ys)
            x2_r = int(w * xe); y2_r = int(h * ye)
            if x2_r <= x1_r or y2_r <= y1_r:
                continue
            region_gray = gray[y1_r:y2_r, x1_r:x2_r]
            if region_gray.size == 0:
                continue
            mean_val = np.mean(region_gray)
            if mean_val < 30 or mean_val > 235:
                continue
            _, binary = cv2.threshold(region_gray, thresh, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                rx, ry, rbw, rbh = cv2.boundingRect(cnt)
                if area < min_a or rbw < min_w or rbh < min_h:
                    continue
                if rbw / max(rbh, 1) < 1.0:
                    continue
                region_area = (x2_r - x1_r) * (y2_r - y1_r)
                if area > region_area * 0.5:
                    continue
                pad_w = int(rbw * 0.25) + 12
                pad_h = int(rbh * 0.4) + 10
                bx1 = max(0, x1_r + rx - pad_w)
                by1 = max(0, y1_r + ry - pad_h)
                bx2 = min(w, x1_r + rx + rbw + pad_w)
                by2 = min(h, y1_r + ry + rbh + pad_h)
                all_boxes.append((bx1, by1, bx2, by2))

        if not all_boxes:
            logger.info("Auto-detect: no watermark regions found")
            return []

        all_boxes = [(max(0, b[0]), max(0, b[1]), min(w, b[2]), min(h, b[3])) for b in all_boxes]
        all_boxes = [b for b in all_boxes if (b[2] - b[0]) > 8 and (b[3] - b[1]) > 4]
        if not all_boxes:
            return []

        regions = self._merge_regions(all_boxes, w, h)
        if len(regions) > 10:
            regions.sort(key=lambda r: (r[2] - r[0]) * (r[3] - r[1]), reverse=True)
            regions = regions[:10]

        total_area = sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in regions)
        max_area = w * h * 0.15
        if total_area > max_area:
            regions.sort(key=lambda r: (r[2] - r[0]) * (r[3] - r[1]), reverse=True)
            filtered = []
            cur_area = 0
            for r in regions:
                a = (r[2] - r[0]) * (r[3] - r[1])
                # Always keep at least the first (largest) region
                if not filtered or cur_area + a <= max_area:
                    filtered.append(r)
                    cur_area += a
                else:
                    break
            regions = filtered

        logger.info(f"CV-fallback detected {len(regions)} watermark regions")
        return regions

    # ================================================================
    # Inpainting (delegates to Inpainter)
    # ================================================================

    def _run_inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Run inpainting via Inpainter module."""
        if self.inpainter_loaded:
            return self.inpainter.inpaint_lama(image, mask)
        else:
            raise RuntimeError("Inpainter not loaded")

    def _blend_result(self, original: np.ndarray, inpainted: np.ndarray,
                      mask: np.ndarray) -> np.ndarray:
        """Blend inpainted result with original. Poisson > alpha blend fallback."""
        if self.inpainter_loaded:
            from core.inpainter import Inpainter
            # Prioritize Poisson blending for seamless boundaries
            try:
                return Inpainter.poisson_blend(original, inpainted, mask)
            except Exception as e:
                logger.warning(f"Poisson blend failed: {e}, using alpha blend")
                return Inpainter.blend(original, inpainted, mask)
        # Fallback blending
        orig_h, orig_w = original.shape[:2]
        inh, inw = inpainted.shape[:2]
        if (inh, inw) != (orig_h, orig_w):
            inpainted = cv2.resize(inpainted, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        mh, mw = mask.shape[:2]
        if (mh, mw) != (orig_h, orig_w):
            mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        mask_float = mask.astype(np.float32) / 255.0
        mask_soft = cv2.GaussianBlur(mask_float, (5, 5), 1.5)
        mask_3ch = np.stack([mask_soft] * 3, axis=2)
        final = inpainted.astype(np.float32) * mask_3ch + original.astype(np.float32) * (1 - mask_3ch)
        return np.clip(final, 0, 255).astype(np.uint8)

    # ================================================================
    # Image Inpainting
    # ================================================================

    def inpaint_image(self,
                      image_path: str,
                      output_path: Optional[str] = None,
                      mask_path: Optional[str] = None,
                      bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
                      auto_detect: bool = False,
                      feather: int = 12) -> str:
        """
        Inpaint a single image.

        Args:
            image_path: Path to input image
            output_path: Output path (auto-generated if None)
            mask_path: Optional pre-made mask image
            bboxes: List of (x1, y1, x2, y2) bounding boxes for manual mode
            auto_detect: Whether to auto-detect watermarks
            feather: Feather radius for mask edges

        Returns:
            Path to output image
        """
        self._ensure_modules()
        logger.info(f"Processing image: {image_path}")
        start_time = time.time()

        image = self._load_image(image_path)
        h, w = image.shape[:2]

        # --- Build mask ---
        if mask_path and os.path.exists(mask_path):
            logger.info(f"Using pre-generated mask: {mask_path}")
            mask = self._load_image(mask_path)
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        else:
            mask_masks = []
            # Manual mode: use smaller feather for precise small-region selection
            manual_feather = 6 if bboxes else feather
            if bboxes:
                mask_masks.append(self._create_mask_from_bboxes((h, w), bboxes, manual_feather))
            if auto_detect:
                detected = self._detect_watermark_auto(image)
                if not detected:
                    logger.warning("Auto-detect found NO watermark regions. Returning original image.")
                    logger.warning("Tip: Try manual selection mode, or install rembg/ultralytics for AI detection.")
                else:
                    mask_masks.append(self._create_mask_from_bboxes((h, w), detected, feather))
            if mask_masks:
                mask = np.clip(np.maximum.reduce(mask_masks), 0, 255).astype(np.uint8)
            else:
                mask = np.zeros((h, w), dtype=np.uint8)

        # --- Empty mask check ---
        if np.max(mask) < 10:
            logger.warning("Mask is empty - returning original image unchanged.")
            if output_path is None:
                output_path = str(OUTPUT_DIR / f"result_{Path(image_path).stem}.png")
            self._save_image(image, output_path)
            return output_path

        mask_pixels = np.sum(mask > 10)
        logger.info(f"Mask has {mask_pixels} pixels to inpaint ({mask_pixels / (w * h) * 100:.1f}% of image)")

        # --- Inpaint + Blend ---
        logger.info("Running LaMa inpainting...")
        result_np = self._run_inpaint(image, mask)
        final_np = self._blend_result(image, result_np, mask)
        logger.info("Inpainting completed successfully")

        if output_path is None:
            output_path = str(OUTPUT_DIR / f"result_{Path(image_path).stem}.png")
        self._save_image(final_np, output_path)

        elapsed = time.time() - start_time
        logger.info(f"[DONE] Image saved: {output_path} ({elapsed:.1f}s)")
        return output_path

    # ================================================================
    # Video Inpainting
    # ================================================================

    def _create_video_writer(self, output_path: str, fps: float,
                              width: int, height: int) -> Tuple[cv2.VideoWriter, str]:
        """Create video writer with H.264 priority."""
        codecs_to_try = [
            ('avc1', '.mp4'),
            ('X264', '.mp4'),
            ('H264', '.mp4'),
            ('mp4v', '.mp4'),
            ('MJPG', '.avi'),
        ]
        for codec, ext in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                test_path = str(Path(output_path).with_suffix(ext))
                test_writer = cv2.VideoWriter(test_path, fourcc, fps, (width, height))
                if test_writer.isOpened():
                    logger.info(f"Using codec: {codec} -> {test_path}")
                    return test_writer, test_path
                test_writer.release()
            except Exception as e:
                logger.warning(f"Codec {codec} failed: {e}")
                continue
        raise RuntimeError("Cannot create video writer with any available codec")

    def inpaint_video(self,
                      video_path: str,
                      output_path: Optional[str] = None,
                      bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
                      auto_detect: bool = False,
                      start_frame: int = 0,
                      end_frame: Optional[int] = None,
                      progress_callback=None) -> str:
        """
        Inpaint video watermarks frame by frame with tracking re-detection.

        Args:
            video_path: Path to input video
            output_path: Output path (auto-generated if None)
            bboxes: Manual bounding boxes (static mask mode)
            auto_detect: Auto-detect watermark (re-detects every 30 frames)
            start_frame: Start frame index
            end_frame: End frame index (None = all frames)
            progress_callback: Optional callback(percent) for progress

        Returns:
            Path to output video
        """
        self._ensure_modules()
        logger.info(f"Processing video: {video_path}")
        start_time = time.time()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps > 120:
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if end_frame is None:
            end_frame = total_frames

        logger.info(f"Video: {width}x{height}, {fps}fps, {total_frames} frames")
        logger.info(f"Processing frames {start_frame}-{end_frame}")

        if output_path is None:
            output_path = str(OUTPUT_DIR / f"result_{Path(video_path).stem}.mp4")

        writer, final_output = self._create_video_writer(output_path, fps, width, height)

        if self.detector is None and auto_detect:
            self.load_detector()

        # --- Determine mask strategy ---
        use_static_mask = (bboxes is not None and not auto_detect)
        re_detect_interval = 60  # Extended: KCF tracker bridges the gap

        first_frame_mask = None
        tracker = None
        tracker_bbox = None  # (x, y, w, h) for KCF
        prev_frame_rgb = None  # For EMA temporal smoothing

        if use_static_mask:
            first_frame_mask = self._create_mask_from_bboxes((height, width), bboxes, feather=6)
            if np.max(first_frame_mask) > 10:
                logger.info(f"Static mask: {np.sum(first_frame_mask > 30)} pixels")
            else:
                logger.warning("Static mask is empty - all frames will be written as-is")

        # --- Multi-frame first-frame detection (auto mode) ---
        # Scan first 5 frames to find the best watermark region, avoiding
        # single-frame detection failure aborting the entire video.
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        if auto_detect and not use_static_mask:
            best_area = 0
            best_bbox = None
            best_frame_rgb = None
            scan_frames = min(5, end_frame - start_frame)
            for scan_i in range(scan_frames):
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detected = self._detect_watermark_auto(frame_rgb)
                if detected:
                    for bbox in detected:
                        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        if area > best_area:
                            best_area = area
                            best_bbox = bbox
                            best_frame_rgb = frame_rgb.copy()
            # Reset to start frame after scanning
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            if best_bbox:
                first_frame_mask = self._create_mask_from_bboxes(
                    (height, width), [best_bbox], feather=12
                )
                # Initialize KCF tracker from the best detection
                try:
                    x1, y1, x2, y2 = best_bbox
                    tracker = _create_kcf_tracker()
                    if tracker is not None:
                        tracker.init(best_frame_rgb, (x1, y1, x2 - x1, y2 - y1))
                        logger.info(f"KCF tracker initialized from multi-frame scan: "
                                    f"({x1},{y1},{x2-x1},{y2-y1}) area={best_area}")
                    else:
                        logger.warning("KCF tracker unavailable (OpenCV contrib/legacy not installed)")
                except Exception as e:
                    logger.warning(f"KCF tracker init failed: {e}")
                    tracker = None
                logger.info(f"Initial watermark mask from {scan_frames}-frame scan: "
                            f"{np.sum(first_frame_mask > 30)} pixels")
            else:
                logger.warning(f"No watermark detected in first {scan_frames} frames. "
                               "Video will be written as-is. Try manual selection mode.")

        frame_idx = start_frame
        failed_frames = 0

        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # --- Dynamic mask update (auto-detect + KCF tracking) ---
            if auto_detect and not use_static_mask:
                if first_frame_mask is None or frame_idx % re_detect_interval == 0:
                    detected = self._detect_watermark_auto(frame_rgb)
                    if detected and len(detected) > 0:
                        new_mask = self._create_mask_from_bboxes((height, width), detected)
                        if np.max(new_mask) > 10:
                            first_frame_mask = new_mask
                            # Initialize KCF tracker from first detection bbox
                            if tracker is None:
                                try:
                                    x1, y1, x2, y2 = detected[0]
                                    tracker_bbox = (x1, y1, x2 - x1, y2 - y1)
                                    tracker = _create_kcf_tracker()
                                    if tracker is not None:
                                        tracker.init(frame_rgb, tracker_bbox)
                                        logger.info(f"KCF tracker initialized: {tracker_bbox}")
                                    else:
                                        logger.warning("KCF tracker unavailable (OpenCV contrib/legacy not installed)")
                                except Exception as e:
                                    logger.warning(f"KCF tracker init failed: {e}")
                                    tracker = None
                            if frame_idx == start_frame:
                                logger.info(f"Initial watermark mask: {np.sum(new_mask > 30)} pixels")
                            else:
                                logger.info(f"Re-detected watermark at frame {frame_idx}")
                    elif tracker is not None:
                        # Detection failed but tracker may still work
                        logger.info(f"Re-detection failed at frame {frame_idx}, relying on tracker")
                elif tracker is not None:
                    # Use KCF tracker to update mask position between re-detections
                    try:
                        success, updated_bbox = tracker.update(frame_rgb)
                        if success:
                            x, y, w, h = [int(v) for v in updated_bbox]
                            # Expand bbox slightly to cover watermark edges
                            pad = 10
                            current_bbox = (
                                max(0, x - pad), max(0, y - pad),
                                min(width, x + w + pad), min(height, y + h + pad)
                            )
                            first_frame_mask = self._create_mask_from_bboxes(
                                (height, width), [current_bbox], feather=12
                            )
                        else:
                            logger.info(f"KCF tracker lost at frame {frame_idx}, re-detecting immediately")
                            tracker = None
                            # Immediate re-detection: don't wait for re_detect_interval
                            detected = self._detect_watermark_auto(frame_rgb)
                            if detected and len(detected) > 0:
                                best_bbox = max(detected, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                                first_frame_mask = self._create_mask_from_bboxes(
                                    (height, width), [best_bbox], feather=12
                                )
                                try:
                                    x1, y1, x2, y2 = best_bbox
                                    tracker = _create_kcf_tracker()
                                    if tracker is not None:
                                        tracker.init(frame_rgb, (x1, y1, x2 - x1, y2 - y1))
                                        logger.info(f"KCF re-initialized at frame {frame_idx}: "
                                                    f"({x1},{y1},{x2-x1},{y2-y1})")
                                    else:
                                        logger.warning("KCF tracker unavailable (OpenCV contrib/legacy not installed)")
                                except Exception as e2:
                                    logger.warning(f"KCF re-init failed: {e2}")
                    except Exception as e:
                        logger.warning(f"KCF tracker update failed: {e}")
                        tracker = None
                        # Immediate re-detection on error too
                        detected = self._detect_watermark_auto(frame_rgb)
                        if detected and len(detected) > 0:
                            best_bbox = max(detected, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                            first_frame_mask = self._create_mask_from_bboxes(
                                (height, width), [best_bbox], feather=12
                            )
                            try:
                                x1, y1, x2, y2 = best_bbox
                                tracker = _create_kcf_tracker()
                                if tracker is not None:
                                    tracker.init(frame_rgb, (x1, y1, x2 - x1, y2 - y1))
                                else:
                                    logger.warning("KCF tracker unavailable (OpenCV contrib/legacy not installed)")
                            except Exception as e2:
                                logger.warning(f"KCF re-init after error failed: {e2}")

            # --- Inpaint or pass-through ---
            if first_frame_mask is not None and np.max(first_frame_mask) > 10:
                try:
                    result_np = self._run_inpaint(frame_rgb, first_frame_mask)
                    final_np = self._blend_result(frame_rgb, result_np, first_frame_mask)

                    # Temporal EMA smoothing: reduce flicker across consecutive frames
                    if prev_frame_rgb is not None:
                        alpha = 0.3  # EMA weight for new frame
                        final_np = (final_np.astype(np.float32) * alpha +
                                    prev_frame_rgb.astype(np.float32) * (1 - alpha))
                        final_np = np.clip(final_np, 0, 255).astype(np.uint8)
                    prev_frame_rgb = final_np.copy()

                    final_bgr = cv2.cvtColor(final_np, cv2.COLOR_RGB2BGR)
                    writer.write(final_bgr)
                    if not writer.isOpened():
                        raise RuntimeError(f"Video writer closed unexpectedly at frame {frame_idx}")
                except Exception as e:
                    logger.error(f"Frame {frame_idx} inpainting failed: {e}")
                    writer.write(frame)  # Write original frame as fallback
                    failed_frames += 1
                    prev_frame_rgb = None  # Reset EMA on failure
            else:
                writer.write(frame)
                prev_frame_rgb = None  # Reset EMA on pass-through

            frame_idx += 1
            if progress_callback:
                progress_callback((frame_idx - start_frame) / (end_frame - start_frame) * 100)
            if frame_idx % 30 == 0:
                logger.info(f"Progress: {frame_idx}/{end_frame} frames")

        cap.release()
        writer.release()

        if os.path.exists(final_output):
            file_size = os.path.getsize(final_output)
            logger.info(f"Output file size: {file_size / 1024:.1f}KB")
            if file_size < 1024:
                logger.warning("Output file is too small, might be corrupted!")

        elapsed = time.time() - start_time
        logger.info(f"[DONE] Video saved: {final_output} ({elapsed:.1f}s, {failed_frames} failed frames)")
        return final_output


# ================================================================
# Singleton
# ================================================================

_engine_instance = None

def get_engine(device: str = 'auto') -> InpaintingEngine:
    global _engine_instance
    if _engine_instance is None:
        logger.info("Creating new engine instance...")
        _engine_instance = InpaintingEngine(device=device)
        success = _engine_instance.load_model()
        if not success:
            logger.error("Failed to load inpainting model!")
            raise RuntimeError(
                "Cannot load LaMa inpainting model. "
                "Please check network or run: pip install simple-lama-inpainting"
            )
        if _engine_instance.yolo_available and _engine_instance.u2net_available:
            logger.info("Engine ready: LaMa + YOLO + U2-Net + CV")
        elif _engine_instance.yolo_available:
            logger.info("Engine ready: LaMa + YOLO + CV")
        elif _engine_instance.u2net_available:
            logger.info("Engine ready: LaMa + U2-Net + CV")
        else:
            logger.info("Engine ready: LaMa + CV-only (install rembg/ultralytics for AI)")
    return _engine_instance