"""
Core Inpainting Engine - Three-stage pipeline orchestration.
Version 12.0 - Detection -> Mask -> Inpaint -> Blend. Strict error handling.
SAM segmentation, optical flow video processing, Poisson edge fusion.
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
import yaml

# Add SAM 2.1 to path
SAM2_DIR = Path(r"D:\AI\watermark_remover\models\sam")
if str(SAM2_DIR) not in sys.path:
    sys.path.insert(0, str(SAM2_DIR))

logger = logging.getLogger('engine')

# Load config
def _load_config():
    cfg_path = Path(r"D:\AI\watermark_remover\config.yaml")
    if cfg_path.exists():
        with open(cfg_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}

CONFIG = _load_config()
PATH_CFG = CONFIG.get("paths", {})
VIDEO_CFG = CONFIG.get("video", {})

MODEL_DIR = Path(PATH_CFG.get("models_dir", r"D:\AI\watermark_remover\models"))
CACHE_DIR = Path(PATH_CFG.get("cache_dir", r"D:\AI\watermark_remover\cache"))
OUTPUT_DIR = Path(PATH_CFG.get("output_dir", r"D:\AI\watermark_remover\output"))
for d in [MODEL_DIR, CACHE_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

REDETECT_INTERVAL = VIDEO_CFG.get("redetect_interval", 10)
TEMPORAL_ALPHA = VIDEO_CFG.get("temporal_alpha", 0.2)
SCENE_CHANGE_THRESHOLD = VIDEO_CFG.get("scene_change_threshold", 30.0)

# Import custom exception
from core.detector import WatermarkNotFoundError


class InpaintingEngine:
    """
    Image/video watermark removal engine.
    v12.0: Three-stage pipeline - detect -> mask -> inpaint -> blend.
    SAM + GrabCut segmentation, optical flow video, Poisson edge blending.
    """

    def __init__(self, device: str = 'auto'):
        self.device = self._get_device(device)
        self.inpainter = None
        self.segmenter = None
        self.detector = None
        self.video_processor = None
        self.inpainter_loaded = False
        self.segmenter_loaded = False
        self.detector_loaded = False
        self.video_processor_loaded = False
        self.yolo_available = False
        self.u2net_available = False
        self.sam_available = False
        logger.info(f"Engine v12.0 initialized on device: {self.device}")

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
    # Module Loading
    # ================================================================

    def load_inpainter(self) -> bool:
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
        try:
            from core.segmenter import Segmenter
            self.segmenter = Segmenter(device=self.device)
            self.segmenter_loaded = True

            # Try loading SAM
            sam_ok = self.segmenter.load_sam()
            self.sam_available = sam_ok
            if sam_ok:
                logger.info("[OK] Segmenter: SAM + GrabCut")
            else:
                logger.info("[OK] Segmenter: GrabCut only")
            return True
        except Exception as e:
            logger.warning(f"[WARN] Segmenter load failed: {e}")
            return False

    def load_detector(self) -> bool:
        try:
            from core.detector import WatermarkDetector
            self.detector = WatermarkDetector(device=str(self.device))
            self.detector_loaded = True

            yolo_ok = self.detector.load_yolo()
            self.yolo_available = yolo_ok

            u2_ok = self.detector.load_u2net()
            self.u2net_available = u2_ok

            if yolo_ok and u2_ok:
                logger.info("[OK] Detector: YOLO + U2-Net")
            elif yolo_ok:
                logger.info("[OK] Detector: YOLO only")
            elif u2_ok:
                logger.info("[OK] Detector: U2-Net only")
            else:
                logger.warning("[WARN] Detector: No AI models available. Manual mode only.")
            return True
        except Exception as e:
            logger.warning(f"[WARN] Detector import failed: {e}")
            self.detector = None
            return False

    def load_video_processor(self) -> bool:
        try:
            from core.video_processor import VideoProcessor
            self.video_processor = VideoProcessor(device=self.device)
            self.video_processor._init_optical_flow()
            self.video_processor_loaded = True
            logger.info("[OK] VideoProcessor loaded (DIS optical flow)")
            return True
        except Exception as e:
            logger.warning(f"[WARN] VideoProcessor load failed: {e}")
            return False

    def load_model(self) -> bool:
        ok = self.load_inpainter()
        if not ok:
            return False
        self.load_segmenter()
        self.load_detector()
        self.load_video_processor()
        return True

    def _ensure_modules(self):
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
    # Three-Stage Pipeline: Detect -> Mask -> Inpaint -> Blend
    # ================================================================

    def _detect_watermark_auto(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Auto-detect watermarks using YOLO + saliency fallback.
        Raises WatermarkNotFoundError if no watermark is found.
        """
        if self.detector is None or not self.detector_loaded:
            raise RuntimeError("Detector not loaded. Cannot auto-detect watermarks.")

        regions = self.detector.detect_watermarks(image, use_yolo=True, use_saliency=True)
        # detect_watermarks now raises WatermarkNotFoundError if nothing found
        logger.info(f"Auto-detected {len(regions)} watermark regions")
        return regions

    def _process_image_pipeline(self, image: np.ndarray,
                                 bboxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Execute the three-stage pipeline on a single image.
        Returns the final blended result.
        """
        # Stage 1: Generate mask
        if not self.segmenter or not self.segmenter_loaded:
            raise RuntimeError("Segmenter not loaded")
        mask = self.segmenter.generate_mask(image, bboxes)

        # Stage 2: Inpaint
        inpainted = self.inpainter.inpaint(image, mask)

        # Stage 3: Blend
        from core.inpainter import Inpainter
        final = Inpainter.blend(image, inpainted, mask)
        return final

    # ================================================================
    # Image Inpainting
    # ================================================================

    def inpaint_image(self,
                      image_path: str,
                      output_path: Optional[str] = None,
                      bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
                      auto_detect: bool = False) -> str:
        """
        Inpaint a single image.

        Args:
            image_path: Path to input image
            output_path: Output path (auto-generated if None)
            bboxes: List of (x1, y1, x2, y2) bounding boxes for manual mode
            auto_detect: Whether to auto-detect watermarks

        Returns:
            Path to output image

        Raises:
            WatermarkNotFoundError: If no watermark is detected in auto mode
            RuntimeError: If required modules are not loaded
        """
        self._ensure_modules()
        logger.info(f"Processing image: {image_path}")
        start_time = time.time()

        image = self._load_image(image_path)
        h, w = image.shape[:2]

        # Determine bboxes
        if auto_detect:
            detected = self._detect_watermark_auto(image)
            if bboxes:
                bboxes = bboxes + detected
            else:
                bboxes = detected

        if not bboxes:
            raise ValueError(
                "No watermark regions specified. "
                "Provide bboxes or enable auto_detect."
            )

        # Run pipeline
        final_np = self._process_image_pipeline(image, bboxes)

        if output_path is None:
            output_path = str(OUTPUT_DIR / f"result_{Path(image_path).stem}.png")
        self._save_image(final_np, output_path)

        elapsed = time.time() - start_time
        logger.info(f"[DONE] Image saved: {output_path} ({elapsed:.1f}s)")
        return output_path

    # ================================================================
    # Video Inpainting (with optical flow temporal smoothing)
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

    def _scene_changed(self, prev_frame: Optional[np.ndarray],
                       curr_frame: np.ndarray) -> bool:
        """Detect scene change via frame difference."""
        if prev_frame is None:
            return True
        diff = cv2.absdiff(prev_frame, curr_frame)
        mean_diff = np.mean(diff)
        return mean_diff > SCENE_CHANGE_THRESHOLD

    def inpaint_video(self,
                      video_path: str,
                      output_path: Optional[str] = None,
                      bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
                      auto_detect: bool = False,
                      start_frame: int = 0,
                      end_frame: Optional[int] = None,
                      progress_callback=None) -> str:
        """
        Inpaint video watermarks with optical flow temporal smoothing.
        Auto mode: re-detects every REDETECT_INTERVAL frames.
        Manual mode: uses static mask from bboxes.

        Raises WatermarkNotFoundError if auto-detect fails on first frame.
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

        # Determine mask strategy
        use_static_mask = (bboxes is not None and not auto_detect)
        static_mask = None
        current_mask = None
        prev_frame_gray = None
        failed_frames = 0

        if use_static_mask:
            if not self.segmenter or not self.segmenter_loaded:
                raise RuntimeError("Segmenter not loaded")
            # Pre-generate static mask from bboxes
            dummy_image = np.zeros((height, width, 3), dtype=np.uint8)
            static_mask = self.segmenter.generate_mask(dummy_image, bboxes)
            if np.max(static_mask) <= 10:
                raise ValueError("Generated mask is empty. Check bbox coordinates.")
            current_mask = static_mask
            logger.info(f"Static mask: {np.sum(current_mask > 30)} pixels")

        # Initialize video processor for optical flow
        if self.video_processor is not None and self.video_processor_loaded:
            self.video_processor.reset()
            logger.info("VideoProcessor: optical flow temporal smoothing enabled")
        else:
            logger.info("VideoProcessor: not available, using basic frame-by-frame processing")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame

        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Auto-detect: re-detect periodically or on scene change
            if auto_detect and not use_static_mask:
                frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
                need_detect = (
                    frame_idx == start_frame or
                    frame_idx % REDETECT_INTERVAL == 0 or
                    self._scene_changed(prev_frame_gray, frame_gray)
                )
                if need_detect:
                    try:
                        detected = self._detect_watermark_auto(frame_rgb)
                        if not self.segmenter or not self.segmenter_loaded:
                            raise RuntimeError("Segmenter not loaded")
                        current_mask = self.segmenter.generate_mask(frame_rgb, detected)
                        if frame_idx == start_frame:
                            logger.info(f"Initial mask: {np.sum(current_mask > 30)} pixels")
                        else:
                            logger.info(f"Re-detected at frame {frame_idx}")
                    except WatermarkNotFoundError:
                        # Detection failed, keep previous mask if any
                        if current_mask is None:
                            raise WatermarkNotFoundError(
                                "No watermark detected in the first frames of the video. "
                                "Try manual selection mode."
                            )
                        logger.info(f"Re-detection failed at frame {frame_idx}, using previous mask")
                prev_frame_gray = frame_gray

            # Process frame
            if current_mask is not None and np.max(current_mask) > 10:
                try:
                    if self.video_processor is not None and self.video_processor_loaded:
                        # Use video processor with optical flow temporal smoothing
                        result_np = self.video_processor.process_frame(
                            frame_rgb, frame_idx, end_frame,
                            current_mask if not use_static_mask else None,
                            self.inpainter,
                            static_mask=static_mask if use_static_mask else None
                        )
                    else:
                        # Basic frame-by-frame with EMA temporal smoothing
                        inpainted = self.inpainter.inpaint(frame_rgb, current_mask)
                        from core.inpainter import Inpainter
                        result_np = Inpainter.blend(frame_rgb, inpainted, current_mask)

                    final_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
                    writer.write(final_bgr)
                    if not writer.isOpened():
                        raise RuntimeError(f"Video writer closed unexpectedly at frame {frame_idx}")
                except Exception as e:
                    logger.error(f"Frame {frame_idx} inpainting failed: {e}")
                    writer.write(frame)
                    failed_frames += 1
            else:
                writer.write(frame)

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
    # Preview Mask (for manual mode validation)
    # ================================================================

    def preview_mask(self, image: np.ndarray,
                     bboxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Generate a mask preview image for user validation.
        Returns an overlay image showing the mask regions.
        """
        if not self.segmenter or not self.segmenter_loaded:
            self.load_segmenter()
        if not self.segmenter:
            raise RuntimeError("Segmenter not loaded")

        mask = self.segmenter.generate_mask(image, bboxes)

        # Create overlay: red tint on mask regions
        overlay = image.copy()
        mask_bin = (mask > 30).astype(np.uint8)
        overlay[mask_bin > 0] = overlay[mask_bin > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
        return overlay.astype(np.uint8)


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
        parts = ["LaMa"]
        if _engine_instance.yolo_available:
            parts.append("YOLO")
        if _engine_instance.u2net_available:
            parts.append("U2-Net")
        if _engine_instance.sam_available:
            parts.append("SAM")
        if _engine_instance.video_processor_loaded:
            parts.append("OpticalFlow")
        logger.info(f"Engine ready: {' + '.join(parts)}")
    return _engine_instance