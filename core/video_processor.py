"""
Video Processor - DIS optical flow guided temporal smoothing.
Version 12.0 - Optical flow mask propagation, frame fusion, scene change detection.
"""
import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Callable
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
VIDEO_CFG = CONFIG.get("video", {})
REDETECT_INTERVAL = VIDEO_CFG.get("redetect_interval", 10)
TEMPORAL_WEIGHT = VIDEO_CFG.get("temporal_weight", 0.7)
TEMPORAL_ALPHA = VIDEO_CFG.get("temporal_alpha", 0.2)
SCENE_CHANGE_THRESHOLD = VIDEO_CFG.get("scene_change_threshold", 30.0)
USE_OPTICAL_FLOW = VIDEO_CFG.get("use_optical_flow", True)


class VideoProcessor:
    """
    Video watermark removal with temporal consistency.
    - DIS optical flow for mask propagation between frames
    - Frame fusion to reduce flickering
    - Scene change detection for re-detection
    """

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cpu')
        self.optical_flow = None
        self.flow_initialized = False
        self.prev_frame_gray = None
        self.prev_result = None
        self.prev_mask = None
        self.frame_count = 0
        logger.info(f"VideoProcessor on device: {self.device}")

    def _init_optical_flow(self):
        """Initialize DIS optical flow (Fast, good for real-time video)."""
        if not USE_OPTICAL_FLOW:
            logger.info("Optical flow disabled in config")
            return False

        try:
            # DIS optical flow: fast and good enough for smooth motion
            self.optical_flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            self.flow_initialized = True
            logger.info("[OK] DIS optical flow initialized")
            return True
        except Exception as e:
            logger.warning(f"DIS optical flow init failed: {e}. Trying Farneback fallback...")
            try:
                # Farneback fallback: always available
                self.optical_flow = None  # Will use calcOpticalFlowFarneback
                self.flow_initialized = True
                logger.info("[OK] Farneback optical flow (fallback) initialized")
                return True
            except Exception as e2:
                logger.warning(f"Optical flow unavailable: {e2}. Temporal smoothing only.")
                self.flow_initialized = False
                return False

    def _compute_flow(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute optical flow between two grayscale frames.
        Returns flow array (H, W, 2) or None.
        """
        if not self.flow_initialized or prev_gray is None or curr_gray is None:
            return None

        try:
            if self.optical_flow is not None:
                # DIS optical flow
                flow = self.optical_flow.calc(prev_gray, curr_gray, None)
            else:
                # Farneback fallback
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, curr_gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
            return flow
        except Exception as e:
            logger.debug(f"Flow computation failed: {e}")
            return None

    def _warp_frame(self, frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """
        Warp a frame (or mask) using optical flow.
        Maps pixels from frame to current frame coordinates.
        """
        h, w = flow.shape[:2]
        flow_map = np.zeros_like(flow)
        flow_map[:, :, 0] = flow[:, :, 0] + np.arange(w)
        flow_map[:, :, 1] = flow[:, :, 1] + np.arange(h)[:, np.newaxis]
        flow_map = flow_map.astype(np.float32)

        warped = cv2.remap(frame, flow_map, None, cv2.INTER_LINEAR)
        return warped

    def _warp_mask(self, mask: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """
        Warp a binary mask using optical flow.
        Uses nearest-neighbor interpolation to preserve binary values.
        """
        h, w = flow.shape[:2]
        flow_map = np.zeros_like(flow)
        flow_map[:, :, 0] = flow[:, :, 0] + np.arange(w)
        flow_map[:, :, 1] = flow[:, :, 1] + np.arange(h)[:, np.newaxis]
        flow_map = flow_map.astype(np.float32)

        warped = cv2.remap(mask, flow_map, None, cv2.INTER_NEAREST)
        return warped

    def _scene_changed(self, prev_gray: Optional[np.ndarray],
                       curr_gray: np.ndarray) -> bool:
        """Detect scene change via frame difference."""
        if prev_gray is None:
            return True
        diff = cv2.absdiff(prev_gray, curr_gray)
        mean_diff = np.mean(diff)
        return mean_diff > SCENE_CHANGE_THRESHOLD

    def reset(self):
        """Reset state for new video processing."""
        self.prev_frame_gray = None
        self.prev_result = None
        self.prev_mask = None
        self.frame_count = 0
        if not self.flow_initialized and USE_OPTICAL_FLOW:
            self._init_optical_flow()

    def process_frame(self,
                      frame_rgb: np.ndarray,
                      frame_idx: int,
                      total_frames: int,
                      current_mask: Optional[np.ndarray],
                      inpainter,
                      static_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Process a single video frame with temporal consistency.

        Args:
            frame_rgb: RGB uint8 (H, W, 3)
            frame_idx: Current frame index
            total_frames: Total frames in video
            current_mask: Current frame's mask (from detection), or None
            inpainter: Inpainter instance with inpaint() and blend()
            static_mask: Pre-computed static mask for manual mode

        Returns:
            Processed RGB uint8 (H, W, 3) frame
        """
        self.frame_count += 1
        frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        # Determine the mask to use
        mask_to_use = current_mask if current_mask is not None else static_mask

        # If no mask, return original frame
        if mask_to_use is None or np.max(mask_to_use) <= 10:
            self.prev_frame_gray = frame_gray
            self.prev_result = None
            self.prev_mask = None
            return frame_rgb

        h, w = frame_rgb.shape[:2]

        # --- Optical flow propagation of previous result ---
        warped_prev_result = None
        warped_prev_mask = None
        if self.flow_initialized and self.prev_frame_gray is not None and self.prev_result is not None:
            flow = self._compute_flow(self.prev_frame_gray, frame_gray)
            if flow is not None:
                # Warp previous result to current frame coordinates
                warped_prev_result = self._warp_frame(self.prev_result, flow)
                if self.prev_mask is not None:
                    warped_prev_mask = self._warp_mask(self.prev_mask, flow)

        # --- Inpaint current frame ---
        try:
            inpainted = inpainter.inpaint(frame_rgb, mask_to_use)
            # Use static blend method from Inpainter
            from core.inpainter import Inpainter
            current_result = Inpainter.blend(frame_rgb, inpainted, mask_to_use)
        except Exception as e:
            logger.error(f"Frame {frame_idx} inpainting failed: {e}")
            current_result = frame_rgb

        # --- Temporal fusion: blend current result with warped previous result ---
        if warped_prev_result is not None and warped_prev_mask is not None:
            # Weighted fusion: more weight on current frame, less on warped previous
            w_curr = TEMPORAL_WEIGHT
            w_prev = 1.0 - TEMPORAL_WEIGHT

            # Only fuse in the mask region
            mask_bin = (mask_to_use > 10).astype(np.float32) / 255.0
            mask_3ch = np.stack([mask_bin] * 3, axis=2)

            # Warped previous may have artifacts at borders, so only use it where mask is valid
            if warped_prev_mask is not None:
                prev_mask_valid = (warped_prev_mask > 10).astype(np.float32) / 255.0
                prev_mask_3ch = np.stack([prev_mask_valid] * 3, axis=2)
                # Fuse only where both masks are valid
                fusion_mask = mask_3ch * prev_mask_3ch
                anti_fusion = 1.0 - fusion_mask

                fused = (current_result.astype(np.float32) * anti_fusion +
                         (warped_prev_result.astype(np.float32) * w_prev +
                          current_result.astype(np.float32) * w_curr) * fusion_mask)
            else:
                fused = (current_result.astype(np.float32) * (1.0 - mask_3ch) +
                         (warped_prev_result.astype(np.float32) * w_prev +
                          current_result.astype(np.float32) * w_curr) * mask_3ch)

            final_result = np.clip(fused, 0, 255).astype(np.uint8)
        else:
            # No previous frame to fuse with, use EMA temporal smoothing
            if self.prev_result is not None:
                final_result = (current_result.astype(np.float32) * TEMPORAL_ALPHA +
                                self.prev_result.astype(np.float32) * (1.0 - TEMPORAL_ALPHA))
                final_result = np.clip(final_result, 0, 255).astype(np.uint8)
            else:
                final_result = current_result

        # Update state
        self.prev_frame_gray = frame_gray
        self.prev_result = final_result.copy()
        self.prev_mask = mask_to_use.copy()

        return final_result