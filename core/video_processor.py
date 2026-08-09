"""
Video Processor - DIS optical flow guided temporal smoothing.
Version 12.0 - SSIM scene change, forced optical flow on detect fail, mask boundary smoothing.
"""
import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Callable
import cv2
import torch
import yaml
from skimage.metrics import structural_similarity as ssim

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
TEMPORAL_WEIGHT = VIDEO_CFG.get("temporal_weight", 0.45)
TEMPORAL_ALPHA = VIDEO_CFG.get("temporal_alpha", 0.2)
SCENE_CHANGE_THRESHOLD_SSIM = VIDEO_CFG.get("scene_change_threshold_ssim", 0.65)
USE_OPTICAL_FLOW = VIDEO_CFG.get("use_optical_flow", True)
FORCE_FLOW_ON_DETECT_FAIL = VIDEO_CFG.get("force_flow_on_detect_fail", True)
MASK_BOUNDARY_SMOOTH_KERNEL = VIDEO_CFG.get("mask_boundary_smooth_kernel", 5)


class VideoProcessor:
    """
    Video watermark removal with temporal consistency.
    - DIS optical flow for mask propagation between frames
    - SSIM-based scene change detection
    - Forced optical flow propagation when detection fails
    - Mask boundary smoothing across frames
    """

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cpu')
        self.optical_flow = None
        self.flow_initialized = False
        self.prev_frame_gray = None
        self.prev_result = None
        self.prev_mask = None
        self.prev_inpainted = None
        self.frame_count = 0
        logger.info(f"VideoProcessor on device: {self.device}")

    def _init_optical_flow(self):
        """Initialize DIS optical flow (Fast, good for real-time video)."""
        if not USE_OPTICAL_FLOW:
            logger.info("Optical flow disabled in config")
            return False

        try:
            self.optical_flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            self.flow_initialized = True
            logger.info("[OK] DIS optical flow initialized")
            return True
        except Exception as e:
            logger.warning(f"DIS optical flow init failed: {e}. Trying Farneback fallback...")
            try:
                self.optical_flow = None  # Will use calcOpticalFlowFarneback
                self.flow_initialized = True
                logger.info("[OK] Farneback optical flow (fallback) initialized")
                return True
            except Exception as e2:
                logger.warning(f"Optical flow unavailable: {e2}. Temporal smoothing only.")
                self.flow_initialized = False
                return False

    def _compute_flow(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> Optional[np.ndarray]:
        """Compute optical flow between two grayscale frames."""
        if not self.flow_initialized or prev_gray is None or curr_gray is None:
            return None
        try:
            if self.optical_flow is not None:
                flow = self.optical_flow.calc(prev_gray, curr_gray, None)
            else:
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
        """Warp a frame using optical flow."""
        h, w = flow.shape[:2]
        flow_map = np.zeros_like(flow)
        flow_map[:, :, 0] = flow[:, :, 0] + np.arange(w)
        flow_map[:, :, 1] = flow[:, :, 1] + np.arange(h)[:, np.newaxis]
        flow_map = flow_map.astype(np.float32)
        warped = cv2.remap(frame, flow_map, None, cv2.INTER_LINEAR)
        return warped

    def _warp_mask(self, mask: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """Warp a binary mask using optical flow (nearest-neighbor interpolation)."""
        h, w = flow.shape[:2]
        flow_map = np.zeros_like(flow)
        flow_map[:, :, 0] = flow[:, :, 0] + np.arange(w)
        flow_map[:, :, 1] = flow[:, :, 1] + np.arange(h)[:, np.newaxis]
        flow_map = flow_map.astype(np.float32)
        warped = cv2.remap(mask, flow_map, None, cv2.INTER_NEAREST)
        return warped

    def _scene_changed(self, prev_gray: Optional[np.ndarray],
                       curr_gray: np.ndarray) -> bool:
        """
        Detect scene change using SSIM (Structural Similarity).
        SSIM is more robust than mean pixel difference for detecting
        actual scene cuts vs. lighting changes.
        """
        if prev_gray is None:
            return True

        try:
            # Ensure same size
            if prev_gray.shape != curr_gray.shape:
                curr_gray = cv2.resize(curr_gray, (prev_gray.shape[1], prev_gray.shape[0]))

            score = ssim(prev_gray, curr_gray, data_range=255)
            is_changed = score < SCENE_CHANGE_THRESHOLD_SSIM

            if is_changed:
                logger.info(f"Scene change detected: SSIM={score:.3f} < {SCENE_CHANGE_THRESHOLD_SSIM}")
            return is_changed
        except Exception as e:
            logger.warning(f"SSIM scene change detection failed: {e}, falling back to mean diff")
            # Fallback to mean difference
            diff = cv2.absdiff(prev_gray, curr_gray)
            mean_diff = np.mean(diff)
            return mean_diff > 30.0

    def _smooth_mask_boundary(self, mask: np.ndarray,
                               prev_mask: Optional[np.ndarray],
                               kernel_size: int = MASK_BOUNDARY_SMOOTH_KERNEL) -> np.ndarray:
        """
        Apply temporal Gaussian smoothing to mask boundary pixels.
        This reduces flickering at the edges of the inpainted region.

        Args:
            mask: Current frame mask (0-255)
            prev_mask: Previous frame mask, or None
            kernel_size: Gaussian kernel size for temporal smoothing

        Returns:
            Smoothed mask
        """
        if prev_mask is None or kernel_size <= 0:
            return mask

        if mask.shape != prev_mask.shape:
            prev_mask = cv2.resize(prev_mask, (mask.shape[1], mask.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

        # Find boundary pixels: where mask differs from prev_mask
        mask_bin = (mask > 30).astype(np.float32) / 255.0
        prev_bin = (prev_mask > 30).astype(np.float32) / 255.0
        boundary = np.abs(mask_bin - prev_bin) > 0.1  # Pixels that changed

        if boundary.sum() < 10:
            return mask

        # Apply Gaussian blur only to boundary pixels
        ksize = kernel_size * 2 + 1
        if ksize % 2 == 0:
            ksize += 1
        mask_smooth = cv2.GaussianBlur(mask.astype(np.float32), (ksize, ksize), kernel_size)

        # Blend: original mask everywhere except boundary where we use smoothed
        boundary_float = boundary.astype(np.float32)
        result = mask.astype(np.float32) * (1.0 - boundary_float) + mask_smooth * boundary_float

        return np.clip(result, 0, 255).astype(np.uint8)

    def reset(self):
        """Reset state for new video processing."""
        self.prev_frame_gray = None
        self.prev_result = None
        self.prev_mask = None
        self.prev_inpainted = None
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

        # If no mask at all, return original frame
        if mask_to_use is None or np.max(mask_to_use) <= 10:
            self.prev_frame_gray = frame_gray
            self.prev_result = None
            self.prev_mask = None
            self.prev_inpainted = None
            return frame_rgb

        # Apply temporal mask boundary smoothing
        mask_to_use = self._smooth_mask_boundary(mask_to_use, self.prev_mask)

        h, w = frame_rgb.shape[:2]

        # --- Optical flow propagation of previous result ---
        warped_prev_result = None
        warped_prev_mask = None
        if self.flow_initialized and self.prev_frame_gray is not None and self.prev_result is not None:
            flow = self._compute_flow(self.prev_frame_gray, frame_gray)
            if flow is not None:
                warped_prev_result = self._warp_frame(self.prev_result, flow)
                if self.prev_mask is not None:
                    warped_prev_mask = self._warp_mask(self.prev_mask, flow)

        # --- Inpaint current frame ---
        try:
            inpainted = inpainter.inpaint(frame_rgb, mask_to_use)
            from core.inpainter import Inpainter
            current_result = Inpainter.blend(frame_rgb, inpainted, mask_to_use)
        except Exception as e:
            logger.error(f"Frame {frame_idx} inpainting failed: {e}")
            current_result = frame_rgb

        # --- Temporal fusion: blend current result with warped previous result ---
        if warped_prev_result is not None and warped_prev_mask is not None:
            # Lower weight on current frame (0.45) to reduce flickering
            w_curr = TEMPORAL_WEIGHT
            w_prev = 1.0 - TEMPORAL_WEIGHT  # 0.55 - more weight on stable previous

            mask_bin = (mask_to_use > 10).astype(np.float32) / 255.0
            mask_3ch = np.stack([mask_bin] * 3, axis=2)

            prev_mask_valid = (warped_prev_mask > 10).astype(np.float32) / 255.0
            prev_mask_3ch = np.stack([prev_mask_valid] * 3, axis=2)

            # Fuse only where both masks are valid
            fusion_mask = mask_3ch * prev_mask_3ch
            anti_fusion = 1.0 - fusion_mask

            fused = (current_result.astype(np.float32) * anti_fusion +
                     (warped_prev_result.astype(np.float32) * w_prev +
                      current_result.astype(np.float32) * w_curr) * fusion_mask)

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
        self.prev_inpainted = inpainted.copy()

        return final_result

    def propagate_mask_with_flow(self, prev_mask: np.ndarray,
                                  prev_gray: np.ndarray,
                                  curr_gray: np.ndarray) -> Optional[np.ndarray]:
        """
        Force optical flow propagation of mask when detection fails.
        This ensures the mask continues to track the watermark even when
        the detector temporarily loses it.

        Args:
            prev_mask: Previous frame's mask (0-255)
            prev_gray: Previous frame grayscale
            curr_gray: Current frame grayscale

        Returns:
            Propagated mask, or None if flow is unavailable
        """
        if not self.flow_initialized:
            return None

        flow = self._compute_flow(prev_gray, curr_gray)
        if flow is None:
            return None

        warped = self._warp_mask(prev_mask, flow)
        logger.debug(f"Mask propagated via optical flow (detection failed)")
        return warped