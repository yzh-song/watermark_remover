"""
Segmenter Module - Precise mask generation with GrabCut refinement.
Version 12.0 - GrabCut for all masks, standardized dilation/feathering, mask validation.
"""
import logging
import numpy as np
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
MASK_CFG = CONFIG.get("mask", {})
DEFAULT_FEATHER = MASK_CFG.get("feather", 6)
DILATE_KERNEL = MASK_CFG.get("dilate_kernel", 5)
DILATE_ITER = MASK_CFG.get("dilate_iter", 2)
GRABCUT_ITER = MASK_CFG.get("grabcut_iter", 2)
MIN_MASK_PIXELS = MASK_CFG.get("min_mask_pixels", 100)


class Segmenter:
    """
    Mask generator for watermark regions.
    Uses GrabCut refinement for precise pixel-level masks.
    Standardized dilation and feathering for consistent results.
    """

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cpu')
        logger.info(f"Segmenter on device: {self.device}")

    @staticmethod
    def mask_from_bbox(image_shape: Tuple[int, int],
                       bbox: Tuple[int, int, int, int],
                       feather: int = DEFAULT_FEATHER) -> np.ndarray:
        """
        Create soft-edged mask from bounding box.
        Uses dilation + Gaussian feather for smooth transitions.
        """
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(x1 + 1, min(int(x2), w))
        y2 = max(y1 + 1, min(int(y2), h))
        mask[y1:y2, x1:x2] = 255

        # Dilate to cover watermark edges
        ksize = DILATE_KERNEL
        if ksize % 2 == 0:
            ksize += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        mask = cv2.dilate(mask, kernel, iterations=DILATE_ITER)

        # Gaussian feather for soft edges
        gauss_ksize = feather * 2 + 1
        if gauss_ksize % 2 == 0:
            gauss_ksize += 1
        mask_float = mask.astype(np.float32) / 255.0
        mask_feathered = cv2.GaussianBlur(mask_float, (gauss_ksize, gauss_ksize), feather // 2)
        return (mask_feathered * 255).astype(np.uint8)

    @staticmethod
    def refine_with_grabcut(image: np.ndarray,
                            mask: np.ndarray,
                            bbox: Tuple[int, int, int, int],
                            iterations: int = GRABCUT_ITER) -> np.ndarray:
        """
        Refine a coarse mask using OpenCV GrabCut.
        Produces pixel-accurate boundaries.
        """
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        x1 = max(0, x1 - 10); y1 = max(0, y1 - 10)
        x2 = min(w, x2 + 10); y2 = min(h, y2 + 10)

        if x2 <= x1 or y2 <= y1:
            logger.warning(f"GrabCut: invalid bbox after expansion: ({x1},{y1})-({x2},{y2})")
            return (mask > 128).astype(np.uint8) * 255

        mask_bin = (mask > 128).astype(np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        # Initialize GrabCut mask: 0=bg, 1=fg, 2=prob_bg, 3=prob_fg
        gc_mask = np.where(mask_bin > 0, 1, 0).astype(np.uint8)

        try:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            gc_mask, _, _ = cv2.grabCut(
                image_bgr, gc_mask, (x1, y1, x2 - x1, y2 - y1),
                bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_MASK
            )
            refined = np.where((gc_mask == 1) | (gc_mask == 3), 255, 0).astype(np.uint8)
            return refined
        except Exception as e:
            logger.warning(f"GrabCut failed: {e}, using binarized mask")
            return mask_bin * 255

    def generate_mask(self,
                      image: np.ndarray,
                      bboxes: List[Tuple[int, int, int, int]],
                      feather: int = DEFAULT_FEATHER) -> np.ndarray:
        """
        Generate combined mask for all watermark regions.
        Uses GrabCut refinement for precise boundaries.

        Args:
            image: RGB uint8 numpy array (H, W, 3)
            bboxes: List of (x1, y1, x2, y2) bounding boxes
            feather: Feather radius for mask edges

        Returns:
            Grayscale mask (H, W) uint8, 0=keep, 255=inpaint
            Raises ValueError if mask is empty.
        """
        h, w = image.shape[:2]
        masks = []

        for bbox in bboxes:
            # Step 1: Create coarse bbox mask
            coarse = self.mask_from_bbox((h, w), bbox, feather)

            # Step 2: Refine with GrabCut
            refined = self.refine_with_grabcut(image, coarse, bbox)

            # Step 3: Dilate and feather the refined mask
            ksize = DILATE_KERNEL
            if ksize % 2 == 0:
                ksize += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            refined = cv2.dilate(refined, kernel, iterations=DILATE_ITER)

            gauss_ksize = feather * 2 + 1
            if gauss_ksize % 2 == 0:
                gauss_ksize += 1
            refined_float = refined.astype(np.float32) / 255.0
            refined_soft = cv2.GaussianBlur(refined_float, (gauss_ksize, gauss_ksize), feather // 2)
            masks.append((refined_soft * 255).astype(np.uint8))

        if not masks:
            raise ValueError("No valid bboxes provided for mask generation")

        combined = np.maximum.reduce(masks)
        combined = np.clip(combined, 0, 255).astype(np.uint8)

        # Validate mask
        mask_pixels = np.sum(combined > 10)
        if mask_pixels < MIN_MASK_PIXELS:
            raise ValueError(
                f"Generated mask is too small ({mask_pixels} pixels < {MIN_MASK_PIXELS} minimum). "
                "Watermark region may be too faint or bbox is incorrect."
            )

        logger.info(f"Mask generated: {mask_pixels} pixels ({mask_pixels / (w * h) * 100:.1f}% of image)")
        return combined