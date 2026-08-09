"""
Segmenter Module - Precise mask generation with SAM + GrabCut refinement.
Version 12.0 - SAM for precise segmentation, GrabCut fallback, standardized post-processing.
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
DEFAULT_FEATHER = MASK_CFG.get("feather", 22)
DILATE_KERNEL = MASK_CFG.get("dilate_kernel", 5)
DILATE_ITER = MASK_CFG.get("dilate_iter", 2)
GRABCUT_ITER = MASK_CFG.get("grabcut_iter", 5)
MIN_MASK_PIXELS = MASK_CFG.get("min_mask_pixels", 100)
ERODE_BOUNDARY = MASK_CFG.get("erode_boundary", 3)
USE_SAM = MASK_CFG.get("use_sam", True)
SAM_MODEL_PATH = MASK_CFG.get("sam_model", "models/sam_vit_h.pth")
SAM_MODEL_TYPE = MASK_CFG.get("sam_model_type", "vit_h")

MODEL_DIR = Path(CONFIG.get("paths", {}).get("models_dir", r"D:\AI\watermark_remover\models"))


class Segmenter:
    """
    Mask generator for watermark regions.
    Priority: SAM > GrabCut > bbox dilation.
    Standardized post-processing: close + feather for all masks.
    """

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cpu')
        self.sam_model = None
        self.sam_loaded = False
        self.sam_predictor = None
        logger.info(f"Segmenter on device: {self.device}")

    # ================================================================
    # SAM Integration
    # ================================================================

    def load_sam(self) -> bool:
        """Load SAM 2.1 (Segment Anything Model 2) for precise mask generation."""
        if not USE_SAM:
            logger.info("SAM disabled in config. Using GrabCut fallback.")
            return False

        sam_checkpoint = Path(r"D:\AI\watermark_remover\models\sam\checkpoints\sam2.1_hiera_base_plus.pt")
        # Also try the path from config
        if not sam_checkpoint.exists():
            sam_checkpoint = MODEL_DIR / SAM_MODEL_PATH.split("/")[-1]
        if not sam_checkpoint.exists():
            sam_checkpoint = MODEL_DIR / "sam" / "checkpoints" / "sam2.1_hiera_base_plus.pt"

        if not sam_checkpoint.exists():
            logger.warning(
                f"SAM 2.1 checkpoint not found. Searched:\n"
                f"  - {MODEL_DIR / 'sam' / 'checkpoints' / 'sam2.1_hiera_base_plus.pt'}\n"
                f"Download from: https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt\n"
                f"Falling back to GrabCut."
            )
            return False

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            config_file = "configs/sam2.1/" + SAM_MODEL_TYPE  # e.g. "configs/sam2.1/sam2.1_hiera_b+"
            logger.info(f"Loading SAM 2.1 checkpoint: {sam_checkpoint}")
            logger.info(f"SAM 2.1 config: {config_file}")

            sam = build_sam2(
                config_file=config_file,
                ckpt_path=str(sam_checkpoint),
                device=str(self.device),
                mode="eval",
                apply_postprocessing=True,
            )
            self.sam_predictor = SAM2ImagePredictor(sam)
            self.sam_loaded = True
            logger.info(f"[OK] SAM 2.1 ({SAM_MODEL_TYPE}) loaded for precise segmentation")
            return True

        except ImportError as e:
            logger.warning(
                f"SAM 2.1 import failed: {e}. "
                f"Ensure hydra-core and iopath are installed: pip install hydra-core iopath. "
                f"Falling back to GrabCut."
            )
            return False
        except Exception as e:
            logger.error(f"SAM 2.1 load failed: {e}. Falling back to GrabCut.")
            import traceback
            logger.debug(traceback.format_exc())
            return False

    def _sam_generate_mask(self, image: np.ndarray,
                           bboxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Generate precise mask using SAM 2.1 for each bbox.
        Returns combined binary mask (0-255).
        """
        if not self.sam_loaded:
            raise RuntimeError("SAM 2.1 not loaded")

        self.sam_predictor.set_image(image)
        h, w = image.shape[:2]
        combined = np.zeros((h, w), dtype=np.uint8)

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            x1 = max(0, int(x1)); y1 = max(0, int(y1))
            x2 = min(w, int(x2)); y2 = min(h, int(y2))
            if x2 <= x1 or y2 <= y1:
                continue

            input_box = np.array([[x1, y1, x2, y2]])
            try:
                masks, scores, _ = self.sam_predictor.predict(
                    box=input_box,
                    multimask_output=True
                )
                # Pick the mask with highest score
                best_idx = np.argmax(scores)
                best_mask = (masks[best_idx] * 255).astype(np.uint8)
                combined = cv2.bitwise_or(combined, best_mask)
                logger.debug(f"SAM 2.1 mask score: {scores[best_idx]:.3f} for bbox ({x1},{y1})-({x2},{y2})")
            except Exception as e:
                logger.warning(f"SAM 2.1 predict failed for bbox ({x1},{y1})-({x2},{y2}): {e}")

        return combined

    # ================================================================
    # GrabCut Fallback
    # ================================================================

    @staticmethod
    def mask_from_bbox(image_shape: Tuple[int, int],
                       bbox: Tuple[int, int, int, int],
                       feather: int = DEFAULT_FEATHER) -> np.ndarray:
        """
        Create soft-edged mask from bounding box.
        Uses dilation + Gaussian feather for smooth transitions.
        bbox coordinates are clamped to image boundaries to prevent out-of-bounds errors.
        """
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        x1, y1, x2, y2 = bbox
        # Boundary clamp: ensure all coordinates are strictly within image bounds
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(x1 + 1, min(int(x2), w))
        y2 = max(y1 + 1, min(int(y2), h))
        if x2 <= x1 or y2 <= y1:
            logger.warning(f"mask_from_bbox: invalid bbox after clamp: ({x1},{y1})-({x2},{y2})")
            x2 = min(x1 + 10, w)
            y2 = min(y1 + 10, h)
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

    # ================================================================
    # Mask Post-processing
    # ================================================================

    def _post_process_mask(self, mask: np.ndarray, feather: int = DEFAULT_FEATHER) -> np.ndarray:
        """
        Unified mask post-processing:
        1. Binarize (threshold 30)
        2. Morphological close (fill small holes)
        3. Dilate slightly
        4. Erode boundary (reduce over-coverage at edges)
        5. Gaussian feather for soft edges
        Returns 0-255 uint8 soft mask.
        """
        # Binarize
        mask_bin = (mask > 30).astype(np.uint8) * 255

        # Morphological close to fill holes
        close_ksize = 5
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        mask_closed = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, close_kernel, iterations=2)

        # Dilate slightly
        dilate_ksize = DILATE_KERNEL
        if dilate_ksize % 2 == 0:
            dilate_ksize += 1
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize))
        mask_dilated = cv2.dilate(mask_closed, dilate_kernel, iterations=DILATE_ITER)

        # Erode boundary to reduce over-coverage on edges
        if ERODE_BOUNDARY > 0:
            erode_ksize = ERODE_BOUNDARY
            if erode_ksize % 2 == 0:
                erode_ksize += 1
            erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_ksize, erode_ksize))
            mask_eroded = cv2.erode(mask_dilated, erode_kernel, iterations=1)
            logger.debug(f"Eroded mask boundary: kernel={erode_ksize}, "
                         f"pixels before={np.sum(mask_dilated > 0)}, after={np.sum(mask_eroded > 0)}")
            mask_dilated = mask_eroded

        # Gaussian feather for soft edges
        gauss_ksize = feather * 2 + 1
        if gauss_ksize % 2 == 0:
            gauss_ksize += 1
        mask_float = mask_dilated.astype(np.float32) / 255.0
        mask_feathered = cv2.GaussianBlur(mask_float, (gauss_ksize, gauss_ksize), feather // 2)
        return (mask_feathered * 255).astype(np.uint8)

    # ================================================================
    # Main Mask Generation
    # ================================================================

    def generate_mask(self,
                      image: np.ndarray,
                      bboxes: List[Tuple[int, int, int, int]],
                      feather: int = DEFAULT_FEATHER) -> np.ndarray:
        """
        Generate combined mask for all watermark regions.
        Priority: SAM > GrabCut > bbox dilation.

        Args:
            image: RGB uint8 numpy array (H, W, 3)
            bboxes: List of (x1, y1, x2, y2) bounding boxes
            feather: Feather radius for mask edges

        Returns:
            Grayscale mask (H, W) uint8, 0=keep, 255=inpaint
            Raises ValueError if mask is empty.
        """
        h, w = image.shape[:2]

        if not bboxes:
            raise ValueError("No valid bboxes provided for mask generation")

        # Strategy 1: SAM (precise pixel-level segmentation)
        if USE_SAM and self.sam_loaded:
            try:
                mask = self._sam_generate_mask(image, bboxes)
                if mask.max() > 0:
                    mask = self._post_process_mask(mask, feather)
                    mask_pixels = np.sum(mask > 10)
                    if mask_pixels >= MIN_MASK_PIXELS:
                        logger.info(f"SAM mask: {mask_pixels} pixels ({mask_pixels / (w * h) * 100:.1f}%)")
                        return mask
                    logger.warning("SAM mask too small, falling back to GrabCut")
            except Exception as e:
                logger.warning(f"SAM mask generation failed: {e}")

        # Strategy 2: GrabCut (pixel-level refinement from bbox)
        masks = []
        for bbox in bboxes:
            # Step 1: Create coarse bbox mask
            coarse = self.mask_from_bbox((h, w), bbox, feather)

            # Step 2: Refine with GrabCut
            refined = self.refine_with_grabcut(image, coarse, bbox, iterations=GRABCUT_ITER)

            # Step 3: Post-process
            refined_soft = self._post_process_mask(refined, feather)
            masks.append(refined_soft)

        if not masks:
            raise ValueError("Mask generation failed for all bboxes")

        combined = np.maximum.reduce(masks)
        combined = np.clip(combined, 0, 255).astype(np.uint8)

        # Validate mask
        mask_pixels = np.sum(combined > 10)
        if mask_pixels < MIN_MASK_PIXELS:
            raise ValueError(
                f"Generated mask is too small ({mask_pixels} pixels < {MIN_MASK_PIXELS} minimum). "
                "Watermark region may be too faint or bbox is incorrect."
            )

        logger.info(f"GrabCut mask: {mask_pixels} pixels ({mask_pixels / (w * h) * 100:.1f}% of image)")
        return combined