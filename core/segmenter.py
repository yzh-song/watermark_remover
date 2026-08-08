"""
Segmenter Module - Fine mask generation with SAM2 / MobileSAM / CV fallback
Version 11.0 - Edge-aware feathering with Guided Filter + Joint Bilateral Filter
"""
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import torch

logger = logging.getLogger(__name__)

MODEL_DIR = Path(r"D:\AI\watermark_remover\models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Check for ximgproc (edge-aware filters)
_HAS_XIMGPROC = hasattr(cv2, 'ximgproc')


class Segmenter:
    """
    Multi-level mask generator for watermark regions.
    Level 1: Bounding box + dilation + feather (fast, always available)
    Level 2: SAM2 precise mask (accurate, requires model)
    Level 3: GrabCut refinement (medium, uses OpenCV built-in)
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.sam_model = None
        self.sam_loaded = False
        self.sam_type = None  # 'sam2' or 'mobile_sam'
        logger.info(f"Segmenter on device: {device}")

    # ================================================================
    # Level 1: Bounding Box Mask (always available)
    # ================================================================

    @staticmethod
    def mask_from_bbox(image_shape: Tuple[int, int],
                       bbox: Tuple[int, int, int, int],
                       feather: int = 12,
                       guide_image: np.ndarray = None) -> np.ndarray:
        """
        Create soft-edged mask from bounding box.
        Uses dilation + edge-aware feathering (Guided Filter > Gaussian fallback).
        Edge-aware feathering preserves sharp boundaries at natural edges,
        avoiding the blurry halo effect of pure Gaussian smoothing.
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
        dilate_px = max(feather, 8)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)
        )
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Edge-aware feathering: Guided Filter preserves edges, avoids blurry halo
        if _HAS_XIMGPROC and guide_image is not None:
            try:
                guide_gray = cv2.cvtColor(guide_image, cv2.COLOR_RGB2GRAY)
                mask_float = mask.astype(np.float32) / 255.0
                # Guided filter: preserve edges from guide image
                radius = max(feather, 10)
                eps = 1e-4
                mask_feathered = cv2.ximgproc.guidedFilter(
                    guide_gray.astype(np.float32), mask_float, radius, eps
                )
                return (np.clip(mask_feathered, 0, 1) * 255).astype(np.uint8)
            except Exception as e:
                logger.debug(f"GuidedFilter failed: {e}, falling back to Gaussian")

        # Fallback: Gaussian feather (less precise but always works)
        mask_float = mask.astype(np.float32) / 255.0
        mask_feathered = cv2.GaussianBlur(
            mask_float, (feather * 2 + 1, feather * 2 + 1), feather // 2
        )
        return (mask_feathered * 255).astype(np.uint8)

    # ================================================================
    # Level 2: SAM2 Precise Mask
    # ================================================================

    def load_sam(self, model_type: str = 'mobile_sam') -> bool:
        """
        Load SAM model. Prefers MobileSAM for speed, SAM2 for accuracy.
        Args:
            model_type: 'mobile_sam' (fast) or 'sam2' (accurate)
        """
        try:
            if model_type == 'mobile_sam':
                return self._load_mobile_sam()
            else:
                return self._load_sam2()
        except Exception as e:
            logger.warning(f"SAM load failed: {e}")
            return False

    def _load_mobile_sam(self) -> bool:
        """Load MobileSAM (lightweight, fast)."""
        try:
            from ultralytics import SAM
            model_path = MODEL_DIR / "mobile_sam.pt"
            if not model_path.exists():
                logger.info("Downloading MobileSAM model...")
                self.sam_model = SAM("mobile_sam.pt")
                # Save to models dir
                import shutil
                shutil.copy("mobile_sam.pt", str(model_path))
            else:
                self.sam_model = SAM(str(model_path))
            self.sam_loaded = True
            self.sam_type = 'mobile_sam'
            logger.info("[OK] MobileSAM loaded")
            return True
        except ImportError:
            logger.warning("ultralytics not installed, SAM unavailable")
            return False
        except Exception as e:
            logger.warning(f"MobileSAM load failed: {e}")
            return False

    def _load_sam2(self) -> bool:
        """Load SAM2 (high accuracy, larger)."""
        try:
            from segment_anything import sam_model_registry, SamPredictor
            model_path = MODEL_DIR / "sam_vit_h_4b8939.pth"
            if not model_path.exists():
                logger.warning(f"SAM2 model not found: {model_path}")
                return False
            self.sam_model = sam_model_registry["vit_h"](checkpoint=str(model_path))
            self.sam_model.to(self.device)
            self.sam_loaded = True
            self.sam_type = 'sam2'
            logger.info("[OK] SAM2 loaded")
            return True
        except ImportError:
            logger.warning("segment-anything not installed, SAM2 unavailable")
            return False
        except Exception as e:
            logger.warning(f"SAM2 load failed: {e}")
            return False

    def _predict_sam_mask(self, image: np.ndarray,
                          bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Generate fine mask using SAM for a given bounding box."""
        if self.sam_type == 'mobile_sam':
            return self._predict_mobile_sam(image, bbox)
        else:
            return self._predict_sam2(image, bbox)

    def _predict_mobile_sam(self, image: np.ndarray,
                            bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """MobileSAM prediction via ultralytics."""
        from ultralytics import SAM
        x1, y1, x2, y2 = bbox
        results = self.sam_model(image, bboxes=[bbox])
        if results and len(results) > 0 and results[0].masks is not None:
            mask = results[0].masks.data[0].cpu().numpy()
            mask = (mask * 255).astype(np.uint8)
            return mask
        return None

    def _predict_sam2(self, image: np.ndarray,
                      bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """SAM2 prediction via segment-anything."""
        from segment_anything import SamPredictor
        predictor = SamPredictor(self.sam_model)
        predictor.set_image(image)
        x1, y1, x2, y2 = bbox
        input_box = np.array([x1, y1, x2, y2])
        masks, scores, _ = predictor.predict(
            box=input_box[None, :],
            multimask_output=False
        )
        if masks is not None and len(masks) > 0:
            mask = (masks[0] * 255).astype(np.uint8)
            return mask
        return None

    # ================================================================
    # Level 3: GrabCut Refinement
    # ================================================================

    @staticmethod
    def refine_with_grabcut(image: np.ndarray,
                            mask: np.ndarray,
                            bbox: Tuple[int, int, int, int],
                            iterations: int = 3) -> np.ndarray:
        """
        Refine a coarse mask using OpenCV GrabCut.
        Produces pixel-accurate boundaries.
        """
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        x1 = max(0, x1 - 10); y1 = max(0, y1 - 10)
        x2 = min(w, x2 + 10); y2 = min(h, y2 + 10)

        # Create initial trimap from mask
        mask_bin = (mask > 128).astype(np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        # Use mask as initial GC mask: 0=bg, 1=fg, 2=prob_bg, 3=prob_fg
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
            logger.warning(f"GrabCut failed: {e}")
            return mask_bin

    # ================================================================
    # Main Interface
    # ================================================================

    def generate_mask(self,
                      image: np.ndarray,
                      bboxes: List[Tuple[int, int, int, int]],
                      method: str = 'auto',
                      feather: int = 12) -> np.ndarray:
        """
        Generate combined mask for all watermark regions.

        Args:
            image: RGB uint8 numpy array (H, W, 3)
            bboxes: List of (x1, y1, x2, y2) bounding boxes
            method: 'auto' (try SAM then fallback), 'sam', 'grabcut', 'bbox'
            feather: Feather radius for bbox masks

        Returns:
            Grayscale mask (H, W) uint8, 0=keep, 255=inpaint
        """
        h, w = image.shape[:2]
        masks = []

        for bbox in bboxes:
            if method == 'auto' and self.sam_loaded:
                fine_mask = self._predict_sam_mask(image, bbox)
                if fine_mask is not None:
                    # Edge-preserving smoothing: Joint Bilateral Filter keeps edges sharp
                    if _HAS_XIMGPROC:
                        try:
                            guide = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                            fine_float = fine_mask.astype(np.float32) / 255.0
                            # Joint bilateral: edge-aware smoothing guided by original image
                            fine_soft = cv2.ximgproc.jointBilateralFilter(
                                guide, fine_float.astype(np.float32),
                                d=9, sigmaColor=50, sigmaSpace=10
                            )
                            masks.append((np.clip(fine_soft, 0, 1) * 255).astype(np.uint8))
                        except Exception as e:
                            logger.debug(f"JointBilateralFilter failed: {e}, using Gaussian")
                            fine_float = fine_mask.astype(np.float32) / 255.0
                            fine_soft = cv2.GaussianBlur(fine_float, (11, 11), 3)
                            masks.append((fine_soft * 255).astype(np.uint8))
                    else:
                        # Fallback: Gaussian softening
                        fine_float = fine_mask.astype(np.float32) / 255.0
                        fine_soft = cv2.GaussianBlur(fine_float, (11, 11), 3)
                        masks.append((fine_soft * 255).astype(np.uint8))
                    continue

            if method in ('grabcut', 'auto') and not self.sam_loaded:
                # Use bbox mask + grabcut refinement
                coarse = Segmenter.mask_from_bbox((h, w), bbox, feather, guide_image=image)
                refined = Segmenter.refine_with_grabcut(image, coarse, bbox)
                masks.append(refined)
                continue

            # Fallback: bbox mask with edge-aware feathering
            masks.append(Segmenter.mask_from_bbox((h, w), bbox, feather, guide_image=image))

        if not masks:
            return np.zeros((h, w), dtype=np.uint8)

        combined = np.maximum.reduce(masks)
        return np.clip(combined, 0, 255).astype(np.uint8)

    def generate_mask_from_saliency(self,
                                     image: np.ndarray,
                                     saliency_map: np.ndarray,
                                     threshold: float = 0.3) -> np.ndarray:
        """
        Generate mask from a saliency/attention map (e.g. from U2-Net).
        Applies thresholding, morphology, and feathering.
        """
        h, w = image.shape[:2]
        if saliency_map.shape[:2] != (h, w):
            saliency_map = cv2.resize(saliency_map, (w, h))

        # Threshold
        _, binary = cv2.threshold(
            (saliency_map * 255).astype(np.uint8) if saliency_map.max() <= 1.0
            else saliency_map.astype(np.uint8),
            int(threshold * 255), 255, cv2.THRESH_BINARY
        )

        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Soft edges
        binary_float = binary.astype(np.float32) / 255.0
        soft = cv2.GaussianBlur(binary_float, (15, 15), 5)
        return (soft * 255).astype(np.uint8)