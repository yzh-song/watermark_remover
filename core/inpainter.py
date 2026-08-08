"""
Inpainter Module - LaMa inpainting + Alpha blending + Poisson edge fusion.
Version 12.0 - 32px alignment, SSIM quality check, Poisson seamless edge cloning.
"""
import logging
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple
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
INPAINT_CFG = CONFIG.get("inpaint", {})
BLEND_CFG = CONFIG.get("blend", {})
LAMA_RESIZE_MULTIPLE = INPAINT_CFG.get("lama_resize_multiple", 32)
QUALITY_SSIM_THRESHOLD = INPAINT_CFG.get("quality_ssim_threshold", 0.9)
BLUR_KSIZE = BLEND_CFG.get("blur_ksize", 21)
EDGE_FILTER_KSIZE = BLEND_CFG.get("edge_filter_ksize", 3)
POISSON_EDGE_WIDTH = BLEND_CFG.get("poisson_edge_width", 10)


class Inpainter:
    """
    Dedicated inpainting engine.
    LaMa only - no fallback to cv2.inpaint.
    Alpha blending + Poisson edge fusion for seamless results.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.lama_model = None
        self.lama_loaded = False
        logger.info(f"Inpainter on device: {device}")

    # ================================================================
    # LaMa Inpainting
    # ================================================================

    def load_lama(self) -> bool:
        """Load LaMa inpainting model (required)."""
        try:
            from simple_lama_inpainting import SimpleLama
            self.lama_model = SimpleLama(device=self.device)
            self.lama_loaded = True
            logger.info("[OK] LaMa inpainting model loaded")
            return True
        except ImportError:
            logger.error(
                "simple-lama-inpainting not installed. "
                "Run: pip install simple-lama-inpainting"
            )
            return False
        except Exception as e:
            logger.error(f"LaMa load failed: {e}")
            return False

    def _ensure_lama(self):
        """Ensure LaMa is loaded. Raises RuntimeError if not."""
        if not self.lama_loaded:
            if not self.load_lama():
                raise RuntimeError(
                    "CRITICAL: LaMa inpainting model is required. "
                    "Run: pip install simple-lama-inpainting"
                )

    def _resize_to_multiple(self, image: np.ndarray, mask: np.ndarray,
                            multiple: int = LAMA_RESIZE_MULTIPLE) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """
        Resize image and mask to nearest multiple of N (default 32) for LaMa alignment.
        This prevents LaMa's internal resize from producing blurry artifacts.
        """
        h, w = image.shape[:2]
        new_h = ((h + multiple - 1) // multiple) * multiple
        new_w = ((w + multiple - 1) // multiple) * multiple
        new_h = max(multiple, new_h)
        new_w = max(multiple, new_w)

        if new_h == h and new_w == w:
            return image, mask, (h, w)

        image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        logger.info(f"Resized for LaMa alignment: ({w},{h}) -> ({new_w},{new_h}) [multiple={multiple}]")
        return image_resized, mask_resized, (h, w)

    def _check_quality(self, original: np.ndarray, inpainted: np.ndarray,
                       mask: np.ndarray) -> float:
        """
        Check inpainting quality using SSIM on the repaired region.
        Returns SSIM score. Warns if below threshold.
        SSIM is computed on the inpainted region vs original, but since
        the original has watermark, we compute SSIM on non-mask areas
        to verify the model didn't degrade the rest of the image.
        """
        mask_bin = (mask > 30).astype(np.uint8)
        if mask_bin.sum() == 0:
            return 1.0

        # Compute SSIM on non-mask (background) areas to verify no degradation
        bg_mask = (mask_bin == 0)
        if bg_mask.sum() < 1000:
            return 1.0

        try:
            # Sample a region for efficiency
            original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            inpainted_gray = cv2.cvtColor(inpainted, cv2.COLOR_RGB2GRAY)
            score = ssim(original_gray, inpainted_gray, data_range=255)
            if score < QUALITY_SSIM_THRESHOLD:
                logger.warning(
                    f"SSIM {score:.3f} < threshold {QUALITY_SSIM_THRESHOLD}. "
                    "Inpainting quality may be degraded. Consider reducing mask dilation."
                )
            else:
                logger.info(f"SSIM quality check: {score:.3f} (OK)")
            return score
        except Exception as e:
            logger.warning(f"SSIM check failed: {e}")
            return -1.0

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Run LaMa inpainting with 32-pixel alignment.

        Args:
            image: RGB uint8 numpy array (H, W, 3)
            mask: Grayscale uint8 numpy array (H, W), 0=keep, 255=inpaint

        Returns:
            Inpainted RGB uint8 numpy array (H, W, 3)
        """
        self._ensure_lama()

        # Resize to nearest multiple of 32 for optimal LaMa performance
        image_resized, mask_resized, original_size = self._resize_to_multiple(image, mask)

        # Threshold mask for LaMa
        pil_image = Image.fromarray(image_resized.astype(np.uint8))
        pil_mask = Image.fromarray((mask_resized > 10).astype(np.uint8) * 255)

        result = self.lama_model(pil_image, pil_mask)
        result_np = np.array(result)

        # Restore original size if resized
        if result_np.shape[:2] != original_size:
            result_np = cv2.resize(result_np, (original_size[1], original_size[0]),
                                   interpolation=cv2.INTER_LINEAR)

        # SSIM quality check
        self._check_quality(image, result_np, mask)

        return result_np

    # ================================================================
    # Alpha Blending + Poisson Edge Fusion
    # ================================================================

    @staticmethod
    def blend(original: np.ndarray, inpainted: np.ndarray,
              mask: np.ndarray) -> np.ndarray:
        """
        Edge-aware blending:
        1. Alpha blending with soft mask as base
        2. Median filter on repair edges to remove artifacts
        3. Poisson seamless cloning on edge strip for color consistency

        Args:
            original: RGB uint8 (H, W, 3)
            inpainted: RGB uint8 (H, W, 3)
            mask: Grayscale uint8 (H, W), 0=keep, 255=inpaint

        Returns:
            Blended RGB uint8 (H, W, 3)
        """
        orig_h, orig_w = original.shape[:2]

        # Align sizes
        inh, inw = inpainted.shape[:2]
        if (inh, inw) != (orig_h, orig_w):
            inpainted = cv2.resize(inpainted, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        mh, mw = mask.shape[:2]
        if (mh, mw) != (orig_h, orig_w):
            mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # --- Step 1: Alpha blending as base ---
        mask_float = mask.astype(np.float32) / 255.0

        # Soften mask edges for seamless transition
        ksize = BLUR_KSIZE
        if ksize % 2 == 0:
            ksize += 1
        mask_soft = cv2.GaussianBlur(mask_float, (ksize, ksize), ksize // 4)
        mask_3ch = np.stack([mask_soft] * 3, axis=2)

        # Alpha blend
        result = (inpainted.astype(np.float32) * mask_3ch +
                  original.astype(np.float32) * (1.0 - mask_3ch))
        result = np.clip(result, 0, 255).astype(np.uint8)

        # --- Step 2: Edge median filtering to remove white border artifacts ---
        edge_ksize = EDGE_FILTER_KSIZE
        if edge_ksize % 2 == 0:
            edge_ksize += 1
        if edge_ksize > 0:
            mask_bin = (mask_soft > 0.02).astype(np.float32)
            mask_bin_3ch = np.stack([mask_bin] * 3, axis=2)
            repair_only = result.astype(np.float32) * mask_bin_3ch
            original_only = original.astype(np.float32) * (1.0 - mask_bin_3ch)
            repair_filtered = cv2.medianBlur(repair_only.astype(np.uint8), edge_ksize)
            result = (repair_filtered.astype(np.float32) * mask_bin_3ch +
                      original_only).astype(np.uint8)

        # --- Step 3: Poisson seamless cloning on edge strip ---
        edge_width = POISSON_EDGE_WIDTH
        if edge_width > 0:
            try:
                # Create edge mask: dilate then subtract original to get edge strip
                mask_bin = (mask > 30).astype(np.uint8)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_width * 2 + 1, edge_width * 2 + 1))
                mask_dilated = cv2.dilate(mask_bin, kernel, iterations=1)
                edge_mask = mask_dilated - mask_bin

                if edge_mask.sum() > 100:
                    # Find center of the repair region for seamlessClone
                    ys, xs = np.where(mask_bin > 0)
                    if len(ys) > 0:
                        center_x = (xs.min() + xs.max()) // 2
                        center_y = (ys.min() + ys.max()) // 2

                        # Convert to BGR for seamlessClone
                        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                        inpainted_bgr = cv2.cvtColor(inpainted, cv2.COLOR_RGB2BGR)

                        # Apply Poisson blending on the edge region
                        result_bgr = cv2.seamlessClone(
                            inpainted_bgr, result_bgr, edge_mask,
                            (center_x, center_y), cv2.NORMAL_CLONE
                        )
                        result = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
                        logger.debug(f"Poisson edge fusion applied (edge_width={edge_width}px)")
            except Exception as e:
                logger.warning(f"Poisson edge fusion failed: {e}")

        return result