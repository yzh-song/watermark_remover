"""
Inpainter Module - LaMa inpainting + Alpha blending.
Version 12.0 - LaMa only, alpha blending only, size alignment, edge filtering.
"""
import logging
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional
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
INPAINT_CFG = CONFIG.get("inpaint", {})
BLEND_CFG = CONFIG.get("blend", {})
LAMA_RESIZE_THRESHOLD = INPAINT_CFG.get("lama_resize_threshold", 1024)
BLUR_KSIZE = BLEND_CFG.get("blur_ksize", 21)
EDGE_FILTER_KSIZE = BLEND_CFG.get("edge_filter_ksize", 3)


class Inpainter:
    """
    Dedicated inpainting engine.
    LaMa only - no fallback to cv2.inpaint.
    Alpha blending only - no Poisson blending.
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

    def _resize_for_lama(self, image: np.ndarray, mask: np.ndarray):
        """
        Resize image and mask to fit within LaMa's optimal size range.
        If the longest dimension exceeds LAMA_RESIZE_THRESHOLD, scale down
        to a multiple of 8 for alignment.
        """
        h, w = image.shape[:2]
        max_dim = max(h, w)
        if max_dim <= LAMA_RESIZE_THRESHOLD:
            # Already small enough, just ensure 8-pixel alignment
            new_h = h - (h % 8) if h % 8 != 0 else h
            new_w = w - (w % 8) if w % 8 != 0 else w
            if new_h != h or new_w != w:
                image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                return image_resized, mask_resized, (h, w)
            return image, mask, (h, w)

        # Scale down to threshold
        scale = LAMA_RESIZE_THRESHOLD / max_dim
        new_h = int(h * scale)
        new_w = int(w * scale)
        # Align to 8
        new_h = new_h - (new_h % 8)
        new_w = new_w - (new_w % 8)
        new_h = max(8, new_h)
        new_w = max(8, new_w)

        image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        logger.info(f"Resized for LaMa: ({w},{h}) -> ({new_w},{new_h})")
        return image_resized, mask_resized, (h, w)

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Run LaMa inpainting with size alignment.

        Args:
            image: RGB uint8 numpy array (H, W, 3)
            mask: Grayscale uint8 numpy array (H, W), 0=keep, 255=inpaint

        Returns:
            Inpainted RGB uint8 numpy array (H, W, 3)
        """
        self._ensure_lama()

        # Resize for optimal LaMa performance
        image_resized, mask_resized, original_size = self._resize_for_lama(image, mask)

        # Threshold mask for LaMa
        pil_image = Image.fromarray(image_resized.astype(np.uint8))
        pil_mask = Image.fromarray((mask_resized > 10).astype(np.uint8) * 255)

        result = self.lama_model(pil_image, pil_mask)
        result_np = np.array(result)

        # Restore original size if resized
        if result_np.shape[:2] != original_size:
            result_np = cv2.resize(result_np, (original_size[1], original_size[0]),
                                   interpolation=cv2.INTER_LINEAR)

        return result_np

    # ================================================================
    # Alpha Blending
    # ================================================================

    @staticmethod
    def blend(original: np.ndarray, inpainted: np.ndarray,
              mask: np.ndarray) -> np.ndarray:
        """
        Alpha blending of inpainted region into original.
        Uses soft mask for natural transitions + edge median filtering.

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

        # Normalize mask to 0-1
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

        # Edge filtering: median filter on repair region to remove white edges
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

        return result