"""
Inpainter Module - LaMa inpainting + Alpha blending + Poisson edge fusion.
Version 12.0 - Padding-based resize, color correction, auto-retry, boundary-aware Poisson.
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
QUALITY_SSIM_THRESHOLD = INPAINT_CFG.get("quality_ssim_threshold", 0.85)
AUTO_RETRY = INPAINT_CFG.get("auto_retry", True)
MAX_RETRIES = INPAINT_CFG.get("max_retries", 2)
RETRY_MASK_EXPAND = INPAINT_CFG.get("retry_mask_expand", 0.10)
BLUR_KSIZE = BLEND_CFG.get("blur_ksize", 31)
EDGE_FILTER_KSIZE = BLEND_CFG.get("edge_filter_ksize", 3)
POISSON_EDGE_WIDTH = BLEND_CFG.get("poisson_edge_width", 10)
COLOR_CORRECTION = BLEND_CFG.get("color_correction", True)


class Inpainter:
    """
    Dedicated inpainting engine.
    LaMa only - no fallback to cv2.inpaint.
    Alpha blending + Poisson edge fusion for seamless results.
    v12.0: Padding-based resize, color correction, auto-retry with mask expansion.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.lama_model = None
        self.lama_loaded = False
        logger.info(f"Inpainter on device: {device}")

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
        if not self.lama_loaded:
            if not self.load_lama():
                raise RuntimeError(
                    "CRITICAL: LaMa inpainting model is required. "
                    "Run: pip install simple-lama-inpainting"
                )

    def _resize_with_padding(self, image: np.ndarray, mask: np.ndarray,
                             multiple: int = LAMA_RESIZE_MULTIPLE) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
        """
        Pad image and mask to the nearest multiple of N using reflection padding.
        Padding preserves original image details better than resize.
        """
        h, w = image.shape[:2]
        new_h = ((h + multiple - 1) // multiple) * multiple
        new_w = ((w + multiple - 1) // multiple) * multiple
        pad_top = 0
        pad_bottom = new_h - h
        pad_left = 0
        pad_right = new_w - w
        if pad_bottom == 0 and pad_right == 0:
            return image, mask, (0, 0, 0, 0)
        image_padded = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)
        mask_padded = cv2.copyMakeBorder(
            mask, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        logger.debug(f"Padded for LaMa: ({w},{h}) -> ({new_w},{new_h})")
        return image_padded, mask_padded, (pad_top, pad_bottom, pad_left, pad_right)

    def _expand_mask(self, mask: np.ndarray, expand_ratio: float) -> np.ndarray:
        """Expand mask by a given ratio using dilation for auto-retry."""
        mask_bin = (mask > 30).astype(np.uint8) * 255
        if mask_bin.sum() == 0:
            return mask
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask
        x, y, bw, bh = cv2.boundingRect(np.vstack(contours))
        expand_px = max(int(max(bw, bh) * expand_ratio), 5)
        ksize = expand_px * 2 + 1
        if ksize % 2 == 0:
            ksize += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        expanded = cv2.dilate(mask_bin, kernel, iterations=1)
        logger.info(f"Auto-retry: mask expanded by {expand_px}px ({expand_ratio*100:.0f}%)")
        return expanded.astype(np.uint8)

    def _check_quality(self, original: np.ndarray, inpainted: np.ndarray,
                       mask: np.ndarray) -> float:
        """Check inpainting quality using SSIM on the whole image."""
        try:
            original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
            inpainted_gray = cv2.cvtColor(inpainted, cv2.COLOR_RGB2GRAY)
            if original_gray.shape != inpainted_gray.shape:
                inpainted_gray = cv2.resize(inpainted_gray,
                                            (original_gray.shape[1], original_gray.shape[0]))
            score = ssim(original_gray, inpainted_gray, data_range=255)
            if score < QUALITY_SSIM_THRESHOLD:
                logger.warning(f"SSIM {score:.3f} < threshold {QUALITY_SSIM_THRESHOLD}")
            else:
                logger.info(f"SSIM quality check: {score:.3f} (OK)")
            return score
        except Exception as e:
            logger.warning(f"SSIM check failed: {e}")
            return -1.0

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Run LaMa inpainting with padding-based alignment and auto-retry.
        When SSIM < threshold, automatically expands mask and retries.
        """
        self._ensure_lama()
        best_result = None
        best_ssim = -1.0
        current_mask = mask.copy()
        original_h, original_w = image.shape[:2]

        for attempt in range(MAX_RETRIES + 1):
            image_padded, mask_padded, pads = self._resize_with_padding(image, current_mask)
            pil_image = Image.fromarray(image_padded.astype(np.uint8))
            pil_mask = Image.fromarray((mask_padded > 10).astype(np.uint8) * 255)
            result = self.lama_model(pil_image, pil_mask)
            result_np = np.array(result)

            pad_top, pad_bottom, pad_left, pad_right = pads
            if pad_bottom > 0 or pad_right > 0:
                result_np = result_np[pad_top:result_np.shape[0] - pad_bottom,
                                      pad_left:result_np.shape[1] - pad_right]
                if result_np.shape[:2] != (original_h, original_w):
                    result_np = cv2.resize(result_np, (original_w, original_h),
                                           interpolation=cv2.INTER_LINEAR)

            ssim_score = self._check_quality(image, result_np, current_mask)
            if ssim_score > best_ssim:
                best_ssim = ssim_score
                best_result = result_np

            if AUTO_RETRY and attempt < MAX_RETRIES and ssim_score < QUALITY_SSIM_THRESHOLD:
                current_mask = self._expand_mask(current_mask, RETRY_MASK_EXPAND)
                logger.info(f"Auto-retry attempt {attempt + 2}/{MAX_RETRIES + 1}")
            else:
                break

        if best_ssim < QUALITY_SSIM_THRESHOLD and best_ssim >= 0:
            logger.warning(f"Best SSIM {best_ssim:.3f} still below threshold")
        return best_result

    # ================================================================
    # Color Correction
    # ================================================================

    @staticmethod
    def _color_correction(original: np.ndarray, inpainted: np.ndarray,
                          mask: np.ndarray) -> np.ndarray:
        """
        Mean-variance color alignment on the inpainted region.
        Matches the color statistics of the inpainted region to the surrounding background.
        """
        mask_bin = (mask > 30).astype(np.uint8)
        if mask_bin.sum() == 0:
            return inpainted
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask_dilated = cv2.dilate(mask_bin, kernel, iterations=1)
        border_mask = mask_dilated - mask_bin
        if border_mask.sum() < 100:
            return inpainted

        result = inpainted.copy().astype(np.float32)
        for c in range(3):
            ref_pixels = original[:, :, c][border_mask > 0]
            ref_mean = np.mean(ref_pixels)
            ref_std = np.std(ref_pixels) + 1e-6
            inp_pixels = inpainted[:, :, c][mask_bin > 0]
            inp_mean = np.mean(inp_pixels)
            inp_std = np.std(inp_pixels) + 1e-6
            channel = result[:, :, c]
            channel[mask_bin > 0] = (channel[mask_bin > 0] - inp_mean) * (ref_std / inp_std) + ref_mean
        return np.clip(result, 0, 255).astype(np.uint8)

    # ================================================================
    # Alpha Blending + Poisson Edge Fusion
    # ================================================================

    @staticmethod
    def _is_mask_touching_boundary(mask: np.ndarray, margin: int = 5) -> bool:
        """Check if the mask touches the image boundary. Poisson fusion is disabled in this case."""
        mask_bin = (mask > 30).astype(np.uint8)
        h, w = mask_bin.shape
        if np.any(mask_bin[:margin, :] > 0):
            return True
        if np.any(mask_bin[-margin:, :] > 0):
            return True
        if np.any(mask_bin[:, :margin] > 0):
            return True
        if np.any(mask_bin[:, -margin:] > 0):
            return True
        return False

    @staticmethod
    def blend(original: np.ndarray, inpainted: np.ndarray,
              mask: np.ndarray) -> np.ndarray:
        """
        Edge-aware blending:
        1. Color correction (mean-variance alignment) on inpainted region
        2. Alpha blending with soft mask as base
        3. Median filter on repair edges to remove artifacts
        4. Poisson seamless cloning on edge strip (disabled if mask touches boundary)
        """
        orig_h, orig_w = original.shape[:2]

        # Align sizes
        inh, inw = inpainted.shape[:2]
        if (inh, inw) != (orig_h, orig_w):
            inpainted = cv2.resize(inpainted, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        mh, mw = mask.shape[:2]
        if (mh, mw) != (orig_h, orig_w):
            mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # --- Step 0: Color correction ---
        if COLOR_CORRECTION:
            try:
                inpainted = Inpainter._color_correction(original, inpainted, mask)
                logger.debug("Color correction applied")
            except Exception as e:
                logger.warning(f"Color correction failed: {e}")

        # --- Step 1: Alpha blending ---
        mask_float = mask.astype(np.float32) / 255.0
        ksize = BLUR_KSIZE
        if ksize % 2 == 0:
            ksize += 1
        mask_soft = cv2.GaussianBlur(mask_float, (ksize, ksize), ksize // 4)
        mask_3ch = np.stack([mask_soft] * 3, axis=2)
        result = (inpainted.astype(np.float32) * mask_3ch +
                  original.astype(np.float32) * (1.0 - mask_3ch))
        result = np.clip(result, 0, 255).astype(np.uint8)

        # --- Step 2: Edge median filtering ---
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

        # --- Step 3: Poisson edge fusion (disabled if mask touches boundary) ---
        edge_width = POISSON_EDGE_WIDTH
        if edge_width > 0 and not Inpainter._is_mask_touching_boundary(mask):
            try:
                mask_bin = (mask > 30).astype(np.uint8)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_width * 2 + 1, edge_width * 2 + 1))
                mask_dilated = cv2.dilate(mask_bin, kernel, iterations=1)
                edge_mask = mask_dilated - mask_bin
                if edge_mask.sum() > 100:
                    ys, xs = np.where(mask_bin > 0)
                    if len(ys) > 0:
                        center_x = (xs.min() + xs.max()) // 2
                        center_y = (ys.min() + ys.max()) // 2
                        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                        inpainted_bgr = cv2.cvtColor(inpainted, cv2.COLOR_RGB2BGR)
                        result_bgr = cv2.seamlessClone(
                            inpainted_bgr, result_bgr, edge_mask,
                            (center_x, center_y), cv2.NORMAL_CLONE)
                        result = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
                        logger.debug(f"Poisson edge fusion applied (edge_width={edge_width}px)")
            except Exception as e:
                logger.warning(f"Poisson edge fusion failed: {e}")
        elif edge_width > 0:
            logger.debug("Poisson fusion disabled: mask touches image boundary")

        return result