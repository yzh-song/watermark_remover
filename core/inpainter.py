"""
Inpainter Module - Dedicated image inpainting with LaMa + optional SDXL
Version 11.0 - Poisson blending as primary fusion strategy
"""
import logging
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional
import cv2
import torch

logger = logging.getLogger(__name__)

MODEL_DIR = Path(r"D:\AI\watermark_remover\models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class Inpainter:
    """
    Dedicated inpainting engine.
    Primary: LaMa (fast, high quality)
    Optional: SDXL Inpainting (highest quality, slower)
    NO fallback to cv2.inpaint - quality is non-negotiable.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.lama_model = None
        self.lama_loaded = False
        self.sdxl_pipe = None
        logger.info(f"Inpainter on device: {device}")

    # ================================================================
    # LaMa Inpainting (Primary)
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
        if not self.lama_loaded:
            if not self.load_lama():
                raise RuntimeError(
                    "CRITICAL: LaMa inpainting model is required. "
                    "Run: pip install simple-lama-inpainting"
                )

    def inpaint_lama(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Run LaMa inpainting on image with mask.
        Args:
            image: RGB uint8 numpy array (H, W, 3)
            mask: Grayscale uint8 numpy array (H, W), 0=keep, 255=inpaint
        Returns:
            Inpainted RGB uint8 numpy array (H, W, 3)
        """
        self._ensure_lama()
        pil_image = Image.fromarray(image.astype(np.uint8))
        # Threshold mask for LaMa (binary mask works best)
        pil_mask = Image.fromarray((mask > 10).astype(np.uint8) * 255)
        result = self.lama_model(pil_image, pil_mask)
        return np.array(result)

    # ================================================================
    # SDXL Inpainting (Optional high-quality)
    # ================================================================

    def load_sdxl(self) -> bool:
        """Load SDXL Inpainting pipeline (optional, high quality)."""
        try:
            from diffusers import StableDiffusionXLInpaintPipeline
            self.sdxl_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                variant="fp16" if self.device.type == 'cuda' else None,
            )
            self.sdxl_pipe = self.sdxl_pipe.to(self.device)
            if self.device.type == 'cuda':
                self.sdxl_pipe.enable_xformers_memory_efficient_attention()
            logger.info("[OK] SDXL Inpainting pipeline loaded")
            return True
        except ImportError:
            logger.warning("diffusers not installed, SDXL unavailable")
            return False
        except Exception as e:
            logger.warning(f"SDXL load failed: {e}")
            return False

    def inpaint_sdxl(self, image: np.ndarray, mask: np.ndarray,
                     prompt: str = "clean background, seamless, photorealistic",
                     strength: float = 0.99) -> np.ndarray:
        """Run SDXL inpainting (slower but higher quality)."""
        if self.sdxl_pipe is None:
            if not self.load_sdxl():
                raise RuntimeError("SDXL not available, use LaMa instead")
        pil_image = Image.fromarray(image.astype(np.uint8))
        pil_mask = Image.fromarray((mask > 10).astype(np.uint8) * 255)
        result = self.sdxl_pipe(
            prompt=prompt,
            image=pil_image,
            mask_image=pil_mask,
            strength=strength,
            guidance_scale=7.5,
        ).images[0]
        return np.array(result)

    # ================================================================
    # Blending Utilities
    # ================================================================

    @staticmethod
    def blend(original: np.ndarray, inpainted: np.ndarray,
              mask: np.ndarray) -> np.ndarray:
        """
        Soft alpha blending of inpainted region into original.
        Uses gaussian-softened mask for natural transitions.
        """
        orig_h, orig_w = original.shape[:2]

        # Align sizes
        inh, inw = inpainted.shape[:2]
        if (inh, inw) != (orig_h, orig_w):
            inpainted = cv2.resize(inpainted, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        mh, mw = mask.shape[:2]
        if (mh, mw) != (orig_h, orig_w):
            mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        mask_float = mask.astype(np.float32) / 255.0

        # Soft edges for natural look
        if np.max(mask_float) > 0.99:
            mask_3ch = np.stack([mask_float] * 3, axis=2)
        else:
            mask_bin = (mask_float > 0.5).astype(np.float32)
            mask_soft = cv2.GaussianBlur(mask_bin, (5, 5), 1.5)
            mask_3ch = np.stack([mask_soft] * 3, axis=2)

        final = inpainted.astype(np.float32) * mask_3ch + original.astype(np.float32) * (1 - mask_3ch)
        return np.clip(final, 0, 255).astype(np.uint8)

    @staticmethod
    def poisson_blend(original: np.ndarray, inpainted: np.ndarray,
                      mask: np.ndarray) -> np.ndarray:
        """
        Seamless blending for watermark removal.
        Strategy: Enhanced alpha blend as primary (predictable, natural-looking).
        NORMAL_CLONE as optional enhancement for highly textured regions.
        MIXED_CLONE is NOT used (preserves original color, re-introduces watermark).
        """
        try:
            # Align all dimensions first
            orig_h, orig_w = original.shape[:2]
            inh, inw = inpainted.shape[:2]
            if (inh, inw) != (orig_h, orig_w):
                inpainted = cv2.resize(inpainted, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            mh, mw = mask.shape[:2]
            if (mh, mw) != (orig_h, orig_w):
                mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            # Use lower threshold (30) matching LaMa's threshold, not 128
            # Feathered masks have many pixels in 30-128 range that still need blending
            mask_bin = (mask > 30).astype(np.uint8)
            mask_pixels = int(np.sum(mask_bin))
            if mask_pixels < 10:
                return original

            # Morphological OPEN: remove isolated noise without shrinking core mask
            kernel = np.ones((3, 3), np.uint8)
            mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel)
            if np.sum(mask_bin) == 0:
                return original

            # ===== Primary: Enhanced alpha blend with controlled feathering =====
            # Soften mask edges for seamless transition (no visible boundaries)
            mask_float = mask.astype(np.float32) / 255.0
            # Fixed 21px kernel: large enough for smooth transition, small enough to avoid white halos
            blur_ksize = 31
            if blur_ksize % 2 == 0:
                blur_ksize += 1
            mask_soft = cv2.GaussianBlur(mask_float, (blur_ksize, blur_ksize), 5)
            mask_3ch = np.stack([mask_soft] * 3, axis=2)
            result = (inpainted.astype(np.float32) * mask_3ch +
                      original.astype(np.float32) * (1.0 - mask_3ch))
            result = np.clip(result, 0, 255).astype(np.uint8)

            # Post-process: light median filter on repair region to remove white edge artifacts
            mask_bin_3ch = np.stack([(mask_soft > 0.02).astype(np.float32)] * 3, axis=2)
            repair_only = result.astype(np.float32) * mask_bin_3ch
            original_only = original.astype(np.float32) * (1.0 - mask_bin_3ch)
            repair_filtered = cv2.medianBlur(repair_only.astype(np.uint8), 3)
            result = (repair_filtered.astype(np.float32) * mask_bin_3ch +
                      original_only).astype(np.uint8)

            # Color correction: align repair region mean with original to suppress white edges
            mask_soft_bin = (mask_soft > 0.02).astype(np.float32)
            for c in range(3):
                orig_c = original[..., c].astype(np.float32)
                res_c = result[..., c].astype(np.float32)
                rep_mean = np.sum(res_c * mask_soft_bin) / max(np.sum(mask_soft_bin), 1)
                orig_mean = np.sum(orig_c * mask_soft_bin) / max(np.sum(mask_soft_bin), 1)
                result[..., c] = np.clip(
                    res_c - rep_mean + orig_mean, 0, 255
                ).astype(np.uint8)

            # ===== Optional: NORMAL_CLONE for textured regions =====
            # Only try when mask_bin is contiguous and centroid is well inside the mask
            roi_pixels = original[mask_bin > 0]
            if roi_pixels.size > 0:
                roi_var = np.var(roi_pixels.astype(np.float32))
                if roi_var > 80:
                    try:
                        # Find centroid of binary mask
                        ys, xs = np.where(mask_bin > 0)
                        if len(xs) > 0:
                            cx = int(np.mean(xs))
                            cy = int(np.mean(ys))
                            # Validate centroid: must be inside mask_bin region with margin
                            if (mask_bin[cy, cx] > 0 and
                                5 <= cx < orig_w - 5 and 5 <= cy < orig_h - 5):
                                # Ensure mask_bin is contiguous uint8
                                mask_bin_clean = np.ascontiguousarray(mask_bin)
                                original_bgr = np.ascontiguousarray(
                                    cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
                                inpainted_bgr = np.ascontiguousarray(
                                    cv2.cvtColor(inpainted, cv2.COLOR_RGB2BGR))
                                # Verify dimensions match
                                if (inpainted_bgr.shape[:2] == original_bgr.shape[:2] == mask_bin_clean.shape[:2]):
                                    clone_result = cv2.seamlessClone(
                                        inpainted_bgr, original_bgr, mask_bin_clean,
                                        (cx, cy), cv2.NORMAL_CLONE
                                    )
                                    result = cv2.cvtColor(clone_result, cv2.COLOR_BGR2RGB)
                    except Exception:
                        pass  # Fall through to alpha blend result

            return result
        except Exception as e:
            logger.warning(f"Poisson blend failed: {e}, using alpha blend")
            return Inpainter.blend(original, inpainted, mask)

    # ================================================================
    # Main Inpaint Interface
    # ================================================================

    def inpaint(self, image: np.ndarray, mask: np.ndarray,
                method: str = 'lama') -> np.ndarray:
        """
        Main inpainting entry point.

        Args:
            image: RGB uint8 numpy array (H, W, 3)
            mask: Grayscale uint8 numpy array (H, W), 0=keep, 255=inpaint
            method: 'lama' (default) or 'sdxl'

        Returns:
            Inpainted RGB uint8 numpy array (H, W, 3)
        """
        if method == 'sdxl':
            result = self.inpaint_sdxl(image, mask)
        else:
            result = self.inpaint_lama(image, mask)
        return result