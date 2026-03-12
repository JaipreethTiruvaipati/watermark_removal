"""Generate binary watermark masks for IOPaint.

For each watermarked image, produces a white-on-black PNG mask where white
pixels indicate the watermark region that IOPaint should inpaint.

Detection strategy (multi-color-space):
  - LAB space  → catches blue/grey watermarks (shift in a* and b* channels)
  - HSV space  → catches red/maroon watermarks (hue near 0/180)
  - Luminance  → catches pure-grey watermarks (neutral, not quite white)

The mask is dilated to cover anti-aliased edges around watermark text.
Returns True if a significant watermark was found, False otherwise (clean img).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Mask generation helpers
# ---------------------------------------------------------------------------

def _blue_grey_mask(lab: np.ndarray, bgr: np.ndarray) -> np.ndarray:
    """Detect blue and grey watermarks via LAB color space.

    Blue watermarks have elevated b* (yellow-blue axis, blue = negative b*).
    Grey watermarks have low saturation and mid-range L*.
    """
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    # Blue watermark: b* channel shifted toward blue (<= 120 in OpenCV encoding
    # where 128 is neutral). Also require mid-high luminance (not dark text).
    blue_mask = (b.astype(np.int16) < 118) & (L > 100) & (L < 240)

    # Grey watermark: near-neutral color (a*, b* close to 128) + mid luminance
    a_neutral = np.abs(a.astype(np.int16) - 128) < 10
    b_neutral = np.abs(b.astype(np.int16) - 128) < 10
    grey_mask = a_neutral & b_neutral & (L > 80) & (L < 220)

    # For grey, additionally require it's NOT white background (L < 235)
    grey_mask &= (L < 235)

    # Combine
    combined = blue_mask | grey_mask

    # Remove isolated noise pixels (require at least some neighbourhood density)
    combined_u8 = combined.astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    # Opening removes tiny specks
    combined_u8 = cv2.morphologyEx(combined_u8, cv2.MORPH_OPEN, kernel)

    return combined_u8


def _red_mask(hsv: np.ndarray) -> np.ndarray:
    """Detect red/maroon watermarks in HSV space."""
    lower_red1 = np.array([0,   50,  50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50,  50])
    upper_red2 = np.array([180, 255, 255])
    lower_maroon = np.array([0,  40,  30])
    upper_maroon = np.array([15, 255, 200])

    m1 = cv2.inRange(hsv, lower_red1, upper_red1)
    m2 = cv2.inRange(hsv, lower_red2, upper_red2)
    m3 = cv2.inRange(hsv, lower_maroon, upper_maroon)
    return m1 | m2 | m3


def generate_mask(img_path: str, mask_path: str, min_coverage: float = 0.003) -> bool:
    """Generate a binary mask for img_path and save to mask_path.

    Args:
        img_path:     Path to input image.
        mask_path:    Where to save the binary mask PNG.
        min_coverage: Fraction of pixels that must be watermark for image to
                      be considered watermarked (default 0.3%).

    Returns:
        True  → watermark found, mask saved.
        False → image appears clean, no mask saved.
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"  [WARN] Cannot read {img_path}")
        return False

    total_pixels = img_bgr.shape[0] * img_bgr.shape[1]

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    blue_grey = _blue_grey_mask(lab, img_bgr)
    red        = _red_mask(hsv)

    # Combine all signals
    combined = cv2.bitwise_or(blue_grey, red)

    # Dilate generously to cover anti-aliased edges (watermarks often have them)
    kernel = np.ones((7, 7), np.uint8)
    dilated = cv2.dilate(combined, kernel, iterations=3)

    coverage = np.count_nonzero(dilated) / total_pixels

    if coverage < min_coverage or coverage > 0.85:
        # Too little → probably clean; too much → probably a false positive
        return False

    # Save mask
    Path(mask_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(mask_path, dilated)
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def process_directory(input_dir: str, mask_dir: str) -> dict[str, bool]:
    """Generate masks for all images in input_dir, saving to mask_dir.

    Returns mapping of image filename → has_watermark.
    """
    input_path  = Path(input_dir)
    mask_path   = Path(mask_dir)
    mask_path.mkdir(parents=True, exist_ok=True)

    exts   = {".jpg", ".jpeg", ".png"}
    images = sorted(f for f in input_path.iterdir() if f.suffix.lower() in exts)

    results: dict[str, bool] = {}
    wm_count = 0

    for i, img_path in enumerate(images):
        out_mask = mask_path / (img_path.stem + ".png")
        has_wm   = generate_mask(str(img_path), str(out_mask))
        results[img_path.name] = has_wm
        if has_wm:
            wm_count += 1

        if (i + 1) % 20 == 0 or (i + 1) == len(images):
            print(f"  Masks: {i + 1}/{len(images)} done — {wm_count} watermarked so far")

    print(f"\nMask generation complete: {wm_count}/{len(images)} images have watermarks")
    return results


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python generate_masks.py <input_dir> <mask_dir>")
        sys.exit(1)
    process_directory(sys.argv[1], sys.argv[2])
