"""
Physics Diagram Cleaner v3
==========================
Removes watermarks (blue rings, red text, beige overlays) and enhances
scanned physics diagrams to sharp black-on-white grayscale.

Core technique: Morphological background division
  background = DILATE(gray, large kernel)   <- smooth background without ink
  normalized = gray / background * 255      <- background becomes uniform white

Usage:
    python cleaner_v3.py                          # processes ./input → ./output
    python cleaner_v3.py --input /path/in --output /path/out
    python cleaner_v3.py --input single_image.png --output cleaned.png

Requirements:
    pip install opencv-python-headless numpy
"""

import cv2
import numpy as np
import os
import sys
import argparse
from pathlib import Path

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


# ──────────────────────────────────────────────────────────────────────────────
# Core processing
# ──────────────────────────────────────────────────────────────────────────────

def clean_image(img: np.ndarray) -> np.ndarray:
    """
    Full pipeline: watermark removal + contrast enhancement.
    Input : BGR uint8 image
    Output: grayscale uint8 image (clean, high-contrast, white background)
    """

    # ── Step 1: Convert to grayscale ─────────────────────────────────────────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── Step 2: Morphological background estimation ───────────────────────────
    # Dilating with a large kernel replaces every pixel with the maximum
    # value in a big neighbourhood. Since ink is dark and surrounded by light
    # background, the ink pixels get "erased" → we get a smooth map of just
    # the background brightness (including any watermark tinting).
    kernel_size = max(61, (max(img.shape[:2]) // 20) | 1)  # scales with image size, must be odd
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    background = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)

    # ── Step 3: Divide out the background ────────────────────────────────────
    # Normalises the image so background → 255 everywhere, regardless of
    # whether the tint came from a blue ring, beige overlay, or colour cast.
    normalized = np.clip(
        255.0 * gray.astype(np.float64) / (background.astype(np.float64) + 1e-6),
        0, 255
    ).astype(np.uint8)

    # ── Step 4: Remove red watermark text ────────────────────────────────────
    # e.g. "TG ~ @bohring_bot" — these are fully saturated red pixels that
    # the background division alone won't remove (they're dark, not light).
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red_mask = (
        ((hue <= 12) | (hue >= 165)) & (sat > 80) & (val > 80)
    ).astype(np.uint8) * 255

    if red_mask.sum() > 500:
        kernel_r = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        red_dilated = cv2.dilate(red_mask, kernel_r, iterations=2)
        normalized[red_dilated > 0] = 255

    # ── Step 5: Hard clamp — any near-white pixel becomes pure white ──────────
    # Kills any remaining ghost tints (values 216–254 → 255).
    # Diagram ink is always well below 215 after normalisation.
    normalized[normalized > 215] = 255

    # ── Step 6: Stretch ink tones for better contrast ─────────────────────────
    dark_mask = normalized < 200
    if dark_mask.sum() > 100:
        lo = float(np.percentile(normalized[dark_mask], 2))
        normalized[dark_mask] = np.clip(
            (normalized[dark_mask].astype(np.float32) - lo) / max(200 - lo, 1) * 180,
            0, 200
        ).astype(np.uint8)

    # ── Step 7: Unsharp mask — sharpen diagram lines ─────────────────────────
    blur = cv2.GaussianBlur(normalized, (0, 0), 1.2)
    sharp = cv2.addWeighted(normalized, 1.6, blur, -0.6, 0)

    return sharp


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def process_file(src: Path, dst: Path) -> None:
    img = cv2.imread(str(src))
    if img is None:
        print(f"  [SKIP] Cannot read: {src}")
        return

    result = clean_image(img)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), result)

    white_pct = (result > 240).mean() * 100
    dark_pct  = (result < 100).mean() * 100
    print(f"  [OK] {dst.name}  — white={white_pct:.1f}%  ink={dark_pct:.2f}%")


def process_batch(in_dir: Path, out_dir: Path) -> None:
    files = sorted(f for f in in_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXTS)
    if not files:
        print(f"No supported images found in {in_dir}")
        return

    print(f"Found {len(files)} image(s) in {in_dir}\n")
    for f in files:
        print(f"Processing: {f.name}")
        process_file(f, out_dir / (f.stem + "_clean.png"))

    print(f"\nDone. Outputs in: {out_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Physics Diagram Cleaner v3")
    parser.add_argument("--input",  default="input",  help="Input file or directory")
    parser.add_argument("--output", default="output", help="Output file or directory")
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)

    if src.is_file():
        # Single file mode
        out_path = dst if dst.suffix else dst / (src.stem + "_clean.png")
        print(f"Processing single file: {src}")
        process_file(src, out_path)
    elif src.is_dir():
        process_batch(src, dst)
    else:
        print(f"ERROR: Input path not found: {src}")
        sys.exit(1)


if __name__ == "__main__":
    main()
