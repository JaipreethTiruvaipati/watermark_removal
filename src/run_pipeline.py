"""
Watermark Removal Pipeline — jaipreeth_tiruvaipati output
=========================================================

Uses IOPaint's LaMa inpainting model directly via Python API.
Detects blue/grey/red watermarks automatically and inpaints them.

Usage:
    cd watermark-removal/
    python src/run_pipeline.py --input samples/watermarked --output jaipreeth_tiruvaipati --clean-input samples/clean

Options:
    --input         Directory of watermarked images
    --output        Output directory (will be created)
    --clean-input   Directory of clean images (copied unchanged)
    --device        cpu / cuda / mps  (default: cpu)
    --limit         Process only first N images (for testing)
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

# ─── Ensure src/ on path ──────────────────────────────────────────────────────
_SRC = Path(__file__).parent
sys.path.insert(0, str(_SRC))

from remove_watermark import remove_watermark as _cv_fallback


# ─────────────────────────────────────────────────────────────────────────────
# 1.  WATERMARK MASK GENERATION (multi-color-space)
# ─────────────────────────────────────────────────────────────────────────────

def make_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Return a binary uint8 mask (255 = watermark, 0 = content).

    Detects:
    - Blue watermarks  — low b* in CIE-LAB space
    - Grey watermarks  — neutral colour + mid-luminance in LAB
    - Red watermarks   — hue near 0/180 in HSV
    - Yellowish tint patches — warm-tone low-contrast patches
    """
    h, w = img_bgr.shape[:2]
    total = h * w

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    # ── Blue mask ─────────────────────────────────────────────────────────────
    # In OpenCV LAB: b* neutral = 128. Values < 118 → blue shift.
    blue = (b.astype(np.int16) < 120) & (L > 90) & (L < 240)

    # ── Grey mask ─────────────────────────────────────────────────────────────
    # Very close to neutral on both a* and b* axes, mid luminance.
    a_neut = np.abs(a.astype(np.int16) - 128) < 8
    b_neut = np.abs(b.astype(np.int16) - 128) < 8
    grey = a_neut & b_neut & (L > 70) & (L < 230)

    # ── Red / maroon mask (HSV) ────────────────────────────────────────────────
    m1 = cv2.inRange(hsv, np.array([0,   50,  50]), np.array([12, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([168, 50,  50]), np.array([180,255, 255]))
    red = (m1 | m2).astype(bool)

    # ── Yellowish tint patches (warm semi-transparent box watermarks) ──────────
    # a* slightly positive (a* > 130), b* slightly positive (b* > 130), mid-L
    warm = (a.astype(np.int16) > 130) & (b.astype(np.int16) > 130) & (L > 100) & (L < 230)

    combined = (blue | grey | red | warm).astype(np.uint8) * 255

    # Remove noise: opening to eliminate isolated pixels
    k3 = np.ones((3, 3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k3)

    # Dilate to cover anti-aliased edges
    k7 = np.ones((7, 7), np.uint8)
    combined = cv2.dilate(combined, k7, iterations=3)

    coverage = np.count_nonzero(combined) / total
    # If coverage is suspiciously small or almost everything, not a watermark
    if coverage < 0.003 or coverage > 0.88:
        return np.zeros((h, w), dtype=np.uint8)

    return combined


def has_watermark(mask: np.ndarray) -> bool:
    return np.count_nonzero(mask) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 2.  IOPAINT INPAINTING  (Python API — no subprocess)
# ─────────────────────────────────────────────────────────────────────────────

def _load_lama(device: str):
    """Load LaMa model via IOPaint's Python API. Returns model or None."""
    try:
        from iopaint.model_manager import ModelManager
        from iopaint.schema import InpaintRequest, HDStrategy

        class _LaMaWrapper:
            def __init__(self):
                # ModelManager(name, device) — confirmed API
                self.manager = ModelManager(name="lama", device=device)
                self._req    = InpaintRequest(hd_strategy=HDStrategy.CROP)

            def inpaint(self, img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
                # manager(image, mask, config) — confirmed API
                result = self.manager(img_bgr, mask, self._req)
                return result

        return _LaMaWrapper()
    except Exception as e:
        print(f"  [WARN] Could not load LaMa model: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PER-IMAGE PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def process_image(
    img_path: Path,
    output_dir: Path,
    lama_model,         # LaMa wrapper or None
) -> str:
    """Process one image. Returns 'iopaint' | 'fallback' | 'clean' | 'error'."""
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"  [ERROR] Cannot read {img_path.name}")
        return "error"

    mask = make_mask(img_bgr)
    out_stem = output_dir / (img_path.stem + ".png")

    if not has_watermark(mask):
        # No watermark detected — copy as-is
        shutil.copy2(img_path, output_dir / img_path.name)
        return "clean"

    # ── Try LaMa inpainting ──────────────────────────────────────────────────
    if lama_model is not None:
        try:
            result = lama_model.inpaint(img_bgr, mask)
            cv2.imwrite(str(out_stem), result)
            return "iopaint"
        except Exception as e:
            print(f"  [WARN] LaMa failed on {img_path.name}: {e} — using fallback")

    # ── Fallback: histogram-LUT → grayscale ─────────────────────────────────
    cleaned_bytes = _cv_fallback(str(img_path))
    if cleaned_bytes:
        out_stem.write_bytes(cleaned_bytes)
        return "fallback"

    # Last resort: copy original
    shutil.copy2(img_path, output_dir / img_path.name)
    return "error"


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PIPELINE ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    input_dir: str,
    output_dir: str,
    clean_dir: str | None = None,
    device: str = "cpu",
    limit: int | None = None,
) -> None:
    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    exts   = {".jpg", ".jpeg", ".png"}
    images = sorted(f for f in input_path.iterdir() if f.suffix.lower() in exts)[:limit]

    print(f"\n{'='*62}")
    print(f"  Watermark Removal Pipeline  →  {output_dir}")
    print(f"  Watermarked images : {len(images)}")
    print(f"  Device             : {device}")
    print(f"{'='*62}\n")

    # ── Load LaMa once ────────────────────────────────────────────────────────
    print("[ Loading LaMa model … ]")
    lama = _load_lama(device)
    if lama:
        print("  LaMa model loaded ✓\n")
    else:
        print("  LaMa unavailable — will use histogram-LUT fallback for all images\n")

    # ── Process watermarked images ────────────────────────────────────────────
    counts = {"iopaint": 0, "fallback": 0, "clean": 0, "error": 0}

    for i, img_path in enumerate(images):
        status = process_image(img_path, output_path, lama)
        counts[status] += 1

        label = {"iopaint": "LaMa ✓", "fallback": "fallback ✓", "clean": "clean (skip)", "error": "ERROR"}.get(status, status)
        print(f"  [{i+1:>3}/{len(images)}] {img_path.name:<18}  →  {label}")

    # ── Copy clean images from separate folder ────────────────────────────────
    clean_copied = 0
    if clean_dir and Path(clean_dir).exists():
        clean_imgs = sorted(f for f in Path(clean_dir).iterdir() if f.suffix.lower() in exts)
        for img in clean_imgs:
            shutil.copy2(img, output_path / img.name)
        clean_copied = len(clean_imgs)
        print(f"\n  Copied {clean_copied} clean images from {clean_dir}")

    # ── Summary ──────────────────────────────────────────────────────────────
    total_out = sum(1 for f in output_path.iterdir() if f.suffix.lower() in exts | {".png"})
    print(f"\n{'='*62}")
    print(f"  DONE")
    print(f"  LaMa inpainting   : {counts['iopaint']}")
    print(f"  Fallback (LUT)    : {counts['fallback']}")
    print(f"  Clean (no WM)     : {counts['clean']}")
    print(f"  Clean dir copied  : {clean_copied}")
    print(f"  Errors            : {counts['error']}")
    print(f"  Total output      : {total_out} files  →  {output_path.resolve()}")
    print(f"{'='*62}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Watermark removal pipeline using LaMa inpainting")
    p.add_argument("--input",       required=True)
    p.add_argument("--output",      required=True)
    p.add_argument("--clean-input", default=None)
    p.add_argument("--device",      default="cpu", help="cpu | cuda | mps")
    p.add_argument("--limit",       type=int, default=None, help="Process only first N images")
    return p.parse_args()


if __name__ == "__main__":
    a = _args()
    run_pipeline(
        input_dir = a.input,
        output_dir= a.output,
        clean_dir = a.clean_input,
        device    = a.device,
        limit     = a.limit,
    )
