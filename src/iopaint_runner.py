"""IOPaint batch runner for watermark removal.

Calls `iopaint run` via subprocess to use the LaMa inpainting model.
Falls back to the existing histogram-LUT approach (remove_watermark.py)
for any images that IOPaint cannot process.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def run_iopaint_batch(
    image_dir: str,
    mask_dir:  str,
    output_dir: str,
    model: str   = "lama",
    device: str  = "cpu",
) -> list[str]:
    """Run IOPaint on all images in image_dir using masks from mask_dir.

    Only images that have a corresponding mask file will be processed.
    Returns list of filenames that failed (so caller can fall back).

    Args:
        image_dir:  Directory of source images.
        mask_dir:   Directory containing binary mask PNGs (same stem as images).
        output_dir: Where IOPaint puts the cleaned images.
        model:      IOPaint model name (default: lama).
        device:     Compute device — 'cpu', 'cuda', or 'mps' (Apple Silicon).

    Returns:
        List of image filenames that failed IOPaint processing.
    """
    image_path  = Path(image_dir)
    mask_path   = Path(mask_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check which images actually have masks (i.e., were detected as watermarked)
    exts   = {".jpg", ".jpeg", ".png"}
    images = sorted(f for f in image_path.iterdir() if f.suffix.lower() in exts)
    to_process = [img for img in images
                  if (mask_path / (img.stem + ".png")).exists()]

    if not to_process:
        print("  [INFO] No masked images found — nothing to inpaint.")
        return []

    print(f"  Running IOPaint ({model}) on {len(to_process)} images (device={device}) …")

    # IOPaint expects --image to point to a folder and --mask to a folder.
    # It matches files by filename (stem). We create a temp subfolder with only
    # the images that have masks, to avoid IOPaint erroring on missing masks.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_img_dir:
        tmp_img_path = Path(tmp_img_dir)
        for img in to_process:
            # Symlink or copy into tmp dir
            dest = tmp_img_path / img.name
            shutil.copy2(img, dest)

        cmd = [
            sys.executable, "-m", "iopaint", "run",
            f"--model={model}",
            f"--device={device}",
            f"--image={tmp_img_path}",
            f"--mask={mask_path}",
            f"--output={output_path}",
        ]

        try:
            result = subprocess.run(cmd, capture_output=False, text=True)
        except FileNotFoundError:
            # Try iopaint as a direct command
            cmd[0:2] = ["iopaint"]
            try:
                result = subprocess.run(cmd, capture_output=False, text=True)
            except FileNotFoundError:
                print("  [ERROR] IOPaint not found. Install with: pip install iopaint")
                return [img.name for img in to_process]

    if result.returncode != 0:
        print(f"  [WARN] IOPaint exited with code {result.returncode}")
        # Return all as failed so caller can use fallback
        return [img.name for img in to_process]

    # Determine which outputs were actually created
    failed: list[str] = []
    for img in to_process:
        # IOPaint may output as .png or same extension
        out_name_png = output_path / (img.stem + ".png")
        out_name_orig = output_path / img.name
        if not out_name_png.exists() and not out_name_orig.exists():
            failed.append(img.name)

    if failed:
        print(f"  [WARN] {len(failed)} images missing from IOPaint output → will use fallback")

    return failed


def check_iopaint_installed() -> bool:
    """Return True if iopaint package is importable."""
    try:
        import iopaint  # noqa: F401
        return True
    except ImportError:
        return False
