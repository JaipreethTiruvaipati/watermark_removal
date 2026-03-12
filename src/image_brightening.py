import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
from run_pipeline import make_mask

def process_image(input_path, output_path):
    img = cv2.imread(input_path)
    if img is None:
        return False
        
    mask = make_mask(img)
    
    if np.count_nonzero(mask) > 0:
        # Approach 3: Subtracting the watermark / Brightening the masked area
        # Convert mask to 3 channels
        mask_3c = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Where mask is white, we artificially increase brightness
        # This assumes the watermark darkened the original image.
        # We add a flat value (e.g., 50) to the pixels under the mask.
        brighten_factor = 70
        
        # Create an image that is all zeros, except where the mask is, it is `brighten_factor`
        brighten_layer = np.zeros_like(img, dtype=np.uint16)
        brighten_layer[mask > 0] = brighten_factor
        
        # Add to original image and clip to 255
        cleaned16 = img.astype(np.uint16) + brighten_layer
        cleaned = np.clip(cleaned16, 0, 255).astype(np.uint8)
        
    else:
        cleaned = img
        
    cv2.imwrite(output_path, cleaned)
    return True

def main():
    input_dir = "samples/watermarked"
    output_dir = "samples/brighten_cleaned"
    
    os.makedirs(output_dir, exist_ok=True)
    images = glob.glob(os.path.join(input_dir, "*.[jp][pn]*")) # Process all
    
    for img_path in tqdm(images, desc="Image Brightening"):
        filename = os.path.basename(img_path)
        out_path = os.path.join(output_dir, filename)
        process_image(img_path, out_path)

if __name__ == "__main__":
    main()
