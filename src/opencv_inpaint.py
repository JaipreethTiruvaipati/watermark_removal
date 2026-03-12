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
        # Use OpenCV Telea Inpainting
        # Radius = 3
        cleaned = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    else:
        cleaned = img
        
    cv2.imwrite(output_path, cleaned)
    return True

def main():
    input_dir = "samples/watermarked"
    output_dir = "samples/opencv_inpaint_cleaned"
    
    os.makedirs(output_dir, exist_ok=True)
    images = glob.glob(os.path.join(input_dir, "*.[jp][pn]*")) # Process all
    
    for img_path in tqdm(images, desc="OpenCV Inpainting"):
        filename = os.path.basename(img_path)
        out_path = os.path.join(output_dir, filename)
        process_image(img_path, out_path)

if __name__ == "__main__":
    main()
