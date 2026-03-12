import cv2
import numpy as np
import os
import glob
from tqdm import tqdm

def process_image(input_path, output_path):
    """
    Reads an image, extracts only the darkest (black) parts,
    and sets everything else (background and colored watermarks) to white.
    """
    # Read the image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Failed to read {input_path}")
        return False

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Binary Thresholding
    # Pixels with intensity < 100 will become black (0)
    # Pixels with intensity >= 100 will become white (255)
    # You can adjust the threshold value (100) if the black text appears too thin or thick.
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    
    cv2.imwrite(output_path, thresh)
    return True

def main():
    # Define input and output directories
    input_dir = "../samples/watermarked"
    output_dir = "../samples/black_only_clean"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all jpg/png files
    images = glob.glob(os.path.join(input_dir, "*.[jp][pn]*"))
    print(f"Found {len(images)} images to process.")
    
    success_count = 0
    for img_path in tqdm(images, desc="Filtering Images"):
        filename = os.path.basename(img_path)
        out_path = os.path.join(output_dir, filename)
        
        if process_image(img_path, out_path):
            success_count += 1
            
    print(f"\nProcessing complete! Successfully cleaned {success_count}/{len(images)} images.")
    print(f"Cleaned images are saved in: {output_dir}")

if __name__ == "__main__":
    main()
