import cv2
import numpy as np
import os
import glob

def process_image(input_path, output_path):
    # Read the image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Failed to read {input_path}")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Simple Binary Thresholding
    # Black is low value (e.g., < 100), others are higher.
    # We will set anything above 120 to white (255) and anything below to black (0)
    # Using THRESH_BINARY
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    
    cv2.imwrite(output_path, thresh)

def main():
    input_dir = "../samples/watermarked"
    output_dir = "../samples/black_only_output"
    
    os.makedirs(output_dir, exist_ok=True)
    
    images = glob.glob(os.path.join(input_dir, "*.jpg"))[:5] # Test on first 5
    for img_path in images:
        filename = os.path.basename(img_path)
        out_path = os.path.join(output_dir, filename)
        process_image(img_path, out_path)
        print(f"Processed {filename}")

if __name__ == "__main__":
    main()
