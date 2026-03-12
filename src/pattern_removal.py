import cv2
import numpy as np
import glob
import os
from tqdm import tqdm

def process_image(img_path, template_img, output_path):
    img = cv2.imread(img_path)
    if img is None:
        return False
    
    # Resize image to match template if needed
    if img.shape != template_img.shape:
        img = cv2.resize(img, (template_img.shape[1], template_img.shape[0]))
    
    # Method: Division
    # Watermarks are often applied by multiplying a watermark image (where background is white=255)
    # with the original image. To reverse this, we divide the watermarked image by the watermark template.
    # We use cv2.divide with a scale of 255.
    
    # We must avoid divide by zero, though template mostly shouldn't be zero.
    # cv2.divide handles this automatically.
    cleaned = cv2.divide(img, template_img, scale=255)
    
    cv2.imwrite(output_path, cleaned)
    return True

def main():
    input_dir = "samples/watermarked"
    output_dir = "samples/pattern_cleaned"
    template_path = "samples/extracted_watermark_template.png"
    
    os.makedirs(output_dir, exist_ok=True)
    
    template_img = cv2.imread(template_path)
    if template_img is None:
        print("Template not found!")
        return
        
    images = glob.glob(os.path.join(input_dir, "*.[jp][pn]*"))[:10] # Test on 10
    
    for img_path in tqdm(images, desc="Pattern Matching (Division)"):
        filename = os.path.basename(img_path)
        out_path = os.path.join(output_dir, filename)
        process_image(img_path, template_img, out_path)

if __name__ == "__main__":
    main()
