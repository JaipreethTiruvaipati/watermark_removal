import cv2
import numpy as np
import glob
import os

def extract_watermark(image_dir, output_path, num_images=50):
    """
    Extracts a static watermark from a set of images by taking the median.
    Because the underlying image content changes but the watermark stays the same,
    the median of many images will reveal just the watermark (and a dull gray background).
    """
    image_paths = glob.glob(os.path.join(image_dir, "*.[jp][pn]*"))
    # Sort or shuffle to get a good mix
    image_paths = image_paths[:num_images]
    
    if not image_paths:
        print("No images found.")
        return

    print(f"Reading {len(image_paths)} images to extract watermark...")
    
    # Read first image to get dimensions
    first_img = cv2.imread(image_paths[0])
    h, w, c = first_img.shape
    
    # Pre-allocate array for all images
    # Use a list to save memory before stacking
    images = []
    
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            # Resize if necessary to match the first image strictly
            if img.shape != (h, w, c):
                img = cv2.resize(img, (w, h))
            images.append(img)
            
    print("Computing median. This may take a minute...")
    # Stack images along a new axis
    image_stack = np.stack(images, axis=0)
    
    # Calculate median along the image axis
    median_img = np.median(image_stack, axis=0).astype(np.uint8)
    
    cv2.imwrite(output_path, median_img)
    print(f"Watermark template saved to: {output_path}")

if __name__ == "__main__":
    extract_watermark("samples/watermarked", "samples/extracted_watermark_template.png", num_images=60)
