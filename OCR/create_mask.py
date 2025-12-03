import cv2
import numpy as np
import os
from glob import glob


def create_black_pixel_masks(buffer_px=5, threshold=200):
    """
    Creates binary masks for all grayscale images in 'image_prep/processed'
    where pixels darker than the threshold are highlighted,
    with a dilation buffer around them.

    Saves masks to the 'masks' folder (both folders are adjacent to the script).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(script_dir, "image_prep", "processed")
    output_folder = os.path.join(script_dir, "masks")
    os.makedirs(output_folder, exist_ok=True)

    image_paths = glob(os.path.join(input_folder, "*.jpg")) + glob(os.path.join(input_folder, "*.png"))

    if not image_paths:
        print(f"No images found in {input_folder}")
        return

    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Skipping {img_path}, could not read.")
            continue

        mask = (img < threshold).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (buffer_px * 2 + 1, buffer_px * 2 + 1))
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)

        # Save as PNG
        filename = os.path.splitext(os.path.basename(img_path))[0] + ".png"
        mask_path = os.path.join(output_folder, filename)
        cv2.imwrite(mask_path, dilated_mask)

        print(f"Mask saved: {mask_path}")


if __name__ == "__main__":
    create_black_pixel_masks(buffer_px=5, threshold=200)
