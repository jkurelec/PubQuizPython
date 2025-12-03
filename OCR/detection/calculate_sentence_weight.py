import os
import cv2
import numpy as np


def calculate_sentence_weight(mask_dir="data/train/masks"):
    """
    Calculate the sentence weight for BCE loss by analyzing the mask images.

    :param mask_dir: Path to the directory containing mask images.
    :return: The calculated sentence weight.
    """
    total_pixels = 0
    total_foreground_pixels = 0

    # Loop through all mask files in the directory
    for file_name in os.listdir(mask_dir):
        file_path = os.path.join(mask_dir, file_name)
        if file_name.endswith(".png"):
            mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Error loading image: {file_path}")
                continue

            # Ensure mask is binary (0 and 255), normalize if needed
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            # Count pixels
            total_pixels += binary_mask.size
            total_foreground_pixels += np.sum(binary_mask > 0)

    # Calculate sentence weight
    if total_foreground_pixels == 0:
        raise ValueError("No foreground pixels found in the mask images.")

    sentence_weight = total_pixels / total_foreground_pixels
    print(sentence_weight)
    print(total_foreground_pixels / total_pixels)
    return sentence_weight


if __name__ == "__main__":
    calculate_sentence_weight()
