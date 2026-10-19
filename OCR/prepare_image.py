import os
import tempfile
import cv2
import shutil
import numpy as np
from PIL import Image

def preprocess_image(input_image, processed_folder, full_sheets_folder):
    # Create the output file path in the full sheets folder
    base = os.path.basename(input_image)
    processed_image_path = os.path.join(full_sheets_folder, f"{os.path.splitext(base)[0]}_processed.jpg")

    # Read and clean
    # cleaned = clean_image(input_image)
    cleaned = clean_small_image(input_image)

    # Resize + pad to square
    final_image = resize_and_pad(cleaned)

    # Save
    cv2.imwrite(processed_image_path, final_image)
    print(f"Processed image saved as: {processed_image_path}")

    # Move original to processed folder
    shutil.move(input_image, os.path.join(processed_folder, base))
    print(f"Moved original image to: {os.path.join(processed_folder, base)}")


def preprocess_single_image(img):
    """
    Preprocess a single image: save it temporarily to disk, clean, resize, and pad, then return it.
    """
    # If input is bytes, convert to temp file
    if isinstance(img, bytes) or isinstance(img, np.ndarray):
        # Convert NumPy array to bytes if needed
        if isinstance(img, np.ndarray):
            _, img_bytes = cv2.imencode(".png", img)
            img_bytes = img_bytes.tobytes()
        else:
            img_bytes = img

        # Create temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name

        # Pass temp path to clean_image
        cleaned = clean_small_image(tmp_path)

        # Remove temp file
        os.remove(tmp_path)
    elif isinstance(img, str):
        cleaned = clean_image(img)
    else:
        raise TypeError("Input must be a file path, bytes, or NumPy array")

    # Resize and pad
    final_image = resize_and_pad(cleaned)

    return final_image


def clean_small_image(input_image):
    image = cv2.imread(input_image)
    if image is None:
        print("Error loading:", input_image)
        return None

    # 1. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Very light denoise (preserve edges)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # 3. Soft contrast boost (NOT threshold)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # 4. DO NOT threshold — keep grayscale for handwriting
    return enhanced


def clean_image(input_image):
    image = cv2.imread(input_image)
    if image is None:
        print(f"Error: Could not load image {input_image}")
    else:
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1
        print(f"Image dimensions: {width}x{height}, Channels: {channels}")
    if image is None:
        print(f"Error: Could not load image {input_image}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.medianBlur(gray, 3)

    thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 21, 2)

    blurred = cv2.medianBlur(thresholded, 5)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(blurred)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    noise_removed = cv2.morphologyEx(contrast_enhanced, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(noise_removed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 100:
            cv2.drawContours(noise_removed, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    return cv2.medianBlur(noise_removed, 3)


def resize_and_pad(image, size=640):
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # create square canvas and paste resized image centered
    delta_w, delta_h = size - new_w, size - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0]  # black padding
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded


def clean_image2(input_image):
    # Load the image
    image = cv2.imread(input_image)

    if image is None:
        print(f"Error: Could not load image {input_image}")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Median Blur to reduce noise (preserves edges)
    blurred = cv2.medianBlur(gray, 3)

    # Apply adaptive thresholding to preserve details in handwritten text
    thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 7, 5)

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(thresholded)

    # Morphological opening to remove small noise (dots)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    noise_removed = cv2.morphologyEx(contrast_enhanced, cv2.MORPH_OPEN, kernel)

    # Contour detection to further remove small dots or unwanted elements
    contours, _ = cv2.findContours(noise_removed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 50:
            cv2.drawContours(noise_removed, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    # Optionally, apply another median blur to smooth any remaining artifacts after cleaning
    return cv2.medianBlur(noise_removed, 3)


def main():
    # Define the paths
    raw_sheets_folder = 'image_prep'
    todo_folder = os.path.join(raw_sheets_folder, 'todo')
    processed_folder = os.path.join(raw_sheets_folder, 'done')
    # full_sheets_folder = os.path.join(raw_sheets_folder, 'processedv2')
    full_sheets_folder = os.path.join(raw_sheets_folder, 'smallProcessed')

    os.makedirs(processed_folder, exist_ok=True)
    os.makedirs(full_sheets_folder, exist_ok=True)

    # Loop through the images and process them
    for filename in os.listdir(todo_folder):
        file_path = os.path.join(todo_folder, filename)  # Create the absolute path
        if os.path.isfile(file_path):  # Check if it's a file
            print(f"Processing: {file_path}")
            preprocess_image(file_path, processed_folder, full_sheets_folder)  # Pass the full path



def resize_images_in_folder(input_folder, output_folder, target_width=1080, target_height=1440):
    """
    Resize all images in input_folder to target dimensions using OpenCV,
    keeping aspect ratio and padding with black.
    """
    os.makedirs(output_folder, exist_ok=True)
    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(extensions):
            continue

        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            img = cv2.imread(input_path)
            if img is None:
                print(f"Failed to load {filename}")
                continue

            h, w = img.shape[:2]

            # Compute scaling factor
            scale_w = target_width / w
            scale_h = target_height / h
            scale = min(scale_w, scale_h)

            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Compute padding
            delta_w = target_width - new_w
            delta_h = target_height - new_h
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)

            # Pad with black
            padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])

            # Save
            cv2.imwrite(output_path, padded)
            print(f"Resized {filename} -> {target_width}x{target_height}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

def rotate_image(file_path, direction="right"):
    """
    Rotate an image in-place.

    :param file_path: path to the image
    :param direction: "right" (90° CW) or "left" (90° CCW)
    """
    # Load image
    img = cv2.imread(file_path)
    if img is None:
        print(f"Failed to load {file_path}")
        return

    # Rotate
    if direction.lower() == "right":
        rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif direction.lower() == "left":
        rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        print("Invalid direction. Use 'left' or 'right'.")
        return

    # Overwrite original file
    cv2.imwrite(file_path, rotated)
    print(f"Rotated {file_path} {direction}")

if __name__ == '__main__':
    # # rotate_image("image_prep/smallProcessed/IMG_20251021_172853510.jpg")  # rotate 90° CCW
    # resize_images_in_folder("image_prep/smallProcessed", "image_prep/todo")

    main()
