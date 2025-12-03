import os
import cv2
import numpy as np
import re



# def deskew_image(img, debug=False):
#     # Convert to grayscale if needed
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
#
#     # Invert since background is white
#     inverted = cv2.bitwise_not(gray)
#
#     # Threshold to binary
#     _, binary = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
#     # Find coordinates of foreground pixels
#     coords = np.column_stack(np.where(binary > 0))
#     if len(coords) < 10:
#         return img  # skip empty images
#
#     # Compute PCA to find dominant angle
#     mean, eigenvectors, _ = cv2.PCACompute2(coords.astype(np.float32), mean=np.array([]))
#     angle = np.arctan2(eigenvectors[0,1], eigenvectors[0,0]) * 180 / np.pi
#
#     # Constrain to reasonable range
#     if abs(angle) > 45:
#         angle = 0
#
#     if debug:
#         print(f"Detected angle: {angle:.2f}°")
#
#     # Rotate the image around its center
#     (h, w) = img.shape[:2]
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, angle, 1.0)
#     return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
#
#
# def show_before_after(path):
#     img = cv2.imread(path)
#     fixed = deskew_image(img)
#
#     # Convert BGR to RGB for matplotlib
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     fixed_rgb = cv2.cvtColor(fixed, cv2.COLOR_BGR2RGB)
#
#     plt.figure(figsize=(10, 4))
#     plt.subplot(1, 2, 1)
#     plt.title('Original')
#     plt.imshow(img_rgb)
#     plt.axis('off')
#
#     plt.subplot(1, 2, 2)
#     plt.title('Deskewed')
#     plt.imshow(fixed_rgb)
#     plt.axis('off')
#     plt.show()


def resize_with_aspect(img, target_height=64, target_width=512):
    h, w = img.shape[:2]
    scale = target_height / h if h > target_height else 1.0
    new_h = int(h * scale)
    new_w = int(w * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_top = (target_height - new_h) // 2
    pad_bottom = target_height - new_h - pad_top
    pad_left = 0
    pad_right = max(0, target_width - new_w)

    img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                             cv2.BORDER_CONSTANT, value=255)
    if img.shape[1] > target_width:
        img = img[:, :target_width]
    return img


def preprocess_and_save_images_as_png(input_dir, output_dir, labels_path=None,
                                      target_height=64, target_width=512):
    os.makedirs(output_dir, exist_ok=True)

    # Get and sort images by creation time
    image_paths = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(('.jpg', '.png'))
    ]
    image_paths.sort(key=os.path.getctime)

    # Open labels file once
    label_file = open(labels_path, "w", encoding="utf-8") if labels_path else None

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: cannot read {img_path}")
            continue

        img = resize_with_aspect(img, target_height, target_width)
        img_uint8 = (cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, img_uint8)

        # Write label line immediately
        if label_file:
            label_file.write(f'{filename} ""\n')

        print(f"Processed and wrote: {filename}")

    if label_file:
        label_file.close()
        print(f"Labels saved to {labels_path}")

    print(f"Preprocessed images saved as PNG to {output_dir}")


def preprocess_tuple_images(results_list, target_height=64, target_width=512):
    """
    Resize and normalize images in a list of (Number_img, Answer_img) tuples.

    Args:
        results_list (list): List of tuples [(Number_img, Answer_img), ...] as NumPy arrays.
        target_height (int): Target height after resizing.
        target_width (int): Target width after resizing.

    Returns:
        list: List of preprocessed tuples [(Number_img_proc, Answer_img_proc), ...]
    """
    processed_list = []
    for number_img, answer_img in results_list:
        # Ensure grayscale
        if number_img.ndim == 3:
            number_img = cv2.cvtColor(number_img, cv2.COLOR_BGR2GRAY)
        if answer_img.ndim == 3:
            answer_img = cv2.cvtColor(answer_img, cv2.COLOR_BGR2GRAY)

        # Resize with aspect ratio
        number_proc = resize_with_aspect(number_img, target_height, target_width)
        answer_proc = resize_with_aspect(answer_img, target_height, target_width)

        # Normalize to [0,1] float32
        number_proc = number_proc.astype(np.float32) / 255.0
        answer_proc = answer_proc.astype(np.float32) / 255.0

        processed_list.append((number_proc, answer_proc))

    return processed_list


if __name__ == "__main__":
    input_folder = 'further_data'
    images_output_folder = os.path.join("small_dataset", "images")
    labels_file = os.path.join("small_dataset", "labels.txt")

    preprocess_and_save_images_as_png(input_folder, images_output_folder, labels_file)

