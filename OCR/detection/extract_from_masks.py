import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


def extract_individual_sentence_regions(image, sentence_mask, threshold=0.8):
    """
    Function to extract individual sentence regions from the original grayscale image
    based on the sentence mask. Each detected region is saved as an individual image.

    :param image: The original input image tensor (CxHxW or HxW).
    :param sentence_mask: The sentence mask tensor (HxW).
    :param threshold: The confidence threshold for the mask (default: 0.5).

    :return: A list of individual sentence images extracted from the grayscale image.
    """
    # Check the shape of the image tensor
    print("Image shape:", image.shape)

    # Convert image to grayscale based on the number of channels
    if len(image.shape) == 4 and image.shape[1] == 1:  # Image is (batch_size, C=1, H, W)
        grayscale_image = image[0, 0].cpu().numpy()  # Extract HxW (ignore batch and channel)
    elif len(image.shape) == 3:  # Image is (C, H, W)
        if image.shape[0] == 1:  # Single channel grayscale image
            grayscale_image = image.cpu().numpy()[0]  # HxW (ignore channel dimension)
        else:
            image_np = image.cpu().numpy().transpose(1, 2, 0)  # HxWxC
            grayscale_image = np.mean(image_np, axis=2)  # Convert to grayscale
    elif len(image.shape) == 2:  # If image is already (H, W)
        grayscale_image = image.cpu().numpy()  # HxW
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    # Clip values to be in the range [0, 1] for display
    grayscale_image = np.clip(grayscale_image, 0, 1)

    # Move sentence mask to CPU and convert to NumPy
    sentence_mask_np = sentence_mask.cpu().numpy()

    # Check if the sentence mask is valid
    if sentence_mask_np is None or sentence_mask_np.size == 0:
        raise ValueError("The sentence mask is empty or None.")

    # Apply the confidence threshold to the sentence mask
    mask_thresholded = np.where(sentence_mask_np >= threshold, 255, 0).astype(np.uint8)  # Binary mask (0 for low, 255 for high)

    # Check the shape and type of the thresholded mask
    print("Mask thresholded shape before squeeze:", mask_thresholded.shape)
    print("Unique values in mask_thresholded:", np.unique(mask_thresholded))

    # Ensure the mask is a 2D array by squeezing it if it has a singleton dimension
    if mask_thresholded.ndim == 3:
        mask_thresholded = mask_thresholded[0]  # Remove the batch dimension

    # Ensure it is indeed 2D
    if mask_thresholded.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {mask_thresholded.ndim}D mask.")

    # Find contours (connected regions) in the thresholded mask
    contours, _ = cv2.findContours(mask_thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract individual sentence images
    sentence_images = []
    for contour in contours:
        # Get bounding box around each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Ensure the bounding box is within image dimensions
        if x < 0 or y < 0 or x + w > grayscale_image.shape[1] or y + h > grayscale_image.shape[0]:
            print(f"Warning: Bounding box out of image bounds: x={x}, y={y}, w={w}, h={h}")
            continue

        if w > 400:
            # Crop the corresponding region from the grayscale image
            sentence_image = grayscale_image[y:y + h, x:x + w]

            # Append the cropped sentence image to the list
            sentence_images.append(sentence_image)

    return sentence_images


def extract_individual_sentence_regions2(image, sentence_mask, threshold=0.9):
    """
    Extract individual sentence regions from the original grayscale image
    based on the sentence mask. Each region is extracted and non-relevant
    parts are filled with white pixels to form a rectangle.

    :param image: The original input image tensor (CxHxW or HxW).
    :param sentence_mask: The sentence mask tensor (HxW or 1xHxW).
    :param threshold: The confidence threshold for the mask (default: 0.8).

    :return: A list of individual sentence images extracted from the grayscale image.
    """
    # Convert image to grayscale (if needed)
    if len(image.shape) == 4 and image.shape[1] == 1:  # (batch_size, C=1, H, W)
        grayscale_image = image[0, 0].detach().cpu().numpy()  # HxW
    elif len(image.shape) == 3:  # (C, H, W)
        if image.shape[0] == 1:  # Single-channel grayscale
            grayscale_image = image[0].detach().cpu().numpy()  # HxW
        else:  # Convert multi-channel to grayscale
            image_np = image.detach().cpu().numpy().transpose(1, 2, 0)  # HxWxC
            grayscale_image = np.mean(image_np, axis=2)  # HxW
    elif len(image.shape) == 2:  # Already HxW
        grayscale_image = image.detach().cpu().numpy()
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    grayscale_image = np.clip(grayscale_image, 0, 1)  # Ensure valid range

    # Squeeze the mask to remove singleton dimensions, if any
    sentence_mask = sentence_mask.squeeze()

    # Convert mask to binary using the threshold
    mask_np = (sentence_mask.detach().cpu().numpy() >= threshold).astype(np.uint8) * 255

    # image_np = image.squeeze().detach().cpu().numpy()
    # plt.figure(figsize=(8, 6))
    # plt.imshow(image_np, cmap='gray')
    # plt.title("Sentence Mask")
    # plt.axis('off')
    # plt.show()
    # plt.figure(figsize=(8, 6))
    # plt.imshow(mask_np, cmap='gray')
    # plt.title("Sentence Mask")
    # plt.axis('off')
    # plt.show()

    # Debugging: Check dimensions and values of mask_np
    if mask_np.ndim != 2:
        raise ValueError(f"mask_np must be 2D, but got shape: {mask_np.shape}")
    if not np.any(mask_np):
        raise ValueError("mask_np is empty (all zeros). Check sentence_mask or threshold.")

    # Find contours for connected regions
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # List to hold sentence images
    sentence_images = []

    for contour in contours:
        # Get bounding box of the region
        x, y, w, h = cv2.boundingRect(contour)

        if w > 250:
            # Extract the mask for the specific region
            region_mask = np.zeros_like(mask_np)
            cv2.drawContours(region_mask, [contour], -1, 255, thickness=cv2.FILLED)

            # Apply the region mask to the grayscale image
            cropped_region = grayscale_image[y:y + h, x:x + w]  # Region from grayscale image
            cropped_mask = region_mask[y:y + h, x:x + w]  # Region from mask

            # Create an output image with white background
            output_image = np.ones_like(cropped_region, dtype=np.float32)

            # Place the extracted region into the white background
            output_image[cropped_mask == 255] = cropped_region[cropped_mask == 255]

            # Append the processed region to the list
            sentence_images.append(output_image)

            plt.figure(figsize=(6, 2))
            plt.imshow(output_image, cmap='gray')
            plt.title(f"Sentence {+1}")
            plt.axis('off')
            plt.show()

    return sentence_images


def main(image, sentence_output):
    # Assuming `image` is your input image tensor and `sentence_output` is the mask
    individual_sentences = extract_individual_sentence_regions2(image, sentence_output)

    output_folder = "sentence_regions"
    os.makedirs(output_folder, exist_ok=True)

    # Save each extracted sentence region to the output folder
    for i, sentence_img in enumerate(individual_sentences):
        plt.figure()
        plt.imshow(sentence_img, cmap='gray')
        plt.axis('off')
        plt.title(f'Sentence Region {i + 1}')
        output_path = os.path.join(output_folder, f"sentence_region_{i + 1}.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()  # Close the plot to free up memory
        print(f"Saved Sentence Region {i + 1} to {output_path}")


def extract_images_from_mask_for_ocr(image, sentence_output, filename):
    individual_sentences = extract_individual_sentence_regions2(image, sentence_output)
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    output_folder = os.path.join(parent_dir, "classifier", "further_data2")
    os.makedirs(output_folder, exist_ok=True)

    for i, sentence_img in enumerate(individual_sentences):
        if sentence_img.max() <= 1.0:
            sentence_img = (sentence_img * 255).astype(np.uint8)
        output_path = os.path.join(output_folder, f"{filename}_{i + 1}.png")
        cv2.imwrite(output_path, sentence_img)


if __name__ == '__main__':
    main()
