import os
import random
from PIL import Image


def augment_rotate_images(
        input_dir="data/trainSmall2",
        output_dir="data/trainSmall",
        image_ext=".jpg",
        mask_ext=".png",
        num_rotations=5
):
    """
    Rotates grayscale images and corresponding masks by random angles.

    Args:
        input_dir (str): Path to input data containing 'images' and 'masks' subfolders.
        output_dir (str): Output directory for augmented data.
        image_ext (str): Extension for input images.
        mask_ext (str): Extension for mask images.
        num_rotations (int): Number of rotated versions to create per image.
    """
    img_input_dir = os.path.join(input_dir, "images")
    mask_input_dir = os.path.join(input_dir, "masks")
    img_output_dir = os.path.join(output_dir, "images")
    mask_output_dir = os.path.join(output_dir, "masks")

    os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(mask_output_dir, exist_ok=True)

    for filename in os.listdir(img_input_dir):
        if not filename.lower().endswith(image_ext):
            continue

        img_path = os.path.join(img_input_dir, filename)
        mask_path = os.path.join(mask_input_dir, os.path.splitext(filename)[0] + mask_ext)

        if not os.path.exists(mask_path):
            print(f"⚠️ No mask found for {filename}, skipping.")
            continue

        img = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        for i in range(num_rotations):
            angle = random.uniform(-10, 10)

            # Rotate both with black fill
            rotated_img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
            rotated_mask = mask.rotate(angle, resample=Image.NEAREST, fillcolor=0)

            # Save with descriptive names
            base_name = os.path.splitext(filename)[0]
            new_img_name = f"{base_name}_rot{int(angle)}_{i}{image_ext}"
            new_mask_name = f"{base_name}_rot{int(angle)}_{i}{mask_ext}"

            rotated_img.save(os.path.join(img_output_dir, new_img_name))
            rotated_mask.save(os.path.join(mask_output_dir, new_mask_name))

            print(f"✅ Saved rotated pair: {new_img_name} (angle {angle:.1f}°)")


if __name__ == "__main__":
    augment_rotate_images(num_rotations=30)
