import os
import random
from PIL import Image, ImageDraw, ImageFont

# -------------------------------
# PARAMETERS
# -------------------------------
fonts_dir = "fonts"
output_dir = "images"
os.makedirs(output_dir, exist_ok=True)

charset = "abcčćdđefghijklmnopqrsštuvwxyzž0123456789-&' "
num_images_per_font = 1000
image_width = 512
image_height = 64
min_font_size = 28
max_font_size = 40


# -------------------------------
# HELPER FUNCTION
# -------------------------------
def generate_image(text, font_path):
    """Generate a simple greyscale image with full text visible."""
    font_size = random.randint(min_font_size, max_font_size)
    font = ImageFont.truetype(font_path, font_size)

    # Greyscale image
    img = Image.new("L", (image_width, image_height), color=255)
    draw = ImageDraw.Draw(img)

    # Get text size using textbbox
    bbox = draw.textbbox((0, 0), text, font=font)  # (x0, y0, x1, y1)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Center text horizontally and vertically
    # x_pos = max((image_width - text_width) // 2, 0)
    x_pos = 20
    y_pos = max((image_height - text_height) // 2, 0)

    draw.text((x_pos, y_pos), text, font=font, fill=0)
    return img


# -------------------------------
# MAIN GENERATION LOOP
# -------------------------------
def main():
    labels_file = "labels.txt"
    font_files = [f for f in os.listdir(fonts_dir) if f.lower().endswith(".ttf")]
    if not font_files:
        print("No .ttf fonts found in the fonts/ folder!")
        return

    total_images = 0

    # Open labels file
    with open(labels_file, "w", encoding="utf-8") as f_labels:
        for font_file in font_files:
            font_path = os.path.join(fonts_dir, font_file)
            for i in range(num_images_per_font):
                text_len = random.randint(1, 12)
                text = "".join(random.choice(charset) for _ in range(text_len))

                # Generate image
                img = generate_image(text, font_path)

                # Save image
                img_filename = f"{font_file[:-4]}_{i}.png"
                img.save(os.path.join(output_dir, img_filename))

                # Write label
                f_labels.write(f'{img_filename} "{text}"\n')

                total_images += 1
                print(f"Created {i} images")
    print(f"Done! Generated {total_images} images and wrote labels to {labels_file}")


if __name__ == "__main__":
    main()