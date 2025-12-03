import os
import random
import shutil

import cv2
import torchvision
from matplotlib import pyplot as plt
from ultralytics import YOLO
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib

matplotlib.use("TkAgg")

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# ===============================
# 1️⃣ Split Dataset into Train/Val
# ===============================
def split_yolo_dataset(images_dir="small_data/images", labels_dir="small_data/labels", output_dir="small_dataset_split", train_ratio=0.8):
    train_img_dir = os.path.join(output_dir, "train/images")
    val_img_dir = os.path.join(output_dir, "val/images")
    train_lbl_dir = os.path.join(output_dir, "train/labels")
    val_lbl_dir = os.path.join(output_dir, "val/labels")

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    images = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(images)
    train_count = int(len(images) * train_ratio)

    for i, img_file in enumerate(images):
        base_name = os.path.splitext(img_file)[0]
        lbl_file = base_name + ".txt"

        if i < train_count:
            shutil.copy(os.path.join(images_dir, img_file), train_img_dir)
            shutil.copy(os.path.join(labels_dir, lbl_file), train_lbl_dir)
        else:
            shutil.copy(os.path.join(images_dir, img_file), val_img_dir)
            shutil.copy(os.path.join(labels_dir, lbl_file), val_lbl_dir)

    print(f"Dataset split: {train_count} train / {len(images)-train_count} val")

# ===============================
# 2️⃣ Load Classes from classes.txt
# ===============================
def load_classes(classes_file="data/classes.txt"):
    with open(classes_file, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    print(f"Loaded classes: {class_names}")
    return class_names



def extract_yolo_detections(model_path="runs/yolo/weights/best.pt",classes_file="small_data/classes.txt",images_dir="further_data2",output_dir="small_detections_crops",conf_threshold=0.4,pad=0,device=0
):
    """
    Runs YOLO detection on all images in a folder and saves cropped detections by class.

    Args:
        model_path (str): Path to YOLO model weights (.pt)
        classes_file (str): Path to classes.txt containing class names (one per line)
        images_dir (str): Directory containing input images
        output_dir (str): Root output directory for cropped detections
        conf_threshold (float): Confidence threshold for detections
        pad (int): Optional pixel padding around bounding boxes
        device (int): GPU index (default 0) or 'cpu'
    """
    print(f"🔍 Starting extraction on device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Load class names
    with open(classes_file, "r") as f:
        class_names = [line.strip() for line in f.readlines()]

    # Load YOLO model
    model = YOLO(model_path)

    # Make sure output folder exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all image files
    image_files = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"Found {len(image_files)} images in '{images_dir}'")

    for img_name in image_files:
        img_path = os.path.join(images_dir, img_name)
        print(f"\n🖼️ Processing: {img_name}")

        # Run YOLO prediction
        results = model.predict(source=img_path, conf=conf_threshold, device=device, save=False)
        result = results[0]

        # Load original image
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Could not read {img_path}, skipping.")
            continue

        # Iterate detections
        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            # cls_id = int(box.cls)
            conf = float(box.conf)
            # label = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"

            # Add padding
            if pad > 0:
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(img.shape[1], x2 + pad)
                y2 = min(img.shape[0], y2 + pad)

            # Crop region
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Save inside class folder
            # class_dir = os.path.join(output_dir, label)
            # os.makedirs(class_dir, exist_ok=True)

            crop_filename = f"{os.path.splitext(img_name)[0]}_{i}_{conf:.2f}.jpg"
            # crop_path = os.path.join(class_dir, crop_filename)
            crop_path = os.path.join(output_dir, crop_filename)
            cv2.imwrite(crop_path, crop)

            print(f"  ✅ Saved: {crop_path}")

    print("\n🎯 Extraction complete!")
    print(f"All crops saved in: {os.path.abspath(output_dir)}")


def extract_sentence_detections(model, class_names, sentence_images, conf_threshold=0.8,device=0):
    results_list = []

    for idx, img in enumerate(sentence_images):
        if img.ndim == 2:  # grayscale
            img_bgr = cv2.cvtColor((img*255).astype('uint8'), cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = (img*255).astype('uint8')

        # Run YOLO prediction
        results = model.predict(source=img_bgr, conf=conf_threshold, device=device, save=False)
        result = results[0]

        # Keep track of Number and Answer/Multiple detections
        number_dets = []
        other_dets = []

        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            cls_id = int(box.cls)
            label = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"

            crop = img_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            if label.lower() == "number":
                number_dets.append(crop)
            elif label.lower() in ["answer", "multiple"]:
                other_dets.append(crop)

        # Only keep valid ones: 1 Number and 1+ Answer/Multiple
        if len(number_dets) == 1 and len(other_dets) >= 1:
            for det in other_dets:
                results_list.append((number_dets[0], det))

    print(f"✅ Total valid detections: {len(results_list)}")
    return results_list


# ===============================
# 4️⃣ Main Setup
# ===============================
if __name__ == "__main__":
    print(torch.cuda.is_available())
    images_dir = "small_data/images"
    labels_dir = "small_data/labels"
    extract_yolo_detections()
    classes_file = "small_data/classes.txt"
    output_dir = "small_dataset_split"
    # #
    # # # Split dataset
    # split_yolo_dataset(images_dir, labels_dir, output_dir=output_dir, train_ratio=0.8)
    #
    # # Load class names
    class_names = load_classes(classes_file)
    num_classes = len(class_names)

    # Create YOLOv8 dataset YAML
    yolo_yaml_path = "yolo_dataset.yaml"
#     yaml_content = f"""train: {output_dir}/train/images
# val: {output_dir}/val/images
#
# nc: {num_classes}
# names: {class_names}
# """
#     with open(yolo_yaml_path, "w") as f:
#         f.write(yaml_content)
#     print(f"YOLO dataset YAML saved to {yolo_yaml_path}")

    # ===============================
    # 5️⃣ Train YOLOv8 Model
    # ===============================
    model = YOLO("runs/yolo/weights/best.pt")
    # model = YOLO("yolo11n.pt")

    model.train(
        data=yolo_yaml_path,
        epochs=50,
        imgsz=512,
        batch=64,
        project="runs",
        name="yolo",
        exist_ok=True,
        device=0
    )

    print("YOLOv8 training finished!")

    img_path = "small_dataset_split/val/images/0e9bc8f9-IMG_20251021_173048681_processed_3.png"
    result = model.predict(source=img_path, conf=0.6, save=False)

    # YOLOv8 returns results[0].plot() as a numpy array with boxes drawn
    img_with_boxes = result[0].plot()

    plt.figure(figsize=(10, 10))
    plt.imshow(img_with_boxes)
    plt.axis('off')
    plt.show()