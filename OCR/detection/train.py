import json
import os
import random
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import torch.optim as optim
from model import YOLOOBBDataset, OBBDetector, OBBCriterion
matplotlib.use('TkAgg')


# Assuming you already have this dataset
dataset = YOLOOBBDataset("data", img_size=640)
loader = DataLoader(dataset, batch_size=4, shuffle=True)


def draw_obb(img, poly):
    """
    Draws oriented bounding box (OBB) from normalized coordinates.
    poly: [cls, x1, y1, x2, y2, x3, y3, x4, y4]
    """
    cls = int(poly[0])
    color = (0, 0, 255) if cls == 0 else (255, 0, 0)  # BGR: red for 0, blue for 1

    h, w = img.shape[:2]
    pts = np.array([
        [poly[1]*w, poly[2]*h],
        [poly[3]*w, poly[4]*h],
        [poly[5]*w, poly[6]*h],
        [poly[7]*w, poly[8]*h]
    ], dtype=np.int32)

    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
    return img



def show_random_obb_image(dataset_path="data"):
    images_dir = os.path.join(dataset_path, "images")
    labels_dir = os.path.join(dataset_path, "labels")

    # Pick a random image
    image_name = random.choice(os.listdir(images_dir))
    image_path = os.path.join(images_dir, image_name)
    label_path = os.path.join(labels_dir, os.path.splitext(image_name)[0] + ".txt")

    # Load grayscale image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not read {image_path}")
        return

    h, w = img.shape[:2]
    assert h == 640 and w == 640, f"Expected 640x640, got {w}x{h}"

    # Convert to RGB so we can draw colored boxes
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Load label file
    if not os.path.exists(label_path):
        print(f"No label file for {image_name}")
        return

    with open(label_path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) != 9:
            print(f"Skipping malformed line: {line}")
            continue

        cls = int(parts[0])
        coords = list(map(float, parts[1:]))

        # Convert normalized (0–1) to pixel coordinates
        pts = np.array([
            [coords[0] * w, coords[1] * h],
            [coords[2] * w, coords[3] * h],
            [coords[4] * w, coords[5] * h],
            [coords[6] * w, coords[7] * h],
        ], dtype=np.int32).reshape((-1, 1, 2))

        # Class color mapping
        color = (255, 0, 0) if cls == 0 else (0, 0, 255)  # red=class0, blue=class1

        cv2.polylines(img_rgb, [pts], isClosed=True, color=color, thickness=2)

    # Display using matplotlib
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
    plt.title(f"Random image: {image_name}")
    plt.axis("off")
    plt.show()


def get_random_prediction_image(model, dataset, device='cuda', best_model_path="best_model.pth"):
    model = model.to(device)

    # Load best model if exists
    if os.path.exists(best_model_path):
        try:
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            print(f"Loaded best model from '{best_model_path}'")
        except Exception as e:
            print(f"Failed to load best model: {e}")

    model.eval()

    # Pick random sample
    idx = random.randint(0, len(dataset) - 1)
    img_tensor, target = dataset[idx]
    img_batch = img_tensor.unsqueeze(0).to(device)  # [1,1,H,W]

    with torch.no_grad():
        pred_boxes, pred_conf, pred_cls = model(img_batch)  # [B,8,H,W], [B,1,H,W], [B,num_classes,H,W]
        B = pred_boxes.shape[0]

        # Flatten spatial dims
        pred_boxes = pred_boxes.view(B, 8, -1).permute(0, 2, 1)[0]  # [num_preds, 8]
        pred_conf  = pred_conf.view(B, -1)[0]                         # [num_preds]
        pred_cls   = pred_cls.view(B, pred_cls.shape[1], -1).permute(0, 2, 1)[0]  # [num_preds, num_classes]

        # Apply sigmoid
        pred_boxes = torch.sigmoid(pred_boxes)
        pred_conf  = torch.sigmoid(pred_conf)
        pred_cls   = torch.sigmoid(pred_cls)

        # Convert image for visualization
        img_np = (img_tensor.squeeze(0).cpu().numpy() * 255).astype('uint8')
        img_color = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        h, w = img_color.shape[:2]

        # Convert targets and preds to same scale
        target_boxes = target[:, 1:].to(device)  # [num_targets, 8]

        if target_boxes.numel() == 0 or pred_boxes.numel() == 0:
            print("No targets or predictions for this image.")
            return

        # Compute IoU
        ious = quad_iou_matrix(pred_boxes, target_boxes)  # [num_preds, num_targets]

        # Generate distinct colors for each target
        num_targets = target_boxes.shape[0]
        colors = [(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
                  for c in plt.cm.get_cmap('hsv', num_targets)(range(num_targets))[:, :3]]

        # For each target, pick best prediction
        best_pred_indices = ious.argmax(dim=0)  # [num_targets]
        for t_i, p_i in enumerate(best_pred_indices):
            p_i = min(p_i.item(), pred_boxes.shape[0] - 1)  # clamp to avoid out-of-bounds
            poly = pred_boxes[p_i].cpu().numpy()
            pts = np.array([[poly[0] * w, poly[1] * h],
                            [poly[2] * w, poly[3] * h],
                            [poly[4] * w, poly[5] * h],
                            [poly[6] * w, poly[7] * h]], dtype=np.int32)

            color = colors[t_i]  # unique color for this target
            cv2.polylines(img_color, [pts], isClosed=True, color=color, thickness=2)
            cv2.putText(img_color, f"target {t_i}", tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)

        plt.figure(figsize=(6, 6))
        plt.imshow(img_color[..., ::-1])
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def quad_iou_matrix(pred_boxes, target_boxes, eps=1e-6):
    """
    Approximate IoU between all pairs of quadrilateral boxes using
    vectorized GPU math (no Shapely). Works entirely on CUDA.

    Args:
        pred_boxes: [N, 8]
        target_boxes: [M, 8]
    Returns:
        IoU matrix [N, M]
    """
    # reshape to [N,4,2] and [M,4,2]
    pred = pred_boxes.reshape(-1, 4, 2)
    targ = target_boxes.reshape(-1, 4, 2)
    N, M = pred.shape[0], targ.shape[0]

    # Polygon area via Shoelace formula
    def polygon_area(poly):
        x = poly[..., 0]
        y = poly[..., 1]
        return 0.5 * torch.abs(torch.sum(x * torch.roll(y, shifts=-1, dims=-1) - y * torch.roll(x, shifts=-1, dims=-1), dim=-1))

    area_pred = polygon_area(pred)  # [N]
    area_targ = polygon_area(targ)  # [M]

    # Pairwise center distance proxy
    center_pred = pred.mean(dim=1)  # [N,2]
    center_targ = targ.mean(dim=1)  # [M,2]
    dists = torch.cdist(center_pred, center_targ)  # [N,M]

    # Convert distance to overlap proxy (closer = higher overlap)
    k = 5.0
    overlap = torch.exp(-k * dists)

    inter_area = overlap * torch.min(area_pred[:, None], area_targ[None, :])
    union_area = area_pred[:, None] + area_targ[None, :] - inter_area

    iou = inter_area / (union_area + eps)
    return iou.clamp(0,1)


def train(model, dataset, epochs=20, batch_size=4, lr=0.1, device='cuda',
          best_model_path="best_model.pth", best_loss_json="best_loss.json",
          conf_thresh=0.3, max_preds=20):
    """
    Train OBBDetector with OBBCriterion.

    conf_thresh: minimum confidence to keep a prediction
    max_preds: maximum number of predictions per image
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=lambda x: tuple(zip(*x)))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = OBBCriterion()

    # Load best loss from JSON if exists
    if os.path.exists(best_loss_json):
        try:
            with open(best_loss_json, 'r') as f:
                best_loss = json.load(f).get("best_loss", float('inf'))
        except Exception:
            best_loss = float('inf')
    else:
        best_loss = float('inf')

    # Load best model if exists
    if os.path.exists(best_model_path):
        try:
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            print(f"Loaded best model from '{best_model_path}' with loss {best_loss:.4f}")
        except Exception as e:
            print(f"Failed to load best model: {e}")

    # Helper: limit predictions per image
    def limit_preds(pred_boxes, pred_conf, pred_cls, conf_thresh, max_preds):
        B, num_preds, _ = pred_boxes.shape
        limited_boxes, limited_conf, limited_cls = [], [], []

        for b in range(B):
            mask = pred_conf[b] > conf_thresh
            boxes_b = pred_boxes[b][mask]
            conf_b  = pred_conf[b][mask]
            cls_b   = pred_cls[b][mask]

            # Keep only top max_preds by confidence
            if boxes_b.shape[0] > max_preds:
                conf_sorted, idx = conf_b.sort(descending=True)
                idx = idx[:max_preds]
                boxes_b = boxes_b[idx]
                conf_b  = conf_b[idx]
                cls_b   = cls_b[idx]

            # Pad if fewer predictions than max_preds
            pad = max_preds - boxes_b.shape[0]
            if pad > 0:
                boxes_b = torch.cat([boxes_b, torch.zeros(pad, 8, device=boxes_b.device)], dim=0)
                conf_b  = torch.cat([conf_b, torch.zeros(pad, device=conf_b.device)], dim=0)
                cls_b   = torch.cat([cls_b, torch.zeros(pad, cls_b.shape[1], device=cls_b.device)], dim=0)

            limited_boxes.append(boxes_b)
            limited_conf.append(conf_b)
            limited_cls.append(cls_b)

        return torch.stack(limited_boxes), torch.stack(limited_conf), torch.stack(limited_cls)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for imgs, targets in tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            imgs = torch.stack(imgs).to(device)
            targets = [t.to(device) for t in targets]

            optimizer.zero_grad()
            pred_boxes, pred_conf, pred_cls = model(imgs)

            # Apply confidence and max_preds limits
            pred_boxes, pred_conf, pred_cls = limit_preds(pred_boxes, pred_conf, pred_cls,
                                                          conf_thresh, max_preds)

            # Compute loss
            loss = criterion((pred_boxes, pred_conf, pred_cls), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(loader)
        print(f"Epoch {epoch + 1} loss: {epoch_loss:.4f}")

        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), best_model_path)
            with open(best_loss_json, 'w') as f:
                json.dump({"best_loss": best_loss}, f)
            print(f"New best model saved with loss {best_loss:.4f}")


if __name__ == '__main__':
    dataset = YOLOOBBDataset("data", img_size=640)
    model = OBBDetector(num_classes=2)
    # show_random_obb_image()
    train(model, dataset, epochs=10, batch_size=4, lr=0.01, device='cuda' if torch.cuda.is_available() else 'cpu')
    # train_FPDIOU(model, dataset, epochs=10, batch_size=2, lr=0.01, device='cuda' if torch.cuda.is_available() else 'cpu')

    get_random_prediction_image(model, dataset, device='cuda' if torch.cuda.is_available() else 'cpu')
