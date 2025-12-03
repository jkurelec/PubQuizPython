import os
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset
import cv2


class OBBDetector(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.dropout1 = nn.Dropout2d(0.2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.dropout2 = nn.Dropout2d(0.3)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.dropout3 = nn.Dropout2d(0.4)

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.dropout4 = nn.Dropout2d(0.5)

        self.pred = nn.Conv2d(256, self.num_classes + 8 + 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.dropout3(x)
        x = self.conv4(x)
        x = self.dropout4(x)

        pred = self.pred(x)  # [B, num_classes+8+1, H, W]

        B, C, H, W = pred.shape
        pred = pred.permute(0,2,3,1).contiguous().view(B, -1, C)  # [B, H*W, C]

        pred_boxes = torch.sigmoid(pred[:, :, :8])   # [B, num_preds, 8]
        pred_conf  = torch.sigmoid(pred[:, :, 8:9])  # [B, num_preds, 1]
        pred_cls   = pred[:, :, 9:]                  # [B, num_preds, num_classes]

        pred_conf = pred_conf.squeeze(-1)  # [B, num_preds]

        return pred_boxes, pred_conf, pred_cls


class OBBCriterion(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.box_loss_fn = nn.SmoothL1Loss(reduction='none')

    def forward(self, preds, targets, iou_threshold=0.3):
        """
        preds: tuple of (pred_boxes [B,N,8], pred_conf [B,N], pred_cls [B,N,num_classes])
        targets: list of B tensors, each [num_objs, 9] (class + 8 coords)
        """
        pred_boxes, pred_conf, pred_cls = preds
        total_loss = 0.0
        B = pred_boxes.shape[0]

        for b in range(B):
            boxes = pred_boxes[b]    # [num_preds, 8]
            confs = pred_conf[b]     # [num_preds]
            cls_logits = pred_cls[b]  # [num_preds, num_classes]
            target = targets[b]       # [num_objs, 9]

            if target.numel() == 0:
                conf_loss = F.mse_loss(confs, torch.zeros_like(confs))
                total_loss += conf_loss
                continue

            target_boxes = target[:, 1:].to(boxes.device)   # [num_objs, 8]

            # --- Compute exact IoU ---
            ious = self.quad_iou_exact(boxes, target_boxes)  # [num_preds, num_objs]

            # Find best match for each prediction
            best_iou_per_pred, best_target_idx = ious.max(dim=1)
            matched_targets = target[best_target_idx]

            # --- Box loss weighted by IoU ---
            box_loss = (self.box_loss_fn(boxes, matched_targets[:, 1:]) * best_iou_per_pred.unsqueeze(1)).mean()

            # --- Classification loss ---
            target_cls_onehot = F.one_hot(matched_targets[:, 0].long(), num_classes=self.num_classes).float()
            cls_loss = (F.binary_cross_entropy_with_logits(cls_logits, target_cls_onehot, reduction='none') *
                        best_iou_per_pred.unsqueeze(1)).mean()

            # --- Confidence loss ---
            conf_target = (best_iou_per_pred > iou_threshold).float()
            conf_loss = F.mse_loss(confs, conf_target)

            total_loss += box_loss + cls_loss + conf_loss

        return total_loss / B

    def quad_iou_exact(self, boxes1, boxes2, eps=1e-6):
        """
        Exact IoU between two sets of quadrilaterals (convex) on GPU using
        vectorized PyTorch operations.

        boxes1: [N,8], boxes2: [M,8]
        returns: [N,M] IoU matrix
        """
        N, M = boxes1.shape[0], boxes2.shape[0]

        # Reshape to [N,4,2] and [M,4,2]
        poly1 = boxes1.view(N,4,2)
        poly2 = boxes2.view(M,4,2)

        # Compute area of polygons using shoelace formula
        def poly_area(poly):
            x = poly[...,0]
            y = poly[...,1]
            return 0.5 * torch.abs(torch.sum(x*torch.roll(y, shifts=-1, dims=-1) - y*torch.roll(x, shifts=-1, dims=-1), dim=-1))

        area1 = poly_area(poly1)  # [N]
        area2 = poly_area(poly2)  # [M]

        # Compute pairwise intersection areas
        inter_area = torch.zeros((N,M), device=boxes1.device)

        for i in range(N):
            p1 = poly1[i]  # [4,2]
            # Expand p1 to [M,4,2] to intersect with poly2
            p1_expand = p1.unsqueeze(0).repeat(M,1,1)
            inter_area[i] = self.quad_intersection_area(p1_expand, poly2)  # [M]

        union_area = area1[:,None] + area2[None,:] - inter_area
        iou = inter_area / (union_area + eps)
        return iou.clamp(0,1)

    def quad_intersection_area(self, poly1, poly2, eps=1e-9):
        """
        Compute exact intersection area between two batches of convex quadrilaterals
        using Sutherland–Hodgman clipping (fully vectorized PyTorch version).
        poly1, poly2: [B,4,2]  (each quad = 4 points in order)
        Returns: [B] intersection areas
        """
        B = poly1.shape[0]
        inter_areas = torch.zeros(B, device=poly1.device)

        for i in range(B):
            subject = poly1[i]
            clipper = poly2[i]

            # Perform Sutherland–Hodgman polygon clipping
            output_list = subject
            for j in range(4):  # 4 edges of the clip polygon
                A = clipper[j]
                Bp = clipper[(j + 1) % 4]

                edge_vec = Bp - A
                new_output = []
                if len(output_list) == 0:
                    break

                prev_point = output_list[-1]
                for curr_point in output_list:
                    # Inside test: (Bp - A) x (P - A) >= 0 (left of edge)
                    cross_curr = edge_vec[0] * (curr_point[1] - A[1]) - edge_vec[1] * (curr_point[0] - A[0])
                    cross_prev = edge_vec[0] * (prev_point[1] - A[1]) - edge_vec[1] * (prev_point[0] - A[0])

                    if cross_curr >= 0:
                        if cross_prev < 0:
                            # Compute intersection point
                            inter_pt = self._compute_intersection(prev_point, curr_point, A, Bp)
                            if inter_pt is not None:
                                new_output.append(inter_pt)
                        new_output.append(curr_point)
                    elif cross_prev >= 0:
                        inter_pt = self._compute_intersection(prev_point, curr_point, A, Bp)
                        if inter_pt is not None:
                            new_output.append(inter_pt)
                    prev_point = curr_point

                output_list = torch.stack(new_output) if len(new_output) > 0 else torch.empty((0, 2),
                                                                                              device=poly1.device)

            if output_list.shape[0] >= 3:
                x = output_list[:, 0]
                y = output_list[:, 1]
                area = 0.5 * torch.abs(torch.sum(x * torch.roll(y, -1) - y * torch.roll(x, -1)))
                inter_areas[i] = area

        return inter_areas

    def _compute_intersection(self, p1, p2, a, b):
        """Compute line intersection between segment p1–p2 and clip edge a–b."""
        p1, p2, a, b = p1.float(), p2.float(), a.float(), b.float()
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = a
        x4, y4 = b

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if torch.abs(denom) < 1e-9:
            return None

        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        return torch.tensor([px, py], device=p1.device)

class YOLOOBBDataset(Dataset):
    def __init__(self, root_dir, img_size=640):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "images")
        self.lbl_dir = os.path.join(root_dir, "labels")
        self.img_files = sorted([
            f for f in os.listdir(self.img_dir)
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ])
        self.img_size = img_size

        # Read class names (optional)
        classes_path = os.path.join(root_dir, "classes.txt")
        if os.path.exists(classes_path):
            with open(classes_path, "r") as f:
                self.class_names = [line.strip() for line in f.readlines()]
        else:
            self.class_names = []
        self.num_classes = len(self.class_names)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        lbl_path = os.path.join(self.lbl_dir, os.path.splitext(img_name)[0] + ".txt")

        # Load grayscale image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype("float32") / 255.0
        img = torch.from_numpy(img).unsqueeze(0)  # [1, H, W]

        # Load polygon labels
        polys = []
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 9:
                        continue  # skip malformed
                    cls = int(parts[0])
                    coords = list(map(float, parts[1:9]))  # x1 y1 x2 y2 x3 y3 x4 y4
                    polys.append([cls] + coords)

        polys = torch.tensor(polys, dtype=torch.float32) if polys else torch.zeros((0, 9))
        return img, polys
