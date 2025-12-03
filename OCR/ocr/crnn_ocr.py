import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import matplotlib
import random
import json

from OCR.ocr.preprocess_sentences import preprocess_tuple_images

matplotlib.use("TkAgg")

# =====================
# DATASET
# =====================

class HandwritingDataset(Dataset):
    def __init__(self, root, charset):
        self.image_dir = os.path.join(root, "images")
        label_path = os.path.join(root, "labels.txt")
        self.samples = []
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                # parse line: {imagename} "{text}"
                if '"' in line:
                    name, text = line.strip().split('"', 1)
                    name = name.strip()
                    text = text.strip('"').strip()
                    self.samples.append((os.path.join(self.image_dir, name), text))
        self.charset = charset
        self.char_to_idx = {c: i + 1 for i, c in enumerate(charset)}  # 0 reserved for CTC blank

    def __len__(self):
        return len(self.samples)

    def encode(self, text):
        return torch.tensor([self.char_to_idx[c] for c in text if c in self.char_to_idx], dtype=torch.long)

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = torch.from_numpy(img).unsqueeze(0).float() / 255.0  # [1, H, W]
        label = self.encode(text)
        return img, label


class HandwritingDataset2(Dataset):
    def __init__(self, root, charset):
        """
        root: path to folder containing images and labels.txt
        charset: string of all allowed characters
        """
        # self.image_dir = os.path.join(root, "external", "images")
        # label_path = os.path.join(root, "external", "labels.txt")
        self.image_dir = os.path.join(root, "images")
        label_path = os.path.join(root, "labels.txt")
        self.samples = []

        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                if '"' in line:
                    name, text = line.strip().split('"', 1)
                    name = name.strip()
                    text = text.strip('"').strip()
                    # Skip samples with empty labels
                    if text:
                        img_path = os.path.join(self.image_dir, name)
                        if os.path.exists(img_path):
                            self.samples.append((img_path, text))

        self.charset = charset
        self.char_to_idx = {c: i + 1 for i, c in enumerate(charset)}  # 0 reserved for CTC blank

    def __len__(self):
        return len(self.samples)

    def encode(self, text):
        return torch.tensor([self.char_to_idx[c] for c in text if c in self.char_to_idx], dtype=torch.long)

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = torch.from_numpy(img).unsqueeze(0).float() / 255.0  # [1, H, W]
        label = self.encode(text)
        return img, label

    def train_val_split(self, val_ratio=0.2, seed=None):
        """
        Dynamically split dataset into train and validation subsets.
        Returns: (train_dataset, val_dataset)
        """
        if seed is not None:
            torch.manual_seed(seed)
        val_size = int(len(self) * val_ratio)
        train_size = len(self) - val_size
        return random_split(self, [train_size, val_size])


def collate_fn(batch):
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs)  # [B, 1, H, W]
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    labels = torch.cat(labels)
    return imgs, labels, label_lengths

# =====================
# MODEL
# =====================

class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d((8, 1))
        )
        self.lstm = nn.LSTM(512, 256, bidirectional=True, num_layers=2, batch_first=True)
        self.fc = nn.Linear(512, num_classes + 1)  # +1 for CTC blank

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        assert h == 1, f"Height after CNN should be 1 but got {h}"
        x = x.squeeze(2).permute(0, 2, 1)  # [B, W, C]
        x, _ = self.lstm(x)
        x = self.fc(x)
        x = x.permute(1, 0, 2)  # (T, B, C)
        return x

# =====================
# DECODING & EVALUATION
# =====================

def decode_predictions(logits, charset):
    logits = F.log_softmax(logits, dim=2)
    preds = logits.argmax(2).permute(1, 0)  # [B, T]
    results = []
    idx_to_char = {i + 1: c for i, c in enumerate(charset)}
    for seq in preds:
        prev = -1
        text = ""
        for p in seq:
            if p != prev and p != 0:
                text += idx_to_char.get(p.item(), "")
            prev = p
        results.append(text)
    return results

def decode_batch_labels(labels, lengths, charset):
    idx_to_char = {i + 1: c for i, c in enumerate(charset)}
    res = []
    idx = 0
    for l in lengths:
        seq = labels[idx:idx + l]
        res.append("".join([idx_to_char.get(c.item(), "") for c in seq]))
        idx += l
    return res


def levenshtein_distance(s1, s2):
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)

    s1 = s1.lower()
    s2 = s2.lower()

    n, m = len(s1), len(s2)
    d = np.zeros((n + 1, m + 1), dtype=int)

    for i in range(n + 1):
        d[i, 0] = i
    for j in range(m + 1):
        d[0, j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            d[i, j] = min(
                d[i - 1, j] + 1,      # deletion
                d[i, j - 1] + 1,      # insertion
                d[i - 1, j - 1] + cost  # substitution
            )

    return d[n, m]


def get_similarity(s1, s2):
    if not s1 or not s2:
        return 0.0
    distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    return 1.0 - distance / max_len


def evaluate(model, loader, charset, device):
    model.eval()
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    total_loss = 0
    similarities = []
    all_true_chars = []
    all_pred_chars = []

    with torch.no_grad():
        for imgs, labels, label_lengths in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            log_probs = F.log_softmax(logits, dim=2)
            input_lengths = torch.full((imgs.size(0),), logits.size(0), dtype=torch.long, device=device)
            loss = criterion(log_probs, labels, input_lengths, label_lengths)
            total_loss += loss.item()

            preds = decode_predictions(logits, charset)
            truths = decode_batch_labels(labels, label_lengths, charset)

            for p, t in zip(preds, truths):
                similarities.append(get_similarity(p, t))
                # Extend char lists for precision/recall/F1 (character-level)
                all_true_chars.extend(list(t))
                all_pred_chars.extend(list(p))

    avg_similarity = np.mean(similarities) if similarities else 0.0

    # Compute char-level metrics (need aligned charset)
    # Create mapping for consistent encoding
    idx = {c: i for i, c in enumerate(sorted(set(charset)))}
    y_true = [idx.get(c, -1) for c in all_true_chars if c in idx]
    y_pred = [idx.get(c, -1) for c in all_pred_chars if c in idx]

    # Pad or truncate to equal length for fair comparison
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    acc = np.mean([p == t for p, t in zip(y_pred, y_true)]) if len(y_true) > 0 else 0.0
    prec = precision_score(y_true, y_pred, average='micro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='micro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)

    return total_loss / len(loader), avg_similarity, acc, prec, rec, f1

# =====================
# TRAINING LOOP
# =====================

def train(model, train_loader, val_loader, charset, epochs=20, lr=1e-3, device='cuda'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    best_loss_file = "best_small_loss.json"
    best_model_file = "best_small_ocr_model.pth"
    last_model_file = "last_small_ocr_model.pth"
    best_loss = 0
        # float('inf')

    # Load previous best if exists
    if os.path.exists(best_model_file):
        model.load_state_dict(torch.load(best_model_file, map_location=device))
    if os.path.exists(best_loss_file):
        with open(best_loss_file, "r") as f:
            best_loss = json.load(f).get("best_loss", float('inf'))
        print(f"Loaded previous best model with loss {best_loss:.4f}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, labels, label_lengths in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            log_probs = F.log_softmax(logits, dim=2)
            input_lengths = torch.full((imgs.size(0),), logits.size(0), dtype=torch.long, device=device)
            loss = criterion(log_probs, labels, input_lengths, label_lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_loss, avg_similarity,  acc, prec, rec, f1 = evaluate(model, val_loader, charset, device)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Average Similarity: {avg_similarity:.4f} | Val Loss: {val_loss:.4f} | "
              f"Acc: {acc:.3f} | F1: {f1:.3f}")

        # Save best model
        if avg_similarity > best_loss:
            best_loss = avg_similarity
            torch.save(model.state_dict(), best_model_file)
            with open(best_loss_file, "w") as f:
                json.dump({"best_loss": best_loss}, f)
            print(f"Saved new best model with loss {best_loss:.4f}")

        # # Show 4 random validation images every 10 epochs
        # if (epoch + 1) % 10 == 0:
        #     model.eval()
        #     indices = random.sample(range(len(val_loader.dataset)), 4)
        #     plt.figure(figsize=(12, 3))
        #     for i, idx in enumerate(indices):
        #         img, label = val_loader.dataset[idx]
        #         img_input = img.unsqueeze(0).to(device)
        #         with torch.no_grad():
        #             logits = model(img_input)
        #             pred_text = decode_predictions(logits, charset)[0]
        #         true_text = decode_batch_labels(label, torch.tensor([len(label)]), charset)[0]  # <-- fix here
        #
        #         plt.subplot(1, 4, i + 1)
        #         plt.imshow(img.squeeze(0).cpu(), cmap='gray')
        #         plt.title(f"P: {pred_text}\nT: {true_text}")
        #         plt.axis('off')
        #     plt.show()
        #     model.train()

        if epoch == epochs - 1:
            torch.save(model.state_dict(), last_model_file)


def ocr_tuple_list(model, charset, results_list, target_height=64, target_width=512, device='cuda'):
    processed_list = preprocess_tuple_images(results_list, target_height, target_width)
    results_json = []

    with torch.no_grad():
        for number_img, answer_img in processed_list:
            # Convert to tensors and add batch & channel dims
            number_tensor = torch.from_numpy(number_img).unsqueeze(0).unsqueeze(0).to(device)
            answer_tensor = torch.from_numpy(answer_img).unsqueeze(0).unsqueeze(0).to(device)

            # Forward pass
            number_logits = model(number_tensor)  # (T, B, C)
            answer_logits = model(answer_tensor)

            # Decode predictions
            number_text = decode_predictions(number_logits, charset)[0]
            answer_text = decode_predictions(answer_logits, charset)[0]

            results_json.append({
                "number": number_text,
                "answer": answer_text
            })

    return results_json


# =====================
# MAIN
# =====================

if __name__ == "__main__":
    # charset = "abcčćdđefghijklmnopqrsštuvwxyzž0123456789-'& "
    charset = "abcčćdđefghijklmnopqrsštuvwxyzž0123456789-'&,) "
    # train_ds = HandwritingDataset("dataset/train", charset)
    # val_ds = HandwritingDataset("dataset/val", charset)
    dataset = HandwritingDataset2("small_dataset", charset)

    # Dynamic 80/20 split
    train_ds, val_ds = dataset.train_val_split(val_ratio=0.2, seed=42)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, collate_fn=collate_fn)

    model = CRNN(num_classes=len(charset))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # for i in range(10):
    train(model, train_loader, val_loader, charset, epochs=500, lr=1e-3, device=device)
