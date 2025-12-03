import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = []

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths= []
        self.labels = []
        idx = 0
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.jpg'):
                    image_path = os.path.join(subdir, file)
                    label = idx
                    self.image_paths.append(image_path)
                    self.labels.append(label)
            if "_".join(subdir.split("\\")[-3:]).startswith("cursive") or "_".join(subdir.split("\\")[-3:]).startswith("print"):
                idx += 1
                if "_".join(subdir.split("\\")[-3:]) not in classes:
                    classes.append("_".join(subdir.split("\\")[-3:]))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])

        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


# Define the OCR Model
class OCRModel(nn.Module):
    def __init__(self):
        super(OCRModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8, 256)  # First fully connected layer
        self.dropout1 = nn.Dropout(p=0.5)  # Dropout layer after fc1
        self.fc2 = nn.Linear(256, 128)  # Second fully connected layer
        self.dropout2 = nn.Dropout(p=0.5)  # Dropout layer after fc2
        self.fc3 = nn.Linear(128, 124)  # Output layer for 124 classes

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # Apply dropout after first fully connected layer
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)  # Apply dropout after second fully connected layer
        x = self.fc3(x)
        return x


def custom_loss_function(predictions, targets, class_names):
    """
    Custom loss function to ignore differences in case and style (print/cursive)
    and only penalize when the predicted letter does not match the target letter.

    Args:
    - predictions (torch.Tensor): The predicted logits from the model.
    - targets (torch.Tensor): The true labels.
    - class_names (list): List of class names in the format 'print_upper_a'.

    Returns:
    - loss (torch.Tensor): Computed loss.
    """
    # Get predicted labels (class indices)
    predicted_labels = torch.argmax(predictions, dim=1)

    # Map class indices to the letter part (last part of 'print_upper_a')
    predicted_letters = [class_names[pred].split('_')[-1] for pred in predicted_labels]
    true_letters = [class_names[target].split('_')[-1] for target in targets]

    # Penalize only if letters don't match
    correct_mask = torch.tensor(
        [pred.lower() == true.lower() for pred, true in zip(predicted_letters, true_letters)],
        device=predictions.device  # Directly specify the device
    ).float()

    # Calculate CrossEntropy loss normally
    base_loss = F.cross_entropy(predictions, targets, reduction='none')

    # Ignore loss where the letter part is correct
    loss = base_loss * (1 - correct_mask)

    # Average the loss
    return loss.mean()


def custom_accuracy(y_true, y_pred):
    """
    Calculate accuracy considering if the last part of class names (letters) is the same.
    """
    y_true_classes = [classes[label] for label in y_true]
    y_pred_classes = [classes[pred] for pred in y_pred]

    correct = sum(
        1 for true, pred in zip(y_true_classes, y_pred_classes)
        if true.split('_')[-1].lower() == pred.split('_')[-1].lower()
    )

    return correct / len(y_true) if len(y_true) > 0 else 0


def custom_precision(y_true, y_pred):
    """
    Calculate precision considering if the last part of class names (letters) is the same.
    """
    y_true_classes = [classes[label] for label in y_true]
    y_pred_classes = [classes[pred] for pred in y_pred]

    true_positive = sum(
        1 for true, pred in zip(y_true_classes, y_pred_classes)
        if true.split('_')[-1].lower() == pred.split('_')[-1].lower() and true == pred
    )

    predicted_positive = sum(
        1 for pred in y_pred_classes
        if pred.split('_')[-1].lower() in [true.split('_')[-1].lower() for true in y_true_classes]
    )

    return true_positive / predicted_positive if predicted_positive > 0 else 0


def custom_recall(y_true, y_pred):
    """
    Calculate recall considering if the last part of class names (letters) is the same.
    """
    y_true_classes = [classes[label] for label in y_true]
    y_pred_classes = [classes[pred] for pred in y_pred]

    true_positive = sum(
        1 for true, pred in zip(y_true_classes, y_pred_classes)
        if true.split('_')[-1].lower() == pred.split('_')[-1].lower() and true == pred
    )

    actual_positive = sum(
        1 for true in y_true_classes
        if true.split('_')[-1].lower() in [pred.split('_')[-1].lower() for pred in y_pred_classes]
    )

    return true_positive / actual_positive if actual_positive > 0 else 0


def custom_f1_score(y_true, y_pred):
    """
    Calculate F1 score considering if the last part of class names (letters) is the same.
    """
    precision = custom_precision(y_true, y_pred)
    recall = custom_recall(y_true, y_pred)

    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


# Training Function
def train_model(model, train_loader, validation_loader, criterion, optimizer, num_epochs=100, patience=15):
    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        model = model.to(device)
        model.train()
        running_loss = 0.0
        all_labels = []
        all_preds = []

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)  # Move to GPU
            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            all_labels.extend(labels.cpu().numpy())
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())

        epoch_loss = running_loss / len(train_loader)
        accuracy = custom_accuracy(all_labels, all_preds)
        precision = custom_precision(all_labels, all_preds)
        recall = custom_recall(all_labels, all_preds)
        f1 = custom_f1_score(all_labels, all_preds)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}, '
              f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

        # Validation after each epoch
        model.eval()
        val_labels = []
        val_preds = []
        with torch.no_grad():
            for val_images, val_labels_batch in validation_loader:
                val_images, val_labels_batch = val_images.to(device), val_labels_batch.to(device)
                val_outputs = model(val_images)
                _, val_preds_batch = torch.max(val_outputs, 1)
                val_labels.extend(val_labels_batch.cpu().numpy())
                val_preds.extend(val_preds_batch.cpu().numpy())

        val_accuracy = custom_accuracy(val_labels, val_preds)
        val_precision = custom_precision(val_labels, val_preds)
        val_recall = custom_recall(val_labels, val_preds)
        val_f1 = custom_f1_score(val_labels, val_preds)

        print(f'Validation - Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, '
              f'Recall: {val_recall:.4f}, F1: {val_f1:.4f}')

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
            print("Best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

# Validation Function
def validate_model(model, validation_loader):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)  # Move to GPU
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)

    print(f'Validation Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')

# Testing Function
def test_model(model, test_loader):
    model.eval()
    all_labels = []
    all_preds = []
    sample_images = []  # For displaying images
    sample_labels = []  # For displaying actual labels
    sample_preds = []   # For displaying predicted labels

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # Move to GPU
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            # Collecting sample images and their labels for visualization
            if len(sample_images) < 16:  # Limit to 16 images
                sample_images.extend(images.cpu().numpy())
                sample_labels.extend(labels.cpu().numpy())
                sample_preds.extend(preds.cpu().numpy())

    custom_test_accuracy = custom_accuracy(all_labels, all_preds)
    custom_test_precision = custom_precision(all_labels, all_preds)
    custom_test_recall = custom_recall(all_labels, all_preds)
    custom_test_f1 = custom_f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)

    print(f'Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
    print(f'Custom test Accuracy: {custom_test_accuracy:.4f}, Precision: {custom_test_precision:.4f}, Recall: {custom_test_recall:.4f}, F1: {custom_test_f1:.4f}')

    # Plotting a 4x4 grid of sample predictions
    plot_predictions(sample_images[:16], sample_labels[:16], sample_preds[:16])


def plot_predictions(images, labels, preds):
    """Plot a 4x4 grid of images with actual and predicted labels, including class names."""
    # Plot the predictions
    plt.figure(figsize=(10, 10))
    for i in range(16):  # Plot 16 images in a 4x4 grid
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i][0], cmap='gray')  # Display the first channel (grayscale)

        # Get true and predicted labels in human-readable format
        true_label = classes[labels[i]]
        pred_label = classes[preds[i]]

        # Set the plot title with true and predicted class names
        plt.title(f'True: {true_label}\nPred: {pred_label}')
        plt.axis('off')

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

# Main function to load data and start training
def main():
    dataset_path = os.path.abspath('dataset')
    train_path = os.path.join(dataset_path, 'train')
    test_path = os.path.join(dataset_path, 'test')
    validation_path = os.path.join(dataset_path, 'validation')

    transform = transforms.Compose([
        transforms.Grayscale(),  # Convert images to grayscale
        transforms.Resize((32, 32)),  # Resize images
        transforms.ToTensor()  # Convert images to tensor
    ])

    # Load datasets
    train_dataset = CustomDataset(train_path, transform=transform)
    test_dataset = CustomDataset(test_path, transform=transform)
    validation_dataset = CustomDataset(validation_path, transform=transform)

    # DataLoader
    num_workers = os.cpu_count()-4  # Adjust as needed

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=num_workers)
    validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=True, num_workers=num_workers)

    model = OCRModel().to(device)
    criterion = lambda outputs, labels: custom_loss_function(outputs, labels, class_names=classes)
    #criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    # model.load_state_dict(torch.load('best_model.pth', weights_only=True))
    train_model(model, train_loader, validation_loader, criterion, optimizer)

    # Test the model with the saved best weights
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))
    test_model(model, test_loader)

def only_test_model():
    dataset_path = os.path.abspath('dataset')
    test_path = os.path.join(dataset_path, 'test')
    transform = transforms.Compose([
        transforms.Grayscale(),  # Convert images to grayscale
        transforms.Resize((32, 32)),  # Resize images
        transforms.ToTensor()  # Convert images to tensor
    ])
    test_dataset = CustomDataset(test_path, transform=transform)
    num_workers = os.cpu_count() - 4
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=num_workers)
    model = OCRModel().to(device)
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))
    test_model(model, test_loader)


if __name__ == '__main__':
    # for _ in range(10):
    #     only_test_model()
    main()
